"""Bounded DeepSeek Search & Fetch client for Mosaic's read-only web tools.

DeepSeek receives only the Qwen-selected search query or a URL which its own
search endpoint previously returned in the same request. It is never used for
language-model inference, and this module never reads Mosaic conversations or
memories.
"""

from __future__ import annotations

from dataclasses import dataclass
import ipaddress
import json
import re
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


DEEPSEEK_API_BASE = "https://api.deepseek.com"
SEARCH_ENDPOINT = f"{DEEPSEEK_API_BASE}/v1/tools/search"
FETCH_ENDPOINT = f"{DEEPSEEK_API_BASE}/v1/tools/fetch"
MAX_SEARCH_RESULTS = 4
MAX_SEARCH_RESULT_TOKENS = 512
MAX_FETCH_CONTENT_CHARS = 6_000
MAX_HTTP_RESPONSE_BYTES = 512 * 1024
HTTP_TIMEOUT_SECONDS = 15.0


class WebSearchError(RuntimeError):
    """A user-safe DeepSeek Search & Fetch failure."""


class WebSearchConfigurationError(WebSearchError):
    """Web search was requested before a key was configured."""


HttpPost = Callable[[str, Mapping[str, str], bytes, float], tuple[int, bytes]]
_configured_api_key: str | None = None


def configure_web_search(api_key: str | None) -> None:
    """Accept a key supplied from Android's Keystore-backed settings.

    The key deliberately remains process-memory only on the Python side. It is
    never written to Mosaic's SQLite database or included in status/debug data.
    """
    global _configured_api_key
    _configured_api_key = api_key.strip() if isinstance(api_key, str) and api_key.strip() else None


def configured_web_client() -> "DeepSeekWebClient":
    return DeepSeekWebClient(_configured_api_key)


def canonical_http_url(value: str) -> str:
    """Validate and canonicalize a public HTTP(S) URL without fetching it."""
    if not isinstance(value, str) or not value or len(value) > 8_192:
        raise ValueError("URL must be a non-empty HTTP(S) URL up to 8192 characters.")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("URL must use http or https and include a host.")
    try:
        _ = parsed.port
    except ValueError as error:
        raise ValueError("URL contains an invalid port.") from error
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URL must not include a username or password.")
    host = parsed.hostname.lower()
    if host == "localhost" or host.endswith(".local"):
        raise ValueError("Local network URLs are not allowed.")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None and (address.is_private or address.is_loopback or address.is_link_local
                                or address.is_multicast or address.is_reserved or address.is_unspecified):
        raise ValueError("Private or local-network URLs are not allowed.")
    # Fragments are browser-only and are not part of the remotely fetched page.
    return urlunsplit((parsed.scheme.lower(), parsed.netloc, parsed.path or "/", parsed.query, ""))


def _trim(value: Any, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _context_token_units(value: str) -> list[str]:
    """Conservatively approximate Qwen context tokens without loading a second tokenizer.

    Long ASCII runs are charged in four-character pieces; non-ASCII characters
    and punctuation are charged individually. This intentionally overcounts
    normal prose and prevents public search snippets from consuming the small
    mobile-agent context budget.
    """
    units: list[str] = []
    for match in re.finditer(r"[^\W_]+|[^\s]", value, flags=re.UNICODE):
        piece = match.group(0)
        if piece.isascii() and piece.isalnum():
            units.extend(piece[index:index + 4] for index in range(0, len(piece), 4))
        else:
            units.extend(piece)
    return units


def _truncate_context(value: str, budget: int) -> str:
    if budget <= 0:
        return ""
    units = _context_token_units(value)
    if len(units) <= budget:
        return value
    # Reassemble a compact, readable prefix. URLs are never passed here.
    return "".join(units[:max(1, budget - 1)]).rstrip() + "…"


def _default_http_post(url: str, headers: Mapping[str, str], body: bytes, timeout: float) -> tuple[int, bytes]:
    request = Request(url, data=body, headers=dict(headers), method="POST")
    try:
        with urlopen(request, timeout=timeout) as response:  # nosec B310: fixed HTTPS DeepSeek endpoint
            payload = response.read(MAX_HTTP_RESPONSE_BYTES + 1)
            if len(payload) > MAX_HTTP_RESPONSE_BYTES:
                raise WebSearchError("DeepSeek returned a response that was too large.")
            return int(response.status), payload
    except HTTPError as error:
        payload = error.read(8_192)
        return int(error.code), payload
    except (URLError, TimeoutError, OSError) as error:
        raise WebSearchError("Could not reach DeepSeek. Check your internet connection and try again.") from error


@dataclass
class DeepSeekWebClient:
    """Minimal client for DeepSeek's official raw Search and Fetch endpoints."""

    api_key: str | None
    http_post: HttpPost = _default_http_post
    timeout_seconds: float = HTTP_TIMEOUT_SECONDS

    def _headers(self) -> dict[str, str]:
        key = (self.api_key or "").strip()
        if not key:
            raise WebSearchConfigurationError(
                "Web search is not configured. Add a DeepSeek API key in Settings."
            )
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def _request(self, endpoint: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        try:
            status, raw = self.http_post(endpoint, self._headers(), encoded, self.timeout_seconds)
        except WebSearchError:
            raise
        except Exception as error:
            raise WebSearchError("Could not reach DeepSeek. Check your internet connection and try again.") from error
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise WebSearchError("DeepSeek returned an invalid response.") from error
        if not isinstance(decoded, Mapping):
            raise WebSearchError("DeepSeek returned an invalid response.")
        if status != 200:
            remote_error = decoded.get("error")
            message = remote_error.get("message") if isinstance(remote_error, Mapping) else None
            if status == 401:
                raise WebSearchConfigurationError("DeepSeek rejected the configured API key.")
            if status == 429:
                raise WebSearchError("DeepSeek rate limit reached. Please try again shortly.")
            if status in {408, 504}:
                raise WebSearchError("DeepSeek timed out while reading the web. Please try again.")
            if status == 402:
                raise WebSearchError("DeepSeek account balance is unavailable for web search.")
            if status == 404:
                raise WebSearchError("DeepSeek could not retrieve usable content from that webpage.")
            detail = _trim(message, 240)
            raise WebSearchError(f"DeepSeek web request failed ({status})" + (f": {detail}" if detail else "."))
        return decoded

    def search(self, query: str, max_results: int) -> dict[str, Any]:
        response = self._request(SEARCH_ENDPOINT, {
            "query": query,
            "count": max_results,
            "need_content": False,
        })
        raw_results = response.get("results", [])
        if not isinstance(raw_results, list):
            raise WebSearchError("DeepSeek returned an invalid search result list.")
        results: list[dict[str, str | None]] = []
        remaining_tokens = MAX_SEARCH_RESULT_TOKENS
        for raw in raw_results[:max_results]:
            if not isinstance(raw, Mapping):
                continue
            try:
                url = canonical_http_url(str(raw.get("url", "")))
            except ValueError:
                continue
            date = _trim(raw.get("date"), 80) or None
            fixed_cost = len(_context_token_units(url)) + (len(_context_token_units(date)) if date else 0)
            if fixed_cost >= remaining_tokens:
                break
            title = _truncate_context(_trim(raw.get("title"), 300) or url, remaining_tokens - fixed_cost)
            title_cost = len(_context_token_units(title))
            snippet = _truncate_context(
                _trim(raw.get("snippet") or raw.get("summary"), 500),
                remaining_tokens - fixed_cost - title_cost,
            )
            result_cost = fixed_cost + title_cost + len(_context_token_units(snippet))
            if result_cost > remaining_tokens:
                break
            results.append({"title": title or url, "url": url, "date": date, "snippet": snippet})
            remaining_tokens -= result_cost
        return {"ok": True, "query": query, "result_count": len(results), "results": results,
                "provider": "DeepSeek Search", "untrusted_content": True}

    def fetch(self, url: str) -> dict[str, Any]:
        response = self._request(FETCH_ENDPOINT, {"url": url})
        content = response.get("content")
        if not isinstance(content, str) or not content.strip():
            raise WebSearchError("DeepSeek returned no readable webpage content.")
        truncated = len(content) > MAX_FETCH_CONTENT_CHARS
        return {
            "ok": True,
            "url": url,
            "title": _trim(response.get("title"), 300) or url,
            "content": content[:MAX_FETCH_CONTENT_CHARS],
            "truncated": truncated,
            "result_count": 1,
            "provider": "DeepSeek Fetch",
            "untrusted_content": True,
        }
