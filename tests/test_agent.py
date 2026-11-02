from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

from mosaic.agent import AgentLoop, AgentProtocolError, ToolRegistry
from mosaic.storage import configure_database
from mosaic.web import FETCH_ENDPOINT, SEARCH_ENDPOINT, DeepSeekWebClient, WebSearchError, canonical_http_url


class ScriptedGateway:
    """Explicit test double: it never invokes Qwen, Chaquopy, or llama.cpp."""

    def __init__(self, replies: list[dict[str, Any]], vector: list[float] | None = None,
                 embedding_error: Exception | None = None) -> None:
        self._replies = list(replies)
        self._vector = vector or [1.0, 0.0]
        self._embedding_error = embedding_error
        self.generated: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
        self.embedded_queries: list[str] = []

    def generate(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Mapping[str, Any]:
        self.generated.append((messages, tools))
        return self._replies.pop(0)

    def embed_query(self, text: str) -> tuple[list[float], dict[str, float]]:
        self.embedded_queries.append(text)
        if self._embedding_error:
            raise self._embedding_error
        return self._vector, {"embedding_query_ms": 1.0, "embedding_model_load_ms": 2.0}


def tool_call(name: str, arguments: Any, call_id: str = "call-1") -> dict[str, Any]:
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}},
    ]}


class AgentLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.store = configure_database(str(Path(self.temp.name) / "mosaic.sqlite3"))
        for name, vector in {"cat.jpg": [1.0, 0.0], "car.jpg": [0.0, 1.0], "banana.jpg": [0.242535625, 0.9701425]}.items():
            source = self.store.sources.register("image", f"/private/{name}", f"fingerprint-{name}")
            self.store.embeddings.save(source, "embeddinggemma-2-gguf", vector, normalized=True)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def loop(self, gateway: ScriptedGateway, max_tool_calls: int = 2) -> AgentLoop:
        return AgentLoop(gateway, ToolRegistry(self.store, gateway.embed_query), max_tool_calls=max_tool_calls)

    def web_loop(self, gateway: ScriptedGateway, client: DeepSeekWebClient) -> AgentLoop:
        return AgentLoop(gateway, ToolRegistry(self.store, gateway.embed_query, client))

    def test_direct_answer_never_invokes_embedding_or_tool(self) -> None:
        gateway = ScriptedGateway([{"role": "assistant", "content": "Hello! I am well.", "parser": "test"}])
        result = self.loop(gateway).run([{"role": "user", "content": "Hello! How are you?"}])
        self.assertEqual("Hello! I am well.", result.answer)
        self.assertEqual((), result.tool_calls)
        self.assertEqual([], gateway.embedded_queries)
        self.assertEqual(1, len(gateway.generated))

    def test_search_images_executes_real_sqlite_vector_ranking_then_resumes(self) -> None:
        gateway = ScriptedGateway([
            tool_call("search_images", {"query": "a banana"}),
            {"role": "assistant", "content": "The banana photo is banana.jpg.", "parser": "test"},
        ], vector=[0.242535625, 0.9701425])
        result = self.loop(gateway).run([{"role": "user", "content": "Find the banana."}])
        self.assertEqual("The banana photo is banana.jpg.", result.answer)
        self.assertEqual(["a banana"], gateway.embedded_queries)
        self.assertTrue(result.tool_executions[0].ok)
        self.assertEqual("banana.jpg", result.tool_executions[0].payload["results"][0]["filename"])
        second_context = gateway.generated[1][0]
        self.assertEqual("assistant", second_context[-2]["role"])
        self.assertEqual("tool", second_context[-1]["role"])
        self.assertEqual("call-1", second_context[-1]["tool_call_id"])

    def test_invalid_arguments_are_returned_as_a_structured_tool_failure(self) -> None:
        gateway = ScriptedGateway([
            tool_call("search_images", {"query": "cat", "top_k": "3"}),
            {"role": "assistant", "content": "I could not run that image search.", "parser": "test"},
        ])
        result = self.loop(gateway).run([{"role": "user", "content": "Find a cat."}])
        self.assertFalse(result.tool_executions[0].ok)
        self.assertEqual("ValueError", result.tool_executions[0].payload["error"]["type"])
        self.assertEqual([], gateway.embedded_queries)

    def test_unknown_tool_is_not_executed(self) -> None:
        gateway = ScriptedGateway([
            tool_call("run_python", {"code": "import os"}),
            {"role": "assistant", "content": "That tool is unavailable.", "parser": "test"},
        ])
        result = self.loop(gateway).run([{"role": "user", "content": "Do something unsafe."}])
        self.assertFalse(result.tool_executions[0].ok)
        self.assertIn("not authorized", result.tool_executions[0].payload["error"]["message"])
        self.assertEqual([], gateway.embedded_queries)

    def test_embedding_failure_becomes_a_tool_result(self) -> None:
        gateway = ScriptedGateway([
            tool_call("search_images", {"query": "cat"}),
            {"role": "assistant", "content": "Image search is unavailable right now.", "parser": "test"},
        ], embedding_error=RuntimeError("Embedding model unavailable"))
        result = self.loop(gateway).run([{"role": "user", "content": "Find a cat."}])
        self.assertFalse(result.tool_executions[0].ok)
        self.assertEqual("RuntimeError", result.tool_executions[0].payload["error"]["type"])

    def test_model_cannot_exceed_two_tool_calls(self) -> None:
        gateway = ScriptedGateway([{
            "role": "assistant", "content": "", "tool_calls": [
                {"id": "one", "type": "function", "function": {"name": "search_images", "arguments": {"query": "cat"}}},
                {"id": "two", "type": "function", "function": {"name": "search_images", "arguments": {"query": "car"}}},
                {"id": "three", "type": "function", "function": {"name": "search_images", "arguments": {"query": "banana"}}},
            ],
        }])
        with self.assertRaisesRegex(AgentProtocolError, "exceeded"):
            self.loop(gateway).run([{"role": "user", "content": "Find pictures."}])
        self.assertEqual([], gateway.embedded_queries)

    def test_non_object_arguments_are_rejected_before_execution(self) -> None:
        gateway = ScriptedGateway([tool_call("search_images", "[]")])
        with self.assertRaisesRegex(AgentProtocolError, "arguments must be a JSON object"):
            self.loop(gateway).run([{"role": "user", "content": "Find a cat."}])

    def test_web_search_executes_compact_deepseek_result_then_resumes(self) -> None:
        requests: list[tuple[str, dict[str, Any]]] = []

        def post(url: str, headers: Mapping[str, str], body: bytes, timeout: float) -> tuple[int, bytes]:
            self.assertEqual("Bearer test-key", headers["Authorization"])
            payload = __import__("json").loads(body)
            requests.append((url, payload))
            return 200, b'{"results":[{"title":"Release notes","url":"https://example.com/release","date":"2026-01-01T00:00:00Z","snippet":"Current release details"}]}'

        gateway = ScriptedGateway([
            tool_call("web_search", {"query": "Mosaic release notes", "max_results": 4}),
            {"role": "assistant", "content": "The release notes are available at https://example.com/release.", "parser": "test"},
        ])
        result = self.web_loop(gateway, DeepSeekWebClient("test-key", http_post=post)).run(
            [{"role": "user", "content": "What are the current release notes?"}]
        )
        self.assertTrue(result.tool_executions[0].ok)
        self.assertEqual(SEARCH_ENDPOINT, requests[0][0])
        self.assertEqual(False, requests[0][1]["need_content"])
        payload = result.tool_executions[0].payload
        self.assertEqual("https://example.com/release", payload["results"][0]["url"])
        self.assertTrue(payload["untrusted_content"])

    def test_web_fetch_is_restricted_to_same_request_search_urls(self) -> None:
        gateway = ScriptedGateway([
            tool_call("web_fetch", {"url": "https://example.com/not-from-search"}),
            {"role": "assistant", "content": "I need a search result before reading a page.", "parser": "test"},
        ])
        result = self.web_loop(gateway, DeepSeekWebClient("test-key", http_post=lambda *_: (_ for _ in ()).throw(AssertionError()))).run(
            [{"role": "user", "content": "Read this page."}]
        )
        self.assertFalse(result.tool_executions[0].ok)
        self.assertIn("returned by web_search", result.tool_executions[0].payload["error"]["message"])

    def test_web_search_then_fetch_returns_bounded_untrusted_content(self) -> None:
        calls = 0

        def post(url: str, headers: Mapping[str, str], body: bytes, timeout: float) -> tuple[int, bytes]:
            nonlocal calls
            calls += 1
            if url == SEARCH_ENDPOINT:
                return 200, b'{"results":[{"title":"Article","url":"https://example.com/article","snippet":"Snippet"}]}'
            self.assertEqual(FETCH_ENDPOINT, url)
            return 200, ('{"title":"Article","content":"' + ('x' * 7000) + '"}').encode()

        gateway = ScriptedGateway([
            tool_call("web_search", {"query": "current public article"}, "search-1"),
            tool_call("web_fetch", {"url": "https://example.com/article"}, "fetch-1"),
            {"role": "assistant", "content": "I read the article.", "parser": "test"},
        ])
        result = self.web_loop(gateway, DeepSeekWebClient("test-key", http_post=post)).run(
            [{"role": "user", "content": "Find and read an article."}]
        )
        self.assertEqual(2, calls)
        fetch = result.tool_executions[1].payload
        self.assertTrue(fetch["truncated"])
        self.assertEqual(6000, len(fetch["content"]))

    def test_web_search_without_configured_key_is_safe_tool_failure(self) -> None:
        gateway = ScriptedGateway([
            tool_call("web_search", {"query": "current news"}),
            {"role": "assistant", "content": "Web search needs configuration.", "parser": "test"},
        ])
        result = self.loop(gateway).run([{"role": "user", "content": "Search current news."}])
        self.assertFalse(result.tool_executions[0].ok)
        self.assertEqual("WebSearchConfigurationError", result.tool_executions[0].payload["error"]["type"])

    def test_private_memory_results_are_never_followed_by_a_web_query(self) -> None:
        gateway = ScriptedGateway([
            tool_call("search_memories", {"query": "Mosaic"}, "memory-1"),
            tool_call("web_search", {"query": "Mosaic"}, "web-1"),
            {"role": "assistant", "content": "I kept the memory result local.", "parser": "test"},
        ])
        result = self.web_loop(gateway, DeepSeekWebClient("test-key")).run(
            [{"role": "user", "content": "Compare my memories with current news."}]
        )
        self.assertTrue(result.tool_executions[0].ok)
        self.assertFalse(result.tool_executions[1].ok)
        self.assertIn("private memory", result.tool_executions[1].payload["error"]["message"])

    def test_web_client_handles_rate_limit_and_invalid_fetch_urls_without_leaking_key(self) -> None:
        client = DeepSeekWebClient(
            "secret-do-not-log",
            http_post=lambda *_: (429, b'{"error":{"message":"too many requests"}}'),
        )
        with self.assertRaisesRegex(WebSearchError, "rate limit") as raised:
            client.search("current public information", 4)
        self.assertNotIn("secret-do-not-log", str(raised.exception))
        for value in ("file:///private/data", "https://user:pass@example.com", "http://127.0.0.1/admin"):
            with self.assertRaises(ValueError):
                canonical_http_url(value)

    def test_web_client_accepts_empty_results_and_maps_connection_failure(self) -> None:
        empty = DeepSeekWebClient("test-key", http_post=lambda *_: (200, b'{"results":[]}'))
        self.assertEqual([], empty.search("no matches expected", 4)["results"])
        offline = DeepSeekWebClient("test-key", http_post=lambda *_: (_ for _ in ()).throw(TimeoutError()))
        with self.assertRaisesRegex(WebSearchError, "internet connection"):
            offline.search("current public information", 4)


if __name__ == "__main__":
    unittest.main()
