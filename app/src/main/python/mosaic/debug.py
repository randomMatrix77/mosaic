"""ADB-facing diagnostic commands which remain owned by the Python backend.

The Android debug receiver supplies a JSON request and writes the returned JSON
to an app-private file. This module deliberately exposes a small, explicit
command set rather than a Python shell or arbitrary method invocation.
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping

from mosaic.agent import ToolCall, ToolRegistry
from mosaic.core import MosaicCore
from mosaic.inference import AndroidAgentGateway, get_model_paths
from mosaic.memory import MemoryService
from mosaic.storage import get_store
from mosaic.web import MAX_SEARCH_RESULTS, canonical_http_url, configured_web_client


def _require_object(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("Debug request must be a JSON object.")
    return value


def _required_text(request: Mapping[str, Any], name: str) -> str:
    value = request.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value.strip()


def _optional_text(request: Mapping[str, Any], name: str) -> str | None:
    value = request.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string when provided.")
    return value.strip() or None


def _core() -> MosaicCore:
    return MosaicCore()


def execute_command(request: Mapping[str, Any]) -> dict[str, Any]:
    """Run one supported debug command and return JSON-serializable data."""
    operation = _required_text(request, "operation")
    core = _core()

    if operation == "status":
        store = get_store()
        return {
            "ok": True,
            "operation": operation,
            "database_path": store.database.path,
            "python": {"implementation": platform.python_implementation(), "version": sys.version.split()[0]},
            "commands": [
                "status", "conversation.create", "conversation.list", "conversation.messages",
                "agent.chat", "web.search", "web.fetch", "images.search", "embedding.text", "qwen.vision",
                "memory.ingest_conversation", "memory.search", "memory.claims", "memory.evidence", "memory.delete_claim",
                "memory.supersede", "memory.curate",
            ],
        }

    if operation == "conversation.create":
        return {"ok": True, "operation": operation, "conversation": core.create_conversation(_optional_text(request, "title"))}

    if operation == "conversation.list":
        return {"ok": True, "operation": operation, "conversations": json.loads(core.list_conversations_json())}

    if operation == "conversation.messages":
        conversation_id = _required_text(request, "conversation_id")
        return {"ok": True, "operation": operation, "messages": json.loads(core.get_messages_json(conversation_id))}

    if operation == "agent.chat":
        conversation_id = _optional_text(request, "conversation_id")
        if conversation_id is None:
            conversation_id = core.create_conversation(_optional_text(request, "title"))["id"]
        result = core.process_message(conversation_id, _required_text(request, "message"))
        return {"ok": True, "operation": operation, "result": result}

    if operation == "web.search":
        max_results = request.get("max_results", 1)
        if isinstance(max_results, bool) or not isinstance(max_results, int) or not 1 <= max_results <= MAX_SEARCH_RESULTS:
            raise ValueError(f"max_results must be an integer from 1 to {MAX_SEARCH_RESULTS}.")
        return {"ok": True, "operation": operation,
                "result": configured_web_client().search(_required_text(request, "query"), max_results)}

    if operation == "web.fetch":
        result = configured_web_client().fetch(canonical_http_url(_required_text(request, "url")))
        # Confirm actual content without copying an untrusted page into ADB output.
        return {"ok": True, "operation": operation, "url": result["url"], "title": result["title"],
                "content_chars": len(result["content"]), "truncated": result["truncated"]}

    if operation == "images.search":
        query = _required_text(request, "query")
        top_k = request.get("top_k", 3)
        gateway = AndroidAgentGateway(get_model_paths())
        execution = ToolRegistry(get_store(), gateway.embed_query).execute(
            ToolCall("debug-search-images", "search_images", {"query": query, "top_k": top_k})
        )
        return {"ok": execution.ok, "operation": operation, "result": execution.payload,
                "elapsed_ms": execution.elapsed_ms}

    if operation == "embedding.text":
        text = _required_text(request, "text")
        vector, metrics = AndroidAgentGateway(get_model_paths()).embed_query(text)
        response: dict[str, Any] = {
            "ok": True, "operation": operation, "dimension": len(vector), "metrics": metrics,
            "preview": vector[:8],
        }
        if request.get("include_vector") is True:
            response["vector"] = vector
        return response

    if operation == "memory.ingest_conversation":
        conversation_id = _required_text(request, "conversation_id")
        gateway = AndroidAgentGateway(get_model_paths())
        result = MemoryService(get_store()).ingest_conversation(conversation_id, gateway)
        return {"ok": True, "operation": operation, "claim_ids": [item.id for item in result.claims],
                "processed_message_ids": list(result.skipped_message_ids), "model_metrics": dict(result.model_metrics),
                "embedding_metrics": dict(result.embedding_metrics)}

    if operation == "memory.search":
        gateway = AndroidAgentGateway(get_model_paths())
        result = MemoryService(get_store()).search(_required_text(request, "query"), gateway, request.get("top_k", 5))
        return {"ok": True, "operation": operation, "result": result}

    if operation == "memory.claims":
        status = request.get("status", "current")
        if status is not None and not isinstance(status, str):
            raise ValueError("status must be a string or null.")
        claims = get_store().memory.claims(status=status)
        return {"ok": True, "operation": operation, "claims": [
            {"id": item.id, "fact": item.description, "status": item.status, "assertion_type": item.assertion_type,
             "evidence": [reference.id for reference in get_store().memory.evidence_for_claim(item.id)]}
            for item in claims]}

    if operation == "memory.evidence":
        evidence = get_store().memory.resolve_evidence(_required_text(request, "evidence_id"))
        return {"ok": True, "operation": operation, "evidence": {
            "id": evidence.reference.id, "claim_id": evidence.reference.claim_id, "conversation_id": evidence.reference.conversation_id,
            "message_id": evidence.reference.message_id, "quote": evidence.quote,
            "span_available": evidence.reference.span_available,
            "context": [{"id": item.id, "role": item.role, "content": item.content} for item in evidence.context],
        }}

    if operation == "memory.delete_claim":
        claim_id = _required_text(request, "claim_id")
        get_store().memory.delete_claim(claim_id)
        return {"ok": True, "operation": operation, "claim_id": claim_id, "status": "deleted"}

    if operation == "memory.supersede":
        old_claim_id = _required_text(request, "old_claim_id")
        replacement_claim_id = _required_text(request, "replacement_claim_id")
        get_store().memory.supersede(old_claim_id, replacement_claim_id)
        return {"ok": True, "operation": operation, "old_claim_id": old_claim_id,
                "replacement_claim_id": replacement_claim_id}

    if operation == "memory.curate":
        raw_ids = request.get("claim_ids")
        if raw_ids is not None and (not isinstance(raw_ids, list) or not all(isinstance(item, str) for item in raw_ids)):
            raise ValueError("claim_ids must be an array of IDs when provided.")
        gateway = AndroidAgentGateway(get_model_paths())
        result = MemoryService(get_store()).curate(gateway, raw_ids)
        return {"ok": True, "operation": operation, "operations": list(result.applied_operations),
                "model_metrics": dict(result.model_metrics)}

    if operation == "qwen.vision":
        image_name = _required_text(request, "image_name")
        if Path(image_name).name != image_name:
            raise ValueError("image_name must be a filename inside the app-private debug image directory.")
        paths = get_model_paths()
        image_path = Path(paths.debug_images) / image_name
        if not image_path.is_file():
            raise ValueError(f"No debug image named {image_name!r} is available.")
        answer, metrics = AndroidAgentGateway(paths).generate_with_image(
            _required_text(request, "message"), str(image_path)
        )
        return {"ok": True, "operation": operation, "image_name": image_name,
                "answer": answer, "metrics": metrics}

    raise ValueError(f"Unsupported debug operation: {operation}.")


def run_command_json(serialized_request: str) -> str:
    """Chaquopy entry point. All failures are structured for ADB consumption."""
    try:
        request = _require_object(json.loads(serialized_request))
        return json.dumps(execute_command(request), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except Exception as error:
        return json.dumps({"ok": False, "error": {"type": type(error).__name__, "message": str(error)}},
                          ensure_ascii=False, separators=(",", ":"))
