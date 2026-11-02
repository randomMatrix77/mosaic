"""Python-owned application logic for Mosaic chat and persistence."""

from __future__ import annotations

import json
import platform
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

from mosaic.agent import AgentLoop, ToolCall, ToolRegistry
from mosaic.embeddings import EMBEDDING_MODEL_IDENTIFIER
from mosaic.events import GenerationCancelled, RequestEventEmitter
from mosaic.inference import AndroidAgentGateway, ModelPaths, configure_model_paths, get_model_paths
from mosaic.skill_registry import SkillRegistry
from mosaic.storage import CustomCapability, Conversation, Message, configure_database, get_store
from mosaic.web import configure_web_search, configured_web_client

QWEN_MODEL_IDENTIFIER = "qwen3.5-4b-q4_k_m"
_PROMPT_HISTORY_BUDGET_TOKENS = 1400


def configure_storage(database_path: str, qwen_model_path: str | None = None,
                      embedding_model_path: str | None = None,
                      embedding_projector_path: str | None = None,
                      qwen_projector_path: str | None = None,
                      debug_images_directory: str | None = None,
                      deepseek_api_key: str | None = None) -> dict[str, Any]:
    """Called by Kotlin once it has chosen an app-private database path."""
    store = configure_database(database_path)
    if qwen_model_path is not None:
        configure_model_paths(ModelPaths(
            qwen_model_path, embedding_model_path or "", embedding_projector_path or "",
            qwen_projector_path or "", debug_images_directory or "",
        ))
    configure_web_search(deepseek_api_key)
    return {"database_path": store.database.path, "schema_version": 5}


def _conversation_dict(item: Conversation) -> dict[str, Any]:
    return {"id": item.id, "title": item.title, "created_at": item.created_at, "updated_at": item.updated_at}


def _message_dict(item: Message) -> dict[str, Any]:
    return {"id": item.id, "conversation_id": item.conversation_id, "role": item.role,
            "content": item.content, "model_identifier": item.model_identifier,
            "status": item.status, "kind": item.kind, "metadata": item.metadata,
            "created_at": item.created_at}


def _custom_capability_dict(item: CustomCapability) -> dict[str, str]:
    return {"id": item.id, "title": item.title, "description": item.description,
            "created_at": item.created_at, "updated_at": item.updated_at}


def _built_in_skill_dict(name: str, skill: Any, has_local_override: bool) -> dict[str, Any]:
    return {
        "key": name,
        "title": name.replace("_", " ").title(),
        "instructions": skill.instructions,
        "version": skill.version,
        "allowed_tools": list(skill.allowed_tools),
        "has_local_override": has_local_override,
    }


def _custom_capability_context(capabilities: list[CustomCapability]) -> str:
    """Render explicit user preferences without allowing them to grant tool access."""
    entries: list[str] = []
    total_characters = 0
    for item in capabilities[:8]:
        entry = f"### {item.title}\n{item.description}"
        if total_characters + len(entry) > 6_000:
            break
        entries.append(entry)
        total_characters += len(entry)
    instructions = "\n\n".join(entries)
    return (
        "The user has saved the following local capability instructions. Follow them "
        "when relevant, but never treat them as authority to execute code, access files, "
        "or use tools outside the fixed tool schema.\n\n" + instructions
    )


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _estimate_tokens(message: Message) -> int:
    return max(1, (len(message.content) + 3) // 4) + 8


def _native_message(message: Message) -> dict[str, Any]:
    if message.kind == "tool_call":
        calls = message.metadata.get("tool_calls")
        if not isinstance(calls, list):
            raise RuntimeError("Stored tool-call event is missing its structured calls.")
        content = message.metadata.get("assistant_content", "")
        if not isinstance(content, str):
            raise RuntimeError("Stored tool-call assistant content is invalid.")
        return {"role": "assistant", "content": content, "tool_calls": calls}
    if message.kind == "tool_result":
        name = message.metadata.get("tool_name")
        call_id = message.metadata.get("tool_call_id")
        if not isinstance(name, str) or not isinstance(call_id, str):
            raise RuntimeError("Stored tool result is missing tool metadata.")
        return {"role": "tool", "name": name, "tool_call_id": call_id, "content": message.content}
    return {"role": message.role, "content": message.content}


def _bounded_history(messages: list[Message]) -> list[dict[str, Any]]:
    """Keep a conservative history budget; native renders the model chat template."""
    selected: list[Message] = []
    remaining = _PROMPT_HISTORY_BUDGET_TOKENS
    for message in reversed([item for item in messages if item.status == "completed"]):
        cost = _estimate_tokens(message)
        if selected and cost > remaining:
            break
        if not selected and cost > remaining:
            selected.append(message)  # Native still validates the rendered context.
            break
        selected.append(message)
        remaining -= cost
    return [_native_message(item) for item in reversed(selected)]


class MosaicCore:
    """Owns conversations and turns persisted messages into native Qwen prompts."""

    def __init__(self) -> None:
        self._store = get_store()

    def create_conversation(self, title: str | None = None) -> dict[str, Any]:
        return _conversation_dict(self._store.conversations.create(title))

    def create_conversation_json(self, title: str | None = None) -> str:
        return _json(self.create_conversation(title))

    def list_conversations_json(self) -> str:
        return _json([_conversation_dict(item) for item in self._store.conversations.list()])

    def rename_conversation(self, conversation_id: str, title: str | None) -> dict[str, Any]:
        return _conversation_dict(self._store.conversations.rename(conversation_id, title))

    def delete_conversation(self, conversation_id: str) -> None:
        self._store.conversations.delete(conversation_id)

    def get_messages_json(self, conversation_id: str) -> str:
        return _json([_message_dict(item) for item in self._store.conversations.messages(conversation_id)])

    def list_custom_capabilities_json(self) -> str:
        """Return user-authored agent instructions stored in the local SQLite ledger."""
        return _json([_custom_capability_dict(item) for item in self._store.capabilities.list()])

    def create_custom_capability_json(self, title: str, description: str) -> str:
        return _json(_custom_capability_dict(self._store.capabilities.create(title, description)))

    def update_custom_capability_json(self, capability_id: str, title: str, description: str) -> str:
        return _json(_custom_capability_dict(self._store.capabilities.update(capability_id, title, description)))

    def delete_custom_capability(self, capability_id: str) -> None:
        self._store.capabilities.delete(capability_id)

    def list_built_in_skills_json(self) -> str:
        registry = SkillRegistry()
        values = []
        for name in registry.names():
            override = self._store.skill_overrides.get(name)
            values.append(_built_in_skill_dict(name, registry.get(name, override), override is not None))
        return _json(values)

    def update_built_in_skill_json(self, name: str, instructions: str) -> str:
        registry = SkillRegistry()
        # Validate the stable internal identifier before writing any SQLite row.
        registry.get(name)
        override = self._store.skill_overrides.save(name, instructions)
        return _json(_built_in_skill_dict(name, registry.get(name, override), True))

    def reset_built_in_skill(self, name: str) -> None:
        SkillRegistry().get(name)  # validates the fixed name
        self._store.skill_overrides.delete(name)

    def list_memory_claims_json(self) -> str:
        """Return real derived claims for the UI without exposing SQLite to Kotlin."""
        claims = []
        for claim in self._store.memory.claims(status=None):
            if claim.status == "deleted":
                continue
            evidence = self._store.memory.evidence_for_claim(claim.id)
            claims.append({
                "id": claim.id,
                "fact": claim.description,
                "subject": claim.subject.name,
                "subject_type": claim.subject.entity_type,
                "predicate": claim.predicate,
                "status": claim.status,
                "assertion_type": claim.assertion_type,
                "evidence_count": len(evidence),
                "created_at": claim.created_at,
                "updated_at": claim.updated_at,
            })
        return _json(claims)

    def get_memory_claim_json(self, claim_id: str) -> str:
        """Resolve a claim and every still-valid provenance anchor for its detail view."""
        claim = next((item for item in self._store.memory.claims(status=None) if item.id == claim_id), None)
        if claim is None or claim.status == "deleted":
            raise ValueError("Memory does not exist or has been deleted.")
        evidence_items = []
        for reference in self._store.memory.evidence_for_claim(claim_id):
            try:
                resolved = self._store.memory.resolve_evidence(reference.id)
                conversation = self._store.conversations.require(resolved.message.conversation_id)
                evidence_items.append({
                    "id": reference.id,
                    "conversation_id": resolved.message.conversation_id,
                    "conversation_title": conversation.title,
                    "message_id": resolved.message.id,
                    "message_role": resolved.message.role,
                    "message_created_at": resolved.message.created_at,
                    "quote": resolved.quote,
                    "message_content": resolved.message.content,
                    "span_available": reference.span_available,
                    "context": [_message_dict(item) for item in resolved.context],
                })
            except RuntimeError as error:
                # A fingerprint mismatch must be visible as unavailable evidence,
                # never substituted with a potentially wrong quote.
                evidence_items.append({
                    "id": reference.id,
                    "conversation_id": reference.conversation_id,
                    "conversation_title": None,
                    "message_id": reference.message_id,
                    "message_role": None,
                    "message_created_at": None,
                    "quote": None,
                    "message_content": None,
                    "span_available": False,
                    "unavailable_reason": str(error),
                    "context": [],
                })
        return _json({
            "id": claim.id,
            "fact": claim.description,
            "subject": claim.subject.name,
            "subject_type": claim.subject.entity_type,
            "predicate": claim.predicate,
            "status": claim.status,
            "assertion_type": claim.assertion_type,
            "evidence_count": len(evidence_items),
            "created_at": claim.created_at,
            "updated_at": claim.updated_at,
            "evidence": evidence_items,
        })

    def delete_memory_claim(self, claim_id: str) -> None:
        """Tombstone a derived claim; authoritative conversation text is untouched."""
        self._store.memory.delete_claim(claim_id)

    def list_indexed_images_json(self) -> str:
        """Expose only real SQLite-backed image sources to the image browser."""
        values = []
        for source, embedding in self._store.embeddings.for_model(EMBEDDING_MODEL_IDENTIFIER, "image"):
            path = Path(source.reference)
            values.append({
                "id": source.id,
                "filename": path.name,
                "path": str(path) if path.is_file() else None,
                "available": path.is_file(),
                "mime_type": source.mime_type,
                "origin": source.metadata.get("origin", source.metadata.get("test_collection", "Mosaic indexed image")),
                "dimension": embedding.dimension,
            })
        return _json(values)

    def search_indexed_images_json(self, query: str, top_k: int = 12) -> str:
        """Perform a genuine EmbeddingGemma query, not filename filtering."""
        gateway = AndroidAgentGateway(get_model_paths())
        execution = ToolRegistry(self._store, gateway.embed_query).execute(
            ToolCall("image-browser", "search_images", {"query": query, "top_k": min(max(int(top_k), 1), 10)})
        )
        if not execution.ok:
            raise RuntimeError(str(execution.payload.get("error", "Image search failed.")))
        by_id = {source.id: source for source, _ in self._store.embeddings.for_model(EMBEDDING_MODEL_IDENTIFIER, "image")}
        results = []
        for item in execution.payload.get("results", []):
            if not isinstance(item, dict) or not isinstance(item.get("source_id"), str):
                continue
            source = by_id.get(item["source_id"])
            if source is None:
                continue
            path = Path(source.reference)
            results.append({
                "id": source.id, "filename": path.name, "path": str(path) if path.is_file() else None,
                "available": path.is_file(), "mime_type": source.mime_type,
                "origin": source.metadata.get("origin", source.metadata.get("test_collection", "Mosaic indexed image")),
                "similarity": item.get("similarity"),
            })
        return _json({"results": results, "elapsed_ms": execution.elapsed_ms})

    def index_image_json(self, image_path: str, mime_type: str | None = None) -> str:
        """Persist a real image embedding for an app-private image selected by the user.

        Kotlin copies the Photo Picker URI into private storage before calling this
        method. Python owns the durable source record and embedding; it never
        derives a vector from a filename or caller-provided description.
        """
        selected = Path(image_path).resolve()
        if not selected.is_file():
            raise ValueError("The selected image is no longer available.")
        if selected.stat().st_size <= 0:
            raise ValueError("The selected image is empty.")
        digest = sha256()
        with selected.open("rb") as source_file:
            while block := source_file.read(1024 * 1024):
                digest.update(block)
        fingerprint = digest.hexdigest()
        gateway = AndroidAgentGateway(get_model_paths())
        vector, metrics = gateway.embed_image(str(selected))
        source = self._store.sources.register(
            "image", str(selected), fingerprint, mime_type=mime_type,
            metadata={"origin": "User-selected image"},
        )
        embedding = self._store.embeddings.save(
            source, EMBEDDING_MODEL_IDENTIFIER, vector, normalized=True,
        )
        return _json({
            "id": source.id,
            "filename": selected.name,
            "path": str(selected),
            "available": True,
            "mime_type": source.mime_type,
            "origin": source.metadata.get("origin", "Mosaic indexed image"),
            "dimension": embedding.dimension,
            "metrics": metrics,
        })

    def local_status_json(self) -> str:
        """Small, portable status payload for UI transparency; no native calls required."""
        paths = get_model_paths()
        model_paths = {
            "qwen": paths.qwen,
            "embeddinggemma": paths.embedding,
            "embedding_projector": paths.projector,
            "qwen_projector": paths.qwen_projector,
        }
        return _json({
            "database_path": self._store.database.path,
            "database_bytes": Path(self._store.database.path).stat().st_size if Path(self._store.database.path).exists() else 0,
            "conversations": len(self._store.conversations.list()),
            "memories": len(self._store.memory.claims(status="current")),
            "indexed_images": len(self._store.embeddings.for_model(EMBEDDING_MODEL_IDENTIFIER, "image")),
            "models": {name: {"present": Path(path).is_file(), "bytes": Path(path).stat().st_size if Path(path).is_file() else 0}
                       for name, path in model_paths.items()},
        })

    def process_message(self, conversation_id: str, message: str) -> dict[str, Any]:
        return self._process_message(conversation_id, message, None, None)

    def process_message_stream(self, conversation_id: str, message: str, event_sink: Any) -> dict[str, Any]:
        """Run a real request while forwarding typed, user-safe runtime events."""
        emitter = RequestEventEmitter(str(uuid4()), event_sink)
        emitter.emit("request_started", conversation_id=conversation_id)
        return self._process_message(conversation_id, message, emitter, event_sink)

    def process_image_message_stream(self, conversation_id: str, message: str, image_path: str, event_sink: Any) -> dict[str, Any]:
        """Run a real Qwen vision request for a Kotlin-imported app-private image.

        Vision's current JNI entry point is complete-response only. Events report
        real model phases, while the final answer is withheld until native
        generation returns instead of pretending to stream tokens.
        """
        paths = get_model_paths()
        root = Path(paths.debug_images).resolve()
        selected = Path(image_path).resolve()
        if root not in selected.parents or not selected.is_file():
            raise ValueError("Selected image is unavailable or outside Mosaic's private attachment storage.")
        return self._process_verified_image_stream(
            conversation_id, message, selected, event_sink,
            {"attachment_name": selected.name, "attachment_kind": "image"},
        )

    def process_indexed_image_message_stream(self, conversation_id: str, message: str,
                                             source_id: str, event_sink: Any) -> dict[str, Any]:
        """Ask vision about an image explicitly selected from Mosaic's own index.

        Kotlin supplies only the opaque source ID. The authoritative SQLite source
        record resolves the path here, so neither UI text nor model output can
        cause arbitrary filesystem access.
        """
        source = next((item for item, _ in self._store.embeddings.for_model(
            EMBEDDING_MODEL_IDENTIFIER, "image") if item.id == source_id), None)
        if source is None:
            raise ValueError("The indexed image is no longer available.")
        selected = Path(source.reference).resolve()
        if not selected.is_file():
            raise ValueError("The indexed image source is no longer available.")
        return self._process_verified_image_stream(
            conversation_id, message, selected, event_sink,
            {"attachment_name": selected.name, "attachment_kind": "indexed_image", "source_id": source.id},
        )

    def _process_verified_image_stream(self, conversation_id: str, message: str, selected: Path,
                                       event_sink: Any, attachment_metadata: dict[str, Any]) -> dict[str, Any]:
        """Shared real Qwen vision flow for a picked or a stored approved image."""
        emitter = RequestEventEmitter(str(uuid4()), event_sink)
        emitter.emit("request_started", conversation_id=conversation_id)
        prompt = message.strip() or "Describe this image."
        self._store.conversations.require(conversation_id)
        paths = get_model_paths()
        user_message = self._store.conversations.add_message(
            conversation_id, "user", f"[Image: {selected.name}]\n{prompt}",
            metadata=attachment_metadata,
        )
        try:
            emitter.emit("model_loading", model="Qwen vision")
            emitter.emit("prompt_processing", display_text="Analyzing selected image")
            answer, metrics = AndroidAgentGateway(paths).generate_with_image(prompt, str(selected))
            emitter.emit("answer_started")
            assistant_message = self._store.conversations.add_message(
                conversation_id, "assistant", answer, QWEN_MODEL_IDENTIFIER, status="completed",
                metadata={"vision_attachment_name": selected.name, "vision_metrics": metrics},
            )
            emitter.emit("answer_completed", message_id=assistant_message.id)
        except Exception:
            error = sys.exc_info()[1]
            emitter.emit("request_failed", error_type=type(error).__name__ if error else "RuntimeError",
                         message=str(error) if error else "Unknown image request failure")
            raise
        return {
            "kind": "mosaic.local_vision", "request_id": emitter.request_id,
            "conversation_id": conversation_id, "user_message_id": user_message.id,
            "assistant_message": _message_dict(assistant_message), "metrics": metrics,
        }

    def _process_message(self, conversation_id: str, message: str,
                         emitter: RequestEventEmitter | None, event_sink: Any | None) -> dict[str, Any]:
        normalized = message.strip()
        if not normalized:
            raise ValueError("A message is required")
        self._store.conversations.require(conversation_id)
        user_message = self._store.conversations.add_message(conversation_id, "user", normalized)
        prompt_messages = _bounded_history(self._store.conversations.messages(conversation_id, include_incomplete=False))
        registry = SkillRegistry()
        responder_override = self._store.skill_overrides.get("responder")
        responder_skill = registry.get("responder", responder_override)
        # The static responder rules are an actual part of the model prompt;
        # editing a built-in skill therefore changes the relevant local task,
        # while Python still owns the tool allow-list below.
        prompt_messages.insert(0, {"role": "system", "content": responder_skill.instructions})
        custom_capabilities = self._store.capabilities.list()
        if custom_capabilities:
            prompt_messages.insert(0, {
                "role": "system",
                "content": _custom_capability_context(custom_capabilities),
            })
        try:
            paths = get_model_paths()
            # The event sink is also a Kotlin NativeTokenListener. Keeping it
            # outside a Python dynamic proxy avoids re-entering Chaquopy while
            # the synchronous JNI generation call is active.
            gateway = AndroidAgentGateway(paths, emitter.emit if emitter else None, event_sink if emitter else None)
            agent = AgentLoop(
                gateway, ToolRegistry(self._store, gateway.embed_query, configured_web_client()),
                allowed_tools=responder_skill.allowed_tools,
                on_tool_round=lambda reply, executions: self._persist_tool_round(conversation_id, reply, executions),
            )
            answer_started = False

            def answer_delta(piece: str) -> None:
                nonlocal answer_started
                if not emitter:
                    return
                if not answer_started:
                    answer_started = True
                    emitter.emit("answer_started")
                emitter.emit("answer_delta", delta=piece)

            def tool_activity(state: str, call: Any, execution: Any) -> None:
                if not emitter:
                    return
                names = {
                    "search_memories": "Searching personal memories",
                    "search_images": "Searching indexed images",
                    "web_search": "Searching the web",
                    "web_fetch": "Reading webpage",
                }
                metadata: dict[str, Any] = {"tool": call.name, "display_name": names.get(call.name, "Using a local tool")}
                if execution and execution.ok:
                    results = execution.payload.get("results")
                    result_count = execution.payload.get("result_count")
                    if not isinstance(result_count, int) and isinstance(results, list):
                        result_count = len(results)
                    if isinstance(result_count, int):
                        metadata["result_count"] = result_count
                        if call.name == "web_search":
                            metadata["display_text"] = f"Found {result_count} source{'s' if result_count != 1 else ''}"
                        elif call.name == "web_fetch":
                            metadata["display_text"] = "Read webpage"
                        else:
                            metadata["display_text"] = f"Found {result_count} record{'s' if result_count != 1 else ''}"
                    summaries: list[dict[str, Any]] = []
                    if call.name == "search_images" and isinstance(results, list):
                        for item in results[:5]:
                            if isinstance(item, dict) and isinstance(item.get("source_id"), str) and isinstance(item.get("filename"), str):
                                summaries.append({"kind": "image", "id": item["source_id"], "label": item["filename"]})
                    if call.name == "search_memories" and isinstance(results, list):
                        for item in results[:5]:
                            if isinstance(item, dict) and isinstance(item.get("claim_id"), str) and isinstance(item.get("fact"), str):
                                summaries.append({"kind": "memory", "id": item["claim_id"], "label": item["fact"]})
                    if call.name == "web_search" and isinstance(results, list):
                        for item in results[:4]:
                            if isinstance(item, dict) and isinstance(item.get("url"), str):
                                url = item["url"]
                                title = item.get("title") if isinstance(item.get("title"), str) else url
                                summaries.append({"kind": "web", "id": sha256(url.encode("utf-8")).hexdigest(),
                                                  "label": title, "url": url})
                    if call.name == "web_fetch" and isinstance(execution.payload.get("url"), str):
                        url = execution.payload["url"]
                        title = execution.payload.get("title") if isinstance(execution.payload.get("title"), str) else url
                        summaries.append({"kind": "web", "id": sha256(url.encode("utf-8")).hexdigest(),
                                          "label": title, "url": url})
                    if summaries:
                        metadata["source_summaries"] = summaries
                emitter.emit(f"tool_{state}", **metadata)

            agent_result = agent.run_streaming(prompt_messages, answer_delta if emitter else None, tool_activity if emitter else None)
            assistant_message = self._store.conversations.add_message(
                conversation_id, "assistant", agent_result.answer, QWEN_MODEL_IDENTIFIER, status="completed"
            )
            if emitter:
                # An empty answer is rejected by AgentLoop, so this represents
                # a real persisted completion rather than an optimistic UI state.
                emitter.emit("answer_completed", message_id=assistant_message.id)
        except GenerationCancelled as error:
            if emitter:
                emitter.emit("request_cancelled", message=str(error))
                return {
                    "kind": "mosaic.local_inference", "request_id": emitter.request_id,
                    "conversation_id": conversation_id, "user_message_id": user_message.id,
                    "status": "cancelled",
                }
            raise
        except Exception:
            # A failed or interrupted generation is never persisted as completed.
            if emitter:
                error = sys.exc_info()[1]
                emitter.emit("request_failed", error_type=type(error).__name__ if error else "RuntimeError",
                             message=str(error) if error else "Unknown request failure")
            raise
        return {
            "kind": "mosaic.local_inference", "conversation_id": conversation_id,
            **({"request_id": emitter.request_id} if emitter else {}),
            "user_message_id": user_message.id, "assistant_message": _message_dict(assistant_message),
            "provider": "llama.cpp via AndroidInferenceProvider",
            "history_messages_sent": len(prompt_messages),
            "agent": {
                "tool_calls": [call.to_message() for call in agent_result.tool_calls],
                "tool_results": [execution.payload for execution in agent_result.tool_executions],
                "native_replies": [
                    {"content": reply.content, "tool_calls": [call.to_message() for call in reply.tool_calls],
                     "raw_output": reply.raw_output, "parser": reply.parser, "metrics": dict(reply.metrics)}
                    for reply in agent_result.native_replies
                ],
                "total_latency_ms": agent_result.total_latency_ms,
            },
            "python": {"implementation": platform.python_implementation(), "version": sys.version.split()[0]},
        }

    def _persist_tool_round(self, conversation_id: str, reply: Any, executions: Any) -> None:
        """Persist structured agent events without treating them as chat replies."""
        by_call_id = {item.call.id: item for item in executions}
        calls = [item.to_message() for item in reply.tool_calls]
        self._store.conversations.add_message(
            conversation_id, "assistant", _json(calls), QWEN_MODEL_IDENTIFIER, kind="tool_call",
            metadata={"tool_calls": calls, "assistant_content": reply.content,
                      "raw_output": reply.raw_output, "parser": reply.parser},
        )
        for call in reply.tool_calls:
            execution = by_call_id[call.id]
            self._store.conversations.add_message(
                conversation_id, "tool", _json(execution.payload), kind="tool_result",
                metadata={"tool_name": call.name, "tool_call_id": call.id,
                          "elapsed_ms": execution.elapsed_ms, "ok": execution.ok},
            )
