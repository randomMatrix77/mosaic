"""Python-owned application logic for Mosaic chat and persistence."""

from __future__ import annotations

import json
import platform
import sys
from typing import Any

from mosaic.inference import AndroidInferenceProvider
from mosaic.storage import Conversation, Message, configure_database, get_store

QWEN_MODEL_IDENTIFIER = "qwen3.5-4b-q4_k_m"
_PROMPT_HISTORY_BUDGET_TOKENS = 1400


def configure_storage(database_path: str) -> dict[str, Any]:
    """Called by Kotlin once it has chosen an app-private database path."""
    store = configure_database(database_path)
    return {"database_path": store.database.path, "schema_version": 1}


def _conversation_dict(item: Conversation) -> dict[str, Any]:
    return {"id": item.id, "title": item.title, "created_at": item.created_at, "updated_at": item.updated_at}


def _message_dict(item: Message) -> dict[str, Any]:
    return {"id": item.id, "conversation_id": item.conversation_id, "role": item.role,
            "content": item.content, "model_identifier": item.model_identifier,
            "status": item.status, "created_at": item.created_at}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _estimate_tokens(message: Message) -> int:
    return max(1, (len(message.content) + 3) // 4) + 8


def _bounded_history(messages: list[Message]) -> list[dict[str, str]]:
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
    return [{"role": item.role, "content": item.content} for item in reversed(selected)]


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

    def process_message(self, conversation_id: str, message: str) -> dict[str, Any]:
        normalized = message.strip()
        if not normalized:
            raise ValueError("A message is required")
        self._store.conversations.require(conversation_id)
        user_message = self._store.conversations.add_message(conversation_id, "user", normalized)
        provider = AndroidInferenceProvider()
        if not provider.is_loaded:
            raise RuntimeError("Import the Qwen GGUF before sending a conversation message.")
        prompt_messages = _bounded_history(self._store.conversations.messages(conversation_id, include_incomplete=False))
        try:
            answer = provider.generate_chat(prompt_messages)
            assistant_message = self._store.conversations.add_message(
                conversation_id, "assistant", answer, QWEN_MODEL_IDENTIFIER, status="completed"
            )
        except Exception:
            # A failed or interrupted generation is never persisted as completed.
            raise
        return {
            "kind": "mosaic.local_inference", "conversation_id": conversation_id,
            "user_message_id": user_message.id, "assistant_message": _message_dict(assistant_message),
            "provider": "llama.cpp via AndroidInferenceProvider",
            "history_messages_sent": len(prompt_messages),
            "python": {"implementation": platform.python_implementation(), "version": sys.version.split()[0]},
        }
