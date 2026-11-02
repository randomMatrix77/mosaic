from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from mosaic.core import MosaicCore, _bounded_history, _custom_capability_context
from mosaic.storage import configure_database


class AgentEventStorageTests(unittest.TestCase):
    def test_custom_capability_persists_and_is_bounded_agent_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = configure_database(str(Path(directory) / "mosaic.sqlite3"))
            saved = store.capabilities.create("Recipe helper", "Prefer concise vegetarian recipes.")
            self.assertEqual([saved], store.capabilities.list())
            self.assertIn("Recipe helper", _custom_capability_context(store.capabilities.list()))
            updated = store.capabilities.update(saved.id, "Recipe assistant", "Prefer concise vegan recipes.")
            self.assertEqual(saved.id, updated.id)
            self.assertEqual("Recipe assistant", updated.title)
            self.assertEqual("Prefer concise vegan recipes.", updated.description)
            store.skill_overrides.save("responder", "Use terse local answers.")
            self.assertEqual("Use terse local answers.", store.skill_overrides.get("responder"))
            store.skill_overrides.delete("responder")
            self.assertIsNone(store.skill_overrides.get("responder"))
            with self.assertRaisesRegex(ValueError, "already exists"):
                store.capabilities.create("recipe ASSISTANT", "Duplicate name")
            store.capabilities.delete(saved.id)
            self.assertEqual([], store.capabilities.list())

    def test_built_in_skills_are_exposed_and_overridable_without_new_tools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            configure_database(str(Path(directory) / "mosaic.sqlite3"))
            core = MosaicCore()
            packaged = json.loads(core.list_built_in_skills_json())
            responder = next(item for item in packaged if item["key"] == "responder")
            self.assertEqual(["search_images", "search_memories", "web_search", "web_fetch"], responder["allowed_tools"])
            updated = json.loads(core.update_built_in_skill_json("responder", "Answer with short local replies."))
            self.assertTrue(updated["has_local_override"])
            self.assertEqual(["search_images", "search_memories", "web_search", "web_fetch"], updated["allowed_tools"])
            self.assertEqual("Answer with short local replies.", updated["instructions"])
            core.reset_built_in_skill("responder")
            restored = next(item for item in json.loads(core.list_built_in_skills_json()) if item["key"] == "responder")
            self.assertFalse(restored["has_local_override"])

    def test_tool_events_round_trip_into_native_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = configure_database(str(Path(directory) / "mosaic.sqlite3"))
            conversation = store.conversations.create("Agent history")
            store.conversations.add_message(conversation.id, "user", "Find a banana")
            tool_call = {
                "id": "call-1", "type": "function",
                "function": {"name": "search_images", "arguments": {"query": "banana"}},
            }
            store.conversations.add_message(
                conversation.id, "assistant", json.dumps([tool_call]), kind="tool_call",
                metadata={"tool_calls": [tool_call], "assistant_content": "", "parser": "test"},
            )
            store.conversations.add_message(
                conversation.id, "tool", '{"ok":true,"results":[{"filename":"banana.jpg"}]}', kind="tool_result",
                metadata={"tool_name": "search_images", "tool_call_id": "call-1", "ok": True},
            )
            store.conversations.add_message(conversation.id, "assistant", "The match is banana.jpg.")
            messages = store.conversations.messages(conversation.id, include_incomplete=False)
            history = _bounded_history(messages)
            self.assertEqual("assistant", history[1]["role"])
            self.assertEqual("search_images", history[1]["tool_calls"][0]["function"]["name"])
            self.assertEqual("tool", history[2]["role"])
            self.assertEqual("call-1", history[2]["tool_call_id"])

    def test_v1_database_migrates_without_erasing_messages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.sqlite3"
            connection = sqlite3.connect(path)
            connection.executescript(
                """
                CREATE TABLE conversations (id TEXT PRIMARY KEY, title TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant')),
                    content TEXT NOT NULL,
                    model_identifier TEXT,
                    status TEXT NOT NULL DEFAULT 'completed' CHECK (status IN ('completed', 'failed', 'pending')),
                    created_at TEXT NOT NULL
                );
                CREATE INDEX messages_conversation_chronological ON messages(conversation_id, created_at, id);
                CREATE TABLE sources (id TEXT PRIMARY KEY, modality TEXT NOT NULL, reference TEXT NOT NULL, mime_type TEXT,
                    content_fingerprint TEXT NOT NULL, metadata_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
                    UNIQUE(modality, reference, content_fingerprint));
                CREATE TABLE embeddings (id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                    model_identifier TEXT NOT NULL, content_fingerprint TEXT NOT NULL, dimension INTEGER NOT NULL CHECK (dimension > 0),
                    normalized INTEGER NOT NULL CHECK (normalized IN (0, 1)), vector_f32 BLOB NOT NULL, created_at TEXT NOT NULL,
                    UNIQUE(source_id, model_identifier, content_fingerprint));
                INSERT INTO conversations VALUES ('conversation', 'Legacy', 'now', 'now');
                INSERT INTO messages VALUES ('message', 'conversation', 'user', 'Persist me', NULL, 'completed', 'now');
                PRAGMA user_version = 1;
                """
            )
            connection.commit()
            connection.close()
            store = configure_database(str(path))
            message = store.conversations.messages("conversation")[0]
            self.assertEqual("Persist me", message.content)
            self.assertEqual("chat", message.kind)
            with sqlite3.connect(path) as upgraded:
                self.assertEqual(5, upgraded.execute("PRAGMA user_version").fetchone()[0])
                self.assertIsNotNone(upgraded.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'custom_capabilities'"
                ).fetchone())


if __name__ == "__main__":
    unittest.main()
