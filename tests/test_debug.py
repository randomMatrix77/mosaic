from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mosaic.debug import run_command_json
from mosaic.storage import configure_database


class DebugCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        configure_database(str(Path(self.temp.name) / "mosaic.sqlite3"))

    def tearDown(self) -> None:
        self.temp.cleanup()

    @staticmethod
    def invoke(request: dict) -> dict:
        return json.loads(run_command_json(json.dumps(request)))

    def test_status_lists_explicit_commands(self) -> None:
        result = self.invoke({"operation": "status"})
        self.assertTrue(result["ok"])
        self.assertIn("agent.chat", result["commands"])

    def test_create_list_and_read_conversation(self) -> None:
        created = self.invoke({"operation": "conversation.create", "title": "ADB test"})
        self.assertTrue(created["ok"])
        conversation_id = created["conversation"]["id"]
        listed = self.invoke({"operation": "conversation.list"})
        self.assertEqual(conversation_id, listed["conversations"][0]["id"])
        messages = self.invoke({"operation": "conversation.messages", "conversation_id": conversation_id})
        self.assertEqual([], messages["messages"])

    def test_invalid_or_unsupported_requests_are_structured_errors(self) -> None:
        malformed = json.loads(run_command_json("[]"))
        self.assertFalse(malformed["ok"])
        unsupported = self.invoke({"operation": "shell.execute"})
        self.assertFalse(unsupported["ok"])
        self.assertIn("Unsupported debug operation", unsupported["error"]["message"])


if __name__ == "__main__":
    unittest.main()
