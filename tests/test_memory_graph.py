from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

import mosaic.core as core_module
from mosaic.agent import AgentLoop, ToolRegistry
from mosaic.core import MosaicCore
from mosaic.inference import ModelPaths, configure_model_paths
from mosaic.memory import MemoryProtocolError, MemoryService, parse_extraction, parse_extraction_tool_arguments
from mosaic.skill_registry import SkillRegistry
from mosaic.storage import configure_database


class ExtractorDouble:
    """Explicit local test double: it never calls Qwen or EmbeddingGemma."""

    def __init__(self, output: str, vectors: dict[str, list[float]] | None = None) -> None:
        self.output = output
        self.vectors = vectors or {}

    def generate_skill(self, messages: list[dict[str, Any]]) -> tuple[str, Mapping[str, float]]:
        return self.output, {"test_extractor_ms": 1.0}

    def embed_query(self, text: str) -> tuple[list[float], dict[str, float]]:
        return self.vectors.get(text, [1.0, 0.0]), {"test_embedding_ms": 1.0}


class GraphGateway:
    def __init__(self, replies: list[dict[str, Any]]) -> None:
        self.replies = replies

    def generate(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Mapping[str, Any]:
        return self.replies.pop(0)

    def embed_query(self, text: str) -> tuple[list[float], dict[str, float]]:
        return [1.0, 0.0], {"test_embedding_ms": 1.0}


class MemoryGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.store = configure_database(str(Path(self.temp.name) / "mosaic.sqlite3"))

    def tearDown(self) -> None:
        self.temp.cleanup()

    def message(self, title: str, content: str):
        conversation = self.store.conversations.create(title)
        return conversation, self.store.conversations.add_message(conversation.id, "user", content)

    def add_claim(self, message, literal: str, start: int | None = None, end: int | None = None):
        evidence: dict[str, Any] = {"conversation_id": message.conversation_id, "message_id": message.id}
        if start is not None:
            evidence.update(start_offset=start, end_offset=end)
        return self.store.memory.add_claim(subject_name="Mosaic", subject_type="project", predicate="uses",
                                           literal_value=literal, evidence=[evidence], embedding=[1.0, 0.0],
                                           embedding_model="embeddinggemma-2-gguf")

    def test_one_message_can_support_multiple_claims_and_resolve_span(self) -> None:
        _, message = self.message("one", "Mosaic uses Python and Kotlin.")
        first = self.add_claim(message, "Python", 12, 18)
        second = self.store.memory.add_claim(subject_name="Mosaic", subject_type="project", predicate="uses",
                                             literal_value="Kotlin", evidence=[{"message_id": message.id, "start_offset": 23, "end_offset": 29}])
        self.assertEqual(1, len(self.store.memory.evidence_for_claim(first.id)))
        reference = self.store.memory.evidence_for_claim(second.id)[0]
        resolved = self.store.memory.resolve_evidence(reference.id)
        self.assertEqual("Kotlin", resolved.quote)
        self.assertEqual(message.id, resolved.message.id)

    def test_core_exposes_claims_and_resolved_evidence_without_leaking_sqlite_to_ui(self) -> None:
        conversation, message = self.message("provenance", "Mosaic uses Python and Kotlin.")
        claim = self.add_claim(message, "Python", 12, 18)
        core = MosaicCore()
        listed = json.loads(core.list_memory_claims_json())
        self.assertEqual(claim.id, listed[0]["id"])
        self.assertEqual("Mosaic uses Python", listed[0]["fact"])
        detail = json.loads(core.get_memory_claim_json(claim.id))
        self.assertEqual(conversation.id, detail["evidence"][0]["conversation_id"])
        self.assertEqual(message.id, detail["evidence"][0]["message_id"])
        self.assertEqual("Python", detail["evidence"][0]["quote"])

    def test_core_lists_only_real_sqlite_backed_indexed_images(self) -> None:
        source = self.store.sources.register("image", "/missing/cat.jpg", "cat-fingerprint", metadata={"test_collection": "fixture"})
        self.store.embeddings.save(source, "embeddinggemma-2-gguf", [1.0, 0.0], normalized=True)
        images = json.loads(MosaicCore().list_indexed_images_json())
        self.assertEqual(1, len(images))
        self.assertEqual("cat.jpg", images[0]["filename"])
        self.assertFalse(images[0]["available"])
        self.assertEqual(2, images[0]["dimension"])

    def test_user_selected_image_is_embedded_and_persisted_by_python(self) -> None:
        """Use an explicit test double only: production invokes Android JNI here."""
        image_path = Path(self.temp.name) / "selected.jpg"
        image_path.write_bytes(b"real selected-image fixture bytes")
        configure_model_paths(ModelPaths("", "", ""))
        captured: dict[str, Any] = {}

        class ImageGatewayDouble:
            def __init__(self, _paths: ModelPaths) -> None:
                pass

            def embed_image(self, path: str):
                captured["path"] = path
                return [0.6, 0.8], {"test_embedding_image_ms": 1.0}

        original_gateway = core_module.AndroidAgentGateway
        core_module.AndroidAgentGateway = ImageGatewayDouble  # type: ignore[assignment]
        try:
            indexed = json.loads(MosaicCore().index_image_json(str(image_path), "image/jpeg"))
        finally:
            core_module.AndroidAgentGateway = original_gateway  # type: ignore[assignment]

        self.assertEqual(str(image_path.resolve()), captured["path"])
        self.assertTrue(indexed["available"])
        self.assertEqual("User-selected image", indexed["origin"])
        stored = self.store.embeddings.for_model("embeddinggemma-2-gguf", "image")
        self.assertEqual(1, len(stored))
        self.assertEqual(2, len(stored[0][1].vector))
        self.assertAlmostEqual(0.6, stored[0][1].vector[0], places=6)
        self.assertAlmostEqual(0.8, stored[0][1].vector[1], places=6)

    def test_indexed_image_vision_resolves_only_an_approved_sqlite_source(self) -> None:
        image_path = Path(self.temp.name) / "approved-cat.jpg"
        image_path.write_bytes(b"fixture image bytes")
        source = self.store.sources.register("image", str(image_path), "approved-fingerprint")
        self.store.embeddings.save(source, "embeddinggemma-2-gguf", [1.0, 0.0], normalized=True)
        core = MosaicCore()
        captured: dict[str, Any] = {}

        def verified(conversation_id, message, selected, event_sink, metadata):
            captured.update(conversation_id=conversation_id, message=message, selected=selected, metadata=metadata)
            return {"kind": "test"}

        core._process_verified_image_stream = verified  # type: ignore[method-assign]
        result = core.process_indexed_image_message_stream("conversation-id", "Describe it", source.id, object())
        self.assertEqual({"kind": "test"}, result)
        self.assertEqual(image_path.resolve(), captured["selected"])
        self.assertEqual(source.id, captured["metadata"]["source_id"])
        with self.assertRaisesRegex(ValueError, "no longer available"):
            core.process_indexed_image_message_stream("conversation-id", "Describe it", "fabricated-source", object())

    def test_one_claim_can_have_evidence_from_multiple_conversations(self) -> None:
        _, first = self.message("first", "Mosaic uses Python.")
        _, second = self.message("second", "The Mosaic backend is Python.")
        claim = self.store.memory.add_claim(subject_name="Mosaic", subject_type="project", predicate="uses",
                                            literal_value="Python", evidence=[{"message_id": first.id}, {"message_id": second.id}])
        evidence = self.store.memory.evidence_for_claim(claim.id)
        self.assertEqual({first.conversation_id, second.conversation_id}, {item.conversation_id for item in evidence})
        self.assertTrue(all(not item.span_available for item in evidence))

    def test_invalid_message_and_span_are_rejected_without_claim(self) -> None:
        _, message = self.message("source", "short text")
        with self.assertRaisesRegex(ValueError, "outside"):
            self.add_claim(message, "Python", 0, 999)
        with self.assertRaisesRegex(ValueError, "does not exist"):
            self.store.memory.add_claim(subject_name="Mosaic", subject_type="project", predicate="uses", literal_value="Python",
                                        evidence=[{"message_id": "fabricated"}])
        self.assertEqual([], self.store.memory.claims())

    def test_correction_preserves_old_claim_as_superseded(self) -> None:
        _, old_message = self.message("old", "Mosaic uses Runtime A.")
        old = self.add_claim(old_message, "Runtime A")
        _, new_message = self.message("new", "Mosaic now uses Runtime B.")
        replacement = self.add_claim(new_message, "Runtime B")
        self.store.memory.supersede(old.id, replacement.id)
        self.assertEqual("superseded", self.store.memory.claims(status=None)[0].status)
        self.assertEqual([replacement.id], [item.id for item in self.store.memory.claims()])
        self.assertEqual(1, len(self.store.memory.evidence_for_claim(old.id)))

    def test_curator_can_only_apply_validated_supersession_or_evidence_link(self) -> None:
        _, old_message = self.message("old", "Mosaic uses Runtime A.")
        old = self.add_claim(old_message, "Runtime A")
        _, new_message = self.message("new", "Mosaic uses Runtime B.")
        replacement = self.add_claim(new_message, "Runtime B")
        curated = MemoryService(self.store).curate(ExtractorDouble(json.dumps({"operations": [
            {"op": "supersede", "old_claim_id": old.id, "replacement_claim_id": replacement.id},
        ]})))
        self.assertEqual("superseded", next(item for item in self.store.memory.claims(status=None) if item.id == old.id).status)
        self.assertEqual("supersede", curated.applied_operations[0]["op"])

    def test_conversation_and_memory_deletion_do_not_leave_or_resurrect_evidence(self) -> None:
        conversation, message = self.message("privacy", "Mosaic uses Python.")
        claim = self.add_claim(message, "Python")
        self.store.conversations.delete(conversation.id)
        deleted = next(item for item in self.store.memory.claims(status="deleted") if item.id == claim.id)
        self.assertEqual("deleted", deleted.status)
        self.assertEqual([], self.store.memory.evidence_for_claim(claim.id))
        _, retained_message = self.message("retained", "Mosaic uses Kotlin.")
        retained = self.add_claim(retained_message, "Kotlin")
        self.store.memory.delete_claim(retained.id)
        with self.assertRaisesRegex(ValueError, "deleted"):
            self.store.memory.add_claim(subject_name="Mosaic", subject_type="project", predicate="uses",
                                        literal_value="Kotlin", evidence=[{"message_id": retained_message.id}])

    def test_extractor_rejects_invalid_output_before_writes_and_tracks_processed_messages(self) -> None:
        conversation, message = self.message("extract", "Mosaic uses Python.")
        bad = ExtractorDouble(json.dumps({"claims": [{"subject": {"name": "Mosaic"}, "predicate": "uses",
                                                       "literal_value": "Python", "evidence": [{"message_id": "fake"}]}]}))
        with self.assertRaises(MemoryProtocolError):
            MemoryService(self.store).ingest_conversation(conversation.id, bad)
        self.assertEqual([], self.store.memory.claims())
        output = json.dumps({"claims": [{"subject": {"name": "Mosaic", "type": "project"}, "predicate": "uses",
                                           "literal_value": "Python", "assertion_type": "explicit",
                                           "evidence": [{"message_id": message.id, "start_offset": 12, "end_offset": 18}]}]})
        result = MemoryService(self.store).ingest_conversation(conversation.id, ExtractorDouble(output))
        self.assertEqual(1, len(result.claims))
        repeated = MemoryService(self.store).ingest_conversation(conversation.id, ExtractorDouble(output))
        self.assertEqual((), repeated.claims)

    def test_extractor_refuses_obvious_contact_data_without_consent(self) -> None:
        conversation, message = self.message("private", "My email is alex@example.com.")
        output = json.dumps({"claims": [{"subject": {"name": "Alex", "type": "person"}, "predicate": "email",
                                           "literal_value": "alex@example.com", "assertion_type": "explicit",
                                           "evidence": [{"message_id": message.id}]}]})
        with self.assertRaisesRegex(MemoryProtocolError, "sensitive"):
            MemoryService(self.store).ingest_conversation(conversation.id, ExtractorDouble(output))
        self.assertEqual([], self.store.memory.claims())

    def test_cross_conversation_retrieval_and_agent_tool(self) -> None:
        _, first = self.message("one", "Mosaic uses Python.")
        self.add_claim(first, "Python")
        _, second = self.message("two", "Mosaic uses Kotlin.")
        self.add_claim(second, "Kotlin")
        result = MemoryService(self.store).search("What does Mosaic use?", ExtractorDouble(""), 5)
        self.assertEqual(2, len(result["results"]))
        gateway = GraphGateway([
            {"role": "assistant", "content": "", "tool_calls": [{"id": "memory-1", "type": "function", "function": {"name": "search_memories", "arguments": {"query": "Mosaic technology"}}}]},
            {"role": "assistant", "content": "Mosaic uses Python and Kotlin."},
        ])
        loop = AgentLoop(gateway, ToolRegistry(self.store, gateway.embed_query), allowed_tools=("search_memories",))
        answer = loop.run([{"role": "user", "content": "What technologies am I using?"}])
        self.assertTrue(answer.tool_executions[0].ok)
        self.assertEqual("Mosaic uses Python and Kotlin.", answer.answer)

    def test_skill_files_are_loaded_and_permissions_are_fixed(self) -> None:
        registry = SkillRegistry()
        self.assertEqual(("search_images", "search_memories", "web_search", "web_fetch"), registry.get("responder").allowed_tools)
        self.assertEqual((), registry.get("memory_extractor").allowed_tools)
        with self.assertRaises(ValueError):
            registry.get("not-a-skill")

    def test_extraction_parser_accepts_one_json_object_after_qwen_thinking_wrapper(self) -> None:
        self.assertEqual((), parse_extraction("<think>reasoning</think>\n{\"claims\": []}"))
        with self.assertRaises(MemoryProtocolError):
            parse_extraction('{"claims": []} trailing')

    def test_native_extractor_tool_arguments_are_normalized_and_strict(self) -> None:
        proposal = parse_extraction_tool_arguments({"claims": [{
            "subject_name": "Mosaic", "subject_type": "project", "predicate": "uses",
            "literal_value": "Python", "assertion_type": "explicit", "evidence_message_ids": ["message-id"],
        }]})[0]
        self.assertEqual("Mosaic uses Python", proposal.description)
        self.assertEqual("message-id", proposal.evidence[0]["message_id"])
        synonym = parse_extraction_tool_arguments({"claims": [{
            "subject_name": "Mosaic", "subject_type": "project", "predicate": "uses",
            "literal_value": "Python", "assertion_type": "fact", "evidence_message_ids": ["message-id"],
        }]})[0]
        self.assertEqual("explicit", synonym.assertion_type)

    def test_v2_ledger_migrates_to_graph_schema_without_erasing_messages(self) -> None:
        conversation, message = self.message("legacy", "Keep this ledger message.")
        path = Path(self.temp.name) / "mosaic.sqlite3"
        with sqlite3.connect(path) as connection:
            connection.executescript("""
                DROP TABLE claim_embeddings;
                DROP TABLE claim_evidence;
                DROP TABLE processed_memory_messages;
                DROP TABLE memory_events;
                DROP TABLE graph_claims;
                DROP TABLE graph_entities;
                PRAGMA user_version = 2;
            """)
        migrated = configure_database(str(path))
        self.assertEqual(message.id, migrated.conversations.messages(conversation.id)[0].id)
        with sqlite3.connect(path) as connection:
            self.assertEqual(5, connection.execute("PRAGMA user_version").fetchone()[0])


if __name__ == "__main__":
    unittest.main()
