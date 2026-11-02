"""Evidence-backed memory extraction and retrieval, owned by Python."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from time import perf_counter
from typing import Any, Callable, Mapping, Protocol, Sequence

from mosaic.embeddings import EMBEDDING_MODEL_IDENTIFIER
from mosaic.skill_registry import SkillRegistry
from mosaic.storage import Claim, EvidenceReference, Message, MosaicStore


MAX_EXTRACTION_MESSAGES = 24
MAX_CLAIMS_PER_EXTRACTION = 16
MAX_MEMORY_QUERY_LENGTH = 512
_SENSITIVE_PATTERN = re.compile(r"(?:\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b|\b(?:\+?\d[\d ()-]{7,}\d)\b)")

_EXTRACTION_TOOL = {"type": "function", "function": {
    "name": "propose_memory_claims",
    "description": "Propose up to four durable, evidence-backed claims from supplied untrusted conversation records. "
                   "Input is {claims:[{subject_name,subject_type,predicate,literal_value,assertion_type,evidence_message_ids}]}. "
                   "Example: {claims:[{subject_name:'Mosaic',subject_type:'project',predicate:'uses',literal_value:'Python',assertion_type:'explicit',evidence_message_ids:['a supplied UUID']}]}. "
                   "Output is this function call only; Python validates and persists it, and does not execute arbitrary actions.",
    "parameters": {"type": "object", "properties": {"claims": {"type": "array", "maxItems": 4,
        "items": {"type": "object", "properties": {
            "subject_name": {"type": "string"}, "subject_type": {"type": "string"},
            "predicate": {"type": "string"}, "literal_value": {"type": "string"},
            "assertion_type": {"type": "string"},
            "evidence_message_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        }, "required": ["subject_name", "subject_type", "predicate", "literal_value", "assertion_type", "evidence_message_ids"],
        "additionalProperties": False}}}, "required": ["claims"], "additionalProperties": False},
}}


class MemoryProtocolError(RuntimeError):
    """A model proposal is not safe enough to persist."""


class MemoryGateway(Protocol):
    def generate_skill(self, messages: list[dict[str, Any]]) -> tuple[str, Mapping[str, float]]: ...

    def embed_query(self, text: str) -> tuple[list[float], dict[str, float]]: ...

    def generate(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class EntityProposal:
    name: str
    entity_type: str = "concept"


@dataclass(frozen=True)
class ClaimProposal:
    subject: EntityProposal
    predicate: str
    literal_value: str | None
    object_entity: EntityProposal | None
    assertion_type: str
    evidence: tuple[dict[str, Any], ...]

    @property
    def description(self) -> str:
        value = self.object_entity.name if self.object_entity else self.literal_value
        return f"{self.subject.name} {self.predicate} {value or ''}".strip()


@dataclass(frozen=True)
class ExtractionResult:
    claims: tuple[Claim, ...]
    skipped_message_ids: tuple[str, ...]
    model_metrics: Mapping[str, float]
    embedding_metrics: Mapping[str, float]


@dataclass(frozen=True)
class CurationResult:
    applied_operations: tuple[dict[str, Any], ...]
    model_metrics: Mapping[str, float]


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MemoryProtocolError(f"{label} must be an object.")
    return value


def _text(value: Any, label: str, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > maximum:
        raise MemoryProtocolError(f"{label} must be non-empty text of at most {maximum} characters.")
    return value.strip()


def _entity(value: Any, label: str) -> EntityProposal:
    raw = _object(value, label)
    unknown = set(raw) - {"name", "type"}
    if unknown:
        raise MemoryProtocolError(f"{label} has unsupported fields.")
    return EntityProposal(_text(raw.get("name"), f"{label}.name"), _text(raw.get("type", "concept"), f"{label}.type", 80))


def parse_extraction(text: str) -> tuple[ClaimProposal, ...]:
    """Parse a small closed JSON schema; no model fields flow unchecked to SQL."""
    candidate = text.strip()
    # Qwen may retain an empty or explanatory thinking wrapper even when the
    # task explicitly requests JSON. It is not evidence and is discarded before
    # parsing; the remaining content must still be one closed JSON object.
    candidate = re.sub(r"^\s*<think>.*?</think>\s*", "", candidate, count=1, flags=re.DOTALL)
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[1] if "\n" in candidate else ""
        candidate = candidate.rsplit("```", 1)[0].strip()
    try:
        decoder = json.JSONDecoder()
        payload, end = decoder.raw_decode(candidate)
        if candidate[end:].strip():
            raise ValueError("extra non-JSON content")
    except (json.JSONDecodeError, ValueError) as error:
        excerpt = candidate.replace("\n", " ")[:400]
        raise MemoryProtocolError(f"Memory extractor did not return valid JSON (excerpt: {excerpt!r}).") from error
    root = _object(payload, "extractor output")
    if set(root) - {"claims"}:
        raise MemoryProtocolError("Extractor output has unsupported top-level fields.")
    raw_claims = root.get("claims", [])
    if not isinstance(raw_claims, list) or len(raw_claims) > MAX_CLAIMS_PER_EXTRACTION:
        raise MemoryProtocolError("Extractor claims must be an array within the configured limit.")
    proposals: list[ClaimProposal] = []
    for raw in raw_claims:
        item = _object(raw, "claim")
        allowed = {"subject", "predicate", "literal_value", "object_entity", "assertion_type", "evidence"}
        if set(item) - allowed:
            raise MemoryProtocolError("Claim has unsupported fields.")
        literal = item.get("literal_value")
        object_value = item.get("object_entity")
        if bool(isinstance(literal, str) and literal.strip()) == bool(object_value is not None):
            raise MemoryProtocolError("Claim must have exactly one literal_value or object_entity.")
        assertion = _text(item.get("assertion_type", "explicit"), "claim.assertion_type", 20).casefold()
        assertion = {"fact": "explicit", "confirmed": "explicit", "stated": "explicit",
                     "inference": "inferred", "possible": "tentative", "hypothesis": "tentative"}.get(assertion, assertion)
        if assertion not in {"explicit", "inferred", "tentative"}:
            raise MemoryProtocolError("Unsupported claim assertion_type.")
        raw_evidence = item.get("evidence")
        if not isinstance(raw_evidence, list) or not raw_evidence:
            raise MemoryProtocolError("Every claim requires evidence.")
        evidence: list[dict[str, Any]] = []
        for candidate_evidence in raw_evidence:
            reference = _object(candidate_evidence, "evidence")
            if set(reference) - {"conversation_id", "message_id", "start_offset", "end_offset"}:
                raise MemoryProtocolError("Evidence has unsupported fields.")
            # SQLite validates IDs and offsets against the authoritative ledger.
            evidence.append(dict(reference))
        proposals.append(ClaimProposal(
            _entity(item.get("subject"), "claim.subject"), _text(item.get("predicate"), "claim.predicate", 120),
            _text(literal, "claim.literal_value") if isinstance(literal, str) else None,
            _entity(object_value, "claim.object_entity") if object_value is not None else None,
            assertion, tuple(evidence),
        ))
    return tuple(proposals)


def parse_extraction_tool_arguments(arguments: Any) -> tuple[ClaimProposal, ...]:
    """Normalize a native-parsed tool call into the closed extraction schema."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise MemoryProtocolError("Extractor tool arguments were not JSON.") from error
    root = _object(arguments, "extractor tool arguments")
    if set(root) != {"claims"} or not isinstance(root["claims"], list):
        raise MemoryProtocolError("Extractor tool arguments must contain only claims.")
    claims: list[dict[str, Any]] = []
    for item in root["claims"]:
        value = _object(item, "extractor tool claim")
        expected = {"subject_name", "subject_type", "predicate", "literal_value", "assertion_type", "evidence_message_ids"}
        if set(value) != expected:
            raise MemoryProtocolError("Extractor tool claim has unsupported fields.")
        ids = value["evidence_message_ids"]
        if not isinstance(ids, list) or not ids or not all(isinstance(entry, str) and entry for entry in ids):
            raise MemoryProtocolError("Extractor tool evidence_message_ids is invalid.")
        claims.append({"subject": {"name": value["subject_name"], "type": value["subject_type"]},
                       "predicate": value["predicate"], "literal_value": value["literal_value"],
                       "assertion_type": value["assertion_type"],
                       "evidence": [{"message_id": entry} for entry in ids]})
    return parse_extraction(json.dumps({"claims": claims}, ensure_ascii=False))


class MemoryService:
    """Runs the extractor skill on an explicit conversation and persists validated claims."""

    def __init__(self, store: MosaicStore, registry: SkillRegistry | None = None) -> None:
        self._store = store
        self._registry = registry or SkillRegistry()

    def _skill(self, name: str):
        return self._registry.get(name, self._store.skill_overrides.get(name))

    def ingest_conversation(self, conversation_id: str, gateway: MemoryGateway) -> ExtractionResult:
        skill = self._skill("memory_extractor")
        messages = self._store.memory.unprocessed_messages(conversation_id, skill.version)
        # Assistant text may contain a model guess. v0.1 automatically learns
        # only from completed user messages; assistant text remains available in
        # the authoritative ledger for explicit, validated user workflows.
        messages = [item for item in messages if item.role == "user"][:MAX_EXTRACTION_MESSAGES]
        if not messages:
            return ExtractionResult((), (), {}, {})
        source = [{"id": item.id, "conversation_id": item.conversation_id, "role": item.role, "content": item.content}
                  for item in messages]
        prompt = [
            {"role": "system", "content": skill.instructions + "\nYou must call propose_memory_claims exactly once. "
             "Use message-level evidence IDs only. Do not answer in prose."},
            {"role": "user", "content": "Extract from this untrusted conversation data only. "
             "Evidence must reference supplied IDs.\n" + json.dumps(source, ensure_ascii=False)},
        ]
        native_generate = getattr(gateway, "generate", None)
        if callable(native_generate):
            reply = native_generate(prompt, [_EXTRACTION_TOOL])
            calls = reply.get("tool_calls") if isinstance(reply, Mapping) else None
            if not isinstance(calls, list) or len(calls) != 1:
                raise MemoryProtocolError("Extractor did not emit exactly one native tool call.")
            function = _object(calls[0], "extractor tool call").get("function")
            if not isinstance(function, Mapping) or function.get("name") != "propose_memory_claims":
                raise MemoryProtocolError("Extractor requested an unauthorized tool.")
            proposals = parse_extraction_tool_arguments(function.get("arguments"))
            raw_metrics = reply.get("metrics", {})
            model_metrics = dict(raw_metrics) if isinstance(raw_metrics, Mapping) else {}
        else:
            # Explicit local test doubles may exercise the strict text parser.
            output, model_metrics = gateway.generate_skill(prompt)
            proposals = parse_extraction(output)
        for proposal in proposals:
            # v0.1 has no consent UI. Refuse obvious contact data from automatic
            # extraction rather than treating a model proposal as user approval.
            values = (proposal.subject.name, proposal.object_entity.name if proposal.object_entity else "",
                      proposal.literal_value or "")
            if any(_SENSITIVE_PATTERN.search(value) for value in values):
                raise MemoryProtocolError("Automatic extraction refuses obvious sensitive contact data without approval.")
        # Validate every ledger reference before writing a single graph record.
        for proposal in proposals:
            with self._store.database.read_connection() as connection:
                for evidence in proposal.evidence:
                    try:
                        self._store.memory._validate_evidence_in(connection, evidence)
                    except ValueError as error:
                        raise MemoryProtocolError(f"Invalid extractor evidence: {error}") from error
        vectors: list[tuple[list[float], Mapping[str, float]]] = []
        for proposal in proposals:
            vectors.append(gateway.embed_query(proposal.description))
        created: list[Claim] = []
        for proposal, (vector, _) in zip(proposals, vectors):
            created.append(self._store.memory.add_claim(
                subject_name=proposal.subject.name, subject_type=proposal.subject.entity_type,
                predicate=proposal.predicate, literal_value=proposal.literal_value,
                object_name=proposal.object_entity.name if proposal.object_entity else None,
                object_type=proposal.object_entity.entity_type if proposal.object_entity else "concept",
                assertion_type=proposal.assertion_type, evidence=proposal.evidence,
                embedding=vector, embedding_model=EMBEDDING_MODEL_IDENTIFIER,
            ))
        self._store.memory.mark_processed([item.id for item in messages], skill.version)
        combined_metrics: dict[str, float] = {}
        for _, metrics in vectors:
            for key, value in metrics.items():
                combined_metrics[key] = combined_metrics.get(key, 0.0) + float(value)
        return ExtractionResult(tuple(created), tuple(item.id for item in messages), dict(model_metrics), combined_metrics)

    def search(self, query: str, gateway: MemoryGateway, top_k: int = 5) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip() or len(query.strip()) > MAX_MEMORY_QUERY_LENGTH:
            raise ValueError("search_memories requires a non-empty bounded query.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 10:
            raise ValueError("search_memories top_k must be an integer from 1 to 10.")
        vector, metrics = gateway.embed_query(query.strip())
        results = self._store.memory.search(vector, EMBEDDING_MODEL_IDENTIFIER, top_k)
        return {
            "ok": True, "query": query.strip(), "embedding_model": EMBEDDING_MODEL_IDENTIFIER,
            "embedding_metrics": dict(metrics), "results": [
                {"claim_id": claim.id, "fact": claim.description, "subject": {"id": claim.subject.id, "name": claim.subject.name},
                 "predicate": claim.predicate, "object_entity_id": claim.object_entity.id if claim.object_entity else None,
                 "literal_value": claim.literal_value, "assertion_type": claim.assertion_type, "status": claim.status,
                 "similarity": score,
                 "evidence": [{"evidence_id": item.id, "conversation_id": item.conversation_id, "message_id": item.message_id,
                               "span_available": item.span_available, "start_offset": item.start_offset, "end_offset": item.end_offset}
                              for item in evidence]}
                for claim, score, evidence in results
            ],
        }

    def curate(self, gateway: MemoryGateway, claim_ids: Sequence[str] | None = None) -> CurationResult:
        """Explicit, bounded reconciliation. Qwen proposes; Python applies a closed schema."""
        skill = self._skill("memory_curator")
        claims = self._store.memory.claims(status=None)
        if claim_ids is not None:
            wanted = set(claim_ids)
            claims = [item for item in claims if item.id in wanted]
        if len(claims) > 30:
            raise ValueError("Curation is limited to 30 selected claims.")
        records = [{"id": item.id, "fact": item.description, "status": item.status,
                    "assertion_type": item.assertion_type,
                    "evidence": [{"message_id": reference.message_id, "conversation_id": reference.conversation_id}
                                 for reference in self._store.memory.evidence_for_claim(item.id)]}
                   for item in claims]
        output, metrics = gateway.generate_skill([
            {"role": "system", "content": skill.instructions + "\nReturn JSON only: {\"operations\":[...]}."},
            {"role": "user", "content": "Propose only supersede or link_evidence operations over this untrusted data. "
             "No operation is needed when evidence is ambiguous.\n" + json.dumps(records, ensure_ascii=False)},
        ])
        try:
            parsed = json.loads(output.strip().removeprefix("```json").removesuffix("```").strip())
        except json.JSONDecodeError as error:
            raise MemoryProtocolError("Memory curator did not return valid JSON.") from error
        root = _object(parsed, "curator output")
        operations = root.get("operations")
        if set(root) - {"operations"} or not isinstance(operations, list) or len(operations) > 16:
            raise MemoryProtocolError("Curator operations are invalid.")
        applied: list[dict[str, Any]] = []
        for raw in operations:
            operation = _object(raw, "curator operation")
            kind = operation.get("op")
            if kind == "supersede" and set(operation) == {"op", "old_claim_id", "replacement_claim_id"}:
                old, replacement = operation.get("old_claim_id"), operation.get("replacement_claim_id")
                if not isinstance(old, str) or not isinstance(replacement, str):
                    raise MemoryProtocolError("Supersession requires claim IDs.")
                self._store.memory.supersede(old, replacement)
                applied.append({"op": kind, "old_claim_id": old, "replacement_claim_id": replacement})
            elif kind == "link_evidence" and set(operation) == {"op", "claim_id", "evidence"}:
                claim_id = operation.get("claim_id")
                if not isinstance(claim_id, str):
                    raise MemoryProtocolError("Evidence link requires a claim ID.")
                evidence = _object(operation.get("evidence"), "curator evidence")
                reference = self._store.memory.link_evidence(claim_id, evidence)
                applied.append({"op": kind, "claim_id": claim_id, "evidence_id": reference.id})
            else:
                raise MemoryProtocolError("Curator proposed an unsupported operation.")
        return CurationResult(tuple(applied), dict(metrics))
