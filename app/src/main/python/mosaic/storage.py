"""Small, portable SQLite persistence layer for Mosaic.

Android only supplies the database location. All schema and data access logic
lives here so the same module can later be used by a desktop Python app.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import hashlib
import math
from pathlib import Path
import sqlite3
import struct
from typing import Any, Iterator, Mapping, Sequence
from uuid import uuid4


SCHEMA_VERSION = 5


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def encode_float32(values: Sequence[float]) -> bytes:
    """Validate and encode a vector as little-endian IEEE-754 float32 bytes."""
    if not values:
        raise ValueError("Embedding vectors must not be empty.")
    converted = [float(value) for value in values]
    if not all(math.isfinite(value) for value in converted):
        raise ValueError("Embedding vectors must contain only finite values.")
    try:
        blob = struct.pack(f"<{len(converted)}f", *converted)
    except (OverflowError, struct.error) as error:
        raise ValueError("Embedding values cannot be represented as float32.") from error
    decoded = struct.unpack(f"<{len(converted)}f", blob)
    if not all(math.isfinite(value) for value in decoded):
        raise ValueError("Embedding values overflowed float32.")
    return blob


def decode_float32(blob: bytes, dimension: int) -> list[float]:
    if dimension <= 0 or len(blob) != dimension * 4:
        raise ValueError("Stored embedding blob size does not match its dimension.")
    values = list(struct.unpack(f"<{dimension}f", blob))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Stored embedding contains non-finite float32 values.")
    return values


@dataclass(frozen=True)
class Conversation:
    id: str
    title: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class Message:
    id: str
    conversation_id: str
    role: str
    content: str
    model_identifier: str | None
    status: str
    kind: str
    metadata: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class Source:
    id: str
    modality: str
    reference: str
    mime_type: str | None
    content_fingerprint: str
    metadata: dict[str, Any]
    created_at: str


@dataclass(frozen=True)
class Embedding:
    id: str
    source_id: str
    model_identifier: str
    content_fingerprint: str
    dimension: int
    normalized: bool
    vector: list[float]
    created_at: str


@dataclass(frozen=True)
class CustomCapability:
    """A user-authored instruction which is safely included in Qwen context."""

    id: str
    title: str
    description: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class Entity:
    id: str
    name: str
    normalized_name: str
    entity_type: str
    status: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class Claim:
    id: str
    subject: Entity
    predicate: str
    object_entity: Entity | None
    literal_value: str | None
    assertion_type: str
    status: str
    created_at: str
    updated_at: str

    @property
    def description(self) -> str:
        value = self.object_entity.name if self.object_entity else self.literal_value
        return f"{self.subject.name} {self.predicate} {value or ''}".strip()


@dataclass(frozen=True)
class EvidenceReference:
    id: str
    claim_id: str
    conversation_id: str
    message_id: str
    start_offset: int | None
    end_offset: int | None
    message_fingerprint: str
    span_available: bool
    created_at: str


@dataclass(frozen=True)
class ResolvedEvidence:
    reference: EvidenceReference
    quote: str | None
    message: Message
    context: tuple[Message, ...]


class Database:
    def __init__(self, path: str) -> None:
        if not path:
            raise ValueError("A SQLite database path is required.")
        self.path = str(Path(path))
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    @contextmanager
    def read_connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        # These connection/database settings must be applied outside a transaction.
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
        with self.transaction() as connection:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version > SCHEMA_VERSION:
                raise RuntimeError(f"Database schema {version} is newer than Mosaic supports.")
            if version < 1:
                connection.executescript(
                    """
                    CREATE TABLE conversations (
                        id TEXT PRIMARY KEY,
                        title TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE messages (
                        id TEXT PRIMARY KEY,
                        conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                        role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
                        content TEXT NOT NULL,
                        model_identifier TEXT,
                        status TEXT NOT NULL DEFAULT 'completed'
                            CHECK (status IN ('completed', 'failed', 'pending')),
                        kind TEXT NOT NULL DEFAULT 'chat'
                            CHECK (kind IN ('chat', 'tool_call', 'tool_result')),
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL
                    );
                    CREATE INDEX messages_conversation_chronological
                        ON messages(conversation_id, created_at, id);
                    CREATE TABLE sources (
                        id TEXT PRIMARY KEY,
                        modality TEXT NOT NULL,
                        reference TEXT NOT NULL,
                        mime_type TEXT,
                        content_fingerprint TEXT NOT NULL,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        UNIQUE(modality, reference, content_fingerprint)
                    );
                    CREATE TABLE embeddings (
                        id TEXT PRIMARY KEY,
                        source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                        model_identifier TEXT NOT NULL,
                        content_fingerprint TEXT NOT NULL,
                        dimension INTEGER NOT NULL CHECK (dimension > 0),
                        normalized INTEGER NOT NULL CHECK (normalized IN (0, 1)),
                        vector_f32 BLOB NOT NULL,
                        created_at TEXT NOT NULL,
                        UNIQUE(source_id, model_identifier, content_fingerprint)
                    );
                    CREATE INDEX embeddings_model_source ON embeddings(model_identifier, source_id);
                    """
                )
                connection.execute("PRAGMA user_version = 2")
                version = 2
            if version < 2:
                # SQLite cannot widen the existing role CHECK constraint in
                # place. Rebuild only this small table, preserving every prior
                # conversation/message and marking legacy records as chat.
                connection.executescript(
                    """
                    DROP INDEX IF EXISTS messages_conversation_chronological;
                    ALTER TABLE messages RENAME TO messages_legacy;
                    CREATE TABLE messages (
                        id TEXT PRIMARY KEY,
                        conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                        role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
                        content TEXT NOT NULL,
                        model_identifier TEXT,
                        status TEXT NOT NULL DEFAULT 'completed'
                            CHECK (status IN ('completed', 'failed', 'pending')),
                        kind TEXT NOT NULL DEFAULT 'chat'
                            CHECK (kind IN ('chat', 'tool_call', 'tool_result')),
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL
                    );
                    INSERT INTO messages
                        (id, conversation_id, role, content, model_identifier, status, kind, metadata_json, created_at)
                    SELECT id, conversation_id, role, content, model_identifier, status, 'chat', '{}', created_at
                    FROM messages_legacy;
                    DROP TABLE messages_legacy;
                    CREATE INDEX messages_conversation_chronological
                        ON messages(conversation_id, created_at, id);
                    """
                )
                connection.execute("PRAGMA user_version = 2")
                version = 2
            if version < 3:
                # The conversation ledger remains authoritative. These tables are a
                # derived graph and never rewrite existing conversations/messages.
                connection.executescript(
                    """
                    CREATE TABLE graph_entities (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        normalized_name TEXT NOT NULL,
                        entity_type TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'active'
                            CHECK (status IN ('active', 'deleted')),
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE(normalized_name, entity_type)
                    );
                    CREATE TABLE graph_claims (
                        id TEXT PRIMARY KEY,
                        subject_entity_id TEXT NOT NULL REFERENCES graph_entities(id) ON DELETE RESTRICT,
                        predicate TEXT NOT NULL,
                        object_entity_id TEXT REFERENCES graph_entities(id) ON DELETE RESTRICT,
                        literal_value TEXT,
                        assertion_type TEXT NOT NULL
                            CHECK (assertion_type IN ('explicit', 'inferred', 'tentative')),
                        status TEXT NOT NULL DEFAULT 'current'
                            CHECK (status IN ('current', 'superseded', 'deleted')),
                        dedupe_key TEXT NOT NULL UNIQUE,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        CHECK ((object_entity_id IS NOT NULL AND literal_value IS NULL)
                           OR (object_entity_id IS NULL AND literal_value IS NOT NULL))
                    );
                    CREATE INDEX graph_claims_subject_status
                        ON graph_claims(subject_entity_id, status, predicate);
                    CREATE TABLE claim_evidence (
                        id TEXT PRIMARY KEY,
                        claim_id TEXT NOT NULL REFERENCES graph_claims(id) ON DELETE CASCADE,
                        conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                        message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                        start_offset INTEGER,
                        end_offset INTEGER,
                        message_fingerprint TEXT NOT NULL,
                        span_available INTEGER NOT NULL CHECK (span_available IN (0, 1)),
                        created_at TEXT NOT NULL,
                        CHECK ((span_available = 0 AND start_offset IS NULL AND end_offset IS NULL)
                           OR (span_available = 1 AND start_offset >= 0 AND end_offset > start_offset))
                    );
                    CREATE INDEX claim_evidence_claim ON claim_evidence(claim_id, created_at);
                    CREATE INDEX claim_evidence_message ON claim_evidence(message_id, claim_id);
                    CREATE TABLE claim_embeddings (
                        claim_id TEXT NOT NULL REFERENCES graph_claims(id) ON DELETE CASCADE,
                        model_identifier TEXT NOT NULL,
                        dimension INTEGER NOT NULL CHECK (dimension > 0),
                        normalized INTEGER NOT NULL CHECK (normalized IN (0, 1)),
                        vector_f32 BLOB NOT NULL,
                        content_fingerprint TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY(claim_id, model_identifier)
                    );
                    CREATE TABLE memory_events (
                        id TEXT PRIMARY KEY,
                        claim_id TEXT,
                        event_type TEXT NOT NULL
                            CHECK (event_type IN ('created', 'evidence_linked', 'superseded', 'deleted', 'conversation_deleted')),
                        payload_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL
                    );
                    CREATE INDEX memory_events_claim ON memory_events(claim_id, created_at);
                    CREATE TABLE processed_memory_messages (
                        message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
                        skill_version TEXT NOT NULL,
                        message_fingerprint TEXT NOT NULL,
                        processed_at TEXT NOT NULL,
                        PRIMARY KEY(message_id, skill_version)
                    );
                    """
                )
                connection.execute("PRAGMA user_version = 3")
                version = 3
            if version < 4:
                # User-authored capabilities are local instruction records, not
                # executable tools. Keeping them separate preserves the fixed
                # Python-enforced tool allow-list.
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS custom_capabilities (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        normalized_title TEXT NOT NULL UNIQUE,
                        description TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS custom_capabilities_updated
                        ON custom_capabilities(updated_at DESC, id DESC);
                    """
                )
                connection.execute("PRAGMA user_version = 4")
                version = 4
            if version < 5:
                # Built-in skills remain packaged and named by Python. This
                # table stores only a user's local instruction override.
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS skill_overrides (
                        skill_name TEXT PRIMARY KEY,
                        instructions TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    """
                )
                connection.execute("PRAGMA user_version = 5")


class ConversationRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    @staticmethod
    def _conversation(row: sqlite3.Row) -> Conversation:
        return Conversation(row["id"], row["title"], row["created_at"], row["updated_at"])

    @staticmethod
    def _message(row: sqlite3.Row) -> Message:
        return Message(row["id"], row["conversation_id"], row["role"], row["content"],
                       row["model_identifier"], row["status"], row["kind"],
                       json.loads(row["metadata_json"]), row["created_at"])

    def create(self, title: str | None = None) -> Conversation:
        now = _now()
        conversation = Conversation(str(uuid4()), title.strip() if title and title.strip() else None, now, now)
        with self._database.transaction() as connection:
            connection.execute("INSERT INTO conversations VALUES (?, ?, ?, ?)",
                               (conversation.id, conversation.title, now, now))
        return conversation

    def list(self) -> list[Conversation]:
        with self._database.read_connection() as connection:
            rows = connection.execute("SELECT * FROM conversations ORDER BY updated_at DESC, id DESC").fetchall()
        return [self._conversation(row) for row in rows]

    def require(self, conversation_id: str) -> Conversation:
        with self._database.read_connection() as connection:
            row = connection.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        if not row:
            raise ValueError("Conversation does not exist.")
        return self._conversation(row)

    def rename(self, conversation_id: str, title: str | None) -> Conversation:
        self.require(conversation_id)
        with self._database.transaction() as connection:
            connection.execute("UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                               (title.strip() if title and title.strip() else None, _now(), conversation_id))
            row = connection.execute("SELECT * FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
        return self._conversation(row)

    def delete(self, conversation_id: str) -> None:
        with self._database.transaction() as connection:
            # Derived memory must never retain dangling references to erased
            # source material. Claims with other evidence remain valid.
            MemoryGraphRepository(self._database).handle_conversation_deletion(connection, conversation_id)
            cursor = connection.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            if cursor.rowcount != 1:
                raise ValueError("Conversation does not exist.")

    def add_message(self, conversation_id: str, role: str, content: str,
                    model_identifier: str | None = None, status: str = "completed", kind: str = "chat",
                    metadata: Mapping[str, Any] | None = None) -> Message:
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError("Unsupported message role.")
        if status not in {"completed", "failed", "pending"}:
            raise ValueError("Unsupported message status.")
        if kind not in {"chat", "tool_call", "tool_result"}:
            raise ValueError("Unsupported message kind.")
        if kind == "tool_call" and role != "assistant":
            raise ValueError("Tool-call events must have the assistant role.")
        if kind == "tool_result" and role != "tool":
            raise ValueError("Tool-result events must have the tool role.")
        if kind == "chat" and role == "tool":
            raise ValueError("Tool messages must use the tool_result kind.")
        if not content.strip():
            raise ValueError("Messages must not be empty.")
        metadata_dict = dict(metadata or {})
        metadata_json = json.dumps(metadata_dict, sort_keys=True, separators=(",", ":"))
        message = Message(str(uuid4()), conversation_id, role, content, model_identifier, status,
                          kind, metadata_dict, _now())
        with self._database.transaction() as connection:
            connection.execute(
                "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (message.id, message.conversation_id, message.role, message.content,
                 message.model_identifier, message.status, message.kind, metadata_json, message.created_at),
            )
            connection.execute("UPDATE conversations SET updated_at = ? WHERE id = ?",
                               (message.created_at, conversation_id))
            if connection.execute("SELECT changes()").fetchone()[0] != 1:
                raise ValueError("Conversation does not exist.")
        return message

    def messages(self, conversation_id: str, include_incomplete: bool = True) -> list[Message]:
        self.require(conversation_id)
        where = "conversation_id = ?" if include_incomplete else "conversation_id = ? AND status = 'completed'"
        with self._database.read_connection() as connection:
            # Timestamps have millisecond precision, so consecutive tool events can
            # share one timestamp. SQLite's insertion order is the required tiebreaker
            # for replaying an assistant tool-call before its corresponding tool result.
            rows = connection.execute(
                f"SELECT * FROM messages WHERE {where} ORDER BY created_at, rowid",
                (conversation_id,),
            ).fetchall()
        return [self._message(row) for row in rows]


class CustomCapabilityRepository:
    """Small SQLite ledger for user-authored, non-executable agent instructions."""

    def __init__(self, database: Database) -> None:
        self._database = database

    @staticmethod
    def _capability(row: sqlite3.Row) -> CustomCapability:
        return CustomCapability(row["id"], row["title"], row["description"], row["created_at"], row["updated_at"])

    @staticmethod
    def _validate(title: str, description: str) -> tuple[str, str, str]:
        normalized_title = title.strip()
        normalized_description = description.strip()
        if not 1 <= len(normalized_title) <= 100:
            raise ValueError("Capability name must be between 1 and 100 characters.")
        if not 1 <= len(normalized_description) <= 4_000:
            raise ValueError("Capability description must be between 1 and 4,000 characters.")
        return normalized_title, normalized_title.casefold(), normalized_description

    def list(self) -> list[CustomCapability]:
        with self._database.read_connection() as connection:
            rows = connection.execute(
                "SELECT * FROM custom_capabilities ORDER BY updated_at DESC, id DESC"
            ).fetchall()
        return [self._capability(row) for row in rows]

    def create(self, title: str, description: str) -> CustomCapability:
        title, normalized_title, description = self._validate(title, description)
        now = _now()
        capability = CustomCapability(str(uuid4()), title, description, now, now)
        try:
            with self._database.transaction() as connection:
                connection.execute(
                    "INSERT INTO custom_capabilities VALUES (?, ?, ?, ?, ?, ?)",
                    (capability.id, capability.title, normalized_title, capability.description,
                     capability.created_at, capability.updated_at),
                )
        except sqlite3.IntegrityError as error:
            raise ValueError("A capability with that name already exists.") from error
        return capability

    def update(self, capability_id: str, title: str, description: str) -> CustomCapability:
        title, normalized_title, description = self._validate(title, description)
        now = _now()
        try:
            with self._database.transaction() as connection:
                cursor = connection.execute(
                    """UPDATE custom_capabilities
                       SET title = ?, normalized_title = ?, description = ?, updated_at = ?
                       WHERE id = ?""",
                    (title, normalized_title, description, now, capability_id),
                )
                if cursor.rowcount != 1:
                    raise ValueError("Capability does not exist.")
                row = connection.execute(
                    "SELECT * FROM custom_capabilities WHERE id = ?", (capability_id,)
                ).fetchone()
        except sqlite3.IntegrityError as error:
            raise ValueError("A capability with that name already exists.") from error
        if row is None:
            raise RuntimeError("Capability update did not return a saved record.")
        return self._capability(row)

    def delete(self, capability_id: str) -> None:
        with self._database.transaction() as connection:
            cursor = connection.execute("DELETE FROM custom_capabilities WHERE id = ?", (capability_id,))
            if cursor.rowcount != 1:
                raise ValueError("Capability does not exist.")


class SkillOverrideRepository:
    """Stores local text overrides for the fixed built-in skill identities."""

    def __init__(self, database: Database) -> None:
        self._database = database

    @staticmethod
    def _validate(instructions: str) -> str:
        value = instructions.strip()
        if not 1 <= len(value) <= 6_000:
            raise ValueError("Skill instructions must be between 1 and 6,000 characters.")
        return value

    def get(self, skill_name: str) -> str | None:
        with self._database.read_connection() as connection:
            row = connection.execute(
                "SELECT instructions FROM skill_overrides WHERE skill_name = ?", (skill_name,)
            ).fetchone()
        return row["instructions"] if row else None

    def save(self, skill_name: str, instructions: str) -> str:
        value = self._validate(instructions)
        with self._database.transaction() as connection:
            connection.execute(
                """INSERT INTO skill_overrides(skill_name, instructions, updated_at) VALUES (?, ?, ?)
                   ON CONFLICT(skill_name) DO UPDATE SET instructions = excluded.instructions,
                                                         updated_at = excluded.updated_at""",
                (skill_name, value, _now()),
            )
        return value

    def delete(self, skill_name: str) -> None:
        with self._database.transaction() as connection:
            connection.execute("DELETE FROM skill_overrides WHERE skill_name = ?", (skill_name,))


class SourceRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    @staticmethod
    def _source(row: sqlite3.Row) -> Source:
        return Source(row["id"], row["modality"], row["reference"], row["mime_type"],
                      row["content_fingerprint"], json.loads(row["metadata_json"]), row["created_at"])

    def register(self, modality: str, reference: str, content_fingerprint: str,
                 mime_type: str | None = None, metadata: Mapping[str, Any] | None = None) -> Source:
        if not modality or not reference or not content_fingerprint:
            raise ValueError("Source modality, reference, and content fingerprint are required.")
        metadata_json = json.dumps(dict(metadata or {}), sort_keys=True, separators=(",", ":"))
        with self._database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM sources WHERE modality = ? AND reference = ? AND content_fingerprint = ?",
                (modality, reference, content_fingerprint),
            ).fetchone()
            if row:
                return self._source(row)
            source = Source(str(uuid4()), modality, reference, mime_type, content_fingerprint,
                            dict(metadata or {}), _now())
            connection.execute("INSERT INTO sources VALUES (?, ?, ?, ?, ?, ?, ?)",
                               (source.id, source.modality, source.reference, source.mime_type,
                                source.content_fingerprint, metadata_json, source.created_at))
        return source

    def delete(self, source_id: str) -> None:
        with self._database.transaction() as connection:
            cursor = connection.execute("DELETE FROM sources WHERE id = ?", (source_id,))
            if cursor.rowcount != 1:
                raise ValueError("Source does not exist.")

    def list(self, modality: str | None = None) -> list[Source]:
        with self._database.read_connection() as connection:
            rows = connection.execute("SELECT * FROM sources WHERE modality = ? ORDER BY created_at, id", (modality,)).fetchall() if modality else connection.execute("SELECT * FROM sources ORDER BY created_at, id").fetchall()
        return [self._source(row) for row in rows]


class EmbeddingRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    @staticmethod
    def _embedding(row: sqlite3.Row) -> Embedding:
        return Embedding(row["id"], row["source_id"], row["model_identifier"],
                         row["content_fingerprint"], row["dimension"], bool(row["normalized"]),
                         decode_float32(row["vector_f32"], row["dimension"]), row["created_at"])

    def save(self, source: Source, model_identifier: str, vector: Sequence[float], normalized: bool) -> Embedding:
        if not model_identifier:
            raise ValueError("Embedding model identifier is required.")
        blob = encode_float32(vector)
        dimension = len(vector)
        with self._database.transaction() as connection:
            row = connection.execute(
                "SELECT * FROM embeddings WHERE source_id = ? AND model_identifier = ? AND content_fingerprint = ?",
                (source.id, model_identifier, source.content_fingerprint),
            ).fetchone()
            if row:
                existing = self._embedding(row)
                if existing.dimension != dimension:
                    raise ValueError("Existing embedding has a different dimension for this source/model/version.")
                return existing
            embedding = Embedding(str(uuid4()), source.id, model_identifier, source.content_fingerprint,
                                  dimension, normalized, list(vector), _now())
            connection.execute("INSERT INTO embeddings VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                               (embedding.id, embedding.source_id, embedding.model_identifier,
                                embedding.content_fingerprint, embedding.dimension, int(embedding.normalized),
                                blob, embedding.created_at))
        return embedding

    def for_model(self, model_identifier: str, modality: str | None = None) -> list[tuple[Source, Embedding]]:
        query = """SELECT e.*, s.id AS s_id, s.modality AS s_modality, s.reference AS s_reference,
                          s.mime_type AS s_mime_type, s.content_fingerprint AS s_fingerprint,
                          s.metadata_json AS s_metadata_json, s.created_at AS s_created_at
                   FROM embeddings e JOIN sources s ON s.id = e.source_id
                   WHERE e.model_identifier = ?"""
        params: tuple[Any, ...] = (model_identifier,)
        if modality:
            query += " AND s.modality = ?"
            params += (modality,)
        query += " ORDER BY s.created_at, s.id"
        with self._database.read_connection() as connection:
            rows = connection.execute(query, params).fetchall()
        pairs: list[tuple[Source, Embedding]] = []
        for row in rows:
            source = Source(row["s_id"], row["s_modality"], row["s_reference"], row["s_mime_type"],
                            row["s_fingerprint"], json.loads(row["s_metadata_json"]), row["s_created_at"])
            pairs.append((source, self._embedding(row)))
        return pairs


class MemoryGraphRepository:
    """Derived knowledge graph with evidence anchored in the immutable ledger."""

    def __init__(self, database: Database) -> None:
        self._database = database

    @staticmethod
    def message_fingerprint(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _entity(row: sqlite3.Row, prefix: str = "") -> Entity:
        return Entity(row[f"{prefix}id"], row[f"{prefix}name"], row[f"{prefix}normalized_name"],
                      row[f"{prefix}entity_type"], row[f"{prefix}status"],
                      row[f"{prefix}created_at"], row[f"{prefix}updated_at"])

    @staticmethod
    def _reference(row: sqlite3.Row) -> EvidenceReference:
        return EvidenceReference(row["id"], row["claim_id"], row["conversation_id"], row["message_id"],
                                 row["start_offset"], row["end_offset"], row["message_fingerprint"],
                                 bool(row["span_available"]), row["created_at"])

    @staticmethod
    def _normal_name(name: str) -> str:
        normalized = " ".join(name.split()).casefold()
        if not normalized:
            raise ValueError("Entity names must not be empty.")
        return normalized

    @staticmethod
    def _dedupe_key(subject_id: str, predicate: str, object_id: str | None,
                    literal_value: str | None, assertion_type: str) -> str:
        raw = "\x1f".join((subject_id, predicate.casefold().strip(), object_id or "",
                             (literal_value or "").casefold().strip(), assertion_type))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _entity_in(self, connection: sqlite3.Connection, name: str, entity_type: str) -> Entity:
        normalized = self._normal_name(name)
        cleaned_type = entity_type.strip().casefold()
        if not cleaned_type:
            raise ValueError("Entity type is required.")
        row = connection.execute("SELECT * FROM graph_entities WHERE normalized_name = ? AND entity_type = ?",
                                 (normalized, cleaned_type)).fetchone()
        if row:
            return self._entity(row)
        now = _now()
        entity = Entity(str(uuid4()), " ".join(name.split()), normalized, cleaned_type, "active", now, now)
        connection.execute("INSERT INTO graph_entities VALUES (?, ?, ?, ?, ?, ?, ?)",
                           (entity.id, entity.name, entity.normalized_name, entity.entity_type,
                            entity.status, entity.created_at, entity.updated_at))
        return entity

    def entity(self, name: str, entity_type: str = "concept") -> Entity:
        with self._database.transaction() as connection:
            return self._entity_in(connection, name, entity_type)

    def _message_in(self, connection: sqlite3.Connection, message_id: str) -> Message:
        row = connection.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
        if not row:
            raise ValueError("Evidence message does not exist.")
        return ConversationRepository._message(row)

    def _validate_evidence_in(self, connection: sqlite3.Connection, evidence: Mapping[str, Any]) -> tuple[Message, int | None, int | None, bool]:
        message_id = evidence.get("message_id")
        if not isinstance(message_id, str) or not message_id:
            raise ValueError("Evidence requires a stored message_id.")
        message = self._message_in(connection, message_id)
        if evidence.get("conversation_id") not in (None, message.conversation_id):
            raise ValueError("Evidence conversation_id does not match its message.")
        start = evidence.get("start_offset")
        end = evidence.get("end_offset")
        if start is None and end is None:
            return message, None, None, False
        if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, int) or not isinstance(end, int):
            raise ValueError("Evidence span offsets must both be integers or both be omitted.")
        if not 0 <= start < end <= len(message.content):
            raise ValueError("Evidence span offsets are outside the stored message text.")
        return message, start, end, True

    def _event_in(self, connection: sqlite3.Connection, claim_id: str | None, event_type: str,
                  payload: Mapping[str, Any] | None = None) -> None:
        connection.execute("INSERT INTO memory_events VALUES (?, ?, ?, ?, ?)",
                           (str(uuid4()), claim_id, event_type,
                            json.dumps(dict(payload or {}), sort_keys=True, separators=(",", ":")), _now()))

    def add_claim(self, *, subject_name: str, subject_type: str, predicate: str,
                  literal_value: str | None = None, object_name: str | None = None,
                  object_type: str = "concept", assertion_type: str = "explicit",
                  evidence: Sequence[Mapping[str, Any]], embedding: Sequence[float] | None = None,
                  embedding_model: str | None = None) -> Claim:
        """Validate the whole proposal before one transaction changes the graph."""
        if assertion_type not in {"explicit", "inferred", "tentative"}:
            raise ValueError("Unsupported assertion type.")
        if not predicate or not predicate.strip():
            raise ValueError("Claim predicate is required.")
        if bool(literal_value and literal_value.strip()) == bool(object_name and object_name.strip()):
            raise ValueError("A claim needs exactly one object entity or literal value.")
        if not evidence:
            raise ValueError("Claims require at least one evidence reference.")
        if embedding is not None and not embedding_model:
            raise ValueError("An embedding model identifier is required with a claim embedding.")
        # Encode before entering the transaction so a bad float cannot partially write a claim.
        embedding_blob = encode_float32(embedding) if embedding is not None else None
        with self._database.transaction() as connection:
            validated = [self._validate_evidence_in(connection, item) for item in evidence]
            subject = self._entity_in(connection, subject_name, subject_type)
            object_entity = self._entity_in(connection, object_name, object_type) if object_name and object_name.strip() else None
            literal = literal_value.strip() if literal_value and literal_value.strip() else None
            key = self._dedupe_key(subject.id, predicate, object_entity.id if object_entity else None, literal, assertion_type)
            row = connection.execute("SELECT * FROM graph_claims WHERE dedupe_key = ?", (key,)).fetchone()
            if row:
                claim_id = row["id"]
                # A deliberately deleted memory is a tombstone. Re-ingestion never
                # resurrects it silently, even when old conversation text remains.
                if row["status"] == "deleted":
                    raise ValueError("This memory was deleted and cannot be recreated automatically.")
                created = False
            else:
                claim_id, now = str(uuid4()), _now()
                connection.execute("""INSERT INTO graph_claims
                    VALUES (?, ?, ?, ?, ?, ?, 'current', ?, ?, ?)""",
                    (claim_id, subject.id, predicate.strip(), object_entity.id if object_entity else None,
                     literal, assertion_type, key, now, now))
                self._event_in(connection, claim_id, "created", {"assertion_type": assertion_type})
                created = True
            for message, start, end, span_available in validated:
                fingerprint = self.message_fingerprint(message.content)
                exists = connection.execute("""SELECT 1 FROM claim_evidence
                    WHERE claim_id = ? AND message_id = ? AND start_offset IS ? AND end_offset IS ?""",
                    (claim_id, message.id, start, end)).fetchone()
                if not exists:
                    connection.execute("INSERT INTO claim_evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                       (str(uuid4()), claim_id, message.conversation_id, message.id, start, end,
                                        fingerprint, int(span_available), _now()))
                    self._event_in(connection, claim_id, "evidence_linked", {"message_id": message.id})
            if embedding_blob is not None:
                description_fingerprint = self.message_fingerprint(
                    f"{subject.name}\n{predicate.strip()}\n{object_entity.name if object_entity else literal}\n{assertion_type}")
                connection.execute("""INSERT OR REPLACE INTO claim_embeddings
                    (claim_id, model_identifier, dimension, normalized, vector_f32, content_fingerprint, created_at)
                    VALUES (?, ?, ?, 1, ?, ?, ?)""",
                    (claim_id, embedding_model, len(embedding), embedding_blob, description_fingerprint, _now()))
            return self._claim_in(connection, claim_id)

    def _claim_in(self, connection: sqlite3.Connection, claim_id: str) -> Claim:
        row = connection.execute("""SELECT c.*, s.id AS s_id, s.name AS s_name, s.normalized_name AS s_normalized_name,
                    s.entity_type AS s_entity_type, s.status AS s_status, s.created_at AS s_created_at, s.updated_at AS s_updated_at,
                    o.id AS o_id, o.name AS o_name, o.normalized_name AS o_normalized_name,
                    o.entity_type AS o_entity_type, o.status AS o_status, o.created_at AS o_created_at, o.updated_at AS o_updated_at
                FROM graph_claims c JOIN graph_entities s ON s.id = c.subject_entity_id
                LEFT JOIN graph_entities o ON o.id = c.object_entity_id WHERE c.id = ?""", (claim_id,)).fetchone()
        if not row:
            raise ValueError("Claim does not exist.")
        subject = self._entity(row, "s_")
        obj = self._entity(row, "o_") if row["o_id"] else None
        return Claim(row["id"], subject, row["predicate"], obj, row["literal_value"], row["assertion_type"],
                     row["status"], row["created_at"], row["updated_at"])

    def evidence_for_claim(self, claim_id: str) -> list[EvidenceReference]:
        with self._database.read_connection() as connection:
            self._claim_in(connection, claim_id)
            rows = connection.execute("SELECT * FROM claim_evidence WHERE claim_id = ? ORDER BY created_at, id", (claim_id,)).fetchall()
        return [self._reference(row) for row in rows]

    def link_evidence(self, claim_id: str, evidence: Mapping[str, Any]) -> EvidenceReference:
        with self._database.transaction() as connection:
            claim = self._claim_in(connection, claim_id)
            if claim.status == "deleted":
                raise ValueError("Cannot link evidence to a deleted memory.")
            message, start, end, span_available = self._validate_evidence_in(connection, evidence)
            row = connection.execute("""SELECT * FROM claim_evidence WHERE claim_id = ? AND message_id = ?
                AND start_offset IS ? AND end_offset IS ?""", (claim_id, message.id, start, end)).fetchone()
            if row:
                return self._reference(row)
            reference = EvidenceReference(str(uuid4()), claim_id, message.conversation_id, message.id, start, end,
                                         self.message_fingerprint(message.content), span_available, _now())
            connection.execute("INSERT INTO claim_evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                               (reference.id, reference.claim_id, reference.conversation_id, reference.message_id,
                                reference.start_offset, reference.end_offset, reference.message_fingerprint,
                                int(reference.span_available), reference.created_at))
            self._event_in(connection, claim_id, "evidence_linked", {"message_id": message.id})
            return reference

    def resolve_evidence(self, evidence_id: str, context_messages: int = 1) -> ResolvedEvidence:
        if context_messages < 0 or context_messages > 10:
            raise ValueError("context_messages must be between 0 and 10.")
        with self._database.read_connection() as connection:
            row = connection.execute("SELECT * FROM claim_evidence WHERE id = ?", (evidence_id,)).fetchone()
            if not row:
                raise ValueError("Evidence reference does not exist.")
            reference = self._reference(row)
            message = self._message_in(connection, reference.message_id)
            if self.message_fingerprint(message.content) != reference.message_fingerprint:
                raise RuntimeError("Evidence source text changed; its saved offsets are stale.")
            all_rows = connection.execute("SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at, rowid",
                                          (message.conversation_id,)).fetchall()
            messages = [ConversationRepository._message(item) for item in all_rows]
            index = next(item for item, value in enumerate(messages) if value.id == message.id)
            context = tuple(messages[max(0, index - context_messages):index + context_messages + 1])
        quote = message.content[reference.start_offset:reference.end_offset] if reference.span_available else None
        return ResolvedEvidence(reference, quote, message, context)

    def claims(self, *, status: str | None = "current") -> list[Claim]:
        query = "SELECT id FROM graph_claims" + (" WHERE status = ?" if status else "") + " ORDER BY created_at, id"
        with self._database.read_connection() as connection:
            rows = connection.execute(query, (status,) if status else ()).fetchall()
            return [self._claim_in(connection, row["id"]) for row in rows]

    def search(self, vector: Sequence[float], model_identifier: str, top_k: int) -> list[tuple[Claim, float, list[EvidenceReference]]]:
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        if not vector:
            raise ValueError("Memory query vector must not be empty.")
        ranked: list[tuple[Claim, float, list[EvidenceReference]]] = []
        with self._database.read_connection() as connection:
            rows = connection.execute("""SELECT ce.claim_id, ce.dimension, ce.normalized, ce.vector_f32
                FROM claim_embeddings ce JOIN graph_claims c ON c.id = ce.claim_id
                WHERE ce.model_identifier = ? AND c.status = 'current'""", (model_identifier,)).fetchall()
            for row in rows:
                if not row["normalized"] or row["dimension"] != len(vector):
                    continue
                claim = self._claim_in(connection, row["claim_id"])
                evidence_rows = connection.execute("SELECT * FROM claim_evidence WHERE claim_id = ?", (claim.id,)).fetchall()
                score = sum(a * b for a, b in zip(vector, decode_float32(row["vector_f32"], row["dimension"])))
                ranked.append((claim, score, [self._reference(item) for item in evidence_rows]))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:top_k]

    def mark_processed(self, message_ids: Sequence[str], skill_version: str) -> None:
        if not skill_version:
            raise ValueError("Skill version is required.")
        with self._database.transaction() as connection:
            for message_id in message_ids:
                message = self._message_in(connection, message_id)
                connection.execute("INSERT OR IGNORE INTO processed_memory_messages VALUES (?, ?, ?, ?)",
                                   (message.id, skill_version, self.message_fingerprint(message.content), _now()))

    def unprocessed_messages(self, conversation_id: str, skill_version: str) -> list[Message]:
        with self._database.read_connection() as connection:
            rows = connection.execute("""SELECT m.* FROM messages m LEFT JOIN processed_memory_messages p
                ON p.message_id = m.id AND p.skill_version = ?
                WHERE m.conversation_id = ? AND m.status = 'completed' AND m.kind = 'chat' AND p.message_id IS NULL
                ORDER BY m.created_at, m.rowid""", (skill_version, conversation_id)).fetchall()
        return [ConversationRepository._message(row) for row in rows]

    def supersede(self, old_claim_id: str, replacement_claim_id: str) -> None:
        if old_claim_id == replacement_claim_id:
            raise ValueError("A claim cannot supersede itself.")
        with self._database.transaction() as connection:
            self._claim_in(connection, old_claim_id)
            self._claim_in(connection, replacement_claim_id)
            connection.execute("UPDATE graph_claims SET status = 'superseded', updated_at = ? WHERE id = ?", (_now(), old_claim_id))
            self._event_in(connection, old_claim_id, "superseded", {"replacement_claim_id": replacement_claim_id})

    def delete_claim(self, claim_id: str, reason: str = "user_deleted") -> None:
        with self._database.transaction() as connection:
            self._claim_in(connection, claim_id)
            connection.execute("DELETE FROM claim_evidence WHERE claim_id = ?", (claim_id,))
            connection.execute("DELETE FROM claim_embeddings WHERE claim_id = ?", (claim_id,))
            connection.execute("UPDATE graph_claims SET status = 'deleted', updated_at = ? WHERE id = ?", (_now(), claim_id))
            self._event_in(connection, claim_id, "deleted", {"reason": reason})

    def handle_conversation_deletion(self, connection: sqlite3.Connection, conversation_id: str) -> None:
        """Remove derived references before ledger cascade and tombstone unsupported claims."""
        rows = connection.execute("SELECT DISTINCT claim_id FROM claim_evidence WHERE conversation_id = ?", (conversation_id,)).fetchall()
        claim_ids = [row["claim_id"] for row in rows]
        connection.execute("DELETE FROM claim_evidence WHERE conversation_id = ?", (conversation_id,))
        for claim_id in claim_ids:
            remaining = connection.execute("SELECT COUNT(*) FROM claim_evidence WHERE claim_id = ?", (claim_id,)).fetchone()[0]
            if remaining == 0:
                connection.execute("DELETE FROM claim_embeddings WHERE claim_id = ?", (claim_id,))
                connection.execute("UPDATE graph_claims SET status = 'deleted', updated_at = ? WHERE id = ?", (_now(), claim_id))
                self._event_in(connection, claim_id, "conversation_deleted", {"conversation_id": conversation_id})


class MosaicStore:
    def __init__(self, path: str) -> None:
        self.database = Database(path)
        self.conversations = ConversationRepository(self.database)
        self.capabilities = CustomCapabilityRepository(self.database)
        self.skill_overrides = SkillOverrideRepository(self.database)
        self.sources = SourceRepository(self.database)
        self.embeddings = EmbeddingRepository(self.database)
        self.memory = MemoryGraphRepository(self.database)


_store: MosaicStore | None = None


def configure_database(path: str) -> MosaicStore:
    global _store
    _store = MosaicStore(path)
    return _store


def get_store() -> MosaicStore:
    if _store is None:
        raise RuntimeError("Mosaic storage has not been configured by Android.")
    return _store
