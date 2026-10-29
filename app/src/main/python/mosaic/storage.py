"""Small, portable SQLite persistence layer for Mosaic.

Android only supplies the database location. All schema and data access logic
lives here so the same module can later be used by a desktop Python app.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import sqlite3
import struct
from typing import Any, Iterator, Mapping, Sequence
from uuid import uuid4


SCHEMA_VERSION = 1


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
                        role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant')),
                        content TEXT NOT NULL,
                        model_identifier TEXT,
                        status TEXT NOT NULL DEFAULT 'completed'
                            CHECK (status IN ('completed', 'failed', 'pending')),
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
                connection.execute("PRAGMA user_version = 1")


class ConversationRepository:
    def __init__(self, database: Database) -> None:
        self._database = database

    @staticmethod
    def _conversation(row: sqlite3.Row) -> Conversation:
        return Conversation(row["id"], row["title"], row["created_at"], row["updated_at"])

    @staticmethod
    def _message(row: sqlite3.Row) -> Message:
        return Message(row["id"], row["conversation_id"], row["role"], row["content"],
                       row["model_identifier"], row["status"], row["created_at"])

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
            cursor = connection.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            if cursor.rowcount != 1:
                raise ValueError("Conversation does not exist.")

    def add_message(self, conversation_id: str, role: str, content: str,
                    model_identifier: str | None = None, status: str = "completed") -> Message:
        if role not in {"system", "user", "assistant"}:
            raise ValueError("Unsupported message role.")
        if status not in {"completed", "failed", "pending"}:
            raise ValueError("Unsupported message status.")
        if not content.strip():
            raise ValueError("Messages must not be empty.")
        message = Message(str(uuid4()), conversation_id, role, content, model_identifier, status, _now())
        with self._database.transaction() as connection:
            connection.execute(
                "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?)",
                (message.id, message.conversation_id, message.role, message.content,
                 message.model_identifier, message.status, message.created_at),
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
            rows = connection.execute(f"SELECT * FROM messages WHERE {where} ORDER BY created_at, id", (conversation_id,)).fetchall()
        return [self._message(row) for row in rows]


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


class MosaicStore:
    def __init__(self, path: str) -> None:
        self.database = Database(path)
        self.conversations = ConversationRepository(self.database)
        self.sources = SourceRepository(self.database)
        self.embeddings = EmbeddingRepository(self.database)


_store: MosaicStore | None = None


def configure_database(path: str) -> MosaicStore:
    global _store
    _store = MosaicStore(path)
    return _store


def get_store() -> MosaicStore:
    if _store is None:
        raise RuntimeError("Mosaic storage has not been configured by Android.")
    return _store
