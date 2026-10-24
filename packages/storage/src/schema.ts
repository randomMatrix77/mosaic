/**
 * SQLite schema reserved for the Android Room/SQLite implementation in Milestone C.
 * Vectors are stored as binary Float32 blobs with explicit dimensions/model metadata.
 */
export const initialSchemaSql = `
CREATE TABLE conversations (id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL);
CREATE TABLE messages (id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL REFERENCES conversations(id), role TEXT NOT NULL, content TEXT NOT NULL, created_at INTEGER NOT NULL);
CREATE TABLE attachments (id TEXT PRIMARY KEY, message_id TEXT REFERENCES messages(id), local_uri TEXT NOT NULL, mime_type TEXT NOT NULL, display_name TEXT NOT NULL, extracted_text TEXT, created_at INTEGER NOT NULL);
CREATE TABLE indexed_documents (id TEXT PRIMARY KEY, attachment_id TEXT REFERENCES attachments(id), content_hash TEXT NOT NULL, indexed_at INTEGER, status TEXT NOT NULL);
CREATE TABLE image_references (id TEXT PRIMARY KEY, attachment_id TEXT NOT NULL REFERENCES attachments(id), width INTEGER, height INTEGER, created_at INTEGER NOT NULL);
CREATE TABLE embeddings (id TEXT PRIMARY KEY, source_type TEXT NOT NULL, source_id TEXT NOT NULL, model_id TEXT NOT NULL, dimensions INTEGER NOT NULL, vector BLOB NOT NULL, created_at INTEGER NOT NULL);
CREATE INDEX embeddings_source_idx ON embeddings(source_type, source_id);
CREATE TABLE memory_records (id TEXT PRIMARY KEY, content TEXT NOT NULL, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL, extraction_version TEXT);
CREATE TABLE memory_provenance (memory_id TEXT NOT NULL REFERENCES memory_records(id), source_type TEXT NOT NULL, source_id TEXT NOT NULL, observed_at INTEGER NOT NULL, PRIMARY KEY(memory_id, source_type, source_id));
`;

