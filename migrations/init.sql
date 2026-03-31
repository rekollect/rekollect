-- Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Users (simple -- no auth provider dependency)
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE,
    name TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- API keys
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    key_hash TEXT NOT NULL UNIQUE,
    key_prefix TEXT NOT NULL,
    name TEXT,
    last_used_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT now(),
    revoked_at TIMESTAMPTZ
);
CREATE INDEX ON api_keys(key_hash);
CREATE INDEX ON api_keys(user_id);

-- Documents
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    user_title TEXT,
    user_text TEXT,
    file_type TEXT,
    collection TEXT DEFAULT 'default',
    processing_status TEXT DEFAULT 'pending',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ON documents(user_id);
CREATE INDEX ON documents(collection);

-- Chunks
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    chunk_index INTEGER DEFAULT 0,
    token_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ON chunks(document_id);

-- Embeddings (pgvector)
CREATE TABLE embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id UUID REFERENCES chunks(id) ON DELETE CASCADE,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    embedding vector(1536),
    created_at TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ON embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX ON embeddings(document_id);
CREATE INDEX ON embeddings(user_id);

-- Seed: create a default user + API key for quick start
INSERT INTO users (id, email, name) VALUES
    ('00000000-0000-0000-0000-000000000001', 'admin@localhost', 'Admin');

-- Default API key: rk_dev_rekollect (for local development)
-- bcrypt hash generated for 'rk_dev_rekollect'
INSERT INTO api_keys (user_id, key_hash, key_prefix, name) VALUES
    ('00000000-0000-0000-0000-000000000001',
     crypt('rk_dev_rekollect', gen_salt('bf')),
     'rk_dev_reko', 'Default Dev Key');
