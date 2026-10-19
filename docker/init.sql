-- Runs once, on the first start of an empty data directory, via docker/init.sh.
-- See that file for why this is rendered through psql instead of being mounted
-- straight into docker-entrypoint-initdb.d.
--
-- psql variables supplied by init.sh:
--   :hnsw_m  :hnsw_ef_construction

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------- documents
-- One row per policy document, exactly as it appears in data/kb_documents.json.
-- Deliberately carries no vector: the chunking strategy is an experimental
-- variable, so embeddings live on `chunks`, where several chunkings of the same
-- document can coexist in one table and be compared side by side.
CREATE TABLE IF NOT EXISTS documents (
    doc_id          TEXT PRIMARY KEY,
    title           TEXT        NOT NULL,
    category        TEXT        NOT NULL,
    card_tier       TEXT        NOT NULL,
    effective_year  SMALLINT    NOT NULL,
    policy_code     TEXT,
    content         TEXT        NOT NULL
);

-- -------------------------------------------------------------------- chunks
-- `title` is denormalised from documents because a GENERATED column cannot
-- reference another table, and the weighted tsvector needs the title at 'A'.
-- At 150 documents the redundancy is free.
--
-- `embedding` is NOT NULL on purpose. pgvector silently omits rows with a NULL
-- vector from an HNSW index, so such a row would be visible to a sequential
-- scan but invisible to ANN search -- a wrong Recall@K that no test would
-- catch. Making the state unrepresentable is cheaper than detecting it.
--
-- NOTE: VECTOR(1024) matches APEX_EMBEDDING_DIM. Switching to an embedding
-- model with different dimensions requires recreating this column.
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        TEXT        NOT NULL,
    doc_id          TEXT        NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    chunk_strategy  TEXT        NOT NULL,
    chunk_index     INTEGER     NOT NULL,
    title           TEXT        NOT NULL,
    content         TEXT        NOT NULL,
    embedded_text   TEXT        NOT NULL,
    embedding       VECTOR(1024) NOT NULL,
    category        TEXT        NOT NULL,
    card_tier       TEXT        NOT NULL,
    effective_year  SMALLINT    NOT NULL,
    policy_code     TEXT,

    -- Title and card_tier are weighted 'A'. That is the deliberate answer to
    -- this corpus's main failure mode: near-identical documents that differ
    -- only by which card the rule applies to, plus exactly one number.
    tsv TSVECTOR GENERATED ALWAYS AS (
        setweight(to_tsvector('english', coalesce(title, '')),     'A') ||
        setweight(to_tsvector('english', coalesce(card_tier, '')), 'A') ||
        setweight(to_tsvector('english', coalesce(content, '')),   'B')
    ) STORED,

    PRIMARY KEY (chunk_strategy, chunk_id)
);

-- --------------------------------------------------------------- btree / GIN
CREATE INDEX IF NOT EXISTS chunks_tsv_gin      ON chunks USING GIN (tsv);
CREATE INDEX IF NOT EXISTS chunks_strategy_idx ON chunks (chunk_strategy);
CREATE INDEX IF NOT EXISTS chunks_doc_idx      ON chunks (doc_id);

-- ------------------------------------------------------------------- HNSW
-- Bootstrap index. The parameters mirror DenseConfig's defaults in
-- src/schema/retrieval.py; override them for a fresh cluster with
-- APEX_HNSW_M / APEX_HNSW_EF_CONSTRUCTION (init scripts only run on an empty
-- data directory, so changing them means `make reset`).
--
-- Phase 4 does not rely on this index: indexer.py drops and recreates it while
-- sweeping m / ef_construction / ef_search, and owns the result from then on.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = :hnsw_m, ef_construction = :hnsw_ef_construction);
