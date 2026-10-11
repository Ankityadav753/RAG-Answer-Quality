-- =============================================================================
-- DAI-009: RAG Answer Quality Benchmark
-- SQLite Database Schema Definition
-- =============================================================================

-- Ensure foreign keys are enabled (also enforced at connection-level in database.py)
PRAGMA foreign_keys = ON;

-- -----------------------------------------------------------------------------
-- 1. Documents: Raw reference documents loaded into the system
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source TEXT,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- -----------------------------------------------------------------------------
-- 2. Document Chunks: Segmented text chunks used for retrieval
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS document_chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    token_count INTEGER,
    embedding_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
);

-- -----------------------------------------------------------------------------
-- 3. Benchmark Runs: Metadata for each benchmark experiment execution
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS benchmark_runs (
    id TEXT PRIMARY KEY,
    run_name TEXT NOT NULL,
    config_name TEXT NOT NULL,
    top_k INTEGER NOT NULL DEFAULT 3,
    total_queries INTEGER NOT NULL DEFAULT 0,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    notes TEXT
);

-- -----------------------------------------------------------------------------
-- 4. Run Evaluations: Detailed per-query retrieval & answer quality metrics
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS run_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    query_id TEXT NOT NULL,
    query_text TEXT NOT NULL,
    retrieved_chunk_ids_json TEXT NOT NULL,
    ground_truth_chunk_ids_json TEXT NOT NULL,
    precision_at_k REAL,
    recall_at_k REAL,
    reciprocal_rank REAL,
    groundedness_score REAL,
    answer_text TEXT,
    is_abstention INTEGER DEFAULT 0,
    latency_ms REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES benchmark_runs (id) ON DELETE CASCADE
);

-- -----------------------------------------------------------------------------
-- Indexes: Speed up retrieval lookups and analytical aggregation queries
-- -----------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON document_chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_run_id ON run_evaluations (run_id);
CREATE INDEX IF NOT EXISTS idx_evaluations_query_id ON run_evaluations (query_id);

-- -----------------------------------------------------------------------------
-- 5. Full-Text Search (FTS5): Lexical BM25 index over document chunks
-- -----------------------------------------------------------------------------
CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5(
    chunk_id UNINDEXED,
    document_id UNINDEXED,
    content
);

-- Triggers to keep fts_chunks synchronized with document_chunks
CREATE TRIGGER IF NOT EXISTS trg_chunks_ai AFTER INSERT ON document_chunks
BEGIN
    INSERT INTO fts_chunks (chunk_id, document_id, content)
    VALUES (new.id, new.document_id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS trg_chunks_ad AFTER DELETE ON document_chunks
BEGIN
    DELETE FROM fts_chunks WHERE chunk_id = old.id;
END;
