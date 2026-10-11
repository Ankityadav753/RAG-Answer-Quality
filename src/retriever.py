"""
Retrieval engine for DAI-009: RAG Answer Quality Benchmark.

Implements:
1. RetrievalResult common return structure.
2. BaseRetriever abstract base class.
3. LexicalRetriever: Configuration A (SQLite FTS5 BM25 with deterministic fallback).
4. DenseRetriever: Configuration B (Embedding-based cosine similarity with SQLite caching).
"""

from __future__ import annotations

import json
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

# Ensure project root is available in sys.path
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from db.database import DEFAULT_DB_PATH, get_db_connection
from src.embeddings import (
    BaseEmbeddingModel,
    DeterministicTestDoubleEmbeddingModel,
    get_embedding_model,
)


@dataclass(frozen=True)
class RetrievalResult:
    """Standardized retrieval result returned across all retrieval configurations."""
    chunk_id: str
    document_id: str
    content: str
    score: float
    rank: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert result object to dictionary."""
        return asdict(self)


class BaseRetriever(ABC):
    """Common interface for all RAG retrieval configurations."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        """
        Retrieve top-k relevant chunks for a search query.
        
        Args:
            query: The natural language or keyword query string.
            top_k: Number of ranked chunks to return (must be > 0).
            
        Returns:
            List[RetrievalResult]: Ranked results ordered by relevance score descending.
        """
        pass

    def _validate_top_k(self, top_k: int) -> None:
        """Validate top_k parameter constraint."""
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError(f"top_k must be an integer greater than 0, got {top_k}")


# =============================================================================
# Configuration A: Lexical Retrieval (SQLite FTS5 BM25 + Fallback)
# =============================================================================

class LexicalRetriever(BaseRetriever):
    """
    Configuration A: Lexical search utilizing SQLite FTS5 BM25 ranking.
    Includes a deterministic token-frequency fallback if FTS5 is disabled or unavailable.
    """

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        use_fts: bool = True,
    ) -> None:
        self.db_path = Path(db_path).resolve() if db_path else DEFAULT_DB_PATH
        self.use_fts = use_fts
        self._ensure_fts_synced()

    def _ensure_fts_synced(self) -> None:
        """Verify that fts_chunks contains all chunks from document_chunks."""
        if not self.use_fts:
            return

        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            try:
                # Check if fts_chunks exists
                cursor.execute(
                    "SELECT COUNT(*) AS cnt FROM sqlite_master WHERE type='table' AND name='fts_chunks';"
                )
                if cursor.fetchone()["cnt"] == 0:
                    cursor.execute(
                        "CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks USING fts5("
                        "chunk_id UNINDEXED, document_id UNINDEXED, content);"
                    )

                cursor.execute("SELECT COUNT(*) AS fts_cnt FROM fts_chunks;")
                fts_cnt = cursor.fetchone()["fts_cnt"]

                cursor.execute("SELECT COUNT(*) AS doc_cnt FROM document_chunks;")
                doc_cnt = cursor.fetchone()["doc_cnt"]

                if fts_cnt < doc_cnt:
                    cursor.execute("DELETE FROM fts_chunks;")
                    cursor.execute(
                        "INSERT INTO fts_chunks (chunk_id, document_id, content) "
                        "SELECT id, document_id, content FROM document_chunks;"
                    )
            except Exception:
                # If FTS5 is unsupported by the SQLite engine, fallback safely
                self.use_fts = False

    def _sanitize_fts_query(self, query: str) -> str:
        """Extract clean alphanumeric search terms for FTS5 syntax."""
        tokens = re.findall(r"\b[A-Za-z0-9_-]+\b", query)
        cleaned: List[str] = []
        for t in tokens:
            parts = t.split("-")
            cleaned.extend([p for p in parts if len(p) >= 2])

        if not cleaned:
            return ""

        # Remove common english noise words to focus BM25 on content terms
        stopwords = {
            "what", "is", "the", "for", "and", "in", "to", "of", "a", "an",
            "are", "do", "how", "on", "can", "i", "my", "if", "or", "which"
        }
        filtered = [c for c in cleaned if c.lower() not in stopwords]
        target_tokens = filtered if filtered else cleaned

        # Escape tokens and join with OR
        return " OR ".join(f'"{token}"' for token in target_tokens)

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        self._validate_top_k(top_k)

        if not query or not query.strip():
            return []

        if self.use_fts:
            try:
                return self._retrieve_fts5(query, top_k)
            except Exception:
                # Fall back gracefully if FTS query parsing encounters an unexpected error
                return self._retrieve_python_fallback(query, top_k)
        else:
            return self._retrieve_python_fallback(query, top_k)

    def _retrieve_fts5(self, query: str, top_k: int) -> List[RetrievalResult]:
        fts_match_query = self._sanitize_fts_query(query)
        if not fts_match_query:
            return self._retrieve_python_fallback(query, top_k)

        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT chunk_id, document_id, content, bm25(fts_chunks) AS raw_score
                FROM fts_chunks
                WHERE fts_chunks MATCH ?
                ORDER BY bm25(fts_chunks) ASC
                LIMIT ?;
                """,
                (fts_match_query, top_k),
            )
            rows = cursor.fetchall()

        if not rows:
            # Fallback to lexical token matching if FTS found zero matches
            return self._retrieve_python_fallback(query, top_k)

        results: List[RetrievalResult] = []
        for idx, row in enumerate(rows):
            # In SQLite FTS5 bm25, lower/more negative is better match.
            # We negate it so higher positive score = better match.
            score = round(-1.0 * float(row["raw_score"]), 4)
            results.append(
                RetrievalResult(
                    chunk_id=str(row["chunk_id"]),
                    document_id=str(row["document_id"]),
                    content=str(row["content"]),
                    score=score,
                    rank=idx + 1,
                )
            )
        return results

    def _retrieve_python_fallback(self, query: str, top_k: int) -> List[RetrievalResult]:
        """Deterministic pure-Python token overlap / TF-IDF scoring fallback."""
        query_terms = set(re.findall(r"\b[A-Za-z0-9]+\b", query.lower()))
        if not query_terms:
            return []

        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, document_id, content FROM document_chunks;")
            rows = cursor.fetchall()

        scored: List[tuple[float, str, str, str]] = []
        for row in rows:
            chunk_text = str(row["content"]).lower()
            chunk_tokens = re.findall(r"\b[A-Za-z0-9]+\b", chunk_text)
            if not chunk_tokens:
                continue

            matches = sum(1 for t in chunk_tokens if t in query_terms)
            if matches > 0:
                # Frequency score normalized by text length with match boost
                score = matches / (len(chunk_tokens) ** 0.5)
                scored.append((score, str(row["id"]), str(row["document_id"]), str(row["content"])))

        # Sort descending by score
        scored.sort(key=lambda x: x[0], reverse=True)
        top_rows = scored[:top_k]

        results: List[RetrievalResult] = []
        for rank, (score, chk_id, doc_id, content) in enumerate(top_rows, start=1):
            results.append(
                RetrievalResult(
                    chunk_id=chk_id,
                    document_id=doc_id,
                    content=content,
                    score=round(float(score), 4),
                    rank=rank,
                )
            )
        return results


# =============================================================================
# Configuration B: Dense Vector Retrieval (Sentence Transformers + Cosine Sim)
# =============================================================================

class DenseRetriever(BaseRetriever):
    """
    Configuration B: Dense semantic retrieval using vector embeddings and cosine similarity.
    Caches chunk embeddings directly into SQLite to eliminate redundant encoding.
    """

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        embedding_model: Optional[BaseEmbeddingModel] = None,
    ) -> None:
        self.db_path = Path(db_path).resolve() if db_path else DEFAULT_DB_PATH
        # Default to offline deterministic test double if no model passed
        self.embedding_model = embedding_model or DeterministicTestDoubleEmbeddingModel()

        # In-memory index cache
        self._chunk_ids: List[str] = []
        self._document_ids: List[str] = []
        self._contents: List[str] = []
        self._embeddings_matrix: Optional[np.ndarray] = None
        self._is_indexed = False

    def index_chunks(self, force_recompute: bool = False) -> int:
        """
        Load or compute embeddings for all document chunks in SQLite.
        Persists newly generated embeddings back to `document_chunks.embedding_json`.
        
        Returns:
            int: Number of chunks indexed.
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, document_id, content, embedding_json FROM document_chunks ORDER BY id;"
            )
            rows = cursor.fetchall()

        if not rows:
            self._chunk_ids = []
            self._document_ids = []
            self._contents = []
            self._embeddings_matrix = np.empty((0, self.embedding_model.dimension), dtype=np.float32)
            self._is_indexed = True
            return 0

        chunk_ids: List[str] = []
        doc_ids: List[str] = []
        contents: List[str] = []
        vectors: List[np.ndarray] = []
        missing_indices: List[int] = []
        missing_texts: List[str] = []

        for idx, row in enumerate(rows):
            cid = str(row["id"])
            did = str(row["document_id"])
            text = str(row["content"])
            emb_json = row["embedding_json"]

            chunk_ids.append(cid)
            doc_ids.append(did)
            contents.append(text)

            if not force_recompute and emb_json:
                try:
                    vec = np.array(json.loads(emb_json), dtype=np.float32)
                    vectors.append(vec)
                except Exception:
                    missing_indices.append(idx)
                    missing_texts.append(text)
                    vectors.append(np.zeros(self.embedding_model.dimension, dtype=np.float32))
            else:
                missing_indices.append(idx)
                missing_texts.append(text)
                vectors.append(np.zeros(self.embedding_model.dimension, dtype=np.float32))

        # Compute embeddings for missing chunks in a single batched call
        if missing_texts:
            new_vectors = self.embedding_model.encode(missing_texts)
            # Update SQLite with computed embeddings
            with get_db_connection(self.db_path) as conn:
                cursor = conn.cursor()
                for pos, (orig_idx, vec) in enumerate(zip(missing_indices, new_vectors)):
                    vectors[orig_idx] = vec
                    cid = chunk_ids[orig_idx]
                    cursor.execute(
                        "UPDATE document_chunks SET embedding_json = ? WHERE id = ?;",
                        (json.dumps(vec.tolist()), cid),
                    )

        self._chunk_ids = chunk_ids
        self._document_ids = doc_ids
        self._contents = contents

        matrix = np.array(vectors, dtype=np.float32)
        # Ensure L2 normalization of matrix
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.maximum(norms, 1e-12)
        self._embeddings_matrix = matrix
        self._is_indexed = True

        return len(chunk_ids)

    def retrieve(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        self._validate_top_k(top_k)

        if not query or not query.strip():
            return []

        if not self._is_indexed:
            self.index_chunks()

        if self._embeddings_matrix is None or len(self._chunk_ids) == 0:
            return []

        # Encode query into normalized vector
        query_vec = self.embedding_model.encode([query])[0]
        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec /= q_norm

        # Cosine similarity is the dot product of normalized vectors
        scores = np.dot(self._embeddings_matrix, query_vec)

        # Sort indices in descending order
        ranked_indices = np.argsort(-scores)[:top_k]

        results: List[RetrievalResult] = []
        for rank, idx in enumerate(ranked_indices, start=1):
            results.append(
                RetrievalResult(
                    chunk_id=self._chunk_ids[idx],
                    document_id=self._document_ids[idx],
                    content=self._contents[idx],
                    score=round(float(scores[idx]), 4),
                    rank=rank,
                )
            )

        return results
