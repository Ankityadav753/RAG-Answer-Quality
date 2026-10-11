"""
Unit and integration tests for retrieval configurations (DAI-009).

Tests:
1. LexicalRetriever: Exact keyword matching, query sanitization, FTS5 BM25 scoring.
2. LexicalRetriever fallback: Pure-Python token scoring when FTS5 is disabled.
3. DenseRetriever: Semantic similarity ranking using DeterministicTestDoubleEmbeddingModel.
4. Embedding caching: Chunks cached in SQLite `embedding_json` without redundant recalculation.
5. Common interface: Identical RetrievalResult structure and attributes across both configurations.
6. Edge cases: Top-k boundary limits, empty queries, empty database handling.
7. Error handling: Clear error reporting when sentence-transformers or models are unavailable.

Compatible with both `python -m unittest` and `pytest`.
"""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from db.database import get_db_connection, init_db
from src.embeddings import (
    DeterministicTestDoubleEmbeddingModel,
    EmbeddingModelUnavailableError,
    SentenceTransformerEmbeddingModel,
)
from src.ingest import DEFAULT_DOCS_PATH, ingest_documents
from src.retriever import DenseRetriever, LexicalRetriever, RetrievalResult


class TestRetrievalConfigurations(unittest.TestCase):
    def setUp(self) -> None:
        """Create a clean temporary database and ingest sample documents."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_db_path = Path(self.temp_dir.name) / "test_retrieval.db"
        init_db(self.test_db_path)
        ingest_documents(
            docs_path=DEFAULT_DOCS_PATH,
            db_path=self.test_db_path,
            chunk_size=80,
            chunk_overlap=20,
        )

    def tearDown(self) -> None:
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    # -------------------------------------------------------------------------
    # Configuration A: Lexical Retrieval Tests
    # -------------------------------------------------------------------------

    def test_lexical_exact_keyword_retrieval(self) -> None:
        """Verify that exact keyword search retrieves the relevant policy document."""
        retriever = LexicalRetriever(db_path=self.test_db_path)
        # Search exact SSID and protocol
        results = retriever.retrieve("Campus-Secure-5G WPA3-Enterprise", top_k=3)

        self.assertGreater(len(results), 0)
        top_match = results[0]
        self.assertEqual(top_match.document_id, "DOC-IT-001")
        self.assertIn("Campus-Secure-5G", top_match.content)
        self.assertEqual(top_match.rank, 1)

    def test_lexical_ranking_order_and_top_k(self) -> None:
        """Verify that lexical results are sorted by score descending and obey top-k."""
        retriever = LexicalRetriever(db_path=self.test_db_path)
        results = retriever.retrieve("student university academic policy", top_k=3)

        self.assertEqual(len(results), 3)
        for i in range(len(results) - 1):
            self.assertGreaterEqual(
                results[i].score,
                results[i + 1].score,
                "Results must be ordered by score descending",
            )
            self.assertEqual(results[i].rank, i + 1)

    def test_lexical_empty_query_and_empty_db(self) -> None:
        """Verify that empty queries and empty databases return empty results safely."""
        retriever = LexicalRetriever(db_path=self.test_db_path)
        self.assertEqual(retriever.retrieve("", top_k=3), [])
        self.assertEqual(retriever.retrieve("   \t  ", top_k=3), [])

        # Empty DB test
        empty_db_path = Path(self.temp_dir.name) / "empty.db"
        init_db(empty_db_path)
        empty_retriever = LexicalRetriever(db_path=empty_db_path)
        self.assertEqual(empty_retriever.retrieve("Wi-Fi network", top_k=3), [])

    def test_lexical_fallback_mode(self) -> None:
        """Verify that lexical search works deterministically when FTS5 is disabled."""
        retriever = LexicalRetriever(db_path=self.test_db_path, use_fts=False)
        results = retriever.retrieve("Titan-HPC Slurm Turing Hall", top_k=2)

        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].document_id, "DOC-IT-003")
        self.assertIn("Titan-HPC", results[0].content)

    # -------------------------------------------------------------------------
    # Configuration B: Dense Vector Retrieval Tests
    # -------------------------------------------------------------------------

    def test_dense_semantic_retrieval(self) -> None:
        """Verify dense retrieval ranks semantically relevant chunks at top positions."""
        model = DeterministicTestDoubleEmbeddingModel()
        retriever = DenseRetriever(db_path=self.test_db_path, embedding_model=model)

        # Query uses conversational phrasing
        results = retriever.retrieve(
            "What is the penalty for submitting coursework past the deadline?",
            top_k=3,
        )

        self.assertEqual(len(results), 3)
        top_match = results[0]
        self.assertEqual(top_match.document_id, "DOC-ACAD-002")
        self.assertEqual(top_match.rank, 1)
        self.assertGreaterEqual(top_match.score, -1.0)
        self.assertLessEqual(top_match.score, 1.0)

    def test_dense_ranking_order_and_top_k(self) -> None:
        """Verify dense retrieval scores are strictly descending and ranks are 1-indexed."""
        model = DeterministicTestDoubleEmbeddingModel()
        retriever = DenseRetriever(db_path=self.test_db_path, embedding_model=model)
        results = retriever.retrieve("slurm gpu clusters research computing", top_k=4)

        self.assertEqual(len(results), 4)
        for i in range(len(results) - 1):
            self.assertGreaterEqual(results[i].score, results[i + 1].score)
            self.assertEqual(results[i].rank, i + 1)

    def test_dense_empty_query_and_empty_db(self) -> None:
        """Verify dense retrieval returns empty lists on empty queries or empty tables."""
        model = DeterministicTestDoubleEmbeddingModel()
        retriever = DenseRetriever(db_path=self.test_db_path, embedding_model=model)
        self.assertEqual(retriever.retrieve("", top_k=3), [])
        self.assertEqual(retriever.retrieve("   ", top_k=3), [])

        empty_db_path = Path(self.temp_dir.name) / "dense_empty.db"
        init_db(empty_db_path)
        empty_retriever = DenseRetriever(db_path=empty_db_path, embedding_model=model)
        self.assertEqual(empty_retriever.retrieve("research cluster", top_k=3), [])

    def test_dense_embedding_caching_in_sqlite(self) -> None:
        """Verify embeddings are cached in SQLite and reused on subsequent calls."""
        model = DeterministicTestDoubleEmbeddingModel()
        retriever = DenseRetriever(db_path=self.test_db_path, embedding_model=model)

        # First indexing run
        indexed_count = retriever.index_chunks()
        self.assertGreater(indexed_count, 0)

        # Verify database has embedding_json populated for all chunks
        with get_db_connection(self.test_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) AS cnt FROM document_chunks WHERE embedding_json IS NOT NULL;")
            cached_count = cursor.fetchone()["cnt"]
            self.assertEqual(cached_count, indexed_count)

        # Verify second indexing run reuses cached data without recomputing
        with patch.object(model, "encode", wraps=model.encode) as mock_encode:
            retriever_2 = DenseRetriever(db_path=self.test_db_path, embedding_model=model)
            retriever_2.index_chunks()
            # encode should NOT have been called because all chunks already had embedding_json
            mock_encode.assert_not_called()

    # -------------------------------------------------------------------------
    # Shared Interface & Error Handling Tests
    # -------------------------------------------------------------------------

    def test_common_interface_result_structure(self) -> None:
        """Verify both retrievers return identical RetrievalResult instances."""
        lexical = LexicalRetriever(db_path=self.test_db_path)
        dense = DenseRetriever(db_path=self.test_db_path)

        lex_res = lexical.retrieve("library borrowing", top_k=2)
        dense_res = dense.retrieve("library borrowing", top_k=2)

        for res in lex_res + dense_res:
            self.assertIsInstance(res, RetrievalResult)
            self.assertIsInstance(res.chunk_id, str)
            self.assertIsInstance(res.document_id, str)
            self.assertIsInstance(res.content, str)
            self.assertIsInstance(res.score, float)
            self.assertIsInstance(res.rank, int)

            as_dict = res.to_dict()
            self.assertEqual(
                set(as_dict.keys()),
                {"chunk_id", "document_id", "content", "score", "rank"},
            )

    def test_invalid_top_k_raises_value_error(self) -> None:
        """Verify that non-positive top_k raises ValueError."""
        lexical = LexicalRetriever(db_path=self.test_db_path)
        dense = DenseRetriever(db_path=self.test_db_path)

        for ret in (lexical, dense):
            with self.assertRaises(ValueError):
                ret.retrieve("query", top_k=0)
            with self.assertRaises(ValueError):
                ret.retrieve("query", top_k=-2)
            with self.assertRaises(ValueError):
                ret.retrieve("query", top_k="invalid")  # type: ignore

    def test_missing_model_error_reporting(self) -> None:
        """Verify that SentenceTransformerEmbeddingModel reports clear setup errors when unavailable."""
        # When attempting to load an unavailable model or when sentence_transformers is mocked as missing
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with self.assertRaises(EmbeddingModelUnavailableError) as ctx:
                SentenceTransformerEmbeddingModel("non-existent-model")
            self.assertIn("sentence-transformers", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
