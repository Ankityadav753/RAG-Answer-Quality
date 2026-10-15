"""
Tests for embedding cache safety and model identity validation (DAI-009 Phase 1).

Tests:
1. Storing embedding model identity, vector dimension, and payload in SQLite cache.
2. Invalidation and safe recomputation when switching to a model with different dimension.
3. Invalidation and safe recomputation when switching to a model with different identity.
4. Legacy unversioned list cache handling and dimension safety.
5. Resilient recovery from corrupted cache strings.
6. Dimension-mismatch crash prevention during retrieval calls.
"""

import json
import tempfile
import unittest
from pathlib import Path
from typing import List

import numpy as np

from db.database import get_db_connection, init_db
from src.embeddings import BaseEmbeddingModel, DeterministicTestDoubleEmbeddingModel
from src.ingest import ingest_documents
from src.retriever import DenseRetriever


class DummyTestModel(BaseEmbeddingModel):
    """Configurable embedding model double for cache testing."""

    def __init__(self, name: str, dim: int) -> None:
        self._name = name
        self._dim = dim
        self.call_count = 0

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._name

    def encode(self, texts: List[str]) -> np.ndarray:
        self.call_count += len(texts)
        if not texts:
            return np.empty((0, self._dim), dtype=np.float32)
        # Produce simple deterministic unit vectors of self._dim
        vecs = np.ones((len(texts), self._dim), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms


class TestCacheSafety(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_db_path = Path(self.temp_dir.name) / "cache_safety.db"
        init_db(self.test_db_path)
        ingest_documents(db_path=self.test_db_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_structured_cache_storage(self) -> None:
        """Verify indexed chunks store metadata: model identity, dimension, and vector."""
        model = DeterministicTestDoubleEmbeddingModel(dimension=64)
        retriever = DenseRetriever(db_path=self.test_db_path, embedding_model=model)
        count = retriever.index_chunks()
        self.assertGreater(count, 0)

        with get_db_connection(self.test_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT embedding_json FROM document_chunks WHERE embedding_json IS NOT NULL;")
            rows = cursor.fetchall()
            self.assertEqual(len(rows), count)

            first_payload = json.loads(rows[0]["embedding_json"])
            self.assertIsInstance(first_payload, dict)
            self.assertEqual(first_payload["model"], "deterministic-test-double-v1")
            self.assertEqual(first_payload["dim"], 64)
            self.assertIsInstance(first_payload["vector"], list)
            self.assertEqual(len(first_payload["vector"]), 64)

    def test_dimension_mismatch_prevents_crash_and_recomputes(self) -> None:
        """Verify changing to a model with a different dimension safely recomputes rather than crashing."""
        # 1. Index with Model A (dim 32)
        model_a = DummyTestModel(name="test-model-32d", dim=32)
        retriever_a = DenseRetriever(db_path=self.test_db_path, embedding_model=model_a)
        retriever_a.index_chunks()
        self.assertEqual(retriever_a._embeddings_matrix.shape[1], 32)

        # 2. Switch to Model B (dim 128) on the same database
        model_b = DummyTestModel(name="test-model-128d", dim=128)
        retriever_b = DenseRetriever(db_path=self.test_db_path, embedding_model=model_b)

        # Previously, this would append 32-dim vectors to 128-dim arrays and crash.
        # Now, it must detect dimension and model mismatch, reject cached 32-dim vectors, and recompute.
        indexed_count = retriever_b.index_chunks()
        self.assertGreater(indexed_count, 0)
        self.assertEqual(retriever_b._embeddings_matrix.shape[1], 128)
        self.assertGreater(model_b.call_count, 0)

        # Verify DB records updated to 128-dim
        with get_db_connection(self.test_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT embedding_json FROM document_chunks LIMIT 1;")
            payload = json.loads(cursor.fetchone()["embedding_json"])
            self.assertEqual(payload["model"], "test-model-128d")
            self.assertEqual(payload["dim"], 128)
            self.assertEqual(len(payload["vector"]), 128)

    def test_model_identity_mismatch_recomputes_even_with_same_dimension(self) -> None:
        """Verify models with the same dimension but different identities are not cross-contaminated."""
        model_v1 = DummyTestModel(name="custom-model-v1", dim=64)
        retriever_v1 = DenseRetriever(db_path=self.test_db_path, embedding_model=model_v1)
        retriever_v1.index_chunks()

        model_v2 = DummyTestModel(name="custom-model-v2", dim=64)
        retriever_v2 = DenseRetriever(db_path=self.test_db_path, embedding_model=model_v2)

        # Must recompute because model_name changed from v1 to v2
        retriever_v2.index_chunks()
        self.assertGreater(model_v2.call_count, 0)

        with get_db_connection(self.test_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT embedding_json FROM document_chunks LIMIT 1;")
            payload = json.loads(cursor.fetchone()["embedding_json"])
            self.assertEqual(payload["model"], "custom-model-v2")

    def test_legacy_format_dimension_mismatch_handling(self) -> None:
        """Verify legacy unversioned list format with mismatched dimension is safely recomputed."""
        # Manually inject legacy 10-dim list into SQLite
        with get_db_connection(self.test_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE document_chunks SET embedding_json = ?;", (json.dumps([0.1] * 10),))

        # Run retriever expecting 64-dim
        model = DeterministicTestDoubleEmbeddingModel(dimension=64)
        retriever = DenseRetriever(db_path=self.test_db_path, embedding_model=model)
        retriever.index_chunks()

        self.assertEqual(retriever._embeddings_matrix.shape[1], 64)

    def test_corrupted_json_cache_recovery(self) -> None:
        """Verify corrupted JSON cache strings are gracefully rejected and recomputed."""
        with get_db_connection(self.test_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE document_chunks SET embedding_json = 'CORRUPTED_NOT_JSON';")

        model = DeterministicTestDoubleEmbeddingModel(dimension=64)
        retriever = DenseRetriever(db_path=self.test_db_path, embedding_model=model)
        count = retriever.index_chunks()
        self.assertGreater(count, 0)
        self.assertEqual(retriever._embeddings_matrix.shape[1], 64)

    def test_safe_retrieve_across_model_swap(self) -> None:
        """Verify retrieve() automatically re-indexes if model dimension changes."""
        model_32 = DummyTestModel(name="model-32", dim=32)
        retriever = DenseRetriever(db_path=self.test_db_path, embedding_model=model_32)
        retriever.index_chunks()

        # Swap model to 64-dim
        model_64 = DeterministicTestDoubleEmbeddingModel(dimension=64)
        retriever.embedding_model = model_64

        # Calling retrieve should safely re-index and produce valid results
        results = retriever.retrieve("wireless network", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(retriever._embeddings_matrix.shape[1], 64)


if __name__ == "__main__":
    unittest.main()
