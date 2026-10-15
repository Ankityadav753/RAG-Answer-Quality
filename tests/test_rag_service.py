"""
Unit and integration tests for the unified RAGService (DAI-009 Phase 1).

Tests:
1. Single-question workflow returning answers, citations, and metadata.
2. Unanswerable question workflow triggering abstention guardrail.
3. Service initialization with lexical vs dense retrievers.
4. Custom top-k parameter overrides.
5. Ingestion and auto-initialization verification.
6. Serialization of RAGResponse to dictionary.
"""

import tempfile
import unittest
from pathlib import Path

from db.database import init_db
from src.embeddings import DeterministicTestDoubleEmbeddingModel
from src.ingest import ingest_documents
from src.retriever import DenseRetriever, LexicalRetriever
from src.service import RAGResponse, RAGService


class TestRAGService(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_db_path = Path(self.temp_dir.name) / "test_service.db"
        init_db(self.test_db_path)
        ingest_documents(db_path=self.test_db_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_answerable_query_workflow_lexical(self) -> None:
        """Verify unified workflow produces grounded answer, sources, and no abstention."""
        service = RAGService(
            db_path=self.test_db_path,
            retriever_type="lexical",
            top_k=3,
            auto_init=False,
        )

        resp = service.query("What is the primary WPA3 wireless network SSID for students?")
        self.assertIsInstance(resp, RAGResponse)
        self.assertFalse(resp.is_abstention)
        self.assertIn("Campus-Secure-5G", resp.answer)
        self.assertIn("DOC-IT-001", resp.source_doc_ids)
        self.assertGreater(len(resp.source_chunk_ids), 0)
        self.assertEqual(len(resp.retrieved_chunks), 3)
        self.assertGreater(resp.latency_ms, 0.0)

    def test_unanswerable_query_workflow_abstention(self) -> None:
        """Verify unanswerable query triggers abstention and suppresses sources."""
        service = RAGService(
            db_path=self.test_db_path,
            retriever_type="lexical",
            top_k=3,
            auto_init=False,
        )

        resp = service.query("What are the recreational swimming pool lane hours and lap reservation rules?")
        self.assertIsInstance(resp, RAGResponse)
        self.assertTrue(resp.is_abstention)
        self.assertIn("cannot find sufficient information", resp.answer.lower())
        self.assertEqual(resp.source_doc_ids, [])
        self.assertEqual(resp.source_chunk_ids, [])

    def test_dense_retriever_service_workflow(self) -> None:
        """Verify service operates correctly with DenseRetriever and test double embeddings."""
        emb_model = DeterministicTestDoubleEmbeddingModel()
        service = RAGService(
            db_path=self.test_db_path,
            retriever_type="dense",
            embedding_model=emb_model,
            top_k=2,
            auto_init=False,
        )

        resp = service.query("Tell me about university wireless network.", top_k=2)
        self.assertIsInstance(resp, RAGResponse)
        self.assertEqual(len(resp.retrieved_chunks), 2)
        self.assertIn("dense_cosine", resp.retriever_name)

    def test_top_k_override(self) -> None:
        """Verify query-level top-k overrides service-level top-k."""
        service = RAGService(
            db_path=self.test_db_path,
            retriever_type="lexical",
            top_k=5,
            auto_init=False,
        )

        resp1 = service.query("wireless network", top_k=1)
        self.assertEqual(len(resp1.retrieved_chunks), 1)

        resp2 = service.query("wireless network", top_k=3)
        self.assertEqual(len(resp2.retrieved_chunks), 3)

    def test_response_to_dict_serialization(self) -> None:
        """Verify RAGResponse serializes cleanly to dict."""
        service = RAGService(
            db_path=self.test_db_path,
            retriever_type="lexical",
            top_k=2,
            auto_init=False,
        )

        resp = service.query("SSID for Wi-Fi")
        d = resp.to_dict()
        self.assertIn("question", d)
        self.assertIn("answer", d)
        self.assertIn("is_abstention", d)
        self.assertIn("retrieved_chunks", d)
        self.assertIn("source_chunk_ids", d)
        self.assertIn("source_doc_ids", d)
        self.assertIn("latency_ms", d)
        self.assertIsInstance(d["retrieved_chunks"], list)
        if d["retrieved_chunks"]:
            self.assertIsInstance(d["retrieved_chunks"][0], dict)


if __name__ == "__main__":
    unittest.main()
