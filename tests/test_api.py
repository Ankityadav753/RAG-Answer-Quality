"""
API and Web Interface Tests for DAI-009 Phase 2.

Tests:
1. Health and System Info endpoints (/health, /api/health, /api/info).
2. Frontend index.html and static asset availability.
3. Answerable query execution with lexical retrieval.
4. Out-of-domain query execution triggering automated abstention.
5. Dense vector retrieval execution via API parameter.
6. Validation error handling for whitespace, empty questions, and out-of-range top-k.
7. Unhandled exception masking ensuring zero credential or secret leakage.
"""

import unittest
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
    from src.app import app
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


@unittest.skipUnless(FASTAPI_AVAILABLE, "FastAPI or TestClient not installed in current environment")
class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if FASTAPI_AVAILABLE:
            cls.client = TestClient(app)

    def test_health_endpoint(self) -> None:
        """Verify GET /health returns service readiness and indexed corpus status."""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("is_offline", data)
        self.assertIn("llm_provider", data)
        self.assertIn("embedding_provider", data)
        self.assertGreater(data["chunks_count"], 0)
        self.assertGreater(data["documents_count"], 0)

    def test_api_health_alias(self) -> None:
        """Verify GET /api/health returns identical health response."""
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "healthy")

    def test_api_info_endpoint_has_no_secrets(self) -> None:
        """Verify GET /api/info returns public config without exposing secrets."""
        resp = self.client.get("/api/info")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("available_retrievers", data)
        self.assertIn("lexical", data["available_retrievers"])
        self.assertIn("dense", data["available_retrievers"])

        # Verify no keys or credentials leaked
        body_str = resp.text.lower()
        self.assertNotIn("secret", body_str)
        self.assertNotIn("api_key", body_str)
        self.assertNotIn("password", body_str)

    def test_frontend_index_serves_html(self) -> None:
        """Verify GET / serves single-page application index.html."""
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/html", resp.headers["content-type"])
        self.assertIn("DAI-009: RAG Answer Quality", resp.text)
        self.assertIn('id="question-input"', resp.text)
        self.assertIn('id="ask-button"', resp.text)

    def test_frontend_static_assets(self) -> None:
        """Verify static CSS and JS files are served correctly."""
        resp_css = self.client.get("/static/app.css")
        self.assertEqual(resp_css.status_code, 200)
        self.assertIn("css", resp_css.headers.get("content-type", ""))

        resp_js = self.client.get("/static/app.js")
        self.assertEqual(resp_js.status_code, 200)

    def test_query_answerable_lexical(self) -> None:
        """Verify POST /api/query produces grounded answer with citations."""
        payload = {
            "question": "What is the primary WPA3 wireless network SSID for students?",
            "retriever": "lexical",
            "top_k": 3,
        }
        resp = self.client.post("/api/query", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertFalse(data["is_abstention"])
        self.assertIn("Campus-Secure-5G", data["answer"])
        self.assertIn("DOC-IT-001", data["source_doc_ids"])
        self.assertGreater(len(data["source_chunk_ids"]), 0)
        self.assertEqual(len(data["retrieved_chunks"]), 3)
        self.assertGreater(data["latency_ms"], 0.0)

    def test_query_unanswerable_abstention(self) -> None:
        """Verify POST /api/query abstains on out-of-domain questions."""
        payload = {
            "question": "What are the recreational swimming pool lane hours and lap reservation rules?",
            "retriever": "lexical",
            "top_k": 3,
        }
        resp = self.client.post("/api/query", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertTrue(data["is_abstention"])
        self.assertIn("cannot find sufficient information", data["answer"].lower())
        self.assertEqual(data["source_doc_ids"], [])
        self.assertEqual(data["source_chunk_ids"], [])

    def test_query_dense_retriever(self) -> None:
        """Verify POST /api/query executes with dense retriever."""
        payload = {
            "question": "Titan-HPC GPU computing cluster access keys",
            "retriever": "dense",
            "top_k": 2,
        }
        resp = self.client.post("/api/query", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertIn("dense_cosine", data["retriever_name"])
        self.assertEqual(len(data["retrieved_chunks"]), 2)

    def test_query_invalid_empty_or_whitespace_question(self) -> None:
        """Verify empty or whitespace-only questions return 422."""
        resp1 = self.client.post("/api/query", json={"question": ""})
        self.assertEqual(resp1.status_code, 422)

        resp2 = self.client.post("/api/query", json={"question": "   \n\t  "})
        self.assertEqual(resp2.status_code, 422)

    def test_query_invalid_top_k(self) -> None:
        """Verify out-of-bounds top-k returns 422."""
        resp_low = self.client.post("/api/query", json={"question": "Test?", "top_k": 0})
        self.assertEqual(resp_low.status_code, 422)

        resp_high = self.client.post("/api/query", json={"question": "Test?", "top_k": 50})
        self.assertEqual(resp_high.status_code, 422)

    def test_query_invalid_retriever(self) -> None:
        """Verify unsupported retriever returns 422."""
        resp = self.client.post("/api/query", json={"question": "Test?", "retriever": "quantum"})
        self.assertEqual(resp.status_code, 422)

    def test_unhandled_exception_masks_secrets(self) -> None:
        """Verify internal server errors return generic message without secret leaks."""
        from src.app import get_service

        service = get_service()
        secret_leak = "sk-super-secret-key-that-must-never-be-leaked"

        with patch.object(service, "query", side_effect=RuntimeError(f"Simulated DB crash with {secret_leak}")):
            resp = self.client.post("/api/query", json={"question": "Trigger error"})
            self.assertEqual(resp.status_code, 500)
            self.assertNotIn(secret_leak, resp.text)
            self.assertIn("internal error", resp.json()["detail"].lower())

    def test_favicon_endpoint(self) -> None:
        """Verify favicon returns 204 No Content without error."""
        resp = self.client.get("/favicon.ico")
        self.assertEqual(resp.status_code, 204)

    def test_list_documents_endpoint(self) -> None:
        """Verify GET /api/documents returns registered documents."""
        resp = self.client.get("/api/documents")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("total", data)
        self.assertIn("documents", data)
        self.assertGreaterEqual(data["total"], 11)


if __name__ == "__main__":
    unittest.main()


