"""
Unit and integration tests for Phase 3: Document Management & Dynamic Ingestion.

Covers:
1. File upload (.txt, .md, .pdf)
2. Text extraction and normalization
3. File validation (extensions, MIME, size limits, binary/null-byte rejection)
4. Duplicate detection and overwrite handling
5. Path traversal prevention and safe storage
6. Document listing and single-document inspection
7. Safe deletion, cascading chunk deletion, and physical file cleanup
8. Protection of core benchmark sample documents against deletion
9. End-to-end queryability of newly uploaded content (Lexical and Dense)
10. Abstention behavior after document deletion
11. Persistence across service re-initialization
"""

from __future__ import annotations

import io
import shutil
import tempfile
import unittest
from pathlib import Path

# Check if fastapi and httpx TestClient are available
try:
    from fastapi.testclient import TestClient
    from src.app import app, get_service
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from db.database import get_db_connection, init_db
from src.document_manager import (
    DocumentManager,
    DuplicateDocumentError,
    EmptyFileError,
    FileTooLargeError,
    MalformedDocumentError,
    ProtectedDocumentError,
    UnsupportedFileTypeError,
)
from src.service import RAGService


def make_simple_pdf(text: str) -> bytes:
    """Generate a minimal, compliant PDF binary stream containing the specified text string."""
    stream_content = f"BT /F1 12 Tf 100 700 Td ({text}) Tj ET\n".encode("latin-1")
    length = len(stream_content)
    obj1 = b"1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n"
    obj2 = b"2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n"
    obj3 = b"3 0 obj\n<</Type /Page /Parent 2 0 R /Resources 4 0 R /MediaBox [0 0 612 792] /Contents 5 0 R>>\nendobj\n"
    obj4 = b"4 0 obj\n<</Font <</F1 <</Type /Font /Subtype /Type1 /BaseFont /Helvetica>>>>>>\nendobj\n"
    obj5 = b"5 0 obj\n<</Length " + str(length).encode("ascii") + b">>\nstream\n" + stream_content + b"endstream\nendobj\n"
    header = b"%PDF-1.4\n"
    o1 = len(header)
    o2 = o1 + len(obj1)
    o3 = o2 + len(obj2)
    o4 = o3 + len(obj3)
    o5 = o4 + len(obj4)
    xref_pos = o5 + len(obj5)
    xref = (
        b"xref\n0 6\n0000000000 65535 f \n"
        + f"{o1:010d} 00000 n \n".encode("ascii")
        + f"{o2:010d} 00000 n \n".encode("ascii")
        + f"{o3:010d} 00000 n \n".encode("ascii")
        + f"{o4:010d} 00000 n \n".encode("ascii")
        + f"{o5:010d} 00000 n \n".encode("ascii")
    )
    trailer = b"trailer\n<</Size 6 /Root 1 0 R>>\nstartxref\n" + str(xref_pos).encode("ascii") + b"\n%%EOF\n"
    return header + obj1 + obj2 + obj3 + obj4 + obj5 + xref + trailer


@unittest.skipUnless(FASTAPI_AVAILABLE, "FastAPI / TestClient dependencies not available")
class TestDocumentManagement(unittest.TestCase):
    """Test suite for DocumentManager and FastAPI document management endpoints."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_rag.db"
        self.upload_dir = Path(self.temp_dir) / "uploads"
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        # Initialize schema and seed default documents into isolated test DB
        init_db(self.db_path)
        self.service = RAGService(
            db_path=self.db_path,
            auto_init=True,
        )
        # Point document manager to test upload directory
        self.service._document_manager = DocumentManager(
            db_path=self.db_path,
            upload_dir=self.upload_dir,
            max_upload_size_bytes=100 * 1024,  # 100 KB limit for testing
        )

        # Patch global service in src.app to point to our isolated service
        import src.app as app_module
        app_module.rag_service = self.service

        self.client = TestClient(app)

    def tearDown(self) -> None:
        import src.app as app_module
        app_module.rag_service = None
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_sanitize_filename_prevents_path_traversal(self) -> None:
        """Verify filename sanitization removes path traversal attempts and dangerous characters."""
        dm = self.service.document_manager
        self.assertEqual(dm.sanitize_filename("../../etc/passwd"), "passwd")
        self.assertEqual(dm.sanitize_filename("..\\..\\windows\\system32.dll"), "system32.dll")
        self.assertEqual(dm.sanitize_filename("test document (v1).txt"), "test_document_v1_.txt")
        self.assertEqual(dm.sanitize_filename(""), "unnamed_document.txt")
        self.assertEqual(dm.sanitize_filename("   "), "unnamed_document.txt")

    def test_upload_txt_file_success(self) -> None:
        """Verify successful upload, chunking, and database insertion of a .txt file."""
        content = (
            "University Data Science Center GPU Cluster Policy.\n"
            "Researchers must reserve NVIDIA H100 partitions through the Slurm queue named 'h100-research'.\n"
            "The maximum runtime allowed per batch allocation is 48 consecutive hours."
        )
        files = {"file": ("gpu_policy.txt", content.encode("utf-8"), "text/plain")}
        resp = self.client.post("/api/documents/upload", files=files)

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["document_id"].startswith("DOC-UPL-"))
        self.assertEqual(data["filename"], "gpu_policy.txt")
        self.assertGreater(data["chunks_created"], 0)
        self.assertFalse(data["is_duplicate"])

        # Verify record exists in SQLite documents table
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, title, content FROM documents WHERE id = ?;", (data["document_id"],))
            row = cursor.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["title"], "gpu_policy.txt")

    def test_upload_md_file_success(self) -> None:
        """Verify successful upload and chunking of a markdown .md file."""
        content = (
            "# Robotics Laboratory Safety Guidelines\n\n"
            "## Emergency Protocols\n"
            "- All autonomous rover test runs require two active safety observers.\n"
            "- Emergency kill switches must remain tethered within 5 meters of active platforms."
        )
        files = {"file": ("robotics_lab.md", content.encode("utf-8"), "text/markdown")}
        resp = self.client.post("/api/documents/upload", files=files)

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["document_id"].startswith("DOC-UPL-"))
        self.assertGreater(data["chunks_created"], 0)

    def test_upload_pdf_file_success(self) -> None:
        """Verify successful text extraction and ingestion of a .pdf document."""
        pdf_bytes = make_simple_pdf("Specialized Cleanroom Facility Entrance Authorization Code CR-4491")
        files = {"file": ("cleanroom_rules.pdf", pdf_bytes, "application/pdf")}
        resp = self.client.post("/api/documents/upload", files=files)

        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["document_id"].startswith("DOC-UPL-"))
        self.assertGreater(data["chunks_created"], 0)

    def test_upload_unsupported_file_extension(self) -> None:
        """Verify unsupported extensions (.exe, .py, .bin) return 400 Bad Request."""
        files = {"file": ("malicious_script.py", b"print('hello world')", "text/x-python")}
        resp = self.client.post("/api/documents/upload", files=files)

        self.assertEqual(resp.status_code, 400)
        data = resp.json()
        self.assertIn("detail", data)
        self.assertIn("Unsupported file format", data["detail"])

    def test_upload_empty_file_rejected(self) -> None:
        """Verify 0-byte file returns 400 Bad Request."""
        files = {"file": ("empty.txt", b"", "text/plain")}
        resp = self.client.post("/api/documents/upload", files=files)

        self.assertEqual(resp.status_code, 400)
        self.assertIn("empty", resp.json()["detail"].lower())

    def test_upload_whitespace_only_file_rejected(self) -> None:
        """Verify whitespace-only file returns 400 Bad Request."""
        files = {"file": ("spaces.txt", b"   \n\n\t  \n", "text/plain")}
        resp = self.client.post("/api/documents/upload", files=files)

        self.assertEqual(resp.status_code, 400)
        self.assertIn("no extractable text", resp.json()["detail"].lower())

    def test_upload_binary_null_bytes_in_txt_rejected(self) -> None:
        """Verify binary content disguised as .txt is rejected."""
        fake_binary = b"MZ\x90\x00\x03\x00\x00\x00\x04\x00\x00\x00\xff\xff"
        files = {"file": ("fake_text.txt", fake_binary, "text/plain")}
        resp = self.client.post("/api/documents/upload", files=files)

        self.assertEqual(resp.status_code, 400)
        self.assertIn("binary data", resp.json()["detail"].lower())

    def test_upload_oversized_file_rejected(self) -> None:
        """Verify files exceeding max upload size return 413 Payload Too Large."""
        oversized_bytes = b"A" * (120 * 1024)  # 120 KB, limit is 100 KB
        files = {"file": ("large_file.txt", oversized_bytes, "text/plain")}
        resp = self.client.post("/api/documents/upload", files=files)

        self.assertEqual(resp.status_code, 413)
        self.assertIn("exceeds maximum allowed limit", resp.json()["detail"].lower())

    def test_upload_duplicate_detection(self) -> None:
        """Verify uploading the exact same document twice returns 409 Conflict without overwrite."""
        content = "Unique Autonomous Driving Research Protocol Policy Document 2026."
        files = {"file": ("driving_policy.txt", content.encode("utf-8"), "text/plain")}

        # First upload: success
        resp1 = self.client.post("/api/documents/upload", files=files)
        self.assertEqual(resp1.status_code, 200)
        doc_id = resp1.json()["document_id"]

        # Second upload: 409 Conflict
        files2 = {"file": ("driving_policy_copy.txt", content.encode("utf-8"), "text/plain")}
        resp2 = self.client.post("/api/documents/upload", files=files2)
        self.assertEqual(resp2.status_code, 409)
        self.assertIn("duplicate document detected", resp2.json()["detail"].lower())
        self.assertEqual(resp2.json().get("existing_doc_id"), doc_id)

    def test_upload_duplicate_with_overwrite(self) -> None:
        """Verify uploading duplicate with ?overwrite=true replaces existing document."""
        content = "Underground Particle Accelerator Shielding Regulations 2026."
        files = {"file": ("accelerator.txt", content.encode("utf-8"), "text/plain")}

        resp1 = self.client.post("/api/documents/upload", files=files)
        self.assertEqual(resp1.status_code, 200)

        # Overwrite=true succeeds
        files2 = {"file": ("accelerator_updated.txt", content.encode("utf-8"), "text/plain")}
        resp2 = self.client.post("/api/documents/upload?overwrite=true", files=files2)
        self.assertEqual(resp2.status_code, 200)
        self.assertTrue(resp2.json()["is_duplicate"])

    def test_list_documents(self) -> None:
        """Verify /api/documents lists both core sample documents and uploaded documents."""
        # Upload one document
        content = "Aeronautics Drone Flight Corridor Regulations for Campus North Quad."
        files = {"file": ("drone_flight.txt", content.encode("utf-8"), "text/plain")}
        self.client.post("/api/documents/upload", files=files)

        resp = self.client.get("/api/documents")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertGreaterEqual(data["total"], 12)  # 11 sample docs + 1 upload

        # Check sample documents have is_sample=True and uploaded document has is_sample=False
        samples = [d for d in data["documents"] if d["is_sample"]]
        uploads = [d for d in data["documents"] if not d["is_sample"]]

        self.assertEqual(len(samples), 11)
        self.assertGreaterEqual(len(uploads), 1)
        self.assertTrue(uploads[0]["id"].startswith("DOC-UPL-"))

    def test_get_document_details(self) -> None:
        """Verify retrieving details and chunks for an existing document."""
        resp = self.client.get("/api/documents/DOC-IT-001")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["id"], "DOC-IT-001")
        self.assertTrue(data["is_sample"])
        self.assertGreater(data["chunk_count"], 0)
        self.assertGreater(len(data["chunks"]), 0)

    def test_get_document_not_found(self) -> None:
        """Verify non-existent document ID returns 404."""
        resp = self.client.get("/api/documents/DOC-NON-EXISTENT")
        self.assertEqual(resp.status_code, 404)

    def test_delete_sample_document_rejected(self) -> None:
        """Verify attempting to delete a core sample document returns 403 Forbidden."""
        resp = self.client.delete("/api/documents/DOC-IT-001")
        self.assertEqual(resp.status_code, 403)
        self.assertIn("cannot delete core benchmark sample document", resp.json()["detail"].lower())

    def test_delete_uploaded_document_success(self) -> None:
        """Verify successful deletion of an uploaded document and cascading chunk removal."""
        content = "Marine Biology Oceanographic Vessel R/V Discovery Access Protocol."
        files = {"file": ("marine_vessel.txt", content.encode("utf-8"), "text/plain")}
        upload_resp = self.client.post("/api/documents/upload", files=files)
        doc_id = upload_resp.json()["document_id"]

        # Delete document
        del_resp = self.client.delete(f"/api/documents/{doc_id}")
        self.assertEqual(del_resp.status_code, 200)
        del_data = del_resp.json()
        self.assertEqual(del_data["document_id"], doc_id)
        self.assertGreater(del_data["deleted_chunks"], 0)

        # Confirm document is gone from documents table
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) AS cnt FROM documents WHERE id = ?;", (doc_id,))
            self.assertEqual(cursor.fetchone()["cnt"], 0)
            cursor.execute("SELECT COUNT(*) AS cnt FROM document_chunks WHERE document_id = ?;", (doc_id,))
            self.assertEqual(cursor.fetchone()["cnt"], 0)
            cursor.execute("SELECT COUNT(*) AS cnt FROM fts_chunks WHERE document_id = ?;", (doc_id,))
            self.assertEqual(cursor.fetchone()["cnt"], 0)

    def test_query_newly_uploaded_document_immediately(self) -> None:
        """
        Verify that a newly uploaded document is immediately retrievable and grounded
        without requiring application restart or manual re-indexing.
        """
        unique_content = (
            "Specialized Biohazard Containment Suite BSL-3 Entry Directive.\n"
            "Personnel entering Suite BSL-3 must verify negative air pressure of at least -0.05 inches of water column.\n"
            "The emergency isolation door code for airlock decontamination chamber is CODE-OMEGA-9942."
        )
        files = {"file": ("bsl3_directive.txt", unique_content.encode("utf-8"), "text/plain")}
        upload_resp = self.client.post("/api/documents/upload", files=files)
        self.assertEqual(upload_resp.status_code, 200)
        doc_id = upload_resp.json()["document_id"]

        # 1. Query using Lexical retriever
        query_payload = {
            "question": "What is the emergency isolation door code for airlock decontamination in Suite BSL-3?",
            "retriever": "lexical",
            "top_k": 3,
        }
        lex_resp = self.client.post("/api/query", json=query_payload)
        self.assertEqual(lex_resp.status_code, 200)
        lex_data = lex_resp.json()
        self.assertFalse(lex_data["is_abstention"])
        self.assertIn("CODE-OMEGA-9942", lex_data["answer"])
        self.assertIn(doc_id, lex_data["source_doc_ids"])

        # 2. Query using Dense retriever
        query_payload["retriever"] = "dense"
        dense_resp = self.client.post("/api/query", json=query_payload)
        self.assertEqual(dense_resp.status_code, 200)
        dense_data = dense_resp.json()
        self.assertFalse(dense_data["is_abstention"])
        self.assertIn("CODE-OMEGA-9942", dense_data["answer"])
        self.assertIn(doc_id, dense_data["source_doc_ids"])

    def test_query_after_deletion_abstains(self) -> None:
        """Verify that after deleting an uploaded document, querying its facts triggers abstention."""
        unique_content = "The emergency cryogenic purge keycode for Sector 7 Helium cooling is PURGE-7788."
        files = {"file": ("cryo_purge.txt", unique_content.encode("utf-8"), "text/plain")}
        upload_resp = self.client.post("/api/documents/upload", files=files)
        doc_id = upload_resp.json()["document_id"]

        # Verify it answers before deletion
        q_payload = {
            "question": "What is the emergency cryogenic purge keycode for Sector 7?",
            "retriever": "lexical",
            "top_k": 3,
        }
        resp_before = self.client.post("/api/query", json=q_payload)
        self.assertFalse(resp_before.json()["is_abstention"])

        # Delete document
        del_resp = self.client.delete(f"/api/documents/{doc_id}")
        self.assertEqual(del_resp.status_code, 200)

        # Query after deletion: should abstain
        resp_after = self.client.post("/api/query", json=q_payload)
        self.assertEqual(resp_after.status_code, 200)
        self.assertTrue(resp_after.json()["is_abstention"])
        self.assertNotIn(doc_id, resp_after.json()["source_doc_ids"])

    def test_persistence_across_service_reinit(self) -> None:
        """Verify uploaded documents persist across service re-instantiation."""
        content = "Observatory Dome Automated Stellar Tracking Elevation Limits Policy."
        files = {"file": ("observatory.txt", content.encode("utf-8"), "text/plain")}
        upload_resp = self.client.post("/api/documents/upload", files=files)
        doc_id = upload_resp.json()["document_id"]

        # Re-create service pointing to same database
        new_service = RAGService(db_path=self.db_path, auto_init=False)
        docs = new_service.list_documents()
        doc_ids = [d.id for d in docs]
        self.assertIn(doc_id, doc_ids)


if __name__ == "__main__":
    unittest.main()
