"""
Document management and dynamic ingestion engine for DAI-009: RAG Answer Quality.

Handles:
1. File validation (extensions, MIME, size limits, binary/null-byte rejection).
2. Text extraction for .txt, .md, and .pdf (via lightweight pure-Python pypdf).
3. Deterministic and collision-resistant document ID generation (DOC-UPL-<hash>).
4. Safe filesystem persistence preventing path traversal.
5. Ingestion and transactional indexing into SQLite (documents, chunks, FTS5).
6. Document listing, metadata inspection, and safe cascading deletion.
7. Protection of core benchmark sample documents against deletion.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root in sys.path
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from db.database import DEFAULT_DB_PATH, get_db_connection, init_db
from src.chunker import chunk_text
from src.config import Settings, get_settings

logger = logging.getLogger("rag_answer_quality.document_manager")


# =============================================================================
# Custom Exception Hierarchy
# =============================================================================

class DocumentManagementError(Exception):
    """Base exception for document management and upload operations."""
    pass


class UnsupportedFileTypeError(DocumentManagementError):
    """Raised when an uploaded file extension or media type is not supported."""
    pass


class FileTooLargeError(DocumentManagementError):
    """Raised when file size exceeds the configured maximum upload limit."""
    pass


class EmptyFileError(DocumentManagementError):
    """Raised when uploaded file contains zero bytes or no extractable text."""
    pass


class MalformedDocumentError(DocumentManagementError):
    """Raised when document content is corrupted, binary, or unparseable."""
    pass


class DuplicateDocumentError(DocumentManagementError):
    """Raised when an uploaded document with identical content has already been ingested."""
    def __init__(self, message: str, existing_doc_id: str):
        super().__init__(message)
        self.existing_doc_id = existing_doc_id


class DocumentNotFoundError(DocumentManagementError):
    """Raised when attempting to access or delete a non-existent document ID."""
    pass


class ProtectedDocumentError(DocumentManagementError):
    """Raised when attempting to delete a core benchmark sample document."""
    pass


# =============================================================================
# DTOs
# =============================================================================

@dataclass(frozen=True)
class DocumentMetadata:
    """Document record representation."""
    id: str
    title: str
    source: Optional[str]
    created_at: str
    chunk_count: int
    content_length: int
    is_sample: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class UploadResult:
    """Result of an upload and ingestion operation."""
    document_id: str
    filename: str
    chunks_created: int
    content_length: int
    is_duplicate: bool = False
    storage_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Document Manager Class
# =============================================================================

class DocumentManager:
    """
    Manages document upload, validation, text extraction, safe persistence,
    chunking, SQLite indexing, and cascading deletion.
    """

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        upload_dir: Optional[Union[str, Path]] = None,
        max_upload_size_bytes: Optional[int] = None,
        allowed_extensions: Optional[Tuple[str, ...]] = None,
    ) -> None:
        settings: Settings = get_settings()

        self.db_path = Path(db_path).resolve() if db_path else settings.db_path
        self.upload_dir = Path(upload_dir).resolve() if upload_dir else settings.upload_dir
        self.max_upload_size_bytes = (
            max_upload_size_bytes
            if max_upload_size_bytes is not None
            else settings.max_upload_size_bytes
        )
        self.allowed_extensions = (
            allowed_extensions
            if allowed_extensions is not None
            else settings.allowed_upload_extensions
        )

        # Ensure database and upload directories exist
        init_db(self.db_path)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize uploaded filename to prevent directory traversal and filesystem attacks.
        Only allows alphanumeric characters, dots, dashes, and underscores.
        """
        if not filename or not filename.strip():
            return "unnamed_document.txt"

        # Extract only the base name (strips ../ and directory paths)
        base = Path(filename).name

        # Replace path separators and null bytes
        base = base.replace("/", "").replace("\\", "").replace("\x00", "")

        # Strip dangerous characters, allowing alphanumeric, dots, hyphens, and underscores
        cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", base)
        cleaned = re.sub(r"_+", "_", cleaned)
        cleaned = cleaned.strip("._")

        if not cleaned:
            return "unnamed_document.txt"
        return cleaned

    def validate_file_metadata(self, filename: str, file_size: int) -> str:
        """
        Validate file extension and size constraints.
        
        Returns:
            str: Normalized lower-case extension (including dot, e.g. '.txt').
        """
        if file_size <= 0:
            raise EmptyFileError("Uploaded file is empty (0 bytes).")

        if file_size > self.max_upload_size_bytes:
            limit_mb = self.max_upload_size_bytes / (1024 * 1024)
            actual_mb = file_size / (1024 * 1024)
            raise FileTooLargeError(
                f"File size ({actual_mb:.2f} MB) exceeds maximum allowed limit of {limit_mb:.2f} MB."
            )

        suffix = Path(filename).suffix.lower()
        if suffix not in self.allowed_extensions:
            allowed_str = ", ".join(self.allowed_extensions)
            raise UnsupportedFileTypeError(
                f"Unsupported file format '{suffix}'. Allowed formats: {allowed_str}"
            )

        return suffix

    def extract_text_from_bytes(self, raw_bytes: bytes, extension: str) -> str:
        """
        Extract clean textual content from raw file bytes.
        
        Args:
            raw_bytes: Raw bytes from the uploaded file.
            extension: Lower-case file extension (e.g. '.txt', '.md', '.pdf').
            
        Returns:
            str: Extracted and normalized text content.
        """
        if not raw_bytes:
            raise EmptyFileError("Uploaded file contains zero bytes.")

        if extension in (".txt", ".md"):
            # Check for null bytes indicating non-text binary disguised as .txt/.md
            if b"\x00" in raw_bytes:
                raise MalformedDocumentError(
                    "File appears to contain binary data instead of valid plain text or markdown."
                )

            # Try UTF-8 first (standard)
            try:
                text = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                # Fallback to UTF-8 with BOM or Latin-1
                try:
                    text = raw_bytes.decode("utf-8-sig")
                except UnicodeDecodeError:
                    try:
                        text = raw_bytes.decode("latin-1")
                    except Exception as e:
                        raise MalformedDocumentError(f"Failed to decode text file: {e}") from e

        elif extension == ".pdf":
            try:
                import pypdf
            except ImportError as e:
                raise UnsupportedFileTypeError(
                    "PDF parsing requires 'pypdf' package to be installed."
                ) from e

            try:
                reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
            except Exception as e:
                raise MalformedDocumentError(f"Failed to parse PDF structure: {e}") from e

            if reader.is_encrypted:
                raise MalformedDocumentError(
                    "Encrypted or password-protected PDF files are not supported."
                )

            pages_text: List[str] = []
            for page_idx, page in enumerate(reader.pages):
                try:
                    page_content = page.extract_text() or ""
                    if page_content.strip():
                        pages_text.append(page_content.strip())
                except Exception as e:
                    logger.warning(f"Error extracting text from PDF page {page_idx}: {e}")

            text = "\n\n".join(pages_text)
        else:
            raise UnsupportedFileTypeError(f"Unsupported file extension: {extension}")

        cleaned = text.strip()
        if not cleaned:
            raise EmptyFileError(
                f"Document contains no extractable text content (format: {extension})."
            )

        return cleaned

    @staticmethod
    def generate_document_id(content: str) -> str:
        """
        Generate a deterministic, collision-resistant document ID based on content hash.
        Format: DOC-UPL-<sha256_prefix_12>
        """
        norm_content = "\n".join(line.strip() for line in content.splitlines() if line.strip())
        content_hash = hashlib.sha256(norm_content.encode("utf-8")).hexdigest()[:12]
        return f"DOC-UPL-{content_hash}"

    def check_duplicate(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Check if a document with this ID already exists in the database.
        
        Returns:
            Optional[Dict[str, Any]]: Existing document metadata if found, None otherwise.
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, title, source, created_at FROM documents WHERE id = ?;",
                (doc_id,),
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None

    def upload_document(
        self,
        raw_bytes: bytes,
        original_filename: str,
        overwrite: bool = False,
    ) -> UploadResult:
        """
        End-to-end document upload workflow:
        1. Validates extension and file size.
        2. Sanitizes filename.
        3. Extracts text content.
        4. Generates stable document ID.
        5. Checks for duplicate upload (unless overwrite=True).
        6. Persists raw file safely to disk.
        7. Chunks content and inserts into SQLite (with FTS5 synchronization).
        
        Args:
            raw_bytes: Raw bytes of uploaded file.
            original_filename: User-provided filename.
            overwrite: If True, replaces existing document with matching ID.
            
        Returns:
            UploadResult: Ingestion outcome.
        """
        safe_name = self.sanitize_filename(original_filename)
        extension = self.validate_file_metadata(safe_name, len(raw_bytes))

        # Extract text content
        text_content = self.extract_text_from_bytes(raw_bytes, extension)

        # Generate stable document ID
        doc_id = self.generate_document_id(text_content)

        # Duplicate check
        existing = self.check_duplicate(doc_id)
        if existing and not overwrite:
            raise DuplicateDocumentError(
                f"Duplicate document detected: an identical document already exists with ID '{doc_id}' "
                f"(title: '{existing['title']}').",
                existing_doc_id=doc_id,
            )

        # Store raw file safely in dedicated upload directory
        disk_filename = f"{doc_id}_{safe_name}"
        safe_path = (self.upload_dir / disk_filename).resolve()

        # Ensure safe_path is strictly within upload_dir (prevent traversal)
        if not str(safe_path).startswith(str(self.upload_dir.resolve())):
            raise DocumentManagementError("Security error: path traversal attempt detected.")

        with open(safe_path, "wb") as f:
            f.write(raw_bytes)

        # Chunk text
        chunks = chunk_text(
            text=text_content,
            chunk_size=80,
            chunk_overlap=20,
        )

        # Ingest into SQLite transactionally
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            # 1. Upsert document
            cursor.execute(
                """
                INSERT INTO documents (id, title, source, content)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    source = excluded.source,
                    content = excluded.content;
                """,
                (doc_id, safe_name, f"upload:{safe_name}", text_content),
            )

            # 2. Clear old chunks for idempotency / overwrite
            cursor.execute("DELETE FROM document_chunks WHERE document_id = ?;", (doc_id,))

            # 3. Insert chunks
            for chunk in chunks:
                chunk_id = f"{doc_id}_chk_{chunk.chunk_index:02d}"
                cursor.execute(
                    """
                    INSERT INTO document_chunks (
                        id, document_id, chunk_index, content, token_count
                    ) VALUES (?, ?, ?, ?, ?);
                    """,
                    (
                        chunk_id,
                        doc_id,
                        chunk.chunk_index,
                        chunk.content,
                        chunk.token_count,
                    ),
                )

        logger.info(
            f"[UPLOAD] Successfully ingested document '{doc_id}' ({safe_name}) with {len(chunks)} chunks."
        )

        return UploadResult(
            document_id=doc_id,
            filename=safe_name,
            chunks_created=len(chunks),
            content_length=len(text_content),
            is_duplicate=bool(existing),
            storage_path=str(safe_path),
        )

    def list_documents(self) -> List[DocumentMetadata]:
        """
        List all documents currently in the database, ordered by creation date descending.
        
        Returns:
            List[DocumentMetadata]: Metadata for each document.
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT 
                    d.id,
                    d.title,
                    d.source,
                    COALESCE(d.created_at, CURRENT_TIMESTAMP) as created_at,
                    COUNT(c.id) AS chunk_count,
                    LENGTH(d.content) AS content_length
                FROM documents d
                LEFT JOIN document_chunks c ON d.id = c.document_id
                GROUP BY d.id
                ORDER BY d.created_at DESC, d.id ASC;
                """
            )
            rows = cursor.fetchall()

        results: List[DocumentMetadata] = []
        for r in rows:
            doc_id = str(r["id"])
            # Core benchmark sample docs follow naming convention DOC-IT-XXX or DOC-ACAD-XXX
            is_sample = not doc_id.startswith("DOC-UPL-")

            results.append(
                DocumentMetadata(
                    id=doc_id,
                    title=str(r["title"]),
                    source=str(r["source"]) if r["source"] else None,
                    created_at=str(r["created_at"]),
                    chunk_count=int(r["chunk_count"]),
                    content_length=int(r["content_length"]),
                    is_sample=is_sample,
                )
            )

        return results

    def get_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Retrieve a single document by ID, including its chunks.
        
        Args:
            doc_id: Target document ID.
            
        Returns:
            Dict[str, Any]: Document metadata and chunk details.
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, title, source, content, created_at
                FROM documents WHERE id = ?;
                """,
                (doc_id,),
            )
            doc_row = cursor.fetchone()
            if not doc_row:
                raise DocumentNotFoundError(f"Document with ID '{doc_id}' not found.")

            cursor.execute(
                """
                SELECT id, chunk_index, content, token_count
                FROM document_chunks
                WHERE document_id = ?
                ORDER BY chunk_index ASC;
                """,
                (doc_id,),
            )
            chunk_rows = cursor.fetchall()

        chunks = [
            {
                "chunk_id": str(c["id"]),
                "chunk_index": int(c["chunk_index"]),
                "content": str(c["content"]),
                "token_count": c["token_count"],
            }
            for c in chunk_rows
        ]

        is_sample = not doc_id.startswith("DOC-UPL-")

        return {
            "id": str(doc_row["id"]),
            "title": str(doc_row["title"]),
            "source": str(doc_row["source"]) if doc_row["source"] else None,
            "created_at": str(doc_row["created_at"]),
            "is_sample": is_sample,
            "chunk_count": len(chunks),
            "content": str(doc_row["content"]),
            "chunks": chunks,
        }

    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """
        Safely delete a document and its cascading dependencies:
        - Rejects deletion if document is a core benchmark sample document.
        - Deletes from `documents` (SQLite cascades to `document_chunks`).
        - SQLite triggers automatically remove from `fts_chunks`.
        - Deletes physical uploaded file on disk if it exists.
        
        Args:
            doc_id: Document ID to delete.
            
        Returns:
            Dict[str, Any]: Deletion summary.
        """
        clean_id = doc_id.strip()
        if not clean_id:
            raise DocumentNotFoundError("Document ID cannot be empty.")

        # Protection: Prevent deletion of core benchmark sample documents
        if not clean_id.startswith("DOC-UPL-"):
            raise ProtectedDocumentError(
                f"Cannot delete core benchmark sample document '{clean_id}'. "
                "Core sample documents are protected to maintain benchmark reproducibility."
            )

        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()

            # Check if document exists
            cursor.execute("SELECT id, title FROM documents WHERE id = ?;", (clean_id,))
            doc_row = cursor.fetchone()
            if not doc_row:
                raise DocumentNotFoundError(f"Document with ID '{clean_id}' does not exist.")

            title = str(doc_row["title"])

            # Count chunks for reporting
            cursor.execute(
                "SELECT COUNT(*) AS cnt FROM document_chunks WHERE document_id = ?;",
                (clean_id,),
            )
            chunk_cnt = cursor.fetchone()["cnt"]

            # Delete from documents table (foreign key cascades to document_chunks)
            cursor.execute("DELETE FROM documents WHERE id = ?;", (clean_id,))

        # Delete physical file from uploads folder if present
        deleted_file = False
        try:
            for p in self.upload_dir.glob(f"{clean_id}_*"):
                if p.is_file():
                    p.unlink(missing_ok=True)
                    deleted_file = True
        except Exception as e:
            logger.warning(f"Error removing physical file for '{clean_id}': {e}")

        logger.info(
            f"[DELETE] Deleted document '{clean_id}' ({title}), {chunk_cnt} chunks removed."
        )

        return {
            "document_id": clean_id,
            "title": title,
            "deleted_chunks": chunk_cnt,
            "physical_file_deleted": deleted_file,
            "status": "deleted",
        }
