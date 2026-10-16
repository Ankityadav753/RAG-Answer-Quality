"""
FastAPI application backend for DAI-009: RAG Answer Quality Browser Interface.

Provides:
1. GET /health & GET /api/health - Health check and system readiness metadata.
2. GET /api/info - Public system configuration and active provider status (never exposing secrets).
3. POST /api/query - Context-grounded question answering via RAGService.
4. Static file mounting serving the responsive browser frontend.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

# Ensure project root is available in sys.path
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from src.document_manager import (
    DocumentManagementError,
    DocumentMetadata,
    DocumentNotFoundError,
    DuplicateDocumentError,
    EmptyFileError,
    FileTooLargeError,
    MalformedDocumentError,
    ProtectedDocumentError,
    UnsupportedFileTypeError,
    UploadResult,
)
from src.service import RAGResponse, RAGService

# Directories
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
STATIC_DIR: Path = PROJECT_ROOT / "static"

logger = logging.getLogger("rag_answer_quality")
logging.basicConfig(level=logging.INFO)

# Global service instance managed across app lifecycle
rag_service: Optional[RAGService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager initializing RAGService on startup."""
    global rag_service
    logger.info("[STARTUP] Initializing unified RAGService...")
    rag_service = RAGService.from_env(auto_init=True)
    logger.info(f"[STARTUP] RAGService initialized. Provider: {rag_service.generator.provider.provider_name}")
    yield
    logger.info("[SHUTDOWN] Cleaning up application resources.")


app = FastAPI(
    title="DAI-009 RAG Answer Quality Service",
    description="Context-grounded question answering and benchmark inspection system.",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for local development and browser clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Request & Response Models
# =============================================================================

class QueryRequest(BaseModel):
    """Incoming user query request payload."""
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The natural language question to ask the RAG knowledge base.",
        examples=["What is the primary WPA3 wireless network SSID for students?"],
    )
    retriever: Optional[Literal["lexical", "dense"]] = Field(
        "lexical",
        description="Retrieval strategy: 'lexical' (SQLite FTS5 BM25) or 'dense' (Cosine similarity).",
    )
    top_k: Optional[int] = Field(
        3,
        ge=1,
        le=10,
        description="Number of ranked context chunks to retrieve (1 to 10).",
    )

    @field_validator("question")
    @classmethod
    def validate_question_non_whitespace(cls, v: str) -> str:
        """Ensure question contains actual informative text beyond whitespace."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Question cannot be empty or whitespace only.")
        return cleaned


class ChunkDTO(BaseModel):
    """DTO for a retrieved context chunk."""
    chunk_id: str
    document_id: str
    content: str
    score: float
    rank: int


class QueryResponse(BaseModel):
    """Structured response container returned to API and frontend clients."""
    question: str
    answer: str
    is_abstention: bool
    retrieved_chunks: List[ChunkDTO]
    source_chunk_ids: List[str]
    source_doc_ids: List[str]
    provider: str
    retriever_name: str
    top_k: int
    latency_ms: float
    is_mock: bool


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    is_offline: bool
    llm_provider: str
    embedding_provider: str
    default_retriever: str
    chunks_count: int
    documents_count: int
    version: str


class DocumentDTO(BaseModel):
    """Document metadata DTO."""
    id: str
    title: str
    source: Optional[str] = None
    created_at: str
    chunk_count: int
    content_length: int
    is_sample: bool


class DocumentListResponse(BaseModel):
    """List of all registered documents in the knowledge base."""
    total: int
    documents: List[DocumentDTO]


class UploadResponse(BaseModel):
    """Upload outcome response."""
    document_id: str
    filename: str
    chunks_created: int
    content_length: int
    is_duplicate: bool
    message: str


class DeleteResponse(BaseModel):
    """Document deletion response."""
    document_id: str
    title: str
    deleted_chunks: int
    status: str


# =============================================================================
# Exception Handlers
# =============================================================================

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return user-friendly validation error without internal framework leaks."""
    errors = []
    for err in exc.errors():
        field = " -> ".join(str(loc) for loc in err.get("loc", []))
        msg = err.get("msg", "Validation error")
        errors.append(f"{field}: {msg}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Invalid request parameter(s)", "errors": errors},
    )


@app.exception_handler(UnsupportedFileTypeError)
async def unsupported_file_type_handler(request: Request, exc: UnsupportedFileTypeError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc), "error_type": "UnsupportedFileTypeError"},
    )


@app.exception_handler(EmptyFileError)
async def empty_file_handler(request: Request, exc: EmptyFileError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc), "error_type": "EmptyFileError"},
    )


@app.exception_handler(MalformedDocumentError)
async def malformed_document_handler(request: Request, exc: MalformedDocumentError):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc), "error_type": "MalformedDocumentError"},
    )


@app.exception_handler(FileTooLargeError)
async def file_too_large_handler(request: Request, exc: FileTooLargeError):
    return JSONResponse(
        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        content={"detail": str(exc), "error_type": "FileTooLargeError"},
    )


@app.exception_handler(DuplicateDocumentError)
async def duplicate_document_handler(request: Request, exc: DuplicateDocumentError):
    return JSONResponse(
        status_code=status.HTTP_409_CONFLICT,
        content={
            "detail": str(exc),
            "existing_doc_id": exc.existing_doc_id,
            "error_type": "DuplicateDocumentError",
        },
    )


@app.exception_handler(DocumentNotFoundError)
async def document_not_found_handler(request: Request, exc: DocumentNotFoundError):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": str(exc), "error_type": "DocumentNotFoundError"},
    )


@app.exception_handler(ProtectedDocumentError)
async def protected_document_handler(request: Request, exc: ProtectedDocumentError):
    return JSONResponse(
        status_code=status.HTTP_403_FORBIDDEN,
        content={"detail": str(exc), "error_type": "ProtectedDocumentError"},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Mask unexpected internal exceptions to prevent credential or trace leakage."""
    logger.error(f"[ERROR] Unhandled exception during request {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal error occurred while processing the request."},
    )



# =============================================================================
# API Endpoints
# =============================================================================

def get_service() -> RAGService:
    """Helper to retrieve active RAGService instance."""
    global rag_service
    if rag_service is None:
        rag_service = RAGService.from_env(auto_init=True)
    return rag_service


@app.get("/health", response_model=HealthResponse, tags=["System"])
@app.get("/api/health", response_model=HealthResponse, tags=["System"])
def health_check() -> Dict[str, Any]:
    """Health check endpoint reporting service readiness and indexed corpus status."""
    service = get_service()
    status_data = service.get_status()
    status_data["version"] = "1.0.0"
    return status_data


@app.get("/api/info", tags=["System"])
def system_info() -> Dict[str, Any]:
    """
    Return sanitized public system configuration.
    Never exposes API keys or sensitive environment variables.
    """
    service = get_service()
    status_data = service.get_status()
    return {
        "status": status_data["status"],
        "is_offline": status_data["is_offline"],
        "llm_provider": status_data["llm_provider"],
        "embedding_provider": status_data["embedding_provider"],
        "default_retriever": status_data["default_retriever"],
        "available_retrievers": ["lexical", "dense"],
        "chunks_indexed": status_data["chunks_count"],
        "documents_count": status_data["documents_count"],
    }


@app.post("/api/query", response_model=QueryResponse, tags=["RAG"])
def query_rag(req: QueryRequest) -> QueryResponse:
    """
    Execute context-grounded RAG question answering.
    
    1. Validates input query and parameters.
    2. Retrieves top-k relevant chunks using requested retriever.
    3. Generates grounded answer and performs abstention check.
    4. Returns structured answer, citations, and ranked context chunks.
    """
    service = get_service()

    try:
        resp: RAGResponse = service.query(
            question=req.question,
            top_k=req.top_k,
            retriever_type=req.retriever,
        )

        return QueryResponse(
            question=resp.question,
            answer=resp.answer,
            is_abstention=resp.is_abstention,
            retrieved_chunks=[
                ChunkDTO(
                    chunk_id=c.chunk_id,
                    document_id=c.document_id,
                    content=c.content,
                    score=c.score,
                    rank=c.rank,
                )
                for c in resp.retrieved_chunks
            ],
            source_chunk_ids=resp.source_chunk_ids,
            source_doc_ids=resp.source_doc_ids,
            provider=resp.provider,
            retriever_name=resp.retriever_name,
            top_k=resp.top_k,
            latency_ms=resp.latency_ms,
            is_mock=service.is_mock,
        )
    except Exception as e:
        logger.error(f"[ERROR] Query processing failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred while processing query through RAG pipeline.",
        ) from e


# =============================================================================
# Document Management Endpoints
# =============================================================================

@app.get("/api/documents", response_model=DocumentListResponse, tags=["Document Management"])
def list_documents() -> DocumentListResponse:
    """List all documents registered in the RAG knowledge base."""
    service = get_service()
    docs = service.list_documents()
    dto_list = [
        DocumentDTO(
            id=d.id,
            title=d.title,
            source=d.source,
            created_at=d.created_at,
            chunk_count=d.chunk_count,
            content_length=d.content_length,
            is_sample=d.is_sample,
        )
        for d in docs
    ]
    return DocumentListResponse(total=len(dto_list), documents=dto_list)


@app.get("/api/documents/{doc_id}", tags=["Document Management"])
def get_document(doc_id: str) -> Dict[str, Any]:
    """Retrieve full details and segmented chunks for a specific document."""
    service = get_service()
    return service.get_document(doc_id)


@app.post("/api/documents/upload", response_model=UploadResponse, tags=["Document Management"])
async def upload_document(
    file: UploadFile = File(...),
    overwrite: bool = Query(False, description="Overwrite existing document if duplicate content ID found."),
) -> UploadResponse:
    """
    Upload and dynamically ingest a new document (.txt, .md, .pdf) into the RAG system.
    Immediately updates retrieval indexes for both Lexical and Dense search.
    """
    service = get_service()

    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file missing filename.")

    content = await file.read()

    res = service.upload_document(
        raw_bytes=content,
        filename=file.filename,
        overwrite=overwrite,
    )

    return UploadResponse(
        document_id=res.document_id,
        filename=res.filename,
        chunks_created=res.chunks_created,
        content_length=res.content_length,
        is_duplicate=res.is_duplicate,
        message=f"Document '{res.filename}' successfully ingested ({res.chunks_created} chunks created).",
    )


@app.delete("/api/documents/{doc_id}", response_model=DeleteResponse, tags=["Document Management"])
def delete_document(doc_id: str) -> DeleteResponse:
    """
    Safely delete an uploaded document, cascading deletion across all chunks,
    full-text search indexes, and disk storage.
    Core benchmark sample documents are protected against deletion.
    """
    service = get_service()
    del_res = service.delete_document(doc_id)
    return DeleteResponse(
        document_id=del_res["document_id"],
        title=del_res["title"],
        deleted_chunks=del_res["deleted_chunks"],
        status=del_res["status"],
    )


# =============================================================================
# Frontend Static Files Serving
# =============================================================================


# Serve static assets if static directory exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=FileResponse, tags=["Frontend"])
def serve_index() -> FileResponse:
    """Serve the single-page application frontend index.html."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(index_path, media_type="text/html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Return 204 No Content for browser favicon requests."""
    return Response(status_code=status.HTTP_204_NO_CONTENT)

