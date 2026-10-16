"""
Unified RAG Service for DAI-009: RAG Answer Quality Benchmark.

Connects document ingestion, SQLite database storage, full-text / dense retrievers,
answer generation guardrails, and source attribution into an end-to-end single-question workflow.
"""

from __future__ import annotations

import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Ensure project root is available in sys.path
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from db.database import DEFAULT_DB_PATH, get_db_connection, init_db
from src.config import Settings, get_settings
from src.document_manager import DocumentManager, DocumentMetadata, UploadResult
from src.embeddings import BaseEmbeddingModel, get_embedding_model
from src.generator import AnswerGenerator, GeneratedAnswer, get_llm_provider
from src.ingest import DEFAULT_DOCS_PATH, ingest_documents
from src.retriever import BaseRetriever, DenseRetriever, LexicalRetriever, RetrievalResult



@dataclass(frozen=True)
class RAGResponse:
    """Structured response container for unified RAG query execution."""
    question: str
    answer: str
    is_abstention: bool
    retrieved_chunks: List[RetrievalResult]
    source_chunk_ids: List[str]
    source_doc_ids: List[str]
    provider: str
    retriever_name: str
    top_k: int
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary, serializing chunk objects."""
        data = asdict(self)
        data["retrieved_chunks"] = [c.to_dict() for c in self.retrieved_chunks]
        return data


class RAGService:
    """
    Unified end-to-end RAG service orchestrating document ingestion,
    retrieval, generation, and source attribution.
    """

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        docs_path: Optional[Union[str, Path]] = None,
        upload_dir: Optional[Union[str, Path]] = None,
        retriever_type: Optional[str] = None,
        retriever: Optional[BaseRetriever] = None,
        generator: Optional[AnswerGenerator] = None,
        embedding_model: Optional[BaseEmbeddingModel] = None,
        top_k: Optional[int] = None,
        auto_init: bool = True,
    ) -> None:
        settings: Settings = get_settings()

        self.db_path = Path(db_path).resolve() if db_path else settings.db_path
        self.docs_path = Path(docs_path).resolve() if docs_path else settings.docs_path
        self.upload_dir = Path(upload_dir).resolve() if upload_dir else settings.upload_dir
        self.top_k = top_k if top_k is not None else settings.top_k

        # 1. Initialize or configure Embedding Model (for dense retrieval)
        self.embedding_model = embedding_model or get_embedding_model(
            provider=settings.embedding_provider,
            model_name=settings.openai_embedding_model if settings.embedding_provider == "openai" else None,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            dimension=settings.openai_embedding_dim,
        )

        # 2. Initialize or configure Retriever
        self._dense_retriever: Optional[DenseRetriever] = None
        self._lexical_retriever: Optional[LexicalRetriever] = None

        if retriever is not None:
            self.retriever = retriever
            self.retriever_type = retriever.__class__.__name__
            if isinstance(retriever, DenseRetriever):
                self._dense_retriever = retriever
            elif isinstance(retriever, LexicalRetriever):
                self._lexical_retriever = retriever
        else:
            r_type = (retriever_type or settings.retriever_type).strip().lower()
            self.retriever_type = r_type
            if r_type == "dense":
                self.retriever = DenseRetriever(
                    db_path=self.db_path,
                    embedding_model=self.embedding_model,
                )
                self._dense_retriever = self.retriever
            else:
                self.retriever = LexicalRetriever(db_path=self.db_path)
                self._lexical_retriever = self.retriever

        # 3. Initialize or configure Generator
        if generator is not None:
            self.generator = generator
        else:
            llm_prov = get_llm_provider(
                provider=settings.llm_provider,
                model=settings.openai_model if settings.llm_provider == "openai" else None,
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
            )
            self.generator = AnswerGenerator(provider=llm_prov)

        # 4. Document Manager (lazy-initialized)
        self._document_manager: Optional[DocumentManager] = None

        # 5. Auto-initialize database and ingest default corpus if empty
        if auto_init:
            self.ensure_initialized()

    @property
    def is_mock(self) -> bool:
        """Return True if using deterministic mock LLM provider."""
        from src.generator import DeterministicMockLLMProvider
        return isinstance(self.generator.provider, DeterministicMockLLMProvider)

    @property
    def document_manager(self) -> DocumentManager:
        """Lazy-initialized DocumentManager instance bound to the service database."""
        if self._document_manager is None:
            self._document_manager = DocumentManager(
                db_path=self.db_path,
                upload_dir=self.upload_dir,
            )
        return self._document_manager

    def invalidate_indices(self) -> None:
        """
        Invalidate in-memory retriever chunk caches so subsequent queries
        immediately see newly uploaded or deleted documents.
        """
        if self._dense_retriever and hasattr(self._dense_retriever, "invalidate_index"):
            self._dense_retriever.invalidate_index()
        if isinstance(self.retriever, DenseRetriever):
            self.retriever.invalidate_index()

    def upload_document(
        self,
        raw_bytes: bytes,
        filename: str,
        overwrite: bool = False,
    ) -> UploadResult:
        """Upload and ingest a document, immediately updating retrieval indexes."""
        res = self.document_manager.upload_document(
            raw_bytes=raw_bytes,
            original_filename=filename,
            overwrite=overwrite,
        )
        self.invalidate_indices()
        return res

    def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """Safely delete a document, cascading to chunks and updating retrieval indexes."""
        res = self.document_manager.delete_document(doc_id)
        self.invalidate_indices()
        return res

    def list_documents(self) -> List[DocumentMetadata]:
        """List all documents currently in the knowledge base."""
        return self.document_manager.list_documents()

    def get_document(self, doc_id: str) -> Dict[str, Any]:
        """Retrieve details and chunks for a single document."""
        return self.document_manager.get_document(doc_id)


    @classmethod
    def from_env(cls, **overrides: Any) -> RAGService:
        """Create a RAGService configured from current environment variables."""
        return cls(**overrides)

    def get_status(self) -> Dict[str, Any]:
        """Return system status and component summary."""
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) AS cnt FROM document_chunks;")
            chunks_cnt = cursor.fetchone()["cnt"]
            cursor.execute("SELECT COUNT(*) AS cnt FROM documents;")
            docs_cnt = cursor.fetchone()["cnt"]

        return {
            "status": "healthy",
            "is_offline": self.is_mock,
            "llm_provider": self.generator.provider.provider_name,
            "embedding_provider": self.embedding_model.model_name,
            "default_retriever": self.retriever_type,
            "documents_count": docs_cnt,
            "chunks_count": chunks_cnt,
            "database_path": str(self.db_path),
        }

    def ensure_initialized(self) -> None:
        """Ensure SQLite schema exists and has documents loaded."""
        init_db(self.db_path)
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) AS cnt FROM document_chunks;")
            count = cursor.fetchone()["cnt"]

        if count == 0 and self.docs_path.exists():
            self.ingest(docs_path=self.docs_path)

    def ingest(
        self,
        docs_path: Optional[Union[str, Path]] = None,
        force_recompute_embeddings: bool = False,
    ) -> Dict[str, Any]:
        """
        Ingest documents from JSON corpus into SQLite and update indexes.
        
        Args:
            docs_path: Optional path to JSON corpus.
            force_recompute_embeddings: If True, forces dense retriever to re-encode all chunks.
            
        Returns:
            Dict[str, Any]: Ingestion summary.
        """
        summary = ingest_documents(
            docs_path=docs_path or self.docs_path,
            db_path=self.db_path,
        )

        # If dense retriever is in use, index chunks into memory and cache in SQLite
        if isinstance(self.retriever, DenseRetriever):
            chunks_indexed = self.retriever.index_chunks(force_recompute=force_recompute_embeddings)
            summary["chunks_indexed_dense"] = chunks_indexed

        return summary

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        retriever_type: Optional[str] = None,
    ) -> RAGResponse:
        """
        Execute unified RAG query workflow:
        1. Query validation
        2. Top-k chunk retrieval
        3. Context-grounded answer generation
        4. Structured source attribution and latency tracking
        
        Args:
            question: Natural language question.
            top_k: Optional override for number of retrieved chunks.
            retriever_type: Optional override for retriever ('lexical' or 'dense').
            
        Returns:
            RAGResponse: Structured response with answer, abstention flag, chunks, and sources.
        """
        effective_top_k = top_k if top_k is not None else self.top_k
        clean_question = question.strip() if question else ""

        # Select active retriever for this query
        active_retriever = self.retriever
        if retriever_type:
            r_type = retriever_type.strip().lower()
            if r_type == "dense":
                if self._dense_retriever is None:
                    self._dense_retriever = DenseRetriever(
                        db_path=self.db_path,
                        embedding_model=self.embedding_model,
                    )
                active_retriever = self._dense_retriever
            elif r_type == "lexical":
                if self._lexical_retriever is None:
                    self._lexical_retriever = LexicalRetriever(db_path=self.db_path)
                active_retriever = self._lexical_retriever

        t0 = time.perf_counter()

        # Execute retrieval phase
        retrieved_chunks = active_retriever.retrieve(clean_question, top_k=effective_top_k)

        # Execute generation phase
        gen_answer: GeneratedAnswer = self.generator.generate(clean_question, retrieved_chunks)

        t1 = time.perf_counter()
        latency_ms = round((t1 - t0) * 1000, 2)

        retriever_name = (
            f"dense_cosine({self.embedding_model.model_name})"
            if isinstance(active_retriever, DenseRetriever)
            else "lexical_fts5"
        )

        return RAGResponse(
            question=question,
            answer=gen_answer.answer,
            is_abstention=gen_answer.is_abstention,
            retrieved_chunks=retrieved_chunks,
            source_chunk_ids=gen_answer.source_chunk_ids,
            source_doc_ids=gen_answer.source_doc_ids,
            provider=gen_answer.provider,
            retriever_name=retriever_name,
            top_k=effective_top_k,
            latency_ms=latency_ms,
        )
