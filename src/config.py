"""
Configuration management for DAI-009 RAG Answer Quality system.

Centralizes environment variable discovery, provider settings, and offline/online mode toggles.
Adheres to 12-factor application design: no secrets are hardcoded in code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_ENV_PATH: Path = PROJECT_ROOT / ".env"
DEFAULT_DB_PATH: Path = PROJECT_ROOT / "db" / "rag_bench.db"
DEFAULT_DOCS_PATH: Path = PROJECT_ROOT / "data" / "sample_documents.json"
DEFAULT_UPLOAD_DIR: Path = PROJECT_ROOT / "data" / "uploads"
DEFAULT_MAX_UPLOAD_SIZE_BYTES: int = 5 * 1024 * 1024  # 5 MB
DEFAULT_ALLOWED_EXTENSIONS: tuple[str, ...] = (".txt", ".md", ".pdf")



def load_env_file(env_path: Optional[Path] = None) -> None:
    """
    Lightweight, dependency-free .env file loader using standard library.
    Does not overwrite existing environment variables.
    """
    target = env_path or DEFAULT_ENV_PATH
    if not target.exists():
        return

    try:
        with open(target, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception:
        # Gracefully proceed if .env is unreadable
        pass


# Auto-load .env file upon module import if present
load_env_file()


@dataclass(frozen=True)
class Settings:
    """Immutable application runtime settings."""
    rag_mode: str
    retriever_type: str
    llm_provider: str
    embedding_provider: str
    top_k: int
    db_path: Path
    docs_path: Path
    openai_api_key: Optional[str]
    openai_base_url: Optional[str]
    openai_model: str
    openai_embedding_model: str
    openai_embedding_dim: int
    upload_dir: Path = DEFAULT_UPLOAD_DIR
    max_upload_size_bytes: int = DEFAULT_MAX_UPLOAD_SIZE_BYTES
    allowed_upload_extensions: tuple[str, ...] = DEFAULT_ALLOWED_EXTENSIONS

    @property
    def is_offline(self) -> bool:
        """Return True if offline mode is active."""
        return self.rag_mode.lower() == "offline"


def get_settings() -> Settings:
    """Read current environment configuration and return a validated Settings instance."""
    mode = os.environ.get("RAG_MODE", "offline").strip().lower()
    retriever_type = os.environ.get("RAG_RETRIEVER_TYPE", "lexical").strip().lower()
    llm_prov = os.environ.get("RAG_LLM_PROVIDER", "mock" if mode == "offline" else "openai").strip().lower()
    emb_prov = os.environ.get(
        "RAG_EMBEDDING_PROVIDER", "mock" if mode == "offline" else "openai"
    ).strip().lower()

    try:
        top_k = int(os.environ.get("RAG_TOP_K", "3"))
    except ValueError:
        top_k = 3

    custom_db = os.environ.get("RAG_DB_PATH")
    db_path = Path(custom_db).resolve() if custom_db else DEFAULT_DB_PATH

    custom_docs = os.environ.get("RAG_DOCS_PATH")
    docs_path = Path(custom_docs).resolve() if custom_docs else DEFAULT_DOCS_PATH

    custom_upload = os.environ.get("RAG_UPLOAD_DIR")
    upload_dir = Path(custom_upload).resolve() if custom_upload else DEFAULT_UPLOAD_DIR

    try:
        max_upload_bytes = int(os.environ.get("RAG_MAX_UPLOAD_SIZE_BYTES", str(DEFAULT_MAX_UPLOAD_SIZE_BYTES)))
    except ValueError:
        max_upload_bytes = DEFAULT_MAX_UPLOAD_SIZE_BYTES

    openai_key = os.environ.get("OPENAI_API_KEY")
    openai_base_url = os.environ.get("OPENAI_BASE_URL")
    openai_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    openai_emb_model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    try:
        openai_emb_dim = int(os.environ.get("OPENAI_EMBEDDING_DIM", "1536"))
    except ValueError:
        openai_emb_dim = 1536

    return Settings(
        rag_mode=mode,
        retriever_type=retriever_type,
        llm_provider=llm_prov,
        embedding_provider=emb_prov,
        top_k=top_k,
        db_path=db_path,
        docs_path=docs_path,
        openai_api_key=openai_key,
        openai_base_url=openai_base_url,
        openai_model=openai_model,
        openai_embedding_model=openai_emb_model,
        openai_embedding_dim=openai_emb_dim,
        upload_dir=upload_dir,
        max_upload_size_bytes=max_upload_bytes,
        allowed_upload_extensions=DEFAULT_ALLOWED_EXTENSIONS,
    )

