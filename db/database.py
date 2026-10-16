"""
Database layer for DAI-009: RAG Answer Quality Benchmark.

Provides safe SQLite connection handling, foreign-key enforcement,
and schema initialization routines.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional, Union

# Base directories
BASE_DIR: Path = Path(__file__).resolve().parent
DEFAULT_DB_PATH: Path = BASE_DIR / "rag_bench.db"
DEFAULT_SCHEMA_PATH: Path = BASE_DIR / "schema.sql"


def get_db_path(custom_path: Optional[Union[str, Path]] = None) -> Path:
    """
    Resolve the SQLite database path and ensure its parent directory exists.
    
    If custom_path is omitted, inspects the RAG_DB_PATH environment variable
    before falling back to DEFAULT_DB_PATH.
    
    Args:
        custom_path: Optional override for the database file path.
        
    Returns:
        Path: Resolved absolute path to the database file.
    """
    if custom_path is None:
        env_db = os.environ.get("RAG_DB_PATH")
        target_path = Path(env_db).resolve() if env_db else DEFAULT_DB_PATH
    elif isinstance(custom_path, Path):
        target_path = custom_path.resolve()
    else:
        target_path = Path(custom_path).resolve()

    # Ensure parent directory exists before creating or connecting to the DB
    target_path.parent.mkdir(parents=True, exist_ok=True)
    return target_path


def get_connection(
    db_path: Optional[Union[str, Path]] = None,
) -> sqlite3.Connection:
    """
    Open a connection to the SQLite database with safe defaults.
    
    Enforces:
    - PRAGMA foreign_keys = ON (cascading deletes and relational integrity)
    - sqlite3.Row row factory for dict-like column access
    
    Args:
        db_path: Optional path to the database file or ':memory:'.
        
    Returns:
        sqlite3.Connection: Configured database connection.
    """
    if db_path == ":memory:":
        conn = sqlite3.connect(":memory:")
    else:
        resolved_path = get_db_path(db_path)
        conn = sqlite3.connect(str(resolved_path))

    # Enable foreign keys enforcement for every connection
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_db_connection(
    db_path: Optional[Union[str, Path]] = None,
) -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager for database connections that automatically commits
    transactions on success and closes the connection upon exit.
    """
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(
    db_path: Optional[Union[str, Path]] = None,
    schema_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Initialize the SQLite database by executing the schema.sql file.
    
    Args:
        db_path: Optional custom path for the database file.
        schema_path: Optional custom path to the SQL schema file.
        
    Returns:
        Path: Path to the initialized database file.
    """
    resolved_schema = (
        Path(schema_path).resolve() if schema_path else DEFAULT_SCHEMA_PATH
    )

    if not resolved_schema.exists():
        raise FileNotFoundError(f"Schema file not found at: {resolved_schema}")

    with open(resolved_schema, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    target_db_path = get_db_path(db_path) if db_path != ":memory:" else Path(":memory:")

    with get_db_connection(db_path) as conn:
        conn.executescript(schema_sql)

    return target_db_path


def get_table_names(conn: sqlite3.Connection) -> list[str]:
    """Return a list of all user-defined table names in the database."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
    )
    return [row["name"] for row in cursor.fetchall()]


if __name__ == "__main__":
    print("[INFO] Initializing SQLite database...")
    db_file = init_db()
    print(f"[SUCCESS] Database successfully initialized at: {db_file}")

    with get_db_connection(db_file) as c:
        tables = get_table_names(c)
        print(f"[INFO] Created tables: {', '.join(tables)}")
