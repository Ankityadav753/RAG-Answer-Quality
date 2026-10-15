"""
Local launcher for DAI-009: RAG Answer Quality Browser Application.

Usage:
    python run_app.py
    python run_app.py --port 8080 --host 0.0.0.0
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import os
import uvicorn


def main() -> int:
    default_host = os.environ.get("RAG_APP_HOST", "127.0.0.1")
    default_port = int(os.environ.get("RAG_APP_PORT", "8000"))

    parser = argparse.ArgumentParser(
        description="Run DAI-009 RAG Answer Quality Web Application.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        type=str,
        default=default_host,
        help="Host IP address to bind server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=default_port,
        help="TCP port to listen on.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code modifications.",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("  DAI-009: RAG ANSWER QUALITY BROWSER APPLICATION")
    print("=" * 70)
    print(f"  Server starting at: http://{args.host}:{args.port}/")
    print(f"  API Docs available: http://{args.host}:{args.port}/docs")
    print(f"  Health endpoint   : http://{args.host}:{args.port}/health")
    print("=" * 70)

    uvicorn.run(
        "src.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
