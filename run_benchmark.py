"""
Root CLI entrypoint to execute the DAI-009 RAG Answer Quality Benchmark.

Usage:
    python run_benchmark.py
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.benchmark_runner import main

if __name__ == "__main__":
    sys.exit(main())
