"""
Convenience CLI entrypoint to query the DAI-009 RAG Answer Quality system.

Usage:
    python ask.py -q "What is the student Wi-Fi SSID?"
    python ask.py -q "What is the late assignment penalty?" --retriever dense
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.cli import main

if __name__ == "__main__":
    sys.exit(main())
