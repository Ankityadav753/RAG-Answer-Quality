# DAI-009: RAG Answer Quality Benchmark

A modular, reproducible benchmark harness and unified engine to evaluate retrieval quality and answer groundedness across multiple Retrieval-Augmented Generation (RAG) configurations using Python and SQL.

---

## 1. Problem Statement

Retrieval-Augmented Generation (RAG) grounds language models in external knowledge sources. However, real-world RAG systems face critical failure modes:
1. **Retrieval Misses**: The retriever fails to pull the correct document chunk, causing the generator to hallucinate or provide incorrect answers.
2. **Lexical vs. Semantic Mismatches**: Inverted index keyword search fails on synonyms and paraphrasing, while dense vector retrieval often fails on exact alphanumeric codes, course identifiers, or policy limits.
3. **Ungrounded Hallucinations on Out-of-Domain Queries**: When asked questions about topics absent from the corpus, naive systems fabricate plausibly sounding answers instead of admitting lack of evidence.
4. **Embedding Cache Invalidation & Dimension Mismatch**: Changing embedding models across runs often causes dimension mismatches or corrupts vector indices if model identities are not verified.

This project implements a reproducible benchmark comparing **Configuration A (Lexical FTS5 BM25)** against **Configuration B (Dense Vector Cosine Similarity)** on a curated university policy knowledge base, evaluating retrieval quality, answer groundedness, and automated abstention safeguards.

In **Phase 1**, the engine was modernized with a unified `RAGService`, configurable real model providers (OpenAI API-compatible LLM and embeddings without PyTorch binary dependencies for Python 3.14 compatibility), robust embedding cache safety, sentence overlap prevention, and an interactive CLI.

In **Phase 2**, a responsive **browser-based RAG application** was added on top of `RAGService`, powered by a high-performance **FastAPI** backend and served via a single CLI command (`python run_app.py`). It features real-time model provider detection, dynamic retriever switching, grounded citation tags, ranked context chunk exploration, and visual abstention guardrail indicators.

In **Phase 3**, **document upload, dynamic ingestion, and document management** were added:
- Upload and parse `.txt`, `.md`, and `.pdf` files (using lightweight pure-Python `pypdf`, 100% compatible with Python 3.14).
- Safe file persistence in `data/uploads/` with path traversal defense, file size limits (5 MB default), and binary null-byte rejection.
- Stable, content-hashed document IDs (`DOC-UPL-<hash>`) with automated duplicate detection (409 Conflict) and optional overwrite.
- Immediate queryability across both Lexical (FTS5 BM25) and Dense (Cosine Similarity) retrieval without service restarts.
- Document management API and browser UI for listing, inspecting, and safely deleting documents with cascading chunk and FTS cleanup.
- Permanent protection for core benchmark sample documents (`DOC-IT-XXX`, `DOC-ACAD-XXX`) to ensure benchmark baselines are never compromised.

---

## 2. Technology Stack

- **Core Programming**: Python 3.10+ (tested on Python 3.14 on Windows)
- **Web Backend**: **FastAPI** with **Uvicorn** ASGI server and Pydantic validation
- **Document Processing**: Pure-Python **`pypdf`** (zero C++ or PyTorch dependencies) and **`python-multipart`**
- **Frontend UI**: Vanilla HTML5, modern CSS3 (custom properties, dark mode glassmorphism, responsive flex/grid, modal dialogs), and ES6+ JavaScript (zero external CDN or build tools required)
- **Database & Indexing**: SQLite 3 with native **FTS5** (Full-Text Search) and relational triggers
- **Vector Operations**: NumPy (matrix multiplication and L2 normalization)
- **Unified Engine**: `RAGService` connecting database, retrievers, generator, and source citations
- **Testing Framework**: `pytest`, `unittest`, and `httpx` / Starlette `TestClient` (125 automated tests)
- **Data Serialization**: JSON & standard Markdown reports
- **Deployment & Cloud**: Production-ready for **Render** (via `render.yaml` or Dockerless Python web service)

---

## 3. Project Structure

```text
Rag-Answer-Quality/
├── data/
│   ├── sample_documents.json    # 11 synthetic university IT & academic policy documents
│   ├── ground_truth_qa.json     # 22 benchmark questions with gold document mappings
│   └── uploads/                 # Safe directory for uploaded user documents (.txt, .md, .pdf)
├── db/
│   ├── schema.sql               # SQLite schema (documents, chunks, FTS5 table, runs, evaluations)
│   ├── database.py              # Database connection manager & schema initializer
│   └── rag_bench.db             # Local SQLite database (created at runtime)
├── src/
│   ├── app.py                   # FastAPI REST API (/health, /api/query, /api/documents)
│   ├── service.py               # Unified RAGService & RAGResponse container
│   ├── document_manager.py      # Upload validation, text extraction, safe persistence & deletion
│   ├── cli.py                   # Command-line query interface with text & JSON output
│   ├── config.py                # Environment configuration & .env management
│   ├── chunker.py               # Deterministic text chunker with word-boundary preservation
│   ├── ingest.py                # Idempotent document loading, chunking, and database ingestion
│   ├── embeddings.py            # BaseEmbeddingModel, DeterministicTestDouble, OpenAIEmbeddingModel
│   ├── retriever.py             # LexicalRetriever (Config A) & DenseRetriever with cache safety
│   ├── generator.py             # Context-grounded AnswerGenerator with sentence deduplication
│   ├── evaluator.py             # Evaluation metrics (MRR, Recall@K, Precision@K, Groundedness, F1)
│   ├── benchmark_runner.py      # Comparative benchmark pipeline and reporting engine
│   └── validate_data.py         # Dataset schema and reference validation script
├── static/
│   ├── index.html               # Responsive web UI layout (Query & Document Management tabs)
│   ├── app.css                  # Modern stylesheet with animations, tokens, and modal styles
│   └── app.js                   # Client-side controller for async queries, uploads & deletion
├── tests/
│   ├── test_deployment.py       # Render deployment readiness, PORT, 0.0.0.0 binding & env paths (9 tests)
│   ├── test_document_management.py # Document upload, validation, parsing & deletion tests (19 tests)
│   ├── test_api.py              # Web API & UI tests: endpoints, validation, abstention, static files (14 tests)
│   ├── test_rag_service.py      # Integration tests for unified RAGService (5 tests)
│   ├── test_cli.py              # CLI tests: text, json, abstention, error handling (6 tests)
│   ├── test_provider_config.py  # Provider configuration & API adapter tests (7 tests)
│   ├── test_cache_safety.py     # Embedding cache safety & model mismatch tests (6 tests)
│   ├── test_tie_handling.py     # Benchmark reporting tie-breaker tests (5 tests)
│   ├── test_database.py         # Smoke tests for database layer (5 tests)
│   ├── test_dataset.py          # Data integrity & validation tests (8 tests)
│   ├── test_chunker.py          # Unit tests for text chunking & boundaries (6 tests)
│   ├── test_ingest.py           # Ingestion pipeline & idempotency tests (4 tests)
│   ├── test_retrieval.py        # Retrieval tests: Lexical, Dense, caching, errors (11 tests)
│   ├── test_generator.py        # Generation tests: Grounding, deduplication, citations (11 tests)
│   └── test_evaluator.py        # Evaluation tests: IR metrics, groundedness, persistence (9 tests)
├── docs/
│   ├── render_deployment.md     # Production Render deployment guide and persistence notes
│   ├── architecture.md          # Complete system architecture and Mermaid diagram
│   ├── design_decisions.md      # Key design choices, trade-offs, and disclosures
│   ├── demo_script.md           # Presentation script and demo checklist
│   ├── benchmark_report.md      # Auto-generated comparative benchmark markdown report
│   └── benchmark_results.json   # Machine-readable per-query evaluation metrics
├── .env.example                 # Environment variable template
├── render.yaml                  # Declarative Render Blueprint specification
├── ask.py                       # Root CLI convenience entrypoint for asking questions
├── run_app.py                   # One-command launcher for the browser web application
├── run_benchmark.py             # CLI runner executing comparative benchmark
├── pytest.ini                   # Pytest configuration
├── requirements.txt             # Project dependencies (FastAPI, Uvicorn, NumPy, HTTPX, PyPDF)
└── README.md                    # Main setup and documentation guide

```

---

## 4. Setup & Quickstart

### 1. Prerequisites
- **Python**: Version 3.10 or higher (including Python 3.14).
- **Operating System**: Windows, macOS, or Linux.

### 2. Set Up a Virtual Environment

In Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

*(On macOS/Linux: `source .venv/bin/activate`)*

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Configure Environment (Optional)
By default, the system runs in **Offline Mode** requiring zero configuration or API keys. To configure external providers or custom paths:
```powershell
cp .env.example .env
```

---

---

## 5. Running the Browser-Based Web Application (`run_app.py`)

Phase 2 provides an interactive, modern browser application built with **FastAPI** and responsive vanilla CSS/JS.

### Launch with One Command:
```powershell
python run_app.py
```

Optional CLI flags:
```powershell
python run_app.py --port 8080 --host 0.0.0.0 --reload
```

Once started, navigate to:
- **Web UI**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **Interactive Swagger API Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check Endpoint**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### Web Interface Capabilities:
1. **Dual-Tab Interface**: Seamlessly switch between **Ask & Query** and **Document Knowledge Base** tabs.
2. **Model & Corpus Health Badges**: Header dynamically displays the active provider (`Offline Deterministic` vs `Online Real Provider`), server health, document count, and total indexed chunks.
3. **Interactive Sample Chips**: Clickable sample questions for both answerable policy queries and out-of-domain unanswerable queries (e.g., campus pool hours).
4. **Retrieval Controls**: Select between **Lexical (FTS5 BM25)** and **Dense Vector (Cosine Similarity)** retrieval, and adjust **Top-K** context chunks (1 to 10).
5. **Answer Grounding & Abstention Indicator**: High-contrast badge clearly denotes whether the response is **Evidence-Grounded** or **Refused / Out-of-Domain** with automated abstention.
6. **Attribution & Citations**: Direct visual mapping of cited documents (e.g., `DOC-IT-001`, `DOC-UPL-a1b2c3d4`) and chunk IDs.
7. **Ranked Context Inspector**: Collapsible cards displaying every retrieved chunk with its rank, similarity/BM25 relevance score, parent document, chunk identifier, and snippet text.
8. **Document Upload Dropzone**: Drag-and-drop or select `.txt`, `.md`, and `.pdf` files up to 5 MB. Newly uploaded content is chunked and immediately searchable across both retrieval modes.
9. **Duplicate Ingestion Prevention**: Uploading duplicate content is rejected with a 409 Conflict notification (or overwritten via the overwrite checkbox).
10. **Document Library & Deletion**: Interactive table displaying all indexed documents, chunk counts, and upload timestamps. Uploaded documents can be safely deleted (cascading to chunks and search indexes).
11. **Benchmark Baseline Protection**: Core benchmark sample documents (`DOC-IT-XXX`, `DOC-ACAD-XXX`) are permanently protected against deletion to maintain benchmark reproducibility.
12. **Safe Error Handling**: Input validation (Pydantic 422), upload size enforcement (413), and safe generic 500 error messages that never leak server paths or environment API keys.

### Document Management & REST API Endpoints:
- `GET /health` / `GET /api/health`: Basic and detailed service health.
- `GET /api/info`: System metadata including provider identity, database status, document and chunk counts.
- `POST /api/query`: Execute RAG query with JSON payload.
- `GET /api/documents`: List all registered documents with chunk counts, creation timestamps, and sample/upload tags.
- `GET /api/documents/{doc_id}`: Retrieve detailed document content and segmented chunks.
- `POST /api/documents/upload`: Upload and ingest a `.txt`, `.md`, or `.pdf` file (`multipart/form-data`, optional query `?overwrite=true`).
- `DELETE /api/documents/{doc_id}`: Safely delete an uploaded document and cascade chunk/index deletion. Core baseline documents return 403 Forbidden.

> [!NOTE]
> **Trusted Local Application**: This application is intended for trusted local development and exploration. Uploaded files are stored in `data/uploads/` and indexed in the local SQLite database. Authentication and multi-user tenancy are out of scope for this phase.

---

## 6. Asking Questions via CLI (`ask.py`)

Ask arbitrary questions using either Lexical (FTS5 BM25) or Dense (Cosine similarity) retrieval:

### Answerable Question Example
```powershell
python ask.py -q "What is the primary WPA3 wireless network SSID for students?"
```

Output:
```text
========================================================================
  DAI-009 RAG ANSWER QUALITY QUERY RESULT
========================================================================
Question  : What is the primary WPA3 wireless network SSID for students?
Retriever : lexical_fts5 (top_k=3)
Provider  : mock-deterministic-test-double
Latency   : 1.60 ms
------------------------------------------------------------------------
ANSWER:
Students and faculty must connect to the primary SSID named 'Campus-Secure-5G', which utilizes WPA3-Enterprise security with 802.1X authentication.
------------------------------------------------------------------------
STATUS & CITATIONS:
  Abstention Triggered : NO (Answer Generated)
  Source Documents     : ['DOC-IT-001']
  Source Chunks        : ['DOC-IT-001_chk_00', 'DOC-IT-001_chk_01']
------------------------------------------------------------------------
RETRIEVED CONTEXT CHUNKS (3 total):
  [Rank 1 | Score: 11.1555 | Doc: DOC-IT-001 | Chunk: DOC-IT-001_chk_00]
  The university provides secure high-speed wireless connectivity...
========================================================================
```

### Unanswerable Question (Automated Abstention Guardrail)
```powershell
python ask.py -q "What are the recreational swimming pool lane hours and lap reservation rules?"
```

Output:
```text
========================================================================
  DAI-009 RAG ANSWER QUALITY QUERY RESULT
========================================================================
Question  : What are the recreational swimming pool lane hours and lap reservation rules?
Retriever : lexical_fts5 (top_k=3)
Provider  : mock-deterministic-test-double
Latency   : 1.62 ms
------------------------------------------------------------------------
ANSWER:
I cannot find sufficient information in the provided documentation to answer this question.
------------------------------------------------------------------------
STATUS & CITATIONS:
  Abstention Triggered : YES (Refused - No Evidence)
  Source Documents     : [None]
  Source Chunks        : [None]
========================================================================
```

### Machine-Readable JSON Output
```powershell
python ask.py -q "What is the late penalty?" --format json
```

### Using Dense Retrieval
```powershell
python ask.py -q "Tell me about research computing cluster GPU access" --retriever dense
```

---

## 7. Model Providers: Offline vs. Online Mode

### Offline Mode (Default)
- **Zero API keys required**: Completely reproducible, offline execution.
- **Generator**: `DeterministicMockLLMProvider` extracts grounded factual sentences matching question tokens from retrieved context chunks, preventing ungrounded hallucinations.
- **Embeddings**: `DeterministicTestDoubleEmbeddingModel` generates reproducible 64-dimensional dense vectors using seeded n-gram hashing and projection ($1024 \rightarrow 64$). Cosine similarity reflects semantic and token affinity without downloading external weights.
- **Duplicate Prevention**: Sentences extracted from overlapping chunks are checked for substring containment and semantic token overlap before being included in generated answers.

### Online Mode (Real LLM and Embedding Providers)
- Configure via `.env` or environment variables:
  ```bash
  RAG_MODE=online
  RAG_LLM_PROVIDER=openai
  RAG_EMBEDDING_PROVIDER=openai
  OPENAI_API_KEY=sk-...
  OPENAI_MODEL=gpt-4o-mini
  OPENAI_EMBEDDING_MODEL=text-embedding-3-small
  ```
- **Python 3.14 Compatibility**: `OpenAIEmbeddingModel` and `OpenAICompatibleProvider` use standard library `urllib.request` over HTTPS. They require zero compiled C/C++ or PyTorch dependencies, avoiding Python 3.14 wheel incompatibility.
- **Embedding Cache Safety**: Embeddings stored in SQLite (`document_chunks.embedding_json`) record `{"model": ..., "dim": ..., "vector": [...]}`. If the active embedding model changes (e.g. from 64-dim test-double to 1536-dim OpenAI), `DenseRetriever` detects the mismatch, rejects stale cache entries, and safely recomputes without dimension-mismatch crashes.

---

## 8. Running the Comparative Benchmark

To execute the comparative benchmark comparing Lexical vs. Dense retrieval across all 22 ground-truth questions:

```powershell
python run_benchmark.py
```

### Measured Benchmark Results (Top-K = 3)

| Metric Category | Metric | Configuration A: Lexical (SQLite FTS5 BM25) | Configuration B: Dense Vector (Cosine Similarity) | Performance Winner |
| :--- | :--- | :---: | :---: | :---: |
| **Retrieval Quality** *(Answerable N=19)* | **MRR (Mean Reciprocal Rank)** | **0.9737** | **0.4561** | **Lexical** |
| | **Recall@3** | **1.0000** | **0.5789** | **Lexical** |
| | **Precision@3** | **0.5790** | **0.2105** | **Lexical** |
| **Retrieval Quality** *(All Queries N=22)* | **MRR (Overall)** | **0.8409** | **0.3939** | **Lexical** |
| | **Precision@3 (Overall)** | **0.5000** | **0.1818** | **Lexical** |
| **Answer Quality** | **Groundedness Score** *(Heuristic)* | **1.0000** | **1.0000** | **Tie** |
| **Abstention Guardrails** | **Abstention Precision** | **1.0000** | **0.3333** | **Lexical** |
| | **Abstention Recall** | **0.6667** | **0.6667** | **Tie** |
| | **Abstention F1 Score** | **0.8000** | **0.4444** | **Lexical** |
| **Performance** | **Mean Latency per Query** | **1.30 ms** | **0.70 ms** | **Dense** |

*Note on Groundedness Tie-Breaking*: Equal groundedness scores (1.0000 vs. 1.0000) are accurately reported as a **Tie** rather than defaulting to Lexical.

---

## 9. Running the Automated Test Suite

Run all **125 automated tests** across all 15 test files using `pytest`:

```powershell
pytest -v
```

Or using Python's standard library `unittest`:

```powershell
python -m unittest discover -s tests -v
```

All 125 tests execute in under 5 seconds with zero external network dependencies.

---

## 10. Production Deployment on Render

The application is structured for deployment as a Web Service on [Render](https://render.com).

### Deployment Settings Summary
- **Runtime**: Python 3 (`PYTHON_VERSION=3.11.9`)
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn src.app:app --host 0.0.0.0 --port $PORT` *(or `python run_app.py`)*
- **Health Check Path**: `/health`
- **Default Mode**: `RAG_MODE=offline` (deterministic, zero external API keys required to start)
- **Declarative Blueprint**: Included at [`render.yaml`](render.yaml)

### Important Note on SQLite and Upload Persistence
Render's **free tier** uses ephemeral container filesystems:
- **Ephemeral Filesystem**: When the free instance sleeps (after 15 minutes of inactivity) or redeploys, any runtime changes written to disk—including user-uploaded files in `data/uploads/` and SQLite database records—are reset.
- **Automatic Re-initialization**: The 11 baseline policy documents in `data/sample_documents.json` are committed to Git. The application automatically re-initializes and re-ingests this sample corpus on startup whenever the database is fresh, ensuring the application remains functional immediately upon waking up.
- **Durable Storage Option**: To persist uploaded documents and SQLite state across redeploys, upgrade to a paid Render plan (e.g. Starter tier) with a **Render Persistent Disk** mounted at `/var/data`, and configure:
  - `RAG_DB_PATH=/var/data/rag_bench.db`
  - `RAG_UPLOAD_DIR=/var/data/uploads`

For detailed step-by-step deployment instructions, see [Render Deployment Guide](docs/render_deployment.md).



