# DAI-009: RAG Answer Quality Benchmark

A modular, reproducible benchmark harness to evaluate retrieval quality and answer groundedness across multiple Retrieval-Augmented Generation (RAG) configurations using Python and SQL.

---

## 1. Problem Statement

Retrieval-Augmented Generation (RAG) grounds language models in external knowledge sources. However, real-world RAG systems face critical failure modes:
1. **Retrieval Misses**: The retriever fails to pull the correct document chunk, causing the generator to hallucinate or provide incorrect answers.
2. **Lexical vs. Semantic Mismatches**: Inverted index keyword search fails on synonyms and paraphrasing, while dense vector retrieval often fails on exact alphanumeric codes, course identifiers, or policy limits.
3. **Ungrounded Hallucinations on Out-of-Domain Queries**: When asked questions about topics absent from the corpus, naive systems fabricate plausibly sounding answers instead of admitting lack of evidence.

This project implements a reproducible benchmark comparing **Configuration A (Lexical FTS5 BM25)** against **Configuration B (Dense Vector Cosine Similarity)** on a curated university policy knowledge base, evaluating retrieval quality, answer groundedness, and automated abstention safeguards.

---

## 2. Technology Stack

- **Core Programming**: Python 3.10+ (tested on Python 3.14 on Windows)
- **Database & Indexing**: SQLite 3 with native **FTS5** (Full-Text Search) and relational triggers
- **Vector Operations**: NumPy (matrix multiplication and L2 normalization)
- **Testing Framework**: Python standard library `unittest` (compatible with `pytest`)
- **Data Serialization**: JSON & standard Markdown reports

---

## 3. Project Structure

```text
Rag-Answer-Quality/
├── data/
│   ├── sample_documents.json    # 11 synthetic university IT & academic policy documents
│   └── ground_truth_qa.json     # 22 benchmark questions with gold document mappings
├── db/
│   ├── schema.sql               # SQLite schema (documents, chunks, FTS5 table, runs, evaluations)
│   └── database.py              # Database connection manager & schema initializer
├── src/
│   ├── chunker.py               # Deterministic text chunker with word-boundary preservation
│   ├── ingest.py                # Idempotent document loading, chunking, and database ingestion
│   ├── embeddings.py            # BaseEmbeddingModel, DeterministicTestDouble, SentenceTransformer
│   ├── retriever.py             # LexicalRetriever (Config A) & DenseRetriever (Config B)
│   ├── generator.py             # Context-grounded AnswerGenerator and abstention guardrails
│   ├── evaluator.py             # Evaluation metrics (MRR, Recall@K, Precision@K, Groundedness, F1)
│   ├── benchmark_runner.py      # Comparative benchmark pipeline and reporting engine
│   └── validate_data.py         # Dataset schema and reference validation script
├── tests/
│   ├── test_database.py         # Smoke tests for database layer (5 tests)
│   ├── test_dataset.py          # Data integrity & validation tests (8 tests)
│   ├── test_chunker.py          # Unit tests for text chunking & boundaries (6 tests)
│   ├── test_ingest.py           # Ingestion pipeline & idempotency tests (4 tests)
│   ├── test_retrieval.py        # Retrieval tests: Lexical, Dense, caching, errors (11 tests)
│   ├── test_generator.py        # Generation tests: Grounding, abstention, citations (8 tests)
│   └── test_evaluator.py        # Evaluation tests: IR metrics, groundedness, persistence (9 tests)
├── docs/
│   ├── architecture.md          # Complete system architecture and Mermaid diagram
│   ├── design_decisions.md      # Key design choices, trade-offs, and disclosures
│   ├── demo_script.md           # 5–8 minute presentation script and demo checklist
│   ├── benchmark_report.md      # Auto-generated comparative benchmark markdown report
│   └── benchmark_results.json   # Machine-readable per-query evaluation metrics
├── run_benchmark.py             # CLI runner executing comparative benchmark
├── requirements.txt             # Project dependencies
└── README.md                    # Main setup and documentation guide
```

---

## 4. Setup & Quickstart

### 1. Prerequisites
- **Python**: Version 3.10 or higher.
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

### 4. Initialize Database & Ingest Documents
```powershell
# Initialize SQLite tables and FTS5 virtual table
python db/database.py

# Ingest and chunk documents idempotently
python src/ingest.py
```

Expected output:
```text
[INFO] Starting document ingestion pipeline...
[SUCCESS] Ingestion completed:
  - Database Path      : ...\db\rag_bench.db
  - Documents Ingested : 11
  - Chunks Ingested    : 22
  - Chunk Size / Overlap: 80 / 20 words
```

---

## 5. Running the Comparative Benchmark

To run the automated benchmark comparing Lexical vs. Dense retrieval across all 22 questions:

```powershell
python run_benchmark.py
```

### Measured Benchmark Results (Top-K = 3)

| Evaluation Category | Metric | Configuration A: Lexical (SQLite FTS5 BM25) | Configuration B: Dense Vector (Cosine Similarity) | Performance Winner |
| :--- | :--- | :---: | :---: | :---: |
| **Retrieval Quality** *(Answerable N=19)* | **MRR (Mean Reciprocal Rank)** | **0.9737** | **0.4561** | **Lexical (+0.5176)** |
| | **Recall@3** | **1.0000** | **0.5789** | **Lexical (+0.4211)** |
| | **Precision@3** | **0.5790** | **0.2105** | **Lexical (+0.3685)** |
| **Retrieval Quality** *(All Queries N=22)* | **Overall MRR** | **0.8409** | **0.3939** | **Lexical** |
| | **Overall Precision@3** | **0.5000** | **0.1818** | **Lexical** |
| **Generation Groundedness** | **Groundedness Score** *(Heuristic)* | **1.0000** | **1.0000** | **Tie** |
| **Abstention Safety** | **Abstention Precision** | **1.0000** | **0.3333** | **Lexical** |
| | **Abstention Recall** | **0.3333** | **0.6667** | **Dense** |
| | **Abstention F1 Score** | **0.5000** | **0.4444** | **Lexical** |
| **Execution Performance** | **Mean Latency per Query** | **1.22 ms** | **0.50 ms** | **Dense** |

*Artifacts generated automatically in `docs/`:*
- [`docs/benchmark_report.md`](file:///c:/Users/ankit/Desktop/Rag-Answer-Quality/docs/benchmark_report.md)
- [`docs/benchmark_results.json`](file:///c:/Users/ankit/Desktop/Rag-Answer-Quality/docs/benchmark_results.json)

---

## 6. Inspecting the SQLite Database & SQL Analytics

Run analytical SQL queries directly on `db/rag_bench.db`:

```powershell
python -c "import sqlite3; conn = sqlite3.connect('db/rag_bench.db'); conn.row_factory = sqlite3.Row; c = conn.cursor(); c.execute('SELECT config_name, COUNT(e.id) as query_count, ROUND(AVG(e.precision_at_k), 4) as avg_p, ROUND(AVG(e.recall_at_k), 4) as avg_r, ROUND(AVG(e.reciprocal_rank), 4) as mrr, ROUND(AVG(e.groundedness_score), 4) as avg_groundedness FROM benchmark_runs r JOIN run_evaluations e ON r.id = e.run_id GROUP BY config_name'); [print(dict(r)) for r in c.fetchall()]"
```

---

## 7. Disclosures & Academic Integrity

In compliance with rigorous evaluation practices:
1. **Default Generation Provider**: The default generator (`DeterministicMockLLMProvider`) is a deterministic, rule-based test double that extracts factual sentences directly from retrieved chunks matching question tokens. It is **not** an actual neural Large Language Model.
2. **Groundedness Metric**: Answer groundedness is measured via transparent content-token containment ($\frac{|\text{AnswerTokens} \cap \text{ContextTokens}|}{|\text{AnswerTokens}|}$). It is a heuristic proxy for faithfulness, **not** an LLM-as-a-judge or human evaluation.
3. **Dense Retrieval Implementation**: The default embedding model (`DeterministicTestDoubleEmbeddingModel`) uses a 64-dimensional seeded pseudo-random projection basis ($1024 \rightarrow 64$). This guarantees 100% deterministic, offline execution during grading without downloading heavy neural weights. A configurable adapter for `sentence-transformers` is provided in `src/embeddings.py`.

---

## 8. Assumptions & Limitations

1. **Corpus Scale**: The benchmark uses 11 curated domain policy documents (22 chunks) to keep tests transparent, inspectable, and fast for an interview or assessment.
2. **Lexical Token Containment**: Groundedness checks whether answer claims are vocabulary-contained in the retrieved text. It does not detect semantic contradictions or inversions (e.g., negations).
3. **Network Independence**: The entire suite runs with zero network connectivity.

---

## 9. Running the Automated Test Suite

Run all **51 automated tests** across all 7 test files:
```powershell
python -m unittest discover tests -v
```

All 51 tests execute in under 2 seconds with zero external network calls.
