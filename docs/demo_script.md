# 5–8 Minute Demo Video Plan & Presentation Script (DAI-009)

**Project**: DAI-009 RAG Answer Quality Benchmark  
**Target Duration**: 5 to 8 Minutes  
**Audience**: Technical Assessment Committee / College Grading Evaluators  
**Speaker Role**: Student Lead Developer  

> **Notice**: This document is the presentation script, recording timeline, and preparation checklist. No video has been pre-recorded yet.

---

## 1. Pre-Recording Setup & Checklist

- [ ] **Clean Environment**: Terminal positioned at project root `Rag-Answer-Quality/`.
- [ ] **Terminal Display**: Font size set to 16–18pt for crisp readability; high-contrast dark theme.
- [ ] **Database State**: Ensure database initialized and ingested via `python db/database.py` and `python src/ingest.py`.
- [ ] **Screen Resolution**: Set recording screen to 1080p (1920x1080) at 30 or 60 fps.
- [ ] **Audio Check**: Test microphone level with zero background hum.
- [ ] **Open Tabs / Windows**:
  1. Terminal (PowerShell / Command Prompt).
  2. VS Code / IDE displaying project files (`db/schema.sql`, `src/retriever.py`, `docs/benchmark_report.md`).
  3. Mermaid architecture diagram (`docs/architecture.md`).

---

## 2. Minute-by-Minute Presentation Script

### [0:00 – 1:00] Segment 1: Introduction & Problem Statement
- **What to Show on Screen**: Title slide or `README.md` at top of workspace.
- **Spoken Commentary**:
  > *"Hello! Today I am presenting my technical assessment project, **DAI-009: RAG Answer Quality Benchmark**.*  
  > *In Retrieval-Augmented Generation, retrieval failures directly cause generation hallucinations. If your retriever pulls the wrong context or if the user asks an out-of-domain question, a naive system will fabricate an answer.*  
  > *To solve and evaluate this, I built a modular benchmark comparing two distinct retrieval paradigms—Lexical Search using SQLite FTS5 BM25 versus Dense Vector Retrieval using cosine similarity. The system evaluates retrieval metrics, answer groundedness, and automated abstention safeguards against a curated ground-truth dataset."*

---

### [1:00 – 2:30] Segment 2: System Architecture & Ingestion
- **What to Show on Screen**: `docs/architecture.md` (show Mermaid diagram) and `db/schema.sql`.
- **Action**: Run test suite in terminal:
  ```powershell
  python -m unittest discover tests -v
  ```
- **Spoken Commentary**:
  > *"Let's examine the architecture. The pipeline consists of five key layers:*  
  > *First, raw university policy documents are chunked into 80-word units with 20-word overlap by `src/chunker.py`.*  
  > *Second, `src/ingest.py` stores documents and chunks idempotently into SQLite (`db/rag_bench.db`). We enforce foreign keys and use triggers to automatically synchronize our FTS5 virtual table.*  
  > *Notice that we have 51 automated unit and integration tests passing cleanly across chunking, database integrity, retrieval ranking, generator guardrails, and metric calculations."*

---

### [2:30 – 4:15] Segment 3: Live Benchmark Execution & Retrieval Comparison
- **What to Show on Screen**: Terminal executing the comparative benchmark.
- **Action**: Run the benchmark script:
  ```powershell
  python run_benchmark.py
  ```
- **Spoken Commentary**:
  > *"Now let's run the comparative benchmark across all 22 questions.*  
  > *[Wait 2 seconds while benchmark prints]*  
  > *Here are the actual measured results on screen:*  
  > *On the 19 answerable policy questions:*  
  > *- **Configuration A (Lexical FTS5 BM25)** achieved an **MRR of 0.9737** and a perfect **Recall@3 of 1.0000**.*  
  > *- **Configuration B (Dense Vector Retrieval)** achieved an **MRR of 0.4561** and a **Recall@3 of 0.5789**.*  
  > *Why this massive difference? In technical policy documents, queries often contain exact codes, room numbers, and identifiers like `Campus-Secure-5G` or `POLICY-AI-2026`. Lexical BM25 matches these exact tokens with near-perfect precision, whereas dense projection can disperse rank weights across related IT policies."*

---

### [4:15 – 5:30] Segment 4: Groundedness & Out-of-Domain Abstention
- **What to Show on Screen**: Interactive Python one-liner in terminal or `tests/test_generator.py`.
- **Action**: Run interactive demo:
  ```powershell
  python -c "from src.retriever import LexicalRetriever; from src.generator import AnswerGenerator; r = LexicalRetriever(); g = AnswerGenerator(); q_ans = 'What is the student Wi-Fi SSID and device limit?'; print('Answerable:', g.generate(q_ans, r.retrieve(q_ans, 2)).answer); print('---'); q_unans = 'What is the meal reimbursement rate for conferences?'; print('Unanswerable:', g.generate(q_unans, r.retrieve(q_unans, 2)).answer)"
  ```
- **Spoken Commentary**:
  > *"Now let's look at answer groundedness and safety.*  
  > *For the legitimate question regarding Wi-Fi, the system extracts the exact policy sentences citing `Campus-Secure-5G` and the 4-device limit.*  
  > *Now look at the second question: 'What is the meal reimbursement rate for conferences?' This is an out-of-domain question with no supporting evidence in our knowledge base.*  
  > *Instead of hallucinating a fake dollar amount, the system detects insufficient token evidence and outputs our canonical abstention: 'I cannot find sufficient information in the provided documentation to answer this question.' Both configurations achieved a Groundedness Score of 1.0000 by refusing ungrounded claims."*

---

### [5:30 – 6:45] Segment 5: SQL Analytical Ledger & Persistence
- **What to Show on Screen**: Terminal running an analytical SQL query against SQLite.
- **Action**: Query the database:
  ```powershell
  python -c "import sqlite3; conn = sqlite3.connect('db/rag_bench.db'); conn.row_factory = sqlite3.Row; c = conn.cursor(); c.execute('SELECT config_name, COUNT(e.id) as queries, ROUND(AVG(e.precision_at_k), 4) as avg_p, ROUND(AVG(e.recall_at_k), 4) as avg_r, ROUND(AVG(e.reciprocal_rank), 4) as mrr FROM benchmark_runs r JOIN run_evaluations e ON r.id = e.run_id GROUP BY config_name'); [print(dict(r)) for r in c.fetchall()]"
  ```
- **Spoken Commentary**:
  > *"Every experiment run is persisted in SQLite. Here we execute an analytical aggregation query joining `benchmark_runs` and `run_evaluations`.*  
  > *This demonstrates how Python and SQL integrate directly: SQLite is not just a document store, but an analytical ledger tracking query latencies, precision, recall, and abstention flags per run.*  
  > *Furthermore, machine-readable JSON results and formatted Markdown comparison reports are exported automatically to the `docs/` folder."*

---

### [6:45 – 7:30] Segment 6: Disclosures, Limitations & Conclusion
- **What to Show on Screen**: `docs/design_decisions.md` (show section on Disclosures and Limitations).
- **Spoken Commentary**:
  > *"To ensure academic integrity, I want to highlight three key disclosures:*  
  > *1. The default generator is an offline rule-based test double, not a paid neural LLM.*  
  > *2. Groundedness is computed using transparent lexical token containment.*  
  > *3. The dense retriever uses a deterministic 64-dimensional seeded projection matrix so tests run without downloading heavy PyTorch weights.*  
  > *In summary, this project delivers a complete, testable, and mathematically verified benchmark harness proving how lexical and dense retrieval trade off across domain policies.*  
  > *Thank you, and I look forward to your questions!"*

---

## 3. Key Q&A Prep for Technical Interview

- **Q: Why did Lexical BM25 outperform Dense Vector search in your benchmark?**  
  *A: The corpus features explicit identifiers and codes (e.g., `Campus-Secure-5G`, `GR-APP-01`, `180 consecutive days`, `15-minute grace period`). Lexical inverted indexing has zero semantic ambiguity with exact tokens. Dense retrieval shines when users use colloquial synonyms, but requires larger dimensional models or fine-tuning to disambiguate dense technical codes.*
- **Q: How is your database ingestion idempotent?**  
  *A: `src/ingest.py` uses `INSERT INTO documents ... ON CONFLICT(id) DO UPDATE` and clears prior chunks for that document before re-inserting. Running the script multiple times produces the exact same 11 documents and 22 chunks.*
- **Q: How does the system prevent hallucinations on out-of-domain queries?**  
  *A: The generator calculates the token overlap between question terms and retrieved context sentences. If overlap falls below the evidence threshold, it immediately triggers abstention with empty citations.*
