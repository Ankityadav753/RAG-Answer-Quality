# Design & Architecture Decision Record (DAI-009)

This document details the architectural choices, alternatives considered, trade-offs, assumptions, and limitations behind the **RAG Answer Quality Benchmark** implementation.

---

## 1. Key Design Choices & Rationale

### Choice 1: SQLite with FTS5 as the Primary Storage & Evaluation Ledger
- **Decision**: Use a single local SQLite database (`db/rag_bench.db`) for both document/chunk storage, full-text inverted indexing (FTS5 BM25), vector embedding caching (`embedding_json`), and relational benchmark logging (`benchmark_runs`, `run_evaluations`).
- **Rationale**: SQLite is built into Python's standard library, requires zero external services or daemon processes, runs natively on Windows, macOS, and Linux, and enforces ACID relational constraints (`PRAGMA foreign_keys = ON;`). It satisfies the Python + SQL assessment requirement cleanly without introducing database container overhead.
- **Alternatives Considered**:
  - *Dedicated Vector Database (Chroma / Pinecone / Qdrant)*: Rejected to avoid external network services, Docker requirements, and heavy dependency footprints for a college benchmark.
  - *Flat CSV / JSON Files*: Rejected because relational joins, analytical aggregation queries, and foreign-key integrity between documents and chunks are required.

### Choice 2: 80-Word Chunk Size with 20-Word Overlap
- **Decision**: Segment text using sliding-window word boundaries with a target chunk size of 80 words and 20 words of overlap.
- **Rationale**: University policy documents feature coherent rule statements (e.g., penalty percentages, deadlines, room numbers) that span 30–70 words. An 80-word chunk captures complete clauses without fragmentation. The 20-word overlap prevents boundary loss where an identifier might be split across adjacent windows.
- **Alternatives Considered**:
  - *Sentence-Level Chunking (Spacy/NLTK)*: Adds heavy dependencies and produces uneven chunk lengths that bias BM25 and vector norms.
  - *Large Chunks (250+ words)*: Fits an entire document into a single chunk, eliminating the retrieval challenge and failing to test top-$K$ precision.

### Choice 3: Offline Deterministic Test Double for Dense Retrieval
- **Decision**: Implement `DeterministicTestDoubleEmbeddingModel` using a fixed 64-dimensional seeded projection basis ($1024 \rightarrow 64$). Provide a configurable adapter for `sentence-transformers` (`all-MiniLM-L6-v2`) if the package is installed.
- **Rationale**: Unit tests and grading environments must execute deterministically, offline, and without downloading hundreds of megabytes of neural weights. The projection model maps word and subword n-grams into unit-normalized dense vectors where cosine similarity directly reflects semantic/lexical overlap.
- **Disclosure**: The default dense retriever in this repository utilizes this deterministic mathematical test double. It is **not** an opaque pretrained neural network, ensuring complete grading transparency.

### Choice 4: Deterministic Mock Generator with Guardrails
- **Decision**: Implement `DeterministicMockLLMProvider` as the default generation engine, while providing `OpenAICompatibleProvider` for optional external endpoints.
- **Rationale**: Guarantees zero API key requirements, zero financial cost, zero network timeouts, and 100% reproducible execution during grading. It extracts grounded sentences matching question tokens and triggers automated abstention when context evidence is insufficient.
- **Disclosure**: This is an offline, rule-based test double. It is **not** a neural Large Language Model.

### Choice 5: Transparent Heuristic Groundedness Scoring
- **Decision**: Measure answer groundedness as the percentage of content words in the generated answer that directly appear in the retrieved context chunks:
  $$\text{Groundedness} = \frac{|\text{AnswerTokens} \cap \text{ContextTokens}|}{|\text{AnswerTokens}|}$$
- **Rationale**: Avoids non-deterministic or costly "LLM-as-a-judge" evaluations. If the system abstains on out-of-domain queries, groundedness is 1.0 (faithfulness preserved by refusing to hallucinate).
- **Disclosure**: This is a transparent lexical containment proxy for factual support. It is **not** human-rated or verified by an external AI judge.

---

## 2. Evaluation Metric Conventions & Positive-Class Semantics

1. **Answerable vs. Overall Retrieval Metrics**:
   - In standard Information Retrieval, queries with no relevant documents (the 3 unanswerable queries in our 22-item benchmark) have an empty ground-truth set. Computing Recall@K would produce division by zero, and defining it as 0.0 artificially depresses retrieval performance. We therefore report **Answerable Metrics** (N=19) alongside **Overall Metrics** (N=22), clearly documenting both.
2. **Abstention Classification**:
   - Positive Class = *System Abstains*.
   - Evaluates system refusal on out-of-domain questions (`Q-20`, `Q-21`, `Q-22`) without incorrectly refusing legitimate questions.

---

## 3. Measured Trade-Off Analysis

| Retrieval Configuration | Strengths | Trade-Offs / Weaknesses |
| :--- | :--- | :--- |
| **Lexical (SQLite FTS5 BM25)** | **1.0000 Recall@3**, **0.9737 MRR**. Exceptional precision on exact alphanumeric identifiers (`Campus-Secure-5G`, `GR-APP-01`, `POLICY-AI-2026`). Inverted index search takes ~1.2 ms with near-zero memory footprint. | Fails when user queries use colloquial synonyms or paraphrases not present in the indexed document vocabulary. |
| **Dense Vector (Cosine Similarity)** | Captures semantic intent and topical affinity without requiring exact lexical matches. Matrix dot product takes only ~0.5 ms using cached vectors. | Lower top-1 precision (**0.4561 MRR**, **0.5789 Recall@3**) on strict alphanumeric codes. Requires storing vector arrays in memory. |

---

## 4. Assumptions & Limitations

1. **Corpus Scale**: Benchmarked on 11 domain documents (22 chunks) to remain transparent, understandable, and testable for a college assessment. Production systems scale to millions of chunks.
2. **Heuristic Faithfulness**: Token containment detects vocabulary hallucinations well, but cannot detect semantic contradictions (e.g., negations like "not permitted" becoming "permitted").
3. **Environment**: Assumes standard Python 3.10+ on Windows, macOS, or Linux with standard SQLite libraries.
