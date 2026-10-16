# Render Deployment Guide: DAI-009 RAG Answer Quality

This guide details how to deploy the **DAI-009 RAG Answer Quality** web application to [Render](https://render.com) using either the declarative **Render Blueprint (`render.yaml`)** or manual configuration through the **Render Dashboard**.

---

## 1. Quick Specifications Summary

| Setting | Value / Details |
| :--- | :--- |
| **Service Type** | Web Service |
| **Environment / Runtime** | Python 3 (`PYTHON_VERSION=3.11.9`) |
| **Build Command** | `pip install -r requirements.txt` |
| **Production Start Command** | `uvicorn src.app:app --host 0.0.0.0 --port $PORT` |
| **Alternative Start Command**| `python run_app.py` *(automatically detects `$PORT` and binds to `0.0.0.0`)* |
| **Health Check Path** | `/health` (returns HTTP 200 with readiness & document counts) |
| **Default Mode** | `RAG_MODE=offline` (deterministic, zero external API keys required) |
| **Default Retriever** | `RAG_RETRIEVER_TYPE=lexical` (SQLite FTS5 BM25) |
| **Blueprint File** | [`render.yaml`](../render.yaml) |

---

## 2. Storage Persistence Limitations on Render Free Plan

> [!WARNING]
> **Ephemeral Storage Limitation on Free Tier**:
> Render's **Free plan** runs on ephemeral containers that shut down (spin down) after 15 minutes of inactivity and reset the filesystem on every redeploy or restart.
> 
> - **Uploaded Documents**: Any documents uploaded via the UI or `/api/documents/upload` stored in `data/uploads/` will be **lost** upon container restart or redeploy.
> - **SQLite Database Mutations**: Any evaluations, benchmark runs, or new documents added to the SQLite database will be **reset** to the baseline state.
> - **Baseline Resilience**: The default 11 university policy documents in [`data/sample_documents.json`](../data/sample_documents.json) are tracked in Git. The application automatically re-initializes and re-ingests this baseline corpus on boot whenever the database is empty or fresh. The application will always be operational and capable of answering questions out of the box.

### Configuring Durable Storage (Paid Plan Upgrade)
To achieve durable data persistence across redeploys:
1. Upgrade the Render service to a paid plan (e.g., **Starter** at $7/month).
2. Attach a **Render Persistent Disk** (e.g., 1 GB mounted at `/var/data`).
3. Set the following environment variables:
   - `RAG_DB_PATH=/var/data/rag_bench.db`
   - `RAG_UPLOAD_DIR=/var/data/uploads`
   - `RAG_DOCS_PATH=data/sample_documents.json`

Both the SQLite database and all uploaded files will then persist indefinitely across deploys and restarts.

---

## 3. Deployment Option A: Declarative Blueprint (`render.yaml`)

The repository includes a ready-to-use [`render.yaml`](../render.yaml) blueprint specification:

```yaml
services:
  - type: web
    name: rag-answer-quality
    runtime: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn src.app:app --host 0.0.0.0 --port $PORT
    healthCheckPath: /health
    autoDeploy: false
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.9
      - key: RAG_MODE
        value: offline
      - key: RAG_RETRIEVER_TYPE
        value: lexical
      - key: RAG_LLM_PROVIDER
        value: mock
      - key: RAG_EMBEDDING_PROVIDER
        value: mock
      - key: RAG_TOP_K
        value: "3"
```

### Steps to Deploy via Blueprint:
1. Push your repository to GitHub or GitLab.
2. In the [Render Dashboard](https://dashboard.render.com), click **New +** -> **Blueprint**.
3. Connect your repository.
4. Render will detect `render.yaml` and configure the Web Service automatically.
5. Click **Apply** to trigger the build and launch the application.

---

## 4. Deployment Option B: Manual Web Service Setup

If configuring manually via the Render web dashboard:

1. In Render Dashboard, click **New +** -> **Web Service**.
2. Connect your Git repository.
3. Configure the following fields:
   - **Name**: `rag-answer-quality`
   - **Language / Runtime**: `Python 3`
   - **Branch**: `main` (or active branch)
   - **Region**: Closest region (e.g., Oregon, Frankfurt, Singapore)
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn src.app:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: `Free`
4. Expand **Advanced Settings**:
   - **Health Check Path**: `/health`
   - **Auto-Deploy**: `Yes` or `No` (per preference)
5. Add **Environment Variables**:
   - `PYTHON_VERSION`: `3.11.9`
   - `RAG_MODE`: `offline`
   - `RAG_RETRIEVER_TYPE`: `lexical`
   - `RAG_LLM_PROVIDER`: `mock`
   - `RAG_EMBEDDING_PROVIDER`: `mock`
   - `RAG_TOP_K`: `3`
6. Click **Create Web Service**.

---

## 5. Connecting Real LLM Providers (Optional Online Mode)

The default configuration runs 100% offline with zero external dependencies. If you wish to connect real AI providers on Render:

1. In Render Dashboard, navigate to **Environment** under your Web Service.
2. Update the environment variables:
   - `RAG_MODE`: `online`
   - `RAG_LLM_PROVIDER`: `openai`
   - `RAG_EMBEDDING_PROVIDER`: `openai`
   - `OPENAI_API_KEY`: `sk-proj-...` *(Never commit to git; add only in Render Dashboard)*
   - `OPENAI_MODEL`: `gpt-4o-mini` *(or preferred model)*
   - `OPENAI_EMBEDDING_MODEL`: `text-embedding-3-small`

---

## 6. Pre-flight Verification Checklist

Before pushing or triggering deployment:
- [x] Production start command verified (`uvicorn src.app:app --host 0.0.0.0 --port $PORT` and `python run_app.py`)
- [x] Host binding set to `0.0.0.0` and listening on Render's `$PORT`
- [x] Health check endpoint verified at `/health` and `/api/health`
- [x] Requirements install cleanly with wheels for Linux x86_64
- [x] Zero secrets required to boot the application
- [x] Sample documents auto-ingestion verified on fresh SQLite databases
- [x] All 125 automated tests passing (`pytest`)
