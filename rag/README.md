# CiteRagLab

A production-grade **Retrieval-Augmented Generation (RAG)** backend and Streamlit UI built on top of a Notion document library. CiteRagLab answers questions, compares documents, and summarises topics — grounded entirely in the documents you have ingested. It will not answer questions outside its library.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Vector store | Milvus Lite (milvus-lite 2.5.x) — in-process, no Docker |
| Embeddings | Azure OpenAI `text-embedding-3-large` (3072-dim, COSINE) |
| LLM | Azure OpenAI `gpt-4.1-mini` |
| RAG orchestration | LangChain + LangGraph |
| Document source | Notion API (database query + block extraction) |
| API backend | FastAPI (port 8001) |
| UI | Streamlit (port 8501) |
| Session cache | Redis (optional — graceful no-op if unavailable) |
| Evaluation | RAGAS (faithfulness, answer relevancy, context precision, context recall) |

---

## Project Structure

```
DOCFORGEHUB/
└── rag/
    ├── api/
    │   └── main_rag.py              # FastAPI app — all HTTP routes
    ├── evaluation/
    │   └── ragas_runner_rag.py      # RAGAS evaluation runner
    ├── ingestion/
    │   ├── notion_loader_rag.py     # Notion API client + block extractor
    │   ├── chunker_rag.py           # Token-aware overlapping chunker
    │   ├── embedder_rag.py          # Azure OpenAI batch embedder
    │   └── ingestion_pipeline_rag.py# End-to-end ingest orchestrator
    ├── pipeline/
    │   ├── adaptive_router_rag.py   # LLM intent classifier (LangGraph)
    │   ├── corrective_rag_rag.py    # Corrective retrieval loop (LangGraph)
    │   ├── pipeline_rag.py          # Main RAG pipeline entry point
    │   ├── prompts_rag.py           # All prompt templates + response constants
    │   ├── redis_cache_rag.py       # Session history + retrieval cache
    │   └── reranker_rag.py          # Top-k cap shim (pass-through)
    └── retrieval/
        ├── filters_rag.py           # Metadata filter builder
        ├── milvus_client_rag.py     # Milvus collection lifecycle + search
        └── retriever_rag.py         # Query embedder + search caller
ui/
├── cite_rag_lab_ui_rag.py           # Streamlit UI (Chat / Inspector / Ingest / Eval tabs)
└── api_helpers_rag.py               # HTTP wrappers for the FastAPI backend
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
# Notion
NOTION_API_KEY=secret_xxx
NOTION_ROOT_PAGE_ID=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx   # database ID (with or without dashes)

# Azure OpenAI — LLM
AZURE_OPENAI_LLM_KEY=xxx
AZURE_LLM_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_LLM_API_VERSION=2024-12-01-preview
AZURE_LLM_DEPLOYMENT_41_MINI=gpt-4.1-mini

# Azure OpenAI — Embeddings
AZURE_OPENAI_EMB_KEY=xxx
AZURE_EMB_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_EMB_API_VERSION=2024-12-01-preview
AZURE_EMB_DEPLOYMENT=text-embedding-3-large

# Milvus Lite (local file — no server needed)
MILVUS_URI=./rag_data/milvus.db

# Redis (optional — all operations are no-ops if Redis is unavailable)
REDIS_URL=redis://localhost:6379
```

---

## Installation

```bash
pip install fastapi uvicorn[standard] streamlit langchain langchain-openai \
            langgraph pymilvus[milvus_lite] openai notion-client \
            redis python-dotenv ragas datasets
```

---

## Running

**1. Start the FastAPI backend:**
```bash
uvicorn rag.api.main_rag:app --host 0.0.0.0 --port 8001 --reload
```

**2. Start the Streamlit UI:**
```bash
streamlit run ui/streamlit_uidemo.py --server.port 8501
```

**3. (Optional) Start Redis for session caching:**
```bash
# Ubuntu
sudo service redis start
# macOS
brew services start redis
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Run the full RAG pipeline for one message |
| `DELETE` | `/session` | Clear session history from Redis |
| `GET` | `/retrieval/debug` | Retrieval inspector — raw chunks with scores |
| `POST` | `/ingestion/notion` | Trigger Notion ingestion (full or single page) |
| `POST` | `/evaluation/run` | Run RAGAS evaluation on a supplied dataset |
| `GET` | `/health` | Liveness probe |

---

## Ingestion

**Full ingest** (all pages in the Notion database):
```bash
curl -X POST http://localhost:8001/ingestion/notion
```

**Single page ingest:**
```bash
curl -X POST http://localhost:8001/ingestion/notion \
  -H "Content-Type: application/json" \
  -d '{"page_id": "xxx", "title": "My Doc", "industry": "HR", "doc_type": "Policy", "version": "1.0"}'
```

The Notion integration must be shared on the database. In Notion: open the database → `...` → Connections → add your integration.

---

## UI Tabs

| Tab | Purpose |
|---|---|
| 💬 Chat | Ask questions, compare documents, search the library |
| 🔍 Inspector | Debug retrieval — see raw chunks and similarity scores |
| 📥 Ingest | Trigger Notion ingestion (full or single page) |
| 📊 Evaluation | Run RAGAS evaluation against a question/ground-truth dataset |

---

## Scope Enforcement

Scope is enforced by the **pipeline**, not by prompts:

1. **GREETING short-circuit** — queries classified as greetings or capability questions are intercepted before retrieval and return a hardcoded identity card instantly.
2. **Score gate** — after retrieval, if `avg_score < 0.30` (configurable via `OUT_OF_SCOPE_SCORE_THRESHOLD` in `prompts_rag.py`), the topic is not in the library. The pipeline returns a clean "not found" message without calling the LLM.

Prompts are kept natural and helpful — the LLM focuses on producing good answers, not on policing itself.

---

## RAGAS Evaluation

From the **Evaluation** tab, enter question/ground-truth pairs (one per line, tab-separated):

```
What is the incident response SLA?	Less than 1 hour for P1 incidents
What does the access control policy require?	MFA for all privileged accounts
```

The UI calls `/chat` for each question to generate answers and retrieve context, then sends everything to `/evaluation/run`. Results show faithfulness, answer relevancy, context precision, and context recall.

---

## Tuning

| Parameter | Location | Effect |
|---|---|---|
| `OUT_OF_SCOPE_SCORE_THRESHOLD` | `prompts_rag.py` | Raise to be stricter about off-topic queries |
| `RELEVANCE_THRESHOLD` | `corrective_rag_rag.py` | Score below which the query is rewritten |
| `TARGET_TOKENS` / `MAX_TOKENS` | `chunker_rag.py` | Chunk size at ingestion time |
| `OVERLAP_TOKENS` | `chunker_rag.py` | Context carried between adjacent chunks |
| `top_k` per mode | `adaptive_router_rag.py` `_STRATEGY_MAP` | Chunks retrieved per query mode |