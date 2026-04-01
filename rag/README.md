# CiteRagLab + StateCase

A production-grade **Retrieval-Augmented Generation (RAG)** backend and Streamlit UI built on top of a Notion document library. 

**CiteRagLab** answers questions, compares documents, and summarises topics — grounded entirely in the documents you have ingested. Answers are never hallucinated: they must be derived from retrieved documents with explicit citations. Scope is enforced by two gates: (1) greetings bypass retrieval entirely, (2) low-scoring retrievals return "not found" without LLM calls.

**StateCase** sits on top of CiteRagLab as a tool-calling agent: when the RAG system cannot answer a question, it offers to raise support tickets in Notion with multi-turn confirmation flow. Supports bulk ticket creation ("create for all"). Tickets are deduped with three-layer checking (hash, title substring, key-term search) and sequentially numbered (atomic Redis counter). Includes a dedicated ticket board UI tab with inline editing.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Vector store | Milvus Lite (`pymilvus[milvus_lite]` 2.5.x) — in-process, no Docker |
| Embeddings | Azure OpenAI `text-embedding-3-large` (3072-dim, COSINE) |
| LLM | Azure OpenAI `gpt-4.1-mini` |
| RAG orchestration | LangChain + LangGraph (three compiled graphs) |
| Document source | Notion API (database query + recursive block extraction) |
| API backend | FastAPI on port 8001 |
| UI | Streamlit on port 8501 |
| Session & agent cache | Redis (optional — graceful no-op if unavailable) |
| Evaluation | RAGAS (faithfulness, answer relevancy, context precision, context recall) |

---

## Project Structure

```
DOCFORGEHUB/
└── rag/
    ├── api/
    │   ├── main_rag.py                  # FastAPI app — all CiteRagLab HTTP routes
    │   └── statecase_routes_rag.py      # StateCase APIRouter: /statecase/* CRUD
    ├── evaluation/
    │   └── ragas_runner_rag.py          # RAGAS evaluation with Azure OpenAI judge
    ├── ingestion/
    │   ├── notion_loader_rag.py         # Notion API client + recursive block extractor
    │   ├── chunker_rag.py               # Token-aware overlapping chunker (400/500/60 tokens)
    │   ├── embedder_rag.py              # Azure OpenAI batch embedder (batch_size=32)
    │   └── ingestion_pipeline_rag.py    # End-to-end ingest orchestrator
    ├── pipeline/
    │   ├── adaptive_router_rag.py       # LLM intent classifier (LangGraph 1-node graph)
    │   ├── corrective_rag_rag.py        # Corrective retrieval loop (LangGraph 5-node graph)
    │   ├── pipeline_rag.py              # Main RAG pipeline entry point
    │   ├── prompts_rag.py               # All prompt templates + response constants
    │   ├── redis_cache_rag.py           # Async Redis: session history + retrieval cache
    │   ├── reranker_rag.py              # Top-k cap shim (cross-encoder swap point)
    │   ├── statecase_agent_rag.py       # LangGraph tool-calling agent (5-node graph)
    │   ├── statecase_notion_rag.py      # Notion CRUD for StateCase tickets database
    │   └── statecase_tools_rag.py       # Five @tool functions bound to the agent LLM
    └── retrieval/
        ├── filters_rag.py               # Metadata filter validator + sanitiser
        ├── milvus_client_rag.py         # Milvus collection lifecycle, insert, COSINE search
        └── retriever_rag.py             # Query embedder + Milvus caller + context formatter
ui/
├── cite_rag_lab_ui_rag.py               # Streamlit UI (Chat / Tickets / Inspector / Ingest / Eval tabs)
├── api_helpers_rag.py                   # HTTP wrappers for CiteRagLab endpoints
└── api_helpers_statecase_rag.py         # HTTP wrappers for StateCase /statecase/* endpoints
```

---

## Environment Variables

Create a `.env` file in the project root:

```env
# Notion
NOTION_API_KEY=secret_xxx
NOTION_ROOT_PAGE_ID=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx   # document library database ID

# StateCase tickets database (defaults to the configured Notion DB)
STATECASE_DB_ID=32d89db1-5e5b-8051-a212-f5f983a90a0f

# Azure OpenAI — LLM (GPT-4.1-mini)
AZURE_OPENAI_LLM_KEY=xxx
AZURE_LLM_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_LLM_API_VERSION=2024-12-01-preview
AZURE_LLM_DEPLOYMENT_41_MINI=gpt-4.1-mini

# Azure OpenAI — Embeddings (text-embedding-3-large)
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
pip install fastapi uvicorn[standard] streamlit \
            langchain langchain-openai langgraph \
            pymilvus[milvus_lite] openai \
            notion-client redis python-dotenv \
            ragas datasets
```

---

## Running

**1. Start Redis** (optional but recommended for session memory and StateCase):
```bash
# Ubuntu
sudo service redis start
# macOS
brew services start redis
```

**2. Start the FastAPI backend:**
```bash
uvicorn rag.api.main_rag:app --port 8001 --reload
```

**3. Start the Streamlit UI:**
```bash
streamlit run ui/cite_rag_lab_ui_rag.py --server.port 8501
```

---

## API Endpoints

### CiteRagLab

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Run the full RAG pipeline for one user message |
| `DELETE` | `/session` | Clear session history from Redis |
| `GET` | `/retrieval/debug` | Retrieval inspector — raw chunks with similarity scores |
| `POST` | `/ingestion/notion` | Trigger Notion ingestion (full or single page) |
| `POST` | `/evaluation/run` | Run RAGAS evaluation on a supplied dataset |
| `GET` | `/health` | Liveness probe |

### StateCase

| Method | Path | Description |
|---|---|---|
| `POST` | `/statecase/chat` | Stateful tool-calling agent (RAG + auto-ticketing) |
| `POST` | `/statecase/tickets` | Create a ticket manually |
| `GET` | `/statecase/tickets` | List tickets (optional `?status=` filter) |
| `GET` | `/statecase/tickets/{id}` | Fetch one ticket by Notion page ID |
| `PATCH` | `/statecase/tickets/{id}` | Update ticket status / owner / priority |
| `GET` | `/statecase/health` | StateCase sub-service liveness probe |

---

## Ingestion

**Full ingest** — all pages in the Notion database:
```bash
curl -X POST http://localhost:8001/ingestion/notion
```

**Single page ingest:**
```bash
curl -X POST http://localhost:8001/ingestion/notion \
  -H "Content-Type: application/json" \
  -d '{
    "page_id":  "32689db1...",
    "title":    "Incident Response SLA",
    "industry": "IT",
    "doc_type": "Policy",
    "version":  "2.1"
  }'
```

> **Important:** The Notion integration must be shared on both the document library database and the StateCase tickets database. In Notion: open the database → `...` → Connections → add your integration.

---

## UI Tabs

| Tab | Purpose |
|---|---|
| 💬 Chat | Ask questions, compare documents, search the library. Toggle StateCase agent mode for auto-ticketing. |
| 🎫 Tickets | View ticket board, create tickets manually, update status/priority/owner inline. |
| 🔍 Inspector | Debug retrieval — inspect raw chunks and COSINE similarity scores from Milvus. |
| 📥 Ingest | Trigger Notion ingestion (full database or single page by ID). |
| 📊 Evaluation | Run RAGAS evaluation against a question / ground-truth dataset. |

---

## Scope Enforcement

Scope is enforced by the pipeline, not by LLM prompts — two gates prevent hallucination:

**Gate 1 — GREETING short-circuit:** Queries classified as greetings or identity questions bypass retrieval entirely and return a hardcoded response card instantly. No Azure embedding call, no Milvus query, no LLM call.

**Gate 2 — Score gate:** After retrieval, if `avg_score < 0.30` (configurable via `OUT_OF_SCOPE_SCORE_THRESHOLD` in `prompts_rag.py`), the topic is not in the library. The pipeline returns a clean "not found" message without calling the LLM. In StateCase mode, this triggers an offer to raise a support ticket.

---

## Corrective RAG

Every query goes through a two-pass retrieval loop before reaching the LLM:

1. Embed the query and retrieve top-k × 2 candidate chunks from Milvus.
2. Compute average COSINE score. If `avg_score ≥ 0.65` the results are good — proceed directly.
3. If `avg_score < 0.65`, rewrite the query using the Azure LLM. The rewrite resolves ambiguous pronouns and references using the last 6 user turns from session history.
4. Re-retrieve with the rewritten query and compare scores. Keep whichever pass scored higher.

This means a query like "what about version 2?" automatically resolves to the full explicit topic from context before searching Milvus.

---

## StateCase Confirmation Flow

When RAG cannot answer a question, the StateCase agent stores the question as `pending_ticket_context` in Redis and asks the user "Would you like me to raise a support ticket?" The ticket is only created when the user confirms (recognises "yes", "yeah", "sure", "do it", etc.). The pending context survives across HTTP requests via Redis, so the confirmation can happen in a separate turn.

Multiple unanswered questions accumulate in an `unanswered_queue` in agent memory. If a user says "create a ticket" with several items queued, the agent presents a numbered list and waits for the user to specify which one rather than creating tickets blindly.

---

## RAGAS Evaluation

From the **Evaluation** tab, enter question / ground-truth pairs (one per line, tab-separated):

```
What is the SLA for P1 incidents?	Less than 1 hour for P1 incidents
What does the access control policy require?	MFA for all privileged accounts
```

The UI calls `/chat` for each question to generate answers and retrieve context, then sends everything to `/evaluation/run`. Results show four RAGAS metrics: faithfulness, answer relevancy, context precision, and context recall — all evaluated using Azure OpenAI as the judge.

---

## Tuning Reference

| Parameter | File | Effect |
|---|---|---|
| `OUT_OF_SCOPE_SCORE_THRESHOLD` | `prompts_rag.py` | Raise to be stricter about off-topic queries (default `0.30`) |
| `RELEVANCE_THRESHOLD` | `corrective_rag_rag.py` | Score below which the query is rewritten (default `0.65`) |
| `TARGET_TOKENS` / `MAX_TOKENS` | `chunker_rag.py` | Chunk size at ingestion time (defaults 400 / 500) |
| `OVERLAP_TOKENS` | `chunker_rag.py` | Context overlap between adjacent chunks (default 60) |
| `top_k` per mode | `adaptive_router_rag.py` (`_STRATEGY_MAP`) | Chunks retrieved per query mode (QA=5, COMPARE=10, SUMMARIZE=10, SEARCH=8) |
| `BATCH_SIZE` | `embedder_rag.py` | Azure OpenAI embedding batch size (default 32) |
| `INTER_PAGE_DELAY_SEC` | `ingestion_pipeline_rag.py` | Courtesy pause between page ingests (default 0.5s) |

---

## Redis Key Reference

| Key | TTL | Purpose |
|---|---|---|
| `rag:retrieval:{sha256[:16]}` | 600s | Cached Milvus results per (query, filters) |
| `rag:session:{session_id}` | 24h | Chat turn history for multi-turn context |
| `statecase:memory:{session_id}` | 24h | Agent durable memory: pending ticket context, unanswered queue |
| `statecase:ticket_counter` | Permanent | Atomic counter for sequential SC-XXXX ticket IDs |
| `rag:notion:reads` | 60s | Notion API rate-limit counter (resets each minute) |

---

## Milvus Collection Schema

**Collection:** `notion_documents` | **Index:** AUTOINDEX · COSINE | **Dim:** 3072

| Field | Type | Notes |
|---|---|---|
| `embedding` | `FLOAT_VECTOR(3072)` | Dense vector from text-embedding-3-large |
| `chunk_text` | `VARCHAR(8192)` | Extracted and chunked page content |
| `doc_id` | `VARCHAR(256)` | Notion page UUID |
| `title` | `VARCHAR(512)` | Document title |
| `section` | `VARCHAR(256)` | Most recent heading above this chunk |
| `industry` | `VARCHAR(128)` | Filter tag (Industry select column) |
| `doc_type` | `VARCHAR(128)` | Filter tag (Type select column) |
| `version` | `VARCHAR(32)` | Filter tag (Version rich_text column) |
| `tags` | `VARCHAR(512)` | Multi-select tags, comma-joined |
| `page_id` | `VARCHAR(128)` | Notion page UUID |
| `block_range` | `VARCHAR(64)` | Block indices this chunk spans, e.g. `"12-18"` |

> **Milvus Lite note:** AUTOINDEX is the only supported index type in milvus-lite. Hybrid BM25 + HNSW search requires migrating to a full Milvus server (change `MILVUS_URI` to `http://server:19530`).