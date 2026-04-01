# CiteRagLab & StateCase — Comprehensive Codebase Architecture

## Executive Summary

**CiteRagLab** is an enterprise **Retrieval-Augmented Generation (RAG)** system that lets employees query a company's Notion document library in natural language and receive cited, grounded answers. It is built on a multi-stage LangGraph pipeline with Azure OpenAI (GPT-4.1-mini for answering, text-embedding-3-large for semantic search), Milvus Lite as the local vector store, and Redis for optional session memory and retrieval caching.

**StateCase** is a **LangGraph tool-calling agent** that sits on top of CiteRagLab. When the RAG system cannot answer a question (low retrieval score), StateCase offers to raise a support ticket in a Notion database, tracks ticket lifecycle (create / update / list), and manages the yes/no confirmation flow across conversation turns using Redis-persisted memory.

Both systems share a single FastAPI backend (port 8001), an optional Redis instance, and a single Streamlit UI (port 8501).

---

### Core Capabilities

**CiteRagLab**
- **Adaptive routing**: LangGraph classifies every query into QA / COMPARE / SUMMARIZE / SEARCH / GREETING and adjusts retrieval pool size and response format accordingly. Falls back to heuristics if the LLM is unavailable.
- **Corrective RAG**: if the first retrieval pass scores below 0.65, the query is automatically rewritten using conversation history and re-retrieved; the better-scoring result is kept. Prevents weak retrievals from poisoning the answer.
- **Grounded citations**: every answer carries numbered `[N]` inline citations linking to the exact Notion page, section, document type, version, and COSINE similarity score. Answers are never hallucinated — they must be grounded in retrieved documents.
- **Score gate**: if avg COSINE score < 0.30 (configurable), the topic is out-of-scope and a clean "not found" response is returned without calling the LLM. Zero hallucination on uncovered topics.
- **Multi-turn memory**: session history stored in Redis per session (or in-memory if Redis unavailable); last 6 prior user turns injected into the LLM context for follow-up question awareness.
- **RAGAS evaluation**: built-in evaluation tab measuring faithfulness, answer relevancy, context precision, and context recall using Azure OpenAI as the judge.

**StateCase**
- **Tool-calling agent**: five `@tool`-decorated functions bound to GPT-4.1-mini via `bind_tools()`; the LLM decides which tool to call based on user intent. No hard-coded routing — the LLM orchestrates the flow.
- **Multi-turn confirmation flow**: when RAG cannot answer, the agent stores `pending_ticket_context` in Redis and asks the user "shall I raise a ticket?" For multiple unanswered questions, accumulates them in `unanswered_queue` and asks the user to pick one or create for "all". Tickets are created only on explicit confirmation.
- **Three-layer deduplication**: Layer 1 (exact hash match) prevents double-clicks. Layer 2 (title substring) catches minor rephrasing. Layer 3 (key-term search) catches semantic duplicates. Returns `is_duplicate: true/false` flag so LLM can notify user if ticket already existed.
- **Sequential ticket ID**: atomic Redis `INCR` on `statecase:ticket_counter` generates `SC-0001`, `SC-0002`, … with no race conditions.
- **Multi-ticket creation**: when user says "create tickets for all [questions]", the LLM dispatches multiple `create_support_ticket` calls in one turn. The `update_mem` node uses word-overlap scoring to match which queue items were successfully ticketed, even if the LLM rephrased the questions slightly.
- **UUID auto-resolution**: if the LLM passes a ticket title instead of a Notion page UUID when updating, `find_ticket_by_title()` resolves it automatically. Reduces friction.
- **Ticket board UI**: separate Streamlit tab with status/priority/owner inline editing, summary metrics, and a manual ticket creation form.

---

## Technology Stack

| Layer | Technology | Purpose | Key Decision |
|-------|-----------|---------|--------------|
| **LLM** | Azure OpenAI GPT-4.1-mini (`AzureChatOpenAI`) | RAG answering, adaptive routing, query rewriting, StateCase tool dispatch | `temperature=0.2` for answers; `temperature=0` for routing/classification |
| **Embeddings** | Azure OpenAI `text-embedding-3-large` | 3072-dim chunk vectors (ingestion) and query vectors (retrieval) | Same model and pre-processing for both to keep them in the same vector space |
| **Orchestration** | LangGraph (`StateGraph`, `ToolNode`) | Adaptive router graph, corrective RAG graph, StateCase agent graph | Three independent compiled graphs; each compiled once at module load |
| **LLM SDK** | `langchain-openai`, `langchain-core` | `AzureChatOpenAI`, `@tool`, `bind_tools`, message types | Native Azure function-calling support |
| **Vector Store** | Milvus Lite (`pymilvus[milvus_lite]`) | AUTOINDEX + COSINE dense vector search; runs in-process as a `.db` file | No Docker, no server; upgrade to full Milvus by changing `MILVUS_URI` |
| **REST API** | FastAPI | Async REST gateway on port 8001 | Lifespan for Redis/connection cleanup; CORS for Streamlit on 8501 |
| **Cache + Memory** | Redis (`redis.asyncio`) | Retrieval cache (10 min), session history (24 h), StateCase agent memory (24 h), ticket counter (permanent), Notion rate-limit counter (1 min) | Graceful no-op fallback if Redis is unavailable |
| **Notion (source)** | `notion-client` | Read document library for ingestion; read/write StateCase tickets database | Two separate databases; same API key |
| **Frontend** | Streamlit | Chat UI, Tickets tab, Inspector, Ingest, Evaluation | All Streamlit state in `st.session_state`; no SSR |
| **Evaluation** | RAGAS (`ragas`, `datasets`) | Automated RAG quality measurement | Configured to use Azure OpenAI instead of default OpenAI |
| **Language** | Python 3.12+ | Entire codebase | TypedDict state, async/await, `@tool` decorator |

---

## Repository Layout

```
DOCFORGEHUB/
└── rag/
    ├── api/
    │   ├── __init__.py
    │   ├── main_rag.py                  # FastAPI app on port 8001; all CiteRagLab routes;
    │   │                                # mounts StateCase sub-router; lifespan shutdown hooks
    │   └── statecase_routes_rag.py      # StateCase APIRouter: /statecase/chat, /tickets CRUD
    │
    ├── evaluation/
    │   └── ragas_runner_rag.py          # RAGAS evaluation with Azure OpenAI judge
    │
    ├── ingestion/
    │   ├── notion_loader_rag.py         # Notion API: read document library database,
    │   │                                # recursive block extraction, rate limiting
    │   ├── chunker_rag.py               # Token-aware overlapping chunker (400/500/60 tokens)
    │   ├── embedder_rag.py              # Azure batch embedder: BATCH_SIZE=32, one retry
    │   └── ingestion_pipeline_rag.py    # Orchestrates: loader → chunker → embedder → Milvus
    │
    ├── pipeline/
    │   ├── adaptive_router_rag.py       # LangGraph 1-node graph: classifies query mode
    │   ├── corrective_rag_rag.py        # LangGraph graph: retrieve → score → rewrite? → re-retrieve
    │   ├── pipeline_rag.py              # Main orchestrator: filters → route → retrieve → LLM
    │   ├── prompts_rag.py               # All LLM prompt strings: RAG, COMPARE, SUMMARIZE,
    │   │                                # REFINE_QUERY, OUT_OF_SCOPE, GREETING
    │   ├── reranker_rag.py              # Pass-through shim: enforces top_k cap (swap point for cross-encoder)
    │   ├── redis_cache_rag.py           # Async Redis: retrieval cache, session history,
    │   │                                # Notion rate limiter, graceful no-op fallback
    │   ├── statecase_agent_rag.py       # LangGraph tool-calling agent: load_mem → agent →
    │   │                                # tools → update_mem → agent loop → save_mem
    │   ├── statecase_notion_rag.py      # Notion CRUD for StateCase tickets database:
    │   │                                # create / update / get / list / find_by_title
    │   └── statecase_tools_rag.py       # Five @tool functions: rag_search, create_support_ticket,
    │                                    # update_support_ticket, list_support_tickets, retrieve_chunks
    │
    └── retrieval/
        ├── filters_rag.py               # Validates and sanitises metadata filter dicts
        ├── milvus_client_rag.py         # Milvus Lite: schema, AUTOINDEX, insert, COSINE search
        └── retriever_rag.py             # Embeds query → calls Milvus → formats [N] context block

ui/
    ├── api_helpers_rag.py               # HTTP wrappers for CiteRagLab endpoints
    ├── api_helpers_statecase_rag.py     # HTTP wrappers for StateCase /statecase/* endpoints
    ├── api_helpers.py                   # HTTP wrappers for DocForgeHub endpoints
    ├── cite_rag_lab_ui_rag.py           # Streamlit: Chat, Tickets, Inspector, Ingest, Evaluation tabs
    └── streamlit_uidemo.py              # Main app entry point (renders CiteRagLab + DocForgeHub)

rag/
    ├── rag_data/
    │   └── milvus.db                    # Milvus Lite local database file (auto-created on first ingest)
    ├── CODEBASE_ARCHITECTURE.md         # DocForgeHub architecture (separate system)
    ├── RAG_TECHNICAL_DOC.md             # This file
    └── README.md
```

---

## High-Level System Architecture

```
+─────────────────────────────────────────────────────────────────────────────+
│                        STREAMLIT UI  (Port 8501)                            │
│  ui/cite_rag_lab_ui_rag.py                                                  │
│                                                                             │
│  LEFT SIDEBAR                     MAIN AREA (tabs)                         │
│  ┌──────────────────┐  ┌──────────┬───────────┬──────────┬───────────────┐ │
│  │ 🤖 CiteRagLab    │  │💬 Chat   │🎫 Tickets │🔍 Inspec.│📥 Ingest /   │ │
│  │ + New Chat btn   │  │          │           │          │📊 Evaluation │ │
│  │ Session list     │  │ Filters  │ Board tab │ Milvus   │              │ │
│  │ Auto-ticket ON   │  │ industry │  metrics  │ raw      │ Notion       │ │
│  │ toggle           │  │ version  │  cards    │ chunks + │ ingest +     │ │
│  │ Delete session   │  │          │           │ scores   │ RAGAS eval   │ │
│  │ Session ID       │  │ Messages │ Create    │          │              │ │
│  └──────────────────┘  │ +ticket  │ ticket    │          │              │ │
│                        │ badges   │ form      │          │              │ │
│                        └──────────┴───────────┴──────────┴───────────────┘ │
│                                                                             │
│  ui/api_helpers_rag.py ──── HTTP REST ──────────────────────────────────+  │
│  ui/api_helpers_statecase_rag.py ─── HTTP REST ─────────────────────────+  │
+─────────────────────────────────────────────────────────────────────────────+
                                    │ HTTP (localhost)
                                    │ port 8001
                                    ▼
+─────────────────────────────────────────────────────────────────────────────+
│                     FASTAPI BACKEND  (Port 8001)                            │
│  rag/api/main_rag.py  +  rag/api/statecase_routes_rag.py                   │
│                                                                             │
│  POST /chat                  → run_rag_pipeline()                           │
│  DELETE /session             → Redis DEL rag:session:{id}                  │
│  GET  /retrieval/debug       → retrieve() raw inspector                     │
│  POST /ingestion/notion      → ingest_all_pages() or ingest_page()         │
│  POST /evaluation/run        → run_ragas_evaluation()                       │
│  GET  /health                                                               │
│                                                                             │
│  POST /statecase/chat        → run_statecase_agent()                        │
│  POST /statecase/tickets     → create_ticket()                              │
│  GET  /statecase/tickets     → list_tickets()                               │
│  GET  /statecase/tickets/{id}→ get_ticket()                                 │
│  PATCH /statecase/tickets/{id}→update_ticket()                              │
│  GET  /statecase/health                                                     │
+─────────────────────────────────────────────────────────────────────────────+
         │                     │                      │
         │                     │                      │
         ▼                     ▼                      ▼
+──────────────────+  +──────────────────+  +──────────────────────────────+
│   LANGGRAPH       │  │     REDIS        │  │     MILVUS LITE              │
│   PIPELINE        │  │                  │  │  rag_data/milvus.db          │
│                   │  │ rag:retrieval:*  │  │                              │
│  Adaptive Router  │  │   TTL 600s       │  │  Collection: notion_documents│
│  (1-node graph)   │  │                  │  │  AUTOINDEX + COSINE          │
│                   │  │ rag:session:{id} │  │  dim=3072 (text-emb-3-large) │
│  Corrective RAG   │  │   TTL 86400s     │  │                              │
│  (5-node graph)   │  │                  │  │  Fields:                     │
│                   │  │ statecase:memory │  │   embedding (3072 floats)    │
│  StateCase Agent  │  │   :{session_id}  │  │   chunk_text  VARCHAR 8192   │
│  (5-node graph    │  │   TTL 86400s     │  │   doc_id, title, section     │
│   + ToolNode      │  │                  │  │   industry, doc_type         │
│   + tool loop)    │  │ statecase:ticket │  │   version, tags, page_id     │
│                   │  │   _counter       │  │   block_range                │
+──────────────────+  │   permanent      │  +──────────────────────────────+
         │             │                  │              │
         │             │ rag:notion:reads  │              │
         │             │   TTL 60s         │              │
         │             +──────────────────+              │
         │                                               │
         ▼                                               ▼
+──────────────────────────────+            +──────────────────────────────+
│     AZURE OPENAI              │            │     NOTION API               │
│                               │            │                              │
│  GPT-4.1-mini                 │            │  Document Library DB         │
│  - RAG answering (temp=0.2)   │            │  (NOTION_ROOT_PAGE_ID)       │
│  - Adaptive routing (temp=0)  │            │  → ingestion source          │
│  - Query rewriting (temp=0.3) │            │                              │
│  - StateCase tool dispatch    │            │  StateCase Tickets DB        │
│  - RAGAS judge                │            │  (STATECASE_DB_ID)           │
│                               │            │  → ticket CRUD target        │
│  text-embedding-3-large       │            │                              │
│  - Chunk embedding (ingest)   │            │  API version: 2022-06-28     │
│  - Query embedding (retrieval)│            │  Rate: 0.35s/call, 6 retries │
│  - RAGAS embeddings           │            +──────────────────────────────+
+──────────────────────────────+
```

---

## Redis Key Design

| Key Pattern | Type | TTL | Written By | Read By | Purpose |
|-------------|------|-----|-----------|---------|---------|
| `rag:retrieval:{sha256[:16]}` | String (JSON) | 600s | `POST /chat` after pipeline | `POST /chat` before pipeline | Cache retrieved chunks per (query, filters). Same query in any session skips Milvus+Azure. |
| `rag:session:{session_id}` | String (JSON) | 86400s | `POST /chat` after answering | `POST /chat` before pipeline | Chat history `[{role, content}]`. Used to inject last 6 turns into LLM context and by corrective RAG rewriter to resolve references. |
| `statecase:memory:{session_id}` | String (JSON) | 86400s | `save_mem` node | `load_mem` node | StateCase durable agent memory: `last_question`, `last_ticket_id`, `pending_ticket_context`. Survives across HTTP requests. |
| `statecase:ticket_counter` | Integer | None (permanent) | `_get_next_ticket_id()` via sync Redis | Same function | Atomic counter for sequential `SC-0001`, `SC-0002`, … IDs. |
| `rag:notion:reads` | Integer | 60s | `check_notion_rate_limit()` | Same function | Count of Notion API calls in the current minute. Resets automatically via TTL. |

---

## Milvus Collection Schema

**Collection:** `notion_documents`  
**Index:** AUTOINDEX · COSINE  
**Embedding dimension:** 3072 (text-embedding-3-large)

| Field | Type | Max Length | Notes |
|-------|------|------------|-------|
| `id` | INT64 | — | Primary key, auto-generated |
| `embedding` | FLOAT_VECTOR | dim=3072 | Dense vector |
| `chunk_text` | VARCHAR | 8192 | Truncated to 8000 on insert |
| `doc_id` | VARCHAR | 256 | Notion page UUID |
| `title` | VARCHAR | 512 | Document title |
| `section` | VARCHAR | 256 | Most recent heading above this chunk |
| `industry` | VARCHAR | 128 | Industry filter tag |
| `doc_type` | VARCHAR | 128 | Document type filter tag |
| `version` | VARCHAR | 32 | Version filter tag |
| `tags` | VARCHAR | 512 | Multi-select tags, comma-joined |
| `page_id` | VARCHAR | 128 | Notion page UUID |
| `block_range` | VARCHAR | 64 | e.g. `"12-18"` |

---

## Notion Databases

### Document Library Database (`NOTION_ROOT_PAGE_ID`)
Source-only. Read during ingestion. Each **database row = one document page**. Properties extracted: `Title` (title), `Type` (select → doc_type), `Industry` (select), `Version` (rich_text), `tags` (multi_select).

### StateCase Tickets Database (`STATECASE_DB_ID`)
Read and write. Each row = one support ticket.

| Column | Notion Property Type | API Shape | Notes |
|--------|---------------------|-----------|-------|
| `Question` | **title** (primary) | `{"title": [...]}` | The page name — must use `_title_prop()` |
| `Ticket ID` | rich_text | `{"rich_text": [...]}` | e.g. `"SC-0012"` |
| `Description` | rich_text | standard | One-line summary |
| `Assigned Owner` | rich_text | standard | |
| `Priority` | **select** | `{"select": {"name": ...}}` | Low / Medium / High / Critical |
| `Status` | **status** (native) | `{"status": {"name": ...}}` | Not started / In progress / Done — NOT select type |
| `User Info` | rich_text | standard | `session_id:{id} \| dedup:{hash} \| trace_id:{id}` |
| `Attempted Sources` | rich_text | standard | Comma-joined doc titles RAG tried |

> **Critical:** `Status` uses Notion's native Status type, not Select. Using `{"select": ...}` returns HTTP 400. This applies to both writes (`create`/`update`) and filter queries (`list`).

---

## File-by-File Deep Dive

---

### `rag/ingestion/notion_loader_rag.py`

Reads the document library from Notion. All API calls go through `_notion_call()` which enforces rate limiting and retries.

**Rate limiting constants:**
```python
REQUEST_DELAY_SEC = 0.35   # 1/0.35 = 2.85 req/s, under Notion's 3 req/s hard limit
MAX_RETRIES       = 6      # exponential backoff: 2s, 4s, 8s, 16s, 32s, 64s
MAX_BACKOFF_SEC   = 64.0   # cap — prevents indefinite waiting on outage
MAX_BLOCK_DEPTH   = 5      # recursion cap — prevents infinite loops on circular references
```

**Client:** Pinned to Notion API version `"2022-06-28"` (stable `POST /databases/{id}/query` endpoint). The default SDK version moved database querying to `/data_sources` which requires extra setup.

**`_notion_call(api_fn, **kwargs)`:** Wraps every API call. Sleeps `REQUEST_DELAY_SEC` after every success. On 429: reads `Retry-After` header (uses exponential backoff if absent). Non-429 errors propagate immediately.

**`_extract_blocks_recursive(block_id, depth)`:** Recursively walks block tree. Tables: fetched as child `table_row` blocks converted to `col1 | col2 | col3` strings. Code blocks: wrapped in triple-backtick fences. Depth capped at 5 to prevent runaway recursion. Returns a **flat list** of `{heading, text, block_idx}` — nesting is resolved before the chunker.

**`get_all_pages(database_id)`:** Calls `databases.retrieve()` first to verify integration access (missing access returns zero rows silently; explicit check gives a clear error). Paginates with `next_cursor` until `has_more=False`. Returns `[{page_id, title, doc_type, industry, version, tags}]`.

**`get_page_blocks(page_id)`:** Thin wrapper around `_extract_blocks_recursive`. Returns flat list of `{heading, text, block_idx}`.

---

### `rag/ingestion/chunker_rag.py`

Splits the flat block list into overlapping chunks preserving section context.

```python
TARGET_TOKENS  = 400   # soft flush target
MAX_TOKENS     = 500   # hard cap — flush BEFORE adding a block that would exceed this
OVERLAP_TOKENS = 60    # last ~3-4 sentences seeded into next chunk for continuity
```

**Token counting:** `max(1, len(text) // 4)` — the `1 token ≈ 4 characters` OpenAI rule of thumb.

**`_flush(end_block_idx)`:** Inner function (uses `nonlocal` to mutate `buffer_lines`, `buffer_tokens`, `start_block_idx`). Emits a chunk dict with `chunk_text, doc_id, title, section, industry, doc_type, version, page_id, block_range`. Then seeds the next buffer with the last `OVERLAP_TOKENS` worth of lines (walks reversed, inserts at front to restore order).

**Special handling:**
- **Headings:** Update `current_section` on encounter; every subsequent chunk carries this section label in its metadata until the next heading
- **Table rows:** Accumulated in `table_row_group` before flushing — header row always stays with data rows in the same chunk
- **Code blocks:** Always emitted as a standalone chunk — current buffer flushed first, code block added alone and immediately flushed

**Each chunk dict:** `{chunk_text, doc_id, title, section, industry, doc_type, version, page_id, block_range}`

---

### `rag/ingestion/embedder_rag.py`

Converts chunk text to 3072-dim vectors using Azure OpenAI `text-embedding-3-large`.

```python
EMBED_MODEL = os.getenv("AZURE_EMB_DEPLOYMENT", "text-embedding-3-large")
BATCH_SIZE  = 32   # conservative; Azure allows up to 2048 per request
```

**`embed_chunks(chunks)`:** Replaces `\n` with spaces (OpenAI recommendation for consistent tokenisation). Calls `_batch_embed()`. Mutates each dict in-place adding `"embedding": list[float]`.

**`_batch_embed(texts)`:** Processes in batches of 32. Sorts results by `item.index` before collecting — Azure does not guarantee response order matches input order. One automatic retry with 2s sleep on any API error; second failure propagates.

---

### `rag/ingestion/ingestion_pipeline_rag.py`

Chains all four ingestion stages.

**`ingest_page(page_meta)`:** `get_page_blocks()` → `chunk_page()` → `embed_chunks()` → `insert_chunks()`. Returns 0 (not error) on empty pages. All four stages logged individually.

**`ingest_all_pages(database_id)`:** Calls `ingest_page()` for each database row. Per-page error isolation — one page failure does not abort the run; errors collected in `errors[]`. `INTER_PAGE_DELAY_SEC = 0.5` courtesy pause between pages to keep cumulative Notion API rate safe.

Returns: `{pages_processed, chunks_inserted, pages_skipped, errors}`.

---

### `rag/retrieval/milvus_client_rag.py`

Vector store client. Uses milvus-lite (in-process, no server needed).

**`pkg_resources` mock:** Injected at import time. milvus-lite imports `pkg_resources` only to read its own version; if `setuptools` is not installed this would crash. The mock satisfies the version-check call without requiring the package.

**`_ensure_connected()`:** `connections.connect(uri=MILVUS_URI)` called exactly once per process. Creates the data directory if needed (only for file-based URIs, not `http://`).

**`get_collection()`:** Module-level singleton `_collection`. If collection exists: loads it. If not: creates schema, creates `AUTOINDEX + COSINE` index, loads into memory. Collection must be loaded before searching.

**`_build_filter_expr(filters)`:** Converts clean filter dict to Milvus boolean expression string. Exact match for `industry`, `doc_type`, `version`. `LIKE "%value%"` for `tags` (because tags are comma-joined strings, not arrays).

**`hybrid_search_chunks(query_embedding, query_text, top_k, filters)`:** Dense COSINE search. `query_text` is accepted but unused (reserved for future BM25 hybrid search on full Milvus server). `results[0]` because Milvus returns a list-of-hit-lists for batch queries.

**`insert_chunks(chunks)`:** Column-format data (list per field). `chunk_text[:8000]` safety truncation. `collection.flush()` forces buffer to disk after insert.

---

### `rag/retrieval/filters_rag.py`

Sanitises the raw filter dict from API requests before it reaches Milvus.

**`ALLOWED_FILTER_KEYS = {"industry", "doc_type", "version", "tags"}`:** Whitelist prevents arbitrary field injection into Milvus boolean expressions. Unknown keys logged as warnings and silently dropped.

**`build_filters(raw)`:** Strips whitespace, drops empty values. `"  "` (spaces only) becomes excluded — prevents `industry == "  "` which would match nothing and confuse users.

---

### `rag/retrieval/retriever_rag.py`

Embeds the user query and retrieves matching chunks.

**`embed_text(text)`:** Single embedding call. Replaces `\n` with spaces — **must match** the pre-processing in `embedder_rag.py` exactly; different pre-processing would place query and chunk vectors in different semantic positions, making COSINE scores meaningless.

**`retrieve(query, top_k, filters)`:** Imports `hybrid_search_chunks` locally (avoids circular import; Milvus is only connected when retrieval is actually needed).

**`format_context_for_prompt(chunks)`:** Formats retrieved chunks into numbered `[N]` blocks for the LLM. Each block:
```
[1] HR Policy → Bereavement Leave  (HR Policy v1.0  tags: hr, leave  score: 0.7234)
<chunk_text>
```
The `[N]` numbers match inline citation indices in the LLM's answer. Score is included so the LLM can calibrate confidence.

---

### `rag/pipeline/adaptive_router_rag.py`

Classifies user intent into one of five modes.

**LangGraph graph:** Single node (`classify`), compiled once at module load into `_router_graph`. Single node is used rather than a plain function call to allow future composition into a larger graph without rewriting the classification logic.

**Modes and retrieval parameters:**

| Mode | `top_k` | `llm_mode` | Trigger |
|------|---------|-----------|---------|
| `QA` | 5 | `"qa"` | Direct factual question, follow-ups, conversation references |
| `COMPARE` | 10 | `"compare"` | Compare / contrast two documents or versions |
| `SUMMARIZE` | 10 | `"summarize"` | Overview or summary of a topic |
| `SEARCH` | 8 | `"qa"` | Find or list documents by topic/type |
| `GREETING` | 0 | `"qa"` | Greeting or identity question — pipeline bypassed entirely |

**Two-attempt fallback:** Primary prompt → if LLM returns unrecognised token or throws → fallback prompt (written differently to avoid same failure mode) → if both fail → default `QA`. Pipeline never crashes on classification failure.

**Classification prompt design:** Rich few-shot examples with informal phrasing, typos, and non-English patterns. Key distinction: GREETING = zero information-seeking intent (just "hi", "who are you"). QA includes follow-up questions and conversation history references ("what did I ask before?" is QA not GREETING).

---

### `rag/pipeline/corrective_rag_rag.py`

Two-pass retrieval with automatic query rewriting when the first pass is weak.

**Threshold:** `RELEVANCE_THRESHOLD = 0.65`. Below this avg COSINE score → rewrite the query.

**Graph topology:**
```
retrieve → score → route:
    score ≥ 0.65 → done   (return first-pass results unchanged)
    score < 0.65 → rewrite → retrieve2 → pick_best
```

**State fields:**
```python
class CorrectiveRAGState(TypedDict):
    query:           str
    retrieve_fn:     Callable      # injected closure — captures pool_size from pipeline
    top_k:           int
    filters:         Optional[dict]
    session_history: list[dict]    # user turns only — AI turns excluded
    chunks1:         list[dict]    # first-pass results
    score1:          float
    rewritten:       str           # LLM-rewritten query
    chunks2:         list[dict]    # second-pass results
    score2:          float
    final_chunks:    list[dict]    # winner
    final_query:     str
```

**Why `retrieve_fn` is injected:** The pipeline calls corrective RAG with `pool_size = top_k * 2` — a larger candidate pool than the final `top_k`. If corrective RAG imported `retrieve` directly, this pool size optimisation would be impossible.

**`_node_rewrite`:** Uses `REFINE_QUERY_PROMPT` which tells the LLM to:
- Resolve ambiguous pronouns/references using the last 6 **user turns only** (AI turns excluded — they carried stale topic content that caused queries to drift toward old subjects)
- Keep the rewrite as a document-retrieval query (not general knowledge)
- Return only the rewritten query — `strip('"').strip("'")` removes accidental wrapping quotes

**`_node_pick_best`:** Compares `score2` vs `score1`. Keeps the winner. If the rewrite made things worse, the original first-pass results are returned — rewrites can never degrade quality.

---

### `rag/pipeline/reranker_rag.py`

Transparent pass-through enforcing the `top_k` cap. Milvus COSINE already orders results by score, so `chunks[:top_k]` is the correct operation. The module exists as a clean swap point: replace the body of `rerank()` with a cross-encoder (e.g. `sentence-transformers/cross-encoder/ms-marco-MiniLM-L-6-v2`) without touching any other file.

---

### `rag/pipeline/prompts_rag.py`

All LLM prompt strings centralised in one file. No LLM calls, no database access, no Streamlit dependency.

**`RAG_SYSTEM_PROMPT`:** Instructs Citter to base answers only on numbered context chunks, cite every claim with `[N]`, and includes a **CRITICAL — source of truth rule**: "The numbered context chunks are the ONLY authoritative source. Ignore anything in earlier conversation turns that contradicts them." This rule was added to fix a bug where stale AI turns in session history caused the LLM to anchor on its own prior answers rather than freshly retrieved chunks.

**`COMPARE_SYSTEM_PROMPT`:** Mandates `## Similarities / ## Key Differences / ## Recommendation` structure.

**`SUMMARIZE_SYSTEM_PROMPT`:** Mandates `## Overview / ## Key Points / ## Conclusions` structure.

**`REFINE_QUERY_PROMPT`:** Used by corrective RAG rewrite node. Instructs the LLM to use conversation history to resolve references, keep the rewrite as a retrieval query, and return ONLY the rewritten query (no preamble).

**`OUT_OF_SCOPE_SCORE_THRESHOLD = 0.30`:** Avg COSINE score below this = topic not in library. Pipeline returns `OUT_OF_SCOPE_RESPONSE` without calling the LLM.

**`GREETING_RESPONSE`:** Hardcoded identity card. No LLM call, no retrieval — instant response.

---

### `rag/pipeline/redis_cache_rag.py`

Async Redis wrapper with graceful fallback.

**Client singleton:** `_get_client()` tries `aioredis.from_url(..., socket_connect_timeout=2)` + `ping()`. On failure: returns `None`. Every downstream function checks `if not client: return None/[]` — the pipeline continues without caching if Redis is down.

**Three namespaces:**

**1. Retrieval cache** (`rag:retrieval:{sha256[:16]}`):  
Cache key = `sha256(json.dumps({"q": query, "f": filters}, sort_keys=True))[:16]`. `sort_keys=True` makes the JSON canonical regardless of dict ordering. TTL 600s (10 min) — same query from any session skips Azure embedding + Milvus.

**2. Session history** (`rag:session:{session_id}`):  
`[{role: str, content: str}]` list per session. TTL 86400s (24h). Loaded by `run_rag_pipeline()` before calling the LLM; updated after each answer. Used by corrective RAG rewriter for context. Also written by StateCase agent's `save_mem` node so RAG rewrites stay consistent across mixed chat sessions.

**3. Notion rate limiter** (`rag:notion:reads`):  
Redis `INCR` (atomic) + `EXPIRE 60` (set only on first increment). Counts Notion API calls in the current minute. Returns `False` if count > 100. TTL resets the counter automatically each minute without any cleanup code.

---

### `rag/pipeline/pipeline_rag.py`

Main orchestrator — chains every component.

**`run_rag_pipeline(query, session_history, raw_filters)`:**

| Step | What happens |
|------|-------------|
| 1 | `build_filters()` — sanitise metadata filters |
| 2 | `classify_query()` — LangGraph router → mode + top_k |
| 2a | If `GREETING` → return hardcoded response immediately (no embedding, no Milvus, no LLM) |
| 3 | `corrective_retrieve()` with `pool_size = top_k * 2` |
| 4 | `rerank()` — enforce top_k cap |
| 4b | If `avg_score < 0.30` → return out-of-scope response (no LLM call) |
| 5 | `format_context_for_prompt()` — numbered `[N]` blocks |
| 6 | `AzureChatOpenAI.invoke([system, history turns, user+context])` |
| 7 | Build citations list with `chunk_text` included (for RAGAS evaluation) |

**Session history injection (step 6):** Last 6 prior turns. AI turns replaced with `"[Previous answer — see retrieved context below for current facts.]"` — preserves turn-pair structure for follow-up reference resolution while removing stale factual content that caused context contamination when users switched topics.

**User content label:** `"Retrieved context (authoritative source — answer from this only):\n{context}\n\nQuestion: {rewritten_query}"` — reinforces the system prompt's source-of-truth rule at the message level.

---

### `rag/evaluation/ragas_runner_rag.py`

RAGAS evaluation configured to use Azure OpenAI as the judge.

```python
metrics = [
    Faithfulness(llm=azure_llm),
    AnswerRelevancy(llm=azure_llm, embeddings=azure_embeddings),
    ContextPrecision(llm=azure_llm),
    ContextRecall(llm=azure_llm),
]
```

Dataset column names match RAGAS v0.2+ expectations: `user_input`, `response`, `retrieved_contexts`, `reference`. Results extracted via `result.to_pandas()` and averaged per metric.

---

### `rag/api/main_rag.py`

FastAPI application on port 8001.

**Lifespan:** Logs startup/shutdown; calls `close_rag_redis()` on shutdown to gracefully flush the async Redis connection.

**CORS:** Allows `localhost:8501` and `localhost:8502` (Streamlit).

**StateCase sub-router:** `app.include_router(statecase_router)` adds all `/statecase/*` routes without modifying any existing route.

**`POST /chat` flow:**
1. `get_session_history(session_id)` — load prior turns from Redis
2. `run_rag_pipeline(query, session_history, filters)` — full RAG
3. `set_retrieval_cache(message, filters, result["chunks"])` — cache chunks
4. Append `{user, assistant}` turns to history → `set_session_history(session_id, history)` — persist
5. Return `{answer, citations, mode, avg_score, rewritten}`

---

### `rag/pipeline/statecase_notion_rag.py`

All Notion CRUD for the StateCase tickets database.

**Property type map:** Same `_notion_call()` rate-limiting wrapper as `notion_loader_rag.py`.

**Property builders:**
```python
def _title_prop(text)     → {"title": [{"text": {"content": text[:2000]}}]}
def _rich_text_prop(text) → {"rich_text": [{"text": {"content": text[:2000]}}]}
def _select_prop(value)   → {"select": {"name": value}}
def _status_prop(value)   → {"status": {"name": value}}   # distinct from select
def _read_status(props, key) → props[key]["status"]["name"]  # distinct reader
```

**`create_ticket()`:**
1. Validate priority (default `Medium` if invalid)
2. Build `dedup = sha256(f"{session_id}::{question.strip().lower()}")[:16]`
3. Store in `User Info`: `session_id:{id} | dedup:{hash}`
4. `_find_by_dedup(dedup)` — if ticket exists, return it without creating duplicate
5. `_get_next_ticket_id()` — Redis `INCR "statecase:ticket_counter"` → `"SC-{n:04d}"`; falls back to timestamp if Redis unavailable
6. Build props with `Question` as title (not `Ticket ID`), `Status` as `_status_prop("Not started")`
7. `client.pages.create(parent={"database_id": db_id}, properties=props)`

**`list_tickets(status_filter)`:** Filter uses `{"property": "Status", "status": {"equals": ...}}` — **not** `{"select": ...}`. Notion Status type requires its own filter key.

**`find_ticket_by_title(search_term)`:** Queries `Ticket ID` (rich_text contains) first, then `Question` (title contains). Returns first match. Called by `update_support_ticket` tool when the LLM passes a human-readable name instead of a UUID.

---

### `rag/pipeline/statecase_tools_rag.py`

Five `@tool`-decorated functions. The `@tool` decorator auto-generates a Pydantic schema from the function signature and docstring, which `bind_tools()` serialises into the Azure OpenAI function-calling format. The LLM receives this schema and decides which tool to call.

**Why tools instead of direct calls:**
1. LLM chooses which action to take — no hard-coded routing
2. Azure OpenAI validates arguments against Pydantic schema before dispatch
3. Tools compose — bind to any future agent without rewriting logic

| Tool | Wraps | Key behaviour |
|------|-------|--------------|
| `rag_search` | `run_rag_pipeline` | Returns `answerable: bool` (score ≥ 0.30); agent uses this to decide on ticket offer |
| `create_support_ticket` | `create_ticket` | Docstring says: do not call without user confirmation unless explicitly requested |
| `update_support_ticket` | `update_ticket` | UUID regex check: if not a UUID, calls `find_ticket_by_title()` to resolve before PATCH |
| `list_support_tickets` | `list_tickets` | Status filter uses Notion status type (not select) |
| `retrieve_chunks` | `retrieve` | Raw retrieval for debugging; not for answer generation |

**UUID auto-resolution in `update_support_ticket`:**
```python
_UUID_RE = re.compile(r"^[0-9a-f]{8}-?[0-9a-f]{4}-?...$", re.IGNORECASE)
if not _UUID_RE.match(notion_page_id.replace("-", "")):
    ticket = find_ticket_by_title(notion_page_id)
    resolved_page_id = ticket["notion_page_id"]
```

---

### `rag/pipeline/statecase_agent_rag.py`

LangGraph tool-calling agent. Compiled graph runs the full agent loop.

**Graph topology:**
```
load_mem → agent ─── has tool_calls ──→ tools → update_mem → agent (loop)
                └─── no tool calls  ──→ save_mem → END
```

**`StateCaseAgentState`:**
```python
class StateCaseAgentState(TypedDict):
    session_id:          str
    raw_filters:         Optional[dict]
    ticket_priority:     str
    ticket_owner:        str
    messages:            Annotated[list[BaseMessage], add_messages]  # appends, not replaces
    memory:              dict     # persisted to Redis: last_question, pending_ticket_context
    trace_id:            str
    final_response:      str
    final_citations:     list
    final_pipeline_meta: dict
    final_ticket:        Optional[dict]
```

`Annotated[list[BaseMessage], add_messages]`: LangGraph's `add_messages` reducer appends new messages rather than replacing the list — critical for accumulating tool call/result pairs across loop iterations.

**`_node_load_mem`:** Reads `statecase:memory:{session_id}` from Redis. Includes `pending_ticket_context` (question + attempted sources stored when RAG failed, awaiting user yes/no).

**`_node_agent`:** Rebuilds system prompt on every iteration with current memory context — pending ticket context must be visible to the LLM when deciding whether "yes" means ticket confirmation. Calls `_get_llm_with_tools().invoke([system_msg] + messages)`. Tracks tool calls to update memory pre-emptively (e.g. clears `pending_ticket_context` when `create_support_ticket` is called).

**`_build_memory_context(memory, session_id)`:** Injects the real `session_id` into the system prompt with an explicit instruction: `"← ALWAYS use this exact value when calling create_support_ticket"`. Without this, the LLM would guess or hallucinate the session ID.

**`_node_update_memory_after_tools`:** Reads `ToolMessage` results from `messages` list. If `rag_search` returned `answerable=False` → stores `pending_ticket_context` with question, attempted sources, priority, owner. If `answerable=True` → clears stale pending context. If `create_support_ticket` succeeded → clears pending, stores `last_ticket_id`.

**`_extract_tool_result(messages, tool_name)`:** Walks messages in reverse (most recent first), finds `ToolMessage` with matching `.name`, parses JSON content.

**`_node_save_mem`:** Extracts final response from last `AIMessage` with content and no `tool_calls`. Persists memory to `statecase:memory:{session_id}` (24h TTL). Appends user + assistant turn to `rag:session:{session_id}` for cross-system multi-turn consistency.

---

### `rag/api/statecase_routes_rag.py`

`APIRouter(prefix="/statecase")` mounted on the main app.

`UpdateTicketRequest` has all optional fields (`Optional[str] = None`) — `PATCH` semantics: only provided fields are sent to Notion.

---

### `ui/cite_rag_lab_ui_rag.py`

Streamlit UI entry point for CiteRagLab + StateCase.

**Session state keys:**

| Key | Type | Purpose |
|-----|------|---------|
| `crl_sessions` | `dict[str, dict]` | All sessions: `{id: {title, messages}}` |
| `crl_active_session_id` | `str` | Currently displayed session |
| `crl_filters` | `dict` | `{industry, version}` filter values |
| `crl_filter_version_counter` | `int` | Incremented on "Clear Filters" to force widget re-creation |
| `sc_agent_mode` | `bool` | ON = StateCase agent (auto-ticketing); OFF = bare RAG |
| `sc_ticket_list` | `list` | Cached ticket list from Notion |
| `sc_ticket_list_loaded` | `bool` | Whether cache is fresh |

**Duplicate submission guard:**
```python
_submit_key = (session_id, len(messages), user_query)
if st.session_state.get("crl_last_submit_key") == _submit_key:
    return
```
Prevents double-submit on Streamlit reruns after chat input.

**Agent mode routing:**
```python
if sc_agent_mode:
    api_response = call_statecase_chat(...)   # → POST /statecase/chat
else:
    api_response = call_chat(...)             # → POST /chat
```

**Ticket board:** Real Notion status values: `"Not started"`, `"In progress"`, `"Done"`. Metrics count by exact string match. Status icons: `📬 Not started`, `🔄 In progress`, `✅ Done`. Expander labels use plain text (not markdown bold) — `st.expander()` renders its label as plain text; `**SC-0012**` appears as `****SC-0012****`.

---

## Complete Data Flows

---

### Flow 1: CiteRagLab Chat — Standard QA

```
User types "How many days of sick leave do I get?"
│
├─ cite_rag_lab_ui_rag.py: call_statecase_chat() (agent mode ON)
│  OR call_chat() (agent mode OFF)
│
├─ POST /statecase/chat (agent mode) OR POST /chat (bare RAG)
│
│  ── AGENT MODE PATH ─────────────────────────────────────────────────────
│  statecase_agent_rag.py: run_statecase_agent()
│  │
│  ├─ load_mem: GET Redis "statecase:memory:{session_id}"
│  │   → memory = {last_question: "...", ...}
│  │   → session_id injected into system prompt
│  │
│  ├─ agent: LLM sees "ask about document library" → tool_call: rag_search(query, session_id)
│  │
│  ├─ tools: ToolNode dispatches rag_search()
│  │   → (same as bare RAG path below)
│  │   → returns {answer, answerable: True, avg_score: 0.68, citations}
│  │
│  ├─ update_mem: answerable=True → clear pending_ticket_context
│  │
│  ├─ agent (2nd iteration): LLM reads rag_search result → formats answer → no tool_calls
│  │
│  └─ save_mem: persist memory to Redis, append turn to session history
│
│  ── BARE RAG PATH ───────────────────────────────────────────────────────
│  main_rag.py → POST /chat
│  ├─ get_session_history("abc") → [{role: user, content: "..."}, ...]
│  ├─ run_rag_pipeline(query, session_history, filters)
│  │   ├─ build_filters({}) → {}
│  │   ├─ classify_query() → "QA", top_k=5
│  │   ├─ corrective_retrieve(pool_size=10)
│  │   │   ├─ retrieve(): embed_text(query) → [0.12, -0.45, ...] (3072 floats)
│  │   │   │   └─ hybrid_search_chunks() → 10 chunks from Milvus
│  │   │   ├─ avg_score=0.71 ≥ 0.65 → done (no rewrite)
│  │   │   └─ returns chunks1, query unchanged
│  │   ├─ rerank() → top 5 chunks
│  │   ├─ avg_score=0.68 ≥ 0.30 → proceed
│  │   ├─ format_context_for_prompt() → "[1] HR Policy → Sick Leave ..."
│  │   ├─ AzureChatOpenAI.invoke([system, [prev user turns], user+context])
│  │   └─ answer = "Employees receive 10 days of sick leave per year [1][3]."
│  ├─ set_retrieval_cache(query, {}, chunks) → Redis SET TTL 600s
│  ├─ append {user, assistant} turns → set_session_history() → Redis SET TTL 86400s
│  └─ return {answer, citations, mode: "QA", avg_score: 0.68, rewritten: "..."}
│
└─ UI: render message bubble + citations expander + pipeline metadata
```

---

### Flow 2: StateCase — Cannot Answer → Ticket Offer → Confirm → Create

```
User: "What is our current vendor rate card?"
│
├─ agent: rag_search(query="vendor rate card", session_id="abc123")
│   └─ run_rag_pipeline() → avg_score=0.11 < 0.30 → OUT_OF_SCOPE
│   └─ returns {answerable: False, attempted_sources: ["Vendor Policy v1.0"]}
│
├─ update_mem:
│   → memory["pending_ticket_context"] = {
│       question: "What is our current vendor rate card?",
│       session_id: "abc123",
│       attempted_sources: ["Vendor Policy v1.0"],
│       priority: "Medium",
│       owner: "Unassigned"
│     }
│   → Redis SET "statecase:memory:abc123" (TTL 86400s)
│
├─ agent (2nd iteration):
│   System prompt now contains:
│     "PENDING TICKET OFFER: You previously offered to create a ticket for:
│      'What is our current vendor rate card?'. If the user says yes, call
│      create_support_ticket..."
│   LLM response: "I couldn't find the vendor rate card. Would you like me to
│                  raise a support ticket? Reply yes to create one or no to skip."
│
└─ save_mem: memory (with pending) persisted, turn appended to session history

─────────── NEXT TURN ───────────────────────────────────────────────────────────

User: "yes"
│
├─ load_mem: Redis GET "statecase:memory:abc123"
│   → memory["pending_ticket_context"] = {...}   ← restored from Redis
│
├─ agent:
│   System prompt contains pending ticket context
│   LLM sees "yes" + pending context → tool_call: create_support_ticket(
│       question="What is our current vendor rate card?",
│       session_id="abc123",   ← injected from system prompt, not guessed
│       attempted_sources="Vendor Policy v1.0",
│       priority="Medium"
│   )
│
├─ tools: ToolNode dispatches create_support_ticket()
│   → statecase_notion_rag.create_ticket()
│   │   ├─ dedup = sha256("abc123::what is our current vendor rate card")[:16]
│   │   ├─ _find_by_dedup(dedup) → None (first time)
│   │   ├─ _get_next_ticket_id(): Redis INCR "statecase:ticket_counter" → 13
│   │   │   → "SC-0013"
│   │   ├─ client.pages.create(properties={
│   │   │       "Question":          _title_prop("What is our current vendor rate card?"),
│   │   │       "Ticket ID":         _rich_text_prop("SC-0013"),
│   │   │       "Status":            _status_prop("Not started"),
│   │   │       "Priority":          _select_prop("Medium"),
│   │   │       "User Info":         _rich_text_prop("session_id:abc123 | dedup:f8b03998..."),
│   │   │       "Attempted Sources": _rich_text_prop("Vendor Policy v1.0"),
│   │   │   })
│   │   └─ returns {ticket_id: "SC-0013", notion_page_id: "uuid...", url: "..."}
│   └─ create_support_ticket tool returns {success: True, ticket_id: "SC-0013", ...}
│
├─ update_mem:
│   → memory.pop("pending_ticket_context")  ← cleared after creation
│   → memory["last_ticket_id"] = "SC-0013"
│
├─ agent (2nd iteration):
│   LLM reads tool result, formats confirmation response:
│   "✅ Support ticket SC-0013 has been raised! Priority: Medium | Status: Not started
│    [View in Notion](https://notion.so/...)"
│
└─ save_mem: updated memory (no pending) persisted
```

---

### Flow 3: Notion Document Ingestion

```
User clicks "Ingest All Pages" in the Ingest tab
│
├─ call_ingest_notion() → POST /ingestion/notion
│
├─ ingest_all_pages(database_id=NOTION_ROOT_PAGE_ID)
│   ├─ get_all_pages(db_id)
│   │   ├─ client.databases.retrieve(db_id) → verify access
│   │   └─ paginate POST /databases/{id}/query until has_more=False
│   │   → [{page_id, title, doc_type, industry, version, tags}, ...]
│   │
│   └─ for each page_meta: ingest_page(page_meta)
│       ├─ Stage 1: get_page_blocks(page_id)
│       │   └─ _extract_blocks_recursive(page_id, depth=0)
│       │   → [{heading, text, block_idx}, ...]
│       │
│       ├─ Stage 2: chunk_page(page_id, title, blocks, ...)
│       │   → [{chunk_text, doc_id, title, section, industry, ...}, ...]
│       │
│       ├─ Stage 3: embed_chunks(chunks)
│       │   → AzureOpenAI.embeddings.create(model="text-embedding-3-large",
│       │                                   input=batch_of_32_texts)
│       │   → chunks with "embedding": [float x 3072] added in-place
│       │
│       ├─ Stage 4: insert_chunks(embedded_chunks)
│       │   → collection.insert(column_format_data)
│       │   → collection.flush()  ← forces to disk
│       │
│       └─ sleep(0.5s)  ← inter-page courtesy delay
│
└─ return {pages_processed, chunks_inserted, pages_skipped, errors}
```

---

## Design Decisions

### Why LangGraph for the Router, Corrective RAG, and Agent — Three Separate Graphs?

Three graphs, each compiled once at module load:

**Adaptive Router** (1-node graph): A single `classify` node wrapping one LLM call. Looks like overkill, but using LangGraph means this classifier can be embedded as a sub-graph in a larger workflow later without rewriting its logic.

**Corrective RAG** (5-node graph with conditional edges): The branching logic (`score ≥ 0.65 → done, else → rewrite → re-retrieve → pick_best`) is naturally expressed as a graph. State flows between nodes via `CorrectiveRAGState` TypedDict. The graph can be extended (e.g. adding a third retrieval pass) without touching the pipeline.

**StateCase Agent** (5-node graph with ToolNode loop): The `agent → tools → update_mem → agent` loop is the standard LangGraph tool-calling pattern. `ToolNode` dispatches all tool calls from a single `AIMessage` in parallel. `add_messages` reducer appends to the message list so the full conversation including tool call/result pairs accumulates in state.

### Why `bind_tools()` Instead of a Custom Intent Classifier?

The original StateCase design had a separate intent classification node (`CONFIRM_TICKET | DECLINE_TICKET | TICKET_INTENT | CLARIFY | RAG`) that routed to different action nodes. This required:
- A separate LLM call just for classification
- Hard-coded branches for each intent
- A `pending_ticket_context` top-level state field that the router had to check
- Separate `confirm_ticket` and `decline_ticket` nodes

With `bind_tools()`, the LLM receives schemas for all five tools and decides which to call (or none) in a single pass. The confirmation flow is handled by the system prompt: when `pending_ticket_context` is in memory, the prompt tells the LLM "if user says yes, call `create_support_ticket` with this question." The LLM's natural language understanding handles "yes", "sure", "yeah do it", "affirmative" without an explicit intent classifier.

### Why Inject `session_id` Into the System Prompt?

`create_support_ticket` requires `session_id` as an argument. The LLM knows the session_id only if it's told what it is. Without explicit injection, the LLM guesses or generates a random UUID — which then appears in Notion's `User Info` field as a string that doesn't match any Redis key. Fixed by `_build_memory_context(memory, session_id)` which includes `"- Current session_id: {session_id} ← ALWAYS use this exact value"` in the system prompt.

### Why Replace Prior AI Turns With a Placeholder in Session History?

When session history is injected into the LLM context (last 6 turns), raw AI turns carry factual content from previous topics. If a user asked about "leave entitlement" and got a correct answer, then asked "summarise it briefly" — the LLM would see its prior answer and summarise that, even if the new retrieval returned different content. Replacing AI turns with `"[Previous answer — see retrieved context below for current facts.]"` preserves the turn-pair structure (so the LLM can follow pronoun references like "it" or "that") without injecting stale factual claims that could override the freshly retrieved context.

### Why `avg_score < 0.30` as the Out-of-Scope Gate?

COSINE similarity of 0.30 represents very weak semantic match. Below this, the retrieved chunks are unlikely to contain a relevant answer — their topics are semantically distant from the query. Calling the LLM with such weak context would produce hallucinated answers. Returning `OUT_OF_SCOPE_RESPONSE` without the LLM call eliminates hallucination, saves token costs, and (in StateCase mode) triggers the ticket offer flow. The threshold is defined in `prompts_rag.py` as `OUT_OF_SCOPE_SCORE_THRESHOLD` and used by both the pipeline and the StateCase agent's `update_mem` node.

### Why Milvus Lite Instead of a Hosted Vector Database?

Milvus Lite runs in-process with no external service requirement — just a file at `./rag_data/milvus.db`. This eliminates infrastructure setup for development and small deployments. The migration path to full Milvus is a single env var change (`MILVUS_URI=http://milvus-server:19530`) and a schema rebuild to switch `AUTOINDEX → HNSW` and add a sparse vector field for BM25 hybrid search. The `hybrid_search_chunks()` function already accepts `query_text` (unused in milvus-lite) so callers need no changes.

### Why Chunk at 400 Tokens With 60-Token Overlap?

400 tokens (~300 words) is enough context for the LLM to understand a policy paragraph or procedure step without including irrelevant adjacent content. Longer chunks dilute the semantic signal — a 2000-token chunk about "HR policy" would score reasonably for every HR question regardless of whether it covers the specific topic. 60-token overlap (roughly 3-4 sentences) prevents context loss at chunk boundaries — policy statements that span paragraph breaks are not severed in a way that loses their meaning.

### Why Redis Is Optional but Critical

Without Redis the system still works: retrieval cache misses hit Azure + Milvus on every request; session history is lost between requests (single-turn mode); StateCase agent memory resets each turn (no pending ticket context survives). With Redis: retrieval results are cached 10 minutes (same query = zero Azure + Milvus calls); session history enables multi-turn conversation; `pending_ticket_context` survives across HTTP requests enabling the yes/no confirmation flow; ticket counter provides collision-free sequential IDs. The `socket_connect_timeout=2` in `_get_client()` ensures a missing Redis server degrades within 2 seconds rather than hanging.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `AZURE_OPENAI_LLM_KEY` | Yes | Azure OpenAI API key for GPT-4.1-mini |
| `AZURE_LLM_ENDPOINT` | Yes | Azure OpenAI endpoint URL for the LLM |
| `AZURE_LLM_API_VERSION` | No | Default: `2024-12-01-preview` |
| `AZURE_LLM_DEPLOYMENT_41_MINI` | No | Default: `gpt-4.1-mini` |
| `AZURE_OPENAI_EMB_KEY` | Yes | Azure OpenAI API key for embeddings |
| `AZURE_EMB_ENDPOINT` | Yes | Azure OpenAI endpoint URL for embeddings |
| `AZURE_EMB_API_VERSION` | No | Default: `2024-12-01-preview` |
| `AZURE_EMB_DEPLOYMENT` | No | Default: `text-embedding-3-large` |
| `NOTION_API_KEY` | Yes | Notion integration secret (must be shared on both databases) |
| `NOTION_ROOT_PAGE_ID` | Yes | Document library database ID (for ingestion) |
| `STATECASE_DB_ID` | No | Default: `32d89db1-5e5b-8051-a212-f5f983a90a0f` |
| `REDIS_URL` | No | Default: `redis://localhost:6379`. If unset or Redis unreachable, caching silently disabled. |
| `MILVUS_URI` | No | Default: `./rag_data/milvus.db`. Set to `http://server:19530` for full Milvus. |

---

## Performance Characteristics

| Operation | Typical Time | Bottleneck |
|-----------|-------------|-----------|
| GET /health | < 5ms | None |
| POST /chat (Redis retrieval HIT) | 3-5s | Azure LLM call only |
| POST /chat (Redis retrieval MISS, no rewrite) | 5-8s | Azure embedding + Milvus + LLM |
| POST /chat (corrective rewrite triggered) | 10-15s | Azure embedding x2 + LLM rewrite + LLM answer |
| POST /statecase/chat (rag_search) | 7-12s | Extra LLM call for tool dispatch + rag pipeline |
| POST /statecase/chat (ticket create) | 3-6s | Notion API PATCH |
| POST /ingestion/notion (per page) | 5-30s | Notion blocks fetch + Azure embed + Milvus insert |
| GET /retrieval/debug | 2-4s | Azure embedding + Milvus |
| POST /evaluation/run (per question) | 8-15s | RAG pipeline + RAGAS LLM judge |

---

## Known Constraints and Limitations

- **milvus-lite index types**: AUTOINDEX only (no HNSW, no sparse vectors, no BM25). Hybrid search requires migrating to a full Milvus server.
- **Single Azure LLM for all tasks**: routing, rewriting, answering, and StateCase tool dispatch all use GPT-4.1-mini. Rate limits affect all features simultaneously.
- **StateCase ticket counter**: `statecase:ticket_counter` in Redis is permanent. If Redis is wiped, the counter resets to 1. New tickets may get IDs that collide with Notion IDs from before the wipe (though Notion's dedup check will prevent duplicate rows for the same question).
- **Session history grows unbounded**: `rag:session:{id}` accumulates all turns for 24 hours. The pipeline only uses the last 6, but the stored list grows with every message. A very long session will store a large JSON blob in Redis.
- **Notion integration must be shared on both databases**: the document library AND the StateCase tickets database must have the integration added via Notion → Connections. A single missing connection returns zero rows silently.
- **StateCase status options are fixed by Notion**: `"Not started"`, `"In progress"`, `"Done"` are Notion's built-in Status type options. Custom status names require recreating the column in Notion.
- **No horizontal scaling**: FastAPI, Streamlit, Redis, and Milvus Lite all run on one machine. Scaling horizontally would require externalising Redis (already done via env var) and migrating Milvus Lite to a server deployment.