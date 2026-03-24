# CiteRagLab — Codebase Architecture

This document describes every module, every design decision, and how all the pieces fit together. Read this before modifying any pipeline component.

---

## High-Level Request Flow

```
User message
    │
    ▼
[FastAPI /chat]  main_rag.py
    │  Load session history from Redis
    │
    ▼
[run_rag_pipeline]  pipeline_rag.py
    │
    ├─ Step 1: build_filters()
    │
    ├─ Step 2: classify_query()         LLM intent classifier (LangGraph)
    │          ┌────────────────────────────────────────────────────┐
    │          │ GREETING → short-circuit: return identity card     │
    │          └────────────────────────────────────────────────────┘
    │
    ├─ Step 3: corrective_retrieve()    retrieval + optional query rewrite (LangGraph)
    │            ├─ retrieve() pass 1   embed query → Milvus COSINE search
    │            ├─ avg_score()
    │            ├─ score < 0.65 → rewrite query → retrieve() pass 2
    │            └─ pick_best()
    │
    ├─ Step 4: rerank()                 enforce top_k cap (pass-through)
    │
    ├─ Step 4b: score gate              avg_score < 0.30 → OUT_OF_SCOPE_RESPONSE
    │
    ├─ Step 5: format_context_for_prompt()
    │
    ├─ Step 6: AzureChatOpenAI.invoke()
    │
    └─ Step 7: build citations list     includes chunk_text for RAGAS
    │
    ▼
[FastAPI response]
    │  Cache chunks in Redis
    │  Persist session history
    │
    ▼
Streamlit UI  cite_rag_lab_ui_rag.py
```

---

## Module Reference

### `rag/api/main_rag.py`

**Role:** FastAPI application. All HTTP routing and request/response models.

**Routes:**

| Route | Purpose |
|---|---|
| `POST /chat` | Load Redis history → run pipeline → save history → cache chunks |
| `DELETE /session` | Delete Redis key for a session |
| `GET /retrieval/debug` | Direct Milvus search bypassing the pipeline (Inspector tab) |
| `POST /ingestion/notion` | Full or single-page Notion ingest |
| `POST /evaluation/run` | Forward dataset to `run_ragas_evaluation()` |
| `GET /health` | Liveness probe |

The evaluation debug loop is fully inside the `for i` loop — each question's chunks are logged individually with an empty-context warning if any `chunk_text` is `""`.

---

### `rag/pipeline/pipeline_rag.py`

**Role:** Orchestrates the full RAG pipeline. Single entry point: `run_rag_pipeline()`.

**Short-circuits (no LLM call, no retrieval):**
- `mode == "GREETING"` → returns `GREETING_RESPONSE` immediately.
- `avg_score < OUT_OF_SCOPE_SCORE_THRESHOLD` → returns `OUT_OF_SCOPE_RESPONSE` after retrieval, before the LLM call.

**Session history:** last 6 turns from Redis are included in the LLM message list for multi-turn context.

**Citations:** every citation dict includes `chunk_text` — required for RAGAS evaluation. The UI displays only the metadata fields but RAGAS reads `chunk_text`.

---

### `rag/pipeline/adaptive_router_rag.py`

**Role:** Classifies the user query into one of five modes using an LLM.

**Modes:**

| Mode | top_k | Behaviour |
|---|---|---|
| `GREETING` | 0 | Pipeline short-circuits — returns identity card |
| `QA` | 5 | Direct factual question |
| `COMPARE` | 10 | Similarities / Key Differences / Recommendation |
| `SUMMARIZE` | 10 | Overview / Key Points / Conclusions |
| `SEARCH` | 8 | Find/list documents |

**Classification approach — no regex:**
Classification is entirely LLM-driven so informal language, typos, and creative phrasings are handled correctly. Two LLM attempts are made in sequence:

1. `_CLASSIFICATION_PROMPT` — detailed prompt with examples of informal language and the rule "when in doubt between GREETING and anything else, choose GREETING".
2. `_FALLBACK_CLASSIFICATION_PROMPT` — differently worded to avoid repeating a transient model quirk.
3. Both fail → default to `QA`, log error.

LangGraph is used so the router can be composed into a larger agentic graph in future without rewriting classification logic.

---

### `rag/pipeline/corrective_rag_rag.py`

**Role:** Corrective retrieval loop using LangGraph.

**Graph:**

```
retrieve → avg_score
                │
                ├─ score ≥ 0.65  →  done  (return first-pass results)
                │
                └─ score < 0.65  →  rewrite → retrieve2 → pick_best
```

`RELEVANCE_THRESHOLD = 0.65` — configurable at the top of the file.

The rewrite prompt (`REFINE_QUERY_PROMPT`) instructs the LLM to produce a better document-retrieval query — not a general knowledge question or coding request.

`pick_best()` compares both passes and keeps the higher-scoring one. If the rewrite LLM call fails, first-pass results are returned unchanged — the pipeline never crashes due to a rewrite failure.

---

### `rag/pipeline/prompts_rag.py`

**Role:** Single source of truth for all prompts and hardcoded responses.

| Constant | Used by | Purpose |
|---|---|---|
| `RAG_SYSTEM_PROMPT` | `pipeline_rag.py` (QA + SEARCH) | Natural answering prompt |
| `COMPARE_SYSTEM_PROMPT` | `pipeline_rag.py` (COMPARE) | Comparison structure |
| `SUMMARIZE_SYSTEM_PROMPT` | `pipeline_rag.py` (SUMMARIZE) | Summary structure |
| `REFINE_QUERY_PROMPT` | `corrective_rag_rag.py` | Query rewrite |
| `OUT_OF_SCOPE_SCORE_THRESHOLD` | `pipeline_rag.py` | COSINE score below which = not in library |
| `OUT_OF_SCOPE_RESPONSE` | `pipeline_rag.py` | Score gate response — no LLM call |
| `GREETING_RESPONSE` | `pipeline_rag.py` | GREETING mode response — no retrieval, no LLM |

**Design principle:** prompts are natural and helpful. Scope enforcement is the pipeline's responsibility (score gate + GREETING short-circuit). The LLM is told to cite sources and note gaps — nothing more restrictive.

---

### `rag/pipeline/redis_cache_rag.py`

**Role:** Async Redis client with three namespaces.

| Namespace | Key | TTL | Purpose |
|---|---|---|---|
| Retrieval | `rag:retrieval:{sha256[:16]}` | 10 min | Cache chunks for (query, filters) |
| Session | `rag:session:{session_id}` | 24 h | Chat turn history |
| Rate limit | `rag:notion:reads` | 60 s | Notion API call counter |

All operations return silently if Redis is unavailable. The pipeline runs uncached but does not fail.

---

### `rag/pipeline/reranker_rag.py`

**Role:** API compatibility shim. Enforces the `top_k` cap on the score-ordered Milvus results.

No reranking is performed. Milvus COSINE search returns results ordered by score. To add a cross-encoder reranker, replace the body of `rerank()` — no other file needs changing.

---

### `rag/retrieval/retriever_rag.py`

**Role:** Embeds the query and calls Milvus.

1. `embed_text(query)` — Azure OpenAI `text-embedding-3-large` → 3072-dim float vector.
2. `hybrid_search_chunks(embedding, text, top_k, filters)` — COSINE search in Milvus.

`EMBED_MODEL` and Azure credentials must match `embedder_rag.py` exactly so query and document vectors live in the same space and COSINE scores are meaningful.

`format_context_for_prompt()` produces a numbered `[N] title → section (meta)\nchunk_text` block for the LLM.

---

### `rag/retrieval/milvus_client_rag.py`

**Role:** Milvus Lite collection lifecycle and search.

**Index:** `AUTOINDEX + COSINE` — the only ANN index supported by milvus-lite. Scores are COSINE similarity (0.0–1.0).

**Schema:**

| Field | Type | Max Length | Notes |
|---|---|---|---|
| `id` | INT64 PK | — | Auto-generated |
| `embedding` | FLOAT_VECTOR | 3072 | text-embedding-3-large |
| `chunk_text` | VARCHAR | 8192 | Truncated to 8000 at insert |
| `doc_id` | VARCHAR | 256 | Notion page UUID |
| `title` | VARCHAR | 512 | Notion Title column |
| `section` | VARCHAR | 256 | Most recent heading above chunk |
| `industry` | VARCHAR | 128 | Industry select column |
| `doc_type` | VARCHAR | 128 | Type select column |
| `version` | VARCHAR | 32 | Version rich_text column |
| `tags` | VARCHAR | 512 | multi_select, comma-joined |
| `page_id` | VARCHAR | 128 | Notion page UUID |
| `block_range` | VARCHAR | 64 | e.g. "12-18" |

`hybrid_search_chunks()` is named for forward-compatibility — migrating to a full Milvus server only requires replacing the `collection.search()` call with `collection.hybrid_search()` (HNSW + BM25 + RRFRanker) and adding the sparse field to the schema.

---

### `rag/retrieval/filters_rag.py`

**Role:** Validates filter dicts from API requests into Milvus boolean expressions.

Allowed keys: `industry`, `doc_type`, `version` (exact match), `tags` (substring match). Unknown keys are dropped silently. Blank values are stripped.

---

### `rag/ingestion/ingestion_pipeline_rag.py`

**Role:** Four-stage ingestion orchestrator.

```
get_page_blocks() → chunk_page() → embed_chunks() → insert_chunks()
```

`ingest_all_pages()` continues past individual page failures — errors are collected and returned in the summary. `INTER_PAGE_DELAY_SEC = 0.5` keeps cumulative Notion API rate under 3 req/s.

---

### `rag/ingestion/notion_loader_rag.py`

**Role:** Notion API client and recursive block extractor.

**Rate limiting:**
- 0.35s sleep after every successful call (≈ 2.85 req/s).
- HTTP 429: honour `Retry-After` header → exponential back-off (base 2s, max 64s) → up to 6 retries.

**Block types handled:** headings, paragraphs, bullets, numbered lists, toggles, callouts, quotes, code (fenced), tables (pipe-delimited rows), nested children. Recursion capped at `MAX_BLOCK_DEPTH = 5`.

**API version:** pinned to `2022-06-28` — supports `POST /databases/{id}/query`. notion-client v3 defaults to a newer version that moved database querying.

---

### `rag/ingestion/chunker_rag.py`

**Role:** Token-aware overlapping chunker.

| Constant | Value | Effect |
|---|---|---|
| `TARGET_TOKENS` | 400 | Flush buffer when it reaches this size |
| `MAX_TOKENS` | 500 | Hard cap — flush before adding a block that would exceed this |
| `OVERLAP_TOKENS` | 60 | Carry last N tokens into next chunk |

Token estimate: 1 token ≈ 4 characters.

Tables are flushed as a unit so the header row always stays with its data rows. Code blocks are kept intact in one chunk.

---

### `rag/ingestion/embedder_rag.py`

**Role:** Batch-embeds chunks using Azure OpenAI `text-embedding-3-large`.

- Batch size 32. One retry with 2s sleep on API error.
- Adds `"embedding": list[float]` in-place to each chunk dict.
- Azure response sorted by `index` field to guarantee order.

---

### `rag/evaluation/ragas_runner_rag.py`

**Role:** RAGAS evaluation using Azure OpenAI as the judge.

**Metrics:** faithfulness, answer_relevancy, context_precision, context_recall.

**Dataset columns (RAGAS v0.2+):** `user_input`, `response`, `retrieved_contexts`, `reference`.

`retrieved_contexts` must contain actual chunk text strings — not empty strings. This is why `chunk_text` is included in every citation dict in `pipeline_rag.py`.

---

### `ui/cite_rag_lab_ui_rag.py`

**Role:** Streamlit UI — four tabs: Chat, Inspector, Ingest, Evaluation.

**Refusal detection:** sources expander and pipeline metadata footer are hidden when:
- `pipeline_meta.get("mode") == "GREETING"`, or
- `content.strip().startswith(_REFUSAL_PREFIXES)` — covers `OUT_OF_SCOPE_RESPONSE` and legacy cached messages.

**Evaluation flow:** parse `question\tground_truth` lines → call `/chat` per question → extract `chunk_text` from citations → send to `/evaluation/run` → display RAGAS scores.

---

### `ui/api_helpers_rag.py`

**Role:** HTTP wrappers for the FastAPI backend. No Streamlit dependency.

| Function | Endpoint | Timeout |
|---|---|---|
| `call_chat()` | `POST /chat` | 90s |
| `call_retrieval_debug()` | `GET /retrieval/debug` | 30s |
| `call_ingest_notion()` | `POST /ingestion/notion` | 600s |
| `call_run_evaluation()` | `POST /evaluation/run` | 300s |
| `call_delete_session()` | `DELETE /session` | 10s |

---

## Data Flows

### Ingestion

```
Notion database
    │
    ▼  get_all_pages()
    list of {page_id, title, doc_type, industry, version, tags}
    │
    ▼  get_page_blocks()  (per page)
    flat list of {heading, text, block_idx}
    │
    ▼  chunk_page()
    list of {chunk_text, doc_id, title, section, industry, doc_type, version, page_id, block_range}
    │
    ▼  embed_chunks()
    + "embedding": list[float, 3072]
    │
    ▼  insert_chunks()
    persisted in ./rag_data/milvus.db
```

### Query

```
query (str)
    │
    ▼  build_filters()  →  validated filter dict
    │
    ▼  classify_query()  →  mode ∈ {GREETING, QA, COMPARE, SUMMARIZE, SEARCH}
    │
    ├─ GREETING  →  return GREETING_RESPONSE  ← (end)
    │
    ▼  corrective_retrieve()
    │    embed query → Milvus COSINE → list[chunk]
    │    avg_score < 0.65 → LLM rewrite → re-embed → Milvus → pick_best
    │
    ▼  rerank()  →  chunks[:top_k]
    │
    ├─ avg_score < 0.30  →  return OUT_OF_SCOPE_RESPONSE  ← (end)
    │
    ▼  format_context_for_prompt()  →  numbered [N] context string
    │
    ▼  AzureChatOpenAI.invoke()  →  answer (str)
    │
    ▼  build citations  →  [{index, title, section, doc_type, industry,
                              version, tags, page_id, score, chunk_text}]
    │
    ▼  return {answer, citations, chunks, mode, rewritten, avg_score}
```

---

## Key Design Decisions

### Scope enforcement in the pipeline, not in prompts

Hard prompt instructions are fragile — LLMs can always be prompted past them. Instead:

- **GREETING** is detected by the LLM classifier and short-circuited before any retrieval happens.
- **Off-topic queries** are caught by the score gate after retrieval. If the library does not cover the topic, Milvus returns low-scoring results — that signal is deterministic and cannot be prompted around.

Prompts are natural and helpful. The LLM focuses on producing good answers.

### LLM-only intent classification

No regex or keyword matching. Two different prompts are tried in sequence — a transient model quirk on attempt 1 is unlikely to repeat on attempt 2. The "when in doubt, choose GREETING" rule in the primary prompt prevents casual or ambiguous phrasing from leaking into document retrieval.

### milvus-lite for zero-infrastructure local deployment

milvus-lite runs in-process as a single file. No Docker, no server. The tradeoff is dense-only search. `hybrid_search_chunks()` is forward-compatible — migrating to full Milvus only requires changing the body of that one function.

### chunk_text in citations

RAGAS evaluation requires the actual retrieved text. Including `chunk_text` in every citation dict means the evaluation tab can pass real context to RAGAS without an additional database lookup, and without any changes to the API contract.

### Redis is optional

All Redis operations are try/except wrapped and silently no-op on failure. The pipeline is fully functional without Redis — sessions are not persisted between restarts and results are not cached, but nothing breaks.