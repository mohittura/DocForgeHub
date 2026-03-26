# CiteRagLab & StateCase — Deep Code-Level Technical Documentation

> **Who this is for:** A new engineer joining the project with zero prior context. Every file is explained — what it is, why it exists, what every major block of code does, and what would break if you removed it. Nothing is assumed.

---

## Table of Contents

1. [What Was Built and Why](#1-what-was-built-and-why)
2. [Project Folder Layout](#2-project-folder-layout)
3. [The Vocabulary You Need First](#3-the-vocabulary-you-need-first)
4. [CiteRagLab — The RAG System](#4-citeraglab--the-rag-system)
   - 4.1 [notion_loader_rag.py — Fetching Documents from Notion](#41-notion_loader_ragpy--fetching-documents-from-notion)
   - 4.2 [chunker_rag.py — Splitting Text into Searchable Pieces](#42-chunker_ragpy--splitting-text-into-searchable-pieces)
   - 4.3 [embedder_rag.py — Turning Text into Numbers](#43-embedder_ragpy--turning-text-into-numbers)
   - 4.4 [ingestion_pipeline_rag.py — Orchestrating the Ingest](#44-ingestion_pipeline_ragpy--orchestrating-the-ingest)
   - 4.5 [milvus_client_rag.py — The Vector Database](#45-milvus_client_ragpy--the-vector-database)
   - 4.6 [filters_rag.py — Metadata Filtering](#46-filters_ragpy--metadata-filtering)
   - 4.7 [retriever_rag.py — Finding Relevant Chunks at Query Time](#47-retriever_ragpy--finding-relevant-chunks-at-query-time)
   - 4.8 [adaptive_router_rag.py — Classifying What the User Wants](#48-adaptive_router_ragpy--classifying-what-the-user-wants)
   - 4.9 [corrective_rag_rag.py — Fixing Weak Retrievals Automatically](#49-corrective_rag_ragpy--fixing-weak-retrievals-automatically)
   - 4.10 [reranker_rag.py — The Reranker Shim](#410-reranker_ragpy--the-reranker-shim)
   - 4.11 [prompts_rag.py — Every LLM Prompt in One Place](#411-prompts_ragpy--every-llm-prompt-in-one-place)
   - 4.12 [redis_cache_rag.py — Caching and Session Memory](#412-redis_cache_ragpy--caching-and-session-memory)
   - 4.13 [pipeline_rag.py — The Main Orchestrator](#413-pipeline_ragpy--the-main-orchestrator)
   - 4.14 [ragas_runner_rag.py — Measuring RAG Quality](#414-ragas_runner_ragpy--measuring-rag-quality)
   - 4.15 [main_rag.py — The FastAPI Backend](#415-main_ragpy--the-fastapi-backend)
   - 4.16 [api_helpers_rag.py — UI-to-Backend HTTP Wrappers](#416-api_helpers_ragpy--ui-to-backend-http-wrappers)
   - 4.17 [cite_rag_lab_ui_rag.py — The Streamlit UI](#417-cite_rag_lab_ui_ragpy--the-streamlit-ui)
5. [StateCase — The Ticketing Agent](#5-statecase--the-ticketing-agent)
   - 5.1 [statecase_notion_rag.py — The Notion Ticket Database Client](#51-statecase_notion_ragpy--the-notion-ticket-database-client)
   - 5.2 [statecase_tools_rag.py — LangChain Tools](#52-statecase_tools_ragpy--langchain-tools)
   - 5.3 [statecase_agent_rag.py — The LangGraph Tool-Calling Agent](#53-statecase_agent_ragpy--the-langgraph-tool-calling-agent)
   - 5.4 [statecase_routes_rag.py — StateCase FastAPI Routes](#54-statecase_routes_ragpy--statecase-fastapi-routes)
   - 5.5 [api_helpers_statecase_rag.py — StateCase HTTP Wrappers](#55-api_helpers_statecase_ragpy--statecase-http-wrappers)
6. [Bugs Found and How They Were Fixed](#6-bugs-found-and-how-they-were-fixed)
7. [How Everything Connects — End-to-End Request Trace](#7-how-everything-connects--end-to-end-request-trace)
8. [Environment Variables](#8-environment-variables)

---

## 1. What Was Built and Why

### The Problem

A company stores its internal documents in Notion — HR policies, SOPs, handbooks, templates, vendor guides. Employees have questions about these documents all day long. Without a system, they either ask a colleague (who may not know), search Notion manually (slow, keyword-based), or just guess.

### Solution 1: CiteRagLab

CiteRagLab is a **Retrieval-Augmented Generation (RAG)** system. RAG means: instead of asking an LLM to answer from its training data (which goes stale), you first retrieve the relevant document excerpts from your own database, then give those excerpts to the LLM and ask it to answer from them. The LLM cites which excerpt it used, so you can verify the answer is grounded in your actual documents.

The system supports:
- Asking questions and getting cited answers
- Comparing two documents side by side
- Summarising a topic
- Searching for documents by topic, type, or industry
- Multi-turn conversation (remembers what you asked before)
- Filtering by industry, document type, version

### Solution 2: StateCase

Not every question can be answered from existing documents. When the RAG system cannot find a good answer, that question should be tracked as a support ticket so someone can add the missing document or answer manually. StateCase is a **ticketing agent** built on top of CiteRagLab that:
- Detects when the RAG cannot answer (low retrieval score)
- Asks the user whether to raise a ticket
- Creates the ticket in Notion with full context attached
- Allows listing, updating, and managing tickets from the chat interface

---

## 2. Project Folder Layout

```
DOCFORGEHUB/
└── rag/
    ├── api/
    │   ├── main_rag.py                 ← FastAPI app. Starts the HTTP server on port 8001.
    │   └── statecase_routes_rag.py     ← StateCase HTTP endpoints, mounted on main_rag.
    │
    ├── evaluation/
    │   └── ragas_runner_rag.py         ← Runs RAGAS metrics to measure RAG quality.
    │
    ├── ingestion/
    │   ├── notion_loader_rag.py        ← Reads documents from Notion via API.
    │   ├── chunker_rag.py              ← Splits document text into 400-token chunks.
    │   ├── embedder_rag.py             ← Converts chunk text to 3072-dim vectors.
    │   └── ingestion_pipeline_rag.py   ← Chains loader → chunker → embedder → Milvus.
    │
    ├── pipeline/
    │   ├── adaptive_router_rag.py      ← LangGraph: classifies query as QA/COMPARE/etc.
    │   ├── corrective_rag_rag.py       ← LangGraph: retries with rewritten query if weak.
    │   ├── pipeline_rag.py             ← Main orchestrator: runs the full RAG pipeline.
    │   ├── prompts_rag.py              ← All LLM prompt strings in one place.
    │   ├── reranker_rag.py             ← Shim: enforces top_k cap (no cross-encoder yet).
    │   ├── redis_cache_rag.py          ← Async Redis: caches chunks, stores session history.
    │   ├── statecase_agent_rag.py      ← LangGraph agent with bind_tools + ToolNode.
    │   ├── statecase_notion_rag.py     ← Notion CRUD for the StateCase tickets database.
    │   └── statecase_tools_rag.py      ← Five @tool-decorated functions for the agent.
    │
    └── retrieval/
        ├── filters_rag.py              ← Validates and cleans metadata filter dicts.
        ├── milvus_client_rag.py        ← Milvus Lite vector store: insert and search.
        └── retriever_rag.py            ← Embeds query, calls Milvus, formats context.
│
ui/
    ├── api_helpers_rag.py              ← HTTP wrappers for CiteRagLab backend calls.
    ├── api_helpers_statecase_rag.py    ← HTTP wrappers for StateCase backend calls.
    └── cite_rag_lab_ui_rag.py          ← Streamlit UI: Chat, Tickets, Inspector, etc.
```

---

## 3. The Vocabulary You Need First

Before reading the code, you need to understand these terms:

**Embedding / Vector:** A list of ~3000 floating-point numbers that represents the semantic meaning of a piece of text. Two texts that mean similar things will have vectors that are close to each other in mathematical space. This is how we find relevant documents — not by keyword matching but by meaning.

**COSINE similarity:** A number between -1 and 1 that measures how similar two vectors are. 1.0 = identical meaning, 0.0 = unrelated, -1.0 = opposite meaning. A score above 0.65 in this system means good relevance; below 0.30 means the question is probably not covered by the documents.

**Chunk:** A piece of a document — typically 300-500 tokens (roughly 300-400 words). Documents are split into chunks because LLMs have limited context windows, and because one chunk about "bereavement leave" is more useful than dumping an entire 20-page HR policy.

**Milvus / Milvus Lite:** A vector database. Like a SQL database but instead of searching by WHERE clause, you search by "give me the top 5 chunks whose vectors are closest to this query vector". Milvus Lite runs entirely in-process — no server, no Docker, just a file on disk.

**LangChain:** A Python library for building LLM applications. Provides `AzureChatOpenAI` (LLM client), `ChatPromptTemplate` (prompt builder), `@tool` (function decorator), `AIMessage`/`HumanMessage`/`SystemMessage` (message types).

**LangGraph:** A library for building stateful, multi-step LLM workflows as directed graphs. You define nodes (Python functions) and edges (which node runs after which), and LangGraph executes them in order, passing a shared state dict between nodes. Used for the adaptive router, corrective RAG loop, and the StateCase agent.

**ToolNode / bind_tools:** LangChain/LangGraph mechanism for letting the LLM call Python functions. `llm.bind_tools(tools)` tells the LLM what functions exist (via Pydantic schema). If the LLM's response contains a `tool_calls` field, `ToolNode` automatically dispatches those calls to the right function.

**RAG (Retrieval-Augmented Generation):** The full pattern: retrieve relevant text → inject into LLM prompt → LLM generates a grounded answer from that text rather than from its training data.

**Redis:** An in-memory key-value store. Used here for: caching retrieval results (so the same query doesn't hit Milvus+Azure repeatedly), storing session history (so the LLM remembers what was discussed), and counting Notion API calls for rate limiting.

**Session / Session ID:** A short string (e.g. `"86e7f692"`) that identifies one conversation. All messages in one conversation share the same session ID. Session history is stored in Redis keyed by this ID.

---

## 4. CiteRagLab — The RAG System

### 4.1 `notion_loader_rag.py` — Fetching Documents from Notion

**Why this file exists:** Notion is the source of truth for all company documents. Before anything can be searched, it must be read from Notion. This file handles all the complexity of the Notion API — pagination, nested blocks, rate limiting, and the many different block types.

#### Constants at the top

```python
REQUEST_DELAY_SEC = 0.35
```
Notion enforces a hard limit of ~3 requests per second. `1 / 0.35 = 2.85 req/s`, safely below that limit. Every successful API call sleeps this long. Without it, you get HTTP 429 (rate limit exceeded) errors during bulk ingestion.

```python
MAX_RETRIES = 6
BACKOFF_BASE_SEC = 2.0
MAX_BACKOFF_SEC = 64.0
```
When Notion does return 429 (it can happen even with the delay during bursts), we wait and retry. The wait doubles each time: 2s, 4s, 8s, 16s, 32s, 64s. After 6 retries we give up and raise the error. `MAX_BACKOFF_SEC = 64` prevents waiting forever on a persistent outage.

```python
MAX_BLOCK_DEPTH = 5
```
Notion pages can have nested blocks (a toggle inside a callout inside a toggle). We recurse to extract child blocks, but cap at depth 5 to prevent infinite loops if Notion ever returns circular references.

#### `_get_client()` — the Notion client singleton

```python
_notion_client: Optional[Client] = None

def _get_client() -> Client:
    global _notion_client
    if _notion_client is None:
        api_key = os.getenv("NOTION_API_KEY")
        _notion_client = Client(auth=api_key, notion_version="2022-06-28")
    return _notion_client
```

**Why singleton:** Creating an HTTP client is expensive. We create it once and reuse it. `global _notion_client` is the module-level singleton pattern in Python.

**Why `notion_version="2022-06-28"`:** The Notion SDK v3 defaults to API version `2025-09-03` which moved database querying to a different `/data_sources` endpoint. We pin to `2022-06-28` which uses the stable `POST /databases/{id}/query` endpoint.

#### `_notion_call()` — the rate-limited wrapper

```python
def _notion_call(api_fn, **kwargs):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = api_fn(**kwargs)
            time.sleep(REQUEST_DELAY_SEC)  # ← always sleep after success
            return result
        except APIResponseError as api_err:
            if api_err.status != 429:
                raise  # ← non-429 errors propagate immediately
            # ... compute wait time from Retry-After header or exponential backoff
            time.sleep(wait)
            if attempt == MAX_RETRIES:
                raise
```

**Why wrap every call:** Every single Notion API call — whether fetching a page, fetching blocks, or querying a database — goes through this function. This ensures rate limiting is never accidentally bypassed. The alternative (calling the API directly in scattered places) would mean rate limiting is inconsistently applied.

**Why check `api_err.status != 429`:** Only rate-limit errors should trigger retry. A 404 (page not found) or 403 (permission denied) should fail immediately — retrying would be pointless and slow.

**Why prefer `Retry-After` header over computed backoff:** When Notion returns 429, it sometimes includes a `Retry-After` header saying exactly how many seconds to wait. Using it is more accurate than our guess.

#### Property extractor functions

```python
def _prop_title(props: dict) -> str:
def _prop_select(props: dict, column_name: str) -> str:
def _prop_rich_text(props: dict, column_name: str) -> str:
def _prop_multi_select(props: dict, column_name: str) -> list[str]:
```

Notion's API returns properties in a deeply nested dict format. Each of these extracts the plain text value from one property type. They exist because the nesting is the same every time but verbose to write inline — these helpers keep `get_all_pages()` readable.

#### `_block_to_text()` — converting one block to plain text

```python
def _block_to_text(block: dict) -> tuple[str, str]:
    # returns (heading_label, plain_text)
```

Returns a tuple: the first element is non-empty only for headings (it signals the chunker to update the current section label), the second is the text content. Code blocks are wrapped in triple-backtick fences so the chunker can detect them. Table rows are returned as pipe-delimited strings (`col1 | col2 | col3`) so they render correctly.

**Why return a tuple instead of just the text:** The heading flag travels with the text so the chunker knows when a section boundary occurred, without needing a separate pass.

#### `_fetch_children()` — paginated child block fetching

```python
def _fetch_children(block_id: str) -> list[dict]:
    results = []
    cursor = None
    has_more = True
    while has_more:
        kwargs = {"block_id": block_id, "page_size": 100}
        if cursor:
            kwargs["start_cursor"] = cursor
        resp = _notion_call(client.blocks.children.list, **kwargs)
        results.extend(resp.get("results", []))
        cursor = resp.get("next_cursor")
        has_more = resp.get("has_more", False)
    return results
```

Notion paginates at 100 items per response. `has_more: True` means there are more pages. The `next_cursor` value is passed as `start_cursor` on the next request. This loop keeps fetching until `has_more` is False. Without this, long pages with many blocks would be silently truncated at 100.

#### `_extract_blocks_recursive()` — the main extraction engine

```python
def _extract_blocks_recursive(block_id: str, depth: int = 0) -> list[dict]:
```

For each block in the page:
1. If it's a `table` block, immediately fetch its children (which are `table_row` blocks) and convert each row to a pipe-delimited string
2. For all other types, convert using `_block_to_text()` and add to the flat output list
3. If the block has children (`has_children=True`) and is not a table, recurse into the children

**Why a flat output list instead of a tree:** The chunker downstream expects a flat list. Nesting structure is irrelevant for chunking — what matters is text order and heading labels.

**Why the depth cap:** Some Notion pages have very deep nesting (toggles inside callouts inside columns). Without a cap, a pathological page could cause Python to hit its recursion limit and crash.

#### `get_all_pages()` — the public database reader

```python
def get_all_pages(database_id: Optional[str] = None) -> list[dict]:
```

1. Normalises the database ID to dashed UUID format (Notion requires `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`)
2. Calls `client.databases.retrieve()` first to verify the integration has access — if it fails, logs a clear error message explaining how to share the integration on the database
3. Paginates through all rows with `POST /databases/{id}/query`
4. Extracts `title`, `doc_type`, `industry`, `version`, `tags` from each row's properties
5. Returns a list of metadata dicts — one per document

**Why verify access separately:** Without the `retrieve()` call, a missing integration permission would return zero rows silently. The explicit check gives a clear error: "Share the integration on the database in Notion first."

#### `get_page_blocks()` — the public block reader

```python
def get_page_blocks(page_id: str) -> list[dict]:
    blocks = _extract_blocks_recursive(page_id, depth=0)
    return blocks
```

Thin wrapper that calls the recursive extractor and handles exceptions. Returns the flat list of `{heading, text, block_idx}` dicts.

---

### 4.2 `chunker_rag.py` — Splitting Text into Searchable Pieces

**Why this file exists:** You cannot embed an entire document and search it — the vector would be too diluted (mixing information about bereavement leave, sick leave, and parental leave into one blob makes it impossible to retrieve specifically). You cannot embed individual sentences — they are too short to carry enough context. Chunks of ~400 tokens are the sweet spot.

#### Constants

```python
TARGET_TOKENS  = 400  # aim to flush when buffer reaches this
MAX_TOKENS     = 500  # hard cap — flush before adding a block that would exceed this
OVERLAP_TOKENS = 60   # carry last N tokens into next chunk
```

**Why two size limits instead of one:** `TARGET_TOKENS` is a soft target — we flush when we hit it. `MAX_TOKENS` is a hard cap — we flush before adding a block that would push us over it. Without the hard cap, a single very long paragraph could produce a chunk much larger than intended.

**Why overlap:** When a document is split at a sentence boundary, the context on both sides of the cut is lost. If a chunk ends with "employees are entitled to..." and the next chunk starts with "five days of bereavement leave", neither chunk contains the complete thought. Carrying the last 60 tokens (roughly 3-4 sentences) into the start of the next chunk preserves continuity. When the LLM retrieves one of these chunks, it has enough context to answer correctly.

#### `_token_count(text)`

```python
def _token_count(text: str) -> int:
    return max(1, len(text) // 4)
```

`1 token ≈ 4 characters` is OpenAI's rule of thumb for English text. The `max(1, ...)` prevents returning 0 for empty strings (which would cause division-by-zero issues downstream).

#### `_is_table_row()` and `_is_code_block()`

```python
def _is_table_row(text: str) -> bool:
    return "|" in text

def _is_code_block(text: str) -> bool:
    return text.startswith("```")
```

Simple detectors because the notion loader already formatted these blocks in a recognisable way: table rows as pipe-delimited strings, code blocks as triple-backtick fences. These detectors drive the special handling logic below.

#### `chunk_page()` — the main function

```python
def chunk_page(page_id, title, blocks, industry, doc_type, version, doc_id=None):
```

**Initialisation:**
```python
effective_doc_id = doc_id or page_id   # allows overriding the doc ID
chunks: list[dict] = []                # output list
current_section = title                # section label, updated by headings
buffer_lines: list[str] = []           # accumulating text
buffer_tokens = 0                      # token count of buffer
start_block_idx = 0                    # first block index of current chunk
```

All of these are mutable state that changes as we walk through the blocks. `current_section` starts as the page title and updates every time we see a heading block.

#### The inner `_flush()` function

```python
def _flush(end_block_idx: int) -> None:
    nonlocal buffer_lines, buffer_tokens, start_block_idx
    
    chunk_text = "\n".join(buffer_lines).strip()
    if chunk_text:
        chunk = {
            "chunk_text":  chunk_text,
            "doc_id":      effective_doc_id,
            "title":       title,
            "section":     current_section,   # ← captures current heading
            "industry":    industry,
            "doc_type":    doc_type,
            "version":     version,
            "page_id":     page_id,
            "block_range": f"{start_block_idx}-{end_block_idx}",
        }
        chunks.append(chunk)
    
    # Build overlap: walk backwards through buffer, accumulate lines until OVERLAP_TOKENS
    overlap_lines = []
    overlap_tokens = 0
    for line in reversed(buffer_lines):
        line_tokens = _token_count(line)
        if overlap_tokens + line_tokens > OVERLAP_TOKENS:
            break
        overlap_lines.insert(0, line)    # insert at front to preserve order
        overlap_tokens += line_tokens
    
    buffer_lines = overlap_lines         # next chunk starts with the overlap
    buffer_tokens = overlap_tokens
    start_block_idx = end_block_idx
```

**Why `nonlocal`:** `_flush` is a nested function inside `chunk_page`. It needs to modify the outer function's local variables (`buffer_lines`, `buffer_tokens`, `start_block_idx`). In Python, `nonlocal` declares that these refer to the enclosing scope's variables, not new local ones.

**Why `current_section` is not nonlocal:** `current_section` is only read inside `_flush`, not modified. The outer loop modifies it via `current_section = heading`, which is fine without nonlocal.

**Why walk backwards for overlap:** We want the last N tokens of the buffer. Walking forward would require knowing the total in advance. Walking backwards lets us stop as soon as we've accumulated enough. `overlap_lines.insert(0, line)` puts each line at the front to restore the correct reading order.

#### The main loop

```python
for block in blocks:
    heading   = block.get("heading", "")
    text      = block.get("text", "").strip()
    block_idx = block.get("block_idx", 0)
    
    if heading:
        current_section = heading   # update section label for future chunks
    
    if not text:
        continue                    # skip empty blocks (headings with no text below)
```

**Section tracking:** When we encounter a heading, we update `current_section`. All subsequent chunks until the next heading will carry this section label in their metadata. This is crucial for citations — the UI shows "HR Policy → Bereavement Leave" not just "HR Policy".

**Table row accumulation:**
```python
if _is_table_row(text):
    if not table_row_group:
        table_start_idx = block_idx
    table_row_group.append(text)
    continue    # ← do NOT process yet, accumulate more rows
else:
    if table_row_group:
        _flush_table_group(block_idx)   # flush before processing non-table block
```

Tables are tricky because the header row (column names) and data rows are separate blocks. If you flush mid-table, you might put the header in one chunk and the data in another, making both useless. We accumulate all consecutive table rows and flush them as a unit.

**Code block handling:**
```python
if _is_code_block(text):
    code_tokens = _token_count(text)
    if buffer_lines:
        _flush(block_idx)          # flush whatever came before
    buffer_lines.append(text)
    buffer_tokens = code_tokens
    _flush(block_idx + 1)          # immediately flush the code block alone
    continue
```

Code blocks are fenced with triple-backticks. If we split a code block mid-fence (half the code in one chunk, half in the next), the LLM cannot understand either half. So we always flush the current buffer first, then emit the code block as its own chunk and immediately flush again.

**Regular text:**
```python
block_tokens = _token_count(text)

if buffer_tokens + block_tokens > MAX_TOKENS and buffer_lines:
    _flush(block_idx)       # flush before adding to avoid exceeding hard cap

buffer_lines.append(text)
buffer_tokens += block_tokens

if buffer_tokens >= TARGET_TOKENS:
    _flush(block_idx + 1)   # flush when we've hit the soft target
```

The hard cap check comes before appending. If we checked after, the buffer could temporarily exceed MAX_TOKENS. `block_idx + 1` as the end index marks the next block as the start of the next chunk.

---

### 4.3 `embedder_rag.py` — Turning Text into Numbers

**Why this file exists:** Milvus stores vectors, not text. Before inserting chunks into Milvus, their text must be converted to 3072-dimensional float vectors using Azure OpenAI's embedding model. This file handles that conversion for bulk ingestion (many chunks at once).

#### Constants

```python
EMBED_MODEL = os.getenv("AZURE_EMB_DEPLOYMENT", "text-embedding-3-large")
BATCH_SIZE  = 32
```

**Why `text-embedding-3-large`:** It produces 3072-dimensional vectors (vs 1536 for the older `text-embedding-ada-002`). More dimensions = more semantic nuance captured = better retrieval accuracy. The tradeoff is storage size and embedding time, which are acceptable for this use case.

**Why batch size 32:** Azure allows up to 2048 texts per request, but 32 is conservative. It keeps each API call fast (under 2-3 seconds) and makes retry logic simpler — if one batch fails, you only lose 32 texts, not thousands.

#### `_get_client()` — the embedding client singleton

```python
_client: AzureOpenAI | None = None

def _get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("AZURE_OPENAI_EMB_KEY")
        endpoint = os.getenv("AZURE_EMB_ENDPOINT")
        if not api_key:
            raise ValueError("AZURE_OPENAI_EMB_KEY is not set...")
        if not endpoint:
            raise ValueError("AZURE_EMB_ENDPOINT is not set...")
        _client = AzureOpenAI(api_key=api_key, azure_endpoint=endpoint, api_version=...)
    return _client
```

**Why validate env vars explicitly:** If these are missing, the `AzureOpenAI` constructor would succeed but the first API call would fail with a cryptic auth error. Raising a clear `ValueError` at construction time gives a better error message pointing directly at the missing variable.

#### `embed_chunks(chunks)` — the public function

```python
def embed_chunks(chunks: list[dict]) -> list[dict]:
    texts = [c["chunk_text"].replace("\n", " ") for c in chunks]
    embeddings = _batch_embed(texts)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb   # ← mutates in place
    return chunks
```

**Why replace newlines:** OpenAI recommends replacing newlines with spaces before embedding. Newlines can cause inconsistent tokenisation at line boundaries, leading to slightly different vectors for the same content depending on its formatting.

**Why mutate in place:** The chunks dict already exists in memory. Adding the `embedding` key to it avoids creating a copy. The caller (ingestion pipeline) gets back the same list with embeddings attached.

#### `_batch_embed(texts)` — batched embedding with retry

```python
for batch_idx, start in enumerate(range(0, len(texts), BATCH_SIZE)):
    batch = texts[start : start + BATCH_SIZE]
    
    for attempt in range(2):   # ← exactly 2 attempts
        try:
            resp = _get_client().embeddings.create(model=EMBED_MODEL, input=batch)
            batch_embs = [
                item.embedding
                for item in sorted(resp.data, key=lambda x: x.index)
            ]
            all_embeddings.extend(batch_embs)
            break
        except Exception as err:
            if attempt == 0:
                time.sleep(2)   # ← wait 2 seconds before retry
            else:
                raise           # ← second failure propagates
```

**Why sort by `item.index`:** Azure OpenAI does not guarantee that embeddings are returned in input order. The `index` field indicates which input each embedding corresponds to. Sorting by it ensures the embeddings align with the input texts.

**Why exactly 2 attempts:** For bulk ingestion, one retry is enough. If the API fails twice in a row, something systemic is wrong and the user should know. More retries would just slow down an already-failing ingest.

---

### 4.4 `ingestion_pipeline_rag.py` — Orchestrating the Ingest

**Why this file exists:** Each stage (load, chunk, embed, insert) is in a separate file. Something must chain them together. This is the orchestrator.

#### `ingest_page(page_meta)` — single page ingest

```python
def ingest_page(page_meta: dict) -> int:
    page_id  = page_meta["page_id"]
    title    = page_meta.get("title", "(untitled)")
    # ... extract other metadata fields
    
    # Stage 1: Fetch blocks from Notion
    blocks = get_page_blocks(page_id)
    if not blocks:
        return 0   # ← return 0, not error — empty pages are normal
    
    # Stage 2: Chunk
    chunks = chunk_page(page_id=page_id, title=title, blocks=blocks, ...)
    if not chunks:
        return 0
    
    # Stage 3: Embed
    embedded_chunks = embed_chunks(chunks)
    
    # Stage 4: Insert into Milvus
    inserted_count = insert_chunks(embedded_chunks)
    return inserted_count
```

**Why return 0 instead of raising on empty pages:** Some Notion pages are intentionally empty (placeholder pages, dividers). Returning 0 signals "nothing to do" without treating it as an error. The caller can count these as `pages_skipped` in the summary.

#### `ingest_all_pages()` — bulk ingest with error isolation

```python
def ingest_all_pages(database_id=None) -> dict:
    pages = get_all_pages(database_id)
    
    for page_number, page_meta in enumerate(pages, start=1):
        try:
            chunks_inserted = ingest_page(page_meta)
        except Exception as err:
            errors.append(f"Error ingesting page '{page_title}': {err}")
            # ← continue to next page, do not abort
        
        if page_number < total_pages:
            time.sleep(INTER_PAGE_DELAY_SEC)  # ← 0.5s courtesy pause between pages
    
    return {
        "pages_processed": total_pages,
        "chunks_inserted": total_chunks_inserted,
        "pages_skipped":   pages_skipped,
        "errors":          errors,
    }
```

**Why catch exceptions per page:** If one page has a corrupted block structure or a Notion API glitch, you don't want to lose the entire ingest run. The try/except isolates each page. Errors are collected and returned in the summary so the user can see which pages failed.

**Why `INTER_PAGE_DELAY_SEC = 0.5` between pages:** Each page fetch involves multiple Notion API calls (one for the page itself, then potentially many for its blocks depending on nesting depth). Even with `REQUEST_DELAY_SEC` per call, a complex page can trigger 10-20 calls in rapid succession. The 0.5s pause gives a breathing room buffer between pages to keep the sustained rate well under Notion's 3 req/s limit.

---

### 4.5 `milvus_client_rag.py` — The Vector Database

**Why this file exists:** Milvus is where all the chunk vectors live. This file manages the collection schema, connection lifecycle, chunk insertion, and vector search.

#### The `pkg_resources` mock at the top

```python
if "pkg_resources" not in sys.modules:
    try:
        import pkg_resources
    except ModuleNotFoundError:
        # create a minimal mock module
        _mock_pkg = ModuleType("pkg_resources")
        _mock_pkg.get_distribution = ...
        sys.modules["pkg_resources"] = _mock_pkg
```

**Why this exists:** `milvus_lite` imports `pkg_resources` only to read its own version number. `pkg_resources` is part of `setuptools` which may not be installed in all environments. Rather than requiring users to install setuptools just for this, we inject a minimal mock that satisfies the version-check call. Without this, importing `milvus_lite` would crash with `ModuleNotFoundError: No module named 'pkg_resources'`.

#### Connection config

```python
MILVUS_URI      = os.getenv("MILVUS_URI", "./rag_data/milvus.db")
COLLECTION_NAME = "notion_documents"
EMBEDDING_DIM   = 3072
```

`MILVUS_URI` pointing to a `.db` file means milvus-lite mode — the entire vector database is a single file. No server process, no Docker, no external dependency. The data directory is created automatically if it doesn't exist.

#### `_ensure_connected()` — connect exactly once

```python
_connected = False

def _ensure_connected() -> None:
    global _connected
    if _connected:
        return
    if not MILVUS_URI.startswith("http"):
        data_dir = os.path.dirname(os.path.abspath(MILVUS_URI))
        os.makedirs(data_dir, exist_ok=True)
    connections.connect(uri=MILVUS_URI)
    _connected = True
```

**Why the `http` check:** If `MILVUS_URI` starts with `http`, it's a full Milvus server — the data directory logic doesn't apply. This allows the same code to work with both milvus-lite (local file) and a full Milvus server (URL), just by changing the env var.

#### The collection schema

The schema defines exactly what fields are stored for each chunk. Key decisions:

```python
FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=3072)
```
The vector field. `dim=3072` must match the embedding model's output dimension exactly. If you ever switch models, you must drop and recreate the collection.

```python
FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=8192)
```
Storing the actual text alongside the vector means search results include the text — no secondary lookup needed. `8192` chars is generous; chunks are capped at 8000 on insert (`[:8000]`) to leave headroom for multi-byte characters.

```python
FieldSchema(name="tags", dtype=DataType.VARCHAR, max_length=512)
```
Tags are a multi-select in Notion (a list). Milvus doesn't support list fields, so tags are stored as a comma-joined string (`"HR,Policy,Leave"`) and searched with `LIKE "%HR%"`.

#### `get_collection()` — lazy collection init

```python
_collection = None

def get_collection():
    global _collection
    if _collection is not None:
        return _collection
    
    _ensure_connected()
    
    if utility.has_collection(COLLECTION_NAME):
        _collection = Collection(COLLECTION_NAME)
        _collection.load()
        return _collection
    
    # Create the collection with schema
    _collection = Collection(name=COLLECTION_NAME, schema=schema)
    _collection.create_index(
        field_name="embedding",
        index_params={"metric_type": "COSINE", "index_type": "AUTOINDEX"},
    )
    _collection.load()
    return _collection
```

**Why `AUTOINDEX`:** Milvus Lite only supports `FLAT`, `IVF_FLAT`, and `AUTOINDEX`. `AUTOINDEX` is the recommended choice for milvus-lite — it selects the best available index automatically. On a full Milvus server you'd use `HNSW` (Hierarchical Navigable Small World) for faster approximate nearest-neighbour search.

**Why `.load()` after creation/retrieval:** Milvus separates storage from memory. A collection must be "loaded" into memory before it can be searched. Without this call, search would fail with "collection not loaded".

#### `_build_filter_expr(filters)` — Milvus boolean expressions

```python
def _build_filter_expr(filters: Optional[dict]) -> Optional[str]:
    expr_parts = []
    for field_name, filter_key in [
        ("industry", "industry"),
        ("doc_type",  "doc_type"),
        ("version",   "version"),
    ]:
        val = filters.get(filter_key, "")
        if val:
            expr_parts.append(f'{field_name} == "{val}"')
    
    tag_filter = filters.get("tags", "")
    if tag_filter:
        expr_parts.append(f'{FIELD_TAGS} like "%{tag_filter}%"')
    
    return " && ".join(expr_parts) if expr_parts else None
```

**Why `like "%...%"` for tags:** Tags are stored as `"HR,Policy,Leave"`. Exact match (`==`) would only work if the user typed the entire tags string. `LIKE "%HR%"` matches any row where the tags field contains "HR" anywhere — which is what you want for multi-select filtering.

**Why return `None` instead of empty string:** Milvus treats an empty expression differently from no expression. Passing `None` means "no filter" (retrieve from all documents). Passing `""` might cause a parse error.

#### `hybrid_search_chunks()` — the search function

```python
def hybrid_search_chunks(query_embedding, query_text, top_k, filters=None):
    expr = _build_filter_expr(filters)
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE"},
        limit=top_k,
        expr=expr,
        output_fields=["chunk_text", "title", "section", ...]
    )
    hits = [_hit_to_dict(hit) for hit in results[0]]
    return hits
```

**Why `query_text` is not used:** This is milvus-lite. Full Milvus supports hybrid search (combining dense vectors with BM25 sparse vectors for keyword matching). The `query_text` parameter is kept in the signature so migrating to full Milvus only requires changing this one function body, not any callers.

**Why `data=[query_embedding]` (list of one):** Milvus supports batch queries (searching for multiple query vectors at once). We always search for one query at a time, so we wrap it in a list.

**Why `results[0]`:** The search result is a list of hit lists, one per input query. Since we always send one query, we always want `results[0]`.

#### `insert_chunks()` — inserting into Milvus

```python
def insert_chunks(chunks: list[dict]) -> int:
    data = [
        [c["embedding"]            for c in chunks],   # list of vectors
        [c.get("chunk_text","")[:8000] for c in chunks],
        [c.get("doc_id","")        for c in chunks],
        # ... one list per field
    ]
    collection.insert(data)
    collection.flush()   # ← forces data to be written to disk
    return len(chunks)
```

**Why column format (list of lists) instead of row format:** This is Milvus's native insert format — each field's data is one list. Row format would require conversion. Column format allows Milvus to batch-write each field type efficiently.

**Why `flush()` after insert:** `insert()` writes to an in-memory buffer. `flush()` forces the buffer to disk. Without it, data might be lost if the process crashes before the buffer is persisted.

---

### 4.6 `filters_rag.py` — Metadata Filtering

**Why this file exists:** Users can filter search results by industry, document type, version, or tags. The raw filter dict from the API request might contain invalid keys, empty values, or whitespace-only strings — all of which would produce malformed Milvus boolean expressions. This file sanitises the input before it reaches Milvus.

```python
ALLOWED_FILTER_KEYS = {"industry", "doc_type", "version", "tags"}

def build_filters(raw: dict) -> dict:
    clean: dict = {}
    dropped: list[str] = []
    
    for key, value in raw.items():
        if key not in ALLOWED_FILTER_KEYS:
            dropped.append(key)
            continue
        cleaned_value = str(value).strip() if value is not None else ""
        if cleaned_value:                    # ← skip empty/whitespace values
            clean[key] = cleaned_value
    
    if dropped:
        logger.warning("unknown filter key(s) dropped: %s", dropped)
    
    return clean
```

**Why a whitelist (`ALLOWED_FILTER_KEYS`):** Without it, a malicious or buggy client could inject arbitrary Milvus expression fields (e.g. `{"id > 0 OR 1=1": ""}`) which would bypass filters or cause errors. The whitelist ensures only known, safe field names reach the expression builder.

**Why strip whitespace and check for empty:** The UI text inputs might have trailing spaces. A filter of `"  "` (spaces only) would produce `industry == "  "` which would match nothing and confuse users. Stripping and skipping empty values means "empty = no filter".

---

### 4.7 `retriever_rag.py` — Finding Relevant Chunks at Query Time

**Why this file exists:** At query time (when a user asks a question), the question text must be embedded using the same model that was used to embed the chunks (otherwise the vectors live in different spaces and similarity scores are meaningless). This file handles query embedding and delegates to Milvus for the actual search.

#### `embed_text(text)` — single text embedding

```python
def embed_text(text: str) -> list[float]:
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text.replace("\n", " "),   # ← same pre-processing as embedder_rag
    )
    embedding = response.data[0].embedding
    return embedding
```

**Why the same model and pre-processing as `embedder_rag.py`:** If chunks were embedded with `text-embedding-3-large` after replacing newlines, the query must be embedded the same way. If you used a different model or pre-processing, the cosine similarity scores between query and chunks would be meaningless.

#### `retrieve(query, top_k, filters)` — the public retrieval function

```python
def retrieve(query: str, top_k: int = 5, filters: Optional[dict] = None) -> list[dict]:
    from rag.retrieval.milvus_client_rag import hybrid_search_chunks
    
    query_embedding = embed_text(query)
    
    chunks = hybrid_search_chunks(
        query_embedding=query_embedding,
        query_text=query,
        top_k=top_k,
        filters=filters,
    )
    return chunks
```

**Why import `hybrid_search_chunks` inside the function:** Circular import avoidance. `milvus_client_rag` is a heavy module (it connects to the database). Importing it at the module level would slow startup and make testing harder. The local import means Milvus is only connected when retrieval is actually needed.

#### `format_context_for_prompt(chunks)` — building the LLM context block

```python
def format_context_for_prompt(chunks: list[dict]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        location = f"{title} → {section}" if section and section != title else title
        meta_parts = [doc_type, f"v{version}", f"tags: ...", f"score: {score}"]
        
        lines.append(f"[{i}] {location}  ({' '.join(meta_parts)})")
        lines.append(chunk.get("chunk_text", ""))
        lines.append("")   # ← blank line between chunks
    
    return "\n".join(lines)
```

**Why number chunks `[1]`, `[2]`, etc.:** The LLM is instructed to cite sources with `[N]`. These numbers in the context block match the citation indices in the answer. When the user sees `[3]` in the answer, they can open source 3 in the UI.

**Why include score in the context:** The LLM can see if a chunk has a low score (e.g. 0.18) and adjust its confidence accordingly. It might say "based on the closest match I found (score 0.21)..." rather than stating the answer with full confidence.

---

### 4.8 `adaptive_router_rag.py` — Classifying What the User Wants

**Why this file exists:** Not all questions are the same. "What is the SLA for P1 incidents?" is a direct QA question (retrieve 5 chunks). "Compare version 1 and version 2 of the access policy" needs 10 chunks and a structured comparison response. "Summarise the vendor handbook" needs 10 chunks and a summary structure. Having the LLM classify intent first allows the pipeline to adjust retrieval size and response format.

#### The strategy map

```python
_STRATEGY_MAP = {
    "QA":        {"top_k": 5,  "llm_mode": "qa"},
    "COMPARE":   {"top_k": 10, "llm_mode": "compare"},
    "SUMMARIZE": {"top_k": 10, "llm_mode": "summarize"},
    "SEARCH":    {"top_k": 8,  "llm_mode": "qa"},
    "GREETING":  {"top_k": 0,  "llm_mode": "qa"},
}
```

**Why different `top_k` values:** QA needs 5 chunks — too many and the context gets diluted. Comparison and summarisation need 10 — they synthesise across more of the document. GREETING needs 0 — it bypasses the pipeline entirely. Search needs 8 — more than QA because the user wants breadth, not depth.

#### The classification prompt

The prompt uses rich few-shot examples with informal language, typos, and non-standard phrasing. This is deliberate — users in production write messages like "wot r u" or "heyy". The prompt covers these cases so the LLM generalises rather than failing on unexpected input.

**Key distinction — GREETING vs QA:**
```
GREETING examples: "hi", "hello", "who are you", "what can you do"
NOT GREETING: "what did I ask before?", "can you elaborate?", follow-up questions
```
This is important because "what did I ask before?" looks conversational but is an information-seeking QA question about conversation history. A wrong classification here would bypass retrieval for a question that should retrieve from session history.

#### The LangGraph graph

```python
class RouterState(TypedDict):
    query: str
    mode:  str

def _classify_node(state: RouterState) -> RouterState:
    # ... LLM call
    return {"query": query, "mode": raw_mode}

graph = StateGraph(RouterState)
graph.add_node("classify", _classify_node)
graph.set_entry_point("classify")
graph.add_edge("classify", END)
_router_graph = graph.compile()
```

**Why LangGraph for a single node:** This looks like overkill for one node. The reason is **future composability**. LangGraph graphs can be composed — this router graph can be embedded as a sub-graph in a larger graph (e.g. one that includes memory nodes or feedback loops) without rewriting the classification logic.

**Two-attempt fallback:**
```python
for attempt, prompt in enumerate([_CLASSIFICATION_PROMPT, _FALLBACK_CLASSIFICATION_PROMPT], start=1):
    try:
        response = (prompt | llm).invoke({"query": query})
        raw_mode = response.content.strip().upper()
        if raw_mode in VALID_MODES:
            return {"query": query, "mode": raw_mode}
    except Exception:
        pass

# Both failed — default to QA
return {"query": query, "mode": "QA"}
```

**Why two different prompts instead of retry:** A transient API error would be caught by the outer try/except and trigger the second prompt. A consistent model quirk (e.g. the LLM returning "QA." with a period) might succeed on the second prompt if it's written differently enough to avoid the same pattern. The fallback prompt is intentionally written differently to reduce correlation between failures.

---

### 4.9 `corrective_rag_rag.py` — Fixing Weak Retrievals Automatically

**Why this file exists:** Users often write vague, typo-filled, or context-dependent queries. "what about that leave policy from before" is a terrible search query but a perfectly natural conversational message. Without correction, retrieval fails silently and the LLM either hallucinates or gives an out-of-scope response. This module detects weak retrievals and automatically rewrites the query to be more specific.

#### The threshold

```python
RELEVANCE_THRESHOLD = 0.65
```

If the average COSINE score of the first retrieval pass is below 0.65, the query is considered weak and rewriting is triggered. Above 0.65 means good retrieval — no rewrite needed.

**Why 0.65 and not some other value:** This was chosen empirically. Below 0.65 means the retrieved chunks are not confidently relevant. Above 0.65 means at least the top results are a good match. The value is tunable.

#### The state

```python
class CorrectiveRAGState(TypedDict):
    query:           str
    retrieve_fn:     Callable      # ← injected function, not a direct import
    top_k:           int
    filters:         Optional[dict]
    session_history: list[dict]    # ← last 6 turns, user messages only
    chunks1:         list[dict]    # first-pass results
    score1:          float         # avg score of first pass
    rewritten:       str           # LLM-rewritten query
    chunks2:         list[dict]    # second-pass results
    score2:          float         # avg score of second pass
    final_chunks:    list[dict]    # winner
    final_query:     str           # query that produced the winner
```

**Why `retrieve_fn` is injected instead of imported:** The pipeline calls corrective RAG with `pool_size = top_k * 2` — a larger candidate set than the final top_k. If corrective RAG imported `retrieve` directly and called it with its own `top_k`, the pool size optimisation would be impossible. Injecting a callable closure preserves this control.

**Why `session_history` in state:** The rewrite node uses conversation history to resolve ambiguous references. "the previous one" in a query cannot be expanded without knowing what was discussed before. Only user turns are included — prior AI answers carried stale topic content that caused the rewriter to pull queries toward old topics.

#### The graph

```
retrieve → score → route:
    score ≥ 0.65 → done (return first-pass results)
    score < 0.65 → rewrite → retrieve2 → pick_best
```

#### `_node_rewrite` — the query rewriter

```python
def _node_rewrite(state):
    query = state["query"]
    session_history = state.get("session_history") or []
    
    # Build history string from last 6 user turns only
    history_lines = []
    for turn in session_history[-6:]:
        role = turn.get("role", "user").capitalize()
        if role != "User":
            continue   # ← AI turns excluded
        text = turn.get("content", "")[:200].replace("\n", " ")
        history_lines.append(f"{role}: {text}")
    history_str = "\n".join(history_lines) or "(no prior conversation)"
    
    chain = _REWRITE_PROMPT | _get_llm()
    response = chain.invoke({"query": query, "history": history_str})
    rewritten = response.content.strip().strip('"').strip("'")
    
    return {"rewritten": rewritten}
```

**Why strip quotes:** LLMs sometimes return the rewritten query wrapped in quotes (`"bereavement leave entitlement"`). `strip('"').strip("'")` removes both double and single quotes so the rewritten query is clean.

**Why only user turns in history:** This was a bug fix. The original code included AI turns in the history string. When a user switched topics mid-session, the rewriter saw the previous AI answer (about "Product Management documents") and rewrote the new query to be about Product Management — completely wrong. With only user turns, the rewriter has enough context to resolve pronouns without being misled by prior answers.

#### `_node_pick_best` — choosing the better retrieval

```python
def _node_pick_best(state):
    score1 = state["score1"]
    score2 = state["score2"]
    
    if score2 >= score1:
        return {"final_chunks": state["chunks2"], "final_query": state["rewritten"]}
    
    return {"final_chunks": state["chunks1"], "final_query": state["query"]}
```

**Why compare and pick instead of always using the rewrite:** The rewrite might make things worse. If the original query was about "leave entitlements" and the rewrite became "employee leave policies 2024", the rewrite might retrieve a different set of chunks with a lower average score. Comparing scores ensures the rewrite never degrades results.

---

### 4.10 `reranker_rag.py` — The Reranker Shim

**Why this file exists:** In a full RAG system, a reranker (typically a cross-encoder neural network) re-scores the top-k results after retrieval for higher precision. We don't have one yet — Milvus COSINE already orders results well enough. But the pipeline imports from `reranker_rag`, so the file must exist. It's a transparent pass-through today and a swap-in point for a cross-encoder tomorrow.

```python
def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    result = chunks[:top_k]   # ← just enforce the top_k cap
    return result
```

**Why keep `query` as a parameter even though it's not used:** When a cross-encoder is added, it will need the query to score (query, chunk) pairs. Keeping the parameter in the signature means no callers need to change when the reranker is upgraded.

---

### 4.11 `prompts_rag.py` — Every LLM Prompt in One Place

**Why this file exists:** Prompts are the most frequently tuned part of a RAG system. If they're scattered across multiple files, changing one prompt requires hunting through the codebase. Centralising them means one file to edit, one place to audit.

#### `RAG_SYSTEM_PROMPT` — the default QA prompt

```python
RAG_SYSTEM_PROMPT = """You are Citter, a helpful research assistant...

CRITICAL — source of truth rule:
The numbered context chunks in the current message are the ONLY authoritative
source for your answer. Ignore anything stated in earlier conversation turns
that contradicts or is absent from those chunks.
...
"""
```

**Why the "CRITICAL — source of truth rule" section:** This was added to fix a bug. When the user switched topics mid-conversation, the LLM had prior AI turns in its message history that contained answers from the old topic. Without an explicit instruction to distrust history, the LLM would anchor on its own prior output rather than the freshly retrieved chunks. The explicit override rule breaks this anchoring.

#### `REFINE_QUERY_PROMPT` — the corrective RAG rewrite prompt

```python
REFINE_QUERY_PROMPT = """The following search query returned weak or irrelevant results...

Use the recent conversation history below to resolve ambiguous pronouns...

Return ONLY the rewritten query — no explanation, no quotes, no preamble.

Recent conversation history (most recent last):
{history}

Original query: {query}
"""
```

**Why "Return ONLY the rewritten query":** LLMs often prepend explanations like "Here is the rewritten query: ...". That preamble would be included in the search query, breaking retrieval. The instruction eliminates it.

#### `OUT_OF_SCOPE_SCORE_THRESHOLD = 0.30`

```python
OUT_OF_SCOPE_SCORE_THRESHOLD = 0.30
```

If the average COSINE similarity score of the top retrieved chunks is below 0.30, the topic is not covered in the document library. The pipeline returns `OUT_OF_SCOPE_RESPONSE` without calling the LLM. This prevents the LLM from hallucinating answers to questions the library doesn't cover.

#### `GREETING_RESPONSE` — hardcoded identity card

```python
GREETING_RESPONSE = """Hi! I'm **Citter**, your document library research assistant. 🤖
Here's what I can help you with:
...
"""
```

**Why hardcoded instead of LLM-generated:** Greetings are frequent and cheap to handle. Calling the LLM for every "hi" wastes tokens and adds latency. The hardcoded card is instant and consistent.

---

### 4.12 `redis_cache_rag.py` — Caching and Session Memory

**Why this file exists:** Two problems without Redis: (1) the same query hits Azure OpenAI and Milvus on every request — expensive and slow. (2) The pipeline has no memory of previous turns in a conversation — each message is treated as standalone. Redis solves both.

#### The async client singleton

```python
_client = None   # ← module-level singleton

async def _get_client():
    global _client
    if _client is not None:
        return _client      # ← return cached client on subsequent calls
    try:
        import redis.asyncio as aioredis
        c = aioredis.from_url(REDIS_URL, decode_responses=True, socket_connect_timeout=2)
        await c.ping()      # ← verify connection works
        _client = c
        return _client
    except Exception as err:
        logger.warning("Redis unavailable — all cache operations are no-ops")
        return None         # ← return None, not raise
```

**Why `socket_connect_timeout=2`:** Without a timeout, a missing Redis server would hang for ~30 seconds before failing. 2 seconds is fast enough to detect an absent server without blocking the request.

**Why return `None` instead of raising:** Every cache operation checks `if not client: return None/[]`. This means Redis being down is a graceful degradation — the pipeline continues to work without caching, just slower and stateless. Raising an exception would crash every request when Redis is down, which would be unacceptable in production.

#### Three namespaces

```python
"rag:retrieval:{sha256[:16]}"    # → cached chunks list, TTL 10 minutes
"rag:session:{session_id}"       # → chat history list, TTL 24 hours
"rag:notion:reads"               # → integer counter, TTL 60 seconds
```

**Why SHA-256 for retrieval key:** The key must be deterministic (same query + filters always maps to the same key) but compact. `json.dumps({"q": query, "f": filters}, sort_keys=True)` produces a canonical string regardless of dict key ordering. SHA-256 of that string, taking the first 16 hex chars, gives a 16-char key with astronomically low collision probability.

**Why `sort_keys=True`:** Without it, `{"f": {}, "q": "hello"}` and `{"q": "hello", "f": {}}` would hash differently even though they're the same query. Sorting keys makes the JSON representation canonical.

#### Session history design

```python
async def get_session_history(session_id: str) -> list[dict]:
    raw = await client.get(f"rag:session:{session_id}")
    return json.loads(raw) if raw else []

async def set_session_history(session_id: str, history: list[dict]) -> None:
    await client.set(f"rag:session:{session_id}", json.dumps(history), ex=TTL_SESSION)
```

**Why 24-hour TTL:** A user might continue a conversation the next morning. 24 hours is long enough that a session feels persistent within a working day but doesn't accumulate forever.

**Why store full history:** The history is loaded on each request and passed to `run_rag_pipeline()` which uses the last 6 turns for LLM context. Storing the full history (not just 6 turns) means the corrective RAG rewriter can use more history for context resolution even if the pipeline only uses 6 for the LLM call.

#### Notion rate limiter

```python
async def check_notion_rate_limit() -> bool:
    key = "rag:notion:reads"
    count = await client.incr(key)     # ← atomic increment
    if count == 1:
        await client.expire(key, 60)   # ← set 60s expiry on first increment
    return count <= RATE_LIMIT_RPM     # 100 reads per minute
```

**Why `incr` then `expire` instead of `set`:** Redis `INCR` is atomic — even with concurrent requests, the counter increments exactly once per call. Setting the expiry only when count == 1 means the TTL is set once (on the first call in a new minute) and then the counter counts up until the key expires and resets.

---

### 4.13 `pipeline_rag.py` — The Main Orchestrator

**Why this file exists:** It chains every component — filters, router, corrective retrieval, reranker, context formatting, LLM call, citation building — into one function that FastAPI calls.

#### The lazy LLM client

```python
_llm: Optional[AzureChatOpenAI] = None

def _get_llm() -> AzureChatOpenAI:
    global _llm
    if _llm is None:
        _llm = AzureChatOpenAI(
            azure_deployment=..., azure_endpoint=..., api_key=...,
            temperature=0.2,      # ← low temperature for factual answers
            max_tokens=1024,      # ← enough for a thorough answer
        )
    return _llm
```

**Why `temperature=0.2`:** Higher temperature makes the LLM more creative but also more likely to deviate from the retrieved context. 0.2 keeps responses grounded while allowing natural phrasing.

**Why `max_tokens=1024`:** Most answers fit in 512 tokens. 1024 provides headroom for long comparison or summarisation answers without being wasteful.

#### `run_rag_pipeline()` — step by step

**Step 1: Build filters**
```python
filters = build_filters(raw_filters or {})
```
Sanitises the filter dict before it reaches Milvus. Empty/invalid filters become `{}`.

**Step 2: Classify query mode**
```python
mode = classify_query(query)
params = get_retrieval_params(mode)
top_k = params["top_k"]
llm_mode = params["llm_mode"]
```
Runs the LangGraph router. Returns one of: QA, COMPARE, SUMMARIZE, SEARCH, GREETING.

**Step 2a: GREETING short-circuit**
```python
if mode == "GREETING":
    return {
        "answer":    GREETING_RESPONSE,
        "citations": [],
        "chunks":    [],
        "mode":      "GREETING",
        "avg_score": 0.0,
    }
```
If it's a greeting, skip everything — return the hardcoded response immediately. No embeddings, no Milvus, no LLM call.

**Step 3: Corrective retrieval**
```python
pool_size = top_k * 2

def _retrieve_fn(q: str, k: int, f: dict | None) -> list[dict]:
    return retrieve(q, top_k=pool_size, filters=f)   # ← always use pool_size, not k

chunks, rewritten_query = corrective_retrieve(
    query=query,
    retrieve_fn=_retrieve_fn,
    top_k=top_k,
    filters=filters or None,
    session_history=session_history or [],
)
```

**Why `pool_size = top_k * 2`:** The corrective RAG graph retrieves candidates for potential rewriting. Having 2× the final top_k gives it a richer set to compare scores against. The `k` parameter passed to `retrieve_fn` by the graph is ignored — the closure always uses `pool_size`.

**Step 4: Rerank (top_k cap)**
```python
chunks = rerank(query=rewritten_query, chunks=chunks, top_k=top_k)
final_score = avg_score(chunks)
```
Enforces the final top_k cap. `avg_score` computes the mean COSINE score.

**Step 4b: Score gate**
```python
if final_score < OUT_OF_SCOPE_SCORE_THRESHOLD:
    return {"answer": OUT_OF_SCOPE_RESPONSE, "citations": [], ...}
```
If the best chunks are weak, the topic is not covered. Return the out-of-scope response without calling the LLM. This is the signal that StateCase uses to detect "cannot answer" situations.

**Step 5: Build context**
```python
context = format_context_for_prompt(chunks)
```
Formats retrieved chunks into the numbered `[N]` block format.

**Step 6: LLM call with history injection**
```python
system_prompt = SYSTEM_PROMPT_BY_MODE.get(llm_mode, RAG_SYSTEM_PROMPT)
lc_messages = [SystemMessage(content=system_prompt)]

if session_history:
    for turn in session_history[-6:]:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role == "user":
            lc_messages.append(HumanMessage(content=content))
        else:
            lc_messages.append(AIMessage(
                content="[Previous answer — see retrieved context below for current facts.]"
            ))

lc_messages.append(HumanMessage(content=f"Retrieved context...\n{context}\n\nQuestion: {rewritten_query}"))
```

**Why replace AI message content with a placeholder:** This was the context contamination bug fix. The original code injected raw AI message content (e.g. a previous answer about "Product Management bereavement leave"). When the user switched topics, the LLM anchored on its prior output rather than the newly retrieved chunks. Replacing AI turns with a neutral placeholder preserves the turn-pair structure (so the LLM can follow references like "what about the other type?") without injecting stale factual content.

**Why label the user content as "authoritative source":** The phrasing `"Retrieved context (authoritative source — answer from this only)"` reinforces the system prompt's source-of-truth rule. The LLM sees this in the user turn, not just the system prompt, making it harder to ignore.

**Step 7: Build citation list**
```python
citations = []
for i, chunk in enumerate(chunks, start=1):
    citations.append({
        "index":      i,
        "title":      chunk.get("title", ""),
        "section":    chunk.get("section", ""),
        "doc_type":   chunk.get("doc_type", ""),
        "industry":   chunk.get("industry", ""),
        "version":    chunk.get("version", ""),
        "tags":       chunk.get("tags", []),
        "page_id":    chunk.get("page_id", ""),
        "score":      chunk.get("score", 0.0),
        "chunk_text": chunk.get("chunk_text", ""),
    })
```

The `index` field matches the `[N]` numbers in the LLM's answer. `chunk_text` is included so the RAGAS evaluation module can access it for faithfulness scoring without a secondary lookup.

---

### 4.14 `ragas_runner_rag.py` — Measuring RAG Quality

**Why this file exists:** You cannot know if your RAG system is actually good without measuring it. RAGAS (Retrieval-Augmented Generation Assessment) is a framework that measures four metrics:
- **Faithfulness:** Does the answer only use information from the retrieved context? (Detects hallucination)
- **Answer Relevancy:** Is the answer relevant to the question?
- **Context Precision:** Are the retrieved chunks actually useful for the answer?
- **Context Recall:** Does the retrieved context cover what the ground truth answer contains?

RAGAS uses an LLM as a judge — we configure it to use Azure OpenAI rather than its default OpenAI to keep all LLM calls within the same Azure deployment.

#### Building Azure clients for RAGAS

```python
def _build_azure_llm():
    from langchain_openai import AzureChatOpenAI
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_LLM_DEPLOYMENT_41_MINI", "gpt-4.1-mini"),
        ...
    )

def _build_azure_embeddings():
    from langchain_openai import AzureOpenAIEmbeddings
    return AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_EMB_DEPLOYMENT", "text-embedding-3-large"),
        ...
    )
```

RAGAS uses LangChain under the hood. Passing `AzureChatOpenAI` and `AzureOpenAIEmbeddings` instances to each metric tells RAGAS to use Azure instead of looking for `OPENAI_API_KEY`.

#### Dataset column names

```python
data = {
    "user_input":         questions,       # ← RAGAS expects this exact name
    "response":           answers,
    "retrieved_contexts": contexts,
    "reference":          ground_truths,
}
dataset = Dataset.from_dict(data)
```

**Why these exact column names:** RAGAS v0.2+ changed its column name expectations. Using the old names (`question`, `answer`, `contexts`, `ground_truth`) would cause RAGAS to fail silently or raise cryptic errors.

---

### 4.15 `main_rag.py` — The FastAPI Backend

**Why this file exists:** The pipeline logic (Python functions) must be exposed over HTTP so the Streamlit UI (a separate process) can call it. FastAPI provides the HTTP server.

#### Lifespan context manager

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 CiteRagLab API starting up on port 8001")
    yield
    logger.info("🛑 Shutting down — closing Redis connection")
    await close_rag_redis()
```

**Why `asynccontextmanager`:** FastAPI's lifespan parameter replaces the old `startup`/`shutdown` event handlers. The `yield` separates startup code (before yield) from shutdown code (after yield). `close_rag_redis()` gracefully flushes the async Redis connection on shutdown.

#### Mounting the StateCase router

```python
from rag.api.statecase_routes_rag import router as statecase_router
app.include_router(statecase_router)
```

`include_router` adds all routes from the StateCase router (all `/statecase/*` endpoints) to the main app. This is cleaner than defining all routes in one file.

#### The `/chat` route

```python
@app.post("/chat")
async def chat(req: ChatRequest):
    session_history = await get_session_history(req.session_id)
    
    result = run_rag_pipeline(
        query=req.message,
        session_history=session_history,
        raw_filters=req.filters,
    )
    
    await set_retrieval_cache(req.message, req.filters, result["chunks"])
    
    session_history.append({"role": "user",      "content": req.message})
    session_history.append({"role": "assistant",  "content": result["answer"]})
    await set_session_history(req.session_id, session_history)
    
    return {
        "session_id": req.session_id,
        "answer":     result["answer"],
        "citations":  result["citations"],
        "mode":       result["mode"],
        "avg_score":  result["avg_score"],
        "rewritten":  result["rewritten"],
    }
```

**Why load history before pipeline and save after:** The pipeline needs history to understand follow-up questions. The updated history (with the new turn appended) must be saved so the next request can load it.

**Why cache chunks separately from session history:** Retrieval results are cached by `(query, filters)` for 10 minutes — if the same query is asked again in a different session, the Milvus lookup is skipped. Session history is stored separately per session with a 24-hour TTL — it's about conversation state, not query results.

---

### 4.16 `api_helpers_rag.py` — UI-to-Backend HTTP Wrappers

**Why this file exists:** The Streamlit UI is a separate Python process. It cannot import from `rag/` directly — it calls the FastAPI backend over HTTP. This file provides one function per endpoint, hiding the `requests` library calls.

```python
def call_chat(session_id, message, filters=None, base_url=RAG_API_URL) -> dict | None:
    try:
        response = requests.post(
            f"{base_url}/chat",
            json={"session_id": session_id, "message": message, "filters": filters or {}},
            timeout=90,   # ← long timeout for LLM calls
        )
        response.raise_for_status()
        return response.json()
    except Exception as error:
        logger.error("-> /chat FAILED: %s", error)
        return None   # ← return None, not raise
```

**Why `timeout=90`:** LLM API calls can take 10-30 seconds. Corrective RAG adds another 10-20 seconds if a rewrite is triggered. 90 seconds covers the worst case without hanging the UI indefinitely.

**Why return `None` instead of raising:** The UI checks `if api_response:` and shows a friendly error message if None. Raising an exception would require the UI to wrap every call in try/except.

---

### 4.17 `cite_rag_lab_ui_rag.py` — The Streamlit UI

**Why this file exists:** The FastAPI backend is an API — it has no user interface. Streamlit provides a web UI with chat bubbles, forms, and interactive components. This file renders the entire UI.

#### Session state initialisation

```python
def _init_crl_session_state():
    if "crl_sessions" not in st.session_state:
        st.session_state.crl_sessions = {}
    
    if not st.session_state.crl_sessions:
        default_id = str(uuid.uuid4())[:8]
        st.session_state.crl_sessions[default_id] = {"title": "Chat 1", "messages": []}
        st.session_state.crl_active_session_id = default_id
```

**Why check `not st.session_state.crl_sessions`:** This guard ensures the default session is created exactly once, on first load. Streamlit reruns the entire script on every user interaction. Without the guard, a new default session would be created on every rerun.

**Why `crl_filter_version_counter`:**
```python
if "crl_filter_version_counter" not in st.session_state:
    st.session_state.crl_filter_version_counter = 0
```
When the user clicks "Clear Filters", we increment this counter. The filter `st.text_input` widgets use it in their `key` parameter: `key=f"crl_filter_industry_{_fv}"`. Changing the key forces Streamlit to create a new widget instance with no cached value — the only reliable way to programmatically reset a text input.

#### Duplicate submission guard

```python
_submit_key = (active_session_id, len(_get_active_messages()), user_query)
if st.session_state.get("crl_last_submit_key") == _submit_key:
    return
st.session_state["crl_last_submit_key"] = _submit_key
```

Streamlit reruns the script after every interaction, including after the chat input is submitted. Without this guard, the same message would be submitted twice. The key includes the message count so the same text can legitimately be sent again in a later turn.

#### Agent mode toggle

```python
st.session_state.sc_agent_mode = st.toggle("🎫 Auto-ticketing", ...)
```

In the chat tab:
```python
if agent_mode:
    api_response = call_statecase_chat(...)   # StateCase agent with auto-ticketing
else:
    api_response = call_chat(...)             # bare RAG pipeline
```

**Why a toggle instead of always using the agent:** The agent adds latency (extra LLM calls for tool selection). Users who just want fast document answers should have the option to bypass the agent overhead.

#### Ticket badge rendering

```python
if ticket_info and role == "assistant":
    t_id = ticket_info.get("ticket_id", "")
    t_url = ticket_info.get("url", "")
    badge_text = f"🎫 **{t_id}** · Priority: {t_pri} · Status: Open"
    if t_url:
        badge_text += f" · [View in Notion]({t_url})"
    st.info(badge_text)
```

When the StateCase agent creates a ticket, the response includes a `ticket_created` dict. The UI renders it as an `st.info` badge inline under the assistant message.

---

## 5. StateCase — The Ticketing Agent

### 5.1 `statecase_notion_rag.py` — The Notion Ticket Database Client

**Why this file exists:** StateCase tickets live in a separate Notion database from the document library. All CRUD operations for tickets (create, read, update, list, search) are in this one file. The FastAPI routes and LangChain tools call this file — they never call the Notion API directly.

#### The critical schema quirks

When the Notion database was inspected, two properties behaved unexpectedly:

**`Question` is the title column** (not `Ticket ID` as the column order might suggest):
```python
# WRONG — causes HTTP 400: "Question is expected to be title"
"Ticket ID": _title_prop(ticket_id),
"Question":  _rich_text_prop(question),

# CORRECT
"Question":  _title_prop(question),   # title column = the primary page name
"Ticket ID": _rich_text_prop(ticket_id),
```
Every Notion page has exactly one "title" property — it's the column that determines the page name. In this database, `Question` is that column.

**`Status` is a Notion Status type, not a Select**:
```python
# WRONG — causes HTTP 400: "database property status does not match filter select"
"Status": {"select": {"name": "Open"}}

# CORRECT
"Status": {"status": {"name": "Not started"}}
```
Notion has two similar-looking column types: `Select` (pick from a list of options) and `Status` (a special type with built-in categorization — "Not started", "In progress", "Done"). They look identical in the UI but use different API shapes. The same mismatch applies to filter queries.

#### Property builders

```python
def _title_prop(text: str) -> dict:
    return {"title": [{"text": {"content": text[:2000]}}]}

def _rich_text_prop(text: str) -> dict:
    return {"rich_text": [{"text": {"content": str(text)[:2000]}}]}

def _select_prop(value: str) -> dict:
    return {"select": {"name": value}}

def _status_prop(value: str) -> dict:
    return {"status": {"name": value}}
```

**Why `[:2000]`:** Notion's API rejects rich_text content longer than 2000 characters. Truncating to 2000 prevents HTTP 400 errors on very long questions.

#### Ticket ID generation

```python
def _get_next_ticket_id() -> str:
    try:
        import redis
        r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"), socket_connect_timeout=2)
        n = r.incr("statecase:ticket_counter")   # ← atomic increment
        return f"SC-{int(n):04d}"
    except Exception:
        ts = int(datetime.now(timezone.utc).timestamp()) % 100000
        return f"SC-{ts:05d}"
```

**Why Redis INCR:** Multiple users might create tickets concurrently. `INCR` is an atomic Redis operation — it increments and returns the new value in a single step with no race condition. Two concurrent calls to `INCR` always get different values.

**Why timestamp fallback:** If Redis is down, we still need a unique ID. A Unix timestamp modulo 100000 gives a 5-digit number that is unique within a ~27-hour window. It's not sequential but it's functional.

#### Deduplication

```python
def _dedup_key(session_id: str, question: str) -> str:
    raw = f"{session_id}::{question.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
```

Before creating a ticket, `create_ticket()` calls `_find_by_dedup(dedup)` which searches Notion for any ticket whose `User Info` field contains `"dedup:{key}"`. If found, it returns the existing ticket instead of creating a new one.

**Why this matters:** A user might click "yes create ticket" twice, or the network might retry a failed request. Without dedup, each click would create a duplicate ticket. The dedup key is stored in `User Info` as plain text so it survives across HTTP requests.

**Why `strip().lower()` on the question:** Trailing whitespace or capitalisation differences between "what is the SLA?" and "What is the SLA? " should not produce different dedup keys.

#### `find_ticket_by_title()` — resolving names to UUIDs

```python
def find_ticket_by_title(search_term: str) -> dict | None:
    for filter_body in [
        {"property": "Ticket ID", "rich_text": {"contains": search_term}},
        {"property": "Question",  "title":     {"contains": search_term}},
    ]:
        # query Notion, return first match
```

**Why this function exists:** When the LLM calls the `update_support_ticket` tool, it needs to pass a Notion page UUID. But the LLM only knows the ticket title ("Turnover of Turabit") or SC-ID ("SC-0012") from the conversation context. Passing the title as the page ID causes Notion to return HTTP 400: "path.page_id should be a valid uuid".

This function queries Notion twice (once for SC-ID match, once for title match) to resolve the human-readable identifier to the actual UUID that Notion requires.

---

### 5.2 `statecase_tools_rag.py` — LangChain Tools

**Why this file exists:** The original StateCase agent had hard-coded routing — an intent classifier node that branched to different function call nodes. The LLM had no say in what happened. This is brittle: if a new action type is needed, you must add a new intent, a new routing branch, and a new node.

LangChain `@tool` changes this design fundamentally. You describe each capability as a function with a docstring and typed parameters. The LLM receives a Pydantic schema describing what each tool does and what arguments it takes. The LLM decides which tool to call and with what arguments, based on the user message and conversation context. This is true agentic behaviour — the LLM is the router.

#### How `@tool` works

```python
from langchain_core.tools import tool

@tool
def rag_search(query: str, session_id: str, industry: str = "") -> dict:
    """
    Search the document library and return an answer with cited sources.
    
    Args:
        query:      The user's question...
        session_id: The current chat session ID...
        industry:   Optional filter...
    
    Returns a dict with: answer, citations, avg_score, answerable...
    """
    # ... implementation
```

The `@tool` decorator:
1. Reads the function signature and generates a Pydantic model from the parameter types and defaults
2. Reads the docstring and uses it as the tool description
3. When `llm.bind_tools([rag_search, ...])` is called, serialises these schemas into the Azure OpenAI function-calling format

The LLM sees: "there is a function called `rag_search` that takes `query` (required string), `session_id` (required string), and `industry` (optional string, default empty). Call it when the user asks a question about the document library."

#### `rag_search` tool

```python
@tool
def rag_search(query: str, session_id: str, industry: str = "", doc_type: str = "", version: str = "") -> dict:
    # ... calls run_rag_pipeline internally
    answerable = avg_score >= 0.30
    return {
        "answer":     result.get("answer", ""),
        "citations":  result.get("citations", []),
        "avg_score":  avg_score,
        "mode":       result.get("mode", "QA"),
        "answerable": answerable,   # ← the key the agent uses to decide on ticketing
    }
```

**Why `answerable` is a separate boolean:** The system prompt instructs the LLM: "if `answerable=False`, offer to create a ticket". Having an explicit boolean is clearer than asking the LLM to interpret `avg_score < 0.30`. The LLM doesn't need to do math — it just reads a boolean.

#### `update_support_ticket` tool — the UUID resolution fix

```python
@tool
def update_support_ticket(notion_page_id: str, status: str = "", ...) -> dict:
    from rag.pipeline.statecase_notion_rag import update_ticket, find_ticket_by_title
    import re
    
    _UUID_RE = re.compile(
        r"^[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}$",
        re.IGNORECASE,
    )
    
    resolved_page_id = notion_page_id
    if not _UUID_RE.match(notion_page_id.replace("-", "")):
        # Not a UUID — try to find the ticket by title or SC-ID
        ticket = find_ticket_by_title(notion_page_id)
        if not ticket:
            return {"success": False, "error": f"Could not find a ticket matching '{notion_page_id}'"}
        resolved_page_id = ticket["notion_page_id"]
    
    # Now update using the real UUID
    ticket = update_ticket(notion_page_id=resolved_page_id, ...)
    return {**ticket, "success": True}
```

**Why the regex:** A UUID is 32 hex digits in the format `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`. The regex validates this pattern. If `notion_page_id` doesn't match, it's a title or SC-ID, not a UUID, and we need to look it up.

**Why dashes optional in the regex:** Notion sometimes returns UUIDs without dashes in some contexts. The regex handles both `"abc123def456..."` and `"abc123de-f456-..."`.

---

### 5.3 `statecase_agent_rag.py` — The LangGraph Tool-Calling Agent

**Why this file exists:** The five tools exist, but something must:
1. Load session memory from Redis
2. Build the LLM messages (system prompt + history)
3. Call the LLM with tools bound
4. Dispatch whatever tools the LLM chose
5. Update memory based on tool results
6. Loop back to the LLM if more tool calls are needed
7. Save memory and session history back to Redis

This file is the LangGraph graph that orchestrates all of that.

#### The state

```python
class StateCaseAgentState(TypedDict):
    session_id:          str
    raw_filters:         Optional[dict]
    ticket_priority:     str
    ticket_owner:        str
    messages:            Annotated[list[BaseMessage], add_messages]  # ← special reducer
    memory:              dict     # persisted to Redis between requests
    trace_id:            str
    final_response:      str      # extracted in save_mem
    final_citations:     list
    final_pipeline_meta: dict
    final_ticket:        Optional[dict]
```

**Why `Annotated[list[BaseMessage], add_messages]`:** LangGraph state is normally replaced on each update (`state["x"] = new_value`). For `messages`, we want to append, not replace. The `add_messages` reducer tells LangGraph: when a node returns `{"messages": [new_msg]}`, append `new_msg` to the existing list rather than replacing it. This is how the agent loop accumulates tool call/result pairs across multiple node executions.

#### The graph topology

```
load_mem → agent ──── has tool_calls → tools → update_mem → agent (loop)
                └──── no tool_calls → save_mem → END
```

The agent loops between `agent → tools → update_mem → agent` until the LLM produces a response with no tool calls. Then it exits to `save_mem → END`.

#### `_node_load_mem` — hydrating state from Redis

```python
async def _node_load_mem(state):
    redis = await _get_redis()
    raw = await redis.get(f"statecase:memory:{session_id}")
    memory = json.loads(raw) if raw else {}
    pending = memory.get("pending_ticket_context")
    return {"memory": memory, "trace_id": trace_id, "pending_ticket_context": pending}
```

**Why surface `pending_ticket_context` at top level:** The system prompt builder reads `state["memory"]`. But the system prompt needs to include the pending context so the LLM knows to look for a yes/no confirmation. Surfacing it at the top level makes it easy to check without dict lookups.

#### `_node_agent` — the core LLM node

```python
def _node_agent(state):
    memory = state.get("memory", {})
    messages = state.get("messages", [])
    
    system_msg = SystemMessage(
        content=_SYSTEM_PROMPT.format(memory_context=_build_memory_context(memory))
    )
    
    response = _get_llm_with_tools().invoke([system_msg] + messages)
    
    # Track tool calls in memory for update_mem to use
    updated_memory = dict(memory)
    for tc in response.tool_calls:
        if tc["name"] == "rag_search":
            updated_memory["last_question"] = tc["args"].get("query", "")
        if tc["name"] == "create_support_ticket":
            updated_memory.pop("pending_ticket_context", None)
    
    return {"messages": [response], "memory": updated_memory}
```

**Why rebuild the system prompt on every agent call:** The memory changes between iterations (e.g. after `rag_search` returns `answerable=False`, the update_mem node adds `pending_ticket_context` to memory). If the system prompt were static, the LLM wouldn't see the pending context when it comes back for the next turn. Rebuilding ensures the LLM always has the current state.

**Why `[system_msg] + messages`:** The system message is not stored in `state["messages"]` — it's rebuilt fresh on every call. The messages list contains the conversation history (HumanMessages, AIMessages with tool_calls, ToolMessages with results). Prepending the system message gives the LLM its instructions before the conversation history.

#### `_build_memory_context()` — the pending ticket hint

```python
def _build_memory_context(memory: dict) -> str:
    if memory.get("pending_ticket_context"):
        p = memory["pending_ticket_context"]
        lines.append(
            f"- PENDING TICKET OFFER: You previously offered to create a ticket for: "
            f"'{p.get('question','')[:120]}'. "
            f"If the user says yes, call create_support_ticket with this question..."
        )
    return "\n".join(lines) or "(no prior context)"
```

**Why include the full instruction in the memory context:** When the agent sees "yes" from the user, it needs to know: (1) that a ticket was previously offered, (2) what question it was for, (3) which tool to call. Including all three pieces of information in the system prompt — rather than just a flag — allows the LLM to act correctly without additional reasoning.

#### `_node_update_memory_after_tools` — reading tool results

```python
def _node_update_memory_after_tools(state):
    messages = state.get("messages", [])
    memory = dict(state.get("memory", {}))
    
    rag_result = _extract_tool_result(messages, "rag_search")
    ticket_result = _extract_tool_result(messages, "create_support_ticket")
    
    if rag_result:
        if not rag_result.get("answerable", True):
            # Store pending context for next turn
            memory["pending_ticket_context"] = {
                "question": memory.get("last_question", ""),
                "attempted_sources": [...],
                ...
            }
        else:
            memory.pop("pending_ticket_context", None)   # clear on success
    
    if ticket_result and ticket_result.get("success"):
        memory.pop("pending_ticket_context", None)
        memory["last_ticket_id"] = ticket_result.get("ticket_id", "")
    
    return {"memory": memory}
```

**Why a separate node for memory update (instead of doing it in agent):** The agent node runs before tools execute. It can track what tool calls are being made but not their results. By the time `update_mem` runs, the `ToolNode` has already executed all tool calls and their results are in `state["messages"]` as `ToolMessage` objects. This is why memory update must be a separate node after the tool node.

#### `_extract_tool_result()` — finding tool outputs in messages

```python
def _extract_tool_result(messages: list, tool_name: str) -> Optional[dict]:
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            if getattr(msg, "name", "") == tool_name:
                return json.loads(msg.content)
    return None
```

**Why reversed:** The most recent call of a given tool is the relevant one. If `rag_search` was called multiple times (unlikely but possible), we want the latest result.

#### `_should_continue()` — the routing function

```python
def _should_continue(state) -> str:
    last_msg = state.get("messages", [])[-1]
    if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", []):
        return "tools"   # ← loop back
    return "save_mem"    # ← exit to save and finish
```

**Why check for `AIMessage` specifically:** After `ToolNode` runs, the last message is a `ToolMessage`, not an `AIMessage`. The conditional edge runs after the agent node, where the last message is always an `AIMessage`. But the check is defensive — it ensures we don't accidentally loop if the last message isn't from the agent.

#### `_node_save_mem` — persisting state and extracting results

```python
async def _node_save_mem(state):
    # Extract final response from last AIMessage with content (not a tool call)
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", []):
            final_response = msg.content
            break
    
    # Extract ticket from create_support_ticket tool result
    ticket_result = _extract_tool_result(messages, "create_support_ticket")
    
    # Extract citations from rag_search tool result
    rag_result = _extract_tool_result(messages, "rag_search")
    
    # Save memory to Redis
    await redis.set(f"statecase:memory:{session_id}", json.dumps(memory), ex=86400)
    
    # Append to session history for corrective RAG multi-turn
    history.append({"role": "user",      "content": user_msgs[-1].content})
    history.append({"role": "assistant",  "content": final_response})
    await set_session_history(session_id, history)
    
    return {
        "final_response":      final_response,
        "final_citations":     rag_result.get("citations", []) if rag_result else [],
        "final_ticket":        ticket_result if ticket_result and ticket_result.get("success") else None,
    }
```

**Why search reversed for `AIMessage` with content:** The messages list ends with the final AI response. But the loop might have produced multiple `AIMessage` objects (one per agent iteration). We want the last one that has content (text) and no tool_calls (i.e. the final response, not an intermediate one triggering tool calls).

**Why append to session history even for StateCase:** The corrective RAG rewriter in `corrective_rag_rag.py` uses session history to resolve query references ("the previous policy" → what policy was discussed). If StateCase agent calls `rag_search`, that turn should be remembered for future rewrite context.

---

### 5.4 `statecase_routes_rag.py` — StateCase FastAPI Routes

**Why this file exists:** HTTP endpoints for all ticket operations. Mounted on the main FastAPI app via `app.include_router()`.

The key design: `UpdateTicketRequest` has all optional fields:
```python
class UpdateTicketRequest(BaseModel):
    status:         Optional[str] = None
    assigned_owner: Optional[str] = None
    priority:       Optional[str] = None
    description:    Optional[str] = None
```

**Why all optional:** Users update different fields on different requests. `PATCH` semantics mean "update only what's provided". If all fields were required, the client would need to resend unchanged values. With all optional, the route only sends changed fields to Notion.

---

### 5.5 `api_helpers_statecase_rag.py` — StateCase HTTP Wrappers

**Why this file exists:** The Streamlit UI cannot import from `rag/` — it calls HTTP. This file mirrors `api_helpers_rag.py` in structure, providing one function per StateCase endpoint. All follow the same pattern: log entry, `requests.post/get/patch`, log success, return dict or None on failure.

---

## 6. Bugs Found and How They Were Fixed

### Bug 1 — Context Contamination (Wrong Answers After Topic Switch)

**What happened:** A user asked about "leave entitlement" and got a correct answer. Then asked about "bereavement leave". The answer was wrong, but when the same query was asked in a new session it was correct.

**Root cause:** Prior AI answers were injected as raw `AIMessage` turns into the LLM message list. The previous answer about "leave entitlement" contained fabricated numbers for sick leave. When the LLM was asked about bereavement leave, it saw its own prior fabricated answer and defended it as correct, ignoring the freshly retrieved chunks which had the right answer.

**Fix 1 — pipeline_rag.py:** Replace prior AI message content with a neutral placeholder:
```python
# Before
lc_messages.append(AIMessage(content=prior_answer_text))

# After
lc_messages.append(AIMessage(content="[Previous answer — see retrieved context below for current facts.]"))
```

**Fix 2 — corrective_rag_rag.py:** Exclude AI turns from the rewrite history:
```python
for turn in session_history[-6:]:
    role = turn.get("role", "user").capitalize()
    if role != "User":
        continue   # ← skip AI turns entirely
```

**Fix 3 — prompts_rag.py:** Add explicit source-of-truth rule to all system prompts:
```
CRITICAL — source of truth rule:
The numbered context chunks in the current message are the ONLY authoritative source.
Ignore anything stated in earlier conversation turns that contradicts these chunks.
```

### Bug 2 — Memory Industry Filter Bleed (Wrong Industry Context)

**What happened:** In the StateCase agent, a query about "bereavement leave" returned a score of 0.08 (too low) and triggered a false ticket offer. The same query in the bare RAG pipeline scored 0.55 and answered correctly.

**Root cause:** The agent's `load_mem` node restored `industry: "Product Management"` from a previous conversation. The `rag_answer` node injected this as a retrieval filter, restricting Milvus to only Product Management documents when searching for bereavement leave (an HR topic). The score collapsed.

**Fix — statecase_agent_rag.py:**
```python
# Before
merged_filters = {**raw_filters}
if memory.get("industry") and not merged_filters.get("industry"):
    merged_filters["industry"] = memory["industry"]  # ← inject from memory

# After
merged_filters = {**raw_filters}  # ← only what the user explicitly set
```
Memory is now only used for display context and pending ticket tracking. It never touches retrieval filters.

### Bug 3 — Notion Property Type Mismatches

**What happened:** HTTP 400 errors: `"Question is expected to be title"` and `"database property status does not match filter select"`.

**Root cause 1:** The database has `Question` as the title column (the page name) and `Ticket ID` as a rich_text column. The code had them swapped.

**Root cause 2:** `Status` is Notion's native Status type, not a Select. They look identical in the UI but use different API shapes.

**Fix — statecase_notion_rag.py:**
```python
# Added _status_prop builder
def _status_prop(value: str) -> dict:
    return {"status": {"name": value}}

# Fixed create_ticket properties
"Question":  _title_prop(question),      # ← title, not rich_text
"Ticket ID": _rich_text_prop(ticket_id), # ← rich_text, not title
"Status":    _status_prop("Not started"),# ← status, not select

# Fixed list_tickets filter
body["filter"] = {
    "property": "Status",
    "status":   {"equals": status_filter},  # ← status, not select
}
```

### Bug 4 — LLM Passing Title Instead of UUID for Ticket Updates

**What happened:** HTTP 400 errors: `"path.page_id should be a valid uuid, instead was 'Turnover of Turabit'"`.

**Root cause:** The LLM called `update_support_ticket` with `notion_page_id="Turnover of Turabit"` because that's all it knew — it had never seen the actual Notion page UUID.

**Fix — statecase_tools_rag.py + statecase_notion_rag.py:**
Added `find_ticket_by_title()` function that queries Notion by `Ticket ID` (SC-XXXX) or `Question` (title) to resolve the human-readable name to a UUID. Added UUID detection regex in the tool:
```python
_UUID_RE = re.compile(r"^[0-9a-f]{8}-?...$", re.IGNORECASE)
if not _UUID_RE.match(notion_page_id.replace("-", "")):
    ticket = find_ticket_by_title(notion_page_id)
    resolved_page_id = ticket["notion_page_id"]
```

---

## 7. How Everything Connects — End-to-End Request Trace

Here is what happens, start to finish, when a user types "How many days of sick leave do I get?" with auto-ticketing ON:

```
1. User types in Streamlit chat input
   └─ cite_rag_lab_ui_rag.py: call_statecase_chat(session_id="abc", message="How many...")

2. HTTP POST to /statecase/chat
   └─ statecase_routes_rag.py: receives request, calls run_statecase_agent()

3. LangGraph agent starts
   └─ statecase_agent_rag.py: initial_state built, _agent_graph.ainvoke() called

4. Node: load_mem
   └─ redis_cache_rag.py: GET "statecase:memory:abc" → {last_question: "..."}
   └─ memory loaded, pending_ticket_context = None

5. Node: agent (first iteration)
   └─ System prompt built with memory context (no pending ticket)
   └─ AzureChatOpenAI.invoke([system_msg, HumanMessage("How many days...")])
   └─ LLM responds: tool_calls=[{name: "rag_search", args: {query: "...", session_id: "abc"}}]
   └─ Routing: has tool_calls → go to tools

6. Node: tools (ToolNode dispatches rag_search)
   └─ statecase_tools_rag.py: rag_search() called
   └─ redis_cache_rag.py: load session history (async)
   └─ pipeline_rag.py: run_rag_pipeline() called
       ├─ filters_rag.py: build_filters({}) → {}
       ├─ adaptive_router_rag.py: classify_query() → "QA", top_k=5
       ├─ corrective_rag_rag.py: corrective_retrieve()
       │   ├─ retriever_rag.py: embed_text("How many days...")
       │   │   └─ Azure OpenAI embeddings API → [0.123, -0.456, ...] (3072 floats)
       │   ├─ milvus_client_rag.py: hybrid_search_chunks() → 10 chunks
       │   ├─ avg_score=0.71 ≥ 0.65 → done (no rewrite needed)
       │   └─ returns chunks1, score1=0.71
       ├─ reranker_rag.py: rerank() → top 5 chunks
       ├─ avg_score=0.68 ≥ 0.30 → proceed to LLM
       ├─ retriever_rag.py: format_context_for_prompt() → "[1] HR Policy → Sick Leave..."
       ├─ prompts_rag.py: RAG_SYSTEM_PROMPT selected
       ├─ AzureChatOpenAI.invoke([system, history, "Retrieved context:...Question:..."])
       └─ answer="Employees are entitled to 10 days of sick leave per year [1][3]."
   └─ rag_search returns {answer: "...", answerable: True, avg_score: 0.68, citations: [...]}

7. Node: update_mem
   └─ rag_result.answerable = True → clear any pending_ticket_context
   └─ memory["last_question"] = "How many days of sick leave..."

8. Node: agent (second iteration)
   └─ System prompt rebuilt (no pending context)
   └─ AzureChatOpenAI.invoke([system, HumanMessage, AIMessage(tool_calls), ToolMessage(result)])
   └─ LLM reads the rag_search result, sees answerable=True
   └─ LLM responds: content="Employees are entitled to 10 days of sick leave per year [1][3]."
   └─ Routing: no tool_calls → go to save_mem

9. Node: save_mem
   └─ final_response extracted from last AIMessage
   └─ final_citations extracted from rag_search ToolMessage
   └─ redis_cache_rag.py: SET "statecase:memory:abc" = updated memory, TTL=86400
   └─ redis_cache_rag.py: SET "rag:session:abc" = history with new turn, TTL=86400
   └─ returns {final_response, final_citations, final_ticket=None}

10. Back in statecase_routes_rag.py
    └─ returns {session_id, answer, citations, pipeline_meta, ticket_created=None, trace_id}

11. Back in Streamlit
    └─ _append_message("assistant", answer, citations=citations, ticket_created=None)
    └─ st.rerun() → page re-renders with the new message
    └─ Answer shown: "Employees are entitled to 10 days of sick leave..."
    └─ Citations shown: "[1] HR Policy → Sick Leave  QA  score: 0.7234"
```

---

## 8. Environment Variables

```bash
# ── Azure OpenAI (LLM — for answering questions, routing, rewriting) ──────────
AZURE_OPENAI_LLM_KEY=<your-azure-openai-api-key>
AZURE_LLM_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_LLM_API_VERSION=2024-12-01-preview
AZURE_LLM_DEPLOYMENT_41_MINI=gpt-4.1-mini      # the deployment name in Azure portal

# ── Azure OpenAI (Embeddings — for converting text to vectors) ────────────────
AZURE_OPENAI_EMB_KEY=<same-or-different-azure-openai-key>
AZURE_EMB_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_EMB_API_VERSION=2024-12-01-preview
AZURE_EMB_DEPLOYMENT=text-embedding-3-large     # the embedding deployment name

# ── Notion ────────────────────────────────────────────────────────────────────
NOTION_API_KEY=secret_xxxxxxxxxxxx              # from notion.so/my-integrations
NOTION_ROOT_PAGE_ID=31489db15e5b804c9049d062c6cdce54  # document library database ID
STATECASE_DB_ID=32d89db1-5e5b-8051-a212-f5f983a90a0f  # tickets database ID

# ── Redis ─────────────────────────────────────────────────────────────────────
REDIS_URL=redis://localhost:6379

# ── Milvus ────────────────────────────────────────────────────────────────────
MILVUS_URI=./rag_data/milvus.db   # path to the milvus-lite file
```

**Starting the backend:**
```bash
uvicorn rag.api.main_rag:app --host 0.0.0.0 --port 8001 --reload
```

**Starting the UI:**
```bash
streamlit run ui/streamlit_uidemo.py --server.port 8501
```

**First-time setup — ingest documents:**
1. Start the backend
2. Open `http://localhost:8501`
3. Go to the 📥 Ingest tab
4. Click "Ingest All Pages"
5. Wait for the summary (may take several minutes depending on library size)
6. Go to 🔍 Inspector to verify chunks are retrievable
7. Start chatting in 💬 Chat