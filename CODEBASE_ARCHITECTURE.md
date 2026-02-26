# DocForgeHub — Comprehensive Codebase Architecture

## Executive Summary

**DocForgeHub** is an enterprise-grade, AI-powered document generation and management platform designed for SaaS organizations. It transforms user-provided answers into schema-compliant, professionally-written business documents using a dual-LLM architecture, a 5-node LangGraph state machine, and an intelligent cache-first gap analysis system.

### Core Capabilities
- **Schema-Driven Q&A**: Fetches categorized questions from MongoDB and presents them in a paginated, widget-driven Streamlit UI
- **Intelligent Gap Analysis**: Uses a lightweight LLM (Llama-3.3-70b) to detect which document schema sections lack Q&A coverage, then generates targeted questions to fill those gaps — with results persisted for all future users of the same document type
- **Dual-Mode Document Generation**: Supports both single-shot (full document at once) and progressive (section-by-section with memory) generation via a 5-node LangGraph agent
- **Two-Layer Validation**: Deterministic structural checks + LLM-based quality review with automatic retry (up to 2x)
- **PDF Export**: ReportLab converts the Markdown output to a professional A4 PDF with styled headings, tables, and bullets
- **Notion Integration**: Recursive page URL retrieval from a Notion workspace displayed as generation history

### Key Design Innovation: Cache-First Gap Filling
Instead of hallucinating content for uncovered schema sections, DocForgeHub:
1. Calls a lightweight LLM to identify which schema sections the user's answers do not cover
2. Generates one targeted question per uncovered section
3. Presents those questions to the user in the same UI as the core questions
4. Persists the answered gap questions to MongoDB so every subsequent user of that document type gets them for free — reducing LLM cost to O(1) per document type over time

---

## Technology Stack

| Layer | Technology | Version | Purpose | Rationale |
|-------|-----------|---------|---------|-----------|
| **Primary LLM** | Groq + `moonshotai/kimi-k2-instruct-0905` | — | Document generation, quality review, fix retries | Best-in-class prose quality, 200K context window |
| **Analysis LLM** | Groq + `llama-3.3-70b-versatile` | — | Schema gap analysis, structured JSON output | Fast, cheap, strong at structured reasoning — ~5x cheaper than kimi-k2 |
| **Orchestration** | LangGraph (`langgraph`) | >= 0.1.0 | 5-node document generation state machine | Deterministic state management, conditional routing, built-in retry |
| **LLM SDK** | `langchain-groq`, `langchain-core` | >= 0.0.1 | Groq `ChatGroq` model + `SystemMessage`/`HumanMessage` types | Native Groq API integration |
| **REST API** | FastAPI | >= 0.104.1 | Async REST gateway, 9 endpoints | Async/await, Pydantic validation, auto-generated `/docs` |
| **Database** | MongoDB Atlas | — | Q&As, schemas, gap question cache | Flexible JSON schema, aggregation pipeline, upsert support |
| **Async Driver** | Motor | >= 3.3.0 | Non-blocking MongoDB operations | Integrates natively with FastAPI's event loop |
| **Frontend** | Streamlit | >= 1.32.0 | Interactive Q&A UI, document editor, PDF download | `st.cache_data` caching, wide layout, native widget types |
| **Notion** | `notion-client` | >= 2.1.0 | Recursive child-page traversal for history panel | Official Notion SDK |
| **PDF Export** | ReportLab (`reportlab`) | — | Markdown to styled A4 PDF | Pure-Python, no headless browser required |
| **Config** | `python-dotenv` | >= 1.0.0 | `.env` loading at startup | Standard pattern |
| **Language** | Python | 3.12+ | Entire codebase | Type hints, async/await, TypedDict |

---

## Repository Layout

```
DocForgeHub/
|
+-- agent/                             # LangGraph agent + all generation logic
|   +-- __init__.py
|   +-- agent_graph.py                 # AgentState, 5 nodes, graph assembly, public API
|   +-- prompts.py                     # All LLM prompt templates + builder functions
|   +-- schema_helpers.py              # Schema/Q&A formatter + schema inspector functions
|   +-- validation_helpers.py          # Structural heading + table column validation
|
+-- api/                               # FastAPI application
|   +-- __init__.py
|   +-- main.py                        # 9 REST endpoints + Pydantic request models
|   +-- db.py                          # Async Motor singleton (get_db / close_client)
|   +-- helpers.py                     # Notion API: recursive page traversal
|
+-- ui/                                # Streamlit frontend
|   +-- streamlit_uidemo.py            # Main app: sidebar, Q&A panels, editor, PDF
|   +-- api_helpers.py                 # HTTP wrappers for every FastAPI endpoint
|   +-- question_helpers.py            # Question list logic, widget renderer
|   +-- pdf_generator.py               # ReportLab Markdown to A4 PDF
|
+-- automations/                       # Admin / one-time batch data upload scripts
|   +-- mongo_auto.py                  # Upload Q&A JSON files to document_qas
|   +-- required_sections_automation.py  # Upload schema JSON to required_section
|
+-- document_and_questions/            # Local data repository
|   +-- final_filtered_QAs/            # Q&A JSON files, organized by department
|   +-- notion_documents/              # Schema JSON files extracted from Notion
|
+-- .env                               # Secrets (never commit)
+-- requirements.txt
+-- CODEBASE_ARCHITECTURE.md           # This file
+-- progress.md
+-- README.md
```

---

## High-Level System Architecture

```
+------------------------------------------------------------------------------+
|                       STREAMLIT UI  (Port 8501)                              |
|  ui/streamlit_uidemo.py                                                      |
|                                                                              |
|  +-----------------+  +----------------------------+  +------------------+  |
|  |   LEFT SIDEBAR  |  |   QUESTIONS PANEL (col1)   |  |  EDITOR PANEL    |  |
|  |                 |  |                            |  |  (col2)          |  |
|  | Dept select     |  | Paginated Q&A (5/page)     |  |                  |  |
|  | Doc select      |  | - Core questions           |  | Single-shot:     |  |
|  | Mode toggle     |  | - Gap questions (AI badge) |  |   Markdown       |  |
|  |   Single-Shot   |  |                            |  |   editor +       |  |
|  |   Progressive   |  | Controls:                  |  |   preview        |  |
|  |                 |  | - Progress bar             |  |                  |  |
|  | Notion history  |  | - Analyse gaps button      |  | Progressive:     |  |
|  | (scrollable)    |  | - Save gaps button         |  |   Sequential     |  |
|  |                 |  | - Back / Next nav          |  |   subsection     |  |
|  |                 |  | - Generate / Finalize btn  |  |   reveal +       |  |
|  +-----------------+  +----------------------------+  |   generate btn   |  |
|                                                       |                  |  |
|                                                       | PDF export btn   |  |
|                                                       | Publish button   |  |
|                                                       +------------------+  |
|                                                                              |
|  ui/api_helpers.py ------- HTTP REST ------------------------------------+  |
|  ui/question_helpers.py  (pure Python, no HTTP)                             |
|  ui/pdf_generator.py     (pure Python, no HTTP)                             |
+------------------------------------------------------------------------------+
                                     |
                            HTTP/REST (localhost)
                                     |
                                     v
+------------------------------------------------------------------------------+
|                      FASTAPI BACKEND  (Port 8000)                            |
|  api/main.py                                                                 |
|                                                                              |
|  GET  /departments         GET  /document-types    GET  /questions           |
|  GET  /required-section    GET  /get_all_urls                                |
|  POST /gap-questions        POST /save-questions                             |
|  POST /generate             POST /generate-section                           |
|                                                                              |
|  api/db.py --------- Motor (async) ----------------------+                  |
|  api/helpers.py ----- Notion API -------------------+    |                  |
|  agent/* ------------ LangGraph agent ----------+   |    |                  |
+---------------------------------------------------+---+--+------------------+
                                                    |   |  |
                        +--------------------------+    |  |
                        |       +-------------------+   |  |
                        |       |        +----------+   |  |
                        v       v        v              |  |
          +------------------+  +----------------+  +----------+
          |  LANGGRAPH AGENT |  | MONGODB ATLAS  |  |  NOTION  |
          |                  |  |                |  |   API    |
          | Node 1           |  | document_qas   |  |          |
          | analyze_gaps     |  |  - core Q&As   |  | child    |
          | Node 2           |  |  - gap Q&As    |  | _page    |
          | build_prompt     |  |                |  | recursive|
          | Node 3           |  | required_      |  | traversal|
          | generate_doc     |  | section        |  |          |
          | Node 4           |  | (schemas)      |  | returns  |
          | quality_gate     |  |                |  | [{id,    |
          | Node 5           |  | Motor async    |  |  title,  |
          | fix_document     |  | driver         |  |  url}]   |
          +------------------+  +----------------+  +----------+
                    |
                    | Groq API (HTTPS)
                    v
        +------------------------------------------+
        |              GROQ CLOUD                   |
        |                                          |
        |  kimi-k2-instruct-0905                   |
        |  temp=0.1, max_tokens=8192               |
        |  --> Nodes 3, 4, 5 (generation/review)   |
        |                                          |
        |  llama-3.3-70b-versatile                 |
        |  temp=0.2, max_tokens=2048               |
        |  --> Node 1 + /gap-questions (analysis)  |
        +------------------------------------------+
```

---

## MongoDB Data Design

### Collection: `document_qas`

Stores both core Q&A questions (seeded by the admin batch scripts) and AI-generated gap questions (written at runtime by the `/save-questions` endpoint).

```json
{
  "_id": "ObjectId",
  "department": {
    "code": "1",
    "name": "Product Management",
    "slug": "product-management"
  },
  "document_type": "Feature Prioritization Framework",
  "document_name": "Feature prioritization framework",
  "question": "What is the primary objective of this feature prioritization effort?",
  "answer": "",
  "category": "Overview",
  "category_order": 1,
  "question_order": 3,
  "answer_type": "text | select | multi_select | structured_list",
  "options": [],
  "description": "Optional user guidance shown beneath the widget",
  "is_gap_question": false,
  "section_covered": "",
  "answered_at": null,
  "created_at": "ISODate",
  "updated_at": "ISODate"
}
```

**Gap question variant** (written by `POST /save-questions`):
```json
{
  "is_gap_question": true,
  "category": "Additional Information",
  "category_order": 999,
  "question_order": 1002,
  "section_covered": "Risk Assessment",
  "answered_at": "2025-02-20T14:32:11.000Z"
}
```

**Key indexes:**
- `{department.name, document_type}` — primary data lookup by `/questions`
- `{document_type, is_gap_question}` — cache check in `POST /gap-questions`
- `{category_order, question_order}` — sort order for the UI

**Sort rule**: `GET /questions` sorts by `(category_order ASC, question_order ASC)`. Gap questions always sort last because `category_order=999` and `question_order=1000+`.

---

### Collection: `required_section`

Stores the structural schema for each document type — which sections the generated document must contain, in what order, and of what type (text or table).

**Pattern A — Mixed schema** (most document types):
```json
{
  "_id": "ObjectId",
  "department": "Product Management",
  "document_name": "Feature prioritization framework",
  "document_type": "Feature Prioritization Framework",
  "sections": [
    {
      "title": "1. Objective",
      "type": "text",
      "order": 1,
      "subsections": [
        { "title": "1.1 Business Impact", "type": "text",  "order": 1 },
        { "title": "1.2 Success Metrics", "type": "table", "order": 2,
          "columns": ["Metric", "Target", "Owner", "Timeline"] }
      ]
    },
    {
      "title": "2. Feature List",
      "type": "text",
      "order": 2,
      "subsections": [
        { "title": "2.1 Feature Candidates", "type": "table", "order": 1,
          "columns": ["Feature ID", "Name", "Priority Score", "Effort", "Status"] }
      ]
    }
  ]
}
```

**Pattern B — Table-only schema** (e.g. Change Request Log):
```json
{
  "sections": [
    {
      "type": "table",
      "order": 1,
      "columns": ["CRID", "Date", "Requested By", "Change Description", "Priority", "Status"]
    }
  ]
}
```

> **Important:** Pattern B sections intentionally omit the `"title"` key on the section dict. `get_table_section_title()` in `schema_helpers.py` handles this with a 4-level fallback chain: `section["title"]` -> `required_section["document_name"]` -> `required_section["document_type"]` -> `"Data Table"`.

**Key indexes:** `{department, document_name}` — schema lookup used by `/required-section`, `/gap-questions`, and `/generate`

---

## File-by-File Deep Dive

---

### `api/db.py`

Manages the Motor async MongoDB connection using a module-level singleton pattern.

```python
DATABASE_NAME = "document_automation"
_client: AsyncIOMotorClient = None

def get_client() -> AsyncIOMotorClient  # creates on first call, reuses thereafter
def get_db()                            # returns get_client()[DATABASE_NAME]
async def close_client()               # called by FastAPI lifespan on shutdown
```

**Connection lifecycle:**
- `_client` is `None` on module load
- First call to `get_client()` creates `AsyncIOMotorClient(MONGODB_CONNECTION_STRING)`
- FastAPI's `lifespan` context manager calls `close_client()` on app shutdown, setting `_client = None`
- Motor manages the underlying connection pool internally

**Environment variable:** `MONGODB_CONNECTION_STRING` loaded from `.env` via `python-dotenv` at module import

---

### `api/helpers.py`

Provides Notion API utilities. The Notion client is instantiated at module-import time from `NOTION_API_KEY`.

```python
notion_client = Client(auth=os.environ.get("NOTION_API_KEY"))
```

**`get_page_url_from_id(page_id: str) -> str`**

Strips dashes from the API-provided UUID and constructs the Notion web URL:
```
"30589db1-5e5b-8077-9819-dc0d8c532954"  ->  "https://notion.so/30589db15e5b80779819dc0d8c532954"
```

**`retrieve_all_child_pages_recursive(block_id, all_pages=[]) -> list[dict]`**

Recursively walks the Notion block tree starting from a given `block_id`:
- Uses `notion_client.blocks.children.list(block_id=block_id, start_cursor=next_cursor, page_size=100)`
- Cursor-based pagination: loops while `has_more=True`
- For every block of `type == "child_page"`: appends `{id, title, url}` to `all_pages`
- Recurses into each discovered child page with the page's own ID as the new `block_id`
- Per-block `try/except`: errors skip that subtree rather than aborting the entire walk
- Returns the accumulated flat list of all descendants

The root page ID is hardcoded in `main.py`:
```python
root_page_id = "30589db15e5b80779819dc0d8c532954"
```

---

### `api/main.py`

The FastAPI application. Configures CORS, defines all 9 endpoints, and owns all Pydantic request models.

**App setup:**
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await close_client()          # Motor cleanup on shutdown

app = FastAPI(title="DocForge Hub API", lifespan=lifespan)

app.add_middleware(CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

#### `GET /departments`

MongoDB aggregation on `document_qas`:
```python
pipeline = [
    {"$group": {"_id": "$department"}},
    {"$sort":  {"_id.code": 1}},
]
```
Extracts `{code, name, slug}` from each `department` subdocument. Python-sorts the final list by `code` as a second pass. Returns `{departments: [{code, name, slug}]}`.

#### `GET /document-types?department={name}`

```python
pipeline = [
    {"$match": {"department.name": department}},
    {"$group": {"_id": {"document_type": "$document_type", "document_name": "$document_name"}}},
    {"$sort":  {"_id.document_type": 1}},
]
```
Returns `{document_types: [{document_type, document_name}]}`, alphabetically sorted.

#### `GET /questions?document_type={type}`

```python
cursor = db["document_qas"].find(
    {"document_type": document_type},
    {"_id": 0, "_runtime_metadata": 0, "schema_id": 0},
).sort([("category_order", 1), ("question_order", 1)])
questions = await cursor.to_list(length=500)
```
Returns all Q&As for the document type — both core questions and any `is_gap_question=True` records. Gaps sort last because `category_order=999`.

#### `GET /required-section?department={dept}&document_name={name}`

```python
schema_document = await db["required_section"].find_one(
    {"department": department, "document_name": document_name},
    {"_id": 0},
)
```
Returns `{required_section: {...}}` or raises HTTP 404 if not found.

#### `GET /get_all_urls`

Calls `retrieve_all_child_pages_recursive(root_page_id)` synchronously and returns:
```json
{"root_page_id": "...", "page_count": 42, "pages": [{"id": "...", "title": "...", "url": "..."}]}
```

#### `POST /gap-questions` — Cache-First Gap Analysis

**Request model:**
```python
class GapQuestionsRequest(BaseModel):
    department: str
    document_type: str
    document_name: str
    questions_and_answers: List[Dict[str, Any]]
    required_section: Optional[Dict[str, Any]] = None
```

**Flow:**
1. **Cache check**: `db["document_qas"].find_one({document_type, is_gap_question: True})`
   - **Cache hit**: fetch all gap questions for this type → return `{gap_questions, source: "cache", count}` (zero LLM calls)
   - **Cache miss**: continue to step 2
2. **Schema fetch**: use `required_section` from request if provided; otherwise query `required_section` collection by `{department, document_name}`; fallback to `{"sections": []}` if not found
3. **LLM analysis**: `await analyze_gaps_only(department, document_type, qa_list, required_section)` — runs Node 1 in isolation against the lightweight LLM
4. Return `{gap_questions, source: "generated", count}`

#### `POST /save-questions` — Upsert Gap Questions

**Request model:**
```python
class SaveQuestionsRequest(BaseModel):
    department: Dict[str, Any]       # full {code, name, slug} object
    document_type: str
    document_name: str
    gap_questions: List[Dict[str, Any]]
```

**Flow:**
1. Aggregate `MAX(question_order)` for this `document_type` to determine `base_order`
2. For each gap question in the request:
   - Build a full document dict with `is_gap_question=True`, `category_order=999`, `question_order=base_order+1000+i`, `answered_at=datetime.utcnow().isoformat()`
   - Upsert by `{document_type, question, is_gap_question: True}` — prevents duplicate entries if user saves twice
3. Track `saved_count` (new inserts) vs `updated_count` (updates to existing)
4. Return `{saved, updated, total}`

#### `POST /generate` — Full Document Generation

**Request model:**
```python
class GenerateDocumentRequest(BaseModel):
    department: str
    document_type: str
    document_name: str
    questions_and_answers: List[Dict[str, Any]]
    required_section: Dict[str, Any] | None = None
```

**Flow:**
1. If `required_section` is `None`: fetch from MongoDB `required_section` collection by `{department, document_name}`; fallback to `{"sections": []}` if missing
2. `await run_agent(department, document_type, qa_list, required_section)`
3. Return `{generated_document, gap_questions, status, quality_issues, quality_scores, quality_suggestions, retry_count}`

#### `POST /generate-section` — Progressive Single-Section Generation

**Request model:**
```python
class GenerateSectionRequest(BaseModel):
    department: str
    document_type: str
    section: dict           # {title: parent_title, subsections: [...]}
    questions_and_answers: list
    doc_memory: str = ""    # previously generated sections injected for consistency
```

Calls `await generate_single_section(...)` and returns `{section_text: str}`.

---

### `agent/prompts.py`

All LLM prompt templates and their builder functions. Pure Python — no LLM calls, no database access, no Streamlit dependency.

#### Templates

**`SYSTEM_PROMPT_TEMPLATE`** — Main document generation prompt injected in Node 3.

Core rules enforced by the template text:
- **Content elevation (most important)**: never copy-paste user answers verbatim. Vague answers (e.g. "yes", "we use React") must be expanded with industry context, best practices, and concrete details.
- **Structural rules (strict enforcement)**: output must follow the schema section order exactly — same headings, same numbering. No extra sections, no omissions, no renumbering.
- **TABLE RULE**: sections marked `type: table` must output a real Markdown table with the exact columns — not prose, not lists, not a description of the table.
- **TEXT RULE**: sections marked `type: text` output professional prose with paragraphs and/or bullet lists.
- **Inferred content**: if no answer covers a section, infer reasonable content from context and mark it with `*(Recommended based on industry best practices)*`.
- **Absolute prohibitions**: `[TBD]`, `[Company Name]`, `[Insert here]`, lorem ipsum, single-sentence sections, sections starting with "This section...", describing table contents instead of outputting the table.
- **Output format**: starts with `# {document_type}`, uses `##` for major sections, `###` for subsections, includes a version/metadata footer.

**`TABLE_ONLY_PROMPT_TEMPLATE`** — Used when the entire schema is `type: table` with no subsections.

Structure:
- First output line must be `# {document_type}`
- Immediately followed by the Markdown table (no introduction, no other sections)
- Columns injected as pre-formatted `| Col1 | Col2 | Col3 |` and `| --- | --- | --- |` lines
- `min_rows=4`, `max_rows=12` (hardcoded in `build_table_only_prompt`)
- All prose sections, metadata footers, and explanations are explicitly prohibited

**`SCHEMA_GAP_FILLER_PROMPT`** — Legacy prompt for generating supplementary prose content for uncovered schema sections. Used by `build_gap_filler_prompt()`.

**`QUALITY_REVIEW_PROMPT`** — Used in Node 4's LLM quality review. Scores on 5 criteria (completeness, professionalism, depth, actionability, structure — each 1-5) and returns strict JSON:
```json
{
  "scores": {"completeness": 4, "professionalism": 5, "depth": 3, "actionability": 4, "structure": 5},
  "overall_score": 4,
  "passed": true,
  "issues": [],
  "suggestions": ["Expand the Risk section with more concrete mitigations"]
}
```
`passed=true` when `overall_score >= 3`. The `passed` field takes precedence; `overall_score` is used as fallback.

#### Builder Functions

| Function | Template Used | Called By | Notes |
|----------|--------------|-----------|-------|
| `build_system_prompt(department, document_type, required_section, questions_and_answers, supplementary_content)` | `SYSTEM_PROMPT_TEMPLATE` | Node 2 | Wraps supplementary content in a labelled header section if non-empty and not the "all covered" string |
| `build_table_only_prompt(department, document_type, columns, questions_and_answers, supplementary_content)` | `TABLE_ONLY_PROMPT_TEMPLATE` | Node 2 | Pre-formats `columns_header` and `columns_separator` before injection; `min_rows=4`, `max_rows=12` |
| `build_gap_filler_prompt(department, document_type, required_section, questions_and_answers)` | `SCHEMA_GAP_FILLER_PROMPT` | Node 1 (indirect), `api/main.py` | Returns supplementary prose for uncovered sections |
| `build_quality_review_prompt(department, document_type, generated_document)` | `QUALITY_REVIEW_PROMPT` | Node 4 | Injects the full generated document text |

---

### `agent/schema_helpers.py`

Pure functions for parsing `required_section` dicts and formatting data for LLM prompts. Zero side effects.

#### `format_questions_and_answers_for_prompt(qa_list: list[dict]) -> str`

Groups Q&A pairs under `### Category` Markdown headers, emitting a new header each time the category field changes:
```
### Overview
**Q:** What is the primary objective?
**A:** Increase feature adoption by 25% in Q3

### Risk Management
**Q:** What are the key risks?
**A:** (not provided)
```
Handles `structured_list` answer types by serialising `qa_item["answers"]` as indented JSON. Handles plain `list` answers by joining with `, `.

#### `format_required_section_for_prompt(required_section: dict) -> str`

Converts the schema dict into a human-readable Markdown outline for injection into prompts.

Handles all three cases:
- **Table-only** (section has `type: table`, no subsections): emits `## Title` + `WARNING: TABLE FORMAT REQUIRED` block with column header string and explicit instruction to output the actual table
- **Mixed** (sections have `subsections[]`): emits `## Section Title` followed by `  - Subsection Title (type: text)` or `  - Subsection Title WARNING TABLE -- columns: | Col1 | Col2 |`
- **Legacy fallback**: if `sections` is missing entirely, uses `question_categories` list as `- category (order: N)` lines

#### `is_table_only_schema(required_section: dict) -> bool`

Returns `True` if **every** section in `sections[]` has `type == "table"` AND has no `subsections` array. This single check controls which prompt template Node 2 uses and which quality gate path Node 4 takes.

#### `get_table_columns(required_section: dict) -> list[str]`

Scans `sections[]` and returns the `columns` list from the first section with `type == "table"`. Returns `[]` if none found.

#### `get_table_section_title(required_section: dict) -> str`

4-level fallback chain to handle Pattern B schemas that omit `"title"` on the section:
```
section["title"]  ->  required_section["document_name"]
                  ->  required_section["document_type"]
                  ->  "Data Table"
```
Strips whitespace at each level before checking if non-empty.

---

### `agent/validation_helpers.py`

Validates a generated Markdown document against the `required_section` schema. Returns `list[str]` of error messages (empty list = valid).

#### `validate_document_structure(document_text, required_section) -> list[str]`

**Pattern A (table-only):** Returns `[]` immediately. Column validation is handled deterministically inside `quality_gate` Node 4.

**Pattern B (mixed) — Three-stage validation:**

**Stage 1 — Build expected sections:**
Iterates `sections[].subsections[]` (sorted by `order`). Each subsection title becomes a required heading in the `allowlist` dict (`normalised_title -> {title, type, columns}`). Parent section titles (e.g. `"1. Objective"`) are NOT required headings — they go into `skip_headings` instead, along with `document_name` and `document_type`.

**Stage 2 — CHECK 1 (Missing sections):**
For each normalised title in `allowlist`, verifies at least one document heading contains it. Reports: `"Missing required section: '1.1 Business Impact'"`.

**Stage 3 — CHECK 2 (Extra sections):**
For every heading found in the document, verifies it either matches a title in `allowlist` (subsection titles) or matches a title in `skip_headings` (document name, parent section titles). Reports: `"Extra section not in schema: 'Introduction' — remove it, the document must only contain schema-defined sections."`.

**Stage 4 — CHECK 3 (Table columns):**
For every `type: "table"` subsection: finds its heading line in the document, collects all lines until the next heading, verifies a pipe-delimited table with a separator row (`re.search(r"\|[\s\-|]+\|")`), then compares column headers case-insensitively to the schema `columns` list. Reports both missing tables and column mismatches.

#### `_normalise_heading(raw: str) -> str`

Normalisation function enabling tolerant heading comparison:
- Strips `#` markers and surrounding whitespace
- Removes numeric prefixes: `re.sub(r"^\d+(\.\d+)*\.?\s*", "", text)` — removes `"4.1 "`, `"2. "`, `"3.1.2 "` etc.
- Removes punctuation: `re.sub(r"[^\w\s]", "", text)`
- Lowercases

Effect: `"### 4.1 Customer Impact"` normalises to `"customer impact"`, matching `"4.1 Customer Impact"` in the schema which also normalises to `"customer impact"`.

---

### `agent/agent_graph.py`

The central agent file. Defines `AgentState`, all 5 node functions, graph routing, and 3 public async entry points.

#### LLM Instances (Module-Level Singletons)

```python
# Primary document-generation LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="moonshotai/kimi-k2-instruct-0905",
    temperature=0.1,
    max_tokens=8192,
)

# Dedicated question-generation LLM (lighter, faster)
question_gen_llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    max_tokens=2048,
)
```

Both are created once at module import. The comment in the code explains the rationale: keeping gap analysis on a separate model avoids burning kimi-k2's context window on schema analysis tasks that don't require prose quality.

#### `AgentState` TypedDict

All data that flows between nodes is stored in a single `AgentState` dict.

| Field | Type | Direction | Set By | Description |
|-------|------|-----------|--------|-------------|
| `department` | str | input | caller | e.g. "Product Management" |
| `document_type` | str | input | caller | e.g. "Feature Prioritization Framework" |
| `questions_and_answers` | list[dict] | input | caller | List of `{question, answer, category, answer_type}` |
| `required_section` | dict | input | caller | Full schema dict from MongoDB |
| `gap_questions` | list[dict] | output | Node 1 | List of `{question, category, answer_type, section_covered}` |
| `supplementary_content` | str | intermediate | Node 1 | Placeholder notes for uncovered sections |
| `system_prompt` | str | intermediate | Node 2 | Full assembled prompt string sent to kimi-k2 |
| `generated_document` | str | output | Node 3 | Raw Markdown text; overwritten by Node 5 on retry |
| `quality_scores` | dict | output | Node 4 | `{completeness, professionalism, depth, actionability, structure}` |
| `quality_issues` | list[str] | output | Node 4 | Problems found by structural check or LLM review |
| `quality_suggestions` | list[str] | output | Node 4 | Improvement suggestions from LLM reviewer |
| `retry_count` | int | counter | Node 2 (init=0), Node 5 (increment) | Number of fix attempts made |
| `status` | str | output | Node 2 ("generating"), Node 4 ("passed"/"failed") | Final agent status |

#### Graph Topology

```
START
  |
  v
analyze_schema_gaps   (Node 1 -- question_gen_llm / llama-3.3-70b)
  |
  v
build_prompt          (Node 2 -- pure Python, no LLM)
  |
  v
generate_document     (Node 3 -- llm / kimi-k2)
  |
  v
quality_gate          (Node 4 -- deterministic + llm / kimi-k2)
  |
  +-- status == "passed"  -----------------------------------------> END
  |
  +-- status == "failed" AND retry_count < 2
        |
        v
      fix_document    (Node 5 -- llm / kimi-k2)
        |
        +----------------------------------------------------> quality_gate
                                                   (loops max 2x)
  +-- status == "failed" AND retry_count >= 2  -----------> END
```

Conditional routing function:
```python
def decide_after_quality_gate(state: AgentState) -> Literal["fix_document", "end"]:
    if state["status"] == "passed":   return "end"
    if state["retry_count"] >= 2:     return "end"
    return "fix_document"
```

---

#### Node 1: `analyze_schema_gaps`

**LLM:** `question_gen_llm` (llama-3.3-70b, temp=0.2, max=2048 tokens)

**Purpose:** Identify which schema sections the existing Q&A answers do not cover, and generate one targeted question per uncovered section.

Uses an inline system prompt (`_GAP_QUESTION_SYSTEM_PROMPT`) defined in `agent_graph.py` — distinct from the templates in `prompts.py`. The rules:
- A section is "covered" if at least one Q&A answer meaningfully addresses it
- A section is "uncovered" if no answer touches it, or the answer is very thin
- Output: strict JSON array `[{question, category, answer_type, section_covered}]`
- If all sections are covered: return `[]`

**Detailed logic:**
1. Format schema via `format_required_section_for_prompt(required_section)`
2. Format Q&A via `format_questions_and_answers_for_prompt(questions_and_answers)`
3. Construct user message with `DOCUMENT TYPE`, `DEPARTMENT`, formatted schema, and formatted Q&A
4. Call `question_gen_llm.invoke([SystemMessage, HumanMessage])`
5. Strip any accidental markdown fences from raw response (`\`\`\`json` or `\`\`\``)
6. `json.loads(raw)` -> `gap_questions`
7. If `gap_questions` is empty: return `{gap_questions: [], supplementary_content: ""}`
8. If gaps found: build `supplementary_content` — for each gap question, one placeholder line: `"**{section}**: This section requires additional information. Gap question pending user answer: '{question}'"`. This gives Node 3 reference material for uncovered sections even if the user hasn't answered the gap questions yet.
9. Return `{gap_questions, supplementary_content}`
10. On `json.JSONDecodeError` or any exception: log warning, return `{gap_questions: [], supplementary_content: ""}` — **non-fatal**, the agent continues to Node 2

---

#### Node 2: `build_prompt`

**LLM:** None — pure Python.

**Purpose:** Assemble the full `system_prompt` that Node 3 will send to kimi-k2.

**Table-only path** (`is_table_only_schema()` returns True):
1. `get_table_columns(required_section)` -> column list
2. `get_table_section_title(required_section)` -> handles missing `title` key
3. `build_table_only_prompt(department, table_title, columns, formatted_answers, supplementary_content)` -> strict table-only prompt
4. Returns `{system_prompt, retry_count: 0, status: "generating"}`

**Mixed path** (`is_table_only_schema()` returns False):
1. `format_required_section_for_prompt(required_section)` -> Markdown schema outline
2. Extract all subsection titles from `sections[].subsections[]` (sorted by `order`):
   ```python
   required_headings = [
       subsection.get("title", "").strip()
       for section in required_section.get("sections", [])
       for subsection in sorted(section.get("subsections", []), key=lambda s: s.get("order", 0))
       if subsection.get("title", "").strip()
   ]
   ```
3. Build `strict_rule` string injected at the end of the schema text:
   ```
   WARNING STRICT SECTION RULE:
   Your document MUST contain ALL of the following headings and NO others.
   Do NOT add, rename, merge, reorder, or omit any heading:
     - 1.1 Business Impact
     - 1.2 Success Metrics
     - 2.1 Feature Candidates
     ...
   ```
4. `build_system_prompt(department, document_type, formatted_schema + strict_rule, formatted_answers, supplementary_content)` -> full generation prompt
5. Returns `{system_prompt, retry_count: 0, status: "generating"}`

---

#### Node 3: `generate_document`

**LLM:** `llm` (kimi-k2, temp=0.1, max=8192 tokens)

**Purpose:** Call kimi-k2 with the assembled system prompt and obtain the Markdown document.

**Human message construction:**

For table-only schemas: `"Generate the {table_title} as a Markdown table now. Output ONLY the heading and table — no introductions, no descriptions, no extra sections. Just the title and the table with data rows."` Note: uses `get_table_section_title()` to correctly name the document even when the schema section has no `title` key.

For mixed schemas: `"Generate the complete {document_type} document now. Remember: elevate every answer into professional, industry-grade prose. Do NOT copy answers verbatim."`

Sends `[SystemMessage(system_prompt), HumanMessage(human_instruction)]` to `llm.invoke(...)`.

Returns `{generated_document: llm_response.content}`.

---

#### Node 4: `quality_gate`

**LLM:** `llm` (kimi-k2) — for the LLM review portion only; structural checks are deterministic

**Purpose:** Validate the generated document. Sets `status` to `"passed"` or `"failed"`.

**Table-only path (fully deterministic):**
1. Split `generated_document` into lines
2. Find `# ` heading line -> `heading_line`
3. Find all `|`-prefixed table lines -> `table_lines`
4. If `len(table_lines) < 3`: return `status="failed"`, issue: `"TABLE-ONLY SCHEMA: No Markdown table found. Output ONLY: # {doc_name} + a table with columns: {columns}"`
5. Parse `header_line = table_lines[0]` -> `actual_columns`
6. Compare `[col.lower() for col in expected_columns]` vs `[col.lower() for col in actual_columns]`
7. If mismatch: return `status="failed"`, issue: `"Wrong columns. Expected: | ... | Got: | ... |"`
8. **On pass: reconstruct the output** as `heading_line + "\n\n" + "\n".join(table_lines)` — this strips any prose that leaked into the generation. Returns `quality_scores={"structure": 5, "completeness": 5}`, `status="passed"`

**Mixed path:**

Step 1 — Structural check (deterministic):
```python
structure_errors = validate_document_structure(document_text, required_section)
```
If errors found: categorise as missing/extra/table errors and build targeted suggestions:
- `missing` errors -> `"Add ALL missing sections using their EXACT titles from the schema — do not rename them."`
- `extra` errors -> `"REMOVE every heading not in the schema. The document must contain ONLY the sections defined in the schema — nothing more."`
- `table` errors -> `"Ensure every table section contains a real Markdown table with the exact column headers specified in the schema."`
Return `status="failed"` with `quality_scores={"structure": 1}`.

Step 2 — LLM quality review (if structural checks pass):
```python
review_prompt = build_quality_review_prompt(department, document_type, document_text)
messages = [SystemMessage(review_prompt), HumanMessage("Review the document and return the JSON assessment now.")]
review_response = llm.invoke(messages)
```
Parses JSON response (strips markdown fences first). Uses `review_result["passed"]` if present, otherwise derives from `overall_score >= 3`.

Step 3 — Fallback rule-based checks (if LLM review throws any exception):
- Document length < 500 characters
- Contains forbidden placeholders: `TBD`, `to be decided`, `[Company Name]`, `[Insert`, `Lorem ipsum`
- Fewer than 5 headings (`document_text.count("\n#") < 5`)
- Any `## `-split section with < 100 chars of content (thin sections)

---

#### Node 5: `fix_document`

**LLM:** `llm` (kimi-k2, same instance as Node 3)

**Purpose:** Rewrite the document to fix all quality gate failures identified in Node 4.

Constructs a `fix_instruction` HumanMessage that embeds:
- The full original document between `--- DOCUMENT START ---` and `--- DOCUMENT END ---` markers
- All quality issues formatted as a bullet list: `- {issue_msg}`
- All reviewer suggestions formatted as a bullet list: `- {suggestion_msg}`
- 6 explicit fix instructions: (1) fix all issues, (2) expand thin sections, (3) 2-3 sentences minimum per section, (4) remove all placeholder text, (5) add concrete metrics/timelines/action items, (6) output only corrected Markdown

Sends `[SystemMessage(state["system_prompt"]), HumanMessage(fix_instruction)]` to `llm`.

Returns `{generated_document: llm_response.content, retry_count: state["retry_count"] + 1}`.

After Node 5, the graph always routes back to Node 4. If `retry_count >= 2`, `decide_after_quality_gate` routes to `END` regardless of the quality gate result on that final pass.

---

#### Graph Assembly

```python
def build_document_generation_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("analyze_schema_gaps", analyze_schema_gaps)
    graph.add_node("build_prompt",        build_prompt)
    graph.add_node("generate_document",   generate_document)
    graph.add_node("quality_gate",        quality_gate)
    graph.add_node("fix_document",        fix_document)

    graph.set_entry_point("analyze_schema_gaps")
    graph.add_edge("analyze_schema_gaps", "build_prompt")
    graph.add_edge("build_prompt",        "generate_document")
    graph.add_edge("generate_document",   "quality_gate")
    graph.add_conditional_edges(
        "quality_gate",
        decide_after_quality_gate,
        {"fix_document": "fix_document", "end": END},
    )
    graph.add_edge("fix_document", "quality_gate")

    return graph.compile()

document_generation_agent = build_document_generation_graph()  # compiled once at import
```

---

#### Public Async Entry Points

**`async run_agent(department, document_type, questions_and_answers, required_section) -> dict`**

Full 5-node execution. Uses `asyncio.to_thread` to run the synchronous LangGraph `invoke` in a thread pool, freeing the FastAPI event loop during the 30-60s generation:
```python
final_state = await asyncio.to_thread(document_generation_agent.invoke, initial_state)
```
Returns: `{generated_document, gap_questions, status, quality_issues, quality_scores, quality_suggestions, retry_count}`.

**`async analyze_gaps_only(department, document_type, questions_and_answers, required_section) -> list[dict]`**

Runs only Node 1 in isolation — not through the graph. Constructs a minimal `AgentState` with empty output fields, calls `analyze_schema_gaps(state)` directly, returns `result.get("gap_questions", [])`. Used by `POST /gap-questions` to run gap analysis without triggering document generation.

**`async generate_single_section(department, document_type, section, questions_and_answers, doc_memory) -> str`**

Progressive generation entry point. Used by `POST /generate-section`.

Detailed logic:
1. Strip `_`-prefixed internal keys from subsection dicts (e.g. `_parent_title` injected by the UI) using `{k: v for k, v in sub.items() if not k.startswith("_")}`
2. Build `scoped_required_section` — a minimal schema with only the target subsection, using the subsection's own title (not the parent title) as the section title. This prevents the LLM from generating the full document hierarchy for every single step.
3. Prepend two special Q&A entries to `enriched_qa`:
   - `SCOPE CONSTRAINT` entry (category `"_scope"`): instructs the LLM to generate ONLY the named subsection(s), not repeat previously generated content, and not add other headings
   - If `doc_memory` is non-empty: `Previously generated sections` entry (category `"_memory"`): injects the combined text of all previously generated sections so the LLM maintains consistent terminology and decisions
4. Build the full `initial_state` with the scoped schema and enriched Q&A
5. `await asyncio.to_thread(document_generation_agent.invoke, initial_state)` — runs the full 5-node graph, so quality gate and fix retry apply per-section
6. Returns `final_state.get("generated_document", "")`

---

### `ui/api_helpers.py`

Thin synchronous HTTP wrappers around every FastAPI endpoint. Zero Streamlit dependency — can be imported and used by any Python client.

**Base URL:** `FASTAPI_URL = "http://127.0.0.1:8000"` (module constant, overridable via `base_url` parameter on each function).

All functions use `requests` (synchronous HTTP, acceptable because Streamlit's execution model is synchronous). Each function: logs request parameters and response counts, returns `None` or `[]` on any exception (never re-raises), and has an explicit timeout.

| Function | Method + Endpoint | Timeout | Return on success |
|----------|------------------|---------|------------------|
| `fetch_departments()` | GET /departments | 10s | list[dict] |
| `fetch_document_types(department)` | GET /document-types | 10s | list[dict] |
| `fetch_questions(document_type)` | GET /questions | 10s | list[dict] |
| `fetch_notion_page_urls()` | GET /get_all_urls | 30s | list[dict] |
| `call_gap_questions_endpoint(department, document_type, document_name, questions_and_answers)` | POST /gap-questions | 60s | dict |
| `call_save_questions_endpoint(department_obj, document_type, document_name, gap_questions)` | POST /save-questions | 30s | dict |
| `call_generate_endpoint(department, document_type, document_name, questions_and_answers)` | POST /generate | 120s | dict |
| `call_generate_section(department, document_type, section, questions_and_answers, doc_memory)` | POST /generate-section | 90s | dict |

---

### `ui/question_helpers.py`

Pure Python helpers (mostly) for building and querying the unified question list. The only Streamlit-dependent function is `render_question_widget`.

**`QuestionTuple`** type alias: `tuple[str, dict, dict, bool]` = `(widget_key, question_dict, answer_state, is_gap_flag)`

#### `build_unified_question_list(questions, answers_state, gap_questions, gap_answers_state) -> list[QuestionTuple]`

Merges three sources into one ordered flat list:
1. **Core questions** (`is_gap_question` not set or False from `questions` list): `widget_key = f"answer_{idx}"`, `answer_state = answers_state`, `is_gap = False`
2. **MongoDB-persisted gap questions** (`is_gap_question=True` in the `questions` list): `widget_key = f"answer_{idx}"`, `answer_state = answers_state`, `is_gap = True`
3. **Session gap questions** (freshly generated, in `gap_questions` + `gap_answers_state`): `widget_key = f"gap_answer_{idx}"`, `answer_state = gap_answers_state`, `is_gap = True`

This design means core and MongoDB gap answers live in `st.session_state.answers` (same namespace), while session gap answers live in `st.session_state.gap_answers` (separate namespace). When gap questions are saved and the cache clears, they migrate from source 3 to source 2 on next load.

#### Category Helpers

**`build_ordered_categories(all_questions) -> list[str]`**
Iterates the unified list and collects category names in first-appearance order. Uses a `seen` set keyed by lowercased category name for deduplication, but preserves the original casing in the output list.

**`build_category_to_subsection_map(all_questions, schema_subsections) -> dict[str, dict]`**
Builds `{category_lower: subsection_dict}` for categories that have a matching `title` in `schema_subsections`. Used for progress tracking and category-to-schema alignment in the progressive mode section headers.

**`get_page_categories(page_idx, all_questions, page_size=5) -> list[str]`**
Slices `all_questions[page_start:page_end]` and returns the unique category names for that page's 5 questions.

#### Q&A Collection

**`get_subsection_qa(category_name, all_questions) -> list[dict]`**
Filters `all_questions` by category (lowercased match), keeps only questions with non-empty `answer_state[widget_key]`, returns `[{question, answer, category, answer_type}]`.

**`collect_all_answered_qa(all_questions) -> list[dict]`**
Collects all answered Q&A across the entire unified list. Used as the input to `POST /generate` (single-shot) and `POST /generate-section` (progressive).

**`collect_page_answered_qa(page_idx, all_questions, page_size=5) -> list[dict]`**
Collects answered Q&A for the 5 questions on the given page only.

#### Answer Checking

**`question_has_answer(widget_key, answer_state) -> bool`**
Priority check:
1. Checks `st.session_state[f"widget_{widget_key}"]` for the live widget value from the current Streamlit run
2. For list values (multiselect): `len(live_value) > 0`
3. For string values: `str(live_value).strip() != ""`
4. Falls back to `str(answer_state.get(widget_key, "")).strip() != ""`

This two-step check is necessary because Streamlit widgets update `st.session_state` with their key at the start of a run, before the widget is rendered again. Without this, the progress bar would show stale counts.

#### Widget Rendering

**`render_question_widget(question, widget_key, answer_state, is_gap=False)`**

Renders one question as the correct Streamlit input type based on `question["answer_type"]`:

| `answer_type` | Widget | Notes |
|--------------|--------|-------|
| `"text"` (default) | `st.text_area` | Pre-populated from `answer_state.get(widget_key, "")` |
| `"structured_list"` | `st.text_area` | Same as text; help text says "Enter items separated by newlines" |
| `"select"` | `st.selectbox` | Options from `question["options"]`; current value determines `index` |
| `"multi_select"` | `st.multiselect` | Current value split by `", "` to compute `default`; stored as `", "`-joined string |

**Gap question rendering:** When `is_gap=True`, renders a custom HTML label via `st.markdown()`:
```python
st.markdown(
    f"<span style='font-size:0.9rem;font-weight:600;color:#eee;'>"
    f"{question_label} <span class='gap-badge'>AI</span></span>",
    unsafe_allow_html=True,
)
label_for_widget = "\u200b"   # zero-width space (Streamlit requires non-empty label)
```

All widgets use Streamlit key `f"widget_{widget_key}"`. The widget's value is written back to `answer_state[widget_key]` on each render cycle.

---

### `ui/pdf_generator.py`

Converts a Markdown string to a professionally styled A4 PDF using ReportLab Platypus. No Streamlit dependency — usable in any Python context.

#### Colour Palette

| Variable | Hex | Usage |
|----------|-----|-------|
| `_PDF_BRAND` | `#1E3A5F` | H1 headings, document title, H1 horizontal rules |
| `_PDF_ACCENT` | `#2E86AB` | H2/H3 headings, table header background, accent rules |
| `_PDF_MID` | `#8C9BAB` | Subtitle text, metadata |
| `_PDF_DARK` | `#1A1A2E` | All body text |
| `_PDF_TH_BG` | `#2E86AB` | Table header cell background |
| `_PDF_ROW_A` | `#F0F5FA` | Alternating table row background (even rows) |
| `_PDF_BORDER` | `#CBD5E1` | Table cell borders |

#### `_build_pdf_styles() -> dict`

Returns named `ParagraphStyle` instances:

| Key | Font | Size | Notes |
|-----|------|------|-------|
| `doc_title` | Helvetica-Bold | 24 | Document title, centred, `_PDF_BRAND` colour |
| `doc_sub` | Helvetica | 10 | "Generated by DocForgeHub" subtitle, centred |
| `h1` | Helvetica-Bold | 16 | `# ` headings, followed by `HRFlowable(_PDF_BRAND, 1pt)` |
| `h2` | Helvetica-Bold | 13 | `## ` headings, `_PDF_ACCENT` colour |
| `h3` | Helvetica-BoldOblique | 11 | `### ` headings, `_PDF_ACCENT` colour |
| `body` | Helvetica | 10 | Normal paragraphs, numbered lists |
| `bullet` | Helvetica | 10 | Bullet items, `leftIndent=14` |

#### `clean_text_for_pdf(text: str) -> str`

Applies the `_UNICODE_REPLACEMENTS` dict (30 entries): em-dash -> `-`, curly quotes -> straight, `>=`/`<=`, `!=`, arrows (`->`, `<-`, `=>`), Greek letters (`alpha`, `beta`, `mu`, `sigma`, `pi`), currency symbols (`INR`, `EUR`, `GBP`), degree/superscript, checkmarks, etc. Then encodes to `latin-1` with `errors="replace"` to drop any remaining non-latin characters that ReportLab cannot render.

#### `_build_paragraph(text: str, style) -> Paragraph`

Calls `clean_text_for_pdf`, then applies three regex substitutions for inline Markdown:
- `**bold**` -> `<b>bold</b>`
- `*italic*` -> `<i>italic</i>`
- `` `code` `` -> `<font name='Courier'>code</font>`

Returns a `Paragraph` flowable.

#### `parse_markdown_table(lines: list[str]) -> list[list[str]]`

Skips separator rows matched by `re.match(r"^\|[\s\-\|:]+\|$")`. Splits remaining rows on `|`, strips whitespace from each cell, discards empty cells (from leading/trailing `|`). Returns a 2D list of strings.

#### `build_reportlab_table(rows, pdf_styles) -> Table | None`

Pads all rows to `max_cols` width with `""` cells. Renders header row cells as `<b>clean_text</b>` Paragraphs. Body cells as plain body Paragraphs. Column widths: `(A4[0] - 5cm) / max_cols` each.

`TableStyle` applied:
- Blue header background (`_PDF_TH_BG`), white Helvetica-Bold text, font size 9
- Alternating row backgrounds: `[_PDF_ROW_A, colors.white]`
- `_PDF_BORDER` grid (`0.4pt` thickness)
- `7pt` left/right padding, `5pt` top/bottom padding
- `repeatRows=1` (header re-printed on page breaks)
- `VALIGN=TOP` throughout

#### `generate_pdf_from_markdown(markdown_text, document_title="") -> bytes`

Creates a `SimpleDocTemplate` (A4, `2.5cm` margins all sides). If `document_title` is non-empty, prepends: `Spacer(0.8cm)` -> `doc_title` Paragraph -> `HRFlowable(accent, 2pt)` -> `doc_sub` Paragraph -> `Spacer(0.8cm)`.

Main loop: `while line_idx < len(lines)` — explicit index loop (not `for`) to allow the table collector to consume multiple lines in one iteration:
- Empty line -> `Spacer(1, 4)`
- Line starting with `|`: greedily collect all contiguous pipe lines, call `build_reportlab_table`, add `Spacer(1, 6) + table + Spacer(1, 10)`
- `"# "` -> `h1` Paragraph + `HRFlowable(_PDF_BRAND, 1pt, spaceAfter=4)`
- `"## "` -> `h2` Paragraph
- `"### "` -> `h3` Paragraph
- `re.match(r"^[\*\-\+]\s+")` -> `bullet` Paragraph with `"• "` prepended, original bullet stripped
- `re.match(r"^\d+\.\s+")` -> `body` Paragraph
- Everything else -> `body` Paragraph

Returns `pdf_buffer.getvalue()` (raw PDF bytes).

#### `build_safe_pdf_filename(title: str) -> str`

`re.sub(r"[^\w\s\-]", "", title)` -> strip non-word chars -> `re.sub(r"\s+", "_", safe)` -> append `".pdf"`. Defaults to `"document.pdf"` if result is empty.

---

### `ui/streamlit_uidemo.py`

The main Streamlit application. Everything visible to the user is defined here.

#### Startup and Path Setup

```python
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
```
Ensures `agent.*`, `api.*`, and sibling `ui/` modules are importable regardless of working directory (whether launched as `streamlit run ui/streamlit_uidemo.py` from root or as `streamlit run streamlit_uidemo.py` from within `ui/`).

```python
st.set_page_config(page_title="DocForgeHub", page_icon="📄", layout="wide")
```

Custom CSS injected via `st.markdown(..., unsafe_allow_html=True)`:
- `.block-container { padding-top: 1rem }` — reduces default top whitespace
- `.separator-right` / `.separator-left` — `1px solid #444` panel dividers via CSS border
- `.gap-banner` — dark gradient banner (`#1a1a2e` to `#16213e`) with `4px solid #e94560` left border for gap question section announcements
- `.gap-badge` — inline red pill badge (`background: #e94560`, `border-radius: 10px`, `font-size: 0.68rem`) displayed next to gap question labels

#### Cached Data Fetchers

```python
@st.cache_data(ttl=300)   # 5-minute cache
def get_departments_from_fastapi() -> list:
    return fetch_departments()

@st.cache_data(ttl=300)
def get_document_types_from_fastapi(department_name) -> list:
    return fetch_document_types(department_name)

@st.cache_data(ttl=300)
def get_questions_from_fastapi(document_type) -> list:
    return fetch_questions(document_type)

@st.cache_data(ttl=600)   # 10-minute cache
def get_notionpage_urls_from_fastapi() -> list:
    return fetch_notion_page_urls()
```

Cache invalidation: `get_questions_from_fastapi.clear()` is called explicitly after a successful `POST /save-questions` so the newly persisted gap questions appear in the next question load.

#### Session State Keys (Complete Reference)

| Key | Type | Initial Value | Purpose |
|-----|------|---------------|---------|
| `history` | list | Notion pages | Sidebar generation history |
| `answers` | dict | `{}` | `{f"answer_{idx}": str}` for core + MongoDB gap questions |
| `gap_answers` | dict | `{}` | `{f"gap_answer_{idx}": str}` for session gap questions |
| `gap_questions` | list | `[]` | Freshly generated gap questions (not yet saved to MongoDB) |
| `markdown_doc` | str | `""` | Current document content shown in editor |
| `is_generating` | bool | `False` | Disables generate button during API call |
| `is_analyzing` | bool | `False` | Disables analyse button during API call |
| `is_saving` | bool | `False` | Disables save button during API call |
| `gap_source` | str | `""` | `"cache"` or `"generated"` — determines if save button is shown |
| `gap_doc_type` | str | `""` | Which document's gap questions are loaded; triggers auto-clear on doc change |
| `prog_mode` | bool | `False` | True = Progressive mode active |
| `q_page` | int | `0` | Current question page index |
| `prog_sections` | dict | `{}` | `{subsection_title: generated_text}` — progressive mode state |
| `prog_generating` | bool | `False` | Disables per-subsection generate button during API call |
| `prog_current_step` | int | `0` | Index of last generated subsection (used for sequential reveal) |

**Gap question auto-clear on document change:**
```python
if st.session_state.gap_doc_type and st.session_state.gap_doc_type != selected_document:
    st.session_state.gap_questions = []
    st.session_state.gap_answers   = {}
    st.session_state.gap_source    = ""
    st.session_state.gap_doc_type  = ""
```

**Progressive mode stale-key guard:**
```python
if any(isinstance(k, int) for k in st.session_state.prog_sections):
    st.session_state.prog_sections = {}
```
Integer-keyed entries from older session versions are purged on every render.

#### Sidebar Layout

```python
with st.sidebar:
    st.write("<h1>DOC DocForge Hub</h1>", unsafe_allow_html=True)
    selected_department = st.selectbox("Department", department_names)
    selected_document   = st.selectbox("Document", document_names)
    mode_choice = st.radio("Mode", ["Single-Shot", "Progressive"])
    st.session_state.prog_mode = (mode_choice == "Progressive")
    # Notion history (scrollable container, height=270)
    for item in st.session_state.history:
        st.markdown(f"<a href='{url}' ...>{title}</a>", unsafe_allow_html=True)
```

Two lookup dicts are built once per render from the doc types list:
- `document_name_lookup`: `{document_type -> document_name}` — for schema fetches and API payloads
- `department_obj_lookup`: `{dept_name -> {code, name, slug}}` — for save-questions department payload

#### Progressive Mode Schema Loading

When `prog_mode=True` and a valid document is selected, the app makes a direct (uncached) `requests.get(f"{FASTAPI_URL}/required-section", ...)` call to build two data structures:

```python
all_subsections: list[dict]   # flat, ordered subsections with _parent_title injected
subsection_titles: list[str]  # ordered subsection title strings for reveal/generate buttons
```

Building logic:
- For sections with `subsections[]`: iterates sorted subsections, injects `_parent_title = section["title"]`
- For table-only or flat sections: the section itself becomes the single subsection entry with `_parent_title = section["title"]`

#### Pagination

```python
PAGE_SIZE = 5
total_pages = max(1, -(- total_questions // PAGE_SIZE))   # ceiling division
```

`q_page` is clamped to `[0, total_pages-1]` on every render. Page questions slice: `all_questions[page_start:page_end]`.

Two helper functions:
- `get_sec_idx_for_page(page_idx, sections)`: maps page index to schema section index proportionally: `min(int(page_idx / total_pages * len(sections)), len(sections)-1)`
- `get_section_qa_for_sec_idx(section_idx, sections)`: collects answered Q&A for a schema section using three-way category matching (exact category lower, section title lower, or title contained-in-category). Falls back to proportional page slice if no category matches.

#### Questions Panel (`col_questions` — 2/5 width)

1. **Section header**: progressive mode shows `"Page N of M"` + `"Covers: subsection1, subsection2"`; single-shot shows `"Questions"` + page caption
2. **Question widgets**: `for (widget_key, question_data, answer_state, is_gap) in page_questions: render_question_widget(...)`
3. **Progress bar**: `st.progress(answered_count / total_questions)` using live `question_has_answer()` check
4. **Gap question controls** (evaluated each render):
   - If session gap questions exist AND `gap_source == "generated"`: show `"Save gap questions"` button
     - On click: collects answers from `gap_answers`, calls `call_save_questions_endpoint`, shows success/error message, clears question cache
   - Else if no gap questions at all: show `"Analyse schema gaps"` button
     - On click: builds core-only Q&A list, calls `call_gap_questions_endpoint`, populates session gap state, calls `st.rerun()`
5. **Navigation row**: `[Back] [Generate/Finalize/Lock] [Next]`

**Single-shot generate button**: `disabled=not (answered_count == total_questions > 0)`. Label changes to `"Answer all questions first (N remaining)"` when locked.

**Progressive finalize button**: visible only on last page AND `all(t in prog_sections for t in subsection_titles)`. Until all sections are generated, shows `st.info(f"Generate all sections in the preview panel first ({remaining_subs} remaining).")`.

#### Generation Handlers

**Single-shot generation:**
1. Builds `questions_and_answers` from three sources in order: core questions, MongoDB-persisted gap questions (both from `questions` list, filtered by `is_gap_question`), then session gap questions
2. Calls `call_generate_endpoint(department, document_type, document_name, qa_list)` with spinner `"Agent is generating your document... This may take 30-60 seconds."`
3. Stores `result["generated_document"]` in `st.session_state.markdown_doc`
4. If `result["gap_questions"]` non-empty and no session gap questions yet: auto-populates gap state and shows `st.info(f"{N} gap question(s) detected. Answer them and regenerate.")`
5. Shows `st.success` (passed) or `st.warning` (failed with issues), includes retry count
6. If `quality_scores` non-empty: renders `st.expander("Quality Scores")` with one `st.metric` per score

**Progressive finalize:**
1. For each subsection in `all_subsections` not yet in `prog_sections`: calls `call_generate_section(...)` individually with a spinner per section
2. Stitches all `prog_sections` values in `subsection_titles` order: `"\n\n".join(prog_sections[t] for t in subsection_titles if t in prog_sections)`
3. Stores in `markdown_doc`, shows `"Document finalized! View it in the preview panel"`

#### Editor Panel (`col_editor` — 3/5 width)

**Progressive mode sequential reveal:**
- Shows already-generated subsections as `st.expander(f"sub_title", expanded=False)` blocks (always uses schema title, not LLM-generated heading, for the expander label)
- Progress bar + step counter for the current generation position
- `"Generate: {next_sub_title}"` primary button (disabled if `prog_generating`, disabled if not all answers filled)
- On click: calls `call_generate_section(...)` with `doc_memory` = all previously generated sections joined, stores result in `prog_sections`, increments `prog_current_step`, calls `st.rerun()`
- Combined editable preview (`st.text_area`, height=300) of all generated content so far

**Single-shot mode:** Single `st.text_area` (height=450) for the full document.

**Both modes (when `markdown_doc` non-empty):**
- `st.expander("Preview rendered document", expanded=False)` -> `st.markdown(markdown_doc)` — renders the Markdown as formatted HTML
- PDF section: calls `generate_pdf_from_markdown(markdown_doc, selected_document)` inline; `st.download_button("Download PDF", data=pdf_bytes, mime="application/pdf")`; wraps in try/except to show warning if PDF generation fails

**Progressive reset button** (visible only in progressive mode with generated sections):
```python
st.button("Reset Progressive Session")
# Clears: prog_sections, prog_current_step, q_page, markdown_doc
# Calls: st.rerun()
```

---

## Complete Data Flows

### Flow 1: Full Single-Shot Document Generation

```
User fills all questions -> clicks "Generate Document"
|
+- streamlit_uidemo.py:
|  Build qa_list (core + MongoDB gaps + session gaps)
|  call_generate_endpoint(department, document_type, document_name, qa_list)
|
+- api_helpers.py: POST http://127.0.0.1:8000/generate  (timeout=120s)
|
+- api/main.py -- /generate handler:
|  1. required_section not in request?
|     -> db["required_section"].find_one({department, document_name})
|  2. await run_agent(department, document_type, qa_list, required_section)
|
+- agent_graph.py -- run_agent():
|  asyncio.to_thread(document_generation_agent.invoke, initial_state)
|
|  +-- NODE 1: analyze_schema_gaps ----------------------------------------+
|  | format_required_section_for_prompt()                                    |
|  | format_questions_and_answers_for_prompt()                               |
|  | question_gen_llm.invoke([SystemMessage, HumanMessage])                  |
|  | -> JSON parse -> gap_questions list                                      |
|  | -> supplementary_content placeholder notes for uncovered sections       |
|  +---------------------------------------------------------------------------+
|         |
|  +-- NODE 2: build_prompt ------------------------------------------------+
|  | is_table_only_schema()?                                                  |
|  |   YES -> build_table_only_prompt(columns, ...)                          |
|  |   NO  -> format_required_section_for_prompt()                           |
|  |          + extract subsection titles -> strict_rule string               |
|  |          + build_system_prompt(schema + strict_rule, qa, supplementary) |
|  | -> state.system_prompt set; retry_count=0; status="generating"          |
|  +---------------------------------------------------------------------------+
|         |
|  +-- NODE 3: generate_document -------------------------------------------+
|  | llm.invoke([SystemMessage(system_prompt), HumanMessage(instruction)])    |
|  | -> state.generated_document = full Markdown string                       |
|  +---------------------------------------------------------------------------+
|         |
|  +-- NODE 4: quality_gate ------------------------------------------------+
|  | is_table_only_schema()?                                                  |
|  |   YES -> deterministic column check                                      |
|  |          pass: reconstruct output, status="passed"                       |
|  |          fail: wrong columns, status="failed"                            |
|  |   NO  -> validate_document_structure() -> structural errors?             |
|  |            YES -> categorise errors, status="failed"                     |
|  |            NO  -> llm quality review -> JSON scores                      |
|  |                   pass: status="passed"                                  |
|  |                   fail: status="failed" with issues + suggestions        |
|  |                   exception -> fallback rule-based checks                |
|  +---------------------------------------------------------------------------+
|         |
|      status == "passed"  -----------------------------------------> END
|         |
|      status == "failed" AND retry_count < 2
|         |
|  +-- NODE 5: fix_document -----------------------------------------------+
|  | Build fix_instruction: full doc + issues + suggestions + 6 rules        |
|  | llm.invoke([SystemMessage(system_prompt), HumanMessage(fix_instruction)])|
|  | -> updated generated_document; retry_count += 1                         |
|  +---------------------------------------------------------------------------+
|         |
|       -> back to NODE 4  (loops max 2x, then END regardless of status)
|
+- api/main.py returns:
|  {generated_document, gap_questions, status, quality_issues,
|   quality_scores, quality_suggestions, retry_count}
|
+- streamlit_uidemo.py:
   st.session_state.markdown_doc = result["generated_document"]
   Show quality scores expander
   PDF download button appears
```

---

### Flow 2: Cache-First Gap Analysis + Save

```
User clicks "Analyse schema gaps"
|
+- streamlit_uidemo.py:
|  Build current_qa (core questions only, no gap questions)
|  call_gap_questions_endpoint(department, document_type, document_name, qa)
|
+- api_helpers.py: POST /gap-questions  (timeout=60s)
|
+- api/main.py -- /gap-questions:
|  STEP 1 -- Cache check:
|    db["document_qas"].find_one({document_type, is_gap_question: True})
|    +-- FOUND:
|    |   fetch all gap Q for doc type (sorted by question_order)
|    |   return {gap_questions, source: "cache", count}  <- zero LLM calls
|    +-- NOT FOUND:
|  STEP 2 -- Schema fetch:
|    required_section in request? -> use it
|    else -> db["required_section"].find_one({department, document_name})
|            fallback to {"sections": []}
|  STEP 3 -- LLM analysis:
|    await analyze_gaps_only(department, document_type, qa_list, schema)
|      -> calls analyze_schema_gaps(state) directly (not through graph)
|      -> llama-3.3-70b -> JSON array -> gap_questions list
|    return {gap_questions, source: "generated", count}
|
+- streamlit_uidemo.py:
   gap_questions_from_api stored in st.session_state.gap_questions
   gap_source stored in st.session_state.gap_source
   gap_doc_type = selected_document
   Initialize gap_answers dict entries
   st.rerun()  -- gap questions now appear in the Q&A panel with AI badges

User answers gap questions -> clicks "Save gap questions"
|
+- streamlit_uidemo.py:
|  Collect {**gap_question, answer: gap_answers[key]} for each gap question
|  dept_obj = department_obj_lookup[selected_department]
|  call_save_questions_endpoint(dept_obj, document_type, document_name, gap_qa)
|
+- api/main.py -- /save-questions:
|  MAX(question_order) aggregation -> base_order
|  For each gap question: upsert into document_qas with is_gap_question=True,
|    category_order=999, question_order=base_order+1000+i, answered_at=now
|  return {saved, updated, total}
|
+- streamlit_uidemo.py:
   get_questions_from_fastapi.clear()  -- cache invalidated
   st.success(f"Saved {saved} new, updated {updated}")
   [Next user fetching /questions gets gap questions as part of the standard list]
```

---

### Flow 3: Progressive Section-by-Section Generation

```
User selects Progressive mode -> schema loaded automatically:
|
+- streamlit_uidemo.py:
   requests.get(f"{FASTAPI_URL}/required-section", params={...})
   -> all_subsections: [{..., "_parent_title": parent}]  (flat, ordered)
   -> subsection_titles: ["1.1 Business Impact", "1.2 Success Metrics", ...]

User answers all questions -> editor panel shows:
  "Next: 1.1 Business Impact" with [Generate] button

User clicks "Generate: 1.1 Business Impact"
|
+- streamlit_uidemo.py:
|  all_qa = collect_all_answered_qa(all_questions)
|  section_for_api = {title: "_parent_title", subsections: [next_sub_entry]}
|  doc_memory = join all previously generated sections
|  call_generate_section(department, document_type, section_for_api, all_qa, doc_memory)
|
+- api_helpers.py: POST /generate-section  (timeout=90s)
|
+- api/main.py -- /generate-section:
|  await generate_single_section(...)
|
+- agent_graph.py -- generate_single_section():
|  1. Strip _-prefixed keys from subsection dicts
|  2. Build scoped_required_section:
|       sections: [{title: sub_title, subsections: [clean_sub]}]
|  3. Prepend to enriched_qa:
|       - SCOPE CONSTRAINT entry (generate ONLY this subsection)
|       - Previously generated sections entry (doc_memory for consistency)
|  4. asyncio.to_thread(document_generation_agent.invoke, scoped_initial_state)
|     -> full 5-node graph runs on this single subsection
|     -> quality gate + fix retry apply per-section
|  return final_state["generated_document"]
|
+- streamlit_uidemo.py:
   prog_sections["1.1 Business Impact"] = section_result["section_text"]
   prog_current_step = 1
   st.rerun()
   -- "1.1 Business Impact" expander appears (collapsed)
   -- "Next: 1.2 Success Metrics" button revealed

... repeat for each subsection in subsection_titles order ...

All sections generated -> "Finalize Document" button appears (last page):
|
+- streamlit_uidemo.py:
   full_doc = "\n\n".join(prog_sections[t] for t in subsection_titles)
   st.session_state.markdown_doc = full_doc
   st.success("Document finalized! View it in the preview panel")
   PDF download button appears
```

---

## Design Decisions

### Why Two LLMs?

The system separates tasks by model:

**kimi-k2-instruct-0905** (temp=0.1, max=8192): Long-form prose quality for document generation. The 200K context window handles large documents with many Q&A pairs. Low temperature (0.1) produces consistent, professional output. Used by Nodes 3, 4, 5 — the generation and validation steps where prose quality and instruction-following matter most.

**llama-3.3-70b-versatile** (temp=0.2, max=2048): Fast, cheap, strong at structured JSON output. Gap analysis returns a small JSON array — 2048 tokens is more than sufficient. Temperature 0.2 allows slight variation in question phrasing while keeping output structured. Used by Node 1 and `analyze_gaps_only` — the analysis step where structured output matters more than prose quality.

Cost implication: Node 1 calls llama-3.3-70b (~5x cheaper per token than kimi-k2) and the cache-first `/gap-questions` design means it converges to zero LLM calls per document type over time.

### Why `asyncio.to_thread` in `run_agent`?

LangGraph's `graph.invoke()` is synchronous. FastAPI runs on an asyncio event loop. A blocking synchronous call lasting 30-60s would prevent all other requests from being served while the agent runs. `asyncio.to_thread` moves the blocking execution to Python's default thread pool executor, freeing the event loop to handle other requests concurrently.

### Why Motor for MongoDB Instead of pymongo?

FastAPI async handlers run on the event loop. Using synchronous pymongo inside an async handler would block the event loop during every database operation. Motor wraps the same pymongo driver with asyncio compatibility, enabling `await db.collection.find_one(...)` calls that yield back to the event loop while waiting for MongoDB — critical for a server expected to handle multiple concurrent Streamlit sessions.

### Why Schema-Driven Validation Rather Than Pure LLM Trust?

LLMs reliably hallucinate: they add extra sections not in the schema, skip requested sections, rename headings, and produce prose where a table was required. The deterministic `validate_document_structure()` check runs before any LLM review — enforcing structural correctness (exact heading presence/absence, table column matching) with programmatic string comparison. The LLM quality review layer then evaluates the softer qualities (professional tone, depth, actionability) that are genuinely better assessed semantically. This two-layer approach means structural errors never reach users and are automatically fixed, while prose quality issues are surfaced with actionable suggestions.

### Why Upsert for Gap Questions?

If a user triggers gap analysis, answers questions, clicks Save, then later answers more questions and clicks Save again, the upsert on `{document_type, question, is_gap_question: True}` ensures idempotency — the same gap question is updated with the new answer rather than duplicated. The `saved / updated / total` breakdown in the response gives the UI precise feedback. The `MAX(question_order) + 1000 + i` ordering scheme ensures new gap questions always sort after existing ones regardless of when they were added.

### Why Separate `answers` and `gap_answers` Session State?

Core and MongoDB-persisted gap questions both arrive from `GET /questions` and are indexed together by `answer_{idx}` into `st.session_state.answers`. Session gap questions (freshly generated, not yet in MongoDB) live in a separate `st.session_state.gap_answers` dict keyed by `gap_answer_{idx}`.

This separation provides two benefits:
1. When the user saves gaps and the cache clears, the next `GET /questions` call returns the newly saved gaps as regular questions in the main list — they automatically migrate from the `gap_answers` namespace to the `answers` namespace without any explicit migration logic
2. The save button handler can easily collect "only the session gaps" by iterating `st.session_state.gap_questions` and looking up answers in `gap_answers`

### Why `generate_single_section` Runs the Full 5-Node Graph?

Progressive generation uses the same `document_generation_agent` graph as single-shot generation, scoped to a single subsection via `scoped_required_section` and `enriched_qa` with scope constraints. This means every section benefits from the same quality gate and fix retry mechanism as a full document — a section that fails structural validation is automatically rewritten before the user sees it. The alternative (a lightweight single-LLM-call path for sections) would produce inconsistent quality and bypass the validation guarantees the system is built around.

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | Yes | Primary Groq API key (shared by both LLM instances) |
| `GROQ_API_KEY_2` ... `GROQ_API_KEY_7` | Optional | Fallback keys for manual rotation on rate-limit |
| `MONGODB_CONNECTION_STRING` | Yes | Atlas connection URI with credentials |
| `NOTION_API_KEY` | Yes | Notion integration secret for page URL traversal |

---

## Performance Characteristics

| Operation | Typical Time | Bottleneck |
|-----------|-------------|-----------|
| GET /departments (first call) | < 50ms | MongoDB aggregation |
| GET /departments (cached) | ~0ms | Streamlit st.cache_data |
| GET /questions (first call) | < 50ms | MongoDB cursor + list |
| GET /questions (cached) | ~0ms | Streamlit st.cache_data |
| POST /gap-questions (cache hit) | < 100ms | MongoDB find_one |
| POST /gap-questions (cache miss) | 10-15s | llama-3.3-70b LLM call |
| POST /save-questions | < 200ms | MongoDB upsert loop |
| POST /generate (0 retries) | 30-45s | kimi-k2 generation (Node 3) |
| POST /generate (1 retry) | 55-75s | kimi-k2 x2 (Nodes 3 + 5) |
| POST /generate-section | 15-30s | kimi-k2 scoped generation |
| PDF export | < 2s | ReportLab rendering (pure CPU) |

---

## Known Constraints and Limitations

- **Single-server only**: Streamlit sessions and FastAPI share one process. Horizontal scaling would require a message queue or external state store for `AgentState`.
- **Shared gap questions**: Gap questions are shared across all users of a document type — there is no per-user gap question isolation.
- **`GROQ_API_KEY` shared between LLMs**: Both kimi-k2 and llama-3.3-70b use the same key. Rotation to fallback keys (`GROQ_API_KEY_2` ... `7`) is manual.
- **Table-only schema format constraint**: Documents with `type: "table"` must declare it at the `sections[]` level with no `subsections`. Mixed schemas with table-only sections alongside text subsections in the same section are not supported.
- **Notion publishing is stubbed**: The Publish button in the editor panel shows a balloon animation but does not call a Notion write API.
- **Progressive mode requires schema**: Document types without a `required_section` MongoDB record silently fall back to empty `all_subsections`, making the Generate button non-functional in progressive mode.
- **Quality review JSON parse failures**: If kimi-k2 returns malformed JSON in Node 4, the system falls back to rule-based checks which are intentionally lenient to avoid excessive false rejections. A document that the rule-based checks pass may still have prose quality issues.
- **Node 1 is non-fatal**: If `analyze_schema_gaps` throws any exception, the agent continues with `gap_questions=[]` and `supplementary_content=""`. This means gap analysis failure never blocks document generation.