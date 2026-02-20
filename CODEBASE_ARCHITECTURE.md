# DocForgeHub — Codebase Architecture & Approach

## 🎯 Project Overview

**DocForgeHub** is an intelligent document generation and management system that:
- Extracts business documents from Notion
- Generates comprehensive Q&A from documents using LLMs
- Stores Q&As in MongoDB with department/category organization
- **Analyses schema coverage gaps and generates targeted questions to fill them**
- Generates polished, professional business documents from user answers using an agentic workflow
- Provides a Streamlit UI for document generation and management

---

## 🏗️ Architecture Stack

### Technology Stack
- **LLM Provider**: Groq (Kimi-k2 instruct for document generation; Llama-3.3-70b for gap analysis)
- **Agent Framework**: LangGraph (for multi-step workflows)
- **Backend API**: FastAPI (async, CORS-enabled)
- **Database**: MongoDB (async motor driver)
- **Frontend**: Streamlit
- **Document Management**: Notion API
- **Language**: Python

### Core Dependencies
- `langchain-groq` — LLM integration with Groq
- `langgraph` — State graph-based agent orchestration
- `fastapi` — REST API backend
- `motor` — Async MongoDB driver
- `streamlit` — Interactive UI
- `notion-client` — Notion API integration

---

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      STREAMLIT UI FRONTEND                       │
│  (Department / Document / Core Q&A / Gap Q&A / Generated View)  │
└─────────────────────┬───────────────────────────────────────────┘
                      │ HTTP REST
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FASTAPI BACKEND (Port 8000)                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ GET  /departments         → List all departments           │ │
│  │ GET  /document-types      → List docs for department       │ │
│  │ GET  /questions           → List Q&As (incl. gap Qs)       │ │
│  │ GET  /required-section    → Fetch schema from MongoDB      │ │
│  │ POST /gap-questions  ★NEW → Analyse gaps + generate Qs     │ │
│  │ POST /save-questions ★NEW → Persist gap Qs to MongoDB      │ │
│  │ POST /generate            → Trigger agentic document gen   │ │
│  │ GET  /get_all_urls        → Retrieve Notion page URLs      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                            │                                     │
│          ┌─────────────────┼─────────────────┐                  │
│          ▼                 ▼                 ▼                  │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │
│  │  Agent Graph │   │   MongoDB    │   │ Notion API   │        │
│  │   (Document  │   │    Client    │   │  (Page URLs) │        │
│  │  Generation) │   │              │   │              │        │
│  └──────────────┘   └──────────────┘   └──────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow Layers

### Layer 1: Data Extraction & Organization
**Files**: `automations/ques_automation.py`, `automations/automation.py`

1. **Notion Content Extraction** (`NotionContentExtractor`)
   - Connects to Notion via API
   - Recursively retrieves child pages organized by headings
   - Extracts markdown content from pages

2. **LangGraph-based Question Generation** (`GroqLangGraphQuestionGenerator`)
   - Implements a multi-node state graph workflow
   - Nodes:
     - `_analyze_and_detect`: Analyzes document structure & content patterns
     - `_generate_questions`: LLM-powered question generation
     - `_simple_validate`: Rule-based validation (no LLM call)
   - Resilient API calling with fallback across multiple Groq API keys
   - Outputs structured JSON Q&A files

3. **Output Organization**
   - Questions saved to `generated_questions/` by department
   - Structured as: `generated_questions/{department}/{document_name}_questions.json`

---

### Layer 2: Answer Field Addition & Filtering
**Files**: `automations/add_answer_field.py`

1. **QuestionAnswerProcessor**
   - Reads generated question files
   - Adds empty `answer` field to each question
   - Organizes by topics/categories

2. **Output Structure**
   - Final Q&As saved to `final_filtered_QAs/{department}/`
   - Format: `{document_name}_questions.json` with answer fields

---

### Layer 3: MongoDB Integration
**Files**: `automations/mongo_auto.py`, `api/db.py`

1. **DepartmentBasedMongoDBIntegration**
   - Reads Q&A files from `final_filtered_QAs/`
   - Batch inserts into MongoDB
   - Creates collections:
     - `document_qas`: Contains all Q&A pairs organized by department/document
     - `required_section`: Stores document schemas/structure requirements

2. **MongoDB Schema**
   ```python
   document_qas: {
       department: { code, name, slug },
       document_type: str,
       document_name: str,
       question: str,
       answer: str,
       category: str,
       category_order: int,
       question_order: int,
       answer_type: str,           # "text" | "select" | "multi_select" | "structured_list"
       options: list,
       is_gap_question: bool,      # ★ NEW — True for AI-generated gap questions
       section_covered: str,       # ★ NEW — which schema section this covers
       answered_at: datetime       # ★ NEW — timestamp when gap Q was answered & saved
   }

   required_section: {
       department: str,
       document_name: str,
       sections: [{ title, type, subsections/columns, ... }]
   }
   ```

3. **Async Database Connection** (`api/db.py`)
   - Singleton motor AsyncIOMotorClient
   - Lazy initialization on first access
   - Proper lifecycle management with FastAPI lifespan

---

### Layer 4: FastAPI Backend Orchestration
**File**: `api/main.py`

**Key Endpoints**:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/departments` | GET | Returns sorted list of departments from MongoDB |
| `/document-types` | GET | Returns document types for a given department |
| `/questions` | GET | Returns Q&A pairs (core + saved gap questions) for a document type |
| `/required-section` | GET | Fetches document schema/structure template |
| `/gap-questions` | POST | ★ NEW: Analyse schema coverage gaps, return targeted questions |
| `/save-questions` | POST | ★ NEW: Persist answered gap questions to MongoDB |
| `/generate` | POST | Triggers LangGraph agent for document generation |
| `/get_all_urls` | GET | Retrieves all Notion page URLs (for history) |

**`POST /gap-questions` — Two-stage logic**:
```
1. Check MongoDB: are gap questions already saved for this document_type?
      YES → return them immediately (source: "cache", no LLM call)
      NO  → run lightweight LLM gap analysis (source: "generated")
```
This caching layer is the primary mechanism that prevents repeated LLM
calls for the same document type across sessions and users.

**`POST /save-questions` — Upsert with deduplication**:
- Upserts on `(document_type, question, is_gap_question=True)`
- Sets `question_order` to 1000+ so gap questions sort after core ones
- Sets `category_order: 999` → always rendered last in the UI

**CORS Configuration**:
- Allows requests from Streamlit on `localhost:8501` and `127.0.0.1:8501`

---

### Layer 5: LangGraph Agent for Document Generation
**File**: `agent/agent_graph.py`

**Purpose**: Transforms user answers into professional, schema-compliant documents

**Two LLMs are now used**:
| Model | Role | Why |
|-------|------|-----|
| `moonshotai/kimi-k2-instruct-0905` | Primary — document generation, quality review, fixes | Best output quality for long-form prose |
| `llama-3.3-70b-versatile` | Secondary — schema gap analysis only | Faster, cheaper, sufficient for structured JSON output |

**Agent State** (`AgentState`):
```python
# Inputs
department: str
document_type: str
questions_and_answers: list[dict]
required_section: dict

# Intermediates/Outputs
gap_questions: list[dict]       # ★ NEW: AI-generated questions for uncovered sections
supplementary_content: str      # Context notes for the document LLM about gaps
system_prompt: str              # Full LLM prompt
generated_document: str         # Final output
quality_scores: dict            # LLM quality metrics
quality_issues: list[str]       # Validation failures
quality_suggestions: list[str]  # Improvement suggestions
retry_count: int                # Retry attempts
status: str                     # "generating" | "passed" | "failed"
```

**5-Node Workflow**:

```
START
  │
  ▼
┌──────────────────────────────────────────────┐
│ 1. analyze_schema_gaps            ★ NEW NODE  │
│  • Compares schema sections vs Q&A answers   │
│  • Uses lightweight Llama-3.3-70b LLM        │
│  • Outputs: gap_questions (JSON array)        │
│  • Also writes supplementary_content notes   │
│    so doc LLM knows gaps exist               │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────┐
│ 2. build_prompt              │
│  (Format Q&As + schema)      │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ 3. generate_document         │
│  (Primary LLM — Kimi-k2)     │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ 4. quality_gate              │
│  (Validate structure)        │
│  - Table-only: Deterministic │
│  - Mixed: LLM review         │
└──────────────┬───────────────┘
               │
          ┌────┴────┐
          │          │
       PASS       FAIL
          │          │
        END    ┌─────────────────┐
               │ 5. fix_document │
               │  (Retry & fix)  │
               └────┬────────────┘
                    │
                (loop back to
                 quality_gate)
```

**Node Descriptions**:

1. **`analyze_schema_gaps`** ★ NEW (replaces `fill_schema_gaps`)
   - Sends schema + existing Q&As to the lightweight `question_gen_llm`
   - LLM returns a JSON array: `[{question, category, answer_type, section_covered}]`
   - Uncovered sections get a placeholder note in `supplementary_content` so
     the document LLM knows to flag them rather than silently skip
   - Gap questions are returned in `state["gap_questions"]` — surfaced in UI and
     optionally saved to MongoDB
   - **Key difference from the old `fill_schema_gaps`**: instead of hallucinating
     content to fill gaps itself, it asks the *user* for missing information

   Old approach:
   ```
   fill_schema_gaps → primary LLM synthesizes filler content (bypasses user)
   ```
   New approach:
   ```
   analyze_schema_gaps → lightweight LLM asks the right questions
                       → user provides real answers
                       → better document quality, no hallucination
   ```

2. **`build_prompt`** — unchanged
   - Formats Q&As into readable text blocks (organized by category)
   - Formats schema into markdown structure guide
   - Constructs system prompt with LLM instructions
   - Detects table-only vs mixed schemas and selects appropriate prompt template

3. **`generate_document`** — unchanged
   - Calls primary Groq LLM with formatted prompt + system instructions
   - Returns raw markdown document

4. **`quality_gate`** — unchanged
   - **Table-only schemas**: Deterministic validation
     - Extracts markdown table from output
     - Verifies column headers match schema exactly
     - Auto-fixes by extracting only table + heading
   - **Mixed schemas**: LLM-based review with rule-based fallback
     - Validates structure, completeness, professionalism
     - Returns quality scores & issues

5. **`fix_document`** — unchanged
   - If quality gate fails, re-prompts primary LLM with corrections
   - Increments retry counter
   - Loops back to `quality_gate` for re-validation

**Standalone utility** — `analyze_gaps_only()`:
- Runs only node 1 without the full document generation pipeline
- Used by `POST /gap-questions` for on-demand pre-generation gap analysis

---

### Layer 6: Streamlit Frontend UI
**File**: `ui/streamlit_uidemo.py`

**UI Flow**:
1. **Left Sidebar**
   - Department selector (dropdown)
   - Document selector (dropdown)
   - Generation history (clickable links to Notion pages)
   - Auto-clears gap questions when document selection changes

2. **Main Area** (Two-column layout)
   - **Left Column**: Q&A Panel
     - Core questions (from MongoDB, rendered by category)
     - **Gap Questions section** ★ NEW (two sources, visually unified):
       - *MongoDB-persisted gap questions* (`is_gap_question: True`) — loaded
         automatically with `/questions`, rendered identically to core questions
         but with an `AI` badge
       - *Session gap questions* — freshly generated via `POST /gap-questions`,
         shown with a "💾 Save gap questions" button
     - "🔍 Analyse schema gaps" button — triggers on-demand gap analysis
     - "⚡ Generate Document" button — sends all answers (core + gap) to `/generate`
   - **Right Column**: Document Editor
     - Displays generated markdown
     - Editable textarea for refinements
     - Rendered preview (collapsible)
     - Publish to Notion button

3. **Gap Question Lifecycle in the UI**:
```
User clicks "🔍 Analyse schema gaps"
  │
  ▼
POST /gap-questions
  ├── source: "cache" → display immediately, no spinner delay
  └── source: "generated" → ~10s spinner, then display

User fills in gap answers
  │
  ▼
User clicks "💾 Save gap questions"
  │
  ▼
POST /save-questions → upsert to MongoDB
  │
  ▼
questions cache cleared → next load includes gap Qs automatically
```

4. **Answer payload sent to `/generate`**:
   - Core Q&A answers (`st.session_state.answers`)
   - MongoDB-persisted gap Q answers (already in core answers dict)
   - Session gap Q answers (`st.session_state.gap_answers`)
   - All merged into a single `questions_and_answers` list

5. **Session State Management**:
   ```python
   history        # Notion page URLs
   answers        # Core + MongoDB-gap question answers {key: value}
   gap_answers    # Session gap question answers {key: value}
   gap_questions  # Current session gap questions list
   gap_source     # "cache" | "generated"
   gap_doc_type   # document_type the gap questions belong to (for auto-clear)
   markdown_doc   # Generated document text
   is_generating  # Button lock flag
   is_analyzing   # Button lock flag
   is_saving      # Button lock flag
   ```

**API Helpers** (with caching):
- `get_departments_from_fastapi()` — TTL 300s
- `get_document_types_from_fastapi(department)` — TTL 300s
- `get_questions_from_fastapi(document_type)` — TTL 300s (cleared after save)
- `get_notionpage_urls_from_fastapi()` — TTL 600s
- `call_gap_questions_endpoint()` — POST, no cache (always fresh)
- `call_save_questions_endpoint()` — POST, no cache
- `call_generate_endpoint()` — POST, no cache

**Shared widget renderer** — `render_question_widget()`:
- Single function handles all `answer_type` variants: `text`, `structured_list`,
  `select`, `multi_select`
- Accepts `is_gap=True` to inject an `AI` badge without changing widget behaviour
- Eliminates duplicated widget logic between core and gap question rendering

---

## 📁 File Structure

```
DocForgeHub/
├── agent/
│   ├── agent_graph.py           # 5-node LangGraph agent + orchestration
│   │                              ★ analyze_schema_gaps replaces fill_schema_gaps
│   │                              ★ analyze_gaps_only() utility added
│   ├── prompts.py               # System prompt templates & formatting
│   └── __init__.py
│
├── api/
│   ├── main.py                  # FastAPI endpoints
│   │                              ★ POST /gap-questions added
│   │                              ★ POST /save-questions added
│   ├── db.py                    # MongoDB connection (async motor)
│   └── __init__.py
│
├── automations/
│   ├── ques_automation.py       # Question generation with LangGraph
│   ├── automation.py            # Notion content extraction
│   ├── add_answer_field.py      # Add answer fields & organize
│   ├── mongo_auto.py            # MongoDB batch upload
│   ├── required_sections_automation.py  # Schema upload to MongoDB
│   ├── clean_reorder.py         # Data cleanup utilities
│   └── ...
│
├── ui/
│   └── streamlit_uidemo.py     # Streamlit frontend
│                                  ★ Gap questions panel added
│                                  ★ render_question_widget() helper added
│                                  ★ gap_answers session state added
│
├── document_and_questions/
│   ├── final_filtered_QAs/      # Final Q&As by department
│   │   ├── 1._Product_Management/
│   │   ├── 2._Engineering__Software_Development/
│   │   ├── ... (10 departments)
│   │   └── 10._Finance/
│   │
│   └── notion_documents/        # Extracted Notion docs
│       └── ... (same structure)
│
├── progress.md                  # Development log
└── .env                         # Credentials (MongoDB, Groq, Notion)
```

---

## 🔑 Design Patterns & Approaches

### 1. **State Machine via LangGraph**
- Multi-step workflow modeled as directed acyclic graph (DAG)
- Each node is a pure function: `State → dict`
- Conditional routing based on quality gate results
- Built-in retry loop (`fix_document` → `quality_gate`)

### 2. **Async-First Architecture**
- FastAPI with async/await throughout
- Motor (async MongoDB driver)
- Graceful lifespan management (app startup/shutdown)

### 3. **Schema-Driven Generation**
- Documents strictly follow MongoDB `required_section` schema
- Two validation modes:
  - **Deterministic** (table-only): Regex + structural validation
  - **LLM-based** (mixed): Semantic validation with quality scoring

### 4. **User-In-The-Loop Gap Filling** ★ NEW
- Old approach: LLM synthesised supplementary content autonomously (hallucination risk)
- New approach: Lightweight LLM identifies gaps → generates targeted questions → user provides real answers
- Gap questions are persisted to MongoDB so future users benefit immediately
- Cache-first design: gap questions are generated at most once per document type

### 5. **Two-LLM Architecture** ★ NEW
- **Primary LLM** (Kimi-k2): Long-form document generation, quality review, fixes
- **Secondary LLM** (Llama-3.3-70b): Schema gap analysis, structured JSON output
- Separation keeps the heavy model focused on prose quality and the light model on analysis

### 6. **Gap Question Caching via MongoDB** ★ NEW
- `POST /gap-questions` checks MongoDB before calling any LLM
- Once a document type's gaps are analysed and saved, subsequent requests
  return cached results instantly — no further LLM calls needed
- Scales gracefully: the question-generation load is O(1) per document type, not O(users)

### 7. **Resilient API Calls**
- Multiple Groq API keys for fallback
- Retry logic with exponential backoff
- Clear error messages and logging

### 8. **Batch Processing Automation**
- Command-line tools for bulk operations:
  - Extract questions from Notion
  - Add answer fields
  - Upload to MongoDB
  - Manage schemas
- Interactive confirmation prompts
- Progress tracking and summaries

### 9. **Data Organization by Taxonomy**
- Hierarchical: Department → Document Type → Q&A
- MongoDB indexing for fast queries
- Streamlit caching for performance

---

## 🚀 Execution Flow (Complete User Journey)

### Setup Phase (One-time)
1. Extract documents from Notion → `notion_documents/`
2. Generate questions via LangGraph → `generated_questions/`
3. Add answer fields → `final_filtered_QAs/`
4. Upload to MongoDB (documents + schemas)

### Runtime Phase (Per Document Generation)
1. **Streamlit UI**: User selects department + document
2. **API**: Fetch Q&As + schema from MongoDB (includes any saved gap questions)
3. **Streamlit**: User fills in core answers
4. *(Optional)* **User clicks "🔍 Analyse schema gaps"**:
   - `POST /gap-questions` → checks MongoDB cache first
   - Returns gap questions; user fills in answers
   - User clicks "💾 Save gap questions" → `POST /save-questions` → persisted for future users
5. **User clicks "⚡ Generate Document"** → `POST /generate`:
   - FastAPI receives all answers (core + gap)
   - Calls `run_agent()` with Q&As + schema
6. **Agent Graph**:
   - `analyze_schema_gaps` → `build_prompt` → `generate_document`
   - `quality_gate` (validate) → `fix_document` (if needed)
   - Gap questions from the agent also surfaced in UI if not already loaded
7. **Response**: Return generated markdown + quality metrics + any new gap questions
8. **Streamlit**: Display markdown + allow edits + show quality scores
9. **Publish**: Optional upload to Notion

---

## 🛠️ Key Technologies & Why

| Component | Technology | Why |
|-----------|-----------|-----|
| Primary LLM | Groq + Kimi-k2 | Best prose quality, supports long-form generation |
| Gap Analysis LLM | Groq + Llama-3.3-70b | Fast, cheap, excellent at structured JSON output |
| Agent Orchestration | LangGraph | Deterministic multi-step workflows, built-in state management |
| API | FastAPI | Async support, auto docs, CORS middleware |
| Database | MongoDB | Flexible schema, fast aggregation, upsert support |
| Frontend | Streamlit | Rapid prototyping, caching built-in, minimal code |
| Async Driver | Motor | Non-blocking DB operations, FastAPI integration |

---

## 📈 Quality Assurance

### Quality Gate Validations
1. **Structural**: Document follows schema sections exactly
2. **Table Validation**: Markdown tables have correct columns
3. **Completeness**: All required sections present
4. **Professionalism**: LLM scores readability, clarity, tone
5. **Suggestions**: Auto-generated improvement tips

### Retry Mechanism
- Up to 2 retries to fix document (3 total attempts)
- Each retry receives specific failure feedback
- If all retries fail, returns partial document with `status: "failed"`

### Gap Coverage
- Schema sections not addressed by core questions are flagged
- Users are prompted with targeted gap questions before generation
- Gap answers are included in the full Q&A payload — resulting in higher
  completeness scores from the quality gate

---

## 🔐 Security & Configuration

### Environment Variables (.env)
```
GROQ_API_KEY (and _2 through _7)
MONGODB_CONNECTION_STRING
MONGODB_DATABASE
NOTION_API_KEY
```

### CORS Policy
- Restricts API to Streamlit frontend (localhost:8501)
- Prevents unauthorized cross-origin requests

---

## 📊 Summary Table

| Layer | Purpose | Key Files | Tech |
|-------|---------|-----------|------|
| **1. Extraction** | Extract from Notion, generate Q&As | `ques_automation.py` | LangGraph, Notion API |
| **2. Enrichment** | Add answer fields, organize | `add_answer_field.py` | Python utilities |
| **3. Storage** | Persist to MongoDB | `mongo_auto.py` | MongoDB, Motor |
| **4. API** | Serve data & trigger generation | `main.py` | FastAPI, Motor |
| **5. Agent** | Analyse gaps + generate documents | `agent_graph.py` | LangGraph, Groq (×2) |
| **6. Frontend** | User interface | `streamlit_uidemo.py` | Streamlit |

---

## 🎓 Architectural Highlights

✅ **Modular Design**: Each layer independently testable  
✅ **State-Driven**: LangGraph ensures deterministic workflows  
✅ **Scalable**: Async operations, MongoDB indexing  
✅ **Resilient**: Multi-retry loops, API key fallbacks  
✅ **Professional Output**: Content elevation + quality gates  
✅ **User-in-the-Loop**: Gap questions ask users instead of hallucinating  
✅ **Cache-First Gap Analysis**: O(1) LLM calls per document type, not per user  
✅ **User-Friendly**: Streamlit UI with caching & real-time feedback  

---

**Last Updated**: February 19, 2026  
**Architecture Version**: 1.2