# DocForgeHub — Codebase Architecture & Approach

## 🎯 Project Overview

**DocForgeHub** is an intelligent document generation and management system that:
- Extracts business documents from Notion
- Generates comprehensive Q&A from documents using LLMs
- Stores Q&As in MongoDB with department/category organization
- Generates polished, professional business documents from user answers using an agentic workflow
- Provides a Streamlit UI for document generation and management

---

## 🏗️ Architecture Stack

### Technology Stack
- **LLM Provider**: Groq (with Kimi-k2 instruct model)
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
│  (Department / Document Selection / Q&A Input / Generated View)  │
└─────────────────────┬───────────────────────────────────────────┘
                      │ HTTP REST
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     FASTAPI BACKEND (Port 8000)                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ GET /departments         → List all departments            │ │
│  │ GET /document-types      → List docs for department        │ │
│  │ GET /questions           → List Q&As for document          │ │
│  │ GET /required-section    → Fetch schema from MongoDB       │ │
│  │ POST /generate           → Trigger agentic document gen    │ │
│  │ GET /get_all_urls        → Retrieve Notion page URLs       │ │
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
       questions_and_answers: [...],
       timestamp: datetime
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
| `/questions` | GET | Returns Q&A pairs for a document type |
| `/required-section` | GET | Fetches document schema/structure template |
| `/generate` | POST | Triggers LangGraph agent for document generation |
| `/get_all_urls` | GET | Retrieves all Notion page URLs (for history) |

**CORS Configuration**:
- Allows requests from Streamlit on `localhost:8501` and `127.0.0.1:8501`

---

### Layer 5: LangGraph Agent for Document Generation
**File**: `agent/agent_graph.py`

**Purpose**: Transforms user answers into professional, schema-compliant documents

**Agent State** (`AgentState`):
```python
# Inputs
department: str
document_type: str
questions_and_answers: list[dict]
required_section: dict

# Intermediates/Outputs
supplementary_content: str      # Gap-filled content
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
┌──────────────────────────────┐
│ 1. fill_schema_gaps          │
│  (Detect low-answer areas)   │
└──────────────┬───────────────┘
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
│  (LLM calls Groq)            │
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
        END    ▼─────────────────┐
              │ 5. fix_document  │
              │  (Retry & fix)   │
              └────┬─────────────┘
                   │
               (loop back to
                quality_gate)
```

**Node Descriptions**:

1. **fill_schema_gaps**
   - Analyzes Q&A coverage vs. required sections
   - Identifies gaps (sections with no/low answers)
   - Uses LLM to generate supplementary content for missing areas

2. **build_prompt**
   - Formats Q&As into readable text blocks (organized by category)
   - Formats schema into markdown structure guide
   - Constructs system prompt with LLM instructions

3. **generate_document**
   - Calls Groq LLM with formatted prompt + system instructions
   - Returns raw markdown document

4. **quality_gate**
   - **Table-only schemas**: Deterministic validation
     - Extracts markdown table from output
     - Verifies column headers match schema exactly
     - Auto-fixes by extracting only table + heading
   - **Mixed schemas**: LLM-based review with fallback
     - Validates structure, completeness, professionalism
     - Returns quality scores & issues

5. **fix_document**
   - If quality gate fails, re-prompts LLM with corrections
   - Increments retry counter
   - Loops back to quality_gate for re-validation

**Routing Logic**:
- `decide_after_quality_gate()`: Returns `"fix_document"` or `"end"`
- Configurable max retries (typically 2-3 attempts)

**LLM Instructions** (`agent/prompts.py`):
- Transform raw answers into professional prose (no copy-paste)
- Expand vague answers with industry context
- Follow schema structure exactly
- Output markdown tables for table-only sections
- Add metrics/KPIs where appropriate
- Infer content for missing areas with disclaimer

---

### Layer 6: Streamlit Frontend UI
**File**: `ui/streamlit_uidemo.py`

**UI Flow**:
1. **Left Sidebar**
   - Department selector (dropdown)
   - Document selector (dropdown)
   - Generation history (clickable links to Notion pages)

2. **Main Area** (Two-column layout)
   - **Left Column**: Q&A Panel
     - Displays questions for selected document
     - Text inputs for answers
   - **Right Column**: Document Editor
     - Displays generated markdown
     - Editable textarea for refinements
     - Publish to Notion button

3. **Session State Management**
   - Caches: `history`, `answers`, `markdown_doc`, `is_generating`
   - TTL on API calls: 300s for departments/documents, 600s for URLs

**API Helpers** (with caching):
- `get_departments_from_fastapi()`
- `get_document_types_from_fastapi(department)`
- `get_questions_from_fastapi(document_type)`
- `get_notionpage_urls_from_fastapi()`
- `call_generate_endpoint()` → POST to `/generate`

---

## 📁 File Structure

```
DocForgeHub/
├── agent/
│   ├── agent_graph.py           # 5-node LangGraph agent + orchestration
│   ├── prompts.py               # System prompt templates & formatting
│   └── __init__.py
│
├── api/
│   ├── main.py                  # FastAPI endpoints
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
- Built-in retry loop (fix_document → quality_gate)

### 2. **Async-First Architecture**
- FastAPI with async/await throughout
- Motor (async MongoDB driver)
- Graceful lifespan management (app startup/shutdown)

### 3. **Schema-Driven Generation**
- Documents strictly follow MongoDB `required_section` schema
- Two validation modes:
  - **Deterministic** (table-only): Regex + structural validation
  - **LLM-based** (mixed): Semantic validation with quality scoring

### 4. **Content Gap Filling**
- Identifies sections with no/few answers
- LLM generates supplementary content with disclaimer
- Ensures comprehensive document output

### 5. **Resilient API Calls**
- Multiple Groq API keys for fallback
- Retry logic with exponential backoff
- Clear error messages and logging

### 6. **Batch Processing Automation**
- Command-line tools for bulk operations:
  - Extract questions from Notion
  - Add answer fields
  - Upload to MongoDB
  - Manage schemas
- Interactive confirmation prompts
- Progress tracking and summaries

### 7. **Data Organization by Taxonomy**
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
2. **API**: Fetch Q&As + schema from MongoDB
3. **Streamlit**: User fills in answers
4. **POST /generate**:
   - FastAPI receives request
   - Calls `run_agent()` with Q&As + schema
5. **Agent Graph**:
   - fill_schema_gaps → build_prompt → generate_document
   - quality_gate (validate) → fix_document (if needed)
6. **Response**: Return generated markdown + quality metrics
7. **Streamlit**: Display markdown + allow edits
8. **Publish**: Optional upload to Notion

---

## 🛠️ Key Technologies & Why

| Component | Technology | Why |
|-----------|-----------|-----|
| LLM | Groq + Kimi-k2 | Fast inference, cost-effective, supports multiple API keys |
| Agent Orchestration | LangGraph | Deterministic multi-step workflows, built-in state management |
| API | FastAPI | Async support, auto docs, CORS middleware |
| Database | MongoDB | Flexible schema for Q&As, fast aggregation queries |
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
- Up to 3 attempts to fix document
- Each retry receives specific failure feedback
- If all retries fail, returns partial document with `status: "failed"`

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
| **5. Agent** | Generate documents from Q&As | `agent_graph.py` | LangGraph, Groq |
| **6. Frontend** | User interface | `streamlit_uidemo.py` | Streamlit |

---

## 🎓 Architectural Highlights

✅ **Modular Design**: Each layer independently testable  
✅ **State-Driven**: LangGraph ensures deterministic workflows  
✅ **Scalable**: Async operations, MongoDB indexing  
✅ **Resilient**: Multi-retry loops, API key fallbacks  
✅ **Professional Output**: Content elevation + quality gates  
✅ **User-Friendly**: Streamlit UI with caching & real-time feedback  

---

**Last Updated**: February 19, 2026  
**Architecture Version**: 1.0
