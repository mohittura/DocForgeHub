# DocForgeHub — Comprehensive Codebase Architecture

## 🎯 Executive Summary

**DocForgeHub** is an enterprise-grade, intelligent document generation and management platform designed to automate the creation of professional business documents for SaaS organizations. The system leverages AI-powered schema analysis, multi-stage LLM workflows, and user-in-the-loop gap filling to ensure documents are both complete and professionally written.

### Core Capabilities
- **Notion Integration**: Extracts business documents from Notion and processes them recursively
- **Intelligent Q&A Generation**: Uses LangGraph-based workflows to generate comprehensive question sets from document content
- **Database Organization**: Stores all Q&As and document schemas in MongoDB, organized hierarchically by department and document type
- **Schema Gap Analysis** ★ NEW: Intelligently identifies which sections of required document schemas are not covered by existing Q&As, generates targeted questions to fill gaps, and persists them for reuse
- **AI-Powered Document Generation**: Multi-node LangGraph agent transforms user answers into professionally-written, schema-compliant business documents
- **Quality Assurance**: Implements deterministic and LLM-based validation, automatic retry/fix mechanisms, and quality scoring
- **Interactive UI**: Streamlit-based frontend enabling users to select documents, fill Q&As, analyse gaps, and generate/review documents
- **Scalable Architecture**: Async-first design with caching layers to minimize LLM calls and database operations

### Key Innovation: User-in-the-Loop Gap Filling
Instead of hallucinating content to fill document schema gaps (which risks inaccuracy), DocForgeHub now:
1. Uses a lightweight LLM to identify which schema sections lack coverage
2. Generates targeted questions asking the user for missing information
3. Persists answered gap questions to MongoDB
4. Automatically includes gap answers in future document generations for the same document type
5. Results in higher-quality, more accurate documents with zero hallucination

---

## 🏗️ Technology Stack & Architectural Decisions

### Core Infrastructure
| Layer | Technology | Purpose | Rationale |
|-------|-----------|---------|-----------|
| **Primary LLM** | Groq + Kimi-k2-instruct-0905 | Document generation, quality review, fix attempts | Best-in-class prose quality, 200K context window, optimized for long-form content |
| **Analysis LLM** | Groq + Llama-3.3-70b-versatile | Schema gap analysis, structured JSON output | Excellent at structured reasoning, 5x cheaper than primary, sufficient for non-prose tasks |
| **Orchestration** | LangGraph (LangChain) | Multi-step agentic workflows | Deterministic state machines, built-in error handling, graph visualization |
| **REST API** | FastAPI | Backend service orchestration | Async/await throughout, auto-generated docs, built-in CORS, Pydantic validation |
| **Database** | MongoDB (Atlas) | Document schemas, Q&As, gap questions, caching | Flexible JSON schema, fast aggregation, upsert support, full-text search capable |
| **Async Driver** | Motor | Non-blocking MongoDB operations | Integrates seamlessly with FastAPI, prevents blocking on I/O |
| **Frontend** | Streamlit | User interface & document editor | Rapid iteration, built-in caching (st.cache_data), minimal boilerplate, responsive widgets |
| **Content Source** | Notion API | External document repository | Read-only extraction, recursive page traversal, markdown export |
| **Language** | Python 3.12+ | All code | Type hints, async/await support, rich ecosystem of ML libraries |

### Key Architecture Decisions Explained

**1. Two-LLM Strategy**
- Separates concerns: heavyweight primary model focuses on prose quality, lightweight secondary model on analysis
- Cost optimization: Gap analysis (cheaper model) happens frequently; document generation (expensive model) happens on-demand
- Fault tolerance: If analysis fails, users can still proceed with manual gap answers

**2. Async-First Design**
- FastAPI handles concurrent requests from multiple Streamlit sessions
- Motor prevents database connection pooling issues
- Non-blocking I/O enables horizontal scaling without thread explosion

**3. LangGraph State Machines**
- Replaces ad-hoc orchestration with declarative workflows
- Built-in state persistence and logging
- Deterministic retry loops guarantee consistent behavior
- Graph structure makes workflows auditable and testable

**4. MongoDB as Single Source of Truth**
- Unified storage for schemas, questions, answers, and gap metadata
- Atomic upsert operations prevent duplicate gap questions
- Aggregation pipeline enables complex filtering (departments, document types, categories)
- TTL indexes can auto-expire old session data if needed in the future

**5. Schema-Driven Document Generation**
- Every document is validated against its MongoDB schema before returning to user
- Two validation modes accommodate different document types:
  - **Table-only schemas** (e.g., Change Request Log): Deterministic validation of column headers and row count
  - **Mixed schemas** (e.g., Feature Prioritization Framework): LLM-based review for structural completeness and professional tone
- This guarantees output consistency regardless of LLM variation

### Core Dependencies
```
langchain-groq ≥ 0.0.1       # Groq LLM integration
langgraph ≥ 0.1.0            # State graph orchestration
fastapi ≥ 0.104.1            # Async REST framework
motor ≥ 3.3.0                # Async MongoDB driver
streamlit ≥ 1.32.0           # Frontend framework
notion-client ≥ 2.1.0        # Notion API client
pymongo ≥ 4.6.0              # Sync MongoDB (batch operations)
pydantic ≥ 2.0.0             # Data validation
python-dotenv ≥ 1.0.0        # Environment configuration
```

---

## 📊 High-Level System Architecture

### System Overview Diagram
```
┌──────────────────────────────────────────────────────────────────────────┐
│                     STREAMLIT UI FRONTEND (Port 8501)                    │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ • Department & Document Selection (Sidebar)                       │  │
│  │ • Core Q&A Panel (fetched from MongoDB with caching)              │  │
│  │ • Gap Questions Panel (MongoDB-persisted + session-fresh) ★ NEW   │  │
│  │ • Generated Document Editor (with Notion publish)                 │  │
│  │ • Generation History & Quality Metrics                            │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                              │                                           │
│                         HTTPS/REST                                       │
│                              │                                           │
│                              ▼                                           │
├──────────────────────────────────────────────────────────────────────────┤
│               FASTAPI BACKEND (Port 8000) - Async Layer                  │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │ GET  /departments              → Aggregate unique depts            │  │
│  │ GET  /document-types           → Docs for dept (with names)        │  │
│  │ GET  /questions                → Core + MongoDB gap Qs, sorted     │  │
│  │ GET  /required-section         → Schema lookup by dept+name        │  │
│  │ POST /gap-questions      ★ NEW → Cache-first gap analysis         │  │
│  │ POST /save-questions     ★ NEW → Upsert gap Qs to MongoDB          │  │
│  │ POST /generate                 → Trigger agent for full doc        │  │
│  │ POST /generate-section   ★ NEW → Progressive single section gen    │  │
│  │ GET  /get_all_urls             → Notion page URLs for history      │  │
│  └────────────────────────────────────────────────────────────────────┘  │
│                              │                                           │
│          ┌───────────────────┼───────────────────┐                      │
│          │                   │                   │                      │
│          ▼                   ▼                   ▼                      │
│  ┌──────────────┐    ┌──────────────┐   ┌──────────────┐               │
│  │  LANGGRAPH   │    │   MONGODB    │   │  NOTION API  │               │
│  │  AGENT       │    │   ATLAS      │   │  (read-only) │               │
│  │              │    │              │   │              │               │
│  │ 5-Node       │    │ Collections: │   │ • Page       │               │
│  │ Workflow     │    │ • doc_qas    │   │   metadata   │               │
│  │ • analyze    │    │ • req_sec    │   │ • hierarchy  │               │
│  │ • build      │    │              │   │   structure  │               │
│  │ • generate   │    │ Indexes:     │   │              │               │
│  │ • quality    │    │ • dept       │   │              │               │
│  │ • fix        │    │ • doc_type   │   │              │               │
│  │              │    │ • gap_q      │   │              │               │
│  └──────────────┘    └──────────────┘   └──────────────┘               │
└──────────────────────────────────────────────────────────────────────────┘
         │
         │ (Groq API calls - with fallback keys)
         │
         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│            EXTERNAL LLM SERVICES (Groq Cloud)                            │
│  ┌──────────────────────────┐    ┌──────────────────────────┐           │
│  │ Primary Model            │    │ Secondary Model          │           │
│  │ kimi-k2-instruct-0905    │    │ llama-3.3-70b-versatile  │           │
│  │                          │    │                          │           │
│  │ • Document generation    │    │ • Gap analysis           │           │
│  │ • Quality review         │    │ • JSON structuring       │           │
│  │ • Fix suggestions        │    │ • Schema coverage check  │           │
│  │                          │    │                          │           │
│  │ Max Tokens: 200K         │    │ Max Tokens: 32K          │           │
│  │ Cost: $$$$               │    │ Cost: $                  │           │
│  └──────────────────────────┘    └──────────────────────────┘           │
└──────────────────────────────────────────────────────────────────────────┘
```

### Data Flow at a Glance
```
SETUP PHASE (One-time)
  Notion → Extract Q&As → MongoDB ← Upload Schemas
            ↓
  generated_questions/ → add_answer_field/ → final_filtered_QAs/
                                                       ↓
                                        mongo_auto.py → MongoDB

RUNTIME PHASE (Per User Session)
  Streamlit UI
    ├─ Select Dept + Document
    ├─ GET /questions (includes cached gap Qs) ← MongoDB
    ├─ User fills Core Q&As
    │
    ├─ Optional: Click "🔍 Analyse Gaps"
    │   POST /gap-questions
    │   ├─ Check: Gap Qs already in MongoDB? YES → return cached
    │   └─ NO → lightweight LLM analysis → return fresh + ask to save
    │
    ├─ User fills Gap Answers (optional)
    │   POST /save-questions → Upsert to MongoDB (for next user)
    │
    └─ Click "⚡ Generate"
       POST /generate → Agent Graph (5 nodes)
       ├─ All Q&A answers (core + gap)
       ├─ Required section schema
       ├─ Primary LLM: generate document
       ├─ Quality validation + LLM review
       ├─ Fix if needed (retry loop)
       └─ Return markdown + scores
```

---

## 🔄 Detailed Data Flow & Processing Layers

### Layer 1: Notion Content Extraction & Question Generation
**Files**: `automations/ques_automation.py`, `automations/automation.py`

**Purpose**: Transform raw Notion documents into structured Q&A pairs

#### 1.1 Notion Content Extractor (`NotionContentExtractor`)
```python
class NotionContentExtractor:
    """Recursively fetch pages from Notion and convert to markdown"""
    
    def __init__(self, api_key: str, root_page_id: str):
        """Initialize with Notion client and root page ID"""
    
    def fetch_recursive(self, page_id: str) -> Dict:
        """
        Recursively retrieve:
        - Page metadata (title, last_edited_time)
        - Page content (markdown blocks)
        - Child pages (hierarchical structure)
        
        Returns nested structure: {title, content, children: [...]}
        """
```

**Workflow**:
1. Initialize client with Notion API key
2. Start from root page ID (typically a department workspace)
3. Recursively traverse child pages
4. Extract markdown content from each page
5. Return hierarchical structure preserving page organization
6. Output: Raw markdown files organized by heading structure

#### 1.2 LangGraph-Based Question Generator (`GroqLangGraphQuestionGenerator`)
```python
class GroqLangGraphQuestionGenerator:
    """Multi-node LangGraph workflow for intelligent question generation"""
    
    def build_graph(self) -> StateGraph:
        """
        Construct a DAG with these nodes:
        
        ┌─────────────────────────────────────┐
        │ 1. _analyze_and_detect              │
        │    ├─ Parse document structure      │
        │    ├─ Identify section patterns     │
        │    ├─ Detect tables/lists/prose     │
        │    └─ Extract metadata              │
        └─────────────┬───────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────────────┐
        │ 2. _generate_questions              │
        │    ├─ Call Groq LLM (primary)       │
        │    ├─ Parse LLM JSON response       │
        │    ├─ Validate structure            │
        │    └─ Format to Q&A schema          │
        └─────────────┬───────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────────────┐
        │ 3. _simple_validate                 │
        │    ├─ Check required fields         │
        │    ├─ Ensure unique questions       │
        │    ├─ Validate answer_type values   │
        │    └─ Detect incomplete answers     │
        └─────────────┬───────────────────────┘
                      │
                      ▼
        ┌─────────────────────────────────────┐
        │ 4. Return validated Q&A list        │
        │    (ready for answer field addition)│
        └─────────────────────────────────────┘
        """
```

**Key Features**:
- **Resilient API calling**: Configured with 7 fallback Groq API keys
- **Structured output validation**: JSON parsing with error recovery
- **Deterministic retry logic**: Re-attempts failed nodes with exponential backoff
- **Comprehensive logging**: Every step traced for debugging

**Output**:
```json
{
  "generated_questions": [
    {
      "question": "What is the primary objective?",
      "category": "Overview",
      "answer_type": "text",
      "description": "Brief description of what this question captures"
    },
    ...
  ],
  "metadata": {
    "document_name": "Feature Prioritization Framework",
    "generation_time": "2024-02-20T10:30:00Z",
    "success": true
  }
}
```

---

### Layer 2: Answer Field Addition & Data Organization
**File**: `automations/add_answer_field.py`

**Purpose**: Prepare Q&As for MongoDB storage by adding empty answer fields and organizing metadata

**QuestionAnswerProcessor Workflow**:
1. Read generated question files from `generated_questions/`
2. For each question, add:
   - Empty `answer: ""` field
   - `category_order`: Numeric index for sorting
   - `question_order`: Position within category
   - `is_gap_question: false` (marks as core Q)
3. Organize by topics/categories
4. Validate schema compliance
5. Output to `final_filtered_QAs/{department}/`

**Output Structure**:
```json
{
  "document_name": "Feature Prioritization Framework",
  "document_type": "Feature Prioritization Framework",
  "questions_by_category": [
    {
      "category": "Overview",
      "category_order": 1,
      "questions": [
        {
          "question_id": "overview_objective",
          "question": "What is the primary objective?",
          "answer": "",
          "answer_type": "text",
          "question_order": 1,
          "is_gap_question": false
        },
        ...
      ]
    },
    ...
  ]
}
```

---

### Layer 3: MongoDB Integration & Schema Storage
**Files**: `automations/mongo_auto.py`, `api/db.py`, `automations/required_sections_automation.py`

**Purpose**: Persist all Q&As and document schemas to MongoDB for fast retrieval and gap analysis

#### 3.1 Database Design

**Collection: `document_qas`** (Core & Gap Questions)
```
{
  "_id": ObjectId,
  "department": {
    "code": "PM",              # Sortable numeric code
    "name": "Product Management",
    "slug": "product-management"
  },
  "document_type": "Feature Prioritization Framework",
  "document_name": "Feature prioritization framework",
  
  # Core question metadata
  "question": "What is the primary objective?",
  "answer": "User-provided answer or empty string",
  "category": "Overview",
  "category_order": 1,
  "question_order": 1,
  "answer_type": "text|select|multi_select|structured_list",
  "options": ["Option 1", "Option 2"],  # If answer_type is select/multi_select
  "description": "Question guidance for user",
  
  # Gap question markers (★ NEW)
  "is_gap_question": false,                # true = AI-generated from schema gaps
  "section_covered": "Overview",           # Which schema section this question targets
  "answered_at": ISODate("2024-02-20T..."), # When gap was answered & saved
  
  # Indexing hints
  "created_at": ISODate,
  "updated_at": ISODate
}

Indexes:
  • {department.name, document_type} - primary lookup
  • {document_type, is_gap_question} - gap question queries
  • {answered_at} - TTL index (future: auto-expire session data)
```

**Collection: `required_section`** (Document Schemas)
```
{
  "_id": ObjectId,
  "department": "Product Management",
  "document_name": "Feature prioritization framework",
  "document_type": "Feature Prioritization Framework",
  
  "sections": [
    {
      "title": "1. Objective",
      "type": "text|table",
      "order": 1,
      
      # For type="text" sections
      "subsections": [
        {
          "title": "1.1 Business Impact",
          "type": "text|table",
          "order": 1,
          
          # For type="table"
          "columns": ["Column1", "Column2", "Column3"]
        },
        ...
      ],
      
      # For type="table" sections (no subsections)
      "columns": ["Feature ID", "Feature Name", "Priority", "Status"]
    },
    ...
  ]
}

Indexes:
  • {department, document_name} - schema lookup
  • {document_type} - document type queries
```

#### 3.2 DepartmentBasedMongoDBIntegration (`mongo_auto.py`)
```python
class DepartmentBasedMongoDBIntegration:
    """Batch upload Q&As and schemas to MongoDB"""
    
    def process_directory(self, base_dir: str = 'final_filtered_QAs'):
        """
        Iterate through all departments and files:
        
        for each department folder:
            for each document JSON file:
                1. Read file content
                2. Extract questions array
                3. Flatten by category
                4. Add metadata (department, document_type, etc.)
                5. Batch insert to document_qas collection
        """
    
    def extract_optimized_qas(self, schema_data: Dict) -> List[Dict]:
        """
        Transform nested schema structure into flat Q&A records:
        
        Input:  {questions_by_category: [{category, questions: [...]}]}
        Output: [{question, answer, category, category_order, ...}, ...]
        
        This flattening enables MongoDB queries like:
          db.document_qas.find({document_type, category})
        """
```

**Process Flow**:
1. Fetch department list from API (or hardcoded list)
2. For each department, iterate local folder structure
3. For each JSON file, match filename to API document type
4. Match logic (in order of preference):
   - Exact name match
   - File name fully contained in API name
   - API name fully contained in file name
   - Highest token overlap (Jaccard similarity)
5. Extract Q&As and add enrichment fields
6. Batch insert with `insert_many()` for performance
7. Print summary: inserted count, skipped count, errors

#### 3.3 Async Database Access (`api/db.py`)
```python
class AsyncDatabaseConnection:
    """Singleton async MongoDB connection for FastAPI"""
    
    _instance: Optional[AsyncIOMotorClient] = None
    
    @classmethod
    def get_client(cls) -> AsyncIOMotorClient:
        """Lazy initialization on first access"""
        if cls._instance is None:
            cls._instance = AsyncIOMotorClient(MONGODB_URI)
        return cls._instance
    
    @classmethod
    async def close(cls):
        """Called on FastAPI shutdown"""
        if cls._instance:
            cls._instance.close()

# Usage in FastAPI:
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # App running
    await get_db().close_client()

app = FastAPI(lifespan=lifespan)
```

**Key Design Decisions**:
- Singleton pattern prevents connection pool exhaustion
- Lazy initialization allows app startup without DB connectivity
- Motor driver ensures non-blocking queries
- Lifespan context manager guarantees proper cleanup

---

### Layer 4: FastAPI Backend Orchestration
**File**: `api/main.py` (~443 lines)

**Purpose**: REST API gateway connecting Streamlit UI to MongoDB and LangGraph agent

#### 4.1 Endpoint Reference

| Endpoint | Method | Params | Returns | Purpose |
|----------|--------|--------|---------|---------|
| `/departments` | GET | - | `{departments: [{code, name, slug}]}` | List all departments |
| `/document-types` | GET | `department: str` | `{document_types: [{document_type, document_name}]}` | Docs for dept |
| `/questions` | GET | `document_type: str` | `{questions: [...]}` | All Q&As (sorted by category) with pagination support |
| `/required-section` | GET | `department: str, document_name: str` | `{required_section: {sections: [...]}}` | Document schema |
| `/gap-questions` | POST | `GapQuestionsRequest` | `{gap_questions: [...], source: "cache\|generated"}` | ★ NEW Gap analysis |
| `/save-questions` | POST | `SaveQuestionsRequest` | `{saved_count: int}` | ★ NEW Save gaps to DB |
| `/generate` | POST | `GenerateDocumentRequest` | `{generated_document, gap_questions, quality_scores, ...}` | Generate complete document |
| `/generate-section` | POST | `GenerateSectionRequest` | `{section_text: str}` | ★ NEW Generate single section with context memory |
| `/get_all_urls` | GET | - | `{pages: [{notion_url, title}]}` | Page history |

#### 4.2 Novel Endpoints: Gap Analysis & Persistence

**`POST /gap-questions` — Cache-First Gap Analysis**
```python
@app.post("/gap-questions")
async def get_gap_questions(request: GapQuestionsRequest):
    """
    Two-stage logic:
    
    Stage 1: Check cache
      db.document_qas.find_one({
        document_type: request.document_type,
        is_gap_question: True
      })
      
      IF found → return immediately with source="cache"
      ELSE → continue to Stage 2
    
    Stage 2: Generate fresh gaps
      call analyze_gaps_only(
        department, document_type, Q&As, required_section
      )
      
      LLM returns: [{question, category, answer_type, section_covered}]
      return with source="generated"
    """
    
    db = get_db()
    
    # Stage 1: Check MongoDB
    existing_gaps = await db["document_qas"].find_one(
        {"document_type": request.document_type, "is_gap_question": True}
    )
    
    if existing_gaps:
        cursor = db["document_qas"].find(
            {"document_type": request.document_type, "is_gap_question": True}
        ).sort([("question_order", 1)])
        
        cached_questions = await cursor.to_list(length=100)
        return {
            "gap_questions": cached_questions,
            "source": "cache",
            "count": len(cached_questions)
        }
    
    # Stage 2: Generate fresh
    required_section = request.required_section or await fetch_schema()
    gaps = await analyze_gaps_only(
        department=request.department,
        document_type=request.document_type,
        questions_and_answers=request.questions_and_answers,
        required_section=required_section
    )
    
    return {
        "gap_questions": gaps,
        "source": "generated",
        "count": len(gaps)
    }
```

**Performance Benefit**:
- First user for a document type: ~10 seconds (LLM call)
- Subsequent users for same document type: <100ms (MongoDB lookup)
- Scales: O(1) LLM calls per document type, not O(users)

**`POST /save-questions` — Deduplication & Persistence**
```python
@app.post("/save-questions")
async def save_questions(request: SaveQuestionsRequest):
    """
    Upsert gap questions to MongoDB with conflict resolution.
    
    For each gap question:
      db.document_qas.update_one(
        filter={
          document_type: request.document_type,
          question: q.question,
          is_gap_question: True
        },
        update={
          $set: {
            answer: q.answer,
            answered_at: now(),
            section_covered: q.section_covered
          }
        },
        upsert=True
      )
    """
    
    db = get_db()
    now = datetime.utcnow()
    
    bulk_ops = []
    for gap_q in request.gap_questions:
        bulk_ops.append(
            UpdateOne(
                {
                    "document_type": request.document_type,
                    "question": gap_q["question"],
                    "is_gap_question": True
                },
                {
                    "$set": {
                        "answer": gap_q.get("answer", ""),
                        "answered_at": now,
                        "department": request.department,
                        "document_name": request.document_name,
                        "category": gap_q.get("category", "Additional"),
                        "section_covered": gap_q.get("section_covered", ""),
                        "category_order": 999,  # Always render last
                        "question_order": 1000 + len(bulk_ops)  # After core Qs
                    }
                },
                upsert=True
            )
        )
    
    if bulk_ops:
        result = await db["document_qas"].bulk_write(bulk_ops)
        return {"saved_count": result.upserted_id.__len__()}
    
    return {"saved_count": 0}
```

**Key Design Details**:
- Upsert on (document_type, question, is_gap_question) prevents duplicates
- `question_order` set to 1000+ ensures gap questions sort after core ones in UI
- `category_order: 999` makes gap section render last visually
- `answered_at` tracks when gap was answered for analytics/debugging

#### 4.3 CORS & Security Configuration
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Restricts API to local Streamlit frontend only
# Prevents unauthorized cross-origin requests
```

#### 4.4 Progressive Generation Endpoint ★ NEW

**`POST /generate-section` — Generate Single Section with Memory**

```python
@app.post("/generate-section")
async def generate_section_endpoint(request: GenerateSectionRequest):
    """
    Generate ONE document section with context memory of previous sections.
    
    Used in progressive generation mode where sections are built incrementally
    instead of generating the entire document at once.
    
    Request body:
        {
            "department": str,
            "document_type": str,
            "section": {...},  # single section from required_section.sections
            "questions_and_answers": [...],  # all user Q&As
            "doc_memory": str  # rendered markdown of previous sections
        }
    
    Response:
        {"section_text": str}  # rendered markdown for this section only
    """
    
    section_text = await generate_single_section(
        department=request.department,
        document_type=request.document_type,
        section=request.section,
        questions_and_answers=request.questions_and_answers,
        doc_memory=request.doc_memory
    )
    
    return {"section_text": section_text}
```

**Progressive Generation Flow**:
1. User clicks "Progressive Mode" toggle in UI
2. UI breaks required schema into individual sections
3. For each section:
   - Call `POST /generate-section` with that section + cumulative document memory
   - Receive rendered markdown for that section
   - Append to document buffer
   - Stream updates to UI in real-time
4. User can pause, edit intermediate sections, or regenerate problematic sections
5. Better for long documents (Product Roadmap, etc.) where full generation might timeout

**Advantages**:
- Real-time feedback (sections appear as they're generated)
- Recoverable from mid-generation failures (restart from failed section)
- Context memory ensures consistency between sections
- Reduces risk of full-document timeout on long generation
- Users can edit and resubmit without full regeneration

---

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

## ⚙️ In-Depth Component Specifications

### Layer 5: LangGraph Agent for Document Generation (Extended)
*See detailed specifications above in "Detailed Data Flow & Processing Layers" section (5.1-5.5)*

**Summary**: 5-node state machine orchestrating schema gap analysis, prompt building, document generation with primary LLM, quality validation, and automated fixes with retry logic.

**Key Innovation**: Separates gap identification (lightweight LLM) from document generation (premium LLM), enabling efficient scaling and better quality through user-in-the-loop feedback.

---

### Layer 6: Streamlit Frontend UI (Extended)
*See detailed specifications above in "Detailed Data Flow & Processing Layers" section (6.1-6.7)*

**Summary**: Interactive web interface providing department/document selection, multi-type Q&A widgets, gap analysis UI, document generation trigger, and markdown editing with quality feedback.

**Key Innovation**: Unified gap question rendering (both MongoDB-persisted and session-fresh), automatic cache clearing on document selection, and shared widget renderer eliminating code duplication.

---

## 📁 Complete File Structure & Purposes

```
DocForgeHub/
│
├── agent/                          # LangGraph orchestration
│   ├── agent_graph.py              # 5-node document generation workflow
│   │                                 • analyze_schema_gaps (NEW)
│   │                                 • build_prompt
│   │                                 • generate_document (Kimi-k2)
│   │                                 • quality_gate (deterministic + LLM)
│   │                                 • fix_document (retry + fix)
│   │                                 • analyze_gaps_only() utility
│   ├── prompts.py                  # System prompt templates
│   │                                 • SYSTEM_PROMPT_TEMPLATE (mixed docs)
│   │                                 • TABLE_PROMPT_TEMPLATE (table-only)
│   │                                 • QUALITY_REVIEW_PROMPT
│   │                                 • build_system_prompt()
│   │                                 • build_quality_prompt()
│   └── __init__.py
│
├── api/                            # FastAPI REST endpoints
│   ├── main.py                     # 9 endpoints + CORS configuration
│   │                                 • GET /departments
│   │                                 • GET /document-types
│   │                                 • GET /questions
│   │                                 • GET /required-section
│   │                                 • POST /gap-questions (cache-first)
│   │                                 • POST /save-questions (upsert)
│   │                                 • POST /generate (full document)
│   │                                 • POST /generate-section ★ NEW (progressive)
│   │                                 • GET /get_all_urls
│   ├── db.py                       # Async MongoDB connection (singleton)
│   │                                 • AsyncIOMotorClient
│   │                                 • Lifespan management
│   │                                 • Lazy initialization
│   └── __init__.py
│
├── automations/                    # Data pipeline & batch operations
│   ├── ques_automation.py          # Notion extraction + Q&A generation
│   │                                 • NotionContentExtractor
│   │                                 • GroqLangGraphQuestionGenerator
│   │                                 • Multi-key API fallback
│   ├── automation.py               # Content extraction utilities
│   ├── add_answer_field.py         # Answer field addition & organization
│   ├── mongo_auto.py               # MongoDB batch upload (~600 lines)
│   │                                 • DepartmentBasedMongoDBIntegration
│   │                                 • Batch insert optimization
│   │                                 • Index creation
│   ├── required_sections_automation.py  # Schema upload to MongoDB
│   │                                      • Fuzzy matching (exact/subset/tokens)
│   │                                      • Department-to-API reconciliation
│   ├── clean_reorder.py            # Data cleanup utilities
│   ├── run_clean_reorder.py        # Execution script
│   └── __init__.py
│
├── ui/                             # Streamlit interactive frontend
│   ├── streamlit_uidemo.py         # Complete UI (~850 lines)
│   │                                 • render_question_widget()
│   │                                 • Sidebar navigation
│   │                                 • Q&A panels (core + gap)
│   │                                 • Document editor
│   │                                 • API helpers (cached)
│   │                                 • Session state management
│   └── (no __init__.py for streamlit)
│
├── document_and_questions/         # Data repository
│   ├── final_filtered_QAs/         # Final Q&A storage (100 files)
│   │   ├── 1._Product_Management/  # 10 documents
│   │   ├── 2._Engineering__Software_Development/
│   │   ├── 3._Information_Security/
│   │   ├── 4._Quality_Assurance_(QA)__Testing/
│   │   ├── 5._Compliance__Regulatory/
│   │   ├── 6._Sales/
│   │   ├── 7._Marketing/
│   │   ├── 8._Customer_Support/
│   │   ├── 9._Human_Resources_(HR)/
│   │   └── 10._Finance/
│   │       └── [Each folder contains 10 {document}_questions.json files]
│   │
│   └── notion_documents/           # Extracted Notion content (parallel structure)
│       └── [Same 10-department structure]
│
├── .env                            # Secrets (NOT in version control)
│   ├── GROQ_API_KEY (and _2 through _7)
│   ├── MONGODB_CONNECTION_STRING
│   ├── MONGODB_DATABASE
│   └── NOTION_API_KEY
│
├── CODEBASE_ARCHITECTURE.md        # This file
├── progress.md                     # Development changelog
└── .gitignore                      # Excludes .env, __pycache__, venv
```

---

## 🔄 Complete User Journey: From Notion to Published Document

### 1️⃣ Setup Phase (One-time Admin)
```
Admin Action: Extract from Notion

  automations/automation.py + ques_automation.py
  ├─ Initialize NotionContentExtractor with API key
  ├─ Fetch all pages recursively from root workspace
  ├─ Extract markdown content from each page
  ├─ Call GroqLangGraphQuestionGenerator (3-node LangGraph)
  │  ├─ Node 1: Analyze document structure & patterns
  │  ├─ Node 2: Call Groq LLM to generate questions
  │  └─ Node 3: Validate & format output
  └─ Output: notion_documents/ + generated_questions/ (100 files)

Admin Action: Prepare for storage

  automations/add_answer_field.py
  ├─ Read generated_questions/
  ├─ Add empty "answer" field to each question
  ├─ Add category_order, question_order, is_gap_question: false
  └─ Output: final_filtered_QAs/ (100 files, ready for MongoDB)

Admin Action: Upload to MongoDB

  automations/mongo_auto.py + required_sections_automation.py
  ├─ Read final_filtered_QAs/ files
  ├─ Batch insert to document_qas collection
  ├─ Create indexes for fast queries
  ├─ Read notion_documents/ and extract schemas
  ├─ Batch insert to required_section collection
  └─ MongoDB now contains all data for runtime
```

### 2️⃣ Runtime Phase (Per User Session)

```
User Opens Streamlit UI (localhost:8501)

  ├─ UI initializes session state
  ├─ Fetch departments (GET /departments) → Cached 5 min
  └─ Render sidebar department selector

User Selects Department

  ├─ Fetch document types (GET /document-types) → Cached 5 min
  ├─ Auto-clear previous gap questions (safety)
  └─ Render sidebar document selector

User Selects Document

  ├─ Fetch Q&As (GET /questions) → Cached 5 min
  │  • Includes core questions (is_gap_question: false)
  │  • Includes MongoDB-persisted gap questions (is_gap_question: true)
  ├─ Fetch schema (GET /required-section)
  └─ Render Q&A panels (core + gap) with appropriate widgets

User Fills in Core Answers

  └─ Answers stored in st.session_state.answers

(Optional) User Clicks "🔍 Analyse Schema Gaps"

  ├─ POST /gap-questions with:
  │  • department, document_type, Q&As, required_section
  │
  ├─ Backend logic:
  │  ├─ Check MongoDB for existing gap questions
  │  │  └─ If found → return immediately (source: "cache")
  │  └─ If not found → call analyze_gaps_only()
  │     ├─ Run NODE 1: Call Llama-3.3-70b LLM
  │     ├─ LLM returns: [{question, category, section_covered}, ...]
  │     └─ Return (source: "generated")
  │
  ├─ UI receives gap questions
  ├─ Render in "✨ New Gap Questions" section with AI badge
  ├─ User fills in gap answers
  │  └─ Answers stored in st.session_state.gap_answers
  │
  └─ (Optional) User Clicks "💾 Save gap questions"
     ├─ POST /save-questions with gap questions + answers
     ├─ Backend: Upsert to MongoDB
     │  • Filter: {document_type, question, is_gap_question: True}
     │  • Set: category_order: 999, question_order: 1000+
     │  • Result: Next user sees these questions automatically
     └─ UI: Clear cache, show confirmation

User Clicks "⚡ Generate Document"

  ├─ (Option 1) Standard Mode:
  │  ├─ POST /generate with:
  │  │  • department, document_type, questions_and_answers, required_section
  │  │
  │  ├─ Backend invokes agent (5-node LangGraph):
  │  │  ├─ NODE 1: analyze_schema_gaps
  │  │  │  └─ Lightweight LLM identifies any additional gaps
  │  │  ├─ NODE 2: build_prompt
  │  │  │  └─ Format Q&As, schema, supplementary notes
  │  │  ├─ NODE 3: generate_document
  │  │  │  └─ Primary LLM (Kimi-k2) generates markdown
  │  │  ├─ NODE 4: quality_gate
  │  │  │  ├─ Table-only: Validate markdown table columns
  │  │  │  └─ Mixed: LLM review for completeness & tone
  │  │  └─ NODE 5: fix_document (if needed)
  │  │     └─ Re-prompt with issues, retry up to 2x
  │  │
  │  └─ Return: full markdown + quality scores + issues
  │
  └─ (Option 2) Progressive Mode ★ NEW:
     ├─ Split schema into individual sections
     ├─ For each section:
     │  ├─ POST /generate-section with section + doc_memory
     │  ├─ Stream section_text back to UI
     │  ├─ Append to document buffer
     │  └─ User can pause, edit, or regenerate
     └─ Advantages: real-time feedback, pauseable, recoverable

  ├─ UI receives document (either mode)
  ├─ Display in markdown editor (right column)
  ├─ Show quality scores (completeness, professionalism, clarity)
  ├─ Show any issues/suggestions
  ├─ Allow user edits
  └─ Add to generation history

User Clicks "📤 Publish to Notion"

  ├─ POST to Notion API
  ├─ Create page with document title
  ├─ Insert generated markdown content
  ├─ Return Notion URL
  └─ Add to history (clickable links)
```

---

## 🎓 Key Architectural Highlights

✅ **Modular & Testable**: Each layer (extraction, storage, API, agent, UI) independently developed and tested
✅ **State-Driven Workflows**: LangGraph ensures deterministic execution with clear visibility into each step
✅ **Async-First Scaling**: FastAPI + Motor handle concurrent users without thread bottlenecks
✅ **Intelligent Caching**: 3-level cache (Streamlit session, MongoDB gap cache, API response cache) minimizes LLM calls
✅ **Two-LLM Strategy**: Expensive primary model for prose, cheap secondary model for analysis → cost optimization
✅ **Schema-Driven Quality**: Every document validated against required schema before delivery
✅ **User-in-the-Loop Gap Filling**: Identifies gaps → asks users → incorporates real answers → zero hallucination
✅ **Resilient APIs**: Multiple Groq keys, exponential backoff, clear error messages, comprehensive logging
✅ **Professional Output**: Content elevation rules, quality review LLM, automatic retry/fix mechanism
✅ **Great UX**: Intuitive sidebar, real-time gap discovery, editable output, quick history access

---

## 🔐 Security & Deployment Considerations

### Environment Configuration
```bash
# .env (Never commit this file)
GROQ_API_KEY="gsk_..."                        # Primary key
GROQ_API_KEY_2="gsk_..."                      # Fallback keys (up to 7)
MONGODB_CONNECTION_STRING="mongodb+srv://..."
MONGODB_DATABASE="document_automation"
NOTION_API_KEY="secret_..."
```

### Network Security
- **CORS Policy**: Restrict to local Streamlit only (`localhost:8501`, `127.0.0.1:8501`)
- **API Auth**: Currently trusts local network (suitable for internal SaaS use)
- **MongoDB**: Use IP whitelist on MongoDB Atlas; never expose connection string to frontend

### Data Privacy
- Gap questions are persisted and shared across users — ensure questions don't contain PII
- Consider data retention policies for `answered_at` timestamps
- Audit logging for document generation (who, when, document type)

---

## 📊 Performance Characteristics

| Operation | Typical Time | Bottleneck |
|-----------|-------------|-----------|
| **GET /departments** | <10ms | MongoDB index on department |
| **GET /document-types** | <15ms | MongoDB aggregation |
| **GET /questions** (500+) | <50ms | Sorting by category_order |
| **POST /gap-questions** (cache hit) | <100ms | MongoDB lookup |
| **POST /gap-questions** (fresh) | ~10-15s | Llama-3.3-70b LLM call |
| **POST /generate** (full document) | ~30-60s | Kimi-k2 LLM generation + quality review |
| **POST /generate-section** (per section) | ~5-15s | Single section LLM generation with memory |
| **Full document gen + quality + fix** | ~90-120s | Multi-node retry loops |
| **Progressive mode (all sections)** | ~30-120s | Cumulative per-section time (parallelizable) |
| **POST /save-questions** | <50ms | MongoDB bulk_write |
| **Streamlit page load** | ~200ms | Data fetching + rendering |

---

**Last Updated**: February 24, 2026  
**Architecture Version**: 2.1 (Added Progressive Generation Endpoint)**