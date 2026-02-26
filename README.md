# 📄 DocForgeHub

> **AI-Powered Intelligent Document Generation & Management System**

An enterprise-grade platform that transforms user-supplied answers into schema-compliant, professionally-written business documents. Using a dual-LLM architecture, a 5-node LangGraph agent, intelligent gap analysis, and user-in-the-loop feedback, DocForgeHub ensures every generated document is complete, accurate, and publication-ready.

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io/)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green.svg)](https://www.mongodb.com/cloud/atlas)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.1+-orange.svg)](https://langchain-ai.github.io/langgraph/)

---

## 🎯 What is DocForgeHub?

DocForgeHub is an intelligent document generation platform designed for SaaS organizations that need to produce professional business documents at scale. The system:

1. **Stores** comprehensive Q&A pairs and document schemas in MongoDB, organized by department and document type
2. **Surfaces** the right questions to users via a Streamlit UI with paginated, categorized widgets
3. **Identifies** schema coverage gaps with a lightweight LLM and generates targeted follow-up questions
4. **Persists** answered gap questions to MongoDB so future users benefit without repeating the analysis
5. **Generates** professional, schema-compliant documents via a 5-node LangGraph agent
6. **Validates** generated documents deterministically (table schemas) and semantically (LLM-based review)
7. **Publishes** final documents back to Notion and exports them to PDF

### Key Innovation: User-in-the-Loop Gap Filling

Instead of hallucinating missing content, DocForgeHub:
- Uses Llama-3.3-70b to identify which schema sections lack Q&A coverage
- Generates targeted questions that ask users only for what is actually missing
- Persists answered gap questions to MongoDB for reuse across all future users
- Results in higher-quality documents with **zero hallucination risk**

---

## 🏗️ Architecture Overview

```
Streamlit UI (Port 8501)
    │
    │  REST (HTTP)
    ▼
FastAPI Backend (Port 8000)
    ├── MongoDB Atlas  ←→  document_qas + required_section collections
    ├── Notion API         (page URL history)
    └── LangGraph Agent
            ├── Node 1: analyze_gaps
            ├── Node 2: build_prompt
            ├── Node 3: generate_document   ← kimi-k2-instruct (primary LLM)
            ├── Node 4: quality_gate        ← kimi-k2-instruct (review)
            └── Node 5: fix_document        ← kimi-k2-instruct (retry, up to 2x)
```

**For a full architectural deep-dive**, see [CODEBASE_ARCHITECTURE.md](CODEBASE_ARCHITECTURE.md)

### Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Primary LLM** | Groq + `kimi-k2-instruct-0905` | Document generation, quality review, fix attempts |
| **Analysis LLM** | Groq + `llama-3.3-70b-versatile` | Schema gap analysis, structured JSON output |
| **Orchestration** | LangGraph | 5-node state machine for document generation |
| **Backend API** | FastAPI | Async REST gateway (9 endpoints) |
| **Database** | MongoDB Atlas | Q&As, schemas, gap question cache |
| **Async Driver** | Motor | Non-blocking MongoDB operations |
| **Frontend** | Streamlit | Interactive Q&A UI and document editor |
| **Content Source** | Notion API | Page URL history, recursive hierarchy retrieval |
| **PDF Export** | ReportLab | Markdown → professional A4 PDF |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- MongoDB Atlas account (free tier sufficient)
- Groq API key (free tier: ~200K tokens/month)
- Notion workspace & integration key

### 1. Clone & Set Up Environment

```bash
git clone <repository-url>
cd DocForgeHub

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the **project root**:

```env
# Groq API Keys (primary + optional fallbacks)
GROQ_API_KEY="gsk_your_primary_key"
GROQ_API_KEY_2="gsk_fallback_key_1"
# ... up to GROQ_API_KEY_7

# MongoDB Atlas
MONGODB_CONNECTION_STRING="mongodb+srv://user:password@cluster.mongodb.net/"

# Notion
NOTION_API_KEY="secret_your_notion_integration_key"
```

> ⚠️ **Never commit `.env` to version control.** It is already listed in `.gitignore`.

### 3. Seed MongoDB (One-Time Admin Step)

```bash
# Upload Q&A pairs for all departments/document types
cd automations
python mongo_auto.py

# Upload document schemas (required_section collection)
python required_sections_automation.py
```

### 4. Start the Backend

```bash
# From project root
python -m uvicorn api.main:app --reload --port 8000
```

Visit `http://localhost:8000/docs` to confirm the API is running.

### 5. Start the Frontend

```bash
# From the ui/ directory (new terminal)
cd ui
streamlit run streamlit_uidemo.py
```

Navigate to `http://localhost:8501` — the app is ready.

---

## 📁 Project Structure

```
DocForgeHub/
│
├── agent/                          # LangGraph document generation agent
│   ├── __init__.py
│   ├── agent_graph.py              # 5-node state machine + AgentState TypedDict
│   ├── prompts.py                  # System prompt builders (mixed + table-only)
│   ├── schema_helpers.py           # Schema parsing, Q&A → prompt formatting
│   └── validation_helpers.py       # Structure & table column validation
│
├── api/                            # FastAPI REST backend
│   ├── __init__.py
│   ├── main.py                     # 9 endpoints (departments → generate)
│   ├── db.py                       # Async Motor/MongoDB singleton connection
│   └── helpers.py                  # Notion API: recursive page traversal
│
├── ui/                             # Streamlit interactive frontend
│   ├── streamlit_uidemo.py         # Main app (sidebar, Q&A panels, doc editor)
│   ├── api_helpers.py              # HTTP wrappers for all FastAPI endpoints
│   ├── question_helpers.py         # Unified question list, category helpers, widget renderer
│   └── pdf_generator.py            # ReportLab Markdown → A4 PDF export
│
├── automations/                    # Batch data pipeline (admin / one-time)
│   ├── mongo_auto.py               # Upload Q&A JSON files → MongoDB
│   └── required_sections_automation.py  # Upload schema JSON files → MongoDB
│
├── document_and_questions/         # Local data repository
│   ├── final_filtered_QAs/         # Q&A JSON files, organized by department
│   └── notion_documents/           # Schema JSON files extracted from Notion
│
├── .env                            # Environment variables (never commit)
├── .gitignore
├── requirements.txt
├── CODEBASE_ARCHITECTURE.md        # Deep architectural reference
├── progress.md                     # Development changelog
└── README.md                       # This file
```

---

## 💡 How It Works

### User Session Flow

```
1. Select Department + Document Type (sidebar)
        ↓
2. Fetch Q&As from MongoDB (core + previously saved gap questions)
        ↓
3. Fill answers across paginated, categorized question panels
        ↓
4. [Optional] Click "🔍 Analyse Schema Gaps"
   → Llama-3.3-70b checks which schema sections lack coverage
   → Returns targeted gap questions (from cache or freshly generated)
   → User answers gap questions
   → Click "💾 Save" → upserted to MongoDB for future users
        ↓
5. Click "⚡ Generate Document"
   → POST /generate → 5-node LangGraph agent:
        Node 1: analyze_gaps      – identify any remaining uncovered sections
        Node 2: build_prompt      – assemble full system prompt from schema + Q&As
        Node 3: generate_document – kimi-k2 generates complete Markdown document
        Node 4: quality_gate      – deterministic checks + LLM review + scoring
        Node 5: fix_document      – rewrite if quality fails (up to 2 retries)
        ↓
6. Review output: Markdown editor + quality scores + issue list
        ↓
7. [Optional] Export to PDF  |  Publish to Notion
```

---

## 🔌 API Reference

All endpoints are served at `http://localhost:8000`. Interactive docs at `/docs`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/departments` | All departments, sorted by code |
| `GET` | `/document-types?department=` | Document types for a department |
| `GET` | `/questions?document_type=` | All Q&As (core + saved gaps), sorted |
| `GET` | `/required-section?department=&document_name=` | Document schema |
| `POST` | `/gap-questions` | Cache-first gap analysis → gap questions |
| `POST` | `/save-questions` | Upsert answered gap questions to MongoDB |
| `POST` | `/generate` | Full 5-node agent → complete document |
| `POST` | `/generate-section` | Generate one section with doc memory |
| `GET` | `/get_all_urls` | Notion page URL history |

### Example: Generate a Document

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "department": "Product Management",
    "document_type": "Feature Prioritization Framework",
    "document_name": "Feature prioritization framework",
    "questions_and_answers": [
      {"question": "What is the primary objective?", "answer": "Increase retention", "category": "Overview"}
    ]
  }'
```

---

## 🔑 Key Features

| Feature | Detail |
|---------|--------|
| **Dual-LLM routing** | kimi-k2 for prose quality; llama-3.3-70b for cheap structured analysis |
| **5-node LangGraph agent** | Deterministic state machine with automatic retry (up to 2x) |
| **Cache-first gap analysis** | O(1) LLM calls per document type after first user; reuses MongoDB cache |
| **Two validation modes** | Deterministic (table column checks) + LLM-based (semantic structure review) |
| **Schema-driven generation** | Every output validated against MongoDB required_section schema |
| **PDF export** | ReportLab renders Markdown → styled A4 PDF (tables, headings, bullets) |
| **Groq key fallback** | Up to 7 API keys rotated automatically on rate-limit |
| **Async throughout** | FastAPI + Motor → non-blocking, concurrent-session capable |
| **Streamlit caching** | `st.cache_data` with 5–10 min TTL for departments, doc types, questions |

---

## 📊 Supported Scope

DocForgeHub ships with data for **10 departments × 10 document types = 100 document types**:

| Department | Sample Documents |
|-----------|-----------------|
| Product Management | PRD, Feature Prioritization, Roadmap, User Story Backlog |
| Engineering | API Documentation, Deployment Runbook, Coding Standards |
| Quality Assurance | Test Plans, Bug Reports, QA Checklists |
| Security | Security Policies, Incident Reports, Risk Assessment |
| Compliance | Regulatory Checklists, Audit Reports, Policy Documents |
| Sales | Proposals, Case Studies, Pricing Sheets |
| Marketing | Campaign Plans, Content Calendars, Brand Guidelines |
| Support | Knowledge Base, Support Processes, FAQs |
| HR | Onboarding Guides, Policies, Team Handbooks |
| Finance | Budget Documents, Financial Reports, Expense Policies |

---

## 📈 Performance Characteristics

| Operation | Typical Time | Notes |
|-----------|-------------|-------|
| `GET /questions` | < 50 ms | Streamlit cache hit (5 min TTL) |
| `POST /gap-questions` (cache hit) | < 100 ms | Direct MongoDB lookup |
| `POST /gap-questions` (fresh) | 10–15 s | llama-3.3-70b LLM call |
| `POST /generate` | 30–60 s | Full 5-node workflow, kimi-k2 |
| PDF export | < 2 s | Pure ReportLab, no LLM |

---

## 🛠️ Development Guide

### Adding a New Document Type

1. Create a Q&A JSON file: `document_and_questions/final_filtered_QAs/{department}/{doc_name}.json`
2. Create a schema JSON file: `document_and_questions/notion_documents/{department}/{doc_name}.json` (must include a `sections` array)
3. Upload to MongoDB:
   ```bash
   cd automations
   python mongo_auto.py
   python required_sections_automation.py
   ```
4. Streamlit cache clears automatically on next page reload.

### Modifying LLM Prompts

Edit `agent/prompts.py`:
- `SYSTEM_PROMPT_TEMPLATE` — for mixed (text + table) documents
- `TABLE_PROMPT_TEMPLATE` — for table-only documents
- `build_quality_review_prompt()` — for the quality gate node
- `build_gap_filler_prompt()` — for the gap analysis LLM call

### Extending the Agent

Edit `agent/agent_graph.py`:
1. Add a new node function
2. Register it on the `StateGraph` builder
3. Update `AgentState` TypedDict if new fields are needed
4. Wire routing edges (conditional or direct)

---

## 🔐 Security

- **Never commit `.env`** — all secrets live there
- CORS is locked to `localhost:8501` and `127.0.0.1:8501` — update `api/main.py` for production
- Use MongoDB Atlas IP allowlist in production
- Rotate Groq API keys regularly; configure fallbacks via `GROQ_API_KEY_2` … `GROQ_API_KEY_7`

---

## 🐛 Troubleshooting

**Backend won't start**
```bash
# Check port conflict
lsof -i :8000

# Verify MongoDB reachability
python -c "from pymongo import MongoClient; print(MongoClient('YOUR_URI').list_database_names())"
```

**Streamlit blank / errors**
```bash
# Confirm backend is running
curl http://localhost:8000/docs

# Clear Streamlit cache
streamlit cache clear
```

**Gap questions not saving** — Check that `document_type` matches exactly what is stored in MongoDB. Monitor logs for `POST /save-questions` errors.

**Generation > 60 s** — Check Groq API latency. Try reducing the number of Q&As passed (fewer answered questions = shorter prompt).

---

## 📚 Documentation

| File | Purpose |
|------|---------|
| [CODEBASE_ARCHITECTURE.md](CODEBASE_ARCHITECTURE.md) | Complete architectural reference (all files, data flows, design decisions) |
| [progress.md](progress.md) | Development changelog |
| README.md | This quick-start guide |

---

## 📜 License

[Specify your license here]

---

<div align="center">

**Built for document-driven SaaS organizations**

[⬆ Back to top](#-docforgehub)

</div>