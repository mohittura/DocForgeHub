# 📄 DocForgeHub

> **AI-Powered Intelligent Document Generation & Management System**

An enterprise-grade SaaS platform that transforms business documents from Notion into schema-compliant, professionally-written documents. Using multi-LLM orchestration, intelligent gap analysis, and user-in-the-loop feedback, DocForgeHub ensures every generated document is complete, accurate, and publication-ready.

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io/)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green.svg)](https://www.mongodb.com/cloud/atlas)

---

## 🎯 What is DocForgeHub?

DocForgeHub is an intelligent document generation platform designed for organizations that need to create professional business documents at scale. Instead of manually writing documents or using static templates, DocForgeHub:

1. **Extracts** business documents from Notion with hierarchical structure preservation
2. **Generates** comprehensive Q&A pairs from document content using LangGraph workflows
3. **Stores** all Q&As and schemas in MongoDB, organized by department and document type
4. **Identifies** gaps in document coverage and generates targeted questions for users
5. **Generates** professional, schema-compliant documents from user answers using AI
6. **Validates** generated documents against required structure with deterministic and LLM-based checks
7. **Publishes** final documents back to Notion for team collaboration

### Key Innovation: User-in-the-Loop Gap Filling

Unlike traditional document generation systems that hallucinate missing content, DocForgeHub:
- Uses a lightweight LLM to identify which document sections lack coverage
- Generates targeted questions asking users for missing information
- Persists answered questions to MongoDB for immediate reuse by other users
- Results in higher-quality documents with **zero hallucination risk**

---

## 🏗️ Architecture Overview

```
Notion Documents → Q&A Generation → MongoDB Storage → FastAPI Backend → LangGraph Agent → Streamlit UI
                                                              ↓
                                                     Gap Analysis & Caching
```

**For a comprehensive architectural deep-dive**, see [CODEBASE_ARCHITECTURE.md](CODEBASE_ARCHITECTURE.md)

### Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Document Extraction** | Notion API | Pull business documents from Notion |
| **LLM Orchestration** | LangGraph | Multi-step workflows for Q&A generation & document assembly |
| **Primary LLM** | Groq + Kimi-k2 | High-quality long-form document generation |
| **Analysis LLM** | Groq + Llama-3.3-70b | Fast, cost-effective schema gap analysis |
| **Backend API** | FastAPI | Async REST endpoints for all operations |
| **Database** | MongoDB Atlas | Flexible schema storage with fast aggregation |
| **Async Driver** | Motor | Non-blocking MongoDB operations |
| **Frontend** | Streamlit | Interactive Q&A interface & document editor |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- MongoDB Atlas account (free tier sufficient)
- Groq API keys (free tier includes ~200K free tokens/month)
- Notion workspace & API key

### 1. Clone Repository

```bash
git clone <repository-url>
cd DocForgeHub
```

### 2. Set Up Python Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Groq API Keys (primary + fallback)
GROQ_API_KEY="gsk_your_primary_key"
GROQ_API_KEY_2="gsk_fallback_key_1"
GROQ_API_KEY_3="gsk_fallback_key_2"
# ... up to GROQ_API_KEY_7

# MongoDB Atlas
MONGODB_CONNECTION_STRING="mongodb+srv://username:password@cluster.mongodb.net/"
MONGODB_DATABASE="document_automation"

# Notion
NOTION_API_KEY="secret_your_notion_api_key"
```

> ⚠️ **Never commit `.env` to version control!** Add it to `.gitignore`

### 4. Initialize MongoDB Collections

```bash
# From the automations directory
cd automations

# 4a. Upload Q&As (if you have extracted documents)
python mongo_auto.py

# 4b. Upload schemas/requirements
python required_sections_automation.py
```

### 5. Start Backend Server

```bash
# From project root
python -m uvicorn api.main:app --reload --port 8000
```

Expected output:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete
```

### 6. Start Streamlit Frontend

```bash
# From project root (new terminal)
cd ui
streamlit run streamlit_uidemo.py
```

Expected output:
```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
```

### 7. Open in Browser

Navigate to `http://localhost:8501` and start generating documents!

---

## 📁 Project Structure

```
DocForgeHub/
├── agent/                          # LangGraph 5-node document generation agent
│   ├── agent_graph.py              # State machine orchestration
│   ├── prompts.py                  # System prompts for LLM
│   └── __init__.py
│
├── api/                            # FastAPI REST backend
│   ├── main.py                     # 8 endpoints (departments, documents, generation)
│   ├── db.py                       # Async MongoDB connection
│   └── __init__.py
│
├── automations/                    # Data pipeline & batch operations
│   ├── ques_automation.py          # Notion → Q&A generation
│   ├── mongo_auto.py               # Q&A → MongoDB upload
│   ├── required_sections_automation.py  # Schema → MongoDB upload
│   └── ...
│
├── ui/                             # Streamlit interactive frontend
│   └── streamlit_uidemo.py         # Complete user interface
│
├── document_and_questions/         # Data repository
│   ├── final_filtered_QAs/         # Q&A files (by department)
│   └── notion_documents/           # Extracted Notion content
│
├── CODEBASE_ARCHITECTURE.md        # 📘 Detailed architecture documentation
├── README.md                        # This file
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
└── .gitignore                      # Git ignore rules

```

**For detailed file-by-file explanations**, see [CODEBASE_ARCHITECTURE.md - File Structure](CODEBASE_ARCHITECTURE.md#-complete-file-structure--purposes)

---

## 💡 How It Works: Complete User Journey

### Phase 1: Setup (Admin - One-time)

```
Admin extracts Notion documents
    ↓
LangGraph generates Q&A pairs
    ↓
Add answer fields & organize
    ↓
Batch upload to MongoDB (100 documents, 1000+ Q&As)
    ↓
Ready for users!
```

### Phase 2: User Session (Interactive)

```
User selects Department + Document
    ↓
Fetch Q&As from MongoDB (core + cached gap questions)
    ↓
User fills in answers
    ↓
(Optional) User clicks "🔍 Analyse Schema Gaps"
    → Lightweight LLM identifies missing sections
    → Generate targeted questions
    → User provides real answers
    → Save for next user (cache-first design)
    ↓
User clicks "⚡ Generate Document"
    → 5-node LangGraph workflow:
        1. Analyse remaining gaps
        2. Build formatted prompt
        3. Generate document (Kimi-k2 LLM)
        4. Quality gate validation
        5. Fix & retry if needed (up to 2 retries)
    ↓
Display markdown + quality scores
    ↓
(Optional) Edit and publish to Notion
```

---

## 🔑 Key Features

✨ **Intelligent Gap Analysis**
- Automatically identifies which document sections lack Q&A coverage
- Generates targeted questions asking users for missing information
- Persists gap questions to MongoDB for reuse across users
- **Zero hallucination** — no synthetic content

🧠 **Dual-LLM Architecture**
- **Kimi-k2**: Premium model for high-quality document prose
- **Llama-3.3-70b**: Efficient model for structured gap analysis
- Intelligent routing → cost optimization + quality

🔄 **Multi-Node Orchestration**
- LangGraph state machines for deterministic workflows
- Automatic retry logic with progressive fixes
- Clear visibility into each generation step

📊 **Schema-Driven Validation**
- Every document validated against required structure
- Two validation modes:
  - **Deterministic**: For table-only documents (exact column validation)
  - **LLM-based**: For mixed documents (semantic completeness check)
- Auto-fix mechanism with quality scoring

🎨 **Professional Output**
- Content elevation rules (transforms raw answers into polished prose)
- Consistent formatting and tone
- Quality metrics (completeness, professionalism, clarity)

⚡ **Performance & Caching**
- Streamlit session caching (5-10 min TTL)
- MongoDB gap question cache (O(1) LLM calls per document type)
- Async I/O throughout (FastAPI + Motor)
- Typically generates documents in 30-60 seconds

🔐 **Enterprise-Ready**
- CORS security (local Streamlit frontend only)
- Environment-based configuration
- API key fallback mechanism (up to 7 Groq keys)
- Comprehensive error handling & logging

---

## 📊 Supported Document Types

DocForgeHub supports **10 departments** with **10 document types each** (100 total):

| Department | Sample Documents |
|-----------|-----------------|
| **Product Management** | PRD, Feature Prioritization, Roadmap, User Story Backlog |
| **Engineering** | API Documentation, Deployment Runbook, Coding Standards |
| **Quality Assurance** | Test Plans, Bug Reports, QA Checklists |
| **Security** | Security Policies, Incident Reports, Risk Assessment |
| **Compliance** | Regulatory Checklists, Audit Reports, Policy Documents |
| **Sales** | Proposals, Case Studies, Pricing Sheets |
| **Marketing** | Campaign Plans, Content Calendars, Brand Guidelines |
| **Support** | Knowledge Base, Support Processes, FAQs |
| **HR** | Onboarding Guides, Policies, Team Handbooks |
| **Finance** | Budget Documents, Financial Reports, Expense Policies |

---

## 🔌 API Reference

### Core Endpoints

```bash
# Get all departments
GET /departments
→ {departments: [{code, name, slug}, ...]}

# Get documents for a department
GET /document-types?department=Product%20Management
→ {document_types: [{document_type, document_name}, ...]}

# Get Q&As for a document
GET /questions?document_type=Feature%20Prioritization%20Framework
→ {questions: [{question, answer_type, options, is_gap_question}, ...]}

# Get document schema
GET /required-section?department=...&document_name=...
→ {required_section: {sections: [...]}}

# Analyse schema gaps (cache-first)
POST /gap-questions
Body: {department, document_type, document_name, questions_and_answers}
→ {gap_questions: [...], source: "cache|generated"}

# Save gap questions for future users
POST /save-questions
Body: {department, document_type, document_name, gap_questions}
→ {saved_count: 5}

# Generate document from answers
POST /generate
Body: {department, document_type, document_name, questions_and_answers}
→ {generated_document, quality_scores, quality_issues, status}

# Get Notion page URLs (for history)
GET /get_all_urls
→ {pages: [{notion_url, title}, ...]}
```

**For full API documentation**, run the backend and visit `http://localhost:8000/docs`

---

## 🧪 Testing & Validation

### Manual Testing Checklist

- [ ] Backend starts without errors (`http://localhost:8000/docs` accessible)
- [ ] Streamlit UI loads (`http://localhost:8501`)
- [ ] Can select department & document
- [ ] Core Q&As load and render correctly
- [ ] Can fill in answers with different widget types (text, select, etc.)
- [ ] "🔍 Analyse Schema Gaps" returns gap questions within 15 seconds
- [ ] Can save gap questions without errors
- [ ] "⚡ Generate Document" produces markdown output in 30-60 seconds
- [ ] Generated document displays quality scores
- [ ] Can edit and re-submit documents

### Running Tests

```bash
# (If pytest tests are available)
pytest tests/
```

---

## 🛠️ Development Guide

### Adding a New Document Type

1. **Create Q&A file**: `document_and_questions/final_filtered_QAs/{department}/{document_name}_questions.json`
2. **Create schema file**: `document_and_questions/notion_documents/{department}/{document_name}.json` (with `sections` field)
3. **Upload to MongoDB**:
   ```bash
   cd automations
   python mongo_auto.py
   python required_sections_automation.py
   ```
4. **Refresh Streamlit** (cache will clear automatically)

### Modifying LLM Prompts

Edit `agent/prompts.py`:
- `SYSTEM_PROMPT_TEMPLATE` — for mixed documents
- `TABLE_PROMPT_TEMPLATE` — for table-only documents
- `QUALITY_REVIEW_PROMPT` — for validation

### Extending the Agent Workflow

Edit `agent/agent_graph.py`:
1. Define new node function
2. Add to graph builder
3. Update routing logic if needed
4. Update AgentState TypedDict if new fields are needed

---

## 📈 Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| GET /questions | <50ms | Cached in Streamlit (5 min TTL) |
| POST /gap-questions (cache hit) | <100ms | Direct MongoDB lookup |
| POST /gap-questions (fresh) | ~10-15s | Llama-3.3-70b LLM call |
| POST /generate | ~30-60s | Full 5-node workflow with Kimi-k2 |
| Streamlit page load | ~200ms | Data fetching + rendering |

---

## 🔐 Security & Privacy

### Best Practices

1. **Never commit `.env`** to version control
2. **Use IP whitelist** on MongoDB Atlas for production
3. **Rotate API keys** regularly
4. **Review gap questions** before saving (may contain sensitive info)
5. **Audit logging**: Track who generates which documents and when
6. **Data retention**: Consider TTL indexes for old session data

### CORS Configuration

Currently allows only:
- `http://localhost:8501`
- `http://127.0.0.1:8501`

For production, update `api/main.py`:
```python
allow_origins=["https://yourdomain.com"]
```

---

## 🐛 Troubleshooting

### Backend Won't Start

```bash
# Check port 8000 is not in use
lsof -i :8000

# Verify MongoDB connection
python -c "from pymongo import MongoClient; MongoClient('your_uri').list_database_names()"

# Check Groq API key
curl -H "Authorization: Bearer $GROQ_API_KEY" https://api.groq.com/...
```

### Streamlit Won't Load

```bash
# Clear Streamlit cache
streamlit cache clear

# Check FastAPI is running (should see docs at localhost:8000/docs)
curl http://localhost:8000/docs

# Check port 8501 is available
lsof -i :8501
```

### Gap Questions Not Saving

- Verify MongoDB connection string is correct
- Check that `document_type` field exists in request
- Monitor logs for `POST /save-questions` errors

### Documents Generating Slowly (>60s)

- Check Groq API response times (LLM may be busy)
- Verify network latency to MongoDB Atlas
- Try smaller document (fewer Q&As)

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [CODEBASE_ARCHITECTURE.md](CODEBASE_ARCHITECTURE.md) | 📘 Deep architectural dive (2100+ lines) |
| [progress.md](progress.md) | 📝 Development changelog |
| This README | 🚀 Quick start & overview |

---

## 🤝 Contributing

### Workflow

1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes (see Development Guide above)
3. Test locally: `pytest` (if available) + manual testing
4. Commit with clear message: `git commit -m "Add feature: description"`
5. Push: `git push origin feature/your-feature`
6. Open PR with description of changes

### Code Style

- Follow PEP 8
- Use type hints throughout
- Add docstrings for functions and classes
- Keep functions focused and testable

---

## 📜 License

[Specify your license here - e.g., MIT, Apache 2.0, proprietary]

---

## 📧 Support & Contact

For questions, issues, or feature requests:

- **Issues**: Open a GitHub issue
- **Email**: [your-email@domain.com]
- **Documentation**: See [CODEBASE_ARCHITECTURE.md](CODEBASE_ARCHITECTURE.md)

---

## 🚀 What's Next?

### Planned Enhancements

- [ ] Web dashboard for document history & analytics
- [ ] Multi-language support
- [ ] Custom document type builder (no-code schema creation)
- [ ] Batch document generation (generate 100+ at once)
- [ ] Integration with Google Docs & Microsoft Word
- [ ] Advanced quality metrics & audit trails
- [ ] Team collaboration features (comments, approvals)
- [ ] A/B testing for LLM prompts

### Known Limitations

- Currently supports single-user local deployment (Streamlit limitation)
- Table-only documents must have `type: "table"` in schema
- Gap questions limited to 100 per document type (configurable)

---

## 🙏 Acknowledgments

Built with:
- [LangChain](https://python.langchain.com/) & [LangGraph](https://langchain-ai.github.io/langgraph/)
- [Groq](https://groq.com/) for fast LLM inference
- [Streamlit](https://streamlit.io/) for rapid UI development
- [FastAPI](https://fastapi.tiangolo.com/) for modern Python APIs
- [MongoDB](https://www.mongodb.com/) for flexible data storage

---

<div align="center">

**Made with ❤️ for document-driven organizations**

[⬆ Back to top](#-docforgdhub)

</div>
