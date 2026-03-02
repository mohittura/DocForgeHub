import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.db import get_db, close_client
from notion_client import Client
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime
from agent.agent_graph import run_agent, analyze_gaps_only, generate_single_section


@asynccontextmanager #defining the db lifespan in the project
async def lifespan(app: FastAPI):
    yield
    await close_client()


app = FastAPI(title="DocForge Hub API", lifespan=lifespan) # app startup

app.add_middleware( # added cors middleware to allow the local streamlit url
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/departments")
async def get_departments():
    """Return a sorted list of unique department names."""
    db = get_db()
    pipeline = [
        {"$group": {"_id": "$department"}}, # $group will store the department as the primary id
        {"$sort": {"_id.code": 1}}, # $sort will sort the departments based on the id (based on the department code default is 1)
    ]
    results = await db["document_qas"].aggregate(pipeline).to_list(length=100) #will store the departments from the document_qas collection in the mongodb which will be aggregated by the pipeline defined as of group and sort
    departments = []
    for result_item in results: #will store the departments by looping on the results by code, name and slug defined in the mongo client
        dept = result_item["_id"]
        if dept and isinstance(dept, dict):
            departments.append({
                "code": dept.get("code", ""),
                "name": dept.get("name", ""),
                "slug": dept.get("slug", ""),
            })
    # Sort by code
    departments.sort(key=lambda d: d.get("code", "0"))
    return {"departments": departments}


@app.get("/document-types")
async def get_document_types(department: str = Query(..., description="Department name")):
    """Return document types for the given department."""
    db = get_db()
    pipeline = [
        {"$match": {"department.name": department}},
        {"$group": {"_id": {"document_type": "$document_type", "document_name": "$document_name"}}},
        {"$sort": {"_id.document_type": 1}},
    ]
    results = await db["document_qas"].aggregate(pipeline).to_list(length=100)
    doc_types = []
    for result_item in results:
        doc_types.append({
            "document_type": result_item["_id"]["document_type"],
            "document_name": result_item["_id"]["document_name"],
        })
    doc_types.sort(key=lambda document_item: document_item["document_type"]) #sorted by document type
    return {"document_types": doc_types}


@app.get("/questions")
async def get_questions(document_type: str = Query(..., description="Document type")):
    """
    Return questions for the given document type, sorted by category and question order.
    This now includes any AI-generated gap questions that were previously saved to MongoDB.
    Gap questions are tagged with is_gap_question=True so the UI can distinguish them.
    """
    db = get_db()
    cursor = db["document_qas"].find(
        {"document_type": document_type},
        {"_id": 0, "_runtime_metadata": 0, "schema_id": 0},
    ).sort([("category_order", 1), ("question_order", 1)])

    questions = await cursor.to_list(length=500)
    return {"questions": questions}


from api.helpers import (
    retrieve_all_child_pages_recursive,
)


@app.get("/get_all_urls")
def get_all_urls_endpoint():
    root_page_id = "30589db15e5b80779819dc0d8c532954"
    """FastAPI endpoint to retrieve all URLs under a root page ID."""
    if not root_page_id:
        raise HTTPException(status_code=400, detail="Root page ID must be provided")

    # Ensure the root ID is in the correct format (remove dashes if necessary)
    formatted_root_page_id = root_page_id.replace("-", "")

    pages = retrieve_all_child_pages_recursive(formatted_root_page_id)
    return {"root_page_id": root_page_id, "page_count": len(pages), "pages": pages}



@app.get("/required-section")
async def get_required_section(
    department: str = Query(..., description="Department name"),
    document_name: str = Query(..., description="Document name"),
):
    """Return the required section schema for the given department and document name."""
    db = get_db()
    schema_document = await db["required_section"].find_one(
        {"department": department, "document_name": document_name},
        {"_id": 0},
    )
    if not schema_document:
        raise HTTPException(
            status_code=404,
            detail=f"No schema found for department='{department}', document_name='{document_name}'",
        )
    return {"required_section": schema_document}


# ═══════════════════════════════════════════════════════════════
#  NEW: Gap Questions endpoints
# ═══════════════════════════════════════════════════════════════

class GapQuestionsRequest(BaseModel):
    """Request body for POST /gap-questions."""
    department: str
    document_type: str
    document_name: str
    questions_and_answers: List[Dict[str, Any]]
    required_section: Optional[Dict[str, Any]] = None


@app.post("/gap-questions")
async def get_gap_questions(request: GapQuestionsRequest):
    """
    Analyse schema coverage and return AI-generated questions for uncovered sections.

    Flow:
      1. Check MongoDB: have gap questions for this document_type already been
         generated and saved? If yes, return them immediately (no LLM call needed).
      2. If not cached: run the lightweight question-generation LLM.
      3. Return the gap questions to the UI.

    The caller (Streamlit) will display these questions and can then call
    POST /save-questions to persist answered gap questions into MongoDB.

    Response:
        {
            "gap_questions": [...],
            "source": "cache" | "generated",
            "count": <int>
        }
    """
    db = get_db()

    # ── Step 1: Check cache — any already-saved gap questions for this doc type? ──
    existing_gaps = await db["document_qas"].find_one(
        {"document_type": request.document_type, "is_gap_question": True},
        {"_id": 0},
    )
    if existing_gaps:
        # Fetch ALL saved gap questions for this document_type
        cursor = db["document_qas"].find(
            {"document_type": request.document_type, "is_gap_question": True},
            {"_id": 0, "_runtime_metadata": 0, "schema_id": 0},
        ).sort([("question_order", 1)])
        cached_questions = await cursor.to_list(length=100)

        return {
            "gap_questions": cached_questions,
            "source": "cache",
            "count": len(cached_questions),
        }

    # ── Step 2: Fetch schema if not in request ──────────────────────────────────
    required_section = request.required_section
    if not required_section:
        schema_doc = await db["required_section"].find_one(
            {
                "department": request.department,
                "document_name": request.document_name,
            },
            {"_id": 0},
        )
        if schema_doc:
            required_section = schema_doc
        else:
            required_section = {"sections": []}

    # ── Step 3: Run lightweight gap analysis ───────────────────────────────────
    try:
        gap_questions = await analyze_gaps_only(
            department=request.department,
            document_type=request.document_type,
            questions_and_answers=request.questions_and_answers,
            required_section=required_section,
        )

        return {
            "gap_questions": gap_questions,
            "source": "generated",
            "count": len(gap_questions),
        }
    except Exception as error_message:
        print(f"Error in /gap-questions: {error_message}")
        raise HTTPException(status_code=500, detail=str(error_message))


class SaveQuestionsRequest(BaseModel):
    """
    Request body for POST /save-questions.

    Saves answered gap questions into MongoDB's document_qas collection
    so future loads of this document_type include them automatically.
    """
    department: Dict[str, Any]       # full department object {code, name, slug}
    document_type: str
    document_name: str
    gap_questions: List[Dict[str, Any]]   # [{question, answer, category, ...}]


@app.post("/save-questions")
async def save_gap_questions(request: SaveQuestionsRequest):
    """
    Persist answered gap questions into MongoDB document_qas.

    Each gap question is saved as a normal Q&A document with:
      - is_gap_question: True   ← marks it as AI-generated
      - question_order: high value (1000+) so it sorts after existing questions
      - answered_at: timestamp

    Idempotent: if a gap question with the same text already exists
    for this document_type, it is updated (not duplicated).

    Response:
        {"saved": <count>, "updated": <count>}
    """
    db = get_db()

    # Get the max existing question_order to avoid collisions
    pipeline = [
        {"$match": {"document_type": request.document_type}},
        {"$group": {"_id": None, "max_order": {"$max": "$question_order"}}},
    ]
    agg_result = await db["document_qas"].aggregate(pipeline).to_list(length=1)
    base_order = (agg_result[0]["max_order"] if agg_result else 0) or 0

    saved_count = 0
    updated_count = 0

    for gap_question_item in request.gap_questions:
        question_text = gap_question_item.get("question", "").strip()
        if not question_text:
            continue

        document_to_save = {
            "department": request.department,
            "document_type": request.document_type,
            "document_name": request.document_name,
            "question": question_text,
            "answer": gap_question_item.get("answer", ""),
            "category": gap_question_item.get("category", "Additional Information"),
            "category_order": 999,          # always sorts after core categories
            "question_order": base_order + 1000 + request.gap_questions.index(gap_question_item),
            "answer_type": gap_question_item.get("answer_type", "text"),
            "options": gap_question_item.get("options", []),
            "is_gap_question": True,
            "section_covered": gap_question_item.get("section_covered", ""),
            "answered_at": datetime.utcnow().isoformat(),
        }

        # Upsert: match on document_type + question text
        result = await db["document_qas"].update_one(
            {
                "document_type": request.document_type,
                "question": question_text,
                "is_gap_question": True,
            },
            {"$set": document_to_save},
            upsert=True,
        )

        if result.upserted_id:
            saved_count += 1
        else:
            updated_count += 1

    return {
        "saved": saved_count,
        "updated": updated_count,
        "total": saved_count + updated_count,
    }


class GenerateDocumentRequest(BaseModel):
    """
    The body of a POST /generate request.

    Fields:
        department:              e.g. "Product Management"
        document_type:           e.g. "Feature Prioritization Framework"
        document_name:           e.g. "Feature prioritization framework"
        questions_and_answers:   list of {question, answer, category, ...}
        required_section:        the document schema (sections) — optional,
                                 will be fetched from MongoDB if not provided
    """
    department: str
    document_type: str
    document_name: str
    questions_and_answers: List[Dict[str, Any]]
    required_section: Dict[str, Any] | None = None


@app.post("/generate")
async def generate_document(request: GenerateDocumentRequest):
    """
    Run the LangGraph agent to generate a professional Markdown document.

    1. If required_section is not provided in the body, fetch it from MongoDB
    2. Call the agent with (department, document_type, Q&A, required_section)
    3. Return the generated Markdown + quality status
    """

    # ── Fetch schema if not included in the request ──────────────
    required_section = request.required_section

    if not required_section:
        db = get_db()
        schema_doc = await db["required_section"].find_one(
            {
                "department": request.department,
                "document_name": request.document_name,
            },
            {"_id": 0},
        )
        if schema_doc:
            required_section = schema_doc
        else:
            required_section = {"sections": []}

    # ── Run the agent ────────────────────────────────────────────
    try:
        agent_result = await run_agent(
            department=request.department,
            document_type=request.document_type,
            questions_and_answers=request.questions_and_answers,
            required_section=required_section,
        )
    except Exception as agent_error:
        raise HTTPException(status_code=500, detail=f"Agent error: {agent_error}")

    return {
        "generated_document": agent_result["generated_document"],
        "gap_questions": agent_result.get("gap_questions", []),    # NEW
        "status": agent_result["status"],
        "quality_issues": agent_result.get("quality_issues", []),
        "quality_scores": agent_result.get("quality_scores", {}),
        "quality_suggestions": agent_result.get("quality_suggestions", []),
        "retry_count": agent_result.get("retry_count", 0),
    }


# ═══════════════════════════════════════════════════════════════
#  Progressive: Generate a single section
# ═══════════════════════════════════════════════════════════════

class GenerateSectionRequest(BaseModel):
    department: str
    document_type: str
    section: dict
    questions_and_answers: list
    doc_memory: str = ""


@app.post("/generate-section")
async def generate_section_endpoint(request: GenerateSectionRequest):
    """Generate ONE section with memory of previous sections."""
    try:
        section_text = await generate_single_section(
            department=request.department,
            document_type=request.document_type,
            section=request.section,
            questions_and_answers=request.questions_and_answers,
            doc_memory=request.doc_memory,
        )
    except Exception as generation_err:
        raise HTTPException(status_code=500, detail=f"Section generation error: {generation_err}")

    return {"section_text": section_text}

# ═══════════════════════════════════════════════════════════════
#  Notion Publish endpoint
# ═══════════════════════════════════════════════════════════════

from api.notion_publisher import publish_to_notion_database
from api.helpers import notion_client as _notion_client  # reuse the shared client

# Notion database ID where published documents are stored as rows.
# The database must have columns: Title, Type, Industry, Version,
# tags, Created by, Created time  (exact names).
_NOTION_DATABASE_ID = os.environ.get("NOTION_DATABASE_ID", "")


class PublishToNotionRequest(BaseModel):
    """Request body for POST /publish-to-notion."""
    markdown_text: str
    document_title: str
    document_type: str = ""
    industry: str = "General"
    version: str = "1.0"
    tags: List[str] = []
    created_by: str = "DocForgeHub"


@app.post("/publish-to-notion")
async def publish_to_notion(request: PublishToNotionRequest):
    """
    Publish a Markdown document to the configured Notion database as a new row.

    Pipeline
    ────────
    1. Validate that markdown_text is non-empty and NOTION_DATABASE_ID is set.
    2. Create a new database page with structured properties:
         Title, Type, Industry, Version, tags, Created by, Created time.
    3. Parse Markdown → typed Notion blocks (headings, bullets, numbered lists,
       tables, code blocks, quotes, paragraphs with inline formatting).
    4. Push blocks to the page in chunks of ≤95 with rate-limit back-off.

    Response (200):
        {
            "status":        "ok",
            "page_id":       "<notion-page-id>",
            "page_url":      "https://notion.so/<id>",
            "blocks_pushed": <int>
        }
    """
    if not request.markdown_text or not request.markdown_text.strip():
        raise HTTPException(status_code=400, detail="markdown_text must not be empty.")

    if not _NOTION_DATABASE_ID:
        raise HTTPException(
            status_code=503,
            detail="NOTION_DATABASE_ID env var is not set. Add it to your .env file.",
        )

    try:
        result = await asyncio.to_thread(
            publish_to_notion_database,
            markdown_text=request.markdown_text,
            document_title=request.document_title,
            document_type=request.document_type,
            industry=request.industry,
            version=request.version,
            tags=request.tags,
            created_by=request.created_by,
            database_id=_NOTION_DATABASE_ID,
            notion_client_instance=_notion_client,
        )
    except ValueError as value_err:
        raise HTTPException(status_code=400, detail=str(value_err))
    except Exception as publish_err:
        raise HTTPException(status_code=500, detail=f"Notion publish failed: {publish_err}")

    return {
        "status": "ok",
        "page_id": result["page_id"],
        "page_url": result["page_url"],
        "blocks_pushed": result["blocks_pushed"],
    }