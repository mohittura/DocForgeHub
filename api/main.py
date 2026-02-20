import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.db import get_db, close_client
from notion_client import Client
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime
from agent.agent_graph import run_agent, analyze_gaps_only


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
    for r in results: #will store the departments by looping on the results by code, name and slug defined in the mongo client
        dept = r["_id"]
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
    for r in results:
        doc_types.append({
            "document_type": r["_id"]["document_type"],
            "document_name": r["_id"]["document_name"],
        })
    doc_types.sort(key=lambda d: d["document_type"]) #sorted by document type
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


notion_api_key = os.environ.get("NOTION_API_KEY")
if not notion_api_key:
    raise ValueError("notion api key not defined")

notion = Client(auth=notion_api_key)

def get_page_url_from_id(page_id: str) -> str:
    """Constructs the Notion web URL from a page ID."""
    # Notion URLs are typically in the format: https://notion.so
    # The API returns IDs with dashes, so they need to be removed for the URL.
    simple_page_id = page_id.replace("-", "")
    return f"https://notion.so/{simple_page_id}"

def retrieve_all_child_pages_recursive(block_id: str, all_pages: List[Dict] = None) -> List[Dict]:
    """Recursively retrieves all child pages under a given block ID."""
    if all_pages is None:
        all_pages = []

    # The Retrieve block children endpoint is paginated, so we must handle pagination
    has_more = True
    next_cursor = None

    while has_more:
        try:
            response = notion.blocks.children.list(
                block_id=block_id,
                start_cursor=next_cursor,
                page_size=100
            )
            for block in response['results']:
                if block['type'] == 'child_page':
                    page_id = block['id']
                    page_title = block['child_page']['title']
                    page_url = get_page_url_from_id(page_id)
                    all_pages.append({
                        "id": page_id,
                        "title": page_title,
                        "url": page_url
                    })
                    # Recursively call for children of this child page
                    retrieve_all_child_pages_recursive(page_id, all_pages)

            next_cursor = response.get('next_cursor')
            has_more = response.get('has_more')

        except Exception as e:
            print(f"Error retrieving children for block {block_id}: {e}")
            break

    return all_pages

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

    for idx, gq in enumerate(request.gap_questions):
        question_text = gq.get("question", "").strip()
        if not question_text:
            continue

        doc = {
            "department": request.department,
            "document_type": request.document_type,
            "document_name": request.document_name,
            "question": question_text,
            "answer": gq.get("answer", ""),
            "category": gq.get("category", "Additional Information"),
            "category_order": 999,          # always sorts after core categories
            "question_order": base_order + 1000 + idx,
            "answer_type": gq.get("answer_type", "text"),
            "options": gq.get("options", []),
            "is_gap_question": True,
            "section_covered": gq.get("section_covered", ""),
            "answered_at": datetime.utcnow().isoformat(),
        }

        # Upsert: match on document_type + question text
        result = await db["document_qas"].update_one(
            {
                "document_type": request.document_type,
                "question": question_text,
                "is_gap_question": True,
            },
            {"$set": doc},
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