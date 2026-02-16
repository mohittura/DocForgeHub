import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.db import get_db, close_client
from notion_client import Client
from typing import List, Dict, Any


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
    results = await db["document_qas"].aggregate(pipeline).to_list(length=100) #will store the departments from the document_qas collection in the mongodb which will be aggregated by the pipeline defined as of group adn sort
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
    """Return questions for the given document type, sorted by category and question order."""
    db = get_db()
    cursor = db["document_qas"].find(
        {"document_type": document_type},
        {"_id": 0, "_runtime_metadata": 0, "schema_id": 0},
    ).sort([("category_order", 1), ("question_order", 1)])

    questions = await cursor.to_list(length=500)
    return {"questions": questions}


notion_api_key=os.environ.get("NOTION_API_KEY")
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
    root_page_id="30589db15e5b80779819dc0d8c532954"
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


