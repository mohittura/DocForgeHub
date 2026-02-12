import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query
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
