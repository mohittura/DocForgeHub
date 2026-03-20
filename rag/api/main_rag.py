"""
rag/api/main_rag.py

FastAPI application for the CiteRagLab RAG backend.
Runs on port 8001 to avoid clashing with DocForgeHub (port 8000).

Start with:
    uvicorn rag.api.main_rag:app --host 0.0.0.0 --port 8001 --reload

Routes
──────
    POST   /chat               — full RAG pipeline
    DELETE /session            — clear session history from Redis
    GET    /retrieval/debug    — retrieval inspector (chunks + scores)
    POST   /ingestion/notion   — trigger Notion ingestion
    POST   /evaluation/run     — run RAGAS evaluation metrics
    GET    /health             — liveness probe
"""

import sys
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ── Ensure the project root (DOCFORGEHUB/) is importable ────────────────────
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ── Logging (matches DocForgeHub format exactly) ─────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rag.api.main_rag")

# ── Internal imports (after sys.path is set) ─────────────────────────────────
from rag.pipeline.pipeline_rag        import run_rag_pipeline
from rag.pipeline.redis_cache_rag     import (
    get_session_history,
    set_session_history,
    delete_session,
    get_retrieval_cache,
    set_retrieval_cache,
    close_rag_redis,
)
from rag.retrieval.retriever_rag      import retrieve
from rag.retrieval.filters_rag        import build_filters
from rag.ingestion.ingestion_pipeline_rag import ingest_all_pages, ingest_page
from rag.evaluation.ragas_runner_rag  import run_ragas_evaluation


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 CiteRagLab API starting up on port 8001")
    yield
    logger.info("🛑 CiteRagLab API shutting down — closing Redis connection")
    await close_rag_redis()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CiteRagLab API",
    version="1.0.0",
    description="RAG backend for CiteRagLab — Adaptive RAG with Corrective retrieval and RAGAS evaluation",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501", "http://127.0.0.1:8501",   # Streamlit
        "http://localhost:8502", "http://127.0.0.1:8502",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic request models ───────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message:    str
    filters:    Optional[dict] = None


class DeleteSessionRequest(BaseModel):
    session_id: str


class IngestPageRequest(BaseModel):
    page_id:  str
    title:    str
    industry: Optional[str] = "General"
    doc_type: Optional[str] = "Document"
    version:  Optional[str] = "1.0"


class EvalRequest(BaseModel):
    questions:     list[str]
    answers:       list[str]
    contexts:      list[list[str]]
    ground_truths: list[str]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Full RAG pipeline for one user message.

    Loads session history from Redis → runs pipeline → saves updated history.
    Retrieved chunks are cached in Redis for TTL_RETRIEVAL seconds.
    """
    logger.info(
        "📨 POST /chat — session_id=%s  message='%s…'  filters=%s",
        req.session_id,
        req.message[:60],
        req.filters or {},
    )

    # Load prior conversation turns
    session_history = await get_session_history(req.session_id)
    logger.info(
        "   📜 Session history loaded — %d prior turns",
        len(session_history),
    )

    # Run the full RAG pipeline
    try:
        result = run_rag_pipeline(
            query=req.message,
            session_history=session_history,
            raw_filters=req.filters,
        )
    except Exception as err:
        logger.error("   ❌ Pipeline error: %s", err)
        raise HTTPException(status_code=500, detail=str(err))

    # Cache retrieved chunks
    await set_retrieval_cache(req.message, req.filters, result["chunks"])

    # Persist updated session history
    session_history.append({"role": "user",      "content": req.message})
    session_history.append({"role": "assistant",  "content": result["answer"]})
    await set_session_history(req.session_id, session_history)

    logger.info(
        "   ✅ POST /chat complete — mode=%s, chunks=%d, avg_score=%.4f, answer=%d chars",
        result["mode"],
        len(result["chunks"]),
        result["avg_score"],
        len(result["answer"]),
    )

    return {
        "session_id": req.session_id,
        "answer":     result["answer"],
        "citations":  result["citations"],
        "mode":       result["mode"],
        "avg_score":  result["avg_score"],
        "rewritten":  result["rewritten"],
    }


@app.delete("/session")
async def delete_session_route(req: DeleteSessionRequest):
    """Clear a session's chat history from Redis."""
    logger.info("🗑️  DELETE /session — session_id=%s", req.session_id)
    await delete_session(req.session_id)
    logger.info("   ✅ Session %s deleted", req.session_id)
    return {"status": "deleted", "session_id": req.session_id}


@app.get("/retrieval/debug")
async def retrieval_debug(
    query:    str  = Query(..., description="Search query"),
    top_k:    int  = Query(5,   description="Number of chunks to retrieve"),
    industry: str  = Query("",  description="Optional industry filter"),
    doc_type: str  = Query("",  description="Optional document type filter"),
    version:  str  = Query("",  description="Optional version filter"),
):
    """
    Retrieval inspector — returns raw chunks with similarity scores and metadata.
    Use this in the Streamlit Inspector tab to validate retrieval quality.
    """
    logger.info(
        "🔍 GET /retrieval/debug — query='%s…', top_k=%d, industry=%s, doc_type=%s, version=%s",
        query[:60], top_k, industry or "(any)", doc_type or "(any)", version or "(any)",
    )

    filters = build_filters({
        "industry": industry,
        "doc_type": doc_type,
        "version":  version,
    })

    try:
        chunks = retrieve(query=query, top_k=top_k, filters=filters or None)
    except Exception as err:
        logger.error("   ❌ Retrieval failed: %s", err)
        raise HTTPException(status_code=500, detail=str(err))

    logger.info(
        "   ✅ GET /retrieval/debug — %d chunks returned",
        len(chunks),
    )
    return {
        "query":   query,
        "filters": filters,
        "count":   len(chunks),
        "chunks":  chunks,
    }


@app.post("/ingestion/notion")
async def ingest_notion(req: Optional[IngestPageRequest] = None):
    """
    Trigger Notion ingestion.
    - With a body: ingest only the specified page.
    - Without a body (or empty body): ingest ALL pages under NOTION_ROOT_PAGE_ID.
    """
    if req and req.page_id.strip():
        logger.info(
            "📥 POST /ingestion/notion — single page: title='%s', page_id=%s",
            req.title, req.page_id,
        )
        try:
            n = ingest_page({
                "page_id":  req.page_id,
                "title":    req.title,
                "industry": req.industry,
                "doc_type": req.doc_type,
                "version":  req.version,
            })
            logger.info(
                "   ✅ Single-page ingest complete — %d chunks inserted",
                n,
            )
            return {"status": "ok", "chunks_inserted": n}
        except Exception as err:
            logger.error("   ❌ Single-page ingest failed: %s", err)
            raise HTTPException(status_code=500, detail=str(err))
    else:
        logger.info("📥 POST /ingestion/notion — full ingest (all pages)")
        try:
            summary = ingest_all_pages()
            logger.info(
                "   ✅ Full ingest complete — pages=%d, chunks=%d, errors=%d",
                summary["pages_processed"],
                summary["chunks_inserted"],
                len(summary.get("errors", [])),
            )
            return {"status": "ok", **summary}
        except Exception as err:
            logger.error("   ❌ Full ingest failed: %s", err)
            raise HTTPException(status_code=500, detail=str(err))


@app.post("/evaluation/run")
async def run_evaluation(req: EvalRequest):
    """Run RAGAS evaluation on a supplied dataset and return metric scores."""
    logger.info(
        "📊 POST /evaluation/run — %d question(s) to evaluate",
        len(req.questions),
    )
    scores = run_ragas_evaluation(
        questions=req.questions,
        answers=req.answers,
        contexts=req.contexts,
        ground_truths=req.ground_truths,
    )
    if "error" in scores:
        logger.error("   ❌ Evaluation failed: %s", scores["error"])
        raise HTTPException(status_code=500, detail=scores["error"])

    logger.info(
        "   ✅ POST /evaluation/run complete — scores: %s",
        scores,
    )
    return {"status": "ok", "scores": scores}


@app.get("/health")
async def health():
    """Simple liveness probe."""
    logger.info("💚 GET /health — OK")
    return {"status": "ok", "service": "CiteRagLab", "port": 8001}