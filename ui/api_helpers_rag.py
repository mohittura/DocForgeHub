"""
ui/api_helpers_rag.py

FastAPI endpoint helper functions for the CiteRagLab Streamlit UI.

Wraps HTTP calls to the CiteRagLab RAG backend (port 8001).
Follows the exact same style as the existing ui/api_helpers.py:
  - one function per endpoint
  - requests with timeout
  - logger.info() on entry, logger.info() on success, logger.error() on failure
  - returns the parsed dict or None on failure

No Streamlit dependency — purely HTTP wrappers reusable by any Python client.
"""

import logging
import requests

RAG_API_URL = "http://127.0.0.1:8001"   # CiteRagLab RAG backend

logger = logging.getLogger("ui.api_helpers_rag")


def call_chat(
    session_id: str,
    message: str,
    filters: dict | None = None,
    base_url: str = RAG_API_URL,
) -> dict | None:
    """
    POST /chat — run the full RAG pipeline for one user message.

    Returns the API response dict or None on failure.
    Response keys: session_id, answer, citations, mode, avg_score, rewritten
    """
    logger.info(
        "Calling POST /chat — session_id=%s  message='%s…'  filters=%s",
        session_id,
        message[:60],
        filters or {},
    )
    try:
        response = requests.post(
            f"{base_url}/chat",
            json={
                "session_id": session_id,
                "message":    message,
                "filters":    filters or {},
            },
            timeout=90,
        )
        response.raise_for_status()
        result = response.json()
        logger.info(
            "   -> /chat OK — mode=%s, avg_score=%s, answer=%d chars, citations=%d",
            result.get("mode"),
            result.get("avg_score", 0),
            len(result.get("answer", "")),
            len(result.get("citations", [])),
        )
        return result
    except Exception as error:
        logger.error("   -> /chat FAILED: %s", error)
        return None


def call_retrieval_debug(
    query: str,
    top_k: int = 5,
    industry: str = "",
    doc_type: str = "",
    version: str = "",
    base_url: str = RAG_API_URL,
) -> dict | None:
    """
    GET /retrieval/debug — retrieval inspector.

    Returns dict with keys: query, filters, count, chunks
    or None on failure.
    """
    logger.info(
        "Calling GET /retrieval/debug — query='%s…', top_k=%d, industry=%s, doc_type=%s",
        query[:60], top_k, industry or "(any)", doc_type or "(any)",
    )
    try:
        response = requests.get(
            f"{base_url}/retrieval/debug",
            params={
                "query":    query,
                "top_k":    top_k,
                "industry": industry,
                "doc_type": doc_type,
                "version":  version,
            },
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        logger.info(
            "   -> /retrieval/debug OK — %d chunks returned",
            result.get("count", 0),
        )
        return result
    except Exception as error:
        logger.error("   -> /retrieval/debug FAILED: %s", error)
        return None


def call_ingest_notion(
    page_id: str = "",
    title: str = "",
    industry: str = "General",
    doc_type: str = "Document",
    version: str = "1.0",
    base_url: str = RAG_API_URL,
) -> dict | None:
    """
    POST /ingestion/notion — trigger Notion ingestion.

    If page_id is supplied: ingest only that page.
    If page_id is empty:    ingest all pages under NOTION_ROOT_PAGE_ID.
    Returns the API response dict or None on failure.
    """
    if page_id.strip():
        logger.info(
            "Calling POST /ingestion/notion — single page: title=%r, page_id=%s",
            title, page_id,
        )
        request_body = {
            "page_id":  page_id,
            "title":    title,
            "industry": industry,
            "doc_type": doc_type,
            "version":  version,
        }
    else:
        logger.info("Calling POST /ingestion/notion — full ingest (all pages)")
        request_body = {}

    try:
        response = requests.post(
            f"{base_url}/ingestion/notion",
            json=request_body or None,
            timeout=600,   # full ingest can take several minutes
        )
        response.raise_for_status()
        result = response.json()
        logger.info(
            "   -> /ingestion/notion OK — pages=%s, chunks=%s",
            result.get("pages_processed", result.get("chunks_inserted", "?")),
            result.get("chunks_inserted", "?"),
        )
        return result
    except Exception as error:
        logger.error("   -> /ingestion/notion FAILED: %s", error)
        return None


def call_run_evaluation(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
    base_url: str = RAG_API_URL,
) -> dict | None:
    """
    POST /evaluation/run — run RAGAS evaluation metrics.

    Returns dict with keys: status, scores (or error) or None on failure.
    """
    logger.info(
        "Calling POST /evaluation/run — %d question(s)",
        len(questions),
    )
    try:
        response = requests.post(
            f"{base_url}/evaluation/run",
            json={
                "questions":     questions,
                "answers":       answers,
                "contexts":      contexts,
                "ground_truths": ground_truths,
            },
            timeout=300,
        )
        response.raise_for_status()
        result = response.json()
        logger.info(
            "   -> /evaluation/run OK — scores=%s",
            result.get("scores", {}),
        )
        return result
    except Exception as error:
        logger.error("   -> /evaluation/run FAILED: %s", error)
        return None


def call_delete_session(
    session_id: str,
    base_url: str = RAG_API_URL,
) -> bool:
    """
    DELETE /session — clear a session's chat history from Redis.

    Returns True on success, False on failure.
    """
    logger.info("Calling DELETE /session — session_id=%s", session_id)
    try:
        response = requests.delete(
            f"{base_url}/session",
            json={"session_id": session_id},
            timeout=10,
        )
        response.raise_for_status()
        logger.info("   -> /session DELETE OK — session_id=%s cleared", session_id)
        return True
    except Exception as error:
        logger.error("   -> /session DELETE FAILED: %s", error)
        return False