"""
FastAPI endpoint helper functions for the DocForge Hub Streamlit UI.

These functions wrap HTTP calls to the FastAPI backend. They handle
request construction, error handling, and logging. They do NOT depend
on Streamlit at all and can be reused by any Python client.
"""

import logging
import requests

FASTAPI_URL = "http://127.0.0.1:8000"

logger = logging.getLogger("ui.api_helpers")


# ─────────────────────────────────────────────────────────────
#  Cached data fetchers (cache is applied in the Streamlit layer)
# ─────────────────────────────────────────────────────────────

def fetch_departments(base_url: str = FASTAPI_URL) -> list:
    """GET /departments — return the list of department dicts."""
    try:
        response = requests.get(f"{base_url}/departments", timeout=10)
        response.raise_for_status()
        departments = response.json().get("departments", [])
        logger.info(" -> received %d departments", len(departments))
        return departments
    except Exception as error:
        logger.error("Failed to fetch departments: %s", error)
        return []


def fetch_document_types(department_name: str, base_url: str = FASTAPI_URL) -> list:
    """GET /document-types — return document type dicts for a department."""
    try:
        response = requests.get(
            f"{base_url}/document-types",
            params={"department": department_name},
            timeout=10,
        )
        response.raise_for_status()
        document_types = response.json().get("document_types", [])
        logger.info(" -> received %d document types", len(document_types))
        return document_types
    except Exception as error:
        logger.error("Failed to fetch document types: %s", error)
        return []


def fetch_questions(document_type: str, base_url: str = FASTAPI_URL) -> list:
    """GET /questions — return question dicts for a document type."""
    try:
        response = requests.get(
            f"{base_url}/questions",
            params={"document_type": document_type},
            timeout=10,
        )
        response.raise_for_status()
        questions = response.json().get("questions", [])
        logger.info(" -> received %d questions", len(questions))
        return questions
    except Exception as error:
        logger.error("Failed to fetch questions: %s", error)
        return []


def fetch_notion_page_urls(base_url: str = FASTAPI_URL) -> list:
    """GET /get_all_urls — return all published Notion page dicts."""
    try:
        response = requests.get(f"{base_url}/get_all_urls", timeout=30)
        response.raise_for_status()
        pages = response.json().get("pages", [])
        logger.info(" -> received %d pages", len(pages))
        return pages
    except Exception as error:
        logger.error("Failed to fetch published pages: %s", error)
        return []


# ─────────────────────────────────────────────────────────────
#  POST endpoint wrappers
# ─────────────────────────────────────────────────────────────

def call_gap_questions_endpoint(
    department: str,
    document_type: str,
    document_name: str,
    questions_and_answers: list,
    base_url: str = FASTAPI_URL,
) -> dict | None:
    """POST /gap-questions — analyse schema coverage and get gap questions."""
    logger.info(
        "Calling POST /gap-questions — document_type=%s, answers=%d",
        document_type,
        len(questions_and_answers),
    )
    try:
        response = requests.post(
            f"{base_url}/gap-questions",
            json={
                "department": department,
                "document_type": document_type,
                "document_name": document_name,
                "questions_and_answers": questions_and_answers,
            },
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
        logger.info(
            "   -> gap analysis done — source=%s, count=%d",
            result.get("source"),
            result.get("count", 0),
        )
        return result
    except Exception as error:
        logger.error("Gap question fetch failed: %s", error)
        return None


def call_save_questions_endpoint(
    department_obj: dict,
    document_type: str,
    document_name: str,
    gap_questions: list,
    base_url: str = FASTAPI_URL,
) -> dict | None:
    """POST /save-questions — persist answered gap questions to MongoDB."""
    logger.info(
        "Calling POST /save-questions — document_type=%s, questions=%d",
        document_type,
        len(gap_questions),
    )
    try:
        response = requests.post(
            f"{base_url}/save-questions",
            json={
                "department": department_obj,
                "document_type": document_type,
                "document_name": document_name,
                "gap_questions": gap_questions,
            },
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        logger.info("   -> saved=%d, updated=%d", result.get("saved", 0), result.get("updated", 0))
        return result
    except Exception as error:
        logger.error("Save questions failed: %s", error)
        return None


def call_generate_endpoint(
    department: str,
    document_type: str,
    document_name: str,
    questions_and_answers: list,
    base_url: str = FASTAPI_URL,
) -> dict | None:
    """POST /generate — send answers to the agent and get a generated document back."""
    logger.info(
        "Calling POST /generate — department=%s, document_type=%s, answers=%d",
        department,
        document_type,
        len(questions_and_answers),
    )
    try:
        response = requests.post(
            f"{base_url}/generate",
            json={
                "department": department,
                "document_type": document_type,
                "document_name": document_name,
                "questions_and_answers": questions_and_answers,
            },
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()
        logger.info(
            "   -> generation complete — status=%s, length=%d chars",
            result.get("status"),
            len(result.get("generated_document", "")),
        )
        return result
    except Exception as error:
        logger.error("Generation failed: %s", error)
        return None


def call_generate_section(
    department: str,
    document_type: str,
    section: dict,
    questions_and_answers: list,
    doc_memory: str = "",
    base_url: str = FASTAPI_URL,
) -> dict | None:
    """POST /generate-section — generate one section with memory of previous sections."""
    try:
        response = requests.post(
            f"{base_url}/generate-section",
            json={
                "department": department,
                "document_type": document_type,
                "section": section,
                "questions_and_answers": questions_and_answers,
                "doc_memory": doc_memory,
            },
            timeout=90,
        )
        response.raise_for_status()
        return response.json()
    except Exception as error:
        logger.error("Section generation failed: %s", error)
        return None

def call_publish_to_notion_endpoint(
    markdown_text: str,
    document_title: str,
    parent_page_id: str | None = None,
    base_url: str = FASTAPI_URL,
) -> dict | None:
    """POST /publish-to-notion — convert Markdown and create a Notion page."""
    logger.info(
        "Calling POST /publish-to-notion — title=%r, length=%d chars",
        document_title,
        len(markdown_text),
    )
    try:
        response = requests.post(
            f"{base_url}/publish-to-notion",
            json={
                "markdown_text": markdown_text,
                "document_title": document_title,
                "parent_page_id": parent_page_id,
            },
            timeout=120,   # large docs with many chunks can take a while
        )
        response.raise_for_status()
        result = response.json()
        logger.info(
            "   -> published — page_id=%s, blocks=%d",
            result.get("page_id"),
            result.get("blocks_pushed", 0),
        )
        return result
    except Exception as error:
        logger.error("Notion publish failed: %s", error)
        return None