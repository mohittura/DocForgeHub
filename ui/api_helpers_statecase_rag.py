"""
ui/api_helpers_statecase_rag.py

HTTP helper functions for the StateCase ticketing system.

Mirrors api_helpers_rag.py exactly — one function per endpoint,
requests with timeout, logger.info() on entry/success, logger.error() on failure,
returns parsed dict or None on failure.
"""

import logging
import requests
from typing import Optional

RAG_API_URL = "http://127.0.0.1:8001"   # same CiteRagLab backend

logger = logging.getLogger("ui.api_helpers_statecase_rag")


def call_statecase_chat(
    session_id:      str,
    message:         str,
    filters:         dict | None = None,
    ticket_priority: str = "Medium",
    ticket_owner:    str = "Unassigned",
    base_url:        str = RAG_API_URL,
) -> dict | None:
    """
    POST /statecase/chat — stateful agent with auto-ticketing.

    Response keys: session_id, answer, citations, pipeline_meta,
                   ticket_created (None or ticket dict), trace_id, intent
    """
    logger.info(
        "Calling POST /statecase/chat — session_id=%s  message='%s…'",
        session_id, message[:60],
    )
    try:
        response = requests.post(
            f"{base_url}/statecase/chat",
            json={
                "session_id":      session_id,
                "message":         message,
                "filters":         filters or {},
                "ticket_priority": ticket_priority,
                "ticket_owner":    ticket_owner,
            },
            timeout=90,
        )
        response.raise_for_status()
        result = response.json()
        logger.info(
            "   -> /statecase/chat OK — intent=%s  ticket=%s  answer=%d chars",
            result.get("intent"),
            result["ticket_created"]["ticket_id"] if result.get("ticket_created") else "none",
            len(result.get("answer", "")),
        )
        return result
    except Exception as err:
        logger.error("   -> /statecase/chat FAILED: %s", err)
        return None


def call_create_ticket(
    question:          str,
    session_id:        str,
    description:       str = "",
    priority:          str = "Medium",
    assigned_owner:    str = "Unassigned",
    attempted_sources: list[str] | None = None,
    user_info:         str = "",
    base_url:          str = RAG_API_URL,
) -> dict | None:
    """
    POST /statecase/tickets — create a ticket manually.

    Returns dict with keys: status, ticket (or None on failure).
    """
    logger.info(
        "Calling POST /statecase/tickets — session=%s  question='%s…'  priority=%s",
        session_id, question[:60], priority,
    )
    try:
        response = requests.post(
            f"{base_url}/statecase/tickets",
            json={
                "question":          question,
                "session_id":        session_id,
                "description":       description,
                "priority":          priority,
                "assigned_owner":    assigned_owner,
                "attempted_sources": attempted_sources or [],
                "user_info":         user_info,
            },
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        logger.info(
            "   -> /statecase/tickets POST OK — ticket_id=%s",
            result.get("ticket", {}).get("ticket_id", "?"),
        )
        return result
    except Exception as err:
        logger.error("   -> /statecase/tickets POST FAILED: %s", err)
        return None


def call_list_tickets(
    status_filter: str | None = None,
    limit:         int        = 50,
    base_url:      str        = RAG_API_URL,
) -> dict | None:
    """
    GET /statecase/tickets — list tickets with optional status filter.

    Returns dict with keys: status, count, tickets (list).
    """
    logger.info(
        "Calling GET /statecase/tickets — status_filter=%s  limit=%d",
        status_filter or "(all)", limit,
    )
    try:
        params: dict = {"limit": limit}
        if status_filter:
            params["status"] = status_filter

        response = requests.get(
            f"{base_url}/statecase/tickets",
            params=params,
            timeout=30,
        )
        response.raise_for_status()
        result = response.json()
        logger.info(
            "   -> /statecase/tickets GET OK — %d tickets",
            result.get("count", 0),
        )
        return result
    except Exception as err:
        logger.error("   -> /statecase/tickets GET FAILED: %s", err)
        return None


def call_get_ticket(
    notion_page_id: str,
    base_url:       str = RAG_API_URL,
) -> dict | None:
    """GET /statecase/tickets/{id} — fetch one ticket."""
    logger.info("Calling GET /statecase/tickets/%s", notion_page_id)
    try:
        response = requests.get(
            f"{base_url}/statecase/tickets/{notion_page_id}",
            timeout=15,
        )
        response.raise_for_status()
        return response.json()
    except Exception as err:
        logger.error("   -> /statecase/tickets/%s GET FAILED: %s", notion_page_id, err)
        return None


def call_update_ticket(
    notion_page_id: str,
    status:         str | None = None,
    assigned_owner: str | None = None,
    priority:       str | None = None,
    description:    str | None = None,
    base_url:       str        = RAG_API_URL,
) -> dict | None:
    """
    PATCH /statecase/tickets/{id} — update one or more ticket fields.

    Returns updated ticket dict or None on failure.
    """
    logger.info(
        "Calling PATCH /statecase/tickets/%s — status=%s  owner=%s  priority=%s",
        notion_page_id, status, assigned_owner, priority,
    )
    payload = {}
    if status         is not None: payload["status"]         = status
    if assigned_owner is not None: payload["assigned_owner"] = assigned_owner
    if priority       is not None: payload["priority"]       = priority
    if description    is not None: payload["description"]    = description

    try:
        response = requests.patch(
            f"{base_url}/statecase/tickets/{notion_page_id}",
            json=payload,
            timeout=15,
        )
        response.raise_for_status()
        result = response.json()
        logger.info(
            "   -> PATCH OK — ticket_id=%s  new_status=%s",
            result.get("ticket", {}).get("ticket_id", "?"),
            result.get("ticket", {}).get("status", "?"),
        )
        return result
    except Exception as err:
        logger.error("   -> PATCH /statecase/tickets/%s FAILED: %s", notion_page_id, err)
        return None