"""
rag/api/statecase_routes_rag.py

StateCase — FastAPI router for ticket management + stateful agent.

Mounted on the existing CiteRagLab FastAPI app (main_rag.py) at startup.

Routes
──────
    POST   /statecase/chat            — stateful agent (RAG + auto-ticketing)
    POST   /statecase/tickets         — create ticket manually
    GET    /statecase/tickets         — list all tickets (optional ?status= filter)
    GET    /statecase/tickets/{id}    — get one ticket by Notion page ID
    PATCH  /statecase/tickets/{id}    — update ticket status / owner / priority
    GET    /statecase/health          — liveness probe for this sub-service

All endpoints follow the same logging conventions as main_rag.py:
    logger.info("route — detail")
    logger.error("❌ error")
"""

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger("rag.api.statecase_routes_rag")

router = APIRouter(prefix="/statecase", tags=["StateCase"])


# ── Pydantic models ───────────────────────────────────────────────────────────

class StateCaseChatRequest(BaseModel):
    session_id:      str
    message:         str
    filters:         Optional[dict]  = None
    ticket_priority: Optional[str]   = Field(default="Medium", description="Low | Medium | High | Critical")
    ticket_owner:    Optional[str]   = Field(default="Unassigned")


class CreateTicketRequest(BaseModel):
    question:          str
    session_id:        str
    description:       Optional[str]       = ""
    priority:          Optional[str]       = "Medium"
    assigned_owner:    Optional[str]       = "Unassigned"
    attempted_sources: Optional[list[str]] = None
    user_info:         Optional[str]       = ""


class UpdateTicketRequest(BaseModel):
    status:         Optional[str] = None
    assigned_owner: Optional[str] = None
    priority:       Optional[str] = None
    description:    Optional[str] = None


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/chat")
async def statecase_chat(req: StateCaseChatRequest):
    """
    Stateful agent chat endpoint.

    Runs the LangGraph StateCase agent:
      - intent classification (RAG / TICKET_INTENT / CLARIFY)
      - if RAG: run full pipeline → auto-create ticket if out-of-scope
      - if TICKET_INTENT: create ticket immediately
      - persist memory + session history to Redis
    """
    logger.info(
        "📨 POST /statecase/chat — session_id=%s  message='%s…'  filters=%s",
        req.session_id, req.message[:60], req.filters or {},
    )

    from rag.pipeline.statecase_agent_rag import run_statecase_agent

    try:
        result = await run_statecase_agent(
            session_id=req.session_id,
            user_message=req.message,
            raw_filters=req.filters,
            ticket_priority=req.ticket_priority or "Medium",
            ticket_owner=req.ticket_owner or "Unassigned",
        )
    except Exception as err:
        logger.error("❌ POST /statecase/chat error: %s", err)
        raise HTTPException(status_code=500, detail=str(err))

    logger.info(
        "   ✅ POST /statecase/chat — trace=%s  intent=%s  ticket=%s  answer=%d chars",
        result.get("trace_id"),
        result.get("intent"),
        result["ticket_created"]["ticket_id"] if result.get("ticket_created") else "none",
        len(result.get("response", "")),
    )

    return {
        "session_id":     req.session_id,
        "answer":         result["response"],
        "citations":      result["citations"],
        "pipeline_meta":  result["pipeline_meta"],
        "ticket_created": result.get("ticket_created"),
        "trace_id":       result.get("trace_id"),
        "intent":         result.get("intent"),
    }


@router.post("/tickets")
async def create_ticket_endpoint(req: CreateTicketRequest):
    """
    Manually create a StateCase ticket.

    Used by:
      - Streamlit "Create Ticket" form in the Tickets tab
      - Any external workflow that needs to log an unanswered question
    """
    logger.info(
        "🎫 POST /statecase/tickets — session=%s  question='%s…'  priority=%s",
        req.session_id, req.question[:60], req.priority,
    )

    from rag.pipeline.statecase_notion_rag import create_ticket

    try:
        ticket = create_ticket(
            question=req.question,
            session_id=req.session_id,
            description=req.description or "",
            priority=req.priority or "Medium",
            assigned_owner=req.assigned_owner or "Unassigned",
            attempted_sources=req.attempted_sources,
            user_info=req.user_info or "",
        )
    except Exception as err:
        logger.error("❌ POST /statecase/tickets error: %s", err)
        raise HTTPException(status_code=500, detail=str(err))

    logger.info(
        "   ✅ Ticket created — ticket_id=%s  notion_page_id=%s",
        ticket["ticket_id"], ticket["notion_page_id"],
    )
    return {"status": "ok", "ticket": ticket}


@router.get("/tickets")
async def list_tickets_endpoint(
    status: Optional[str] = Query(None, description="Open | In Progress | Resolved | Closed"),
    limit:  int           = Query(50,   description="Max tickets to return"),
):
    """
    List all StateCase tickets.

    Optional ?status= filter: Open | In Progress | Resolved | Closed
    """
    logger.info(
        "📋 GET /statecase/tickets — status_filter=%s  limit=%d",
        status or "(all)", limit,
    )

    from rag.pipeline.statecase_notion_rag import list_tickets

    try:
        tickets = list_tickets(status_filter=status, limit=limit)
    except Exception as err:
        logger.error("❌ GET /statecase/tickets error: %s", err)
        raise HTTPException(status_code=500, detail=str(err))

    logger.info("   ✅ GET /statecase/tickets — %d tickets returned", len(tickets))
    return {"status": "ok", "count": len(tickets), "tickets": tickets}


@router.get("/tickets/{notion_page_id}")
async def get_ticket_endpoint(notion_page_id: str):
    """Fetch a single ticket by its Notion page ID."""
    logger.info("🎫 GET /statecase/tickets/%s", notion_page_id)

    from rag.pipeline.statecase_notion_rag import get_ticket

    try:
        ticket = get_ticket(notion_page_id)
    except Exception as err:
        logger.error("❌ GET /statecase/tickets/%s error: %s", notion_page_id, err)
        raise HTTPException(status_code=404, detail=str(err))

    return {"status": "ok", "ticket": ticket}


@router.patch("/tickets/{notion_page_id}")
async def update_ticket_endpoint(notion_page_id: str, req: UpdateTicketRequest):
    """
    Update an existing ticket.

    Only send the fields you want to change — all others are left untouched.
    """
    logger.info(
        "✏️  PATCH /statecase/tickets/%s — status=%s  owner=%s  priority=%s",
        notion_page_id, req.status, req.assigned_owner, req.priority,
    )

    from rag.pipeline.statecase_notion_rag import update_ticket

    try:
        ticket = update_ticket(
            notion_page_id=notion_page_id,
            status=req.status,
            assigned_owner=req.assigned_owner,
            priority=req.priority,
            description=req.description,
        )
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as err:
        logger.error("❌ PATCH /statecase/tickets/%s error: %s", notion_page_id, err)
        raise HTTPException(status_code=500, detail=str(err))

    logger.info(
        "   ✅ Ticket updated — ticket_id=%s  status=%s",
        ticket["ticket_id"], ticket["status"],
    )
    return {"status": "ok", "ticket": ticket}


@router.get("/health")
async def statecase_health():
    """StateCase sub-service liveness probe."""
    logger.info("💚 GET /statecase/health — OK")
    return {"status": "ok", "service": "StateCase"}