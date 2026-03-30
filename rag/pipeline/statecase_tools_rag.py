"""
rag/pipeline/statecase_tools_rag.py

LangChain @tool definitions for the StateCase agent.

Why tools instead of direct function calls?
────────────────────────────────────────────
The entire project currently calls functions directly from node bodies
(run_rag_pipeline(), create_ticket(), retrieve(), etc.).  This works for
a fixed graph but has three concrete problems:

  1. The LLM cannot decide WHICH action to take — the graph topology is
     hard-coded.  Tools let the LLM choose the right action based on
     context, which is the correct design for an agentic system.

  2. Direct calls produce no structured schema — LangChain tools expose a
     Pydantic schema the LLM can inspect, reason about, and call with
     validated arguments. This enables tool-calling via function calling
     in the Azure OpenAI API (cheaper and more reliable than prompt-based
     routing).

  3. Tools compose — you can bind these to any LangChain agent executor,
     LangGraph ToolNode, or future LLM without changing the tool logic.

Tools defined here
──────────────────
  rag_search          — run the full RAG pipeline for a query
  create_support_ticket — create a StateCase ticket in Notion
  update_support_ticket — update status / owner / priority on a ticket
  list_support_tickets  — list tickets with optional status filter
  retrieve_chunks       — low-level retrieval inspector (for the Inspector tab)

Usage in the agent
──────────────────
  tools = get_all_tools()
  llm_with_tools = _get_llm().bind_tools(tools)

  In LangGraph this replaces the manual prompt | llm chain with a
  ToolNode that dispatches tool calls returned by the LLM automatically.
"""

import logging
from typing import Optional
from langchain_core.tools import tool

logger = logging.getLogger("rag.pipeline.statecase_tools_rag")


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool 1 — RAG search
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def rag_search(
    query: str,
    session_id: str,
    industry: str = "",
    doc_type: str = "",
    version: str = "",
) -> dict:
    """
    Search the document library and return an answer with cited sources.

    Use this tool whenever the user asks a question that should be answered
    from the document library — policies, templates, handbooks, SOPs, etc.

    Args:
        query:      The user's question, as written or rewritten for retrieval.
        session_id: The current chat session ID (for multi-turn history context).
        industry:   Optional filter — restrict retrieval to this industry tag.
        doc_type:   Optional filter — restrict retrieval to this document type.
        version:    Optional filter — restrict retrieval to this document version.

    Returns a dict with:
        answer      (str)        — LLM-generated answer with inline [N] citations
        citations   (list[dict]) — source chunks used (title, section, score, etc.)
        avg_score   (float)      — mean retrieval score; < 0.30 means low confidence
        mode        (str)        — QA | COMPARE | SUMMARIZE | SEARCH
        rewritten   (str)        — query after corrective RAG rewrite (if any)
        answerable  (bool)       — True if score >= 0.30 (document was found)
    """
    from rag.pipeline.pipeline_rag   import run_rag_pipeline
    from rag.pipeline.redis_cache_rag import get_session_history
    import asyncio, concurrent.futures

    logger.info("🔧 [tool:rag_search] query='%s…'  session=%s", query[:60], session_id)

    # Build filters — empty strings are ignored by build_filters
    raw_filters: dict = {}
    if industry: raw_filters["industry"] = industry
    if doc_type:  raw_filters["doc_type"] = doc_type
    if version:   raw_filters["version"]  = version

    # Load session history (sync wrapper around async Redis call)
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as pool:
                history = pool.submit(asyncio.run, get_session_history(session_id)).result(timeout=5)
        else:
            history = loop.run_until_complete(get_session_history(session_id))
    except Exception:
        history = []

    result = run_rag_pipeline(
        query=query,
        session_history=history,
        raw_filters=raw_filters or None,
    )

    avg_score  = result.get("avg_score", 0.0)
    answerable = avg_score >= 0.30
    citations  = result.get("citations", [])

    # Collect all retrieved chunk titles for the "Attempted Sources" Notion field.
    # Citations only contains chunks that PASSED the relevance threshold, so for
    # unanswerable queries it is always empty — that is why sources showed "None".
    # We pull titles from the raw "chunks" list (all Milvus hits before filtering)
    # and fall back to citations if the pipeline didn't surface raw chunks.
    raw_chunks = result.get("chunks", [])
    if raw_chunks:
        attempted_sources = sorted({c.get("title","").strip() for c in raw_chunks if c.get("title","").strip()})
    elif citations:
        attempted_sources = sorted({c.get("title","").strip() for c in citations if c.get("title","").strip()})
    else:
        # Score gate fired — pipeline returned empty chunks AND empty citations.
        # Call the retriever directly so we still get the titles of what was tried.
        try:
            from rag.retrieval.retriever_rag import retrieve
            from rag.retrieval.filters_rag   import build_filters
            _f = build_filters(raw_filters) if raw_filters else None
            attempted_sources = sorted({
                c.get("title","").strip()
                for c in retrieve(query, top_k=5, filters=_f)
                if c.get("title","").strip()
            })
        except Exception as exc:
            logger.warning("attempted_sources fallback failed: %s", exc)
            attempted_sources = []

    logger.info(
        "   ✅ [tool:rag_search] score=%.4f  answerable=%s  citations=%d  attempted_sources=%d",
        avg_score, answerable, len(citations), len(attempted_sources),
    )

    return {
        "answer":            result.get("answer", ""),
        "citations":         citations,
        "avg_score":         avg_score,
        "mode":              result.get("mode", "QA"),
        "rewritten":         result.get("rewritten", query),
        "answerable":        answerable,
        "attempted_sources": attempted_sources,  # all chunk titles tried, even below threshold
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool 2 — Create support ticket
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def create_support_ticket(
    question: str,
    session_id: str,
    description: str = "",
    priority: str = "Medium",
    assigned_owner: str = "Unassigned",
    attempted_sources: str = "",
    user_info: str = "",
) -> dict:
    """
    Create a support ticket in the StateCase Notion database.

    Use this tool when:
      - The user explicitly asks to raise / create / log / escalate a ticket.
      - A rag_search returned answerable=False and the user confirmed they want
        a ticket raised (i.e. they replied yes to the offer).

    Do NOT use this tool automatically without user confirmation unless the
    user has explicitly requested it.

    Args:
        question:          The full user question or issue description.
        session_id:        Current chat session ID.
        description:       Short one-line summary (auto-derived from question if empty).
        priority:          Ticket priority: Low | Medium | High | Critical.
        assigned_owner:    Name or team to assign the ticket to.
        attempted_sources: Comma-separated list of document titles that were tried
                           but did not contain the answer. Leave blank if unknown.
        user_info:         Any extra context about the user or their situation.

    Returns a dict with:
        ticket_id       (str)  — e.g. "SC-0012"
        notion_page_id  (str)  — Notion page UUID for updates
        status          (str)  — always "Not started" on creation
        url             (str)  — direct Notion URL for the ticket
        success         (bool) — True if ticket was created successfully
    """
    from rag.pipeline.statecase_notion_rag import create_ticket

    sources_list = [s.strip() for s in attempted_sources.split(",") if s.strip()] if attempted_sources else []

    logger.info(
        "🔧 [tool:create_support_ticket] question='%s…'  priority=%s  session=%s",
        question[:60], priority, session_id,
    )

    try:
        ticket = create_ticket(
            question=question,
            session_id=session_id,
            description=description,
            priority=priority,
            assigned_owner=assigned_owner,
            attempted_sources=sources_list or None,
            user_info=user_info,
        )
        logger.info("   ✅ [tool:create_support_ticket] ticket_id=%s", ticket.get("ticket_id"))
        return {**ticket, "success": True}
    except Exception as err:
        logger.error("   ❌ [tool:create_support_ticket] error: %s", err)
        return {"success": False, "error": str(err)}


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool 3 — Update support ticket
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def update_support_ticket(
    notion_page_id: str,
    status: str = "",
    assigned_owner: str = "",
    priority: str = "",
    description: str = "",
) -> dict:
    """
    Update an existing StateCase support ticket.

    Use this tool when the user asks to change the status, owner, or priority
    of a ticket they can see in the Tickets tab.

    Only provide the fields you want to change — leave others blank.

    Args:
        notion_page_id: The Notion page UUID of the ticket. If you only know
                        the ticket title or SC-XXXX ID, pass that — this tool
                        will look up the correct UUID automatically.
        status:         New status. Must be exactly one of:
                        "Not started" | "In progress" | "Done"
        assigned_owner: New assignee name or team.
        priority:       New priority: Low | Medium | High | Critical.
        description:    Updated short description.

    Returns the updated ticket dict with success=True, or success=False + error.
    """
    from rag.pipeline.statecase_notion_rag import update_ticket, find_ticket_by_title
    import re

    logger.info(
        "🔧 [tool:update_support_ticket] page_id=%s  status=%s  owner=%s  priority=%s",
        notion_page_id, status or "(unchanged)", assigned_owner or "(unchanged)", priority or "(unchanged)",
    )

    # ── Resolve non-UUID identifiers to the actual Notion page UUID ────────────
    # The LLM often supplies the ticket title or SC-XXXX ID instead of a UUID.
    # Detect this and look up the real page_id before calling the API.
    _UUID_RE = re.compile(
        r"^[0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12}$",
        re.IGNORECASE,
    )
    resolved_page_id = notion_page_id
    if not _UUID_RE.match(notion_page_id.replace("-", "")):
        logger.info(
            "   ℹ️  '%s' is not a UUID — looking up ticket by title/ID…",
            notion_page_id,
        )
        ticket = find_ticket_by_title(notion_page_id)
        if not ticket:
            msg = f"Could not find a ticket matching '{notion_page_id}'. Please check the ticket ID or title."
            logger.error("   ❌ [tool:update_support_ticket] %s", msg)
            return {"success": False, "error": msg}
        resolved_page_id = ticket["notion_page_id"]
        logger.info("   ✅ Resolved to page_id=%s", resolved_page_id)

    try:
        ticket = update_ticket(
            notion_page_id=resolved_page_id,
            status=status or None,
            assigned_owner=assigned_owner or None,
            priority=priority or None,
            description=description or None,
        )
        logger.info("   ✅ [tool:update_support_ticket] ticket_id=%s", ticket.get("ticket_id"))
        return {**ticket, "success": True}
    except Exception as err:
        logger.error("   ❌ [tool:update_support_ticket] error: %s", err)
        return {"success": False, "error": str(err)}


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool 4 — List support tickets
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def list_support_tickets(
    status_filter: str = "",
    limit: int = 20,
) -> dict:
    """
    List support tickets from the StateCase Notion database.

    Use this tool when the user asks to see their tickets, check ticket status,
    or browse open/resolved/in-progress tickets.

    Args:
        status_filter: Filter by status. Leave blank for all tickets.
                       Valid values: "Not started" | "In progress" | "Done"
        limit:         Maximum number of tickets to return (default 20, max 100).

    Returns a dict with:
        tickets  (list[dict]) — list of ticket dicts
        count    (int)        — number of tickets returned
        success  (bool)
    """
    from rag.pipeline.statecase_notion_rag import list_tickets

    logger.info(
        "🔧 [tool:list_support_tickets] status_filter=%s  limit=%d",
        status_filter or "(all)", limit,
    )

    try:
        tickets = list_tickets(
            status_filter=status_filter or None,
            limit=min(limit, 100),
        )
        logger.info("   ✅ [tool:list_support_tickets] returned %d tickets", len(tickets))
        return {"tickets": tickets, "count": len(tickets), "success": True}
    except Exception as err:
        logger.error("   ❌ [tool:list_support_tickets] error: %s", err)
        return {"tickets": [], "count": 0, "success": False, "error": str(err)}


# ═══════════════════════════════════════════════════════════════════════════════
#  Tool 5 — Low-level retrieval inspector
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def retrieve_chunks(
    query: str,
    top_k: int = 5,
    industry: str = "",
    doc_type: str = "",
    version: str = "",
) -> dict:
    """
    Retrieve raw document chunks from Milvus with similarity scores.

    Use this tool for retrieval debugging or when you need to inspect the
    raw evidence before generating an answer.  For normal Q&A, use rag_search
    instead (it includes the LLM answer generation step).

    Args:
        query:    Search query string.
        top_k:    Number of chunks to return (1–20).
        industry: Optional metadata filter.
        doc_type: Optional metadata filter.
        version:  Optional metadata filter.

    Returns a dict with:
        chunks    (list[dict]) — raw chunks with score, title, section, chunk_text
        count     (int)
        avg_score (float)
        success   (bool)
    """
    from rag.retrieval.retriever_rag import retrieve, embed_text
    from rag.retrieval.filters_rag   import build_filters

    logger.info(
        "🔧 [tool:retrieve_chunks] query='%s…'  top_k=%d", query[:60], top_k,
    )

    raw_filters: dict = {}
    if industry: raw_filters["industry"] = industry
    if doc_type:  raw_filters["doc_type"] = doc_type
    if version:   raw_filters["version"]  = version
    filters = build_filters(raw_filters) or None

    try:
        chunks    = retrieve(query=query, top_k=min(top_k, 20), filters=filters)
        avg_score = round(
            sum(c.get("score", 0) for c in chunks) / max(len(chunks), 1), 4
        )
        logger.info(
            "   ✅ [tool:retrieve_chunks] %d chunks  avg_score=%.4f", len(chunks), avg_score,
        )
        return {"chunks": chunks, "count": len(chunks), "avg_score": avg_score, "success": True}
    except Exception as err:
        logger.error("   ❌ [tool:retrieve_chunks] error: %s", err)
        return {"chunks": [], "count": 0, "avg_score": 0.0, "success": False, "error": str(err)}


# ═══════════════════════════════════════════════════════════════════════════════
#  Registry
# ═══════════════════════════════════════════════════════════════════════════════

def get_all_tools() -> list:
    """Return all StateCase tools for binding to an LLM or ToolNode."""
    return [
        rag_search,
        create_support_ticket,
        update_support_ticket,
        list_support_tickets,
        retrieve_chunks,
    ]


def get_rag_tools() -> list:
    """Return only the retrieval-side tools (no ticket mutation)."""
    return [rag_search, retrieve_chunks]


def get_ticket_tools() -> list:
    """Return only the ticket management tools."""
    return [create_support_ticket, update_support_ticket, list_support_tickets]