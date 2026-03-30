"""
rag/pipeline/statecase_agent_rag.py

StateCase — LangGraph tool-calling agent for CiteRagLab.

Architecture (tool-calling version)
────────────────────────────────────
The agent uses Azure OpenAI function-calling (bind_tools) instead of
hard-coded prompt→function dispatch.  The LLM decides which tool to call
based on the user message and conversation context.

LangGraph graph:

    load_mem → agent → tools → agent → ... → save_mem → END
                  ↑___________________|
                  (loop while tool calls returned)
                  ↓ (no more tool calls)
               save_mem → END

Nodes
─────
  load_mem   — hydrate durable memory + pending_ticket_context from Redis
  agent      — AzureChatOpenAI with bind_tools; returns tool_calls or final text
  tools      — LangGraph ToolNode dispatches tool calls to statecase_tools_rag
  save_mem   — persist memory + session history to Redis

Tools available to the LLM (from statecase_tools_rag.py)
─────────────────────────────────────────────────────────
  rag_search              — RAG pipeline: retrieve + answer + score
  create_support_ticket   — create a Notion StateCase ticket
  update_support_ticket   — update ticket status / owner / priority
  list_support_tickets    — list tickets from Notion
  retrieve_chunks         — raw chunk retrieval for inspector/debug

Why tools instead of direct calls?
────────────────────────────────────
  1. The LLM chooses which action to take based on the message — no
     hard-coded intent classification node needed for routing.
  2. Azure OpenAI function-calling validates arguments against a Pydantic
     schema before dispatch — no manual parsing.
  3. Tools compose — bind them to any future agent without rewriting logic.
  4. Multi-step reasoning: the LLM can call rag_search, see answerable=False,
     then call create_support_ticket in the same turn if the user already
     said "raise a ticket" — or it can ask first and wait for confirmation.

"Cannot answer" + confirmation flow
─────────────────────────────────────
  The SYSTEM PROMPT instructs the LLM:
    - Always call rag_search first.
    - If answerable=False, OFFER to create a ticket — do not call
      create_support_ticket yet.
    - Store the pending question in memory (set_pending_context tool-call
      equivalent is handled via the system prompt + memory dict).
    - On the next turn, if user says yes → call create_support_ticket.
    - If user says no → acknowledge and clear pending.

  The agent loop handles multi-turn naturally because the full memory
  (including pending_ticket_context) is injected into every agent call.
"""

import os
import uuid
import json
import logging
import asyncio
import concurrent.futures
from typing import TypedDict, Optional, Annotated
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, ToolMessage, BaseMessage,
)
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from rag.pipeline.statecase_tools_rag import get_all_tools
from rag.pipeline.redis_cache_rag     import get_session_history, set_session_history

load_dotenv()

logger = logging.getLogger("rag.pipeline.statecase_agent_rag")

# ── System prompt ──────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are Citter, a helpful document library assistant with access to tools.
You work inside CiteRagLab — a RAG chat interface with an integrated StateCase ticketing system.

## Your capabilities (5 tools)
| Tool                     | What it does                                               |
|--------------------------|------------------------------------------------------------|
| **rag_search**           | Search the document library → answer with [N] citations    |
| **create_support_ticket**| Create a new ticket in Notion                              |
| **update_support_ticket**| Update an existing ticket (status / owner / priority)      |
| **list_support_tickets** | List tickets from Notion with optional status filter       |
| **retrieve_chunks**      | Low-level retrieval inspector (raw chunks + scores)        |

────────────────────────────────────────────────────────────────────────────────
## INTENT DETECTION — how to figure out what the user wants
────────────────────────────────────────────────────────────────────────────────

Users will often write messy, vague, abbreviated, or misspelled messages.
YOUR JOB is to figure out their intent and call the right tool(s).

### Intent → Document Question (call rag_search)
Trigger words/patterns (case-insensitive, fuzzy):
  "what is", "how to", "explain", "tell me about", "policy", "process",
  "SLA", "SOP", "handbook", "template", "document", "find", "search",
  "look up", "info on", "details about", "describe", any question mark,
  or anything that looks like a knowledge/information question.

Even if the query is:
  - Gibberish-adjacent: "wats the sla for incidents?" → rag_search("What is the SLA for incidents?")
  - Extremely short: "password policy" → rag_search("What is the password policy?")
  - Just a keyword: "escalation" → rag_search("What is the escalation process?")
  - Misspelled: "incidnet respons" → rag_search("incident response process")

ALWAYS rewrite the query into a clean, well-formed search query before passing to rag_search.

### Intent → List Tickets (call list_support_tickets)
Trigger words/patterns:
  "list", "show", "get", "see", "view", "check", "display", "all tickets",
  "my tickets", "tickets", "open tickets", "pending", "whats open",
  "ticket board", "ticket status", "how many tickets"

Even if the query is:
  - "show me everything" (when in ticket context) → list_support_tickets
  - "any open ones?" → list_support_tickets(status_filter="Not started")
  - "tickets" (just the word) → list_support_tickets

### Intent → Update Ticket (call list_support_tickets FIRST, then update_support_ticket)
Trigger words/patterns:
  "update", "change", "edit", "modify", "set", "mark", "move", "assign",
  "reassign", "close", "resolve", "reopen", "make it", "switch to",
  "put it", "mark as", "change status", "set priority", "bump priority",
  "escalate", "de-escalate"

Even if the query is:
  - "close SC-0012" → list tickets → find SC-0012 → update status to "Done"
  - "mark the last one as done" → list tickets → pick most recent → update
  - "change priority of that vendor ticket to high" → list tickets → match by description → update
  - "assign it to John" → list tickets → pick the one from context → update assigned_owner
  - "resolve all open tickets" → list tickets → update each one

CRITICAL WORKFLOW FOR UPDATES:
  1. ALWAYS call list_support_tickets FIRST to get the notion_page_id
  2. Match the ticket the user is referring to (by ticket_id, description, keywords, or recency)
  3. Call update_support_ticket with the notion_page_id from step 1
  4. If the user's reference is ambiguous, list the candidates and ask which one they mean
  5. NEVER guess a notion_page_id — always get it from list_support_tickets

### Intent → Create Ticket (call create_support_ticket)
Trigger words/patterns:
  "create", "raise", "log", "file", "open a ticket", "new ticket",
  "escalate this", "make a ticket", "submit", "report issue",
  "can't find", "need help with", "raise it", "yes" (after you offered)

Even if the query is:
  - "yeah raise one" → create_support_ticket (using pending context)
  - "log this" → create_support_ticket (using last question from context)
  - "yes" / "sure" / "yep" / "do it" / "go ahead" / "please" → check pending_ticket_context → create

### Intent → Confirmation (yes/no to a previous offer)
Words that mean YES: "yes", "yeah", "yep", "sure", "ok", "okay", "do it",
  "go ahead", "please", "yea", "ya", "y", "affirmative", "absolutely",
  "of course", "definitely", "raise it", "create it", "log it"
Words that mean NO: "no", "nah", "nope", "don't", "skip", "cancel",
  "never mind", "forget it", "no thanks", "not now", "n"

If there is a pending_ticket_context in memory and the user gives a YES → create_support_ticket.
If there is a pending_ticket_context in memory and the user gives a NO → acknowledge and clear.

### Intent → Greeting / Meta
Greetings ("hi", "hello", "hey", "sup", "yo") → respond warmly, introduce yourself.
Meta ("what can you do", "help", "?") → list your capabilities briefly.

### Intent → Ambiguous / Can't tell
If you genuinely cannot determine intent:
  - DO NOT refuse. DO NOT say "I can't help with that."
  - Instead, try rag_search with a cleaned-up version of their query.
  - If rag_search returns answerable=False, THEN ask the user to clarify.

────────────────────────────────────────────────────────────────────────────────
## TOOL CALL WORKFLOWS (step-by-step)
────────────────────────────────────────────────────────────────────────────────

### Workflow A — Answer a document question
1. Clean up the user's query (fix typos, expand abbreviations, resolve pronouns using memory context).
2. Call rag_search with the cleaned query.
3. If answerable=true → present the answer with [N] citations. Done.
4. If answerable=false → tell the user, list attempted sources, and ask:
   "Would you like me to raise a support ticket for this?"
   Store the question in pending context. Wait for next turn.

### Workflow B — Create a ticket
1. If there's pending_ticket_context in memory → use it for the question/sources.
2. If the user is explicitly requesting a new ticket → use their current message.
3. Always provide: question (required), session_id (required), description (derive from question if blank), priority (default Medium unless user specified).
4. Call create_support_ticket.
5. Report: ticket ID, priority, status, and Notion URL if available.

### Workflow C — Update a ticket (NEVER SKIP STEP 1)
1. Call list_support_tickets (no filter, limit=100) to get ALL tickets.
2. From the results, find the ticket the user is referring to:
   - If they said "SC-0012" → match by ticket_id
   - If they said "the vendor one" → match by description/question keywords
   - If they said "the last one" / "that one" / "it" → use the most recently created ticket, or the ticket from memory context
   - If ambiguous → list 2-3 candidates and ask "Which one?"
3. Extract the notion_page_id from the matched ticket.
4. Call update_support_ticket with the notion_page_id and the requested changes.
5. Confirm the update to the user.

### Workflow D — List tickets
1. If the user wants all tickets → call list_support_tickets with no filter.
2. If they want a specific status → map their words to: "Not started" | "In progress" | "Done".
   - "open" / "new" / "pending" → "Not started"
   - "in progress" / "working on" / "ongoing" → "In progress"
   - "done" / "closed" / "resolved" / "finished" → "Done"
3. Present as a numbered list with ticket_id, description, status, priority, owner.

────────────────────────────────────────────────────────────────────────────────
## MULTI-STEP REASONING
────────────────────────────────────────────────────────────────────────────────

You can call MULTIPLE tools in a single turn. For example:
- User says "list all tickets and close SC-0005":
  1. Call list_support_tickets
  2. From results, find SC-0005's notion_page_id
  3. Call update_support_ticket to set status="Done"
  4. Present both: the full ticket list AND the update confirmation

- User says "search for password policy and raise a ticket if not found":
  1. Call rag_search("password policy")
  2. If answerable=false → immediately call create_support_ticket
  3. Report both results

────────────────────────────────────────────────────────────────────────────────
## MEMORY CONTEXT
────────────────────────────────────────────────────────────────────────────────
{memory_context}

────────────────────────────────────────────────────────────────────────────────
## HARD RULES
────────────────────────────────────────────────────────────────────────────────
1. NEVER fabricate document content. Only cite what rag_search returns.
2. NEVER create duplicate tickets — the tool handles deduplication.
3. NEVER guess a notion_page_id — always get it from list_support_tickets.
4. NEVER refuse to try. If unsure → call rag_search as a fallback.
5. ALWAYS rewrite messy queries into clean tool arguments.
6. ALWAYS be concise and well-structured in responses.
7. If a tool returns success=false, explain the error clearly and suggest next steps.
8. Use markdown formatting: **bold** for key terms, bullet lists, ## headings for long answers.
9. If the user query is not related to documents or tickets (e.g. "
what is 3+4(5*32) -32 * 0.5?") → dont answer these types of questions. dont do basic math either. instead, say "I'm here to help with questions about our documents and support tickets. For other questions, you might want to try a different assistant or a search engine!".
"""

# ── Lazy LLM with tools bound ─────────────────────────────────────────────────
_llm_with_tools: Optional[AzureChatOpenAI] = None


def _get_llm_with_tools():
    global _llm_with_tools
    if _llm_with_tools is None:
        llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_LLM_DEPLOYMENT_41_MINI", "gpt-4.1-mini"),
            azure_endpoint=os.getenv("AZURE_LLM_ENDPOINT", ""),
            api_key=os.getenv("AZURE_OPENAI_LLM_KEY", ""),
            api_version=os.getenv("AZURE_LLM_API_VERSION", "2024-12-01-preview"),
            temperature=0.2,
            max_tokens=1024,
        )
        _llm_with_tools = llm.bind_tools(get_all_tools())
        logger.info(
            "✅ LLM initialised with %d bound tools: %s",
            len(get_all_tools()),
            [t.name for t in get_all_tools()],
        )
    return _llm_with_tools


# ── LangGraph state ────────────────────────────────────────────────────────────

class StateCaseAgentState(TypedDict):
    # per-turn inputs
    session_id:   str
    raw_filters:  Optional[dict]
    ticket_priority: str
    ticket_owner:    str
    # message history — add_messages reducer appends rather than replaces
    messages:     Annotated[list[BaseMessage], add_messages]
    # durable memory persisted to Redis
    memory:       dict
    # trace ID for log correlation
    trace_id:     str
    # final extracted values for the API response
    final_response:      str
    final_citations:     list
    final_pipeline_meta: dict
    final_ticket:        Optional[dict]


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _sync_run(coro):
    """
    Run an async coroutine from a sync context.
    Always uses a fresh thread — avoids 'no current event loop' in Python 3.10+.
    """
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result(timeout=8)
    except Exception as err:
        logger.warning("⚠️  _sync_run error: %s", err)
        return None


def _sync_redis_set(key: str, value: str, ex: int) -> None:
    """
    Write a Redis key with the sync client.
    Used in _node_save_mem to avoid 'Event loop is closed' — the async
    singleton gets bound to a thread loop opened by _sync_run, which closes
    when that thread exits. The sync client has no such issue.
    """
    try:
        import redis as _redis_sync
        r = _redis_sync.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"),
            socket_connect_timeout=2,
            decode_responses=True,
        )
        r.set(key, value, ex=ex)
        logger.info("💾 statecase memory saved (sync) key=%s", key)
    except Exception as err:
        logger.warning("⚠️  _sync_redis_set error key=%s: %s", key, err)


def _build_memory_context(memory: dict, session_id: str = "") -> str:
    """Format memory dict into a readable context string for the system prompt."""
    lines = []
    if session_id:
        lines.append(f"- Current session_id: {session_id}  ← ALWAYS use this exact value when calling create_support_ticket")
    if memory.get("last_question"):
        lines.append(f"- Last question asked: {memory['last_question'][:120]}")
    if memory.get("last_answer_summary"):
        lines.append(f"- Summary of last answer given: {memory['last_answer_summary'][:150]}")
    if memory.get("pending_ticket_context"):
        p = memory["pending_ticket_context"]
        src = ", ".join(p.get("attempted_sources", [])) or "(none)"
        lines.append(
            f"- PENDING TICKET OFFER: You previously offered to create a ticket for: "
            f"'{p.get('question','')[:120]}'. "
            f"Attempted sources: {src}. "
            f"If the user says yes → call create_support_ticket with this question and attempted_sources='{src}'."
        )
    return "\n".join(lines) if lines else "(no prior context)"


def _extract_tool_result(messages: list, tool_name: str) -> Optional[dict]:
    """
    Find the most recent ToolMessage for a given tool name and parse its content.
    Returns None if not found or not parseable as JSON.
    """
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            # ToolMessage.name holds the tool that was called
            if getattr(msg, "name", "") == tool_name:
                try:
                    return json.loads(msg.content)
                except Exception:
                    return {"raw": msg.content}
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Nodes
# ═══════════════════════════════════════════════════════════════════════════════

async def _node_load_mem(state: StateCaseAgentState) -> dict:
    """Hydrate durable memory from Redis for this session."""
    session_id = state["session_id"]
    trace_id   = state.get("trace_id") or str(uuid.uuid4())[:8]
    memory: dict = {}

    try:
        from rag.pipeline.redis_cache_rag import _get_client as _get_redis
        redis = await _get_redis()
        if redis:
            raw = await redis.get(f"statecase:memory:{session_id}")
            if raw:
                memory = json.loads(raw)
    except Exception as err:
        logger.warning("⚠️  [%s] load_mem error: %s", trace_id, err)

    logger.info(
        "📥 [%s] load_mem — session=%s  memory_keys=%s  has_pending=%s",
        trace_id, session_id, list(memory.keys()),
        bool(memory.get("pending_ticket_context")),
    )
    return {"memory": memory, "trace_id": trace_id}


def _node_agent(state: StateCaseAgentState) -> dict:
    session_id   = state["session_id"]
    memory       = state.get("memory", {})
    trace_id     = state["trace_id"]
    messages     = state.get("messages", [])

    # Build system message with current memory context (includes real session_id)
    system_msg = SystemMessage(
        content=_SYSTEM_PROMPT.format(memory_context=_build_memory_context(memory, session_id))
    )

    # ── Inject prior session history for multi-turn memory ────────────────────
    # We load the last N turns from Redis and prepend them as HumanMessage /
    # AIMessage pairs so the LLM knows what was said earlier in this session.
    #
    # Context-rot guard: AI turns are wrapped with a note that they are prior
    # conversation history — NOT verified facts or source-of-truth.  This lets
    # the agent remember what it said ("summarise it" works) while preventing it
    # from treating its own previous answers as document ground truth.
    #
    # Topic-shift safety: we only inject the last 6 turns (3 user + 3 assistant).
    # Older history stays in Redis but is not sent to the LLM, so stale context
    # from a previous topic does not bleed into the current answer.
    history_msgs: list[BaseMessage] = []
    try:
        raw_history = _sync_run(get_session_history(session_id)) or []
        recent = raw_history[-6:]  # last 3 user + 3 assistant turns
        for turn in recent:
            role    = turn.get("role", "")
            content = turn.get("content", "").strip()
            if not content:
                continue
            if role == "user":
                history_msgs.append(HumanMessage(content=content))
            elif role == "assistant":
                # Wrap AI turns to prevent context rot
                history_msgs.append(AIMessage(
                    content=f"[Prior answer — treat as conversation context only, not as document source-of-truth]\n{content}"
                ))
    except Exception as err:
        logger.warning("⚠️  [%s] could not load session history: %s", trace_id, err)

    logger.info(
        "🤖 [%s] agent — session=%s  history_turns=%d  current_msgs=%d  has_pending=%s",
        trace_id, session_id, len(history_msgs), len(messages),
        bool(memory.get("pending_ticket_context")),
    )

    response = _get_llm_with_tools().invoke([system_msg] + history_msgs + messages)

    tool_calls = getattr(response, "tool_calls", [])
    logger.info(
        "   ← LLM response — tool_calls=%d  content_len=%d",
        len(tool_calls), len(response.content or ""),
    )
    for tc in tool_calls:
        logger.info("      🔧 tool_call: %s(%s)", tc["name"], str(tc["args"])[:120])

    updated_memory = dict(memory)
    for tc in tool_calls:
        if tc["name"] == "rag_search":
            updated_memory["last_question"] = tc["args"].get("query", "")
    for tc in tool_calls:
        if tc["name"] == "create_support_ticket":
            updated_memory.pop("pending_ticket_context", None)

    return {"messages": [response], "memory": updated_memory}


def _node_update_memory_after_tools(state: StateCaseAgentState) -> dict:
    """
    After tools have executed, inspect tool results and update memory.

    Specifically:
    - If rag_search returned answerable=False, store pending_ticket_context
      so the next turn knows to look for a yes/no confirmation.
    - If create_support_ticket succeeded, store the ticket_id in memory.
    - If rag_search returned answerable=True, clear any stale pending context.
    """
    messages      = state.get("messages", [])
    memory        = dict(state.get("memory", {}))
    trace_id      = state["trace_id"]
    session_id    = state["session_id"]

    rag_result    = _extract_tool_result(messages, "rag_search")
    ticket_result = _extract_tool_result(messages, "create_support_ticket")

    if rag_result:
        if not rag_result.get("answerable", True):
            # Use the attempted_sources list the rag_search tool now returns.
            # This contains all Milvus chunk titles retrieved, including those
            # below the relevance threshold — so it's never empty.
            attempted = rag_result.get("attempted_sources", []) or ["(none)"]
            last_question = memory.get("last_question", "")
            memory["pending_ticket_context"] = {
                "question":          last_question,
                "session_id":        session_id,
                "attempted_sources": attempted,
                "trace_id":          trace_id,
                "priority":          state.get("ticket_priority", "Medium"),
                "owner":             state.get("ticket_owner", "Unassigned"),
            }
            logger.info("🤔 [%s] rag_search unanswerable — pending_ticket_context stored (sources=%s)", trace_id, attempted)
        else:
            memory.pop("pending_ticket_context", None)
            # Store a short summary of the answer so the agent can reference it
            # on follow-ups ("summarise it") without storing the full answer
            # (which caused context rot when the full text was used as ground truth)
            answer_text = rag_result.get("answer", "")
            memory["last_answer_summary"] = answer_text[:150].rstrip() + ("…" if len(answer_text) > 150 else "")
            logger.info("✅ [%s] rag_search answerable — pending context cleared, summary stored", trace_id)

    if ticket_result and ticket_result.get("success"):
        memory.pop("pending_ticket_context", None)
        memory["last_ticket_id"] = ticket_result.get("ticket_id", "")
        logger.info(
            "🎫 [%s] ticket created — id=%s", trace_id, memory["last_ticket_id"]
        )

    return {"memory": memory}


async def _node_save_mem(state: StateCaseAgentState) -> dict:
    """
    Persist durable memory to Redis and append this turn to session history.
    Also extract final_response, final_citations, final_ticket from messages.
    """
    session_id = state["session_id"]
    trace_id   = state["trace_id"]
    memory     = state.get("memory", {})
    messages   = state.get("messages", [])

    # ── Extract final response from last AIMessage ─────────────────────────────
    final_response = ""
    final_citations: list = []
    final_ticket:    Optional[dict] = None
    final_meta:      dict = {}

    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", []):
            final_response = msg.content
            break

    # Extract ticket data from tool results
    ticket_result = _extract_tool_result(messages, "create_support_ticket")
    if ticket_result and ticket_result.get("success"):
        final_ticket = ticket_result

    # Extract citations from rag_search tool result
    rag_result = _extract_tool_result(messages, "rag_search")
    if rag_result:
        final_citations = rag_result.get("citations", [])
        final_meta = {
            "mode":      rag_result.get("mode", "QA"),
            "avg_score": rag_result.get("avg_score", 0.0),
            "rewritten": rag_result.get("rewritten", ""),
        }

    # ── Persist memory to Redis ────────────────────────────────────────────────
    # Use sync client: the async singleton may be bound to a closed event loop
    # if _sync_run was called earlier this turn (it creates and closes thread loops).
    _sync_redis_set(f"statecase:memory:{session_id}", json.dumps(memory), ex=86400)

    # ── Append to session history for multi-turn RAG context ──────────────────
    try:
        history = await get_session_history(session_id)
        # Add just the last user message and final AI response
        user_msgs = [m for m in messages if isinstance(m, HumanMessage)]
        if user_msgs:
            history.append({"role": "user", "content": user_msgs[-1].content})
        if final_response:
            history.append({"role": "assistant", "content": final_response})
        await set_session_history(session_id, history)
    except Exception as err:
        logger.warning("⚠️  [%s] save_mem history error: %s", trace_id, err)

    logger.info(
        "💾 [%s] save_mem — session=%s  memory_keys=%s  response=%d chars  ticket=%s",
        trace_id, session_id, list(memory.keys()),
        len(final_response),
        final_ticket.get("ticket_id") if final_ticket else "none",
    )

    return {
        "final_response":      final_response,
        "final_citations":     final_citations,
        "final_pipeline_meta": final_meta,
        "final_ticket":        final_ticket,
    }


# ── ToolNode — dispatches all tool calls returned by the agent ─────────────────
_tool_node = ToolNode(get_all_tools())


# ═══════════════════════════════════════════════════════════════════════════════
#  Conditional routing
# ═══════════════════════════════════════════════════════════════════════════════

def _should_continue(state: StateCaseAgentState) -> str:
    """
    After the agent node: if the last AIMessage has tool_calls → run tools.
    Otherwise → save memory and finish.
    """
    messages = state.get("messages", [])
    last_msg = messages[-1] if messages else None

    if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", []):
        logger.info("   → continuing to tools (%d calls)", len(last_msg.tool_calls))
        return "tools"

    logger.info("   → no tool calls — proceeding to save_mem")
    return "save_mem"


# ═══════════════════════════════════════════════════════════════════════════════
#  Graph
# ═══════════════════════════════════════════════════════════════════════════════

def _build_agent_graph():
    graph = StateGraph(StateCaseAgentState)

    graph.add_node("load_mem",    _node_load_mem)
    graph.add_node("agent",       _node_agent)
    graph.add_node("tools",       _tool_node)
    graph.add_node("update_mem",  _node_update_memory_after_tools)
    graph.add_node("save_mem",    _node_save_mem)

    graph.set_entry_point("load_mem")
    graph.add_edge("load_mem", "agent")

    # After agent: either run tools or finish
    graph.add_conditional_edges(
        "agent",
        _should_continue,
        {"tools": "tools", "save_mem": "save_mem"},
    )

    # After tools: update memory state, then loop back to agent
    graph.add_edge("tools",      "update_mem")
    graph.add_edge("update_mem", "agent")

    graph.add_edge("save_mem", END)

    return graph.compile()


_agent_graph = _build_agent_graph()


# ═══════════════════════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════════════════════

async def run_statecase_agent(
    session_id:      str,
    user_message:    str,
    raw_filters:     dict | None = None,
    ticket_priority: str = "Medium",
    ticket_owner:    str = "Unassigned",
) -> dict:
    """
    Run the StateCase tool-calling agent for one user turn.

    Returns
    ───────
    {
        response       : str          — final text to show in chat
        citations      : list[dict]   — from rag_search tool result
        pipeline_meta  : dict         — mode, avg_score, rewritten
        ticket_created : dict | None  — ticket if one was created this turn
        trace_id       : str          — log correlation ID
        intent         : str          — "TOOL_CALL" or "DIRECT"
    }
    """
    trace_id = str(uuid.uuid4())[:8]

    initial_state: StateCaseAgentState = {
        "session_id":          session_id,
        "raw_filters":         raw_filters or {},
        "ticket_priority":     ticket_priority,
        "ticket_owner":        ticket_owner,
        "messages":            [HumanMessage(content=user_message)],
        "memory":              {},
        "trace_id":            trace_id,
        "final_response":      "",
        "final_citations":     [],
        "final_pipeline_meta": {},
        "final_ticket":        None,
    }

    logger.info(
        "🚀 run_statecase_agent — session=%s  trace=%s  message='%s…'",
        session_id, trace_id, user_message[:60],
    )

    result = await _agent_graph.ainvoke(initial_state)

    ticket = result.get("final_ticket")
    logger.info(
        "✅ run_statecase_agent done — trace=%s  ticket=%s  response=%d chars",
        trace_id,
        ticket["ticket_id"] if ticket else "none",
        len(result.get("final_response", "")),
    )

    return {
        "response":       result.get("final_response", ""),
        "citations":      result.get("final_citations", []),
        "pipeline_meta":  result.get("final_pipeline_meta", {}),
        "ticket_created": ticket,
        "trace_id":       trace_id,
        "intent":         "TOOL_CALL",
    }