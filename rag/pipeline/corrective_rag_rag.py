"""
rag/pipeline/corrective_rag_rag.py

Corrective RAG for CiteRagLab — built with LangChain + LangGraph.

Architecture
────────────
  A LangGraph StateGraph implements the corrective retrieval loop:

    ┌──────────┐
    │  retrieve │  — first retrieval pass using retrieve_fn
    └────┬─────┘
         │
    ┌────▼──────┐
    │   score   │  — compute avg RRF score of retrieved chunks
    └────┬──────┘
         │
    ┌────▼──────────────────────────────────────────────┐
    │  route  (conditional edge)                        │
    │    score ≥ RELEVANCE_THRESHOLD → "done"           │
    │    score <  RELEVANCE_THRESHOLD → "rewrite"       │
    └────┬───────────────────────────────────────────────┘
         │ (rewrite path)
    ┌────▼──────┐
    │  rewrite  │  — AzureChatOpenAI rewrites the query via LangChain chain
    └────┬──────┘
         │
    ┌────▼────────┐
    │  retrieve2  │  — second retrieval pass with the rewritten query
    └────┬────────┘
         │
    ┌────▼──────────────────────────────────────────────┐
    │  pick_best  (conditional edge)                    │
    │    score2 ≥ score1 → keep second-pass results     │
    │    score2 <  score1 → keep first-pass results     │
    └────────────────────────────────────────────────────┘

The graph is compiled once at module load and invoked per request.
Falls back gracefully — if the LLM rewrite fails, first-pass results
are returned unchanged.

Azure env variables required
─────────────────────────────
    AZURE_OPENAI_LLM_KEY         — Azure OpenAI API key
    AZURE_LLM_ENDPOINT           — Azure OpenAI endpoint
    AZURE_LLM_API_VERSION        — API version (default: 2024-12-01-preview)
    AZURE_LLM_DEPLOYMENT_41_MINI — deployment name (gpt-4.1-mini)
"""

import os
import logging
from typing import TypedDict, Callable, Optional
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from rag.pipeline.prompts_rag import REFINE_QUERY_PROMPT

load_dotenv()

logger = logging.getLogger("rag.pipeline.corrective_rag_rag")

RELEVANCE_THRESHOLD = 0.65   # avg RRF score below which we rewrite the query


# ── LangChain rewrite chain ────────────────────────────────────────────────────
# Built lazily (inside the rewrite node) so the Azure client is only
# initialised when an actual rewrite is needed.

_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("human", REFINE_QUERY_PROMPT),
])

_llm: Optional[AzureChatOpenAI] = None


def _get_llm() -> AzureChatOpenAI:
    global _llm
    if _llm is None:
        _llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_LLM_DEPLOYMENT_41_MINI", "gpt-4.1-mini"),
            azure_endpoint=os.getenv("AZURE_LLM_ENDPOINT", ""),
            api_key=os.getenv("AZURE_OPENAI_LLM_KEY", ""),
            api_version=os.getenv("AZURE_LLM_API_VERSION", "2024-12-01-preview"),
            temperature=0.3,
            max_tokens=128,
        )
        logger.info(
            "✅ Corrective RAG AzureChatOpenAI client initialised "
            "(deployment=%s)",
            os.getenv("AZURE_LLM_DEPLOYMENT_41_MINI", "gpt-4.1-mini"),
        )
    return _llm


# ── LangGraph state ────────────────────────────────────────────────────────────

class CorrectiveRAGState(TypedDict):
    # Inputs (set before graph invocation)
    query:          str
    retrieve_fn:    Callable           # (query, top_k, filters) → list[dict]
    top_k:          int
    filters:        Optional[dict]
    session_history: list[dict]        # prior chat turns [{role, content}] for context-aware rewrite

    # Filled by nodes
    chunks1:     list[dict]         # first-pass chunks
    score1:      float              # avg score of first pass
    rewritten:   str                # rewritten query (may equal original)
    chunks2:     list[dict]         # second-pass chunks (may be empty)
    score2:      float              # avg score of second pass

    # Output
    final_chunks: list[dict]
    final_query:  str


# ── Helper ─────────────────────────────────────────────────────────────────────

def avg_score(chunks: list[dict]) -> float:
    """Compute the mean RRF score across a list of retrieved chunks."""
    if not chunks:
        return 0.0
    return sum(c.get("score", 0.0) for c in chunks) / len(chunks)


# ── LangGraph nodes ────────────────────────────────────────────────────────────

def _node_retrieve(state: CorrectiveRAGState) -> dict:
    """Node: first retrieval pass."""
    query       = state["query"]
    retrieve_fn = state["retrieve_fn"]
    top_k       = state["top_k"]
    filters     = state["filters"]

    logger.info(
        "🔄 [node_retrieve] — query='%s…'  top_k=%d  filters=%s",
        query[:60], top_k, filters or {},
    )

    chunks = retrieve_fn(query, top_k, filters)
    score  = avg_score(chunks)

    logger.info(
        "   📊 [node_retrieve] — %d chunks  avg_score=%.4f",
        len(chunks), score,
    )
    return {"chunks1": chunks, "score1": score}


def _node_rewrite(state: CorrectiveRAGState) -> dict:
    """Node: rewrite the query using the Azure LLM, with session history for context."""
    query           = state["query"]
    session_history = state.get("session_history") or []

    # Build a compact history string from the last 6 turns.
    # Truncate each turn to 200 chars to keep the prompt concise.
    # IMPORTANT: only include user turns — prior AI answers contain content
    # from old topics which would misdirect the rewritten query when the
    # user shifts topic mid-session.
    history_lines = []
    for turn in session_history[-6:]:
        role = turn.get("role", "user").capitalize()
        if role != "User":
            continue   # skip AI turns — they carry stale topic content
        text = turn.get("content", "")[:200].replace("\n", " ")
        history_lines.append(f"{role}: {text}")
    history_str = "\n".join(history_lines) if history_lines else "(no prior conversation)"

    logger.info(
        "✏️  [node_rewrite] — rewriting query='%s'  history_turns=%d",
        query, len(history_lines),
    )

    try:
        chain     = _REWRITE_PROMPT | _get_llm()
        response  = chain.invoke({"query": query, "history": history_str})
        rewritten = response.content.strip().strip('"').strip("'")

        if not rewritten or rewritten == query:
            logger.info(
                "   ℹ️  [node_rewrite] — LLM produced identical query, keeping original"
            )
            return {"rewritten": query}

        logger.info(
            "   ✅ [node_rewrite] — original='%s' → rewritten='%s'",
            query, rewritten,
        )
        return {"rewritten": rewritten}

    except Exception as err:
        logger.warning(
            "   ⚠️  [node_rewrite] — LLM rewrite failed (%s), keeping original query",
            err,
        )
        return {"rewritten": query}
def _node_retrieve2(state: CorrectiveRAGState) -> dict:
    """Node: second retrieval pass using the rewritten query."""
    rewritten   = state["rewritten"]
    retrieve_fn = state["retrieve_fn"]
    top_k       = state["top_k"]
    filters     = state["filters"]

    logger.info(
        "🔄 [node_retrieve2] — rewritten='%s…'  top_k=%d",
        rewritten[:60], top_k,
    )

    chunks2 = retrieve_fn(rewritten, top_k, filters)
    score2  = avg_score(chunks2)

    logger.info(
        "   📊 [node_retrieve2] — %d chunks  avg_score=%.4f",
        len(chunks2), score2,
    )
    return {"chunks2": chunks2, "score2": score2}


def _node_pick_best(state: CorrectiveRAGState) -> dict:
    """Node: compare both passes and keep the better result."""
    score1 = state["score1"]
    score2 = state["score2"]

    if score2 >= score1:
        logger.info(
            "   ✅ [node_pick_best] — second pass wins (%.4f ≥ %.4f)",
            score2, score1,
        )
        return {
            "final_chunks": state["chunks2"],
            "final_query":  state["rewritten"],
        }

    logger.info(
        "   ℹ️  [node_pick_best] — first pass kept (%.4f < %.4f)",
        score2, score1,
    )
    return {
        "final_chunks": state["chunks1"],
        "final_query":  state["query"],
    }


# ── Conditional edge: route after scoring ─────────────────────────────────────

def _route_after_score(state: CorrectiveRAGState) -> str:
    """
    If score is above the threshold, skip rewrite and go straight to done.
    Otherwise trigger the rewrite → retrieve2 → pick_best path.
    """
    score = state["score1"]
    if score >= RELEVANCE_THRESHOLD:
        logger.info(
            "   ✅ [route_after_score] — score %.4f ≥ threshold %.2f → done",
            score, RELEVANCE_THRESHOLD,
        )
        return "done"

    logger.info(
        "   ⚠️  [route_after_score] — score %.4f < threshold %.2f → rewrite",
        score, RELEVANCE_THRESHOLD,
    )
    return "rewrite"


def _node_done(state: CorrectiveRAGState) -> dict:
    """Terminal node for the happy path (score already good enough)."""
    logger.info("   ✅ [node_done] — returning first-pass results unchanged")
    return {
        "final_chunks": state["chunks1"],
        "final_query":  state["query"],
    }


# ── Build and compile the LangGraph graph ─────────────────────────────────────

def _build_corrective_graph():
    graph = StateGraph(CorrectiveRAGState)

    graph.add_node("retrieve",   _node_retrieve)
    graph.add_node("rewrite",    _node_rewrite)
    graph.add_node("retrieve2",  _node_retrieve2)
    graph.add_node("pick_best",  _node_pick_best)
    graph.add_node("done",       _node_done)

    graph.set_entry_point("retrieve")

    # After first retrieval, branch on score
    graph.add_conditional_edges(
        "retrieve",
        _route_after_score,
        {
            "done":    "done",
            "rewrite": "rewrite",
        },
    )

    # Rewrite path: rewrite → second retrieve → pick best
    graph.add_edge("rewrite",   "retrieve2")
    graph.add_edge("retrieve2", "pick_best")

    # Both terminal nodes lead to END
    graph.add_edge("done",      END)
    graph.add_edge("pick_best", END)

    return graph.compile()


_corrective_graph = _build_corrective_graph()


# ── Public API ────────────────────────────────────────────────────────────────

def corrective_retrieve(
    query: str,
    retrieve_fn: Callable,
    top_k: int = 5,
    filters: Optional[dict] = None,
    session_history: list[dict] | None = None,
) -> tuple[list[dict], str]:
    """
    Run the LangGraph corrective retrieval graph.

    Retrieves chunks, scores them, and rewrites the query + re-retrieves
    if the average score is below RELEVANCE_THRESHOLD.

    Parameters
    ──────────
    query           : original user query
    retrieve_fn     : callable (query: str, top_k: int, filters: dict|None) → list[dict]
    top_k           : number of chunks to retrieve per pass
    filters         : optional Milvus metadata filters
    session_history : prior chat turns [{role, content}] — used by the rewrite node
                      to resolve ambiguous references using conversation context

    Returns
    ───────
    (final_chunks, final_query_used)
        final_chunks      — best chunk list found (first or second pass)
        final_query_used  — query string that produced those chunks
    """
    logger.info(
        "🔄 corrective_retrieve (LangGraph) — query='%s…'  top_k=%d  filters=%s",
        query[:60], top_k, filters or {},
    )

    initial_state: CorrectiveRAGState = {
        "query":           query,
        "retrieve_fn":     retrieve_fn,
        "top_k":           top_k,
        "filters":         filters,
        "session_history": session_history or [],
        "chunks1":         [],
        "score1":          0.0,
        "rewritten":       query,
        "chunks2":         [],
        "score2":          0.0,
        "final_chunks":    [],
        "final_query":     query,
    }

    result = _corrective_graph.invoke(initial_state)

    final_chunks = result["final_chunks"]
    final_query  = result["final_query"]

    logger.info(
        "   ✅ corrective_retrieve done — %d chunks  final_query='%s…'  rewritten=%s",
        len(final_chunks),
        final_query[:60],
        final_query != query,
    )
    return final_chunks, final_query