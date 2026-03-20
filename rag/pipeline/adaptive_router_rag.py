"""
rag/pipeline/adaptive_router_rag.py

Adaptive RAG router for CiteRagLab — built with LangChain + LangGraph.

Architecture
────────────
  A single-node LangGraph graph wraps a LangChain AzureChatOpenAI call that
  classifies the user query into one of four modes.  The LLM is asked to
  respond with exactly one token (QA / COMPARE / SUMMARIZE / SEARCH).

  If the LLM is unavailable or returns an unrecognised response, the router
  falls back to the compiled regex keyword heuristics — so the pipeline
  always gets a mode back and never crashes.

  LangGraph is used here so the router can be composed into a larger
  agentic graph later (e.g. with memory nodes, feedback loops) without
  rewriting the classification logic.

Modes
─────
    QA        — direct question answering    top_k=5
    COMPARE   — side-by-side comparison      top_k=10
    SUMMARIZE — overview / explanation       top_k=10
    SEARCH    — listing / exploration        top_k=8

Azure env variables required
─────────────────────────────
    AZURE_OPENAI_LLM_KEY         — Azure OpenAI API key for the LLM
    AZURE_LLM_ENDPOINT           — Azure OpenAI endpoint for the LLM
    AZURE_LLM_API_VERSION        — API version (default: 2024-12-01-preview)
    AZURE_LLM_DEPLOYMENT_41_MINI — deployment name (gpt-4.1-mini)
"""

import os
import re
import logging
from typing import TypedDict
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

load_dotenv()

logger = logging.getLogger("rag.pipeline.adaptive_router_rag")

# ── Strategy parameters per mode ──────────────────────────────────────────────
_STRATEGY_MAP: dict[str, dict] = {
    "QA":        {"top_k": 5,  "llm_mode": "qa"},
    "COMPARE":   {"top_k": 10, "llm_mode": "compare"},
    "SUMMARIZE": {"top_k": 10, "llm_mode": "summarize"},
    "SEARCH":    {"top_k": 8,  "llm_mode": "qa"},
}

VALID_MODES = set(_STRATEGY_MAP.keys())

# ── Keyword fallback patterns (used when LLM is unavailable) ──────────────────
_COMPARE_RE = re.compile(
    r"\b(compare|vs\.?|versus|difference|contrast|between|against"
    r"|differ|distinguish|similarities|which is better|what changed)\b",
    re.I,
)
_SUMMARIZE_RE = re.compile(
    r"\b(summarize|summarise|summary|overview|outline|brief|explain"
    r"|describe|what is|what are|tell me about|walk me through"
    r"|key points|main points)\b",
    re.I,
)
_SEARCH_RE = re.compile(
    r"\b(find|search|look up|look for|locate|list|show me|give me"
    r"|fetch|get me|all documents|all policies|all templates"
    r"|examples of|related to)\b",
    re.I,
)

# ── LangChain classification prompt ───────────────────────────────────────────
_CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a query classifier for a document library RAG system.
Classify the user query into exactly one of these four modes:

QA        — direct question about a specific fact or policy detail
COMPARE   — comparing two or more documents, versions, or topics
SUMMARIZE — requesting an overview, summary, or explanation of a topic
SEARCH    — exploring, listing, or finding documents by topic or type

Reply with ONLY the mode name — one word, no punctuation, no explanation.
Valid responses: QA, COMPARE, SUMMARIZE, SEARCH""",
    ),
    ("human", "{query}"),
])


# ── LangGraph state ────────────────────────────────────────────────────────────

class RouterState(TypedDict):
    query: str
    mode:  str   # filled by the classification node


# ── LangGraph node ─────────────────────────────────────────────────────────────

def _classify_node(state: RouterState) -> RouterState:
    """
    LangGraph node: call the Azure LLM to classify the query.
    Falls back to keyword heuristics on any error.
    """
    query = state["query"]

    try:
        llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_LLM_DEPLOYMENT_41_MINI", "gpt-4.1-mini"),
            azure_endpoint=os.getenv("AZURE_LLM_ENDPOINT", ""),
            api_key=os.getenv("AZURE_OPENAI_LLM_KEY", ""),
            api_version=os.getenv("AZURE_LLM_API_VERSION", "2024-12-01-preview"),
            temperature=0,
            max_tokens=10,
        )
        chain    = _CLASSIFICATION_PROMPT | llm
        response = chain.invoke({"query": query})
        raw_mode = response.content.strip().upper()

        # Accept only known modes — anything else falls back to heuristics
        if raw_mode in VALID_MODES:
            logger.info(
                "🗂️  _classify_node (LLM) — mode=%s  query='%s…'",
                raw_mode, query[:60],
            )
            return {"query": query, "mode": raw_mode}

        logger.warning(
            "   ⚠️  LLM returned unexpected mode '%s' — falling back to heuristics",
            raw_mode,
        )

    except Exception as err:
        logger.warning(
            "   ⚠️  LLM classification failed (%s) — falling back to heuristics",
            err,
        )

    # ── Keyword heuristic fallback ────────────────────────────────────────────
    if _COMPARE_RE.search(query):
        mode = "COMPARE"
    elif _SUMMARIZE_RE.search(query):
        mode = "SUMMARIZE"
    elif _SEARCH_RE.search(query):
        mode = "SEARCH"
    else:
        mode = "QA"

    logger.info(
        "🗂️  _classify_node (heuristic fallback) — mode=%s  query='%s…'",
        mode, query[:60],
    )
    return {"query": query, "mode": mode}


# ── Build the LangGraph router graph (compiled once at module load) ────────────

def _build_router_graph():
    graph = StateGraph(RouterState)
    graph.add_node("classify", _classify_node)
    graph.set_entry_point("classify")
    graph.add_edge("classify", END)
    return graph.compile()


_router_graph = _build_router_graph()


# ── Public API ────────────────────────────────────────────────────────────────

def classify_query(query: str) -> str:
    """
    Classify a user query into one of: QA | COMPARE | SUMMARIZE | SEARCH.

    Runs the LangGraph router graph which calls the Azure LLM.
    Falls back to keyword heuristics if the LLM is unavailable.

    Parameters
    ──────────
    query : raw user query string

    Returns
    ───────
    One of: "QA", "COMPARE", "SUMMARIZE", "SEARCH"
    """
    result = _router_graph.invoke({"query": query, "mode": ""})
    return result["mode"]


def get_retrieval_params(mode: str) -> dict:
    """
    Return retrieval strategy parameters for the given mode.

    Parameters
    ──────────
    mode : one of "QA", "COMPARE", "SUMMARIZE", "SEARCH"
           Unknown values fall back to QA.

    Returns
    ───────
    dict with keys:
        top_k    (int) — number of chunks to retrieve from Milvus
        llm_mode (str) — key into prompts_rag.SYSTEM_PROMPT_BY_MODE
    """
    if mode not in _STRATEGY_MAP:
        logger.warning(
            "   ⚠️  get_retrieval_params: unknown mode '%s' — falling back to QA", mode
        )

    params = _STRATEGY_MAP.get(mode, _STRATEGY_MAP["QA"])
    logger.info(
        "   ✅ get_retrieval_params — mode=%s  top_k=%d  llm_mode=%s",
        mode, params["top_k"], params["llm_mode"],
    )
    return params