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
    "GREETING":  {"top_k": 0,  "llm_mode": "qa"},   # bypassed — no retrieval needed
}

VALID_MODES = set(_STRATEGY_MAP.keys())

# ── LangChain classification prompt ───────────────────────────────────────────
# Rich few-shot examples cover informal language, typos, and creative phrasings
# so the LLM generalises correctly without any regex keyword matching.
_CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an intent classifier for a document library assistant called Citter.

Classify the user message into exactly one of these modes:

  GREETING  — the user is greeting the assistant, asking what it can do, asking about
              its identity or capabilities, making small talk, or sending any message
              that is NOT a question about documents in the library.
              This includes informal or creative phrasings like "yo what can u do",
              "wot r u", "tell me bout urself", "heyy", "sup", "who r u", etc.

  QA        — a specific factual question about a document, policy, or topic
              that exists in the library. e.g. "what is the SLA for P1 incidents?"

  COMPARE   — asking to compare, contrast, or find differences between two or more
              documents, versions, or topics. e.g. "how does v1 differ from v2?"

  SUMMARIZE — asking for an overview, summary, or explanation of a topic covered
              in the library. e.g. "summarise the access control policy"

  SEARCH    — asking to find, list, or explore documents by topic, type, or tag.
              e.g. "show me all HR templates"

Rules:
- When in doubt between GREETING and any other mode, choose GREETING.
- Reply with ONLY the mode name — one word, uppercase, no punctuation, no explanation.
- Valid responses: GREETING, QA, COMPARE, SUMMARIZE, SEARCH""",
    ),
    ("human", "{query}"),
])

# ── Fallback classification prompt (used if primary LLM call fails) ───────────
# Identical intent but written differently so a transient model quirk on the
# first call is less likely to recur on the retry.
_FALLBACK_CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Classify this message into one word: GREETING, QA, COMPARE, SUMMARIZE, or SEARCH.

GREETING = anything that is not a genuine document question: hellos, small talk,
           questions about the assistant itself, capability questions, typo-heavy
           casual messages, or anything where the user is not asking about a
           specific document or topic in a library.
QA       = specific factual question about a library document or policy.
COMPARE  = comparing two documents, versions, or topics.
SUMMARIZE = asking for a summary or overview of a library topic.
SEARCH   = asking to find or list documents.

When uncertain, output GREETING.
Output only the single word, nothing else.""",
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
    LangGraph node: classify the query using the Azure LLM.

    Attempt 1 — primary prompt (_CLASSIFICATION_PROMPT), temperature=0.
    Attempt 2 — fallback prompt (_FALLBACK_CLASSIFICATION_PROMPT) if attempt 1
                returns an unrecognised token or raises an exception.
    Last resort — default to QA if both LLM calls fail (e.g. total outage).

    No regex or keyword heuristics — intent classification is entirely LLM-driven
    so informal language, typos, and creative phrasings are handled correctly.
    """
    query = state["query"]

    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_LLM_DEPLOYMENT_41_MINI", "gpt-4.1-mini"),
        azure_endpoint=os.getenv("AZURE_LLM_ENDPOINT", ""),
        api_key=os.getenv("AZURE_OPENAI_LLM_KEY", ""),
        api_version=os.getenv("AZURE_LLM_API_VERSION", "2024-12-01-preview"),
        temperature=0,
        max_tokens=10,
    )

    for attempt, prompt in enumerate(
        [_CLASSIFICATION_PROMPT, _FALLBACK_CLASSIFICATION_PROMPT], start=1
    ):
        try:
            response = (prompt | llm).invoke({"query": query})
            raw_mode = response.content.strip().upper()

            if raw_mode in VALID_MODES:
                logger.info(
                    "🗂️  _classify_node (LLM attempt %d) — mode=%s  query='%s…'",
                    attempt, raw_mode, query[:60],
                )
                return {"query": query, "mode": raw_mode}

            logger.warning(
                "   ⚠️  LLM attempt %d returned unrecognised mode '%s'",
                attempt, raw_mode,
            )

        except Exception as err:
            logger.warning(
                "   ⚠️  LLM attempt %d failed: %s", attempt, err,
            )

    # Both LLM calls failed — default to QA so the pipeline still returns
    # something useful rather than crashing. Logged as an error so it's visible.
    logger.error(
        "   ❌ Both LLM classification attempts failed for query='%s…' — defaulting to QA",
        query[:60],
    )
    return {"query": query, "mode": "QA"}


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