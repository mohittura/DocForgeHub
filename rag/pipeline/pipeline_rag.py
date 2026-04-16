"""
rag/pipeline/pipeline_rag.py

Main RAG pipeline for CiteRagLab.

Full flow
─────────
    User Query
      ↓
    Adaptive Router (LangGraph)   — LLM classifies mode: QA / COMPARE / SUMMARIZE / SEARCH
      ↓                             Falls back to keyword heuristics if LLM unavailable
    Corrective RAG (LangGraph)    — retrieve → score → rewrite if weak → re-retrieve
      ↓                             Each retrieve() runs HNSW + BM25 + RRF in Milvus
    rerank()                       — top_k cap on already-RRF-ranked list (pass-through)
      ↓
    Context Assembly               — numbered [N] citation format (includes tags)
      ↓
    LLM Answer (AzureChatOpenAI)  — grounded answer with inline [N] citations
      ↓
    Response dict                  — answer + citations + metadata

Azure env variables required
─────────────────────────────
    AZURE_OPENAI_LLM_KEY         — Azure OpenAI API key for the LLM
    AZURE_LLM_ENDPOINT           — Azure OpenAI endpoint
    AZURE_LLM_API_VERSION        — API version (default: 2024-12-01-preview)
    AZURE_LLM_DEPLOYMENT_41_MINI — deployment name (gpt-4.1-mini)
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from rag.pipeline.adaptive_router_rag import classify_query, get_retrieval_params
from rag.pipeline.corrective_rag_rag  import corrective_retrieve, avg_score
from rag.pipeline.reranker_rag        import rerank
from rag.pipeline.prompts_rag         import SYSTEM_PROMPT_BY_MODE, RAG_SYSTEM_PROMPT, GREETING_RESPONSE, OUT_OF_SCOPE_RESPONSE, OUT_OF_SCOPE_SCORE_THRESHOLD
from rag.retrieval.retriever_rag      import retrieve, format_context_for_prompt
from rag.retrieval.filters_rag        import build_filters
from langchain_core.messages import AIMessage

load_dotenv()

logger = logging.getLogger("rag.pipeline.pipeline_rag")

# ── Lazy Azure LLM client ─────────────────────────────────────────────────────
_llm: Optional[AzureChatOpenAI] = None


def _get_llm() -> AzureChatOpenAI:
    global _llm
    if _llm is None:
        _llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_LLM_DEPLOYMENT_41_MINI", "gpt-4.1-mini"),
            azure_endpoint=os.getenv("AZURE_LLM_ENDPOINT", ""),
            api_key=os.getenv("AZURE_OPENAI_LLM_KEY", ""),
            api_version=os.getenv("AZURE_LLM_API_VERSION", "2024-12-01-preview"),
            temperature=0.2,
            max_tokens=1024,
        )
        logger.info(
            "✅ Pipeline AzureChatOpenAI initialised (deployment=%s)",
            os.getenv("AZURE_LLM_DEPLOYMENT_41_MINI", "gpt-4.1-mini"),
        )
    return _llm


def run_rag_pipeline(
    query: str,
    session_history: list[dict] | None = None,
    raw_filters: dict | None = None,
) -> dict:
    """
    Execute the full RAG pipeline for one user query.

    Parameters
    ──────────
    query           : raw user query string
    session_history : prior chat turns [{role, content}] from Redis
    raw_filters     : optional filter dict — supported keys:
                        industry, doc_type, version, tags

    Returns
    ───────
    {
        answer    : str         — LLM answer with inline [N] citations
        citations : list[dict]  — {index, title, section, doc_type,
                                   industry, version, tags, page_id, score}
        chunks    : list[dict]  — raw retrieved chunks (for Redis caching)
        mode      : str         — QA | COMPARE | SUMMARIZE | SEARCH
        rewritten : str         — query used for retrieval
        avg_score : float       — mean RRF score of final chunks
    }
    """
    logger.info(
        "🚀 run_rag_pipeline — query='%s…'  filters=%s",
        query[:60], raw_filters or {},
    )

    # ── 1. Validate filters ───────────────────────────────────────────────────
    filters = build_filters(raw_filters or {})
    logger.info("   🔧 Filters: %s", filters or "(none)")

    # ── 2. Classify query mode (LangGraph router) ─────────────────────────────
    mode     = classify_query(query)
    params   = get_retrieval_params(mode)
    top_k    = params["top_k"]
    llm_mode = params["llm_mode"]
    logger.info("   🗂️  Mode=%s  top_k=%d  llm_mode=%s", mode, top_k, llm_mode)

    # ── 2a. Short-circuit for greetings / identity queries ────────────────────
    # No retrieval, no LLM call — return the hardcoded identity card instantly.
    if mode == "GREETING":
        logger.info("   👋 GREETING mode — returning identity response, skipping pipeline")
        return {
            "answer":    GREETING_RESPONSE,
            "citations": [],
            "chunks":    [],
            "mode":      "GREETING",
            "rewritten": query,
            "avg_score": 0.0,
        }

    # ── 3. Corrective retrieval (LangGraph graph) ─────────────────────────────
    # retrieve() runs HNSW + BM25 + RRFRanker inside Milvus — already fused.
    # pool_size = 2 × top_k gives corrective RAG enough candidates.
    pool_size = top_k * 2

    def _retrieve_fn(q: str, k: int, f: dict | None) -> list[dict]:
        return retrieve(q, top_k=pool_size, filters=f)

    chunks, rewritten_query = corrective_retrieve(
        query=query,
        retrieve_fn=_retrieve_fn,
        top_k=top_k,
        filters=filters or None,
        session_history=session_history or [],
    )
    logger.info(
        "   📥 Corrective retrieval done — %d chunks  rewritten=%s",
        len(chunks), rewritten_query != query,
    )

    # ── 4. Enforce top_k cap (RRF already ranked; rerank is a pass-through) ───
    chunks      = rerank(query=rewritten_query, chunks=chunks, top_k=top_k)
    final_score = avg_score(chunks)
    logger.info(
        "   📐 After top_k cap — %d chunks  avg_score=%.4f",
        len(chunks), final_score,
    )

    # ── 4b. Score gate — if best chunks are too weak, the topic is not in the
    #        library. Return a clean response without calling the LLM.
    if final_score < OUT_OF_SCOPE_SCORE_THRESHOLD:
        logger.info(
            "   🚫 Score gate triggered — avg_score=%.4f < threshold=%.2f — "
            "topic not covered in library",
            final_score, OUT_OF_SCOPE_SCORE_THRESHOLD,
        )
        return {
            "answer":    OUT_OF_SCOPE_RESPONSE,
            "citations": [],
            "chunks":    [],
            "mode":      mode,
            "rewritten": rewritten_query,
            "avg_score": round(final_score, 4),
        }

    # ── 5. Build prompt context ────────────────────────────────────────────────
    context = format_context_for_prompt(chunks)
    logger.info("   📝 Context assembled — %d chars", len(context))

    # ── 6. LLM answer (AzureChatOpenAI via LangChain) ────────────────────────
    system_prompt = SYSTEM_PROMPT_BY_MODE.get(llm_mode, RAG_SYSTEM_PROMPT)

    # Build a stripped-down history that only carries user questions,
    # NOT prior assistant answers.
    #
    # Why: prior assistant answers contain content from old topics. When the
    # user shifts topic mid-session the LLM anchors on those old AI turns and
    # hallucinates instead of reading the freshly retrieved context blocks.
    # Keeping only user turns preserves conversational thread awareness
    # (so the LLM understands references like "what about X instead?") without
    # poisoning the answer with stale AI-generated content.
    #
    # The retrieved context blocks in `user_content` are the ONLY source of
    # factual truth — the system prompt enforces this explicitly.
    lc_messages = [SystemMessage(content=system_prompt)]
    if session_history:
        
        prior_turns = session_history[-6:]   # last 3 exchanges max
        for turn in prior_turns:
            role    = turn.get("role", "user")
            content = turn.get("content", "")
            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            else:
                # Replace the prior assistant answer with a neutral placeholder.
                # This keeps the turn-pair structure intact (so the LLM knows
                # a response was given) without injecting old factual content
                # that could contaminate the current answer.
                lc_messages.append(AIMessage(content="[Previous answer — see retrieved context below for current facts.]"))

    user_content = (
        f"Retrieved context (authoritative source — answer from this only):\n"
        f"{context}\n\n"
        f"Question: {rewritten_query}"
    )
    lc_messages.append(HumanMessage(content=user_content))

    logger.info(
        "   🤖 Calling AzureChatOpenAI (deployment=%s) — %d messages",
        os.getenv("AZURE_LLM_DEPLOYMENT_41_MINI", "gpt-4.1-mini"),
        len(lc_messages),
    )

    answer = ""
    try:
        response = _get_llm().invoke(lc_messages)
        answer   = response.content.strip()
        logger.info("   ✅ LLM response received — %d chars", len(answer))
    except Exception as err:
        logger.error("   ❌ LLM call failed: %s", err)
        answer = f"⚠️ I encountered an error generating a response: {err}"

    # ── 7. Build citation list ─────────────────────────────────────────────────
    citations = []
    for i, chunk in enumerate(chunks, start=1):
        citations.append({
            "index":    i,
            "title":    chunk.get("title",    ""),
            "section":  chunk.get("section",  ""),
            "doc_type": chunk.get("doc_type", ""),
            "industry": chunk.get("industry", ""),
            "version":  chunk.get("version",  ""),
            "tags":     chunk.get("tags",     []),
            "page_id":  chunk.get("page_id",  ""),
            "score":    chunk.get("score",    0.0),
            "chunk_text": chunk.get("chunk_text", ""),
        })
    logger.info("   📚 Citations built — %d entries", len(citations))

    result = {
        "answer":    answer,
        "citations": citations,
        "chunks":    chunks,
        "mode":      mode,
        "rewritten": rewritten_query,
        "avg_score": round(final_score, 4),
    }
    logger.info(
        "✅ run_rag_pipeline complete — mode=%s  chunks=%d  avg_score=%.4f  answer=%d chars",
        mode, len(chunks), final_score, len(answer),
    )
    return result