"""
rag/pipeline/pipeline_rag.py

Main RAG pipeline for CiteRagLab.

Full flow
─────────
    User Query
      ↓
    Adaptive Router          classify mode (QA / COMPARE / SUMMARIZE / SEARCH)
      ↓
    Corrective RAG           retrieve → score → rewrite if weak → re-retrieve
      ↓
    Cross-Encoder Rerank     pool (4 × top_k) → final top_k
      ↓
    Context Assembly         numbered [N] citation format (includes tags)
      ↓
    LLM (gpt-4o-mini)        grounded answer with inline [N] citations
      ↓
    Response dict            answer + citations + metadata

Filters supported (matching DocForge Hub Library Notion database columns)
─────────────────────────────────────────────────────────────────────────
    industry  — Industry select column   (exact match)
    doc_type  — Type select column       (exact match)
    version   — Version rich_text column (exact match)
    tags      — tags multi_select column (substring match on comma-joined string)
"""

import os
import logging
from openai import OpenAI
from dotenv import load_dotenv

from rag.pipeline.adaptive_router_rag import classify_query, get_retrieval_params
from rag.pipeline.corrective_rag_rag  import corrective_retrieve, avg_score
from rag.pipeline.reranker_rag        import rerank
from rag.pipeline.prompts_rag         import build_rag_messages
from rag.retrieval.retriever_rag      import retrieve, format_context_for_prompt
from rag.retrieval.filters_rag        import build_filters

load_dotenv()

logger = logging.getLogger("rag.pipeline.pipeline_rag")

LLM_MODEL = "gpt-4o-mini"

_openai_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment / .env")
        _openai_client = OpenAI(api_key=api_key)
        logger.info("✅ Pipeline OpenAI client initialised (model=%s)", LLM_MODEL)
    return _openai_client


def run_rag_pipeline(
    query: str,
    session_history: list[dict] | None = None,
    raw_filters: dict | None = None,
) -> dict:
    """
    Execute the full RAG pipeline for one user query.

    Parameters
    ──────────
    query           : the raw user query string
    session_history : prior chat turns [{role, content}] from Redis
    raw_filters     : optional filter dict — supported keys:
                        industry, doc_type, version, tags

    Returns
    ───────
    {
        answer    : str         — LLM answer with inline [N] citations
        citations : list[dict]  — one dict per retrieved chunk:
                                  {index, title, section, doc_type,
                                   industry, version, tags, page_id, score}
        chunks    : list[dict]  — raw retrieved chunks (used for Redis caching)
        mode      : str         — QA | COMPARE | SUMMARIZE | SEARCH
        rewritten : str         — query used for retrieval (may differ from input)
        avg_score : float       — mean cosine similarity of final chunks
    }
    """
    logger.info(
        "🚀 run_rag_pipeline — query='%s…'  filters=%s",
        query[:60],
        raw_filters or {},
    )

    # ── 1. Validate and clean filters ────────────────────────────────────────
    filters = build_filters(raw_filters or {})
    logger.info("   🔧 Filters after validation: %s", filters or "(none)")

    # ── 2. Classify query mode ────────────────────────────────────────────────
    mode     = classify_query(query)
    params   = get_retrieval_params(mode)
    top_k    = params["top_k"]
    llm_mode = params["llm_mode"]
    logger.info("   🗂️  Mode=%s  top_k=%d  llm_mode=%s", mode, top_k, llm_mode)

    # ── 3. Corrective retrieval (fetch 4 × top_k pool for the reranker) ──────
    pool_size = top_k * 4

    def _retrieve_fn(q: str, k: int, f: dict | None) -> list[dict]:
        """Inner wrapper — corrective_retrieve calls this with its own args."""
        return retrieve(q, top_k=pool_size, filters=f)

    chunks, rewritten_query = corrective_retrieve(
        query=query,
        retrieve_fn=_retrieve_fn,
        top_k=top_k,
        filters=filters or None,
    )
    logger.info(
        "   📥 Corrective retrieval done — %d chunks in pool  rewritten=%s",
        len(chunks),
        rewritten_query != query,
    )

    # ── 4. Cross-encoder rerank: pool → final top_k ───────────────────────────
    chunks      = rerank(query=rewritten_query, chunks=chunks, top_k=top_k)
    final_score = avg_score(chunks)
    logger.info(
        "   📐 After rerank — %d chunks kept  avg_score=%.4f",
        len(chunks), final_score,
    )

    # ── 5. Build prompt context ───────────────────────────────────────────────
    context = format_context_for_prompt(chunks)
    logger.info("   📝 Context assembled — %d chars", len(context))

    # ── 6. LLM call ───────────────────────────────────────────────────────────
    messages = build_rag_messages(
        query=rewritten_query,
        context=context,
        chat_history=session_history,
        mode=llm_mode,
    )
    logger.info(
        "   🤖 Calling LLM '%s' — %d messages in prompt",
        LLM_MODEL, len(messages),
    )

    answer = ""
    try:
        client = _get_client()
        resp   = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=1024,
            temperature=0.2,
        )
        answer = resp.choices[0].message.content.strip()
        logger.info("   ✅ LLM response received — %d chars", len(answer))
    except Exception as err:
        logger.error("   ❌ LLM call failed: %s", err)
        answer = f"⚠️ I encountered an error generating a response: {err}"

    # ── 7. Build citation list ─────────────────────────────────────────────────
    # tags comes back from search_chunks as list[str] — pass it through as-is
    # so the UI can display it and the LLM prompt already contains it via
    # format_context_for_prompt().
    citations = []
    for i, chunk in enumerate(chunks, start=1):
        citations.append({
            "index":    i,
            "title":    chunk.get("title",    ""),
            "section":  chunk.get("section",  ""),
            "doc_type": chunk.get("doc_type", ""),
            "industry": chunk.get("industry", ""),
            "version":  chunk.get("version",  ""),
            "tags":     chunk.get("tags",     []),   # list[str] from multi_select
            "page_id":  chunk.get("page_id",  ""),
            "score":    chunk.get("score",    0.0),
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