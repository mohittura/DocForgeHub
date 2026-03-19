"""
rag/tools/tools_rag.py

Tool functions for the CiteRagLab RAG system.

    search_documents  — semantic search returning ranked chunks
    refine_query      — rewrite a weak query for better retrieval
    compare_documents — fetch two chunk groups for side-by-side comparison
"""

import logging

from rag.retrieval.retriever_rag    import retrieve, format_context_for_prompt
from rag.pipeline.corrective_rag_rag import rewrite_query

logger = logging.getLogger("rag.tools.tools_rag")


def search_documents(
    query: str,
    top_k: int = 5,
    filters: dict | None = None,
) -> list[dict]:
    """
    Semantic search against the Milvus collection.

    Parameters
    ──────────
    query   : natural-language search query
    top_k   : number of chunks to return
    filters : optional {industry, doc_type, version}

    Returns
    ───────
    Ranked list of chunk dicts (chunk_text, title, section, score, …).
    """
    logger.info(
        "🔎 search_documents — query='%s…', top_k=%d, filters=%s",
        query[:60], top_k, filters or {},
    )
    chunks = retrieve(query=query, top_k=top_k, filters=filters)
    logger.info(
        "   ✅ search_documents — %d chunks returned (top score: %.4f)",
        len(chunks),
        chunks[0]["score"] if chunks else 0.0,
    )
    return chunks


def refine_query(query: str) -> str:
    """
    Rewrite a weak or ambiguous query into a retrieval-optimised form.
    Used internally by Corrective RAG; exposed as a standalone tool
    so the UI's Retrieval Inspector can call it directly.

    Returns the rewritten query string (or the original on failure).
    """
    logger.info("✏️  refine_query — rewriting: '%s…'", query[:60])
    rewritten = rewrite_query(query)
    logger.info("   ✅ refine_query — result: '%s…'", rewritten[:60])
    return rewritten


def compare_documents(
    query_a: str,
    query_b: str,
    top_k: int = 5,
    filters: dict | None = None,
) -> dict:
    """
    Retrieve two separate chunk sets and package them for LLM comparison.

    The caller (typically pipeline_rag.py in COMPARE mode) is responsible
    for passing the returned context strings to the LLM with
    COMPARE_SYSTEM_PROMPT.

    Parameters
    ──────────
    query_a : query for the first document group
    query_b : query for the second document group
    top_k   : chunks to retrieve per query
    filters : optional shared metadata filters

    Returns
    ───────
    {
        context_a : str          — formatted context for query_a
        context_b : str          — formatted context for query_b
        chunks_a  : list[dict]   — raw chunks for query_a
        chunks_b  : list[dict]   — raw chunks for query_b
    }
    """
    logger.info(
        "⚖️  compare_documents — "
        "query_a='%s…'  query_b='%s…'  top_k=%d",
        query_a[:40], query_b[:40], top_k,
    )

    chunks_a = retrieve(query_a, top_k=top_k, filters=filters)
    chunks_b = retrieve(query_b, top_k=top_k, filters=filters)

    logger.info(
        "   ✅ compare_documents — %d chunks (A), %d chunks (B)",
        len(chunks_a), len(chunks_b),
    )

    return {
        "context_a": format_context_for_prompt(chunks_a),
        "context_b": format_context_for_prompt(chunks_b),
        "chunks_a":  chunks_a,
        "chunks_b":  chunks_b,
    }