"""
rag/pipeline/reranker_rag.py

Reranker shim for CiteRagLab.

Ranking is handled by Milvus AUTOINDEX + COSINE similarity — results arrive
already ordered by score before this module is called.  No cross-encoder,
no sentence-transformers, no extra inference step.

This module is kept for API compatibility — pipeline_rag.py imports from it
— but it is a transparent pass-through that enforces the top_k cap and logs.

To add a cross-encoder reranker later, replace the body of rerank() here
without touching any other file.
"""

import logging

logger = logging.getLogger("rag.pipeline.reranker_rag")


def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    Return the top_k chunks from the already-RRF-ranked list.

    Milvus AUTOINDEX + COSINE has already ranked the results inside
    hybrid_search_chunks before this function is called.
    The list arriving here is therefore already ordered by score — this
    function simply enforces the top_k cap and logs the outcome.

    Parameters
    ──────────
    query  : user query string (kept for interface compatibility)
    chunks : score-ranked chunk list from hybrid_search_chunks via retrieve()
    top_k  : maximum number of chunks to return

    Returns
    ───────
    chunks[:top_k] — the top-ranked chunks from Milvus COSINE search
    """
    if not chunks:
        logger.warning("   ⚠️  rerank: received empty chunk list — returning []")
        return []

    result = chunks[:top_k]

    logger.info(
        "📐 rerank (RRF pass-through) — input=%d chunks  top_k=%d  "
        "returning=%d  top_score=%.4f",
        len(chunks),
        top_k,
        len(result),
        result[0].get("score", 0.0) if result else 0.0,
    )
    return result