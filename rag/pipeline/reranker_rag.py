"""
rag/pipeline/reranker_rag.py

Reranker shim for CiteRagLab.

Reranking is now handled natively inside Milvus using RRFRanker, which
fuses the HNSW dense ranking and the BM25 sparse ranking before results
ever leave the database.  No cross-encoder model, no sentence-transformers
dependency, no extra inference step.

This module is kept for API compatibility — pipeline_rag.py still imports
from it — but it is now a transparent pass-through that simply slices the
already-fused list to top_k and logs what it received.

If you later want to swap RRF for a cross-encoder (e.g. for a domain that
benefits from deep semantic reranking), replace the body of rerank() here
without touching any other file.
"""

import logging

logger = logging.getLogger("rag.pipeline.reranker_rag")


def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    Return the top_k chunks from the already-RRF-ranked list.

    Milvus's RRFRanker has already fused the dense (HNSW) and sparse (BM25)
    rankings inside hybrid_search_chunks before this function is called.
    The list arriving here is therefore already optimally ordered — this
    function simply enforces the top_k cap and logs the outcome.

    Parameters
    ──────────
    query  : user query string (kept for interface compatibility)
    chunks : RRF-ranked chunk list from hybrid_search_chunks via retrieve()
    top_k  : maximum number of chunks to return

    Returns
    ───────
    chunks[:top_k] — the top-ranked chunks after Milvus RRF fusion
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