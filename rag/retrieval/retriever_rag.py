"""
rag/retrieval/retriever_rag.py

Retrieval engine for CiteRagLab.

Responsibilities:
  1. Embed a user query using text-embedding-3-large.
  2. Run a vector search against the Milvus collection with optional
     metadata filters (industry, doc_type, version, tags).
  3. Return ranked chunk dicts ready for the RAG pipeline.

Each returned chunk dict contains:
    chunk_text, doc_id, title, section,
    industry, doc_type, version, tags,   ← tags is list[str]
    page_id, block_range, score
"""

import os
import logging
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("rag.retrieval.retriever_rag")

EMBED_MODEL = "text-embedding-3-large"

# ── Lazy OpenAI client ────────────────────────────────────────────────────────
_openai_client: Optional[OpenAI] = None


def _get_openai_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        api_key = os.getenv("AZURE_OPENAI_EMB_KEY")
        if not api_key:
            raise ValueError("AZURE_OPENAI_EMB_KEY is not set in environment / .env")
        _openai_client = OpenAI(api_key=api_key)
        logger.info("✅ OpenAI client initialised (model=%s)", EMBED_MODEL)
    return _openai_client


def embed_text(text: str) -> list[float]:
    """
    Embed a single string using OpenAI text-embedding-3-small.
    Newlines are replaced with spaces per OpenAI's recommendation.
    """
    client = _get_openai_client()
    logger.info("🔢 Embedding query (%d chars)…", len(text))

    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text.replace("\n", " "),
    )
    embedding = response.data[0].embedding
    logger.info("   ✅ Embedding produced — dim=%d", len(embedding))
    return embedding


def retrieve(
    query: str,
    top_k: int = 5,
    filters: Optional[dict] = None,
) -> list[dict]:
    """
    Main retrieval function.

    Steps:
      1. Embed the query via OpenAI text-embedding-3-small.
      2. Search the Milvus collection with optional metadata filters.
      3. Return top_k ranked chunk dicts.

    filters supported keys: industry, doc_type, version, tags
    (validated/normalised by filters_rag.build_filters before calling here)

    Each returned dict:
        chunk_text, doc_id, title, section,
        industry, doc_type, version, tags (list[str]),
        page_id, block_range, score
    """
    from rag.retrieval.milvus_client_rag import search_chunks

    logger.info(
        "📥 retrieve — query='%s…'  top_k=%d  filters=%s",
        query[:60], top_k, filters or {},
    )

    query_embedding = embed_text(query)
    chunks = search_chunks(
        query_embedding=query_embedding,
        top_k=top_k,
        filters=filters,
    )

    logger.info(
        "   ✅ retrieve — %d chunks returned  avg_score=%.4f",
        len(chunks),
        sum(c.get("score", 0) for c in chunks) / max(len(chunks), 1),
    )
    return chunks


def format_context_for_prompt(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered context block ready to be
    injected into the LLM system prompt.

    Each entry is prefixed with a [N] citation number that the LLM uses
    inline (e.g. "…as stated in the policy [1]…").

    Format per chunk:
        [N] <title> → <section>  (<doc_type>  v<version>  tags: <tags>)
        <chunk_text>

    Example:
        [1] Offer Letter Templates → Position Details (Offer Letter Templates  v1.0  tags: HR)
        Field | Details
        Job Title | Senior AI Systems Engineer – Defense Intelligence Platforms
        ...
    """
    if not chunks:
        logger.warning("   ⚠️  format_context_for_prompt: no chunks — returning fallback")
        return "No relevant documents found."

    lines = []
    for i, chunk in enumerate(chunks, start=1):
        title    = chunk.get("title",    "Unknown")
        section  = chunk.get("section",  "")
        doc_type = chunk.get("doc_type", "")
        version  = chunk.get("version",  "")
        tags     = chunk.get("tags",     [])
        score    = chunk.get("score",    0)

        # Build citation header line
        location    = f"{title} → {section}" if section and section != title else title
        meta_parts  = []
        if doc_type:
            meta_parts.append(doc_type)
        if version:
            meta_parts.append(f"v{version}")
        if tags:
            meta_parts.append(f"tags: {', '.join(tags)}")
        meta_parts.append(f"score: {score}")
        meta_str = "  ".join(meta_parts)

        lines.append(f"[{i}] {location}  ({meta_str})")
        lines.append(chunk.get("chunk_text", ""))
        lines.append("")   # blank separator between chunks

    logger.info(
        "   ✅ format_context_for_prompt — %d chunks → %d chars",
        len(chunks),
        sum(len(c.get("chunk_text", "")) for c in chunks),
    )
    return "\n".join(lines)