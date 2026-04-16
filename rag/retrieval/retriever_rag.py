"""
rag/retrieval/retriever_rag.py

Retrieval engine for CiteRagLab.

Responsibilities
────────────────
  1. Embed the user query using Azure OpenAI text-embedding-3-large (3072-dim).
     Uses identical Azure config as embedder_rag.py so chunks and queries
     live in the same vector space — cosine similarity scores are valid.
  2. Pass both the dense vector AND the raw query text to hybrid_search_chunks
     in milvus_client_rag, which runs dense-only AUTOINDEX + COSINE (milvus-lite).
  3. Return ranked chunk dicts to the pipeline.

Azure env variables required  (must match embedder_rag.py exactly)
───────────────────────────────────────────────────────────────────
    AZURE_OPENAI_EMB_KEY   — Azure OpenAI API key for embeddings
    AZURE_EMB_ENDPOINT     — Azure OpenAI endpoint
    AZURE_EMB_API_VERSION  — API version (2024-12-01-preview)
    AZURE_EMB_DEPLOYMENT   — deployment name (text-embedding-3-large)

Each returned chunk dict contains:
    chunk_text, doc_id, title, section,
    industry, doc_type, version, tags (list[str]),
    page_id, block_range, score
"""

import os
import logging
from typing import Optional
from openai import AzureOpenAI
from dotenv import load_dotenv
from rag.retrieval.milvus_client_rag import hybrid_search_chunks

load_dotenv()

logger = logging.getLogger("rag.retrieval.retriever_rag")

# ── Azure deployment config — must match embedder_rag.py exactly ─────────────
EMBED_MODEL = os.getenv("AZURE_EMB_DEPLOYMENT", "text-embedding-3-large")
EMBED_DIM   = 3072   # text-embedding-3-large output dimension

# ── Lazy Azure OpenAI client ──────────────────────────────────────────────────
_azure_client: Optional[AzureOpenAI] = None


def _get_azure_client() -> AzureOpenAI:
    global _azure_client
    if _azure_client is None:
        api_key     = os.getenv("AZURE_OPENAI_EMB_KEY")
        endpoint    = os.getenv("AZURE_EMB_ENDPOINT")
        api_version = os.getenv("AZURE_EMB_API_VERSION", "2024-12-01-preview")

        if not api_key:
            raise ValueError("AZURE_OPENAI_EMB_KEY is not set in environment / .env")
        if not endpoint:
            raise ValueError("AZURE_EMB_ENDPOINT is not set in environment / .env")

        _azure_client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        logger.info(
            "✅ Azure OpenAI retriever client initialised "
            "(deployment=%s, api_version=%s)",
            EMBED_MODEL, api_version,
        )
    return _azure_client


def embed_text(text: str) -> list[float]:
    """
    Embed a single string using Azure OpenAI text-embedding-3-large.
    Newlines replaced with spaces per OpenAI recommendation.
    Returns a 3072-dimensional float vector.
    """
    client = _get_azure_client()
    logger.info(
        "🔢 embed_text — %d chars, deployment='%s'",
        len(text), EMBED_MODEL,
    )
    response  = client.embeddings.create(
        model=EMBED_MODEL,
        input=text.replace("\n", " "),
    )
    embedding = response.data[0].embedding
    logger.info("   ✅ Query embedding produced — dim=%d", len(embedding))
    return embedding


def retrieve(
    query: str,
    top_k: int = 5,
    filters: Optional[dict] = None,
) -> list[dict]:
    """
    Main retrieval function — dense vector search with COSINE similarity.

    Steps:
      1. Embed the query with Azure OpenAI text-embedding-3-large.
      2. Call hybrid_search_chunks with the dense vector and the raw query text.
         Milvus runs AUTOINDEX + COSINE (milvus-lite: dense only).
      3. Return top_k ranked chunk dicts.

    filters supported keys: industry, doc_type, version, tags
    (pre-validated by filters_rag.build_filters)
    """
    

    logger.info(
        "📥 retrieve — query='%s…'  top_k=%d  filters=%s",
        query[:60], top_k, filters or {},
    )

    query_embedding = embed_text(query)

    chunks = hybrid_search_chunks(
        query_embedding=query_embedding,
        query_text=query,
        top_k=top_k,
        filters=filters,
    )

    logger.info(
        "   ✅ retrieve — %d chunks  avg_score=%.4f",
        len(chunks),
        sum(c.get("score", 0) for c in chunks) / max(len(chunks), 1),
    )
    return chunks


def format_context_for_prompt(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered context block for the LLM.

    Format per chunk:
        [N] <title> → <section>  (<doc_type>  v<version>  tags: <tags>  score: <score>)
        <chunk_text>
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

        location   = f"{title} → {section}" if section and section != title else title
        meta_parts = []
        if doc_type:
            meta_parts.append(doc_type)
        if version:
            meta_parts.append(f"v{version}")
        if tags:
            meta_parts.append(f"tags: {', '.join(tags)}")
        meta_parts.append(f"score: {score}")

        lines.append(f"[{i}] {location}  ({' '.join(meta_parts)})")
        lines.append(chunk.get("chunk_text", ""))
        lines.append("")

    logger.info(
        "   ✅ format_context_for_prompt — %d chunks → %d chars",
        len(chunks),
        sum(len(c.get("chunk_text", "")) for c in chunks),
    )
    return "\n".join(lines)