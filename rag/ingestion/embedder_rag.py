"""
rag/ingestion/embedder_rag.py

Embedding layer for CiteRagLab.

Wraps Azure OpenAI text-embedding-3-large with:
  - Configurable batch size (default 32)
  - One retry with 2-second sleep on any API error
  - In-place mutation: adds an 'embedding' field to each chunk dict

Azure env variables required
─────────────────────────────
    AZURE_OPENAI_EMB_KEY   — Azure OpenAI API key for embeddings
    AZURE_EMB_ENDPOINT     — Azure OpenAI endpoint (e.g. https://xxx.openai.azure.com/)
    AZURE_EMB_API_VERSION  — API version (2024-12-01-preview)
    AZURE_EMB_DEPLOYMENT   — deployment name (text-embedding-3-large)
"""

import os
import time
import logging
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("rag.ingestion.embedder_rag")

EMBED_MODEL = os.getenv("AZURE_EMB_DEPLOYMENT", "text-embedding-3-large")
BATCH_SIZE  = 32   # conservative — Azure allows up to 2048 inputs per request

_client: AzureOpenAI | None = None


def _get_client() -> AzureOpenAI:
    global _client
    if _client is None:
        api_key     = os.getenv("AZURE_OPENAI_EMB_KEY")
        endpoint    = os.getenv("AZURE_EMB_ENDPOINT")
        api_version = os.getenv("AZURE_EMB_API_VERSION", "2024-12-01-preview")

        if not api_key:
            raise ValueError("AZURE_OPENAI_EMB_KEY is not set in environment / .env")
        if not endpoint:
            raise ValueError("AZURE_EMB_ENDPOINT is not set in environment / .env")

        _client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version,
        )
        logger.info(
            "✅ Azure OpenAI embedder client initialised "
            "(deployment=%s, api_version=%s)",
            EMBED_MODEL, api_version,
        )
    return _client


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Add an 'embedding' key (list[float], dim=3072) to each chunk dict in-place.

    Processes chunks in batches of BATCH_SIZE.
    Retries once on any API error before propagating.

    Returns the same list with embeddings attached.
    """
    if not chunks:
        logger.warning("   ⚠️  embed_chunks: received empty list — skipping")
        return chunks

    logger.info(
        "🔢 embed_chunks — %d chunks to embed (batch_size=%d, deployment=%s)…",
        len(chunks), BATCH_SIZE, EMBED_MODEL,
    )

    texts      = [c["chunk_text"].replace("\n", " ") for c in chunks]
    embeddings = _batch_embed(texts)

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb

    logger.info(
        "   ✅ embed_chunks complete — %d embeddings attached (dim=%d)",
        len(chunks),
        len(embeddings[0]) if embeddings else 0,
    )
    return chunks


def _batch_embed(texts: list[str]) -> list[list[float]]:
    """
    Send texts in batches of BATCH_SIZE to Azure OpenAI.
    Returns a flat list of 3072-dim embedding vectors in input order.
    """
    all_embeddings: list[list[float]] = []
    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx, start in enumerate(range(0, len(texts), BATCH_SIZE)):
        batch = texts[start : start + BATCH_SIZE]
        logger.info(
            "   📦 Embedding batch %d/%d — %d texts",
            batch_idx + 1, total_batches, len(batch),
        )

        for attempt in range(2):
            try:
                resp = _get_client().embeddings.create(
                    model=EMBED_MODEL,
                    input=batch,
                )
                # Azure returns results ordered by index field
                batch_embs = [
                    item.embedding
                    for item in sorted(resp.data, key=lambda x: x.index)
                ]
                all_embeddings.extend(batch_embs)
                logger.info(
                    "      ✅ Batch %d embedded — %d vectors (dim=%d)",
                    batch_idx + 1, len(batch_embs),
                    len(batch_embs[0]) if batch_embs else 0,
                )
                break

            except Exception as err:
                if attempt == 0:
                    logger.warning(
                        "      ⚠️  Embed error on batch %d (attempt 1) — retrying in 2 s: %s",
                        batch_idx + 1, err,
                    )
                    time.sleep(2)
                else:
                    logger.error(
                        "      ❌ Embed failed on batch %d after retry: %s",
                        batch_idx + 1, err,
                    )
                    raise

    return all_embeddings