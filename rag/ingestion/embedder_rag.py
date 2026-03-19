"""
rag/ingestion/embedder_rag.py

Embedding layer for CiteRagLab.

Wraps OpenAI text-embedding-3-small with:
  - Configurable batch size (default 32)
  - One retry with 2-second sleep on any API error
  - In-place mutation: adds an 'embedding' field to each chunk dict
"""

import os
import time
import logging
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("rag.ingestion.embedder_rag")

EMBED_MODEL = "text-embedding-3-large"
BATCH_SIZE  = 32   # conservative — OpenAI allows up to 2048 inputs per request

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("AZURE_OPENAI_EMB_KEY")
        if not api_key:
            raise ValueError("AZURE_OPENAI_EMB_KEY is not set in environment / .env")
        _client = OpenAI(api_key=api_key)
        logger.info("✅ OpenAI embedder client initialised (model=%s, batch_size=%d)", EMBED_MODEL, BATCH_SIZE)
    return _client


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Add an 'embedding' key (list[float]) to each chunk dict in-place.

    Processes chunks in batches of BATCH_SIZE.
    Retries once on any API error before propagating.

    Returns the same list with embeddings attached.
    """
    if not chunks:
        logger.warning("   ⚠️  embed_chunks: received empty list — skipping")
        return chunks

    logger.info(
        "🔢 embed_chunks — %d chunks to embed (batch_size=%d)…",
        len(chunks), BATCH_SIZE,
    )

    texts = [c["chunk_text"].replace("\n", " ") for c in chunks]
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
    Send texts in batches of BATCH_SIZE to OpenAI.
    Returns a flat list of embedding vectors in the same order as input.
    """
    client = _get_client()
    all_embeddings: list[list[float]] = []

    total_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx, start in enumerate(range(0, len(texts), BATCH_SIZE)):
        batch = texts[start : start + BATCH_SIZE]
        logger.info(
            "   📦 Embedding batch %d/%d — %d texts (chars: %d–%d)",
            batch_idx + 1, total_batches,
            len(batch),
            start, start + len(batch) - 1,
        )

        for attempt in range(2):
            try:
                resp = _get_client().embeddings.create(
                    model=EMBED_MODEL,
                    input=batch,
                )
                # Results come back ordered by index field
                batch_embs = [
                    item.embedding
                    for item in sorted(resp.data, key=lambda x: x.index)
                ]
                all_embeddings.extend(batch_embs)
                logger.info(
                    "      ✅ Batch %d embedded successfully (%d vectors)",
                    batch_idx + 1, len(batch_embs),
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