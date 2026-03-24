"""
rag/ingestion/ingestion_pipeline_rag.py

Ingestion pipeline for CiteRagLab.

Full flow per page:
    Notion Database row
        → get_page_blocks()    (rate-limited Notion API calls)
        → chunk_page()         (token-aware overlapping chunker)
        → embed_chunks()       (Azure OpenAI text-embedding-3-large, batched)
        → insert_chunks()      (Milvus AUTOINDEX + COSINE collection)

Rate limiting
─────────────
  Individual Notion API calls are already throttled inside
  notion_loader_rag._notion_call() (REQUEST_DELAY_SEC + 429 back-off).

  Additionally, this pipeline adds a INTER_PAGE_DELAY_SEC pause between
  ingesting consecutive pages so that the cumulative rate — across all the
  blocks.children.list calls a page fetch triggers — stays well under
  the 3 req/s sustained limit when processing many pages in a row.

Usage
─────
    from rag.ingestion.ingestion_pipeline_rag import ingest_all_pages, ingest_page

    # Full run — all pages in the configured Notion database
    summary = ingest_all_pages()

    # Single-page upsert (triggered from the UI Ingest tab)
    n_chunks = ingest_page({
        "page_id":  "32689db1...",
        "title":    "Offer Letter Templates",
        "doc_type": "Offer Letter Templates",
        "industry": "Human Resources (HR)",
        "version":  "1.0",
        "tags":     ["Offer Letter Templates"],
    })
"""

import time
import logging

from rag.ingestion.notion_loader_rag  import get_all_pages, get_page_blocks
from rag.ingestion.chunker_rag        import chunk_page
from rag.ingestion.embedder_rag       import embed_chunks
from rag.retrieval.milvus_client_rag  import insert_chunks

logger = logging.getLogger("rag.ingestion.ingestion_pipeline_rag")

# Extra courtesy pause between pages so the accumulated Notion call rate
# stays comfortably under 3 req/s across a long multi-page run.
INTER_PAGE_DELAY_SEC = 0.5


def ingest_page(page_meta: dict) -> int:
    """
    Ingest a single Notion database page end-to-end.

    page_meta keys:
        page_id  (required)
        title    (str, default "(untitled)")
        doc_type (str, default "Document")
        industry (str, default "General")
        version  (str, default "1.0")
        tags     (list[str], optional — stored as doc_type suffix if present)

    Returns the number of chunks inserted into Milvus (0 if page is empty).
    Every stage is logged so failures are easy to trace.
    """
    page_id  = page_meta["page_id"]
    title    = page_meta.get("title",    "(untitled)")
    doc_type = page_meta.get("doc_type", "Document")
    industry = page_meta.get("industry", "General")
    version  = page_meta.get("version",  "1.0")
    tags     = page_meta.get("tags",     [])

    logger.info(
        "🚀 ingest_page — title='%s'  page_id=%s  doc_type=%s  industry=%s  version=%s  tags=%s",
        title, page_id, doc_type, industry, version, tags,
    )

    # ── Stage 1: Fetch and extract all text blocks from Notion ───────────────
    blocks = get_page_blocks(page_id)
    if not blocks:
        logger.warning(
            "   ⚠️  Stage 1: no text blocks found for page '%s' (%s) — skipping",
            title, page_id,
        )
        return 0
    logger.info("   📄 Stage 1 done — %d blocks extracted", len(blocks))

    # ── Stage 2: Chunk ────────────────────────────────────────────────────────
    chunks = chunk_page(
        page_id=page_id,
        title=title,
        blocks=blocks,
        doc_type=doc_type,
        industry=industry,
        version=version,
    )
    if not chunks:
        logger.warning(
            "   ⚠️  Stage 2: chunking produced 0 chunks for page '%s' — skipping",
            title,
        )
        return 0
    logger.info("   ✂️  Stage 2 done — %d chunks produced", len(chunks))

    # ── Stage 3: Embed ────────────────────────────────────────────────────────
    embedded_chunks = embed_chunks(chunks)
    logger.info("   🔢 Stage 3 done — %d chunks embedded", len(embedded_chunks))

    # ── Stage 4: Insert into Milvus ───────────────────────────────────────────
    inserted_count = insert_chunks(embedded_chunks)
    logger.info(
        "   ✅ ingest_page complete — page='%s'  chunks_inserted=%d",
        title, inserted_count,
    )
    return inserted_count


def ingest_all_pages(database_id: str | None = None) -> dict:
    """
    Discover and ingest every page in the Notion database.

    Parameters
    ──────────
    database_id : optional override for NOTION_ROOT_PAGE_ID env variable.
                  The root page ID in the Notion URL is the database ID.

    Returns a summary dict:
        {
            pages_processed : int,
            chunks_inserted : int,
            pages_skipped   : int,   # pages that had no extractable text
            errors          : list[str],
        }

    Logs progress at every stage.  On error for a single page the pipeline
    continues with the remaining pages — errors are collected and returned
    in the summary rather than aborting the whole run.
    """
    logger.info("🏁 ingest_all_pages — starting full ingest run")

    pages = get_all_pages(database_id)
    total_pages = len(pages)
    logger.info("   📋 %d pages found in database — beginning ingest loop", total_pages)

    total_chunks_inserted = 0
    pages_skipped         = 0
    errors: list[str]     = []

    for page_number, page_meta in enumerate(pages, start=1):
        page_title = page_meta.get("title", page_meta.get("page_id", "?"))
        logger.info(
            "   [%d/%d] Processing page '%s'…",
            page_number, total_pages, page_title,
        )

        try:
            chunks_inserted = ingest_page(page_meta)
            if chunks_inserted == 0:
                pages_skipped += 1
            else:
                total_chunks_inserted += chunks_inserted

        except Exception as err:
            error_message = (
                f"Error ingesting page '{page_title}' "
                f"(page_id={page_meta.get('page_id', '')}): {err}"
            )
            logger.error("   ❌ %s", error_message)
            errors.append(error_message)

        # Courtesy delay between pages — keeps cumulative Notion request rate
        # comfortably under 3 req/s across many consecutive page fetches.
        if page_number < total_pages:
            time.sleep(INTER_PAGE_DELAY_SEC)

    summary = {
        "pages_processed": total_pages,
        "chunks_inserted": total_chunks_inserted,
        "pages_skipped":   pages_skipped,
        "errors":          errors,
    }
    logger.info(
        "✅ ingest_all_pages complete — "
        "pages=%d  chunks=%d  skipped=%d  errors=%d",
        summary["pages_processed"],
        summary["chunks_inserted"],
        summary["pages_skipped"],
        len(summary["errors"]),
    )
    return summary