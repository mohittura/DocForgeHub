"""
rag/ingestion/notion_loader_rag.py

Notion loader for CiteRagLab.

Data layout (DocForge Hub Library database)
───────────────────────────────────────────
  Root URL : https://www.notion.so/31489db15e5b804c9049d062c6cdce54
  Type     : Notion DATABASE — each row is one document page.
  Columns  : Title, Type (select), Industry (select),
             Version (rich_text), tags (multi_select),
             Created by, Created time

  Each row is a full Notion page whose body contains:
    - Headings (heading_1/2/3)
    - Paragraphs, bullets, numbered lists
    - Toggles, callouts, quotes
    - Code blocks (with language label)
    - Tables  (table block → table_row children)
    - Nested child blocks under any of the above

Public API
──────────
    get_all_pages(database_id)  — query every row in the database
    get_page_blocks(page_id)    — recursively extract all text blocks

Rate limiting  (Notion: ~3 req/s sustained, search: 30 req/min)
─────────────
    1. REQUEST_DELAY_SEC sleep after every successful API call.
    2. Exponential back-off on HTTP 429, honouring Retry-After header.
    3. All API calls go through _notion_call() which centralises both.
"""

import os
import time
import logging
from typing import Optional
from notion_client import Client
from notion_client.errors import APIResponseError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("rag.ingestion.notion_loader_rag")

# ── Rate-limit constants ──────────────────────────────────────────────────────
# 0.35 s between calls ≈ 2.85 req/s — safely under the 3 req/s hard limit.
REQUEST_DELAY_SEC = 0.35
MAX_RETRIES       = 6          # max retries on a single 429 before giving up
BACKOFF_BASE_SEC  = 2.0        # exponential back-off base
MAX_BACKOFF_SEC   = 64.0       # cap on back-off wait

# Maximum recursion depth for nested child blocks (prevents infinite loops)
MAX_BLOCK_DEPTH   = 5

# ── Notion client — lazy singleton ────────────────────────────────────────────
_notion_client: Optional[Client] = None


def _get_client() -> Client:
    global _notion_client
    if _notion_client is None:
        api_key = os.getenv("NOTION_API_KEY")
        if not api_key:
            raise ValueError("NOTION_API_KEY is not set in environment / .env")
        # Pin to API version 2022-06-28 — the stable version that retains
        # GET/POST /databases/{id}/query.  The default in notion-client v3
        # is 2025-09-03 which moved database querying to a different
        # /data_sources endpoint that requires extra setup.
        _notion_client = Client(auth=api_key, notion_version="2022-06-28")
        logger.info("✅ Notion client initialised (api_version=2022-06-28)")
    return _notion_client


# ── Rate-limited API call wrapper ─────────────────────────────────────────────

def _notion_call(api_fn, **kwargs):
    """
    Invoke api_fn(**kwargs) with rate-limit protection:

    • Sleeps REQUEST_DELAY_SEC after every successful call so we stay
      safely under Notion's 3 req/s sustained limit.
    • On HTTP 429 waits for the Retry-After header duration (or exponential
      back-off if the header is absent) and retries up to MAX_RETRIES times.
    • All other exceptions propagate immediately.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = api_fn(**kwargs)
            time.sleep(REQUEST_DELAY_SEC)
            return result

        except APIResponseError as api_err:
            if api_err.status != 429:
                raise   # non-rate-limit error — propagate immediately

            # Determine how long to wait
            retry_after_wait = None
            if hasattr(api_err, "headers") and api_err.headers:
                raw = api_err.headers.get("Retry-After") or api_err.headers.get("retry-after")
                if raw:
                    try:
                        retry_after_wait = float(raw)
                    except ValueError:
                        pass

            wait_seconds = retry_after_wait if retry_after_wait is not None else min(
                BACKOFF_BASE_SEC * (2 ** (attempt - 1)),
                MAX_BACKOFF_SEC,
            )

            logger.warning(
                "   ⚠️  Notion 429 rate-limit (attempt %d/%d) — waiting %.1f s",
                attempt, MAX_RETRIES, wait_seconds,
            )
            time.sleep(wait_seconds)

            if attempt == MAX_RETRIES:
                logger.error(
                    "   ❌ Notion 429 — exhausted all %d retries for %s",
                    MAX_RETRIES, api_fn.__name__,
                )
                raise


# ── Property extractors (for database row properties) ─────────────────────────

def _prop_title(props: dict) -> str:
    """Extract the Title property (first property of type 'title')."""
    for val in props.values():
        if val.get("type") == "title":
            return "".join(
                rt.get("plain_text", "") for rt in val.get("title", [])
            ).strip()
    return ""


def _prop_select(props: dict, column_name: str) -> str:
    """Extract a select-type property by exact column name."""
    val = props.get(column_name, {})
    if val.get("type") == "select":
        select_obj = val.get("select") or {}
        return select_obj.get("name", "").strip()
    return ""


def _prop_rich_text(props: dict, column_name: str) -> str:
    """Extract a rich_text-type property by exact column name."""
    val = props.get(column_name, {})
    if val.get("type") == "rich_text":
        return "".join(
            rt.get("plain_text", "") for rt in val.get("rich_text", [])
        ).strip()
    return ""


def _prop_multi_select(props: dict, column_name: str) -> list[str]:
    """Extract a multi_select-type property as a list of tag strings."""
    val = props.get(column_name, {})
    if val.get("type") == "multi_select":
        return [
            opt.get("name", "").strip()
            for opt in val.get("multi_select", [])
            if opt.get("name")
        ]
    return []


# ── Block text conversion ──────────────────────────────────────────────────────

def _rich_text_to_plain(rich_text_list: list) -> str:
    """Collapse a Notion rich_text array to a plain-text string."""
    return "".join(rt.get("plain_text", "") for rt in rich_text_list)


_PLAIN_TEXT_TYPES = {
    "paragraph",
    "bulleted_list_item",
    "numbered_list_item",
    "toggle",
    "quote",
    "callout",
}
_HEADING_TYPES = {"heading_1", "heading_2", "heading_3"}


def _block_to_text(block: dict) -> tuple[str, str]:
    """
    Convert one Notion block dict to (heading_label, plain_text).

    heading_label — non-empty only for heading blocks; the chunker uses
                    this to update the current section label.
    plain_text    — the text to accumulate in the chunk buffer.

    Block type coverage:
      heading_1/2/3     → plain text, marks as heading
      paragraph         → plain text
      bulleted/numbered → plain text of the item line
      toggle/callout    → plain text of the header line
      quote             → plain text
      code              → fenced  ```lang\\n...\\n```
      table_row         → pipe-delimited  col1 | col2 | col3
      everything else   → ("", "") — children handled by caller
    """
    block_type = block.get("type", "")
    content    = block.get(block_type, {})

    if block_type in _HEADING_TYPES:
        text = _rich_text_to_plain(content.get("rich_text", []))
        return text, text   # heading_label == text for headings

    if block_type == "code":
        language  = content.get("language", "plain text")
        code_text = _rich_text_to_plain(content.get("rich_text", []))
        return "", f"```{language}\n{code_text}\n```"

    if block_type in _PLAIN_TEXT_TYPES:
        return "", _rich_text_to_plain(content.get("rich_text", []))

    if block_type == "table_row":
        cells = content.get("cells", [])
        return "", " | ".join(_rich_text_to_plain(cell) for cell in cells)

    return "", ""


# ── Paginated child-block fetcher ──────────────────────────────────────────────

def _fetch_children(block_id: str) -> list[dict]:
    """
    Fetch ALL child blocks of block_id, handling Notion pagination.
    Each page fetch is rate-limited via _notion_call().
    """
    client   = _get_client()
    results  = []
    cursor   = None
    has_more = True

    while has_more:
        kwargs = {"block_id": block_id, "page_size": 100}
        if cursor:
            kwargs["start_cursor"] = cursor

        resp     = _notion_call(client.blocks.children.list, **kwargs)
        results.extend(resp.get("results", []))
        cursor   = resp.get("next_cursor")
        has_more = resp.get("has_more", False)

    return results


# ── Recursive block extractor ──────────────────────────────────────────────────

def _extract_blocks_recursive(
    block_id: str,
    depth: int = 0,
) -> list[dict]:
    """
    Recursively walk all blocks under block_id and return a flat list of
    {heading, text, block_idx} dicts for every text-bearing block found.

    Special handling:
      • table blocks       — fetch their table_row children directly; the
                             first row is the header, subsequent rows are data.
      • has_children=True  — recurse into child blocks for toggles, callouts,
                             nested bullets, etc.

    Depth is capped at MAX_BLOCK_DEPTH to prevent runaway recursion on
    pathologically nested content.
    """
    if depth > MAX_BLOCK_DEPTH:
        logger.warning(
            "   ⚠️  Block recursion capped at depth %d for block_id='%s'",
            MAX_BLOCK_DEPTH, block_id,
        )
        return []

    raw_blocks = _fetch_children(block_id)
    extracted: list[dict] = []
    sequential_idx = 0   # running index within this recursion level

    for raw_block in raw_blocks:
        block_type = raw_block.get("type", "")
        block_id_inner = raw_block.get("id", "")

        # ── Table blocks ──────────────────────────────────────────────────────
        # A table block itself carries no text — all content lives in its
        # table_row children. We fetch them directly here.
        if block_type == "table":
            logger.info(
                "      📋 Table block — fetching rows (depth=%d)", depth
            )
            table_rows = _fetch_children(block_id_inner)
            row_number = 0
            for row_block in table_rows:
                if row_block.get("type") != "table_row":
                    continue
                _, row_text = _block_to_text(row_block)
                if row_text.strip():
                    extracted.append({
                        "heading":   "",
                        "text":      row_text,
                        "block_idx": sequential_idx,
                    })
                    row_number += 1
                sequential_idx += 1
            logger.info(
                "         → %d table rows extracted", row_number
            )
            continue   # table children already processed above

        # ── All other block types ─────────────────────────────────────────────
        heading_label, plain_text = _block_to_text(raw_block)

        if plain_text.strip():
            extracted.append({
                "heading":   heading_label,
                "text":      plain_text,
                "block_idx": sequential_idx,
            })
        sequential_idx += 1

        # ── Recurse into child blocks (toggles, callouts, nested items) ───────
        if raw_block.get("has_children", False) and block_type != "table":
            child_blocks = _extract_blocks_recursive(
                block_id_inner,
                depth=depth + 1,
            )
            extracted.extend(child_blocks)

    return extracted


# ═══════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════

def get_all_pages(database_id: Optional[str] = None) -> list[dict]:
    """
    Query every row in the DocForge Hub Library Notion database and return
    their metadata.

    Falls back to NOTION_ROOT_PAGE_ID env variable if database_id is not
    supplied — the root page ID in the Notion URL is the database ID.

    Returns a list of dicts:
        {page_id, title, doc_type, industry, version, tags}

    Handles database pagination (100 rows per request) automatically.
    All API calls are rate-limited via _notion_call().
    """
    raw_id = (database_id or os.getenv("NOTION_ROOT_PAGE_ID", "")).strip()
    if not raw_id:
        raise ValueError(
            "database_id not supplied and NOTION_ROOT_PAGE_ID is not set in .env"
        )

    # Normalise to the dashed UUID format the Notion SDK expects.
    # Whether the user supplies with or without dashes, we produce:
    #   xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
    clean = raw_id.replace("-", "")
    if len(clean) == 32:
        db_id = f"{clean[0:8]}-{clean[8:12]}-{clean[12:16]}-{clean[16:20]}-{clean[20:32]}"
    else:
        # Unexpected format — use as-is and let Notion return the error
        db_id = raw_id
        logger.warning(
            "   ⚠️  get_all_pages: unexpected ID format '%s' — using as-is", raw_id
        )

    logger.info(
        "🔍 get_all_pages — querying Notion database id='%s'", db_id
    )

    client = _get_client()

    # ── Verify the integration can see the database ───────────────────────────
    # A common failure mode is that the Notion integration hasn't been
    # explicitly shared on the database page. We do a retrieve() call first
    # so the error message is clear rather than silently returning 0 rows.
    try:
        db_meta = _notion_call(client.databases.retrieve, database_id=db_id)
        logger.info(
            "   ✅ Database reachable — title=%r  object=%s",
            db_meta.get("title", [{}])[0].get("plain_text", "(no title)")
            if db_meta.get("title") else "(no title)",
            db_meta.get("object", "?"),
        )
    except Exception as err:
        logger.error(
            "   ❌ Cannot access database '%s': %s\n"
            "   → Make sure the Notion integration has been shared on this database:\n"
            "     1. Open the database in Notion\n"
            "     2. Click '...' → Connections → Add the integration\n"
            "     3. Re-run ingestion",
            db_id, err,
        )
        raise RuntimeError(
            f"Notion integration cannot access database '{db_id}'. "
            f"Share the integration on the database in Notion first. Error: {err}"
        )

    pages    = []
    cursor   = None
    has_more = True
    page_num = 0

    while has_more:
        # With API version 2022-06-28 the database query endpoint is
        # POST /v1/databases/{id}/query — called via client.request() since
        # notion-client v3 removed the databases.query() convenience method.
        body: dict = {"page_size": 100}
        if cursor:
            body["start_cursor"] = cursor

        try:
            resp = _notion_call(
                client.request,
                path=f"databases/{db_id}/query",
                method="POST",
                body=body,
            )
        except Exception as err:
            logger.error(
                "   ❌ databases query failed (page %d) for db_id='%s': %s",
                page_num + 1, db_id, err,
            )
            break

        results = resp.get("results", [])
        page_num += 1
        logger.info(
            "   📋 Query page %d — %d row(s) returned  has_more=%s",
            page_num, len(results), resp.get("has_more", False),
        )

        for row in results:
            if row.get("object") != "page":
                logger.debug(
                    "      Skipping non-page object: %s", row.get("object")
                )
                continue

            row_page_id = row["id"]
            props       = row.get("properties", {})

            title    = _prop_title(props)
            doc_type = _prop_select(props, "Type")
            industry = _prop_select(props, "Industry")
            version  = _prop_rich_text(props, "Version")
            tags     = _prop_multi_select(props, "tags")

            if not title:
                title = row.get("child_page", {}).get("title", "(untitled)")

            logger.info(
                "   📄 Row %d: title='%s'  type=%s  industry=%s  version=%s  tags=%s",
                len(pages) + 1,
                title,
                doc_type or "(none)",
                industry or "(none)",
                version  or "(none)",
                tags or [],
            )

            pages.append({
                "page_id":  row_page_id,
                "title":    title    or "(untitled)",
                "doc_type": doc_type or "Document",
                "industry": industry or "General",
                "version":  version  or "1.0",
                "tags":     tags,
            })

        cursor   = resp.get("next_cursor")
        has_more = resp.get("has_more", False)

    logger.info(
        "   ✅ get_all_pages — %d pages found in database '%s'",
        len(pages), db_id,
    )
    return pages


def get_page_blocks(page_id: str) -> list[dict]:
    """
    Recursively extract all text-bearing blocks from a Notion page.

    Handles headings, paragraphs, bullets, numbered lists, toggles,
    callouts, quotes, code blocks, tables, and nested child blocks.

    Returns a flat list of dicts:
        {heading: str, text: str, block_idx: int}

    Only blocks with non-empty text are included.
    All API calls are rate-limited via _notion_call().
    """
    logger.info(
        "📄 get_page_blocks — extracting blocks for page_id='%s'", page_id
    )

    try:
        blocks = _extract_blocks_recursive(page_id, depth=0)
    except Exception as err:
        logger.error(
            "   ❌ get_page_blocks failed for page_id='%s': %s", page_id, err
        )
        return []

    logger.info(
        "   ✅ get_page_blocks — %d text blocks extracted from page '%s'",
        len(blocks), page_id,
    )
    return blocks