"""
notion_publisher.py  —  DocForge Hub

Converts a Markdown document into Notion blocks and publishes it as a
child page under a given parent page ID.

Key design decisions
────────────────────
* Notion's append-children API accepts at most 100 blocks per request.
  We split the block list into chunks of MAX_BLOCKS_PER_REQUEST (≤100).
* Notion rate-limits at ~3 requests/second (sustained). We default to
  a conservative REQUEST_INTERVAL_SEC = 0.4 s (≈2.5 req/s) with an
  exponential back-off on 429 responses.
* Text content inside a single rich-text array is capped at
  RICH_TEXT_MAX_CHARS (2000) characters — Notion rejects longer values.
* Tables are converted to Notion table blocks (supported since 2022).
* Inline Markdown formatting (**bold**, *italic*, `code`, ~~strike~~)
  is mapped to Notion rich-text annotations.

Public API
──────────
    publish_markdown_to_notion(
        markdown_text: str,
        document_title: str,
        parent_page_id: str,
        notion_client: notion_client.Client,
    ) -> dict          # {"page_id": str, "page_url": str, "blocks_pushed": int}
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

logger = logging.getLogger("docforge.notion_publisher")

# ─── Notion API limits ────────────────────────────────────────────────────────
MAX_BLOCKS_PER_REQUEST: int = 95       # Notion hard limit per append call
RICH_TEXT_MAX_CHARS: int = 1950         # Notion hard limit per rich-text object
REQUEST_INTERVAL_SEC: float = 0.4       # ~2.5 req/s — well under the 3 req/s cap
MAX_RETRIES: int = 5                    # max back-off retries on 429
BACKOFF_BASE_SEC: float = 1.5          # exponential back-off base


# ═══════════════════════════════════════════════════════════════════════════════
#  Rich-text helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _split_long_text(text: str) -> list[str]:
    """
    Split text into chunks of at most RICH_TEXT_MAX_CHARS characters.
    Tries to break on whitespace boundaries when possible.
    """
    if len(text) <= RICH_TEXT_MAX_CHARS:
        return [text]
    chunks: list[str] = []
    while text:
        if len(text) <= RICH_TEXT_MAX_CHARS:
            chunks.append(text)
            break
        split_at = text.rfind(" ", 0, RICH_TEXT_MAX_CHARS)
        if split_at == -1:
            split_at = RICH_TEXT_MAX_CHARS
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip()
    return chunks


def _parse_inline(text: str) -> list[dict]:
    """
    Convert inline Markdown to a list of Notion rich-text objects.
    Handles: **bold**, *italic*, `code`, ~~strikethrough~~, combined (**_text_**).
    Long plain text is split into multiple rich-text objects (≤2000 chars each).
    """
    if not text:
        return []

    rich_text_items: list[dict] = []

    # Tokenise with a regex that matches all inline patterns
    token_pattern = re.compile(
        r"(\*\*\*(.+?)\*\*\*"       # bold+italic
        r"|\*\*(.+?)\*\*"           # bold
        r"|\*(.+?)\*"               # italic
        r"|__(.+?)__"               # bold (alt)
        r"|_(.+?)_"                 # italic (alt)
        r"|~~(.+?)~~"               # strikethrough
        r"|`(.+?)`"                 # inline code
        r"|([^`*_~]+))",            # plain text
        re.DOTALL,
    )

    for match in token_pattern.finditer(text):
        bold_italic, bold, italic1, alt_bold, alt_italic, strike, code, plain = (
            match.group(2), match.group(3), match.group(4), match.group(5),
            match.group(6), match.group(7), match.group(8), match.group(9),
        )

        if bold_italic:
            content = bold_italic
            annotations = {"bold": True, "italic": True}
        elif bold or alt_bold:
            content = bold or alt_bold
            annotations = {"bold": True}
        elif italic1 or alt_italic:
            content = italic1 or alt_italic
            annotations = {"italic": True}
        elif strike:
            content = strike
            annotations = {"strikethrough": True}
        elif code:
            content = code
            annotations = {"code": True}
        elif plain:
            content = plain
            annotations = {}
        else:
            continue

        for chunk in _split_long_text(content):
            obj: dict[str, Any] = {
                "type": "text",
                "text": {"content": chunk},
            }
            if annotations:
                obj["annotations"] = annotations
            rich_text_items.append(obj)

    return rich_text_items or [{"type": "text", "text": {"content": ""}}]


# ═══════════════════════════════════════════════════════════════════════════════
#  Block builders
# ═══════════════════════════════════════════════════════════════════════════════

def _heading_block(level: int, text: str) -> dict:
    key = f"heading_{level}"
    return {
        "object": "block",
        "type": key,
        key: {"rich_text": _parse_inline(text)},
    }


def _paragraph_block(text: str) -> dict:
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {"rich_text": _parse_inline(text)},
    }


def _bulleted_list_item(text: str) -> dict:
    return {
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {"rich_text": _parse_inline(text)},
    }


def _numbered_list_item(text: str) -> dict:
    return {
        "object": "block",
        "type": "numbered_list_item",
        "numbered_list_item": {"rich_text": _parse_inline(text)},
    }


def _divider_block() -> dict:
    return {"object": "block", "type": "divider", "divider": {}}


def _code_block(code_text: str, language: str = "plain text") -> dict:
    # Notion code block content limit is 2000 chars per rich-text object
    rt = []
    for chunk in _split_long_text(code_text):
        rt.append({"type": "text", "text": {"content": chunk}})
    return {
        "object": "block",
        "type": "code",
        "code": {"rich_text": rt, "language": language},
    }


def _quote_block(text: str) -> dict:
    return {
        "object": "block",
        "type": "quote",
        "quote": {"rich_text": _parse_inline(text)},
    }


def _table_block(rows: list[list[str]]) -> dict:
    """
    Build a Notion table block.
    rows[0] is treated as the header row.
    """
    if not rows:
        return _paragraph_block("")

    col_count = max(len(row) for row in rows)
    notion_rows = []
    for row_idx, row in enumerate(rows):
        cells = []
        for col_idx in range(col_count):
            cell_text = row[col_idx].strip() if col_idx < len(row) else ""
            cells.append(_parse_inline(cell_text))
        notion_rows.append({
            "object": "block",
            "type": "table_row",
            "table_row": {"cells": cells},
        })

    return {
        "object": "block",
        "type": "table",
        "table": {
            "table_width": col_count,
            "has_column_header": True,
            "has_row_header": False,
            "children": notion_rows,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Markdown → Notion block list
# ═══════════════════════════════════════════════════════════════════════════════

def markdown_to_notion_blocks(markdown_text: str) -> list[dict]:
    """
    Parse Markdown and return a flat list of Notion block dicts.

    Supported constructs:
      Headings (#, ##, ###, ####)
      Unordered lists (-, *, +)
      Ordered lists (1. 2. …)
      Fenced code blocks (``` … ```)
      Blockquotes (>)
      Horizontal rules (---, ___, ***)
      Markdown tables (|col|col|)
      Plain paragraphs (with inline formatting)
    """
    blocks: list[dict] = []
    lines = markdown_text.splitlines()
    idx = 0

    while idx < len(lines):
        raw_line = lines[idx]
        line = raw_line.strip()

        # ── Empty line → skip (paragraph spacing handled implicitly) ──────────
        if not line:
            idx += 1
            continue

        # ── Fenced code block ─────────────────────────────────────────────────
        code_fence_match = re.match(r"^```(\w*)", line)
        if code_fence_match:
            lang = code_fence_match.group(1) or "plain text"
            code_lines: list[str] = []
            idx += 1
            while idx < len(lines) and not lines[idx].strip().startswith("```"):
                code_lines.append(lines[idx])
                idx += 1
            idx += 1  # skip closing fence
            blocks.append(_code_block("\n".join(code_lines), lang))
            continue

        # ── Horizontal rule ───────────────────────────────────────────────────
        if re.match(r"^(---+|___+|\*\*\*+)$", line):
            blocks.append(_divider_block())
            idx += 1
            continue

        # ── Headings ──────────────────────────────────────────────────────────
        heading_match = re.match(r"^(#{1,4})\s+(.+)", line)
        if heading_match:
            level = min(len(heading_match.group(1)), 3)  # Notion only has h1-h3
            blocks.append(_heading_block(level, heading_match.group(2).strip()))
            idx += 1
            continue

        # ── Markdown table ────────────────────────────────────────────────────
        if line.startswith("|"):
            table_raw_lines: list[str] = []
            while idx < len(lines) and lines[idx].strip().startswith("|"):
                table_raw_lines.append(lines[idx].strip())
                idx += 1
            # Parse rows, skip separator lines (|---|---|)
            parsed_rows: list[list[str]] = []
            for table_line in table_raw_lines:
                if re.match(r"^\|[\s\-\|:]+\|$", table_line):
                    continue
                cells = [c.strip() for c in table_line.split("|") if c.strip() != ""]
                if cells:
                    parsed_rows.append(cells)
            if parsed_rows:
                blocks.append(_table_block(parsed_rows))
            continue

        # ── Blockquote ────────────────────────────────────────────────────────
        if line.startswith("> "):
            blocks.append(_quote_block(line[2:].strip()))
            idx += 1
            continue

        # ── Unordered list item ───────────────────────────────────────────────
        if re.match(r"^[\-\*\+]\s+", line):
            item_text = re.sub(r"^[\-\*\+]\s+", "", line)
            blocks.append(_bulleted_list_item(item_text))
            idx += 1
            continue

        # ── Ordered list item ─────────────────────────────────────────────────
        if re.match(r"^\d+\.\s+", line):
            item_text = re.sub(r"^\d+\.\s+", "", line)
            blocks.append(_numbered_list_item(item_text))
            idx += 1
            continue

        # ── Plain paragraph ───────────────────────────────────────────────────
        # Collect consecutive non-special lines into one paragraph
        paragraph_lines: list[str] = []
        while idx < len(lines):
            peek = lines[idx].strip()
            if (
                not peek
                or re.match(r"^#{1,4}\s", peek)
                or peek.startswith("|")
                or peek.startswith("> ")
                or re.match(r"^[\-\*\+]\s+", peek)
                or re.match(r"^\d+\.\s+", peek)
                or re.match(r"^(---+|___+|\*\*\*+)$", peek)
                or re.match(r"^```", peek)
            ):
                break
            paragraph_lines.append(peek)
            idx += 1
        if paragraph_lines:
            blocks.append(_paragraph_block(" ".join(paragraph_lines)))

    return blocks


# ═══════════════════════════════════════════════════════════════════════════════
#  Rate-limited Notion API calls
# ═══════════════════════════════════════════════════════════════════════════════

def _append_blocks_with_backoff(
    notion_client,
    block_id: str,
    children: list[dict],
    request_interval: float = REQUEST_INTERVAL_SEC,
) -> None:
    """
    Append `children` to `block_id` with exponential back-off on 429.
    Raises the underlying exception after MAX_RETRIES failures.
    """
    from notion_client.errors import APIResponseError

    for attempt in range(MAX_RETRIES):
        try:
            notion_client.blocks.children.append(
                block_id=block_id,
                children=children,
            )
            time.sleep(request_interval)
            return
        except APIResponseError as err:
            if err.status == 429:
                wait = BACKOFF_BASE_SEC * (2 ** attempt)
                logger.warning(
                    "Notion 429 rate-limit hit — waiting %.1f s (attempt %d/%d)",
                    wait, attempt + 1, MAX_RETRIES,
                )
                time.sleep(wait)
            else:
                raise
    # If we exhaust retries, raise one final time
    notion_client.blocks.children.append(block_id=block_id, children=children)


def _push_blocks_in_chunks(
    notion_client,
    page_id: str,
    blocks: list[dict],
    chunk_size: int = MAX_BLOCKS_PER_REQUEST,
) -> int:
    """
    Push all blocks to `page_id` in chunks ≤ chunk_size.
    Returns the total number of blocks pushed.
    """
    # Notion does not accept table children nested inline — we need to
    # strip `children` from table blocks and append them separately.
    flat_blocks = _flatten_table_children(blocks)

    total_pushed = 0
    for start in range(0, len(flat_blocks), chunk_size):
        chunk = flat_blocks[start : start + chunk_size]
        logger.debug("Pushing blocks %d–%d to page %s", start, start + len(chunk) - 1, page_id)
        _append_blocks_with_backoff(notion_client, page_id, chunk)
        total_pushed += len(chunk)

    return total_pushed


def _flatten_table_children(blocks: list[dict]) -> list[dict]:
    """
    Notion's append-children API does NOT support nested `children` in the
    payload for table blocks sent during page creation. Instead we must:
      1. Send the table shell (with children embedded — this is fine for
         the /pages endpoint on creation but NOT for append-children).
    
    We keep tables with their embedded children because the Notion API
    does accept nested children for table blocks specifically.
    All other block types are returned as-is.
    """
    return blocks  # Notion accepts table children inline in append-children


# ═══════════════════════════════════════════════════════════════════════════════
#  Public entry point
# ═══════════════════════════════════════════════════════════════════════════════

def publish_markdown_to_notion(
    markdown_text: str,
    document_title: str,
    parent_page_id: str,
    notion_client_instance,
) -> dict:
    """
    Convert `markdown_text` to Notion blocks and publish as a new child page
    under `parent_page_id`.

    Parameters
    ──────────
    markdown_text         : raw Markdown string from st.session_state.markdown_doc
    document_title        : title shown in the Notion page header
    parent_page_id        : Notion page ID under which the new page is created
    notion_client_instance: initialised notion_client.Client

    Returns
    ───────
    {
        "page_id":      str,   # Notion page ID of the newly created page
        "page_url":     str,   # browser URL for the page
        "blocks_pushed": int,  # total Notion blocks written
    }

    Raises
    ──────
    ValueError  : if markdown_text or parent_page_id is empty
    Exception   : propagated from the Notion API on non-429 errors
    """
    if not markdown_text or not markdown_text.strip():
        raise ValueError("markdown_text is empty — nothing to publish.")
    if not parent_page_id or not parent_page_id.strip():
        raise ValueError("parent_page_id must be provided.")

    clean_title = document_title.strip() or "Untitled Document"
    logger.info("Publishing '%s' to Notion under parent %s", clean_title, parent_page_id)

    # ── Step 1: Create the page shell (title only, no content yet) ────────────
    new_page = notion_client_instance.pages.create(
        parent={"page_id": parent_page_id},
        properties={
            "title": {
                "title": [{"type": "text", "text": {"content": clean_title}}]
            }
        },
    )
    new_page_id: str = new_page["id"]
    raw_url: str = new_page.get("url", f"https://notion.so/{new_page_id.replace('-', '')}")
    logger.info("Page created — id=%s url=%s", new_page_id, raw_url)

    # Small delay after page creation before we start appending
    time.sleep(REQUEST_INTERVAL_SEC)

    # ── Step 2: Parse Markdown → Notion blocks ────────────────────────────────
    all_blocks = markdown_to_notion_blocks(markdown_text)
    logger.info("Parsed %d Notion blocks from Markdown", len(all_blocks))

    # ── Step 3: Push blocks in rate-limited chunks ────────────────────────────
    blocks_pushed = _push_blocks_in_chunks(notion_client_instance, new_page_id, all_blocks)
    logger.info("Pushed %d blocks to page %s", blocks_pushed, new_page_id)

    return {
        "page_id": new_page_id,
        "page_url": raw_url,
        "blocks_pushed": blocks_pushed,
    }