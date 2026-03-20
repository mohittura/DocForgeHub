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
#  Notion code block language normalisation
#
#  Notion's API only accepts a fixed set of language strings. Any language
#  not in that set causes a 400 validation error. This map converts common
#  aliases and unsupported languages (e.g. "http", "sh", "js") to their
#  accepted equivalents. Anything not in the map falls back to "plain text".
# ═══════════════════════════════════════════════════════════════════════════════

_NOTION_LANGUAGE_MAP: dict[str, str] = {
    # Direct matches (already valid — identity entries for safety)
    "abap": "abap", "abc": "abc", "agda": "agda", "arduino": "arduino",
    "ascii art": "ascii art", "assembly": "assembly", "bash": "bash",
    "basic": "basic", "bnf": "bnf", "c": "c", "c#": "c#", "c++": "c++",
    "clojure": "clojure", "coffeescript": "coffeescript", "coq": "coq",
    "css": "css", "dart": "dart", "dhall": "dhall", "diff": "diff",
    "docker": "docker", "ebnf": "ebnf", "elixir": "elixir", "elm": "elm",
    "erlang": "erlang", "f#": "f#", "flow": "flow", "fortran": "fortran",
    "gherkin": "gherkin", "glsl": "glsl", "go": "go", "graphql": "graphql",
    "groovy": "groovy", "haskell": "haskell", "hcl": "hcl", "html": "html",
    "idris": "idris", "java": "java", "javascript": "javascript",
    "json": "json", "julia": "julia", "kotlin": "kotlin", "latex": "latex",
    "less": "less", "lisp": "lisp", "livescript": "livescript",
    "llvm ir": "llvm ir", "lua": "lua", "makefile": "makefile",
    "markdown": "markdown", "markup": "markup", "matlab": "matlab",
    "mathematica": "mathematica", "mermaid": "mermaid", "nix": "nix",
    "notion formula": "notion formula", "objective-c": "objective-c",
    "ocaml": "ocaml", "pascal": "pascal", "perl": "perl", "php": "php",
    "plain text": "plain text", "powershell": "powershell", "prolog": "prolog",
    "protobuf": "protobuf", "purescript": "purescript", "python": "python",
    "r": "r", "racket": "racket", "reason": "reason", "ruby": "ruby",
    "rust": "rust", "sass": "sass", "scala": "scala", "scheme": "scheme",
    "scss": "scss", "shell": "shell", "smalltalk": "smalltalk",
    "solidity": "solidity", "sql": "sql", "swift": "swift", "toml": "toml",
    "typescript": "typescript", "vb.net": "vb.net", "verilog": "verilog",
    "vhdl": "vhdl", "visual basic": "visual basic",
    "webassembly": "webassembly", "xml": "xml", "yaml": "yaml",
    "java/c/c++/c#": "java/c/c++/c#",
    # Common aliases / unsupported languages → nearest valid equivalent
    "http": "plain text",
    "https": "plain text",
    "sh": "shell",
    "zsh": "shell",
    "bash-session": "bash",
    "console": "bash",
    "terminal": "bash",
    "js": "javascript",
    "jsx": "javascript",
    "ts": "typescript",
    "tsx": "typescript",
    "py": "python",
    "python3": "python",
    "rb": "ruby",
    "rs": "rust",
    "cs": "c#",
    "csharp": "c#",
    "cpp": "c++",
    "c++": "c++",
    "objc": "objective-c",
    "dockerfile": "docker",
    "tf": "hcl",
    "terraform": "hcl",
    "proto": "protobuf",
    "proto3": "protobuf",
    "md": "markdown",
    "text": "plain text",
    "txt": "plain text",
    "": "plain text",
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
            raw_lang = code_fence_match.group(1).strip().lower()
            lang = _NOTION_LANGUAGE_MAP.get(raw_lang, "plain text")
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


# ═══════════════════════════════════════════════════════════════════════════════
#  Version resolution
# ═══════════════════════════════════════════════════════════════════════════════

def get_latest_version_for_title(
    document_title: str,
    database_id: str,
    api_key: str,
) -> float | None:
    """
    Query the Notion database for all rows whose Title matches `document_title`
    and return the highest version number found as a float, or None if no
    matching rows exist.

    Uses the direct Notion REST API (same pattern as GET /get_all_urls) to
    bypass notion-client version incompatibilities.

    Version values stored in the database are expected to look like "1.0",
    "2.0", "3.0" etc. Any row whose Version field cannot be parsed as a float
    is silently skipped.
    """
    import requests as _requests

    def to_dashed(s: str) -> str:
        s = s.replace("-", "")
        return f"{s[0:8]}-{s[8:12]}-{s[12:16]}-{s[16:20]}-{s[20:32]}" if len(s) == 32 else s

    clean_db_id = to_dashed(database_id.replace("-", ""))
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
    }

    # Filter: Title property equals document_title (exact match)
    body: dict = {
        "filter": {
            "property": "Title",
            "title": {"equals": document_title.strip()},
        },
        "page_size": 100,
    }

    max_version: float | None = None
    has_more = True
    next_cursor = None

    while has_more:
        if next_cursor:
            body["start_cursor"] = next_cursor

        try:
            resp = _requests.post(
                f"https://api.notion.com/v1/databases/{clean_db_id}/query",
                headers=headers,
                json=body,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as err:
            logger.warning("Version lookup failed (will default to 1.0): %s", err)
            return None

        for page in data.get("results", []):
            props = page.get("properties", {})
            version_text = "".join(
                t.get("plain_text", "")
                for t in props.get("Version", {}).get("rich_text", [])
            ).strip()
            try:
                v = float(version_text)
                if max_version is None or v > max_version:
                    max_version = v
            except (ValueError, TypeError):
                pass  # skip rows with non-numeric version

        has_more = data.get("has_more", False)
        next_cursor = data.get("next_cursor")

    return max_version


def _next_version(latest: float | None) -> str:
    """
    Given the highest version found (or None), return the next version string.

    None  -> "1.0"   (first publish)
    1.0   -> "2.0"
    2.0   -> "3.0"
    """
    if latest is None:
        return "1.0"
    return f"{int(latest) + 1}.0"


# ═══════════════════════════════════════════════════════════════════════════════
#  Database publish
# ═══════════════════════════════════════════════════════════════════════════════

def publish_to_notion_database(
    markdown_text: str,
    document_title: str,
    document_type: str,
    industry: str,
    tags: list[str],
    database_id: str,
    notion_client_instance,
    notion_api_key: str = "",
    version: str = "",
) -> dict:
    """
    Create a new row in the Notion "Lib" database and push the Markdown
    as the page body content.

    Version control
    ───────────────
    If `notion_api_key` is provided, the database is queried for existing
    rows with the same Title. The version is automatically set to:
      - "1.0"  if no prior rows exist for this title
      - "{N+1}.0"  where N is the highest version already published

    If `notion_api_key` is empty (e.g. in tests), `version` is used as-is
    and defaults to "1.0".

    This means publishing the same document type twice produces two rows:
        Row 1:  Feature Prioritization Framework  |  v1.0
        Row 2:  Feature Prioritization Framework  |  v2.0
    Both rows remain in the database — old versions are never overwritten.

    Column names (exactly as shown in Notion):
        Title        — title        (the built-in title column)
        Type         — select
        Industry     — select
        Version      — rich_text
        tags         — multi_select
        Created by   — created_by   (Notion built-in, auto-set, NOT in payload)
        Created time — created_time (Notion built-in, auto-set, NOT in payload)
    """
    if not markdown_text or not markdown_text.strip():
        raise ValueError("markdown_text is empty — nothing to publish.")
    if not database_id or not database_id.strip():
        raise ValueError("database_id must be provided.")

    clean_db_id = database_id.strip().replace("-", "")
    clean_title = document_title.strip() or "Untitled Document"

    # ── Auto-resolve version ──────────────────────────────────────────────────
    if notion_api_key:
        latest = get_latest_version_for_title(clean_title, clean_db_id, notion_api_key)
        resolved_version = _next_version(latest)
        if latest is not None:
            logger.info(
                "   📌 Version control: found existing v%.1f → assigning %s",
                latest, resolved_version,
            )
        else:
            logger.info("   📌 Version control: no prior version found → assigning %s", resolved_version)
    else:
        # Fallback: use the caller-supplied version or default to "1.0"
        resolved_version = version.strip() or "1.0"

    logger.info(
        "Publishing '%s' to database %s (type=%s, industry=%s, v=%s)",
        clean_title, clean_db_id, document_type, industry, resolved_version,
    )

    # Only the 5 writable columns — never send Created by / Created time
    properties: dict = {
        "Title": {
            "title": [{"type": "text", "text": {"content": clean_title[:2000]}}]
        },
        "Type": {
            "rich_text": [{"type": "text", "text": {"content": (document_type or "")[:200]}}]
        },
        "Industry": {
            "select": {"name": industry} if industry else {"name": "General"}
        },
        "Version": {
            "rich_text": [{"type": "text", "text": {"content": resolved_version[:200]}}]
        },
        "tags": {
            "multi_select": [{"name": t.strip()} for t in (tags or []) if t.strip()]
        },
    }

    # ── Step 1: Create the row ────────────────────────────────────────────────
    try:
        new_page = notion_client_instance.pages.create(
            parent={"database_id": clean_db_id},
            properties=properties,
            children=[],
        )
    except Exception as err:
        # Log the full Notion error body so we can see exactly what's wrong
        logger.error("Notion pages.create failed: %s", err)
        raise

    new_page_id: str = new_page["id"]
    raw_url: str = new_page.get(
        "url", f"https://notion.so/{new_page_id.replace('-', '')}"
    )
    logger.info("Row created — page_id=%s url=%s", new_page_id, raw_url)
    time.sleep(REQUEST_INTERVAL_SEC)

    # ── Step 2: Parse Markdown → Notion blocks ────────────────────────────────
    all_blocks = markdown_to_notion_blocks(markdown_text)
    logger.info("Parsed %d blocks from Markdown", len(all_blocks))

    # ── Step 3: Push blocks in rate-limited chunks ────────────────────────────
    blocks_pushed = _push_blocks_in_chunks(notion_client_instance, new_page_id, all_blocks)
    logger.info("Pushed %d blocks to page %s", blocks_pushed, new_page_id)

    return {
        "page_id": new_page_id,
        "page_url": raw_url,
        "blocks_pushed": blocks_pushed,
        "version": resolved_version,
    }