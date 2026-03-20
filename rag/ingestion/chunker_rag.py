"""
rag/ingestion/chunker_rag.py

Text chunker for CiteRagLab.

Splits a Notion page's flat block list into overlapping chunks of
300-500 tokens, preserving section context and citation metadata.

Strategy
────────
- Token estimate : 1 token ≈ 4 characters (OpenAI rule of thumb).
- TARGET_TOKENS  : aim to flush when the buffer reaches this size.
- MAX_TOKENS     : hard cap — flush before adding a block that would exceed this.
- OVERLAP_TOKENS : carry the last N tokens into the next chunk so context
                   is not lost at boundaries.
- Headings       : update current_section on every heading block so every
                   chunk carries the correct section label.
- Tables         : table rows are grouped together. If a run of pipe-delimited
                   rows would overflow the buffer they are flushed as a single
                   unit so the header row always stays with its data rows.
- Code blocks    : fenced code strings are kept intact in one chunk (never
                   split mid-fence).
"""

import logging
from typing import Optional

logger = logging.getLogger("rag.ingestion.chunker_rag")

TARGET_TOKENS  = 400
MAX_TOKENS     = 500
OVERLAP_TOKENS = 60


def _token_count(text: str) -> int:
    """Approximate token count — 1 token ≈ 4 characters."""
    return max(1, len(text) // 4)


def _is_table_row(text: str) -> bool:
    """Return True if the text looks like a pipe-delimited table row."""
    return "|" in text


def _is_code_block(text: str) -> bool:
    """Return True if the text is a fenced code block."""
    return text.startswith("```")


def chunk_page(
    page_id:  str,
    title:    str,
    blocks:   list[dict],
    industry: str = "General",
    doc_type: str = "Document",
    version:  str = "1.0",
    doc_id:   Optional[str] = None,
) -> list[dict]:
    """
    Convert a page's block list into overlapping text chunks.

    Parameters
    ──────────
    page_id   : Notion page UUID (used as page_id in chunk metadata)
    title     : page title (also the default section label)
    blocks    : output of notion_loader_rag.get_page_blocks()
                each dict: {heading: str, text: str, block_idx: int}
    industry  : Milvus metadata filter value
    doc_type  : Milvus metadata filter value
    version   : Milvus metadata filter value
    doc_id    : override for doc_id field (defaults to page_id)

    Returns
    ───────
    A list of chunk dicts, each containing:
        chunk_text, doc_id, title, section, industry,
        doc_type, version, page_id, block_range
    """
    if not blocks:
        logger.warning(
            "   ⚠️  chunk_page: no blocks for page '%s' — returning []", title
        )
        return []

    effective_doc_id = doc_id or page_id
    chunks: list[dict] = []
    current_section   = title
    buffer_lines: list[str] = []
    buffer_tokens     = 0
    start_block_idx   = 0

    logger.info(
        "✂️  chunk_page — page='%s' (%d blocks) doc_type=%s industry=%s version=%s",
        title, len(blocks), doc_type, industry, version,
    )

    def _flush(end_block_idx: int) -> None:
        """
        Flush the current buffer into a chunk dict, then seed the next buffer
        with the last OVERLAP_TOKENS worth of lines for continuity.
        """
        nonlocal buffer_lines, buffer_tokens, start_block_idx

        chunk_text = "\n".join(buffer_lines).strip()
        if chunk_text:
            chunk = {
                "chunk_text":  chunk_text,
                "doc_id":      effective_doc_id,
                "title":       title,
                "section":     current_section,
                "industry":    industry,
                "doc_type":    doc_type,
                "version":     version,
                "page_id":     page_id,
                "block_range": f"{start_block_idx}-{end_block_idx}",
            }
            chunks.append(chunk)
            logger.info(
                "   📦 Chunk %d — section='%s', blocks=%d-%d, ~%d tokens",
                len(chunks), current_section,
                start_block_idx, end_block_idx, buffer_tokens,
            )

        # Build overlap seed for the next chunk
        overlap_lines: list[str] = []
        overlap_tokens = 0
        for line in reversed(buffer_lines):
            line_tokens = _token_count(line)
            if overlap_tokens + line_tokens > OVERLAP_TOKENS:
                break
            overlap_lines.insert(0, line)
            overlap_tokens += line_tokens

        buffer_lines  = overlap_lines
        buffer_tokens = overlap_tokens
        start_block_idx = end_block_idx

    # ── Collect consecutive table rows into a group before flushing ───────────
    # This prevents the header row from ending up in a different chunk than
    # the data rows that follow it.
    table_row_group: list[str] = []
    table_start_idx: int = 0

    def _flush_table_group(end_idx: int) -> None:
        """Flush accumulated table rows as a single unit."""
        nonlocal buffer_lines, buffer_tokens, table_row_group

        if not table_row_group:
            return
        group_text   = "\n".join(table_row_group)
        group_tokens = _token_count(group_text)

        # If the table group alone overflows MAX_TOKENS, flush the current
        # buffer first, then emit the table group as its own chunk.
        if buffer_tokens + group_tokens > MAX_TOKENS and buffer_lines:
            _flush(end_idx)

        buffer_lines.extend(table_row_group)
        buffer_tokens += group_tokens
        table_row_group = []

        if buffer_tokens >= TARGET_TOKENS:
            _flush(end_idx)

    for block in blocks:
        heading   = block.get("heading", "")
        text      = block.get("text", "").strip()
        block_idx = block.get("block_idx", 0)

        # Update section label when we encounter a heading
        if heading:
            current_section = heading
            logger.info("   📌 Section: '%s'", heading)

        if not text:
            continue

        # ── Table rows — accumulate as a group ───────────────────────────────
        if _is_table_row(text):
            if not table_row_group:
                table_start_idx = block_idx
            table_row_group.append(text)
            continue
        else:
            # Non-table block — flush any accumulated table group first
            if table_row_group:
                _flush_table_group(block_idx)

        # ── Code blocks — keep intact, flush buffer before if needed ─────────
        if _is_code_block(text):
            code_tokens = _token_count(text)
            if buffer_lines:
                _flush(block_idx)
            buffer_lines.append(text)
            buffer_tokens = code_tokens
            _flush(block_idx + 1)
            continue

        # ── Regular text block ────────────────────────────────────────────────
        block_tokens = _token_count(text)

        # Flush before adding if adding this block would exceed the hard cap
        if buffer_tokens + block_tokens > MAX_TOKENS and buffer_lines:
            _flush(block_idx)

        buffer_lines.append(text)
        buffer_tokens += block_tokens

        # Flush when we reach the target size
        if buffer_tokens >= TARGET_TOKENS:
            _flush(block_idx + 1)

    # Flush any remaining table rows and buffer content
    if table_row_group:
        _flush_table_group(len(blocks))
    if buffer_lines:
        _flush(len(blocks))

    logger.info(
        "   ✅ chunk_page done — page='%s' → %d chunks produced",
        title, len(chunks),
    )
    return chunks