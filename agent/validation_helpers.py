"""
Document structure validation helpers for the DocForge Hub agent.

Validates a generated Markdown document against the required_section schema.
Checks for missing sections, extra (hallucinated) sections, and table
column correctness.

All functions are pure — they take text + schema and return error lists.
"""

import re
import logging

from agent.schema_helpers import is_table_only_schema

logger = logging.getLogger("agent.validation_helpers")


def _normalise_heading(raw: str) -> str:
    """
    Normalise a heading string for tolerant but content-strict comparison.
    Strips # markers, leading number prefixes like "4.1 ", and punctuation,
    then lowercases so "### 4.1 Customer Impact" matches "4.1 Customer Impact".
    """
    text = raw.strip().lstrip("#").strip()
    text = re.sub(r"^\d+(\.\d+)*\.?\s*", "", text)  # remove "4.1 " style prefixes
    text = re.sub(r"[^\w\s]", "", text)              # remove punctuation
    return text.lower().strip()


def validate_document_structure(document_text: str, required_section: dict) -> list[str]:
    """
    Validate the generated document against the schema.

    Returns a list of error strings (empty = valid).

    Pattern A (table-only): returns [] — quality_gate handles it deterministically.
    Pattern B (flat subsections): two-way check — missing headings and extra
    headings are both flagged, plus table column validation.
    """
    # Pattern A: table-only — handled by quality_gate's column checks
    if is_table_only_schema(required_section):
        return []

    # ── Build expected sections from the schema ──────────────────────────────
    # For Pattern B, all required headings live in subsections[], sorted by order.
    # The parent section title is informational only and NOT a required heading.
    expected_sections = []
    for schema_section in required_section.get("sections", []):
        subsections = schema_section.get("subsections", [])
        if subsections:
            # Pattern B: every subsection title = a required document heading
            for subsection_item in sorted(subsections, key=lambda s: s.get("order", 0)):
                title = subsection_item.get("title", "").strip()
                if title:
                    expected_sections.append({
                        "title": title,
                        "type": subsection_item.get("type", "text"),
                        "columns": subsection_item.get("columns", []),
                    })
        else:
            # Fallback for future schema patterns where section itself is titled
            title = schema_section.get("title", "").strip()
            if title:
                expected_sections.append({
                    "title": title,
                    "type": schema_section.get("type", "text"),
                    "columns": schema_section.get("columns", []),
                })

    if not expected_sections:
        logger.warning("   ⚠️  validate_document_structure: no expected sections found in schema")
        return []

    errors = []
    doc_lines = document_text.split("\n")

    # Extract all headings from the document as (line_index, raw_text) pairs
    doc_headings: list[tuple[int, str]] = [
        (i, line.strip().lstrip("#").strip())
        for i, line in enumerate(doc_lines)
        if line.strip().startswith("#")
    ]
    doc_headings_norm: list[tuple[int, str]] = [
        (i, _normalise_heading(raw)) for i, raw in doc_headings
    ]

    # Normalised allowlist: normalised_title → schema entry
    # This is the single source of truth for what headings are permitted.
    allowlist: dict[str, dict] = {
        _normalise_heading(schema_section["title"]): schema_section
        for schema_section in expected_sections
    }

    # ── CHECK 1: Missing sections ────────────────────────────────────────────
    # Every schema subsection title must appear as a heading in the document.
    for norm_title, schema_entry in allowlist.items():
        found = any(norm_title in doc_norm for _, doc_norm in doc_headings_norm)
        if not found:
            errors.append(f"Missing required section: '{schema_entry['title']}'")

    # ── CHECK 2: Extra sections ──────────────────────────────────────────────
    # Every heading in the document must match something in the allowlist.
    # Headings the LLM invented beyond the schema are flagged.
    #
    # Skip-list: headings that are legitimately present even though they are
    # not subsection titles in the schema:
    #   • The document name (document_name / document_type at the top level)
    #   • Parent section titles (sections[].title) — these wrap subsections
    skip_headings: set[str] = set()
    doc_name = _normalise_heading(required_section.get("document_name", ""))
    doc_type = _normalise_heading(required_section.get("document_type", ""))
    if doc_name:
        skip_headings.add(doc_name)
    if doc_type:
        skip_headings.add(doc_type)
    for schema_section in required_section.get("sections", []):
        parent_title = _normalise_heading(schema_section.get("title", ""))
        if parent_title:
            skip_headings.add(parent_title)

    for (_, raw_heading), (_, norm_heading) in zip(doc_headings, doc_headings_norm):
        # Allow if it matches the allowlist (subsection titles)
        in_allowlist = any(
            allowed in norm_heading or norm_heading in allowed
            for allowed in allowlist
        )
        # Allow if it matches the document name or a parent section title
        in_skip = any(
            skip in norm_heading or norm_heading in skip
            for skip in skip_headings
            if skip
        )
        if not in_allowlist and not in_skip:
            errors.append(
                f"Extra section not in schema: '{raw_heading}' — "
                f"remove it, the document must only contain schema-defined sections."
            )

    # ── CHECK 3: Table content ───────────────────────────────────────────────
    # For type="table" subsections, the content block under that heading must
    # contain a real Markdown table with the correct column headers.
    for norm_title, schema_entry in allowlist.items():
        if schema_entry["type"] != "table":
            continue

        expected_cols = schema_entry.get("columns", [])

        # Find this heading's line index in the document
        heading_line_idx = next(
            (idx for idx, norm in doc_headings_norm if norm_title in norm),
            None,
        )
        if heading_line_idx is None:
            continue  # already caught by CHECK 1

        # Grab lines from this heading until the next heading
        next_heading_idx = next(
            (idx for idx, _ in doc_headings_norm if idx > heading_line_idx),
            len(doc_lines),
        )
        block_lines = doc_lines[heading_line_idx:next_heading_idx]
        block_text = "\n".join(block_lines)

        # Must contain a pipe-delimited table with a separator row
        has_table = "|" in block_text and re.search(r"\|[\s\-|]+\|", block_text)
        if not has_table:
            errors.append(
                f"Section '{schema_entry['title']}' must contain a Markdown table "
                f"(expected columns: {', '.join(expected_cols)})"
            )
            continue

        # Verify the column headers match the schema exactly
        if expected_cols:
            table_lines = [line.strip() for line in block_lines if line.strip().startswith("|")]
            if table_lines:
                actual_cols = [col.strip() for col in table_lines[0].split("|") if col.strip()]
                if [col.lower() for col in expected_cols] != [col.lower() for col in actual_cols]:
                    errors.append(
                        f"Section '{schema_entry['title']}' has wrong table columns. "
                        f"Expected: {expected_cols}. Got: {actual_cols}"
                    )

    if errors:
        logger.warning(
            "   ❌ validate_document_structure: %d error(s): %s",
            len(errors), errors,
        )
    else:
        logger.info(
            "   ✅ validate_document_structure passed — %d sections checked, no extras",
            len(expected_sections),
        )

    return errors