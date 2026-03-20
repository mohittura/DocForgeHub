"""
rag/retrieval/filters_rag.py

Metadata filter builder for Milvus queries.

Validates and normalises raw filter dicts from API requests before they
reach the Milvus boolean expression layer in milvus_client_rag.search_chunks().

Supported filter keys  (all match Notion database columns)
───────────────────────────────────────────────────────────
    industry  — exact match on Industry (select column)
    doc_type  — exact match on Type (select column)
    version   — exact match on Version (rich_text column)
    tags      — substring match on tags (multi_select column,
                stored in Milvus as a comma-joined string)
"""

import logging

logger = logging.getLogger("rag.retrieval.filters_rag")

# All keys the UI is allowed to pass as filters.
# Anything not in this set is dropped before reaching Milvus.
ALLOWED_FILTER_KEYS = {"industry", "doc_type", "version", "tags"}


def build_filters(raw: dict) -> dict:
    """
    Validate and normalise a raw filter dict coming from the API request.

    - Only keys in ALLOWED_FILTER_KEYS are kept.
    - Empty / whitespace-only values are stripped so they never produce
      invalid Milvus boolean expressions.
    - Unknown keys are logged as warnings and dropped silently.

    Returns a clean dict (may be empty if no valid filters were supplied).

    Examples
    ────────
    build_filters({"industry": "Fintech", "foo": "bar"})
    → {"industry": "Fintech"}          # "foo" dropped

    build_filters({"doc_type": "Policy", "tags": "HR"})
    → {"doc_type": "Policy", "tags": "HR"}

    build_filters({"industry": "  ", "version": ""})
    → {}                               # both values are blank
    """
    if not raw:
        logger.info("   ℹ️  build_filters: no raw filters supplied — returning {}")
        return {}

    clean: dict = {}
    dropped: list[str] = []

    for key, value in raw.items():
        if key not in ALLOWED_FILTER_KEYS:
            dropped.append(key)
            continue
        cleaned_value = str(value).strip() if value is not None else ""
        if cleaned_value:
            clean[key] = cleaned_value

    if dropped:
        logger.warning(
            "   ⚠️  build_filters: unknown filter key(s) dropped: %s", dropped
        )

    logger.info("   ✅ build_filters: clean filters = %s", clean)
    return clean