"""
Schema formatting and inspection helpers for the DocForge Hub agent.

These utilities parse the `required_section` document schema (from MongoDB)
and convert it into prompt-ready text or answer structural questions
(e.g. "is this a table-only schema?").

All functions are pure — they take a schema dict and return a value,
with no LLM calls, database access, or side effects.
"""

import json
import logging

logger = logging.getLogger("agent.schema_helpers")


# ═══════════════════════════════════════════════════════════════
#  Q&A → Prompt text
# ═══════════════════════════════════════════════════════════════

def format_questions_and_answers_for_prompt(qa_list: list[dict]) -> str:
    """
    Convert a list of Q&A dicts into a readable Markdown block
    suitable for embedding in an LLM system prompt.

    Groups questions under `### Category` headers when the category changes.
    Handles structured_list answers (JSON) and plain list answers.
    """
    lines = []
    current_category = ""

    for qa_item in qa_list:
        category = qa_item.get("category", "General")
        if category != current_category:
            current_category = category
            lines.append(f"\n### {category}")

        question_text = qa_item.get("question", "")
        answer_value = qa_item.get("answer", "")

        if qa_item.get("answer_type") == "structured_list" and qa_item.get("answers"):
            answer_value = json.dumps(qa_item["answers"], indent=2)
        elif isinstance(answer_value, list):
            answer_value = ", ".join(str(item) for item in answer_value)

        lines.append(f"**Q:** {question_text}")
        lines.append(f"**A:** {answer_value if answer_value else '(not provided)'}")
        lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  Schema → Prompt text
# ═══════════════════════════════════════════════════════════════

def format_required_section_for_prompt(required_section: dict) -> str:
    """
    Convert a `required_section` schema dict into a human-readable
    Markdown outline that describes the expected document structure.

    Handles:
      - Table-only schemas (type=table, no subsections)
      - Mixed schemas (title + flat subsections array)
      - Legacy question_categories fallback
    """
    sections = required_section.get("sections", [])
    document_name = required_section.get("document_name", "")

    if not sections:
        categories = required_section.get("question_categories", [])
        if categories:
            return "\n".join(
                f"- {cat.get('category', 'Unknown')} (order: {cat.get('order', 0)})"
                for cat in categories
            )
        return "No schema sections available"

    lines = []
    for section in sections:
        if section.get("type") == "table" and not section.get("subsections"):
            # Table-only schema: section has type/columns/order but no title or subsections
            table_title = section.get("title", "").strip() or document_name or "Data Table"
            columns = section.get("columns", [])
            lines.append(f"## {table_title}")
            lines.append("")
            lines.append("⚠️  TABLE FORMAT REQUIRED — This entire document is a Markdown table.")
            lines.append(f"Column headers: | {' | '.join(columns)} |")
            lines.append("You MUST output a real Markdown table with these exact columns")
            lines.append("and at least 4-6 realistic data rows based on the user's answers.")
            lines.append("Do NOT describe the table — OUTPUT THE TABLE ITSELF.")
            lines.append("")
            continue

        # Mixed schema: section has a title + flat subsections array
        section_title = section.get("title", "Untitled Section")
        lines.append(f"## {section_title}")

        for subsection in section.get("subsections", []):
            sub_title = subsection.get("title", "")
            sub_type = subsection.get("type", "text")
            columns = subsection.get("columns", [])

            if sub_type == "table" and columns:
                lines.append(f"  - {sub_title} ⚠️ TABLE — columns: | {' | '.join(columns)} |")
                lines.append(f"    (Output a real Markdown table with these columns and realistic rows)")
            else:
                lines.append(f"  - {sub_title} (type: {sub_type})")

        lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  Schema inspection queries
# ═══════════════════════════════════════════════════════════════

def is_table_only_schema(required_section: dict) -> bool:
    """
    Return True if every section in the schema is type='table'
    with no subsections — meaning the entire document is one big table.
    """
    sections = required_section.get("sections", [])
    if not sections:
        return False
    try:
        debug_info = [
            f"type={schema_section.get('type')}, subs={bool(schema_section.get('subsections'))}"
            for schema_section in sections
        ]
        logger.info("   🔍 Checking is_table_only_schema: %s", debug_info)
    except Exception:
        pass
    return all(
        schema_section.get("type") == "table" and not schema_section.get("subsections")
        for schema_section in sections
    )


def get_table_columns(required_section: dict) -> list[str]:
    """Return the column list from the first table-type section, or []."""
    for section in required_section.get("sections", []):
        if section.get("type") == "table":
            return section.get("columns", [])
    return []


def get_table_section_title(required_section: dict) -> str:
    """
    Return the display title for a table-only schema.

    Handles schemas where the table section omits the 'title' key entirely,
    falling back to 'document_name' at the top level of required_section,
    then 'document_type', then a safe generic label.

    This covers the Change Request Log pattern:
        { "type": "table", "columns": [...], "order": 1 }  ← no 'title' key

    Fallback chain:
        section["title"]  →  required_section["document_name"]
                          →  required_section["document_type"]
                          →  "Data Table"
    """
    for section in required_section.get("sections", []):
        if section.get("type") == "table":
            title = section.get("title", "").strip()
            if title:
                return title
    # Fall back to document-level name fields
    return (
        required_section.get("document_name", "").strip()
        or required_section.get("document_type", "").strip()
        or "Data Table"
    )