import os
import json
import logging
import asyncio
from typing import TypedDict, Literal
from dotenv import load_dotenv
import re

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from agent.prompts import (
    build_system_prompt,
    build_table_only_prompt,
    build_gap_filler_prompt,
    build_quality_review_prompt,
)
from agent.schema_helpers import (
    format_questions_and_answers_for_prompt,
    format_required_section_for_prompt,
    is_table_only_schema,
    get_table_columns,
    get_table_section_title,
)
from agent.validation_helpers import (
    validate_document_structure,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("agent.agent_graph")

load_dotenv()

# ── Primary document-generation LLM ─────────────────────────────
llm = AzureChatOpenAI(
    api_key=os.getenv("AZURE_OPENAI_LLM_KEY"),
    azure_endpoint=os.getenv("AZURE_LLM_ENDPOINT"),
    api_version=os.getenv("AZURE_LLM_API_VERSION"),
    azure_deployment=os.getenv("AZURE_LLM_DEPLOYMENT_41_MINI"),
    temperature=0.1,
    max_tokens=8192,
)

# ── Dedicated question-generation LLM (lighter, faster) ──────────
# Using a separate model keeps the question-analysis step cheap and
# avoids burning the main model's context window on schema analysis.
question_gen_llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",   # fast, efficient for structured output
    temperature=0.2,
    max_tokens=2048,
)


# ═══════════════════════════════════════════════════════════════
#  AgentState
# ═══════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    """
    All the data that moves through the graph.

    Inputs (provided when you start the agent):
        department               – e.g. "Product Management"
        document_type            – e.g. "Feature Prioritization Framework"
        questions_and_answers    – list of dicts [{question, answer, category, ...}]
        required_section         – the document schema from MongoDB

    Intermediates / Outputs (filled in by the nodes):
        gap_questions            – NEW: list of generated questions for uncovered sections
        supplementary_content    – synthesized content for uncovered schema sections
        system_prompt            – the full prompt sent to the LLM
        generated_document       – the Markdown document the LLM created
        quality_scores           – dict of scores from LLM quality review
        quality_issues           – list of problems found by the quality gate
        quality_suggestions      – list of improvement suggestions
        retry_count              – how many times we've asked the LLM to fix the doc
        status                   – "generating" | "passed" | "failed"
    """
    department: str
    document_type: str
    questions_and_answers: list[dict]
    required_section: dict

    # NEW — populated by analyze_schema_gaps
    gap_questions: list[dict]          # [{question, category, answer_type, options?}]
    supplementary_content: str         # synthesized filler for uncovered sections

    system_prompt: str
    generated_document: str
    quality_scores: dict
    quality_issues: list[str]
    quality_suggestions: list[str]
    retry_count: int
    status: str


# ═══════════════════════════════════════════════════════════════
#  Formatting & schema helpers — imported from agent.schema_helpers
#  Validation helpers         — imported from agent.validation_helpers
# ═══════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════
#  NODE 1 (NEW): analyze_schema_gaps
# ═══════════════════════════════════════════════════════════════

_GAP_QUESTION_SYSTEM_PROMPT = """\
You are a document-requirements analyst. Your job is to compare a document schema
against a set of existing Q&A answers and identify which schema sections have
INSUFFICIENT information to write good content.

Rules:
1. A section is "covered" if at least one existing QUESTION (not just its answer)
   already asks for the same information — regardless of wording differences.
   "What authentication method is used?" and "Which auth scheme does the API use?"
   cover the same information need — do NOT generate a duplicate.
2. A section is "uncovered" only if NO existing question addresses that information
   need at all.
3. For each genuinely uncovered section, generate EXACTLY ONE targeted question.
4. Questions must be practical, specific, and answerable in 1-3 sentences.
5. You MUST include a "why_not_duplicate" field explaining why this question is not
   already covered by any existing question. Be specific — name the existing question
   it might seem similar to and explain the difference.
6. Output ONLY valid JSON — no markdown, no explanation, no preamble.

FORBIDDEN: Do NOT generate questions that are paraphrases of existing questions.
Examples of forbidden duplicates:
  Existing: "What is the API base URL?"  →  Forbidden: "What URL should clients use?"
  Existing: "What auth method is used?"  →  Forbidden: "How do clients authenticate?"
  Existing: "What are the rate limits?"  →  Forbidden: "How many requests per second?"

Output format (JSON array):
[
  {
    "question": "What are the key risks and mitigation strategies for this project?",
    "category": "Risk Management",
    "answer_type": "text",
    "section_covered": "Risk Assessment",
    "why_not_duplicate": "No existing question asks about risks or mitigations."
  },
  ...
]

If ALL sections are already covered, return an empty array: []
"""


def _extract_key_terms(text: str) -> set[str]:
    """Lowercase words >3 chars with stop words removed — used for Jaccard dedup."""
    stop = {
        "what", "which", "who", "how", "when", "where", "that", "this",
        "with", "your", "have", "will", "does", "from", "about", "into",
        "should", "would", "could", "used", "uses", "using", "does",
    }
    return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in stop}


def _deduplicate_gap_questions(
    gap_questions: list[dict],
    existing_questions: list[str],
    threshold: float = 0.5,
) -> list[dict]:
    """
    Post-generation Jaccard similarity filter.

    Removes any gap question whose key-term overlap with ANY existing question
    exceeds `threshold` (default 50%). Also deduplicates within the gap
    questions themselves in case the LLM generated two near-identical ones.
    """
    def jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    existing_term_sets = [_extract_key_terms(q) for q in existing_questions]
    kept: list[dict] = []
    kept_terms: list[set] = []

    for gq in gap_questions:
        gq_terms = _extract_key_terms(gq.get("question", ""))

        # Check against existing questions
        duplicate = any(
            jaccard(gq_terms, ex_terms) >= threshold
            for ex_terms in existing_term_sets
        )
        if duplicate:
            logger.info(
                "   🗑️  Dedup filtered (vs existing): '%s'", gq.get("question", "")[:80]
            )
            continue

        # Check within the gap questions already kept
        duplicate = any(
            jaccard(gq_terms, kept_t) >= threshold
            for kept_t in kept_terms
        )
        if duplicate:
            logger.info(
                "   🗑️  Dedup filtered (vs gap peers): '%s'", gq.get("question", "")[:80]
            )
            continue

        kept.append(gq)
        kept_terms.append(gq_terms)

    return kept


def analyze_schema_gaps(state: AgentState) -> dict:
    """
    NODE 1: Analyze schema vs existing Q&A to identify coverage gaps.

    Deduplication strategy (two layers):
      1. LLM-level: existing question texts are injected into the prompt so the
         LLM can compare question-to-question (not just answer-to-answer). The
         why_not_duplicate reasoning field scaffolds the LLM into checking each
         candidate against the list before emitting it.
      2. Post-generation: _deduplicate_gap_questions() applies Jaccard similarity
         on key terms to catch any paraphrased duplicates that slipped through.
    """
    logger.info("🔎 Node: analyze_schema_gaps — scanning schema coverage...")

    formatted_schema = format_required_section_for_prompt(state["required_section"])
    formatted_answers = format_questions_and_answers_for_prompt(state["questions_and_answers"])

    # Extract existing question texts (skip _-prefixed internal entries)
    existing_questions = [
        qa["question"]
        for qa in state["questions_and_answers"]
        if not qa.get("category", "").startswith("_") and qa.get("question")
    ]
    existing_questions_block = "\n".join(
        f"{i+1}. {q}" for i, q in enumerate(existing_questions)
    ) if existing_questions else "(none)"

    user_message = f"""
DOCUMENT TYPE: {state['document_type']}
DEPARTMENT: {state['department']}

=== SCHEMA (sections that must be covered) ===
{formatted_schema}

=== EXISTING QUESTIONS (do NOT duplicate any of these) ===
{existing_questions_block}

=== EXISTING Q&A ANSWERS ===
{formatted_answers}

Identify genuinely uncovered sections and generate gap questions now.
Remember: check every candidate against the existing questions list above before including it.
"""

    try:
        messages = [
            SystemMessage(content=_GAP_QUESTION_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]
        response = question_gen_llm.invoke(messages)
        raw = response.content.strip()

        # Strip any accidental markdown fences
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        gap_questions: list[dict] = json.loads(raw)

        if not gap_questions:
            logger.info("   ✅ All schema sections are covered — no gap questions needed")
            return {"gap_questions": [], "supplementary_content": ""}

        # Strip the why_not_duplicate scaffold field before returning
        for q in gap_questions:
            q.pop("why_not_duplicate", None)

        # Post-generation deduplication safety net
        gap_questions = _deduplicate_gap_questions(gap_questions, existing_questions)

        logger.info(
            "   📝 Found %d schema gap(s) — questions generated for: %s",
            len(gap_questions),
            ", ".join(q.get("section_covered", "?") for q in gap_questions),
        )

        supplementary_lines = []
        for gap_question in gap_questions:
            section = gap_question.get("section_covered", "")
            supplementary_lines.append(
                f"**{section}**: This section requires additional information. "
                f"Gap question pending user answer: \"{gap_question['question']}\""
            )

        supplementary_content = "\n".join(supplementary_lines) if supplementary_lines else ""

        return {
            "gap_questions": gap_questions,
            "supplementary_content": supplementary_content,
        }

    except (json.JSONDecodeError, Exception) as error_msg:
        logger.warning("   ⚠️  analyze_schema_gaps failed (non-critical): %s", error_msg)
        return {"gap_questions": [], "supplementary_content": ""}


# ═══════════════════════════════════════════════════════════════
#  NODE 2: build_prompt  (unchanged logic, updated field names)
# ═══════════════════════════════════════════════════════════════

def build_prompt(state: AgentState) -> dict:
    """
    NODE 2: Assemble the full system prompt.

    Combines:
        - The document schema (required_section)
        - The user's Q&A answers (including any gap-question answers)
        - Supplementary content notes from analyze_schema_gaps

    For TABLE-ONLY schemas uses the strict table-only prompt.
    For mixed schemas, appends a strict section allowlist to the prompt so
    the LLM knows upfront exactly which headings to include and nothing more.
    Uses get_table_section_title() to correctly resolve the document title
    for schemas that omit 'title' on the section (e.g. Change Request Log).
    """
    logger.info("📝 Node: build_prompt — assembling system prompt")

    formatted_answers = format_questions_and_answers_for_prompt(
        state["questions_and_answers"]
    )

    if is_table_only_schema(state["required_section"]):
        columns = get_table_columns(state["required_section"])
        # Use get_table_section_title to handle schemas without a section-level 'title'
        table_title = get_table_section_title(state["required_section"])
        logger.info("   📊 Table-only schema — title=%s, columns: %s", table_title, ", ".join(columns))
        system_prompt = build_table_only_prompt(
            department=state["department"],
            document_type=table_title,
            columns=columns,
            questions_and_answers=formatted_answers,
            supplementary_content=state.get("supplementary_content", ""),
        )
    else:
        formatted_schema = format_required_section_for_prompt(state["required_section"])

        # Build the strict allowlist of required headings from the schema subsections
        # and inject it into the prompt so the LLM knows exactly what to generate.
        # The LLM must use ONLY these headings — no additions, renames, or omissions.
        required_headings = []
        for schema_section in state["required_section"].get("sections", []):
            for subsection_item in sorted(schema_section.get("subsections", []), key=lambda s: s.get("order", 0)):
                title = subsection_item.get("title", "").strip()
                if title:
                    required_headings.append(title)

        if required_headings:
            headings_list = "\n".join(f"  - {t}" for t in required_headings)
            strict_rule = (
                f"\n\n⚠️  STRICT SECTION RULE:\n"
                f"Your document MUST contain ALL of the following headings and NO others.\n"
                f"Do NOT add, rename, merge, reorder, or omit any heading:\n"
                f"{headings_list}\n"
            )
        else:
            strict_rule = ""

        system_prompt = build_system_prompt(
            department=state["department"],
            document_type=state["document_type"],
            required_section=formatted_schema + strict_rule,
            questions_and_answers=formatted_answers,
            supplementary_content=state.get("supplementary_content", ""),
        )

    logger.info(
        "   ✅ Prompt built — %d chars, department=%s, document=%s, answers=%d",
        len(system_prompt),
        state["department"],
        state["document_type"],
        len(state["questions_and_answers"]),
    )

    return {
        "system_prompt": system_prompt,
        "retry_count": 0,
        "status": "generating",
    }


# ═══════════════════════════════════════════════════════════════
#  NODE 3: generate_document  (unchanged)
# ═══════════════════════════════════════════════════════════════

def generate_document(state: AgentState) -> dict:
    """NODE 3: Call the primary LLM to generate the Markdown document."""
    logger.info("🤖 Node: generate_document — calling LLM...")

    if is_table_only_schema(state["required_section"]):
        # Use get_table_section_title so the instruction names the document correctly
        # even when the schema section omits 'title' (e.g. Change Request Log pattern)
        table_title = get_table_section_title(state["required_section"])
        human_instruction = (
            f"Generate the {table_title} as a Markdown table now. "
            f"Output ONLY the heading and table — no introductions, no descriptions, "
            f"no extra sections. Just the title and the table with data rows."
        )
    else:
        human_instruction = (
            f"Generate the complete {state['document_type']} document now. "
            f"Remember: elevate every answer into professional, industry-grade prose. "
            f"Do NOT copy answers verbatim."
        )

    messages = [
        SystemMessage(content=state["system_prompt"]),
        HumanMessage(content=human_instruction),
    ]

    llm_response = llm.invoke(messages)
    generated_text = llm_response.content

    logger.info("   ✅ LLM returned %d characters of Markdown", len(generated_text))
    return {"generated_document": generated_text}


# ═══════════════════════════════════════════════════════════════
#  Helper: Structure Validator
#
#  Handles the two real MongoDB schema patterns:
#
#  Pattern A — Table-only (Change Request Log):
#    sections: [{ type: "table", columns: [...], order: 1 }]
#    → No titled headings to validate; quality_gate handles column checking.
#    → Returns [] immediately.
#
#  Pattern B — Mixed with flat subsections (Feature Prioritization Framework):
#    sections: [{ title: "1. Objective", subsections: [{title, type, order}, ...] }]
#    → The parent title ("1. Objective") is NOT a required document heading.
#    → Every subsection title IS a required heading — checked in order.
#    → Two-way enforcement:
#        CHECK 1: every subsection title must appear as a heading in the doc.
#        CHECK 2: no heading in the doc may be absent from the schema allowlist.
#        CHECK 3: type="table" subsections must contain a Markdown table with correct columns.
# ═══════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════
#  NODE 4: quality_gate  (unchanged)
# ═══════════════════════════════════════════════════════════════

def quality_gate(state: AgentState) -> dict:
    """NODE 4: Validate the generated document."""
    logger.info("🔍 Node: quality_gate — reviewing document quality...")

    document_text = state.get("generated_document", "")

    # TABLE-ONLY: deterministic validation
    if is_table_only_schema(state["required_section"]):
        logger.info("   📊 Table-only schema — using deterministic validation")

        expected_columns = get_table_columns(state["required_section"])
        # Use get_table_section_title to correctly resolve title for schemas
        # that omit the 'title' key on the section (e.g. Change Request Log pattern)
        doc_name = get_table_section_title(state["required_section"])

        lines = document_text.split("\n")
        table_lines = []
        heading_line = ""

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("# ") and not stripped.startswith("## "):
                heading_line = stripped
            elif "|" in stripped and stripped.startswith("|"):
                table_lines.append(stripped)

        if len(table_lines) < 3:
            logger.warning("   ❌ No Markdown table found in output")
            return {
                "quality_scores": {},
                "quality_issues": [
                    f"TABLE-ONLY SCHEMA: No Markdown table found. "
                    f"Output ONLY: # {doc_name} + a table with columns: "
                    f"{', '.join(expected_columns)}."
                ],
                "quality_suggestions": [],
                "status": "failed",
            }

        header_line = table_lines[0]
        actual_columns = [col.strip() for col in header_line.split("|") if col.strip()]
        expected_normalized = [col.lower().strip() for col in expected_columns]
        actual_normalized = [col.lower().strip() for col in actual_columns]

        if expected_normalized != actual_normalized:
            logger.warning("   ❌ Column mismatch")
            return {
                "quality_scores": {},
                "quality_issues": [
                    f"Wrong columns. Expected: | {' | '.join(expected_columns)} | "
                    f"Got: | {' | '.join(actual_columns)} |"
                ],
                "quality_suggestions": [],
                "status": "failed",
            }

        if not heading_line:
            heading_line = f"# {doc_name}"

        cleaned_output = heading_line + "\n\n" + "\n".join(table_lines) + "\n"
        logger.info(
            "   ✅ Table-only validation PASSED — %d columns, %d data rows",
            len(actual_columns),
            len(table_lines) - 2,
        )
        return {
            "generated_document": cleaned_output,
            "quality_scores": {"structure": 5, "completeness": 5},
            "quality_issues": [],
            "quality_suggestions": [],
            "status": "passed",
        }

    # MIXED SCHEMAS: strict two-way structural check + LLM quality review
    structure_errors = validate_document_structure(document_text, state["required_section"])
    if structure_errors:
        logger.warning("   ❌ Structural validation failed with %d errors", len(structure_errors))

        # Split errors by type so fix_document gets targeted instructions
        missing = [error_msg for error_msg in structure_errors if error_msg.startswith("Missing")]
        extra   = [error_msg for error_msg in structure_errors if error_msg.startswith("Extra")]
        table   = [error_msg for error_msg in structure_errors if error_msg.startswith("Section")]

        suggestions = []
        if missing:
            suggestions.append(
                "Add ALL missing sections using their EXACT titles from the schema — do not rename them."
            )
        if extra:
            suggestions.append(
                "REMOVE every heading not in the schema. "
                "The document must contain ONLY the sections defined in the schema — nothing more."
            )
        if table:
            suggestions.append(
                "Ensure every table section contains a real Markdown table "
                "with the exact column headers specified in the schema."
            )

        return {
            "quality_scores": {"structure": 1},
            "quality_issues": structure_errors,
            "quality_suggestions": suggestions,
            "status": "failed",
        }

    logger.info("   ✅ Structural validation PASSED")

    try:
        review_prompt = build_quality_review_prompt(
            department=state["department"],
            document_type=state["document_type"],
            generated_document=document_text,
        )
        messages = [
            SystemMessage(content=review_prompt),
            HumanMessage(content="Review the document and return the JSON assessment now."),
        ]
        review_response = llm.invoke(messages)
        review_text = review_response.content

        json_text = review_text
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0].strip()
        elif "```" in json_text:
            json_text = json_text.split("```")[1].split("```")[0].strip()

        review_result = json.loads(json_text)
        scores = review_result.get("scores", {})
        overall_score = review_result.get("overall_score", 3)
        passed = review_result.get("passed", overall_score >= 3)
        issues = review_result.get("issues", [])
        suggestions = review_result.get("suggestions", [])

        logger.info("   📊 Overall: %d/5 — %s", overall_score, "PASSED" if passed else "FAILED")

        if passed:
            return {
                "quality_scores": scores,
                "quality_issues": [],
                "quality_suggestions": suggestions,
                "status": "passed",
            }
        else:
            return {
                "quality_scores": scores,
                "quality_issues": issues,
                "quality_suggestions": suggestions,
                "status": "failed",
            }

    except Exception as review_error:
        logger.warning("   ⚠️  LLM quality review failed, falling back to rules: %s", review_error)

    # Fallback rule-based checks
    issues_found = []
    if len(document_text) < 500:
        issues_found.append("Document is too short (< 500 chars)")

    forbidden = ["TBD", "to be decided", "[Company Name]", "[Insert", "Lorem ipsum"]
    for phrase in forbidden:
        if phrase.lower() in document_text.lower():
            issues_found.append(f"Contains placeholder: '{phrase}'")

    if document_text.count("\n#") < 5:
        issues_found.append("Too few sections (expected at least 5 headings)")

    sections_split = document_text.split("\n## ")
    # Get thin sections (excluding title part)
    thin = [section_text for section_text in sections_split[1:] if len(section_text.strip()) < 100]
    if thin:
        issues_found.append(f"{len(thin)} sections are too thin — expand with detail")

    if issues_found:
        return {
            "quality_scores": {},
            "quality_issues": issues_found,
            "quality_suggestions": [],
            "status": "failed",
        }

    return {
        "quality_scores": {},
        "quality_issues": [],
        "quality_suggestions": [],
        "status": "passed",
    }


# ═══════════════════════════════════════════════════════════════
#  NODE 5: fix_document  (unchanged)
# ═══════════════════════════════════════════════════════════════

def fix_document(state: AgentState) -> dict:
    """NODE 5: Ask the LLM to fix quality issues in the document."""
    current_retry = state["retry_count"] + 1
    logger.info("🔧 Node: fix_document — retry %d/2...", current_retry)

    issues_text = "\n".join(f"- {issue_msg}" for issue_msg in state["quality_issues"])
    suggestions_text = "\n".join(f"- {suggestion_msg}" for suggestion_msg in state.get("quality_suggestions", []))

    fix_instruction = f"""The following document was generated but failed quality review:

--- DOCUMENT START ---
{state['generated_document']}
--- DOCUMENT END ---

## Quality Issues Found:
{issues_text}
"""
    if suggestions_text:
        fix_instruction += f"\n## Reviewer Suggestions:\n{suggestions_text}\n"

    fix_instruction += """
## Instructions:
1. Fix ALL the issues listed above.
2. Expand any thin or superficial sections.
3. Ensure every section has at least 2-3 detailed sentences.
4. Remove all placeholder text.
5. Add concrete metrics, timelines, or action items where appropriate.
6. Output ONLY the corrected Markdown document — no commentary."""

    messages = [
        SystemMessage(content=state["system_prompt"]),
        HumanMessage(content=fix_instruction),
    ]
    llm_response = llm.invoke(messages)
    logger.info("   ✅ Fixed document: %d characters", len(llm_response.content))
    return {"generated_document": llm_response.content, "retry_count": current_retry}


# ═══════════════════════════════════════════════════════════════
#  Routing
# ═══════════════════════════════════════════════════════════════

def decide_after_quality_gate(state: AgentState) -> Literal["fix_document", "end"]:
    if state["status"] == "passed":
        logger.info("✅ Routing → END (quality gate passed)")
        return "end"
    if state["retry_count"] >= 2:
        logger.warning("⚠️  Routing → END (max retries reached)")
        return "end"
    logger.info("🔄 Routing → fix_document")
    return "fix_document"


# ═══════════════════════════════════════════════════════════════
#  Graph assembly
# ═══════════════════════════════════════════════════════════════

def build_document_generation_graph() -> StateGraph:
    """
    Graph topology:

        START
          ↓
        analyze_schema_gaps   ← NEW: lightweight LLM identifies gaps + generates questions
          ↓
        build_prompt
          ↓
        generate_document
          ↓
        quality_gate ──(failed, retry < 2)──→ fix_document ──┐
          ↓ (passed)                                          ↓
         END                                              quality_gate
    """
    logger.info("🔨 Building document generation graph...")

    graph = StateGraph(AgentState)

    graph.add_node("analyze_schema_gaps", analyze_schema_gaps)
    graph.add_node("build_prompt", build_prompt)
    graph.add_node("generate_document", generate_document)
    graph.add_node("quality_gate", quality_gate)
    graph.add_node("fix_document", fix_document)

    graph.set_entry_point("analyze_schema_gaps")
    graph.add_edge("analyze_schema_gaps", "build_prompt")
    graph.add_edge("build_prompt", "generate_document")
    graph.add_edge("generate_document", "quality_gate")
    graph.add_conditional_edges(
        "quality_gate",
        decide_after_quality_gate,
        {"fix_document": "fix_document", "end": END},
    )
    graph.add_edge("fix_document", "quality_gate")

    compiled = graph.compile()
    logger.info("✅ Graph compiled — 5 nodes, entry=analyze_schema_gaps")
    return compiled


document_generation_agent = build_document_generation_graph()


# ═══════════════════════════════════════════════════════════════
#  Lean graph for progressive section generation
#
#  Skips analyze_schema_gaps entirely — gap analysis is a one-time
#  pre-generation step, not something that should run per section.
#  This eliminates the Groq LLM call that caused 413/429 errors and
#  a wasted ~2-3 s on every section request.
#
#  Topology: build_prompt → generate_document → quality_gate → fix_document
# ═══════════════════════════════════════════════════════════════

def build_section_generation_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("build_prompt", build_prompt)
    graph.add_node("generate_document", generate_document)
    graph.add_node("quality_gate", quality_gate)
    graph.add_node("fix_document", fix_document)
    graph.set_entry_point("build_prompt")
    graph.add_edge("build_prompt", "generate_document")
    graph.add_edge("generate_document", "quality_gate")
    graph.add_conditional_edges(
        "quality_gate",
        decide_after_quality_gate,
        {"fix_document": "fix_document", "end": END},
    )
    graph.add_edge("fix_document", "quality_gate")
    compiled = graph.compile()
    logger.info("✅ Section graph compiled — 4 nodes, no gap analysis, entry=build_prompt")
    return compiled


section_generation_agent = build_section_generation_graph()


# ═══════════════════════════════════════════════════════════════
#  Memory summariser
#
#  Condenses previously-generated sections into a ~1500-char
#  "decisions & terminology" digest. This gives the next section's
#  LLM call full awareness of what was written — without the risk
#  of hallucination that comes from arbitrary tail-truncation,
#  and without the token cost of sending the full accumulated doc.
# ═══════════════════════════════════════════════════════════════

_MEMORY_SUMMARY_PROMPT = """\
You are a document consistency assistant. You will be given the text of \
previously generated sections of a {document_type} document.

Produce a concise CONSISTENCY DIGEST in under 1500 characters. Cover:
1. Key decisions, names, versions, and terminology already used
2. Tone and writing style (formal/technical/etc.)
3. Any specific values, numbers, or policies already stated
4. Section titles already written (so the next section does not repeat them)

Rules:
- Write in compact bullet points, not prose.
- Do NOT summarise content verbatim — extract only what is needed for \
the next section to stay consistent.
- Do NOT add any new information not present in the source text.
- Output ONLY the digest — no preamble, no headings.

Previously generated sections:
{doc_memory}
"""

# Bypass threshold: if doc_memory is already under this, skip the LLM call.
_MEMORY_SUMMARY_THRESHOLD = 1_500


def _summarise_doc_memory(doc_memory: str, document_type: str) -> str:
    """
    Summarise accumulated section text into a compact consistency digest.

    Uses the main Azure LLM (not Groq) to avoid burning the Groq daily quota.
    Falls back to tail-truncation if the LLM call fails, so generation is
    never blocked by a summarisation error.
    """
    if not doc_memory.strip() or len(doc_memory) <= _MEMORY_SUMMARY_THRESHOLD:
        return doc_memory  # already short enough — no LLM call needed

    logger.info(
        "   🗜️  Summarising doc_memory: %d chars → target ≤%d chars",
        len(doc_memory), _MEMORY_SUMMARY_THRESHOLD,
    )
    try:
        prompt = _MEMORY_SUMMARY_PROMPT.format(
            document_type=document_type,
            doc_memory=doc_memory,
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        summary = response.content.strip()
        logger.info(
            "   ✅ Memory summarised: %d → %d chars", len(doc_memory), len(summary)
        )
        return summary
    except Exception as err:
        # Non-fatal: fall back to keeping the most recent 1500 chars
        logger.warning(
            "   ⚠️  Memory summarisation failed (%s) — falling back to tail truncation", err
        )
        return "…[earlier sections condensed]\n\n" + doc_memory[-_MEMORY_SUMMARY_THRESHOLD:]


# ═══════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════

async def run_agent(
    department: str,
    document_type: str,
    questions_and_answers: list[dict],
    required_section: dict,
) -> dict:
    """Run the full document generation agent (single-shot mode)."""
    logger.info(
        "🚀 run_agent — department=%s, document_type=%s, answers=%d",
        department, document_type, len(questions_and_answers),
    )
    initial_state: AgentState = {
        "department": department,
        "document_type": document_type,
        "questions_and_answers": questions_and_answers,
        "required_section": required_section,
        "gap_questions": [],
        "supplementary_content": "",
        "system_prompt": "",
        "generated_document": "",
        "quality_scores": {},
        "quality_issues": [],
        "quality_suggestions": [],
        "retry_count": 0,
        "status": "generating",
    }
    final_state = await asyncio.to_thread(
        document_generation_agent.invoke, initial_state
    )
    logger.info(
        "🏁 Agent finished — status=%s, retries=%d, doc=%d chars, gap_questions=%d",
        final_state.get("status", "unknown"),
        final_state.get("retry_count", 0),
        len(final_state.get("generated_document", "")),
        len(final_state.get("gap_questions", [])),
    )
    return {
        "generated_document": final_state.get("generated_document", ""),
        "gap_questions": final_state.get("gap_questions", []),
        "status": final_state.get("status", "unknown"),
        "quality_issues": final_state.get("quality_issues", []),
        "quality_scores": final_state.get("quality_scores", {}),
        "quality_suggestions": final_state.get("quality_suggestions", []),
        "retry_count": final_state.get("retry_count", 0),
    }


# ═══════════════════════════════════════════════════════════════
#  Standalone gap-analysis utility (used by /gap-questions endpoint)
# ═══════════════════════════════════════════════════════════════

async def analyze_gaps_only(
    department: str,
    document_type: str,
    questions_and_answers: list[dict],
    required_section: dict,
) -> list[dict]:
    """Run ONLY schema-gap analysis — no document generation."""
    state: AgentState = {
        "department": department,
        "document_type": document_type,
        "questions_and_answers": questions_and_answers,
        "required_section": required_section,
        "gap_questions": [],
        "supplementary_content": "",
        "system_prompt": "",
        "generated_document": "",
        "quality_scores": {},
        "quality_issues": [],
        "quality_suggestions": [],
        "retry_count": 0,
        "status": "generating",
    }
    result = await asyncio.to_thread(analyze_schema_gaps, state)
    return result.get("gap_questions", [])


# ═══════════════════════════════════════════════════════════════
#  Progressive: generate ONE section with controlled prompt size
#
#  Three controls keep every section prompt under ~15 k chars:
#
#  1. LEAN GRAPH  — section_generation_agent skips analyze_schema_gaps.
#                   No Groq call per section → no 413/429 errors.
#
#  2. QA FILTERING — only Q&A whose category matches the target
#     subsection is included (+ ≤3 global-context answers).
#
#  3. MEMORY SUMMARY — doc_memory is condensed by the LLM into a
#     compact consistency digest instead of being arbitrarily
#     truncated. The LLM sees every key decision made so far —
#     just compressed — eliminating hallucination of already-written
#     content while keeping token cost flat.
# ═══════════════════════════════════════════════════════════════

def _extract_key_terms_for_filter(text: str) -> set[str]:
    """Lowercase words >3 chars, stop-words removed — for QA category matching."""
    stop = {
        "what", "which", "that", "this", "with", "your", "have",
        "will", "does", "from", "about", "into", "when", "where",
    }
    return {w for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in stop}


async def generate_single_section(
    department: str,
    document_type: str,
    section: dict,
    questions_and_answers: list[dict],
    doc_memory: str = "",
) -> str:
    """Generate ONE section using the lean graph, filtered QA, and summarised memory."""

    # ── Strip all _-prefixed UI internal keys from subsection dicts ───────────
    raw_subsections = section.get("subsections", [])
    clean_subsections = [
        {k: v for k, v in sub.items() if not k.startswith("_")}
        for sub in raw_subsections
    ]
    subsection_names = [s.get("title", "") for s in clean_subsections]

    logger.info(
        "📝 generate_single_section — parent='%s', subsections=%s, qa_total=%d, memory=%d chars",
        section.get("title", "Untitled"),
        subsection_names,
        len(questions_and_answers),
        len(doc_memory),
    )

    # ── Scoped required_section ────────────────────────────────────────────────
    parent_section_title = section.get(
        "title", clean_subsections[0]["title"] if clean_subsections else "Document"
    )
    scoped_required_section = {
        "document_type": document_type,
        "document_name": document_type,
        "sections": [{"title": parent_section_title, "subsections": clean_subsections}],
    }

    # ── 1. Filter QA to relevant answers only ─────────────────────────────────
    target_terms = set()
    for name in subsection_names:
        target_terms |= _extract_key_terms_for_filter(name)

    relevant_qa: list[dict] = []
    global_ctx_qa: list[dict] = []

    for qa in questions_and_answers:
        cat = qa.get("category", "")
        if cat.startswith("_"):
            continue
        cat_terms = _extract_key_terms_for_filter(cat)
        if target_terms & cat_terms:
            relevant_qa.append(qa)
        elif len(global_ctx_qa) < 3:
            global_ctx_qa.append(qa)

    if not relevant_qa:
        relevant_qa = [q for q in questions_and_answers if not q.get("category", "").startswith("_")]
        global_ctx_qa = []

    filtered_qa = relevant_qa + global_ctx_qa
    logger.info(
        "   🔎 QA: %d relevant + %d ctx = %d sent (from %d total)",
        len(relevant_qa), len(global_ctx_qa), len(filtered_qa),
        len([q for q in questions_and_answers if not q.get("category", "").startswith("_")]),
    )

    # ── 2. Summarise doc_memory into a consistency digest ─────────────────────
    # Run in a thread since it makes a synchronous LLM call.
    condensed_memory = await asyncio.to_thread(
        _summarise_doc_memory, doc_memory, document_type
    )

    # ── 3. Assemble enriched QA: scope → memory digest → filtered answers ─────
    scope_names = ", ".join(f'"{n}"' for n in subsection_names if n)
    enriched_qa: list[dict] = [
        {
            "question": "SCOPE CONSTRAINT",
            "answer": (
                f"Generate ONLY: {scope_names}. "
                "Do NOT add any other sections or headings. "
                "Previously generated content is summarised below for reference — do NOT repeat or regenerate it."
            ),
            "category": "_scope",
            "answer_type": "text",
        }
    ]
    if condensed_memory.strip():
        enriched_qa.append({
            "question": (
                "Consistency digest of previously generated sections "
                "(reference only — do NOT repeat or regenerate any of this)"
            ),
            "answer": condensed_memory,
            "category": "_memory",
            "answer_type": "text",
        })
    enriched_qa.extend(filtered_qa)

    # ── 4. Run lean section graph (no gap analysis) ────────────────────────────
    initial_state: AgentState = {
        "department": department,
        "document_type": document_type,
        "questions_and_answers": enriched_qa,
        "required_section": scoped_required_section,
        "gap_questions": [],
        "supplementary_content": "",
        "system_prompt": "",
        "generated_document": "",
        "quality_scores": {},
        "quality_issues": [],
        "quality_suggestions": [],
        "retry_count": 0,
        "status": "generating",
    }

    final_state = await asyncio.to_thread(
        section_generation_agent.invoke, initial_state
    )

    section_text = final_state.get("generated_document", "")
    status       = final_state.get("status", "unknown")
    retries      = final_state.get("retry_count", 0)

    logger.info(
        "   ✅ '%s' done — status=%s, retries=%d, %d chars (qa_sent=%d, memory=%d chars)",
        section.get("title", "Untitled"), status, retries,
        len(section_text), len(enriched_qa), len(condensed_memory),
    )
    return section_text