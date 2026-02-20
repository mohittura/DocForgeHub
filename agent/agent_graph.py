import os
import json
import logging
import asyncio
from typing import TypedDict, Literal
from dotenv import load_dotenv
import re

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from agent.prompts import (
    build_system_prompt,
    build_table_only_prompt,
    build_gap_filler_prompt,
    build_quality_review_prompt,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("agent.agent_graph")

load_dotenv()

# ── Primary document-generation LLM ─────────────────────────────
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="moonshotai/kimi-k2-instruct-0905",
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
#  Formatting helpers (unchanged)
# ═══════════════════════════════════════════════════════════════

def format_questions_and_answers_for_prompt(qa_list: list[dict]) -> str:
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


def format_required_section_for_prompt(required_section: dict) -> str:
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
            table_title = section.get("title", document_name or "Data Table")
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


def is_table_only_schema(required_section: dict) -> bool:
    sections = required_section.get("sections", [])
    if not sections:
        return False
    try:
        debug_info = [
            f"type={s.get('type')}, subs={bool(s.get('subsections'))}"
            for s in sections
        ]
        logger.info("   🔍 Checking is_table_only_schema: %s", debug_info)
    except Exception:
        pass
    return all(
        s.get("type") == "table" and not s.get("subsections")
        for s in sections
    )


def get_table_columns(required_section: dict) -> list[str]:
    for section in required_section.get("sections", []):
        if section.get("type") == "table":
            return section.get("columns", [])
    return []


# ═══════════════════════════════════════════════════════════════
#  NODE 1 (NEW): analyze_schema_gaps
# ═══════════════════════════════════════════════════════════════

_GAP_QUESTION_SYSTEM_PROMPT = """\
You are a document-requirements analyst. Your job is to compare a document schema
against a set of existing Q&A answers and identify which schema sections have
INSUFFICIENT information to write good content.

Rules:
1. A section is "covered" if at least one Q&A answer meaningfully addresses it.
2. A section is "uncovered" if no answer touches it, or the answer is very thin.
3. For each uncovered section, generate EXACTLY ONE targeted question that would
   give the document writer enough information to fill that section well.
4. Questions must be practical, specific, and answerable in 1-3 sentences.
5. Output ONLY valid JSON — no markdown, no explanation, no preamble.

Output format (JSON array):
[
  {
    "question": "What are the key risks and mitigation strategies for this project?",
    "category": "Risk Management",
    "answer_type": "text",
    "section_covered": "Risk Assessment"
  },
  ...
]

If ALL sections are already covered, return an empty array: []
"""


def analyze_schema_gaps(state: AgentState) -> dict:
    """
    NODE 1 (NEW): Analyze schema vs existing Q&A to identify coverage gaps.

    Uses a dedicated lightweight LLM to:
      1. Compare every schema section against the existing answers
      2. Generate one targeted question per uncovered section
      3. Return the gap questions so they can be:
         a) Displayed in the Streamlit UI for the user to answer
         b) Saved to MongoDB's document_qas collection for future reuse

    This replaces the old fill_schema_gaps node which used the main LLM
    to synthesize supplementary content — which was wasteful and bypassed
    the user entirely.

    The generated questions are stored in state["gap_questions"] and also
    synthesized into state["supplementary_content"] for the document writer
    (using whatever answers are already in the Q&A payload for gap sections).
    """
    logger.info("🔎 Node: analyze_schema_gaps — scanning schema coverage...")

    formatted_schema = format_required_section_for_prompt(state["required_section"])
    formatted_answers = format_questions_and_answers_for_prompt(state["questions_and_answers"])

    user_message = f"""
DOCUMENT TYPE: {state['document_type']}
DEPARTMENT: {state['department']}

=== SCHEMA (sections that must be covered) ===
{formatted_schema}

=== EXISTING Q&A ANSWERS ===
{formatted_answers}

Identify uncovered sections and generate gap questions now.
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

        logger.info(
            "   📝 Found %d schema gap(s) — questions generated for: %s",
            len(gap_questions),
            ", ".join(q.get("section_covered", "?") for q in gap_questions),
        )

        # Build lightweight supplementary_content from existing answers
        # that touch the gap areas — gives the document LLM something to
        # work with even if the user hasn't answered the gap questions yet.
        supplementary_lines = []
        for gq in gap_questions:
            section = gq.get("section_covered", "")
            supplementary_lines.append(
                f"**{section}**: This section requires additional information. "
                f"Gap question pending user answer: \"{gq['question']}\""
            )

        supplementary_content = "\n".join(supplementary_lines) if supplementary_lines else ""

        return {
            "gap_questions": gap_questions,
            "supplementary_content": supplementary_content,
        }

    except (json.JSONDecodeError, Exception) as err:
        logger.warning("   ⚠️  analyze_schema_gaps failed (non-critical): %s", err)
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
    """
    logger.info("📝 Node: build_prompt — assembling system prompt")

    formatted_answers = format_questions_and_answers_for_prompt(
        state["questions_and_answers"]
    )

    if is_table_only_schema(state["required_section"]):
        columns = get_table_columns(state["required_section"])
        logger.info("   📊 Table-only schema — columns: %s", ", ".join(columns))
        system_prompt = build_table_only_prompt(
            department=state["department"],
            document_type=state["document_type"],
            columns=columns,
            questions_and_answers=formatted_answers,
            supplementary_content=state.get("supplementary_content", ""),
        )
    else:
        formatted_schema = format_required_section_for_prompt(state["required_section"])
        system_prompt = build_system_prompt(
            department=state["department"],
            document_type=state["document_type"],
            required_section=formatted_schema,
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
        human_instruction = (
            f"Generate the {state['document_type']} as a Markdown table now. "
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
#  Helper: Structure Validator  (unchanged)
# ═══════════════════════════════════════════════════════════════

def validate_document_structure(document_text: str, required_section: dict) -> list[str]:
    errors = []
    expected_sections = []
    sections = required_section.get("sections", [])

    for i, section in enumerate(sections, start=1):
        expected_sections.append({
            "number": f"{i}",
            "title": section.get("title", section.get("type", "Section")),
            "type": section.get("type", "text"),
        })
        for j, sub in enumerate(section.get("subsections", []), start=1):
            expected_sections.append({
                "number": f"{i}.{j}",
                "title": sub.get("title", "Subsection"),
                "type": sub.get("type", "text"),
            })

    lines = document_text.split("\n")
    actual_sections = []
    header_pattern = re.compile(r"^(#{2,3})\s+(\d+(?:\.\d+)?)\.?\s+(.*)")
    current_section_index = -1

    for line in lines:
        stripped = line.strip()
        match = header_pattern.match(stripped)
        if match:
            current_section_index += 1
            actual_sections.append({
                "number": match.group(2),
                "title": match.group(3).strip(),
                "content_lines": [],
            })
        elif current_section_index >= 0:
            actual_sections[current_section_index]["content_lines"].append(stripped)

    if not actual_sections:
        return ["Structure check failed: No numbered sections found."]

    if len(actual_sections) != len(expected_sections):
        errors.append(
            f"Structure mismatch: Expected {len(expected_sections)} sections, "
            f"found {len(actual_sections)}."
        )

    for idx, expected in enumerate(expected_sections):
        if idx >= len(actual_sections):
            errors.append(f"Missing section #{idx+1}: '{expected['number']} {expected['title']}'")
            continue

        actual = actual_sections[idx]
        if actual["number"] != expected["number"]:
            errors.append(
                f"Section {idx+1} numbering mismatch: "
                f"Expected '{expected['number']}', found '{actual['number']}'"
            )

        content_text = "\n".join(actual["content_lines"]).strip()
        is_table_content = "|" in content_text and "-|-" in content_text

        if expected["type"] == "table" and not is_table_content:
            errors.append(f"Section {expected['number']} must be a TABLE.")
        elif expected["type"] == "text" and is_table_content:
            errors.append(f"Section {expected['number']} must be TEXT only.")

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
        doc_name = state.get("document_type", "Document")

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
        actual_columns = [c.strip() for c in header_line.split("|") if c.strip()]
        expected_normalized = [c.lower().strip() for c in expected_columns]
        actual_normalized = [c.lower().strip() for c in actual_columns]

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

    # MIXED SCHEMAS: structural + LLM review
    structure_errors = validate_document_structure(document_text, state["required_section"])
    if structure_errors:
        logger.warning("   ❌ Structural validation failed with %d errors", len(structure_errors))
        return {
            "quality_scores": {"structure": 1},
            "quality_issues": structure_errors,
            "quality_suggestions": [
                "Follow the numbered section structure EXACTLY.",
                "Do not skip sections or change orders.",
            ],
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
    thin = [s for s in sections_split[1:] if len(s.strip()) < 100]
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

    issues_text = "\n".join(f"- {i}" for i in state["quality_issues"])
    suggestions_text = "\n".join(f"- {s}" for s in state.get("quality_suggestions", []))

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
#  Public API
# ═══════════════════════════════════════════════════════════════

async def run_agent(
    department: str,
    document_type: str,
    questions_and_answers: list[dict],
    required_section: dict,
) -> dict:
    """
    Run the document generation agent end-to-end.

    Returns:
        generated_document   – final Markdown text
        gap_questions        – NEW: questions for uncovered schema sections
        status               – "passed" or "failed"
        quality_issues       – any remaining issues
        quality_scores       – LLM quality scores
        quality_suggestions  – improvement suggestions
        retry_count          – fix attempts made
    """
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
    """
    Run ONLY the schema-gap analysis (no document generation).

    Used by POST /gap-questions to analyse gaps before generation
    and let the user answer them first.

    Returns list of gap question dicts.
    """
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