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

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="moonshotai/kimi-k2-instruct-0905",
    temperature=0.1,
    max_tokens=8192,
)



class AgentState(TypedDict):
    """
    All the data that moves through the graph.

    Inputs (provided when you start the agent):
        department               – e.g. "Product Management"
        document_type            – e.g. "Feature Prioritization Framework"
        questions_and_answers    – list of dicts [{question, answer, category, ...}]
        required_section         – the document schema from MongoDB

    Intermediates / Outputs (filled in by the nodes):
        supplementary_content    – extra content for uncovered schema sections (from gap filler)
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

    supplementary_content: str
    system_prompt: str
    generated_document: str
    quality_scores: dict
    quality_issues: list[str]
    quality_suggestions: list[str]
    retry_count: int
    status: str


def format_questions_and_answers_for_prompt(qa_list: list[dict]) -> str:
    """
    Convert the list of Q&A dictionaries into a readable text block.

    Example output:
        ### Product Vision
        **Q:** What is the product vision?
        **A:** Build the best...
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

        # Special handling for structured_list answers
        if qa_item.get("answer_type") == "structured_list" and qa_item.get("answers"):
            answer_value = json.dumps(qa_item["answers"], indent=2)
        elif isinstance(answer_value, list):
            answer_value = ", ".join(str(item) for item in answer_value)

        lines.append(f"**Q:** {question_text}")
        lines.append(f"**A:** {answer_value if answer_value else '(not provided)'}")
        lines.append("")

    return "\n".join(lines)


def format_required_section_for_prompt(required_section: dict) -> str:
    """
    Convert the required_section schema into a readable text block
    showing the document structure the LLM should follow.

    Handles two schema patterns:
        Pattern A — Table-only:  section has {type: "table", columns: [...]}
                                 directly (no title, no subsections).
        Pattern B — Mixed:       section has {title: "...", subsections: [...]}
    """
    sections = required_section.get("sections", [])
    document_name = required_section.get("document_name", "")

    if not sections:
        # Fallback: try question_categories format
        categories = required_section.get("question_categories", [])
        if categories:
            return "\n".join(
                f"- {cat.get('category', 'Unknown')} (order: {cat.get('order', 0)})"
                for cat in categories
            )
        return "No schema sections available"

    lines = []
    for section in sections:
        # ── Pattern A: Table-only section (no title, no subsections) ──
        if section.get("type") == "table" and not section.get("subsections"):
            table_title = section.get("title", document_name or "Data Table")
            lines.append(f"## {table_title}")
            lines.append("")
            lines.append("⚠️  TABLE FORMAT REQUIRED — This entire document is a Markdown table.")
            lines.append(f"Column headers: | {' | '.join(columns)} |")
            lines.append("You MUST output a real Markdown table with these exact columns")
            lines.append("and at least 4-6 realistic data rows based on the user's answers.")
            lines.append("Do NOT describe the table — OUTPUT THE TABLE ITSELF.")
            lines.append("")
            continue

        # ── Pattern B: Mixed section with title + subsections ────────
        section_title = section.get("title", "Untitled Section")
        lines.append(f"## {section_title}")

        for subsection in section.get("subsections", []):
            sub_title = subsection.get("title", "")
            sub_type = subsection.get("type", "text")
            columns = subsection.get("columns", [])

            if sub_type == "table" and columns:
                # Strong table directive for subsection-level tables
                lines.append(f"  - {sub_title} ⚠️ TABLE — columns: | {' | '.join(columns)} |")
                lines.append(f"    (Output a real Markdown table with these columns and realistic rows)")
            else:
                lines.append(f"  - {sub_title} (type: {sub_type})")

        lines.append("")

    return "\n".join(lines)


def is_table_only_schema(required_section: dict) -> bool:
    """
    Return True if the schema is ONLY a single table with no subsections.

    Pattern A schemas look like:
        {"sections": [{"type": "table", "columns": [...]}]}
    These should produce table-only output, not prose documents.
    """
    sections = required_section.get("sections", [])
    if not sections:
        return False
        
    # Debug: log the section types/structure to help diagnose issues
    try:
        debug_info = [
            f"type={s.get('type')}, subs={bool(s.get('subsections'))}" 
            for s in sections
        ]
        logger.info("   🔍 Checking is_table_only_schema: %s", debug_info)
    except Exception:
        pass

    # Every section must be a direct table (no subsections)
    # Using 'not s.get("subsections")' handles: missing key, None, and []
    return all(
        s.get("type") == "table" and not s.get("subsections")
        for s in sections
    )


def get_table_columns(required_section: dict) -> list[str]:
    """
    Extract column names from a table-only schema.
    Returns the columns from the first table section.
    """
    for section in required_section.get("sections", []):
        if section.get("type") == "table":
            return section.get("columns", [])
    return []


def fill_schema_gaps(state: AgentState) -> dict:
    """
    NODE 1: Identify schema sections not covered by existing Q&A.

    Compares the required_section schema against the questions_and_answers.
    If there are gaps, asks the LLM to generate supplementary content
    so the document writer has material for every section.

    Why?  Some document types have 15+ sections but only 5-8 questions.
    Without this node, those sections would be empty or very thin.
    """
    logger.info("📋 Node: fill_schema_gaps — checking for uncovered schema sections")

    formatted_schema = format_required_section_for_prompt(state["required_section"])
    formatted_answers = format_questions_and_answers_for_prompt(state["questions_and_answers"])

    gap_filler_prompt = build_gap_filler_prompt(
        department=state["department"],
        document_type=state["document_type"],
        required_section=formatted_schema,
        questions_and_answers=formatted_answers,
    )

    try:
        messages = [
            SystemMessage(content=gap_filler_prompt),
            HumanMessage(content="Analyze the schema vs Q&A and provide supplementary content now."),
        ]
        llm_response = llm.invoke(messages)
        supplementary_content = llm_response.content

        if "All sections are adequately covered" in supplementary_content:
            logger.info("   ✅ All schema sections are covered by existing Q&A")
            return {"supplementary_content": ""}
        else:
            # Count how many sections were supplemented
            section_count = supplementary_content.count("**")
            logger.info(
                "   📝 Generated supplementary content for ~%d uncovered sections",
                section_count // 2,  # each section has opening + closing **
            )
            return {"supplementary_content": supplementary_content}

    except Exception as gap_error:
        logger.warning("   ⚠️  Gap filler failed (non-critical): %s", gap_error)
        return {"supplementary_content": ""}