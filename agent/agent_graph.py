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
    

def build_prompt(state: AgentState) -> dict:
    """
    NODE 2: Assemble the full system prompt.

    Combines:
        - The document schema (required_section)
        - The user's Q&A answers
        - Any supplementary content from the gap filler

    Into the final system prompt that will be sent to the LLM.

    For TABLE-ONLY schemas (e.g. Change Request Log), uses a strict
    table-only prompt that produces ONLY a heading + Markdown table.
    """
    logger.info("📝 Node: build_prompt — assembling system prompt")

    formatted_answers = format_questions_and_answers_for_prompt(
        state["questions_and_answers"]
    )

    if is_table_only_schema(state["required_section"]):
        columns = get_table_columns(state["required_section"])
        logger.info(
            "   📊 Table-only schema detected — columns: %s",
            ", ".join(columns),
        )
        system_prompt = build_table_only_prompt(
            department=state["department"],
            document_type=state["document_type"],
            columns=columns,
            questions_and_answers=formatted_answers,
            supplementary_content=state.get("supplementary_content", ""),
        )
    else:
        formatted_schema = format_required_section_for_prompt(
            state["required_section"]
        )
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



def generate_document(state: AgentState) -> dict:
    """
    NODE 3: Call the LLM to generate the Markdown document.

    Sends the system prompt + a targeted instruction.
    For table-only schemas, the instruction emphasizes table output.
    """
    logger.info("🤖 Node: generate_document — calling LLM...")

    # Tailor the human message based on schema type
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


# ── Helper: Structure Validator ─────────────────────────────────

def validate_document_structure(document_text: str, required_section: dict) -> list[str]:
    """
    Validate that the document follows the schema structure EXACTLY.
    Returns a list of error messages (empty if valid).
    """
    errors = []
    
    expected_sections = []
    sections = required_section.get("sections", [])
    
    for i, section in enumerate(sections, start=1):
        # Pattern A or B main section
        expected_sections.append({
            "number": f"{i}", 
            "title": section.get("title", section.get("type", "Section")),
            "type": section.get("type", "text")
        })
        
        # Subsections
        for j, sub in enumerate(section.get("subsections", []), start=1):
             expected_sections.append({
                "number": f"{i}.{j}", 
                "title": sub.get("title", "Subsection"), 
                "type": sub.get("type", "text")
            })

    lines = document_text.split('\n')
    actual_sections = []
    
    # Regex to capture: ## 1. Title or ### 1.1. Title
    # Group 2 is the number (e.g. "1" or "1.1")
    # Group 3 is the title
    header_pattern = re.compile(r"^(#{2,3})\s+(\d+(?:\.\d+)?)\.?\s+(.*)")
    
    # Track content for type validation
    current_section_index = -1
    
    for line in lines:
        stripped = line.strip()
        match = header_pattern.match(stripped)
        
        if match:
            # Found a new header
            current_section_index += 1
            actual_sections.append({
                "number": match.group(2),
                "title": match.group(3).strip(),
                "content_lines": []
            })
        elif current_section_index >= 0:
            actual_sections[current_section_index]["content_lines"].append(stripped)



    if not actual_sections:
         return ["Structure check failed: No numbered sections found. Output must start with '## 1. [Title]'"]
    
    # Check count mismatch
    if len(actual_sections) != len(expected_sections):
         errors.append(f"Structure mismatch: Expected {len(expected_sections)} sections, found {len(actual_sections)}.")
    
    # Iterate and check strict 1:1 match
    for idx, expected in enumerate(expected_sections):
        if idx >= len(actual_sections):
            errors.append(f"Missing section #{idx+1}: '{expected['number']} {expected['title']}'")
            continue
            
        actual = actual_sections[idx]
        
        # Check number
        if actual["number"] != expected["number"]:
            errors.append(f"Section {idx+1} numbering mismatch: Expected '{expected['number']}', found '{actual['number']}'")
        
        # Check type (Content Validation)
        content_text = "\n".join(actual["content_lines"]).strip()
        
        # Loose table check: looks for pipe bars and separator lines
        is_table_content = "|" in content_text and "-|-" in content_text
        
        if expected["type"] == "table":
            if not is_table_content:
                errors.append(f"Section {expected['number']} ('{expected['title']}') must be a TABLE, but found text.")
        elif expected["type"] == "text":
            if is_table_content:
                errors.append(f"Section {expected['number']} ('{expected['title']}') must be TEXT only, but found a table.")

    return errors


