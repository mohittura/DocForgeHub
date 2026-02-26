"""
Question list building and Q&A collection helpers for DocForge Hub.

These functions build the unified question list from core + gap questions,
compute per-page/per-category slices, and collect answered Q&A for
sending to the generation API.

Most helpers are Streamlit-free: they operate on plain lists of
question tuples and dicts. The only Streamlit dependency is
`render_question_widget` which renders input widgets and can be
replaced when migrating to a different UI framework.
"""

import streamlit as st
from typing import Any


# ─────────────────────────────────────────────────────────────
#  Type alias for the unified question tuple:
#  (widget_key, question_dict, answer_state_dict, is_gap_flag)
# ─────────────────────────────────────────────────────────────
QuestionTuple = tuple[str, dict, dict, bool]


# ═══════════════════════════════════════════════════════════════
#  Building the unified question list
# ═══════════════════════════════════════════════════════════════

def build_unified_question_list(
    questions: list[dict],
    answers_state: dict,
    gap_questions: list[dict],
    gap_answers_state: dict,
) -> list[QuestionTuple]:
    """
    Build a single flat list of (widget_key, question_dict, answer_state, is_gap)
    from core questions, MongoDB-persisted gap questions, and session gap questions.
    """
    all_questions: list[QuestionTuple] = []

    # Core questions (non-gap from MongoDB)
    for question_idx, question in enumerate(questions):
        if question.get("is_gap_question"):
            continue
        all_questions.append((f"answer_{question_idx}", question, answers_state, False))

    # MongoDB-persisted gap questions (is_gap_question=True in main list)
    for question_idx, question in enumerate(questions):
        if not question.get("is_gap_question"):
            continue
        all_questions.append((f"answer_{question_idx}", question, answers_state, True))

    # Session gap questions (freshly generated, stored in gap_answers)
    for gap_idx, gap_question in enumerate(gap_questions):
        all_questions.append((f"gap_answer_{gap_idx}", gap_question, gap_answers_state, True))

    return all_questions


# ═══════════════════════════════════════════════════════════════
#  Category helpers
# ═══════════════════════════════════════════════════════════════

def build_ordered_categories(all_questions: list[QuestionTuple]) -> list[str]:
    """Return unique category names in the order they first appear in the question list."""
    seen: set[str] = set()
    ordered: list[str] = []
    for _, question_dict, _, _ in all_questions:
        category_name = question_dict.get("category", "").strip()
        category_lower = category_name.lower()
        if category_name and category_lower not in seen:
            seen.add(category_lower)
            ordered.append(category_name)
    return ordered


def build_category_to_subsection_map(
    all_questions: list[QuestionTuple],
    schema_subsections: list[dict],
) -> dict[str, dict]:
    """
    Map each question category (lowered) to its matching schema subsection dict.
    Only categories that have a matching subsection title are included.
    """
    subsection_by_title = {
        subsection.get("title", "").strip().lower(): subsection
        for subsection in schema_subsections
    } if schema_subsections else {}

    mapping: dict[str, dict] = {}
    for _, question_dict, _, _ in all_questions:
        category_lower = question_dict.get("category", "").strip().lower()
        if category_lower in subsection_by_title:
            mapping[category_lower] = subsection_by_title[category_lower]
    return mapping


def get_page_categories(
    page_idx: int,
    all_questions: list[QuestionTuple],
    page_size: int = 5,
) -> list[str]:
    """Return unique categories for the questions on a given page index."""
    page_start = page_idx * page_size
    page_end = min(page_start + page_size, len(all_questions))
    seen: set[str] = set()
    page_category_list: list[str] = []
    for _, question_dict, _, _ in all_questions[page_start:page_end]:
        category_name = question_dict.get("category", "").strip()
        if category_name and category_name not in seen:
            seen.add(category_name)
            page_category_list.append(category_name)
    return page_category_list


# ═══════════════════════════════════════════════════════════════
#  Q&A collection
# ═══════════════════════════════════════════════════════════════

def get_subsection_qa(
    category_name: str,
    all_questions: list[QuestionTuple],
) -> list[dict]:
    """Return all answered Q&A dicts for a specific category."""
    category_lower = category_name.strip().lower()
    return [
        {
            "question": question_dict.get("question", ""),
            "answer": answer_state.get(widget_key, ""),
            "category": question_dict.get("category", ""),
            "answer_type": question_dict.get("answer_type", "text"),
        }
        for widget_key, question_dict, answer_state, _ in all_questions
        if question_dict.get("category", "").strip().lower() == category_lower
        and answer_state.get(widget_key, "").strip()
    ]


def collect_all_answered_qa(all_questions: list[QuestionTuple]) -> list[dict]:
    """Gather ALL answered Q&A from every question (core + gap)."""
    qa_list = []
    for widget_key, question_dict, answer_state, _ in all_questions:
        answer_text = answer_state.get(widget_key, "")
        if answer_text.strip():
            qa_list.append({
                "question": question_dict.get("question", ""),
                "answer": answer_text,
                "category": question_dict.get("category", ""),
                "answer_type": question_dict.get("answer_type", "text"),
            })
    return qa_list


def collect_page_answered_qa(
    page_idx: int,
    all_questions: list[QuestionTuple],
    page_size: int = 5,
) -> list[dict]:
    """Gather answered Q&A for the questions on a specific page."""
    page_start = page_idx * page_size
    page_end = min(page_start + page_size, len(all_questions))
    qa_list = []
    for widget_key, question_dict, answer_state, _ in all_questions[page_start:page_end]:
        answer_text = answer_state.get(widget_key, "")
        if answer_text.strip():
            qa_list.append({
                "question": question_dict.get("question", ""),
                "answer": answer_text,
                "category": question_dict.get("category", ""),
                "answer_type": question_dict.get("answer_type", "text"),
            })
    return qa_list


# ═══════════════════════════════════════════════════════════════
#  Answer checking
# ═══════════════════════════════════════════════════════════════

def question_has_answer(widget_key: str, answer_state: dict) -> bool:
    """
    Return True if the question identified by widget_key has a non-empty answer.
    Checks st.session_state widget key first (live value from this run),
    then answer_state (persisted value from a previous run).
    """
    streamlit_widget_key = f"widget_{widget_key}"
    if streamlit_widget_key in st.session_state:
        live_value = st.session_state[streamlit_widget_key]
        if isinstance(live_value, list):
            return len(live_value) > 0
        return str(live_value).strip() != ""
    return str(answer_state.get(widget_key, "")).strip() != ""


# ═══════════════════════════════════════════════════════════════
#  Widget rendering (Streamlit-specific — replace when migrating)
# ═══════════════════════════════════════════════════════════════

def render_question_widget(
    question: dict,
    widget_key: str,
    answer_state: dict,
    is_gap: bool = False,
) -> None:
    """
    Render one question as the correct Streamlit input widget.
    Writes the answer back into answer_state[widget_key].
    is_gap=True adds an AI badge next to the label.
    """
    question_label = question.get("question", "Question")
    answer_type = question.get("answer_type", "text")

    if is_gap:
        st.markdown(
            f"<span style='font-size:0.9rem;font-weight:600;color:#eee;'>"
            f"{question_label} <span class='gap-badge'>AI</span></span>",
            unsafe_allow_html=True,
        )
        label_for_widget = "\u200b"
    else:
        label_for_widget = question_label

    streamlit_widget_key = f"widget_{widget_key}"

    if answer_type == "structured_list":
        answer_state[widget_key] = st.text_area(
            label_for_widget,
            value=answer_state.get(widget_key, ""),
            help="Enter items separated by newlines",
            key=streamlit_widget_key,
        )
    elif answer_type == "select":
        select_options = question.get("options", [])
        current_value = answer_state.get(widget_key, "")
        current_index = select_options.index(current_value) if current_value in select_options else 0
        answer_state[widget_key] = st.selectbox(
            label_for_widget,
            options=select_options,
            index=current_index,
            key=streamlit_widget_key,
        )
    elif answer_type == "multi_select":
        multi_options = question.get("options", [])
        current_value = answer_state.get(widget_key, "")
        default_selected_items = (
            [item.strip() for item in current_value.split(",") if item.strip()]
            if isinstance(current_value, str) and current_value
            else []
        )
        selected_items = st.multiselect(
            label_for_widget,
            options=multi_options,
            default=[item for item in default_selected_items if item in multi_options],
            key=streamlit_widget_key,
        )
        answer_state[widget_key] = ", ".join(selected_items)
    else:
        answer_state[widget_key] = st.text_area(
            label_for_widget,
            value=answer_state.get(widget_key, ""),
            key=streamlit_widget_key,
        )