import sys
import os
import streamlit as st
import requests
import logging

# Ensure the project root is on sys.path so both `ui/` (same-dir) and
# `agent.*` / `api.*` imports resolve correctly regardless of cwd.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from api_helpers import (
    fetch_departments,
    fetch_document_types,
    fetch_questions,
    fetch_notion_page_urls,
    call_gap_questions_endpoint,
    call_save_questions_endpoint,
    call_generate_endpoint,
    call_generate_section,
    FASTAPI_URL,
)
from pdf_generator import generate_pdf_from_markdown, build_safe_pdf_filename
from question_helpers import (
    build_unified_question_list,
    build_ordered_categories,
    collect_all_answered_qa,
    collect_page_answered_qa,
    question_has_answer,
    render_question_widget,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ui.streamlit_uidemo")


st.set_page_config(page_title="DocForgeHub", page_icon="📄", layout="wide")

# -------------------------------------------------
# CSS
# -------------------------------------------------
st.markdown(
    """
    <style>

    /* Vertical separators */
    .block-container {
        padding-top: 1rem;
    }

    .separator-right {
        border-right: 1px solid #444;
        padding-right: 1rem;
    }

    .separator-left {
        border-left: 1px solid #444;
        padding-left: 1rem;
    }

    /* Markdown editor sizing */
    textarea {
        font-family: monospace;
    }

    /* Gap question banner */
    .gap-banner {
        background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
        border-left: 4px solid #e94560;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0 1rem 0;
        font-size: 0.88rem;
        color: #ccc;
    }
    .gap-banner strong { color: #e94560; }

    /* Subtle badge for gap questions */
    .gap-badge {
        display: inline-block;
        background: #e94560;
        color: white;
        font-size: 0.68rem;
        font-weight: 700;
        padding: 1px 7px;
        border-radius: 10px;
        margin-left: 6px;
        vertical-align: middle;
        letter-spacing: 0.04em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------
# Cached API data fetchers (thin wrappers around api_helpers)
# -------------------------------------------------

@st.cache_data(ttl=300)
def get_departments_from_fastapi():
    return fetch_departments()

@st.cache_data(ttl=300)
def get_document_types_from_fastapi(department_name):
    return fetch_document_types(department_name)

@st.cache_data(ttl=300)
def get_questions_from_fastapi(document_type):
    return fetch_questions(document_type)

@st.cache_data(ttl=600)
def get_notionpage_urls_from_fastapi():
    return fetch_notion_page_urls()


# ---------------------------------------------------
# Load the initial data
# ---------------------------------------------------

pages = get_notionpage_urls_from_fastapi()
departments = get_departments_from_fastapi()
department_names = [dept_dict["name"] for dept_dict in departments]


# -------------------------------------------------
# Session State
# -------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = pages or []

if "answers" not in st.session_state:
    st.session_state.answers = {}

if "gap_answers" not in st.session_state:
    st.session_state.gap_answers = {}

if "gap_questions" not in st.session_state:
    st.session_state.gap_questions = []

if "markdown_doc" not in st.session_state:
    st.session_state.markdown_doc = ""

if "is_generating" not in st.session_state:
    st.session_state.is_generating = False

if "is_analyzing" not in st.session_state:
    st.session_state.is_analyzing = False

if "is_saving" not in st.session_state:
    st.session_state.is_saving = False

if "gap_source" not in st.session_state:
    st.session_state.gap_source = ""

if "gap_doc_type" not in st.session_state:
    st.session_state.gap_doc_type = ""

# Progressive / pagination state
if "prog_mode" not in st.session_state:
    st.session_state.prog_mode = False
if "q_page" not in st.session_state:
    st.session_state.q_page = 0
if "prog_sections" not in st.session_state:
    st.session_state.prog_sections = {}       # {category_name: generated_text}
# Purge stale integer keys left from previous sessions
if any(isinstance(category_key, int) for category_key in st.session_state.prog_sections):
    st.session_state.prog_sections = {}
if "prog_generating" not in st.session_state:
    st.session_state.prog_generating = False
# Track which category step we are on in progressive mode (sequential reveal)
if "prog_current_step" not in st.session_state:
    st.session_state.prog_current_step = 0


# =================================================
# LEFT SIDEBAR
# =================================================
with st.sidebar:
    st.write("<h1>📄</br>DocForge Hub</h1>", unsafe_allow_html=True)

    st.subheader("Department")
    selected_department = st.selectbox(
        "Department",
        department_names or ["(no departments found)"],
        label_visibility="collapsed"
    )

    st.subheader("Document")
    is_valid_department = selected_department and selected_department != "(no departments found)"
    doc_types = get_document_types_from_fastapi(selected_department) if is_valid_department else []
    document_names = [doc_dict["document_type"] for doc_dict in doc_types]

    # Build lookups
    document_name_lookup = {
        doc_dict["document_type"]: doc_dict.get("document_name", doc_dict["document_type"])
        for doc_dict in doc_types
    }
    department_obj_lookup = {dept_dict["name"]: dept_dict for dept_dict in departments}

    selected_document = st.selectbox(
        "Document",
        document_names or ["(select a department first)"],
        label_visibility="collapsed",
    )

    # Auto-clear gap questions when the document changes
    if st.session_state.gap_doc_type and st.session_state.gap_doc_type != selected_document:
        st.session_state.gap_questions = []
        st.session_state.gap_answers = {}
        st.session_state.gap_source = ""
        st.session_state.gap_doc_type = ""

    st.subheader("Generation Mode")
    mode_choice = st.radio(
        "Mode",
        ["Single-Shot", "Progressive"],
        index=1 if st.session_state.prog_mode else 0,
        label_visibility="collapsed",
        help="Single-Shot: generates the full document at once. Progressive: generates section-by-section with memory.",
    )
    st.session_state.prog_mode = (mode_choice == "Progressive")

    st.subheader("Generation History")
    history_container = st.container(height=270)
    with history_container:
        for history_item in st.session_state.history:
            st.markdown(
                f"<a href='{history_item.get('url','#')}' style='text-decoration: none; color: beige;'>"
                f"{history_item.get('title','Untitled')}</a>",
                unsafe_allow_html=True,
            )


# =================================================
# MAIN AREA
# =================================================
col_questions, col_editor = st.columns([2, 3])


########################################
# Load questions for the selected document
########################################

is_valid_document = selected_document and selected_document != "(select a department first)"
questions = get_questions_from_fastapi(selected_document) if is_valid_document else []

for question_idx, question_data in enumerate(questions):
    answer_key = f"answer_{question_idx}"
    if answer_key not in st.session_state.answers:
        st.session_state.answers[answer_key] = question_data.get("answer", "") or ""

for gap_idx, gap_question in enumerate(st.session_state.gap_questions):
    gap_answer_key = f"gap_answer_{gap_idx}"
    if gap_answer_key not in st.session_state.gap_answers:
        st.session_state.gap_answers[gap_answer_key] = gap_question.get("answer", "") or ""


# ═══════════════════════════════════════════════════════════════
#  Fetch schema and flatten subsections for progressive mode
# ═══════════════════════════════════════════════════════════════

# all_subsections: flat, ordered list of dicts — each subsection carries
# its parent section title so we can build the correct API payload.
#   [ { "parent_title": "1. Objective", "title": "Description", "type": "list", "order": 1, ... }, ... ]
all_subsections: list[dict] = []
subsection_titles: list[str] = []  # ordered titles for the right-panel reveal

if is_valid_document and st.session_state.prog_mode:
    document_name_for_schema = document_name_lookup.get(selected_document, selected_document)
    try:
        schema_response = requests.get(
            f"{FASTAPI_URL}/required-section",
            params={"department": selected_department, "document_name": document_name_for_schema},
            timeout=15,
        )
        schema_response.raise_for_status()
        schema_data = schema_response.json().get("required_section", schema_response.json())
        schema_sections = schema_data.get("sections", [])

        for section_obj in schema_sections:
            parent_title = section_obj.get("title", "Document")
            subs = section_obj.get("subsections", [])
            if subs:
                for sub in sorted(subs, key=lambda s: s.get("order", 0)):
                    all_subsections.append({
                        **sub,
                        "_parent_title": parent_title,
                    })
            else:
                # Table-only or flat section — treat the section itself as a subsection
                all_subsections.append({
                    "title": parent_title,
                    "type": section_obj.get("type", "text"),
                    "columns": section_obj.get("columns", []),
                    "order": section_obj.get("order", 0),
                    "_parent_title": parent_title,
                })

        subsection_titles = [sub["title"] for sub in all_subsections]
        logger.info("Schema loaded — %d subsections: %s", len(all_subsections), subsection_titles)
    except Exception as schema_err:
        logger.warning("Failed to fetch schema: %s", schema_err)


# ═══════════════════════════════════════════════════════════════
#  Build a UNIFIED question list using helpers
# ═══════════════════════════════════════════════════════════════

all_questions = build_unified_question_list(
    questions=questions,
    answers_state=st.session_state.answers,
    gap_questions=st.session_state.gap_questions,
    gap_answers_state=st.session_state.gap_answers,
)

ordered_categories = build_ordered_categories(all_questions)



# ═══════════════════════════════════════════════════════════════
#  Pagination — always 5 questions per page
# ═══════════════════════════════════════════════════════════════

PAGE_SIZE = 5
total_questions = len(all_questions)
total_pages = max(1, -(-total_questions // PAGE_SIZE))

if st.session_state.q_page >= total_pages:
    st.session_state.q_page = total_pages - 1
if st.session_state.q_page < 0:
    st.session_state.q_page = 0

current_page = st.session_state.q_page
page_start = current_page * PAGE_SIZE
page_end = min(page_start + PAGE_SIZE, total_questions)
page_questions = all_questions[page_start:page_end]


def get_sec_idx_for_page(page_idx: int, sections: list) -> tuple:
    """Map a page index to a schema section index."""
    if not sections:
        return 0, None
    section_index = min(
        int(page_idx / max(total_pages, 1) * len(sections)),
        len(sections) - 1,
    )
    return section_index, sections[section_index]


def get_section_qa_for_sec_idx(section_idx: int, sections: list) -> list:
    """Return ALL answered Q&A for sections[section_idx]."""
    if not sections or section_idx >= len(sections):
        return []

    target_section = sections[section_idx]
    section_title_lower = target_section.get("title", "").lower().strip()
    subsection_title_set = {
        subsection.get("title", "").lower().strip()
        for subsection in target_section.get("subsections", [])
    }

    matched_qa = []
    for widget_key, question_dict, answer_state, is_gap_flag in all_questions:
        question_category_lower = question_dict.get("category", "").lower().strip()
        answer_text = answer_state.get(widget_key, "")
        if not answer_text.strip():
            continue
        if (
            question_category_lower in subsection_title_set
            or question_category_lower == section_title_lower
            or section_title_lower in question_category_lower
        ):
            matched_qa.append({
                "question": question_dict.get("question", ""),
                "answer": answer_text,
                "category": question_dict.get("category", ""),
                "answer_type": question_dict.get("answer_type", "text"),
            })

    # Fallback: if category matching found nothing, use proportional page slice
    if not matched_qa:
        pages_per_section = max(1, total_pages // len(sections))
        fallback_start = section_idx * pages_per_section * PAGE_SIZE
        fallback_end = min(fallback_start + pages_per_section * PAGE_SIZE, total_questions)
        for widget_key, question_dict, answer_state, is_gap_flag in all_questions[fallback_start:fallback_end]:
            answer_text = answer_state.get(widget_key, "")
            if answer_text.strip():
                matched_qa.append({
                    "question": question_dict.get("question", ""),
                    "answer": answer_text,
                    "category": question_dict.get("category", ""),
                    "answer_type": question_dict.get("answer_type", "text"),
                })

    return matched_qa


# ═══════════════════════════════════════════════════════════════
#  QUESTIONS PANEL (paginated)
# ═══════════════════════════════════════════════════════════════

with col_questions:
    st.markdown('<div class="separator-right">', unsafe_allow_html=True)

    if not questions:
        st.info("Select a department and document to load questions.")
    else:
        # ── Section header ──
        if st.session_state.prog_mode and schema_sections:
            section_idx, current_section = get_sec_idx_for_page(current_page, schema_sections)
            section_label = (
                current_section.get("title", f"Page {current_page + 1}")
                if current_section else f"Page {current_page + 1}"
            )
            st.header(f"Page {current_page + 1} of {total_pages}")
            st.subheader(f"📌 {section_label}")
            if current_section:
                section_subsections = current_section.get("subsections", [])
                if section_subsections:
                    subsection_title_list = [subsection.get("title", "") for subsection in section_subsections]
                    st.caption("Covers: " + ", ".join(subsection_title_list))
        else:
            st.header("Questions")
            st.caption(f"Page {current_page + 1} of {total_pages}")

        # ── Render the 5 questions for this page ──
        for widget_key, question_data, answer_state, is_gap_flag in page_questions:
            render_question_widget(
                question=question_data,
                widget_key=widget_key,
                answer_state=answer_state,
                is_gap=is_gap_flag,
            )

        # ── Progress bar ──
        answered_count = sum(
            1 for widget_key, question_dict, answer_state, is_gap_flag in all_questions
            if question_has_answer(widget_key, answer_state)
        )
        st.progress(
            answered_count / max(total_questions, 1),
            text=f"{answered_count} of {total_questions} answered"
        )

        # ── Gap questions controls ──
        mongo_gap_questions = [question for question in questions if question.get("is_gap_question")]
        session_gap_questions = st.session_state.gap_questions
        has_any_gap_questions = bool(mongo_gap_questions or session_gap_questions)

        # Save button for session gap questions
        if session_gap_questions and st.session_state.gap_source == "generated":
            st.markdown("")
            save_col, save_info_col = st.columns([1, 2])
            with save_col:
                save_clicked = st.button(
                    "💾 Save gap questions",
                    disabled=st.session_state.is_saving,
                    help="Saves these questions + your answers to the database.",
                    use_container_width=True,
                )
            with save_info_col:
                st.caption("Save to make these questions permanent.")

            if save_clicked:
                st.session_state.is_saving = True
                gap_qa_to_save = []
                for gap_idx, gap_question in enumerate(session_gap_questions):
                    gap_answer_key = f"gap_answer_{gap_idx}"
                    gap_qa_to_save.append({
                        **gap_question,
                        "answer": st.session_state.gap_answers.get(gap_answer_key, ""),
                    })
                dept_obj = department_obj_lookup.get(selected_department, {"name": selected_department})
                doc_name = document_name_lookup.get(selected_document, selected_document)
                with st.spinner("Saving gap questions to database..."):
                    save_result = call_save_questions_endpoint(
                        department_obj=dept_obj,
                        document_type=selected_document,
                        document_name=doc_name,
                        gap_questions=gap_qa_to_save,
                    )
                st.session_state.is_saving = False
                if save_result:
                    saved_count = save_result.get("saved", 0)
                    updated_count = save_result.get("updated", 0)
                    st.success(f"✅ Saved {saved_count} new, updated {updated_count}.")
                    get_questions_from_fastapi.clear()
                else:
                    st.error("❌ Save failed — check API logs.")

        # Analyse gaps button
        if not has_any_gap_questions and questions:
            st.divider()
            analyse_col, analyse_info_col = st.columns([1, 2])
            with analyse_col:
                analyse_clicked = st.button(
                    "🔍 Analyse schema gaps",
                    disabled=st.session_state.is_analyzing,
                    help="Uses AI to identify which document sections aren't covered.",
                    use_container_width=True,
                )
            with analyse_info_col:
                st.caption("Optional: detect and fill schema coverage gaps.")

            if analyse_clicked and is_valid_document:
                st.session_state.is_analyzing = True
                current_qa_for_gap_analysis = []
                for question_idx, question_item in enumerate(questions):
                    if question_item.get("is_gap_question"):
                        continue
                    answer_key = f"answer_{question_idx}"
                    current_qa_for_gap_analysis.append({
                        "question": question_item.get("question", ""),
                        "answer": st.session_state.answers.get(answer_key, ""),
                        "category": question_item.get("category", ""),
                        "answer_type": question_item.get("answer_type", "text"),
                    })
                doc_name = document_name_lookup.get(selected_document, selected_document)
                with st.spinner("🤖 Analysing schema coverage... this takes ~10 seconds."):
                    gap_analysis_result = call_gap_questions_endpoint(
                        department=selected_department,
                        document_type=selected_document,
                        document_name=doc_name,
                        questions_and_answers=current_qa_for_gap_analysis,
                    )
                st.session_state.is_analyzing = False
                if gap_analysis_result:
                    gap_questions_from_api = gap_analysis_result.get("gap_questions", [])
                    if gap_questions_from_api:
                        st.session_state.gap_questions = gap_questions_from_api
                        st.session_state.gap_source = gap_analysis_result.get("source", "generated")
                        st.session_state.gap_doc_type = selected_document
                        for gap_idx, gap_question in enumerate(gap_questions_from_api):
                            gap_answer_key = f"gap_answer_{gap_idx}"
                            if gap_answer_key not in st.session_state.gap_answers:
                                st.session_state.gap_answers[gap_answer_key] = ""
                        st.rerun()
                    else:
                        st.success("✅ All schema sections are fully covered — no gaps found!")
                else:
                    st.error("❌ Gap analysis failed. Check API logs.")

        st.divider()

        # ── Navigation: Back / Action / Next ──
        nav_back_col, nav_action_col, nav_next_col = st.columns([1, 2, 1])

        with nav_back_col:
            if current_page > 0:
                if st.button("← Back", use_container_width=True):
                    st.session_state.q_page -= 1
                    st.rerun()

        with nav_next_col:
            if current_page < total_pages - 1:
                if st.button("Next →", use_container_width=True):
                    st.session_state.q_page += 1
                    st.rerun()

        with nav_action_col:
            if not st.session_state.prog_mode:
                # ── Single-shot generate button ──
                single_shot_remaining = total_questions - answered_count
                single_shot_all_answered = answered_count == total_questions and total_questions > 0
                generate_button_clicked = st.button(
                    "⚡ Generate Document" if single_shot_all_answered
                    else f"🔒 Answer all questions first ({single_shot_remaining} remaining)",
                    disabled=st.session_state.is_generating or not single_shot_all_answered,
                    use_container_width=True,
                    type="primary",
                )
            else:
                # ── Progressive mode: finalize button only on last page ──
                generate_button_clicked = False
                progressive_all_answered = answered_count == total_questions and total_questions > 0
                progressive_remaining = total_questions - answered_count

                if current_page == total_pages - 1:
                    all_subsections_generated = (
                        subsection_titles
                        and all(t in st.session_state.prog_sections for t in subsection_titles)
                    )
                    if all_subsections_generated:
                        generate_button_clicked = st.button(
                            "📄 Finalize Document" if progressive_all_answered
                            else f"🔒 Finalize ({progressive_remaining} questions remaining)",
                            disabled=not progressive_all_answered,
                            use_container_width=True,
                        )
                    else:
                        remaining_subs = sum(1 for t in subsection_titles if t not in st.session_state.prog_sections)
                        st.info(f"Generate all sections in the preview panel first ({remaining_subs} remaining).")


        # ══════════════════════════════════════════════════════
        # Handle single-shot generate OR progressive finalize
        # ══════════════════════════════════════════════════════
        if generate_button_clicked and questions:
            if not st.session_state.prog_mode:
                # ═══ SINGLE-SHOT GENERATION ═══
                st.session_state.is_generating = True

                questions_and_answers = []

                # Core (non-gap) questions
                for question_idx, question_item in enumerate(questions):
                    if question_item.get("is_gap_question"):
                        continue
                    answer_key = f"answer_{question_idx}"
                    questions_and_answers.append({
                        "question": question_item.get("question", ""),
                        "answer": st.session_state.answers.get(answer_key, ""),
                        "category": question_item.get("category", ""),
                        "answer_type": question_item.get("answer_type", "text"),
                    })

                # MongoDB-persisted gap questions
                for question_idx, question_item in enumerate(questions):
                    if not question_item.get("is_gap_question"):
                        continue
                    answer_key = f"answer_{question_idx}"
                    questions_and_answers.append({
                        "question": question_item.get("question", ""),
                        "answer": st.session_state.answers.get(answer_key, ""),
                        "category": question_item.get("category", "Additional Information"),
                        "answer_type": question_item.get("answer_type", "text"),
                        "is_gap_question": True,
                    })

                # Session gap questions
                for gap_idx, gap_question in enumerate(st.session_state.gap_questions):
                    gap_answer_key = f"gap_answer_{gap_idx}"
                    questions_and_answers.append({
                        "question": gap_question.get("question", ""),
                        "answer": st.session_state.gap_answers.get(gap_answer_key, ""),
                        "category": gap_question.get("category", "Additional Information"),
                        "answer_type": gap_question.get("answer_type", "text"),
                        "is_gap_question": True,
                    })

                logger.info("Generate clicked — sending %d answers to agent", len(questions_and_answers))
                document_name_for_request = document_name_lookup.get(selected_document, selected_document)

                with st.spinner("Agent is generating your document... This may take 30-60 seconds."):
                    generation_result = call_generate_endpoint(
                        department=selected_department,
                        document_type=selected_document,
                        document_name=document_name_for_request,
                        questions_and_answers=questions_and_answers,
                    )

                st.session_state.is_generating = False

                if generation_result:
                    st.session_state.markdown_doc = generation_result.get("generated_document", "")
                    generation_status = generation_result.get("status", "unknown")
                    quality_scores = generation_result.get("quality_scores", {})
                    retry_count = generation_result.get("retry_count", 0)

                    new_gap_questions = generation_result.get("gap_questions", [])
                    if new_gap_questions and not st.session_state.gap_questions:
                        st.session_state.gap_questions = new_gap_questions
                        st.session_state.gap_source = generation_result.get("source", "generated")
                        st.session_state.gap_doc_type = selected_document
                        for gap_idx, gap_question in enumerate(new_gap_questions):
                            gap_answer_key = f"gap_answer_{gap_idx}"
                            if gap_answer_key not in st.session_state.gap_answers:
                                st.session_state.gap_answers[gap_answer_key] = ""
                        st.info(f"💡 {len(new_gap_questions)} gap question(s) detected. Answer them and regenerate.")

                    if generation_status == "passed":
                        st.success(f"✅ Document generated! (retries: {retry_count})")
                    else:
                        st.warning(f"⚠️ Generated with issues (status: {generation_status}, retries: {retry_count})")

                    if quality_scores:
                        with st.expander("📊 Quality Scores", expanded=(generation_status == "passed")):
                            score_cols = st.columns(len(quality_scores))
                            for score_col, (criterion, score) in zip(score_cols, quality_scores.items()):
                                score_col.metric(criterion.replace("_", " ").title(), f"{score}/5")
                else:
                    st.error("❌ Generation failed. Check the API logs for details.")

            else:
                # ═══ PROGRESSIVE FINALIZE ═══
                # Generate any subsection not yet done, then stitch in schema order
                all_qa = collect_all_answered_qa(all_questions)

                for sub_entry in all_subsections:
                    sub_title = sub_entry["title"]
                    if sub_title in st.session_state.prog_sections:
                        continue
                    parent_title = sub_entry.get("_parent_title", "Document")
                    section_for_api = {
                        "title": parent_title,
                        "subsections": [sub_entry],
                    }
                    previously_generated = "\n\n".join(
                        st.session_state.prog_sections[t]
                        for t in subsection_titles
                        if t in st.session_state.prog_sections and t != sub_title
                    )
                    with st.spinner(f"⚡ Generating '{sub_title}'..."):
                        section_result = call_generate_section(
                            department=selected_department,
                            document_type=selected_document,
                            section=section_for_api,
                            questions_and_answers=all_qa,
                            doc_memory=previously_generated,
                        )
                    if section_result and section_result.get("section_text"):
                        st.session_state.prog_sections[sub_title] = section_result["section_text"]

                # Stitch all generated sections in schema order
                full_doc = "\n\n".join(
                    st.session_state.prog_sections[t]
                    for t in subsection_titles
                    if t in st.session_state.prog_sections
                )
                st.session_state.markdown_doc = full_doc
                st.success("🎉 Document finalized! View it in the preview panel →")

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  RIGHT PANEL: Preview / Editor — Sequential Progressive Reveal
# ═══════════════════════════════════════════════════════════════

with col_editor:
    st.markdown('<div class="separator-left">', unsafe_allow_html=True)

    editor_header_col, publish_col = st.columns([4, 1])

    with editor_header_col:
        if st.session_state.prog_mode:
            st.header("Preview")
        else:
            st.header("Markdown View")

    with publish_col:
        publish_clicked = st.button("Publish")
        if publish_clicked:
            if st.session_state.markdown_doc:
                st.balloons()
                st.success("Published! 🎉")
                logger.info("Document published")
            else:
                st.warning("Nothing to publish yet.")

    st.markdown('<div class="scrollable">', unsafe_allow_html=True)

    if st.session_state.prog_mode:
        # ═══════════════════════════════════════════════════════
        #  SEQUENTIAL PROGRESSIVE GENERATION
        #
        #  Walks through EVERY schema subsection in order.
        #  After the user generates one subsection, the result
        #  appears and the NEXT subsection's generate button is
        #  revealed. The user cannot skip ahead.
        # ═══════════════════════════════════════════════════════

        progressive_all_answered = answered_count == total_questions and total_questions > 0

        if not subsection_titles:
            st.info("📋 No schema subsections found. Select a document with a schema to use progressive mode.")
        else:
            # ── Show all already-generated subsections ──
            for sub_idx, sub_title in enumerate(subsection_titles):
                if sub_title not in st.session_state.prog_sections:
                    break  # stop at the first un-generated subsection

                # Always use the schema subsection title — don't extract from
                # generated text (the LLM may put a document-title heading first).
                section_text = st.session_state.prog_sections[sub_title]
                display_title = sub_title
                st.markdown(f"✅ **{display_title}**")
                with st.expander(f"📖 {display_title}", expanded=False):
                    st.markdown(section_text)

            # ── Determine how many subsections have been generated ──
            generated_count = sum(
                1 for t in subsection_titles if t in st.session_state.prog_sections
            )

            if generated_count < len(subsection_titles):
                # Show the generate button for the NEXT un-generated subsection
                next_sub_title = subsection_titles[generated_count]
                next_sub_entry = all_subsections[generated_count]
                parent_title = next_sub_entry.get("_parent_title", "Document")
                sub_type = next_sub_entry.get("type", "text")

                st.divider()
                type_icon = "📊" if sub_type == "table" else "📝"
                st.subheader(f"{type_icon} Next: {next_sub_title}")

                # Show progress
                st.progress(generated_count / len(subsection_titles))
                st.caption(f"Step {generated_count + 1} of {len(subsection_titles)}")

                if not progressive_all_answered:
                    remaining = total_questions - answered_count
                    st.warning(f"🔒 Answer all questions first ({remaining} remaining)")
                else:
                    button_label = f"⚡ Generate: {next_sub_title}"

                    generate_section_clicked = st.button(
                        button_label,
                        disabled=st.session_state.prog_generating,
                        use_container_width=True,
                        type="primary",
                        key=f"prog_gen_{generated_count}",
                    )

                    if generate_section_clicked:
                        # Collect ALL answered Q&A — the agent filters by relevance
                        all_qa = collect_all_answered_qa(all_questions)

                        section_for_api = {
                            "title": parent_title,
                            "subsections": [next_sub_entry],
                        }
                        previously_generated = "\n\n".join(
                            st.session_state.prog_sections[t]
                            for t in subsection_titles
                            if t in st.session_state.prog_sections
                        )

                        logger.info(
                            "Progressive generate — subsection='%s', parent='%s', type='%s', qa_count=%d",
                            next_sub_title, parent_title, sub_type, len(all_qa),
                        )

                        with st.spinner(f"⚡ Generating '{next_sub_title}'..."):
                            section_result = call_generate_section(
                                department=selected_department,
                                document_type=selected_document,
                                section=section_for_api,
                                questions_and_answers=all_qa,
                                doc_memory=previously_generated,
                            )

                        if section_result and section_result.get("section_text"):
                            st.session_state.prog_sections[next_sub_title] = section_result["section_text"]
                            st.session_state.prog_current_step = generated_count + 1
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to generate '{next_sub_title}'. Check API logs.")
            else:
                # All subsections generated!
                st.divider()
                st.success("🎉 All sections generated! Click 'Finalize Document' in the questions panel.")

            # ── Combined editable preview ──
            if st.session_state.prog_sections:
                st.divider()
                full_preview = "\n\n".join(
                    st.session_state.prog_sections[t]
                    for t in subsection_titles
                    if t in st.session_state.prog_sections
                )
                st.session_state.markdown_doc = st.text_area(
                    "Combined Preview",
                    value=full_preview,
                    height=300,
                    label_visibility="collapsed",
                    key="prog_editor",
                )

    else:
        # ── Single-shot mode: simple editor ──
        st.session_state.markdown_doc = st.text_area(
            "Markdown Editor",
            value=st.session_state.markdown_doc,
            height=450,
            label_visibility="collapsed",
        )

    if st.session_state.markdown_doc:
        with st.expander("📖 Preview rendered document", expanded=False):
            st.markdown(st.session_state.markdown_doc)

        st.divider()
        try:
            pdf_bytes = generate_pdf_from_markdown(
                st.session_state.markdown_doc,
                document_title=selected_document,
            )
            st.download_button(
                label="⬇ Download PDF",
                data=pdf_bytes,
                file_name=build_safe_pdf_filename(selected_document),
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as _pdf_err:
            st.warning(f"⚠️ Could not generate PDF: {_pdf_err}")

    # Progressive mode: reset button
    if st.session_state.prog_mode and st.session_state.prog_sections:
        if st.button("🔄 Reset Progressive Session", use_container_width=True):
            st.session_state.prog_sections = {}
            st.session_state.prog_current_step = 0
            st.session_state.q_page = 0
            st.session_state.markdown_doc = ""
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)