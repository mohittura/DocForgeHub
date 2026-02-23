import streamlit as st
import requests
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ui.streamlit_uidemo")

FASTAPI_URL = "http://127.0.0.1:8000"

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
# API helpers to get each and every endpoint
# -------------------------------------------------

@st.cache_data(ttl=300) # ensures safety against mutations by creating a new copy of data (cached data) and holds data for 300 seconds
def get_departments_from_fastapi():
    """To GET the departments from the FASTAPI"""
    try:
        response_received = requests.get(f"{FASTAPI_URL}/departments", timeout=10)
        response_received.raise_for_status() # to check the http response status code
        departments_api = response_received.json().get("departments", [])
        logger.info(" -> received %d departments", len(departments_api))
        return departments_api
    except Exception as error:
        logger.error("Failed to fetch departments: %s", error)
        st.error(f"Failed to load Departments: {error}")
        return []


@st.cache_data(ttl=300)
def get_document_types_from_fastapi(department_name):
    """To GET the document types from the FASTAPI"""
    try:
        response_received = requests.get(
            f"{FASTAPI_URL}/document-types",
            params={"department": department_name},
            timeout=10,
        )
        response_received.raise_for_status()
        document_types_api = response_received.json().get("document_types", [])
        logger.info(" -> received %d document types", len(document_types_api))
        return document_types_api
    except Exception as error:
        logger.error("Failed to fetch document types: %s", error)
        st.error(f"Failed to load Document types: {error}")
        return []


@st.cache_data(ttl=300)
def get_questions_from_fastapi(document_type):
    """To GET the questions for all the document type from the FASTAPI"""
    try:
        response_received = requests.get(
            f"{FASTAPI_URL}/questions",
            params={"document_type": document_type},
            timeout=10,
        )
        response_received.raise_for_status()
        questions_api = response_received.json().get("questions", [])
        logger.info(" -> received %d questions", len(questions_api))
        return questions_api
    except Exception as error:
        logger.error("Failed to fetch questions: %s", error)
        st.error(f"Failed to load Questions: {error}")
        return []


@st.cache_data(ttl=600) # holds data for 600 seconds
def get_notionpage_urls_from_fastapi():
    """To GET all the generated pages url in notion pages from the FASTAPI"""
    try:
        response_received = requests.get(f"{FASTAPI_URL}/get_all_urls", timeout=30)
        response_received.raise_for_status()
        pages_api = response_received.json().get("pages", [])
        logger.info(" -> received %d pages", len(pages_api))
        return pages_api
    except Exception as error:
        logger.error("Failed to fetch published pages: %s", error)
        st.error(f"Failed to load URLs from Notion pages: {error}")
        return []


# ═══════════════════════════════════════════════════════════════
#  NEW: Gap Questions + Save Questions API helpers
# ═══════════════════════════════════════════════════════════════

def call_gap_questions_endpoint(
    department: str,
    document_type: str,
    document_name: str,
    questions_and_answers: list,
):
    """
    POST /gap-questions — Analyse schema coverage and get gap questions.
    Returns the full response dict or None on failure.
    """
    logger.info(
        "Calling POST /gap-questions — document_type=%s, answers=%d",
        document_type,
        len(questions_and_answers),
    )
    try:
        response_received = requests.post(
            f"{FASTAPI_URL}/gap-questions",
            json={
                "department": department,
                "document_type": document_type,
                "document_name": document_name,
                "questions_and_answers": questions_and_answers,
            },
            timeout=60,
        )
        response_received.raise_for_status()
        result = response_received.json()
        logger.info(
            "   → gap analysis done — source=%s, count=%d",
            result.get("source"),
            result.get("count", 0),
        )
        return result
    except Exception as error:
        logger.error("Gap question fetch failed: %s", error)
        return None


def call_save_questions_endpoint(
    department_obj: dict,
    document_type: str,
    document_name: str,
    gap_questions: list,
):
    """
    POST /save-questions — Persist answered gap questions to MongoDB.
    """
    logger.info(
        "Calling POST /save-questions — document_type=%s, questions=%d",
        document_type,
        len(gap_questions),
    )
    try:
        response_received = requests.post(
            f"{FASTAPI_URL}/save-questions",
            json={
                "department": department_obj,
                "document_type": document_type,
                "document_name": document_name,
                "gap_questions": gap_questions,
            },
            timeout=30,
        )
        response_received.raise_for_status()
        result = response_received.json()
        logger.info("   → saved=%d, updated=%d", result.get("saved", 0), result.get("updated", 0))
        return result
    except Exception as error:
        logger.error("Save questions failed: %s", error)
        return None


# ----------------------------------------------------
# Post endpoint for generation
# ----------------------------------------------------
def call_generate_endpoint(
    department: str,
    document_type: str,
    document_name: str,
    questions_and_answers: list,
):
    """
    POST /generate — Send answers to the agent and get a generated document back.

    Returns the full response dict or None on failure.
    """
    logger.info(
        "Calling POST /generate — department=%s, document_type=%s, answers=%d",
        department,
        document_type,
        len(questions_and_answers),
    )

    try:
        response_received = requests.post(
            f"{FASTAPI_URL}/generate",
            json={
                "department": department,
                "document_type": document_type,
                "document_name": document_name,
                "questions_and_answers": questions_and_answers,
            },
            timeout=120,  # generation can take a while
        )
        response_received.raise_for_status()
        result = response_received.json()
        logger.info(
            "   → generation complete — status=%s, length=%d chars",
            result.get("status"),
            len(result.get("generated_document", "")),
        )
        return result
    except Exception as error:
        logger.error("Generation failed: %s", error)
        st.error(f"Document generation failed: {error}")
        return None


def call_generate_section(
    department: str,
    document_type: str,
    section: dict,
    questions_and_answers: list,
    doc_memory: str = "",
):
    """Call POST /generate-section to generate one section with memory."""
    try:
        resp = requests.post(
            f"{FASTAPI_URL}/generate-section",
            json={
                "department": department,
                "document_type": document_type,
                "section": section,
                "questions_and_answers": questions_and_answers,
                "doc_memory": doc_memory,
            },
            timeout=90,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as error:
        logger.error("Section generation failed: %s", error)
        st.error(f"Section generation failed: {error}")
        return None



# ---------------------------------------------------
# Load the initial data from the functions
# ---------------------------------------------------

pages = get_notionpage_urls_from_fastapi()
departments = get_departments_from_fastapi()
department_names = [d["name"] for d in departments]


# -------------------------------------------------
# Session State (this will get updated based on the documents which are published into notion as well as we will cache those documents so that there is no need to call the document again and again and waste api limits)
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

# Track which document_type the gap questions belong to — so we clear
# them automatically when the user switches documents.
if "gap_doc_type" not in st.session_state:
    st.session_state.gap_doc_type = ""

# Progressive / pagination state
if "prog_mode" not in st.session_state:
    st.session_state.prog_mode = False
if "q_page" not in st.session_state:
    st.session_state.q_page = 0
if "prog_sections" not in st.session_state:
    st.session_state.prog_sections = {}       # {section_idx: generated_text}
if "prog_generating" not in st.session_state:
    st.session_state.prog_generating = False


# =================================================
# LEFT SIDEBAR
# =================================================
with st.sidebar:

    st.write("<h1>📄</br>DocForge Hub</h1>", unsafe_allow_html=True)

    st.subheader("Department")
    # ====================================================
    # This will also be generated based on the values stored in mongodb and will load things realtime
    # ====================================================
    selected_department = st.selectbox(
        "Department",
        department_names or ["(no departments found)"],
        label_visibility="collapsed"
    )

    st.subheader("Document")
    valid_dept = selected_department and selected_department != "(no departments found)"
    doc_types = get_document_types_from_fastapi(selected_department) if valid_dept else []
    document_names = [d["document_type"] for d in doc_types] # here document_type is the fastapi endpoint parameter

    # Build a lookup from document_type → document_name (needed for /generate)
    document_name_lookup = {
        dt["document_type"]: dt.get("document_name", dt["document_type"])
        for dt in doc_types
    }

    # Full department object needed for save-questions
    department_obj_lookup = {d["name"]: d for d in departments}

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
        for h in st.session_state.history:
            st.markdown(f"<a href='{h.get('url','#')}' style='text-decoration: none; color: beige;'>{h.get('title','Untitled')}</a>", unsafe_allow_html=True)


# =================================================
# MAIN AREA
# =================================================
col_questions, col_editor = st.columns([2, 3])


########################################
# getting questions as per departments
########################################

valid_document = selected_document and selected_document != "(select a department first)"
questions = get_questions_from_fastapi(selected_document) if valid_document else []

for i, _question in enumerate(questions): # the _question variable is used for internal purpose
    key = f"answer_{i}"
    if key not in st.session_state.answers:
        st.session_state.answers[key] = _question.get("answer", "") or ""

# Seed answer slots for gap questions (if any already loaded)
for i, gq in enumerate(st.session_state.gap_questions):
    key = f"gap_answer_{i}"
    if key not in st.session_state.gap_answers:
        st.session_state.gap_answers[key] = gq.get("answer", "") or ""


# ═══════════════════════════════════════════════════════════════
#  NEW: Helper to render a single question widget
# ═══════════════════════════════════════════════════════════════

def render_question_widget(
    ques: dict,
    widget_key: str,
    state_dict: dict,
    is_gap: bool = False,
) -> None:
    """
    Render one question as the correct Streamlit input widget.
    Writes the answer back into `state_dict[widget_key]`.
    is_gap=True adds a subtle AI badge next to the label.
    """
    label = ques.get("question", f"Question")
    answer_type = ques.get("answer_type", "text")

    # We can't put HTML in widget labels, so show the badge separately
    if is_gap:
        st.markdown(
            f"<span style='font-size:0.9rem;font-weight:600;color:#eee;'>"
            f"{label} <span class='gap-badge'>AI</span></span>",
            unsafe_allow_html=True,
        )
        label_for_widget = "\u200b"   # zero-width space → hides default label
    else:
        label_for_widget = label

    if answer_type == "structured_list":
        state_dict[widget_key] = st.text_area(
            label_for_widget,
            value=state_dict.get(widget_key, ""),
            help="Enter items separated by newlines",
            key=f"widget_{widget_key}",
        )
    elif answer_type == "select":
        options = ques.get("options", [])
        current = state_dict.get(widget_key, "")
        idx = options.index(current) if current in options else 0
        state_dict[widget_key] = st.selectbox(
            label_for_widget,
            options=options,
            index=idx,
            key=f"widget_{widget_key}",
        )
    elif answer_type == "multi_select":
        options = ques.get("options", [])
        current = state_dict.get(widget_key, "")
        default_values = (
            [v.strip() for v in current.split(",") if v.strip()]
            if isinstance(current, str) and current
            else []
        )
        selected_values = st.multiselect(
            label_for_widget,
            options=options,
            default=[v for v in default_values if v in options],
            key=f"widget_{widget_key}",
        )
        state_dict[widget_key] = ", ".join(selected_values)
    else:
        # Default: plain text area
        state_dict[widget_key] = st.text_area(
            label_for_widget,
            value=state_dict.get(widget_key, ""),
            key=f"widget_{widget_key}",
        )


# ═══════════════════════════════════════════════════════════════
#  Fetch schema sections for progressive mode
# ═══════════════════════════════════════════════════════════════

schema_sections = []
if valid_document and st.session_state.prog_mode:
    document_name_for_schema = document_name_lookup.get(selected_document, selected_document)
    try:
        rs_resp = requests.get(
            f"{FASTAPI_URL}/required-section",
            params={"department": selected_department, "document_name": document_name_for_schema},
            timeout=15,
        )
        rs_resp.raise_for_status()
        rs_data = rs_resp.json()
        required_section_data = rs_data.get("required_section", rs_data)
        schema_sections = required_section_data.get("sections", [])
    except Exception:
        pass  # schema_sections stays empty; progressive features degrade gracefully


# ═══════════════════════════════════════════════════════════════
#  Build a UNIFIED question list: core + mongo gap + session gap
#  Each entry: (widget_key, question_dict, state_dict, is_gap)
# ═══════════════════════════════════════════════════════════════

all_questions = []

# Core questions (non-gap from MongoDB)
for i, q in enumerate(questions):
    if q.get("is_gap_question"):
        continue
    all_questions.append((f"answer_{i}", q, st.session_state.answers, False))

# MongoDB-persisted gap questions (is_gap_question=True in main list)
for i, q in enumerate(questions):
    if not q.get("is_gap_question"):
        continue
    all_questions.append((f"answer_{i}", q, st.session_state.answers, True))

# Session gap questions (freshly generated, stored in gap_answers)
for i, gq in enumerate(st.session_state.gap_questions):
    all_questions.append((f"gap_answer_{i}", gq, st.session_state.gap_answers, True))

# ═══════════════════════════════════════════════════════════════
#  Pagination — always 5 questions per page.
#  In progressive mode we also track which schema section each
#  page belongs to, so "Generate This Section" sends the right Q&A.
# ═══════════════════════════════════════════════════════════════

PAGE_SIZE = 5
total_questions = len(all_questions)
total_pages = max(1, -(-total_questions // PAGE_SIZE))  # ceil division

# Clamp page
if st.session_state.q_page >= total_pages:
    st.session_state.q_page = total_pages - 1
if st.session_state.q_page < 0:
    st.session_state.q_page = 0

page = st.session_state.q_page
page_start = page * PAGE_SIZE
page_end = min(page_start + PAGE_SIZE, total_questions)
page_questions = all_questions[page_start:page_end]


def get_sec_idx_for_page(page_idx: int, sections: list) -> tuple:
    """
    Map a page index to a schema section index.

    Questions in MongoDB are ordered by (category_order, question_order).
    Schema sections are ordered the same way. So we do a straightforward
    proportional mapping: spread pages evenly across sections.

    e.g. 27 questions / 5 per page = 6 pages, 6 sections → page N → section N
         27 questions / 5 per page = 6 pages, 3 sections → pages 0-1→sec 0, 2-3→sec 1, 4-5→sec 2
    """
    if not sections:
        return 0, None
    sec_idx = min(
        int(page_idx / max(total_pages, 1) * len(sections)),
        len(sections) - 1,
    )
    return sec_idx, sections[sec_idx]


def get_section_qa_for_sec_idx(sec_idx: int, sections: list) -> list:
    """
    Return ALL answered Q&A for sections[sec_idx].

    Since question categories match subsection titles (not top-level section
    titles), we match against the subsection titles of the target section.
    As a fallback we also include questions whose category matches the
    section title itself.

    If NO questions match by category (e.g. categories aren't set), we fall
    back to the proportional page range so the user always gets output.
    """
    if not sections or sec_idx >= len(sections):
        return []

    sec = sections[sec_idx]
    sec_title = sec.get("title", "").lower().strip()
    # subsection titles are what question categories typically match
    sub_titles = {sub.get("title", "").lower().strip()
                  for sub in sec.get("subsections", [])}

    matched = []
    for wk, q, sd, ig in all_questions:
        cat = q.get("category", "").lower().strip()
        answer = sd.get(wk, "")
        if not answer.strip():
            continue
        if cat in sub_titles or cat == sec_title or sec_title in cat:
            matched.append({
                "question": q.get("question", ""),
                "answer": answer,
                "category": q.get("category", ""),
                "answer_type": q.get("answer_type", "text"),
            })

    # Fallback: if category matching found nothing, use proportional page slice
    if not matched:
        pages_per_sec = max(1, total_pages // len(sections))
        p_start = sec_idx * pages_per_sec * PAGE_SIZE
        p_end = min(p_start + pages_per_sec * PAGE_SIZE, total_questions)
        for wk, q, sd, ig in all_questions[p_start:p_end]:
            answer = sd.get(wk, "")
            if answer.strip():
                matched.append({
                    "question": q.get("question", ""),
                    "answer": answer,
                    "category": q.get("category", ""),
                    "answer_type": q.get("answer_type", "text"),
                })

    return matched


def collect_all_answered_qa():
    """Gather ALL answered Q&A from every question (core + gap)."""
    qa_list = []
    for wk, q, sd, ig in all_questions:
        answer = sd.get(wk, "")
        if answer.strip():
            qa_list.append({
                "question": q.get("question", ""),
                "answer": answer,
                "category": q.get("category", ""),
                "answer_type": q.get("answer_type", "text"),
            })
    return qa_list


def collect_page_answered_qa(page_idx: int):
    """Gather answered Q&A for the 5 questions visible on page_idx."""
    p_start = page_idx * PAGE_SIZE
    p_end = min(p_start + PAGE_SIZE, total_questions)
    page_qs = all_questions[p_start:p_end]
    qa_list = []
    for wk, q, sd, ig in page_qs:
        answer = sd.get(wk, "")
        if answer.strip():
            qa_list.append({
                "question": q.get("question", ""),
                "answer": answer,
                "category": q.get("category", ""),
                "answer_type": q.get("answer_type", "text"),
            })
    return qa_list


# -------------------------------
# QUESTIONS PANEL (paginated)
# -------------------------------

with col_questions:
    st.markdown('<div class="separator-right">', unsafe_allow_html=True)

    if not questions:
        st.info("Select a department and document to load questions.")
    else:
        # ── Section header ──
        if st.session_state.prog_mode and schema_sections:
            sec_idx, current_sec = get_sec_idx_for_page(page, schema_sections)
            if current_sec:
                section_label = current_sec.get("title", f"Page {page + 1}")
            else:
                section_label = f"Page {page + 1}"
            st.header(f"Page {page + 1} of {total_pages}")
            st.subheader(f"📌 {section_label}")
            if current_sec:
                subsections = current_sec.get("subsections", [])
                if subsections:
                    sub_titles = [s.get("title", "") for s in subsections]
                    st.caption("Covers: " + ", ".join(sub_titles))
        else:
            st.header("Questions")
            st.caption(f"Page {page + 1} of {total_pages}")

        # ── Progress bar (counts all questions) ──
        answered_count = sum(
            1 for wk, q, sd, ig in all_questions
            if sd.get(wk, "").strip()
        )
        st.progress(
            answered_count / max(total_questions, 1),
            text=f"{answered_count} of {total_questions} answered"
        )

        # ── Render the 5 questions for this page ──
        for widget_key, ques, state_dict, is_gap in page_questions:
            render_question_widget(
                ques=ques,
                widget_key=widget_key,
                state_dict=state_dict,
                is_gap=is_gap,
            )

        # ── Analyse gaps button + Save button (always visible at bottom) ──
        mongo_gap_questions = [q for q in questions if q.get("is_gap_question")]
        session_gap_questions = st.session_state.gap_questions
        has_any_gap = bool(mongo_gap_questions or session_gap_questions)

        # Save button for freshly generated session gap questions
        if session_gap_questions and st.session_state.gap_source == "generated":
            st.markdown("")
            save_col, info_col = st.columns([1, 2])
            with save_col:
                save_clicked = st.button(
                    "💾 Save gap questions",
                    disabled=st.session_state.is_saving,
                    help="Saves these questions + your answers to the database.",
                    use_container_width=True,
                )
            with info_col:
                st.caption("Save to make these questions permanent.")

            if save_clicked:
                st.session_state.is_saving = True
                gap_qa_to_save = []
                for i, gq in enumerate(session_gap_questions):
                    key = f"gap_answer_{i}"
                    gap_qa_to_save.append({
                        **gq,
                        "answer": st.session_state.gap_answers.get(key, ""),
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
                    saved = save_result.get("saved", 0)
                    updated = save_result.get("updated", 0)
                    st.success(f"✅ Saved {saved} new, updated {updated}.")
                    get_questions_from_fastapi.clear()
                else:
                    st.error("❌ Save failed — check API logs.")

        # Analyse gaps button (shown when no gap questions yet)
        if not has_any_gap and questions:
            st.divider()
            analyse_col, info_col2 = st.columns([1, 2])
            with analyse_col:
                analyse_clicked = st.button(
                    "🔍 Analyse schema gaps",
                    disabled=st.session_state.is_analyzing,
                    help="Uses AI to identify which document sections aren't covered.",
                    use_container_width=True,
                )
            with info_col2:
                st.caption("Optional: detect and fill schema coverage gaps.")

            if analyse_clicked and valid_document:
                st.session_state.is_analyzing = True
                current_qa = []
                for i, ques in enumerate(questions):
                    if ques.get("is_gap_question"):
                        continue
                    key = f"answer_{i}"
                    current_qa.append({
                        "question": ques.get("question", ""),
                        "answer": st.session_state.answers.get(key, ""),
                        "category": ques.get("category", ""),
                        "answer_type": ques.get("answer_type", "text"),
                    })
                doc_name = document_name_lookup.get(selected_document, selected_document)
                with st.spinner("🤖 Analysing schema coverage... this takes ~10 seconds."):
                    gap_result = call_gap_questions_endpoint(
                        department=selected_department,
                        document_type=selected_document,
                        document_name=doc_name,
                        questions_and_answers=current_qa,
                    )
                st.session_state.is_analyzing = False
                if gap_result:
                    gqs = gap_result.get("gap_questions", [])
                    if gqs:
                        st.session_state.gap_questions = gqs
                        st.session_state.gap_source = gap_result.get("source", "generated")
                        st.session_state.gap_doc_type = selected_document
                        for i, gq in enumerate(gqs):
                            key = f"gap_answer_{i}"
                            if key not in st.session_state.gap_answers:
                                st.session_state.gap_answers[key] = ""
                        st.rerun()
                    else:
                        st.success("✅ All schema sections are fully covered — no gaps found!")
                else:
                    st.error("❌ Gap analysis failed. Check API logs.")

        st.divider()

        # ── Navigation: Back / Action / Next ──
        nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])

        with nav_col1:
            if page > 0:
                if st.button("← Back", use_container_width=True):
                    st.session_state.q_page -= 1
                    st.rerun()

        with nav_col3:
            if page < total_pages - 1:
                if st.button("Next →", use_container_width=True):
                    # Progressive mode: auto-generate section for this page before moving
                    if st.session_state.prog_mode and schema_sections:
                        sec_idx, current_sec = get_sec_idx_for_page(page, schema_sections)
                        if current_sec and sec_idx not in st.session_state.prog_sections:
                            sec_qa = get_section_qa_for_sec_idx(sec_idx, schema_sections)
                            if sec_qa:
                                doc_memory = "\n\n".join(
                                    st.session_state.prog_sections[k]
                                    for k in sorted(st.session_state.prog_sections.keys())
                                )
                                sec_title = current_sec.get("title", "")
                                with st.spinner(f"⚡ Generating '{sec_title}'..."):
                                    result = call_generate_section(
                                        department=selected_department,
                                        document_type=selected_document,
                                        section=current_sec,
                                        questions_and_answers=sec_qa,
                                        doc_memory=doc_memory,
                                    )
                                if result and result.get("section_text"):
                                    st.session_state.prog_sections[sec_idx] = result["section_text"]
                    st.session_state.q_page += 1
                    st.rerun()

        with nav_col2:
            if not st.session_state.prog_mode:
                # ── Single-shot: generate button ──
                generate_button_clicked = st.button(
                    "⚡ Generate Document",
                    disabled=st.session_state.is_generating,
                    use_container_width=True,
                    type="primary",
                )
            else:
                # ── Progressive mode buttons ──
                generate_button_clicked = False

                # Always show the Generate Section button on every page (including last)
                if schema_sections:
                    _sec_idx_btn, _current_sec_btn = get_sec_idx_for_page(page, schema_sections)
                    _sec_already_done = _sec_idx_btn in st.session_state.prog_sections
                    _sec_name = _current_sec_btn.get("title", f"Section {_sec_idx_btn + 1}") if _current_sec_btn else f"Section {_sec_idx_btn + 1}"
                    _btn_label = f"✅ Regenerate: {_sec_name}" if _sec_already_done else f"⚡ Generate: {_sec_name}"

                    gen_sec_btn = st.button(
                        _btn_label,
                        disabled=st.session_state.prog_generating,
                        use_container_width=True,
                        type="primary",
                    )
                    if gen_sec_btn:
                        sec_idx, current_sec = _sec_idx_btn, _current_sec_btn
                        if current_sec:
                            sec_qa = get_section_qa_for_sec_idx(sec_idx, schema_sections)
                            if not sec_qa:
                                st.warning("⚠️ Answer at least one question for this section before generating.")
                            else:
                                doc_memory = "\n\n".join(
                                    st.session_state.prog_sections[k]
                                    for k in sorted(st.session_state.prog_sections.keys())
                                    if k != sec_idx
                                )
                                sec_title = current_sec.get("title", "")
                                with st.spinner(f"⚡ Generating '{sec_title}'..."):
                                    result = call_generate_section(
                                        department=selected_department,
                                        document_type=selected_document,
                                        section=current_sec,
                                        questions_and_answers=sec_qa,
                                        doc_memory=doc_memory,
                                    )
                                if result and result.get("section_text"):
                                    st.session_state.prog_sections[sec_idx] = result["section_text"]
                                    st.rerun()
                                else:
                                    st.error("❌ Section generation failed.")
                else:
                    gen_sec_btn = False

                # On the last page, also show Finalize to stitch everything together
                if page == total_pages - 1:
                    generate_button_clicked = st.button(
                        "📄 Finalize Document",
                        use_container_width=True,
                    )

        # ══════════════════════════════════════════════════════
        # Handle single-shot generate OR progressive finalize
        # ══════════════════════════════════════════════════════
        if generate_button_clicked and questions:
            if not st.session_state.prog_mode:
                # ═══ SINGLE-SHOT GENERATION (unchanged) ═══
                st.session_state.is_generating = True

                questions_and_answers = []
                for i, ques in enumerate(questions):
                    if ques.get("is_gap_question"):
                        continue
                    key = f"answer_{i}"
                    questions_and_answers.append({
                        "question": ques.get("question", ""),
                        "answer": st.session_state.answers.get(key, ""),
                        "category": ques.get("category", ""),
                        "answer_type": ques.get("answer_type", "text"),
                    })

                for i, ques in enumerate(questions):
                    if not ques.get("is_gap_question"):
                        continue
                    key = f"answer_{i}"
                    questions_and_answers.append({
                        "question": ques.get("question", ""),
                        "answer": st.session_state.answers.get(key, ""),
                        "category": ques.get("category", "Additional Information"),
                        "answer_type": ques.get("answer_type", "text"),
                        "is_gap_question": True,
                    })

                for i, gq in enumerate(st.session_state.gap_questions):
                    key = f"gap_answer_{i}"
                    questions_and_answers.append({
                        "question": gq.get("question", ""),
                        "answer": st.session_state.gap_answers.get(key, ""),
                        "category": gq.get("category", "Additional Information"),
                        "answer_type": gq.get("answer_type", "text"),
                        "is_gap_question": True,
                    })

                logger.info("Generate clicked — sending %d answers to agent", len(questions_and_answers))
                document_name_for_request = document_name_lookup.get(selected_document, selected_document)

                with st.spinner("Agent is generating your document... This may take 30-60 seconds."):
                    result = call_generate_endpoint(
                        department=selected_department,
                        document_type=selected_document,
                        document_name=document_name_for_request,
                        questions_and_answers=questions_and_answers,
                    )

                st.session_state.is_generating = False

                if result:
                    st.session_state.markdown_doc = result.get("generated_document", "")
                    generation_status = result.get("status", "unknown")
                    quality_scores = result.get("quality_scores", {})
                    retry_count = result.get("retry_count", 0)

                    new_gap_qs = result.get("gap_questions", [])
                    if new_gap_qs and not st.session_state.gap_questions:
                        st.session_state.gap_questions = new_gap_qs
                        st.session_state.gap_source = "generated"
                        st.session_state.gap_doc_type = selected_document
                        for i, gq in enumerate(new_gap_qs):
                            k = f"gap_answer_{i}"
                            if k not in st.session_state.gap_answers:
                                st.session_state.gap_answers[k] = ""
                        st.info(f"💡 {len(new_gap_qs)} gap question(s) detected. Answer them and regenerate.")

                    if generation_status == "passed":
                        st.success(f"✅ Document generated! (retries: {retry_count})")
                    else:
                        st.warning(f"⚠️ Generated with issues (status: {generation_status}, retries: {retry_count})")

                    if quality_scores:
                        with st.expander("📊 Quality Scores", expanded=(generation_status == "passed")):
                            score_cols = st.columns(len(quality_scores))
                            for col, (criterion, score) in zip(score_cols, quality_scores.items()):
                                col.metric(criterion.replace("_", " ").title(), f"{score}/5")
                else:
                    st.error("❌ Generation failed. Check the API logs for details.")

            else:
                # ═══ PROGRESSIVE FINALIZE ═══
                # Generate any remaining schema sections not yet generated
                all_qa = collect_all_answered_qa()
                for s_idx, sec in enumerate(schema_sections):
                    if s_idx in st.session_state.prog_sections:
                        continue   # already generated
                    doc_memory = "\n\n".join(
                        st.session_state.prog_sections[k]
                        for k in sorted(st.session_state.prog_sections.keys())
                    )
                    sec_title = sec.get("title", f"Section {s_idx + 1}")
                    with st.spinner(f"⚡ Generating '{sec_title}'..."):
                        result = call_generate_section(
                            department=selected_department,
                            document_type=selected_document,
                            section=sec,
                            questions_and_answers=all_qa,
                            doc_memory=doc_memory,
                        )
                    if result and result.get("section_text"):
                        st.session_state.prog_sections[s_idx] = result["section_text"]

                # Stitch all sections together
                full_doc = "\n\n".join(
                    st.session_state.prog_sections[k]
                    for k in sorted(st.session_state.prog_sections.keys())
                )
                st.session_state.markdown_doc = full_doc
                st.success("🎉 Document finalized! View it in the preview panel →")

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  RIGHT PANEL: Preview / Editor
# ═══════════════════════════════════════════════════════════════

with col_editor:
    st.markdown('<div class="separator-left">', unsafe_allow_html=True)

    header_col, publish_col = st.columns([4, 1])

    with header_col:
        if st.session_state.prog_mode:
            st.header("Preview")
        else:
            st.header("Markdown View")

    with publish_col:
        submit_publish = st.button("Publish")
        if submit_publish:
            if st.session_state.markdown_doc:
                st.balloons()
                st.success("Published! 🎉")
                logger.info("Document published")
            else:
                st.warning("Nothing to publish yet.")

    st.markdown('<div class="scrollable">', unsafe_allow_html=True)

    if st.session_state.prog_mode and st.session_state.prog_sections:
        # Show each generated section with status
        for sec_idx in sorted(st.session_state.prog_sections.keys()):
            sec_text = st.session_state.prog_sections[sec_idx]
            # Extract title from the generated markdown (first ## heading)
            first_line = sec_text.strip().split("\n")[0] if sec_text else ""
            sec_title = first_line.lstrip("# ").strip() if first_line.startswith("#") else f"Section {sec_idx + 1}"
            st.markdown(f"✅ **{sec_title}**")
            with st.expander(f"📖 {sec_title}", expanded=False):
                st.markdown(sec_text)

        # Show a combined preview
        full_preview = "\n\n".join(
            st.session_state.prog_sections[k]
            for k in sorted(st.session_state.prog_sections.keys())
        )
        st.session_state.markdown_doc = st.text_area(
            "Combined Preview",
            value=full_preview,
            height=300,
            label_visibility="collapsed",
            key="prog_editor",
        )
    else:
        st.session_state.markdown_doc = st.text_area(
            "Markdown Editor",
            value=st.session_state.markdown_doc,
            height=450,
            label_visibility="collapsed",
        )

    if st.session_state.markdown_doc:
        with st.expander("📖 Preview rendered document", expanded=False):
            st.markdown(st.session_state.markdown_doc)

    # Progressive mode: reset button
    if st.session_state.prog_mode and st.session_state.prog_sections:
        if st.button("🔄 Reset Progressive Session", use_container_width=True):
            st.session_state.prog_sections = {}
            st.session_state.q_page = 0
            st.session_state.markdown_doc = ""
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)