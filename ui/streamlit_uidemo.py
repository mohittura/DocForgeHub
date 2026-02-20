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

    st.subheader("Generation History")

    history_container = st.container(height=350)
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


# -------------------------------
# QUESTIONS PANEL
# -------------------------------

with col_questions:
    # =================================================
    # The questions will also be fetched from the mongodb and will change based on the document dropdown selection
    # =================================================
    st.markdown('<div class="separator-right">', unsafe_allow_html=True)

    st.header("Questions")

    if not questions:
        st.info("Select a department and document to load questions.")
    else:
        # ── Core questions ────────────────────────────────────────
        current_category = ""
        for i, ques in enumerate(questions):
            # Skip gap questions already persisted — we'll show them in the gap section
            if ques.get("is_gap_question"):
                continue

            category = ques.get("category", "")
            if category and category != current_category:
                current_category = category
                st.subheader(category)

            render_question_widget(
                ques=ques,
                widget_key=f"answer_{i}",
                state_dict=st.session_state.answers,
                is_gap=False,
            )

        # ── Gap Questions section ────────────────────────────────
        # Two sources of gap questions:
        #   A) Already persisted in MongoDB (loaded as part of /questions)
        #   B) Freshly generated in session (from /gap-questions call)

        # Source A: from MongoDB (is_gap_question=True in the main questions list)
        mongo_gap_questions = [q for q in questions if q.get("is_gap_question")]

        # Source B: freshly generated this session
        session_gap_questions = st.session_state.gap_questions

        has_any_gap = bool(mongo_gap_questions or session_gap_questions)

        if has_any_gap:
            st.divider()
            st.markdown(
                "<div class='gap-banner'>"
                "<strong>🤖 AI-Detected Gap Questions</strong><br>"
                "These questions were generated to cover schema sections not addressed "
                "by the core questionnaire. Answer them to improve document quality."
                "</div>",
                unsafe_allow_html=True,
            )

        # Render MongoDB-persisted gap questions (already in main questions list,
        # so their answers live in st.session_state.answers)
        if mongo_gap_questions:
            st.caption("📦 Previously saved gap questions (loaded from database)")
            for i, ques in enumerate(questions):
                if not ques.get("is_gap_question"):
                    continue
                render_question_widget(
                    ques=ques,
                    widget_key=f"answer_{i}",
                    state_dict=st.session_state.answers,
                    is_gap=True,
                )

        # Render freshly-generated session gap questions
        if session_gap_questions:
            source_label = (
                "⚡ Freshly generated for this session"
                if st.session_state.gap_source == "generated"
                else "📦 Loaded from database"
            )
            if st.session_state.gap_source == "generated":
                st.caption(source_label)

            for i, gq in enumerate(session_gap_questions):
                render_question_widget(
                    ques=gq,
                    widget_key=f"gap_answer_{i}",
                    state_dict=st.session_state.gap_answers,
                    is_gap=True,
                )

            # ── Save gap questions to MongoDB ───────────────────
            if st.session_state.gap_source == "generated":
                st.markdown("")
                save_col, info_col = st.columns([1, 2])
                with save_col:
                    save_clicked = st.button(
                        "💾 Save gap questions",
                        disabled=st.session_state.is_saving,
                        help="Saves these questions + your answers to the database so "
                             "they appear automatically next time this document type is loaded.",
                        use_container_width=True,
                    )
                with info_col:
                    st.caption(
                        "Save to make these questions permanent for this document type. "
                        "Future users won't need to regenerate them."
                    )

                if save_clicked:
                    st.session_state.is_saving = True

                    # Build gap Q&A payload with current answers
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
                        st.success(
                            f"✅ Saved {saved} new question(s), updated {updated}. "
                            f"They'll appear automatically next time!"
                        )
                        # Invalidate the questions cache so next load picks them up
                        get_questions_from_fastapi.clear()
                    else:
                        st.error("❌ Save failed — check API logs.")

        # ── Analyse gaps button (shown when no gap questions yet) ──
        if not has_any_gap and not session_gap_questions and questions:
            st.divider()
            analyse_col, info_col2 = st.columns([1, 2])
            with analyse_col:
                analyse_clicked = st.button(
                    "🔍 Analyse schema gaps",
                    disabled=st.session_state.is_analyzing,
                    help="Uses AI to identify which document sections aren't covered "
                         "by the existing questions and generates targeted questions for them.",
                    use_container_width=True,
                )
            with info_col2:
                st.caption("Optional: detect and fill schema coverage gaps before generating.")

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
                        # Seed answer slots
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

    # ── Generate button ──────────────────────────────────────────
    generate_button_clicked = st.button(
        "⚡ Generate Document",
        disabled=st.session_state.is_generating,
        use_container_width=True,
        type="primary",
    )

    if generate_button_clicked and questions:
        st.session_state.is_generating = True

        # Build full Q&A payload: core + mongo gap + session gap
        questions_and_answers = []

        # Core questions
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

        # MongoDB-persisted gap questions
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

        # Session gap questions
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

        # Look up the document_name from the selected document_type
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
            quality_issues = result.get("quality_issues", [])
            quality_scores = result.get("quality_scores", {})
            quality_suggestions = result.get("quality_suggestions", [])
            retry_count = result.get("retry_count", 0)

            # If the agent itself found NEW gap questions during generation,
            # surface them for the user (in case they weren't loaded via
            # the Analyse button first).
            new_gap_qs = result.get("gap_questions", [])
            if new_gap_qs and not st.session_state.gap_questions:
                st.session_state.gap_questions = new_gap_qs
                st.session_state.gap_source = "generated"
                st.session_state.gap_doc_type = selected_document
                for i, gq in enumerate(new_gap_qs):
                    k = f"gap_answer_{i}"
                    if k not in st.session_state.gap_answers:
                        st.session_state.gap_answers[k] = ""
                st.info(
                    f"💡 {len(new_gap_qs)} gap question(s) were detected during generation. "
                    f"Answer them above and regenerate for a richer document."
                )

            if generation_status == "passed":
                st.success(f"✅ Document generated successfully! (retries: {retry_count})")
            else:
                st.warning(
                    f"⚠️ Document generated with some issues (status: {generation_status}, retries: {retry_count})"
                )

            # Show quality scores if available (from LLM review)
            if quality_scores:
                with st.expander("📊 Quality Scores", expanded=(generation_status == "passed")):
                    score_cols = st.columns(len(quality_scores))
                    for col, (criterion, score) in zip(score_cols, quality_scores.items()):
                        col.metric(criterion.replace("_", " ").title(), f"{score}/5")

            # Show issues if any
            if quality_issues:
                with st.expander("⚠️ Quality Issues"):
                    for issue in quality_issues:
                        st.write(f"- {issue}")

            # Show improvement suggestions
            if quality_suggestions:
                with st.expander("💡 Suggestions for Improvement"):
                    for suggestion in quality_suggestions:
                        st.write(f"- {suggestion}")
        else:
            st.error("❌ Generation failed. Check the API logs for details.")

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# -------------------------------
# MARKDOWN EDITOR PANEL
# -------------------------------
with col_editor:

    st.markdown('<div class="separator-left">', unsafe_allow_html=True)

    header_col, publish_col = st.columns([4, 1])

    with header_col:
        st.header("Markdown View")

    with publish_col:
        submit_publish = st.button("Publish")
        if submit_publish:
            if st.session_state.markdown_doc:
                st.balloons()
                st.success("Published! 🎉")
                logger.info("Document published")
            else:
                st.warning("Nothing to publish yet — generate a document first.")

    st.markdown('<div class="scrollable">', unsafe_allow_html=True)

    st.session_state.markdown_doc = st.text_area(
        "Markdown Editor",
        value=st.session_state.markdown_doc,
        height=450,
        label_visibility="collapsed"
    )

    # Preview tab (rendered Markdown)
    if st.session_state.markdown_doc:
        with st.expander("📖 Preview rendered document", expanded=False):
            st.markdown(st.session_state.markdown_doc)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)