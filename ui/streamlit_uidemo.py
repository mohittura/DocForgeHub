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
        response_received = requests.get(f"{FASTAPI_URL}/document-types", params={"department": department_name}, timeout=10)
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
        response_received = requests.get(f"{FASTAPI_URL}/questions", params={"document_type": document_type}, timeout=10)
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

    request_body = {
        "department": department,
        "document_type": document_type,
        "document_name": document_name,
        "questions_and_answers": questions_and_answers,
    }

    try:
        response = requests.post(
            f"{FASTAPI_URL}/generate",
            json=request_body,
            timeout=120,  # generation can take a while
        )
        response.raise_for_status()
        result = response.json()
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

if "markdown_doc" not in st.session_state:
    st.session_state.markdown_doc = ""

if "is_generating" not in st.session_state:
    st.session_state.is_generating = False


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

    selected_document = st.selectbox(
        "Document",  
        document_names or ["(select a department first)"],
        label_visibility="collapsed",
    )

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

# -------------------------------
# QUESTIONS PANEL
# -------------------------------

with col_questions:
    # =================================================
    # The questions will also be fetched from the mongodb and will change based on the document dropdown selection
    # =================================================
    st.markdown('<div class="separator-right">', unsafe_allow_html=True)

    st.header("Questions")

    st.markdown('<div class="scrollable">', unsafe_allow_html=True)

    if not questions:
        st.info("Select a department and document to load questions.")
    else:
        current_category = ""
        for i, ques in enumerate(questions):
            # Show a category heading when it changes
            category = ques.get("category", "")
            if category and category != current_category:
                current_category = category
                st.subheader(category)

            label = ques.get("question", f"Question {i + 1}")
            key = f"answer_{i}"
            answer_type = ques.get("answer_type", "text")

            # Render the appropriate input widget based on answer_type
            if answer_type == "structured_list":
                st.session_state.answers[key] = st.text_area(
                    label, value=st.session_state.answers.get(key, ""),
                    help="Enter items separated by newlines",
                )
            elif answer_type == "select":
                options = ques.get("options", [])
                current_value = st.session_state.answers.get(key, "")
                selected_index = options.index(current_value) if current_value in options else 0
                st.session_state.answers[key] = st.selectbox(
                    label,
                    options=options,
                    index=selected_index,
                    key=f"select_{i}",
                )
            elif answer_type == "multi_select":
                options = ques.get("options", [])
                current_value = st.session_state.answers.get(key, "")
                default_values = (
                    [v.strip() for v in current_value.split(",") if v.strip()]
                    if isinstance(current_value, str) and current_value
                    else []
                )
                selected_values = st.multiselect(
                    label,
                    options=options,
                    default=[v for v in default_values if v in options],
                    key=f"multiselect_{i}",
                )
                st.session_state.answers[key] = ", ".join(selected_values)
            else:
                # Default: plain text area
                st.session_state.answers[key] = st.text_area(
                    label, value=st.session_state.answers.get(key, ""),
                )

    # ── Generate button ──────────────────────────────────────────
    generate_button_clicked = st.button(
        "Generate Document",
        disabled=st.session_state.is_generating,
        use_container_width=True,
    )

    if generate_button_clicked and questions:
        st.session_state.is_generating = True

        # Build the Q&A payload for the agent
        questions_and_answers = []
        for i, ques in enumerate(questions):
            key = f"answer_{i}"
            questions_and_answers.append({
                "question": ques.get("question", ""),
                "answer": st.session_state.answers.get(key, ""),
                "category": ques.get("category", ""),
                "answer_type": ques.get("answer_type", "text"),
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