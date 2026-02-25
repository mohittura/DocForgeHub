import streamlit as st
import requests
import logging
import re
import io
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable,
)

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
#  Gap Questions + Save Questions API helpers
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
            "   -> gap analysis done — source=%s, count=%d",
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
        logger.info("   -> saved=%d, updated=%d", result.get("saved", 0), result.get("updated", 0))
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
            "   -> generation complete — status=%s, length=%d chars",
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
        response_received = requests.post(
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
        response_received.raise_for_status()
        return response_received.json()
    except Exception as error:
        logger.error("Section generation failed: %s", error)
        st.error(f"Section generation failed: {error}")
        return None


# ---------------------------------------------------
# Load the initial data from the functions
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

# Track which document_type the gap questions belong to so we clear
# them automatically when the user switches documents.
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

    # Build a lookup from document_type -> document_name (needed for /generate)
    document_name_lookup = {
        doc_dict["document_type"]: doc_dict.get("document_name", doc_dict["document_type"])
        for doc_dict in doc_types
    }

    # Full department object needed for save-questions
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

# Seed answer slots for gap questions (if any already loaded)
for gap_idx, gap_question in enumerate(st.session_state.gap_questions):
    gap_answer_key = f"gap_answer_{gap_idx}"
    if gap_answer_key not in st.session_state.gap_answers:
        st.session_state.gap_answers[gap_answer_key] = gap_question.get("answer", "") or ""


# ═══════════════════════════════════════════════════════════════
#  Helper to render a single question widget
# ═══════════════════════════════════════════════════════════════

def render_question_widget(
    question: dict,
    widget_key: str,
    answer_state: dict,
    is_gap: bool = False,
) -> None:
    """
    Render one question as the correct Streamlit input widget.
    Writes the answer back into `answer_state[widget_key]`.
    is_gap=True adds a subtle AI badge next to the label.
    """
    question_label = question.get("question", "Question")
    answer_type = question.get("answer_type", "text")

    # We can't put HTML in widget labels, so show the badge as a separate markdown element
    if is_gap:
        st.markdown(
            f"<span style='font-size:0.9rem;font-weight:600;color:#eee;'>"
            f"{question_label} <span class='gap-badge'>AI</span></span>",
            unsafe_allow_html=True,
        )
        label_for_widget = "\u200b"   # zero-width space -> hides the default widget label
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
        # Default: plain text area
        answer_state[widget_key] = st.text_area(
            label_for_widget,
            value=answer_state.get(widget_key, ""),
            key=streamlit_widget_key,
        )


# ═══════════════════════════════════════════════════════════════
#  PDF generation — ReportLab Platypus
# ═══════════════════════════════════════════════════════════════

_PDF_BRAND  = colors.HexColor("#1E3A5F")
_PDF_ACCENT = colors.HexColor("#2E86AB")
_PDF_MID    = colors.HexColor("#8C9BAB")
_PDF_DARK   = colors.HexColor("#1A1A2E")
_PDF_TH_BG  = colors.HexColor("#2E86AB")
_PDF_ROW_A  = colors.HexColor("#F0F5FA")
_PDF_BORDER = colors.HexColor("#CBD5E1")


def _pdf_styles():
    kw = dict(fontName="Helvetica", textColor=_PDF_DARK, leading=14)
    return {
        "doc_title": ParagraphStyle("DocTitle", fontName="Helvetica-Bold", fontSize=24,
            textColor=_PDF_BRAND, leading=30, spaceAfter=4, alignment=TA_CENTER),
        "doc_sub":   ParagraphStyle("DocSub", fontName="Helvetica", fontSize=10,
            textColor=_PDF_MID, leading=15, spaceAfter=18, alignment=TA_CENTER),
        "h1": ParagraphStyle("H1", fontName="Helvetica-Bold", fontSize=16,
            textColor=_PDF_BRAND, leading=20, spaceBefore=20, spaceAfter=6),
        "h2": ParagraphStyle("H2", fontName="Helvetica-Bold", fontSize=13,
            textColor=_PDF_ACCENT, leading=17, spaceBefore=14, spaceAfter=4),
        "h3": ParagraphStyle("H3", fontName="Helvetica-BoldOblique", fontSize=11,
            textColor=_PDF_ACCENT, leading=15, spaceBefore=10, spaceAfter=3),
        "body":   ParagraphStyle("Body",   **kw, fontSize=10, spaceAfter=5),
        "bullet": ParagraphStyle("Bullet", **kw, fontSize=10, leftIndent=14, spaceAfter=3),
    }


def _pdf_clean(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    for k, v in {
        # Dashes & quotes
        "\u2014": "-", "\u2013": "-",
        "\u2019": "'", "\u2018": "'",
        "\u201c": '"', "\u201d": '"',
        "\u2022": "-", "\u2026": "...", "\u00a0": " ",
        # Math / comparison symbols
        "\u2265": ">=", "\u2264": "<=",
        "\u2260": "!=", "\u2248": "~=",
        "\u00d7": "x",  "\u00f7": "/",
        "\u00b1": "+/-", "\u2212": "-",
        "\u221e": "inf", "\u2205": "{}",
        "\u03b1": "alpha", "\u03b2": "beta", "\u03b3": "gamma",
        "\u03b4": "delta", "\u03bb": "lambda", "\u03bc": "mu",
        "\u03c3": "sigma", "\u03c0": "pi",
        # Arrows
        "\u2192": "->", "\u2190": "<-",
        "\u2194": "<->", "\u21d2": "=>",
        # Misc
        "\u00ae": "(R)", "\u00a9": "(C)", "\u2122": "(TM)",
        "\u2713": "ok", "\u2717": "x",
        "\u00b0": "deg", "\u00b2": "^2", "\u00b3": "^3",
        "\u20b9": "INR", "\u20ac": "EUR", "\u00a3": "GBP",
    }.items():
        text = text.replace(k, v)
    # Drop anything still outside latin-1 rather than showing black boxes
    return text.encode("latin-1", "replace").decode("latin-1")


def _pdf_para(text: str, style) -> Paragraph:
    text = _pdf_clean(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*",     r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`",       r"<font name='Courier'>\1</font>", text)
    return Paragraph(text, style)


def _parse_md_table(lines: list) -> list:
    rows = []
    for ln in lines:
        ln = ln.strip()
        if re.match(r"^\|[\s\-\|:]+\|$", ln):
            continue
        cells = [c.strip() for c in ln.split("|") if c.strip()]
        if cells:
            rows.append(cells)
    return rows


def _build_rl_table(rows: list, S: dict):
    if not rows:
        return None
    n = max(len(r) for r in rows)
    rows = [r + [""] * (n - len(r)) for r in rows]
    header = [Paragraph(f"<b>{_pdf_clean(c)}</b>", S["body"]) for c in rows[0]]
    body   = [[Paragraph(_pdf_clean(c), S["body"]) for c in row] for row in rows[1:]]
    col_w  = (A4[0] - 5 * cm) / n
    tbl = Table([header] + body, colWidths=[col_w] * n, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  _PDF_TH_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("LEADING",       (0, 0), (-1, -1), 13),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [_PDF_ROW_A, colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.4, _PDF_BORDER),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 7),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 7),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return tbl


def generate_pdf_from_markdown(markdown_text: str, document_title: str = "") -> bytes:
    """Render markdown to a professional A4 PDF. Returns raw PDF bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2.5*cm, rightMargin=2.5*cm,
        topMargin=2.5*cm,  bottomMargin=2.5*cm,
        title=document_title or "Document",
        author="DocForgeHub",
    )
    S = _pdf_styles()
    story = []

    if document_title:
        story += [
            Spacer(1, 0.8*cm),
            Paragraph(_pdf_clean(document_title), S["doc_title"]),
            HRFlowable(width="100%", thickness=2, color=_PDF_ACCENT, spaceBefore=4, spaceAfter=4),
            Paragraph("Generated by DocForgeHub", S["doc_sub"]),
            Spacer(1, 0.8*cm),
        ]

    lines = markdown_text.split("\n")
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if not s:
            story.append(Spacer(1, 4))
            i += 1
            continue
        if s.startswith("|"):
            tbl_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                tbl_lines.append(lines[i])
                i += 1
            tbl = _build_rl_table(_parse_md_table(tbl_lines), S)
            if tbl:
                story += [Spacer(1, 6), tbl, Spacer(1, 10)]
            continue
        if s.startswith("# "):
            story.append(_pdf_para(s[2:].strip(), S["h1"]))
            story.append(HRFlowable(width="100%", thickness=1, color=_PDF_BRAND, spaceAfter=4))
        elif s.startswith("## "):
            story.append(_pdf_para(s[3:].strip(), S["h2"]))
        elif s.startswith("### "):
            story.append(_pdf_para(s[4:].strip(), S["h3"]))
        elif re.match(r"^[\*\-\+]\s+", s):
            story.append(_pdf_para("• " + re.sub(r"^[\*\-\+]\s+", "", s), S["bullet"]))
        elif re.match(r"^\d+\.\s+", s):
            story.append(_pdf_para(s, S["bullet"]))
        else:
            story.append(_pdf_para(s, S["body"]))
        i += 1

    doc.build(story)
    return buf.getvalue()


def _safe_pdf_filename(title: str) -> str:
    safe = re.sub(r"[^\w\s\-]", "", title).strip()
    safe = re.sub(r"\s+", "_", safe)
    return (safe or "document") + ".pdf"


# ═══════════════════════════════════════════════════════════════
#  Fetch schema for progressive mode
#  The schema has ONE top-level section whose subsection titles
#  match question category names exactly.
# ═══════════════════════════════════════════════════════════════

raw_schema_section = None
prog_subsections = []
schema_sections = []

if is_valid_document and st.session_state.prog_mode:
    document_name_for_schema = document_name_lookup.get(selected_document, selected_document)
    try:
        schema_response = requests.get(
            f"{FASTAPI_URL}/required-section",
            params={"department": selected_department, "document_name": document_name_for_schema},
            timeout=15,
        )
        schema_response.raise_for_status()
        schema_response_data = schema_response.json()
        schema_response_data = schema_response_data.get("required_section", schema_response_data)
        all_schema_sections = schema_response_data.get("sections", [])
        if all_schema_sections:
            raw_schema_section = all_schema_sections[0]
            prog_subsections = raw_schema_section.get("subsections", [])
            schema_sections = [raw_schema_section]
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════
#  Build a UNIFIED question list: core + mongo gap + session gap
#  Each entry: (widget_key, question_dict, answer_state, is_gap)
# ═══════════════════════════════════════════════════════════════

all_questions = []

# Core questions (non-gap from MongoDB)
for question_idx, question in enumerate(questions):
    if question.get("is_gap_question"):
        continue
    all_questions.append((f"answer_{question_idx}", question, st.session_state.answers, False))

# MongoDB-persisted gap questions (is_gap_question=True in main list)
for question_idx, question in enumerate(questions):
    if not question.get("is_gap_question"):
        continue
    all_questions.append((f"answer_{question_idx}", question, st.session_state.answers, True))

# Session gap questions (freshly generated, stored in gap_answers)
for gap_idx, gap_question in enumerate(st.session_state.gap_questions):
    all_questions.append((f"gap_answer_{gap_idx}", gap_question, st.session_state.gap_answers, True))

# ─────────────────────────────────────────────────────────────────────────────
# Build category helpers — always defined regardless of prog_mode or schema
# ordered_categories     : unique categories in question order
# category_to_subsection : category_name_lower -> subsection dict
# ─────────────────────────────────────────────────────────────────────────────
ordered_categories: list = []
category_to_subsection: dict = {}
seen_category_names: set = set()
subsection_title_lower_map = {
    subsection.get("title", "").strip().lower(): subsection
    for subsection in prog_subsections
} if prog_subsections else {}

for _, question_dict, _, _ in all_questions:
    category_name = question_dict.get("category", "").strip()
    category_name_lower = category_name.lower()
    if category_name and category_name_lower not in seen_category_names:
        seen_category_names.add(category_name_lower)
        ordered_categories.append(category_name)
    if category_name_lower in subsection_title_lower_map:
        category_to_subsection[category_name_lower] = subsection_title_lower_map[category_name_lower]

del seen_category_names, subsection_title_lower_map  # cleanup temps


def get_page_categories(page_idx: int) -> list:
    """Unique categories for the 5 questions on page_idx, in order."""
    page_size = 5  # same value as PAGE_SIZE defined below
    page_range_start = page_idx * page_size
    page_range_end = min(page_range_start + page_size, len(all_questions))
    seen_categories = set()
    page_category_list = []
    for _, question_dict, _, _ in all_questions[page_range_start:page_range_end]:
        category_name = question_dict.get("category", "").strip()
        if category_name and category_name not in seen_categories:
            seen_categories.add(category_name)
            page_category_list.append(category_name)
    return page_category_list


def get_subsection_qa(category_name: str) -> list:
    """All answered Q&A for a specific category."""
    category_name_lower = category_name.strip().lower()
    return [
        {
            "question": question_dict.get("question", ""),
            "answer": answer_state.get(widget_key, ""),
            "category": question_dict.get("category", ""),
            "answer_type": question_dict.get("answer_type", "text"),
        }
        for widget_key, question_dict, answer_state, is_gap_flag in all_questions
        if question_dict.get("category", "").strip().lower() == category_name_lower
        and answer_state.get(widget_key, "").strip()
    ]


def prog_sections_ordered() -> list:
    """Return (category, text) pairs in document order for all generated sections."""
    return [
        (category_name, st.session_state.prog_sections[category_name])
        for category_name in ordered_categories
        if category_name in st.session_state.prog_sections
    ]


# ═══════════════════════════════════════════════════════════════
#  Pagination — always 5 questions per page
# ═══════════════════════════════════════════════════════════════

PAGE_SIZE = 5
total_questions = len(all_questions)
total_pages = max(1, -(-total_questions // PAGE_SIZE))  # ceil division

# Clamp page index within valid range
if st.session_state.q_page >= total_pages:
    st.session_state.q_page = total_pages - 1
if st.session_state.q_page < 0:
    st.session_state.q_page = 0

current_page = st.session_state.q_page
page_start = current_page * PAGE_SIZE
page_end = min(page_start + PAGE_SIZE, total_questions)
page_questions = all_questions[page_start:page_end]


def get_sec_idx_for_page(page_idx: int, sections: list) -> tuple:
    """
    Map a page index to a schema section index.

    Questions in MongoDB are ordered by (category_order, question_order).
    Schema sections are ordered the same way. So we do a straightforward
    proportional mapping: spread pages evenly across sections.

    e.g. 27 questions / 5 per page = 6 pages, 6 sections -> page N -> section N
         27 questions / 5 per page = 6 pages, 3 sections -> pages 0-1->sec 0, 2-3->sec 1, 4-5->sec 2
    """
    if not sections:
        return 0, None
    section_index = min(
        int(page_idx / max(total_pages, 1) * len(sections)),
        len(sections) - 1,
    )
    return section_index, sections[section_index]


def get_section_qa_for_sec_idx(section_idx: int, sections: list) -> list:
    """
    Return ALL answered Q&A for sections[section_idx].

    Since question categories match subsection titles (not top-level section
    titles), we match against the subsection titles of the target section.
    As a fallback we also include questions whose category matches the
    section title itself.

    If NO questions match by category (e.g. categories aren't set), we fall
    back to the proportional page range so the user always gets output.
    """
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


def collect_all_answered_qa():
    """Gather ALL answered Q&A from every question (core + gap)."""
    qa_list = []
    for widget_key, question_dict, answer_state, is_gap_flag in all_questions:
        answer_text = answer_state.get(widget_key, "")
        if answer_text.strip():
            qa_list.append({
                "question": question_dict.get("question", ""),
                "answer": answer_text,
                "category": question_dict.get("category", ""),
                "answer_type": question_dict.get("answer_type", "text"),
            })
    return qa_list


def collect_page_answered_qa(page_idx: int):
    """Gather answered Q&A for the 5 questions visible on page_idx."""
    page_range_start = page_idx * PAGE_SIZE
    page_range_end = min(page_range_start + PAGE_SIZE, total_questions)
    current_page_questions = all_questions[page_range_start:page_range_end]
    qa_list = []
    for widget_key, question_dict, answer_state, is_gap_flag in current_page_questions:
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
#  Helper: check if a question has been answered.
#  Reads the live Streamlit widget key first (reflects the current
#  run's input), then falls back to answer_state for off-page
#  questions that are not rendered this run.
# ═══════════════════════════════════════════════════════════════

def question_has_answer(widget_key: str, answer_state: dict) -> bool:
    """
    Returns True if the question identified by widget_key has a non-empty answer.
    Checks st.session_state widget key first (live value from this run),
    then answer_state (persisted value from a previous run).
    """
    streamlit_widget_key = f"widget_{widget_key}"
    if streamlit_widget_key in st.session_state:
        live_value = st.session_state[streamlit_widget_key]
        if isinstance(live_value, list):        # multi_select returns a list
            return len(live_value) > 0
        return str(live_value).strip() != ""
    return str(answer_state.get(widget_key, "")).strip() != ""


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

        # ── Progress bar — computed AFTER widgets so live widget values are available ──
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

        # Save button for freshly generated session gap questions
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

        # Analyse gaps button (shown when no gap questions exist yet)
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
                # ── Single-shot: generate button — enabled only when all questions are answered ──
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
                # ── Progressive mode: generate per-category ──
                generate_button_clicked = False
                progressive_all_answered = answered_count == total_questions and total_questions > 0
                progressive_remaining = total_questions - answered_count
                current_page_categories = get_page_categories(current_page)
                current_page_categories_done = (
                    bool(current_page_categories)
                    and all(cat in st.session_state.prog_sections for cat in current_page_categories)
                )
                current_page_category_label = (
                    ", ".join(current_page_categories) if current_page_categories else f"Page {current_page + 1}"
                )

                if not progressive_all_answered:
                    progressive_btn_label = f"🔒 Answer all questions first ({progressive_remaining} remaining)"
                elif current_page_categories_done:
                    progressive_btn_label = f"✅ Regenerate: {current_page_category_label}"
                else:
                    progressive_btn_label = f"⚡ Generate: {current_page_category_label}"

                generate_section_clicked = st.button(
                    progressive_btn_label,
                    disabled=(not progressive_all_answered) or st.session_state.prog_generating,
                    use_container_width=True,
                    type="primary",
                )
                if generate_section_clicked and progressive_all_answered:
                    for category_name in current_page_categories:
                        category_name_lower = category_name.strip().lower()
                        subsection_data = category_to_subsection.get(
                            category_name_lower, {"title": category_name, "type": "text"}
                        )
                        category_qa_list = get_subsection_qa(category_name)
                        if not category_qa_list:
                            continue
                        section_for_api_call = {
                            "title": raw_schema_section.get("title", "Document Overview") if raw_schema_section else "",
                            "subsections": [subsection_data],
                        }
                        previously_generated_text = "\n\n".join(
                            st.session_state.prog_sections[cat]
                            for cat in ordered_categories
                            if cat in st.session_state.prog_sections and cat != category_name
                        )
                        with st.spinner(f"⚡ Generating '{category_name}'..."):
                            section_result = call_generate_section(
                                department=selected_department,
                                document_type=selected_document,
                                section=section_for_api_call,
                                questions_and_answers=category_qa_list,
                                doc_memory=previously_generated_text,
                            )
                        if section_result and section_result.get("section_text"):
                            st.session_state.prog_sections[category_name] = section_result["section_text"]
                    st.rerun()

                # On last page also show Finalize
                if current_page == total_pages - 1:
                    generate_button_clicked = st.button(
                        "📄 Finalize Document" if progressive_all_answered
                        else f"🔒 Finalize ({progressive_remaining} questions remaining)",
                        disabled=not progressive_all_answered,
                        use_container_width=True,
                    )


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
                # Generate any category not yet done, then stitch in order
                for category_name in ordered_categories:
                    if category_name in st.session_state.prog_sections:
                        continue
                    category_name_lower = category_name.strip().lower()
                    subsection_data = category_to_subsection.get(
                        category_name_lower, {"title": category_name, "type": "text"}
                    )
                    category_qa_list = get_subsection_qa(category_name)
                    if not category_qa_list:
                        continue
                    section_for_api_call = {
                        "title": raw_schema_section.get("title", "Document Overview") if raw_schema_section else "Document Overview",
                        "subsections": [subsection_data],
                    }
                    previously_generated_text = "\n\n".join(
                        st.session_state.prog_sections[cat]
                        for cat in ordered_categories
                        if cat in st.session_state.prog_sections and cat != category_name
                    )
                    with st.spinner(f"⚡ Generating '{category_name}'..."):
                        section_result = call_generate_section(
                            department=selected_department,
                            document_type=selected_document,
                            section=section_for_api_call,
                            questions_and_answers=category_qa_list,
                            doc_memory=previously_generated_text,
                        )
                    if section_result and section_result.get("section_text"):
                        st.session_state.prog_sections[category_name] = section_result["section_text"]

                # Stitch all generated sections in document order
                full_doc = "\n\n".join(
                    st.session_state.prog_sections[cat]
                    for cat in ordered_categories
                    if cat in st.session_state.prog_sections
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

    if st.session_state.prog_mode and st.session_state.prog_sections:
        # Show each generated section with status
        for generated_category in [cat for cat in ordered_categories if cat in st.session_state.prog_sections]:
            section_text = st.session_state.prog_sections[generated_category]
            # Extract title from the generated markdown (first ## heading)
            first_line = section_text.strip().split("\n")[0] if section_text else ""
            section_display_title = (
                first_line.lstrip("# ").strip() if first_line.startswith("#") else f"Section: {generated_category}"
            )
            st.markdown(f"✅ **{section_display_title}**")
            with st.expander(f"📖 {section_display_title}", expanded=False):
                st.markdown(section_text)

        # Show a combined editable preview
        full_preview = "\n\n".join(
            st.session_state.prog_sections[cat]
            for cat in ordered_categories
            if cat in st.session_state.prog_sections
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

        st.divider()
        try:
            pdf_bytes = generate_pdf_from_markdown(
                st.session_state.markdown_doc,
                document_title=selected_document,
            )
            st.download_button(
                label="⬇ Download PDF",
                data=pdf_bytes,
                file_name=_safe_pdf_filename(selected_document),
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as _pdf_err:
            st.warning(f"⚠️ Could not generate PDF: {_pdf_err}")

    # Progressive mode: reset button
    if st.session_state.prog_mode and st.session_state.prog_sections:
        if st.button("🔄 Reset Progressive Session", use_container_width=True):
            st.session_state.prog_sections = {}
            st.session_state.q_page = 0
            st.session_state.markdown_doc = ""
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)