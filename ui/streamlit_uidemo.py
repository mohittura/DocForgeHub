import streamlit as st
import requests 


FASTAPI_URL= "https://127.0.0.1:8000"

st.set_page_config(layout="wide")

# -------------------------------------------------
# CSS
# -------------------------------------------------
st.markdown(
    """
    <style>

    /* Vertical separators */
    .block-container {
        padding-top: none;
    }

    .separator-right {
        border-right: 1px solid #444;
        padding-right: 1rem;
    }

    .separator-left {
        border-left: 1px solid #444;
        padding-left: none;
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
# Session State (this will get updated based on the documents which are published into notion as well as we will cache those documents so that there is no need to call the document again and again and waste api limits)
# -------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = ["Doc1", "Doc2", "Doc3", "Doc4", "Doc5", "Doc6"]

if "answers" not in st.session_state:
    st.session_state.answers = {
        "q1": "",
        "q2": "",
        "qn": "",
    }

if "markdown_doc" not in st.session_state:
    st.session_state.markdown_doc = ""


# -------------------------------------------------
# Generator (we will replace this with our agent logic(from different file) to generate the docuement)
# -------------------------------------------------
def generate_document():
    st.session_state.markdown_doc = f"""# Generated Document

## Question 1
{st.session_state.answers['q1']}

## Question 2   
{st.session_state.answers['q2']}

## Question N
{st.session_state.answers['qn']}
"""


# =================================================
# LEFT SIDEBAR 
# =================================================
with st.sidebar:

    st.write("<h1>📄</br>DocForge Hub</h1>", unsafe_allow_html=True)

    st.subheader("Department")
    # ====================================================
    # This will also be generated based on the values stored in mongodb and will load things realtime
    # ====================================================
    st.selectbox(
        "Department",
        ["Engineering", "HR", "Finance"],
        label_visibility="collapsed",
    )

    st.subheader("Document")
    st.selectbox(
        "Document",
        ["Spec", "Policy", "Report"],
        label_visibility="collapsed",
    )

    st.subheader("Generation History")

    history_container = st.container(height=350)
    with history_container:
        for h in st.session_state.history:
            st.button(h, use_container_width=True)


# =================================================
# MAIN AREA
# =================================================
col_questions, col_editor = st.columns([2, 3])

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

    st.session_state.answers["q1"] = st.text_area(
        "Question 1",
        value=st.session_state.answers["q1"]
    )

    st.session_state.answers["q2"] = st.text_area(
        "Question 2",
        value=st.session_state.answers["q2"],
    )

    st.session_state.answers["qn"] = st.text_area(
        "Question N",
        value=st.session_state.answers["qn"],
    )

    if st.button("Generate Document"):
        generate_document()

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
            st.balloons()

    st.markdown('<div class="scrollable">', unsafe_allow_html=True)

    st.session_state.markdown_doc = st.text_area(
        "Markdown Editor",
        value=st.session_state.markdown_doc,
        height=450,
        label_visibility="collapsed"
    )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
