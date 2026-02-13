import streamlit as st
import requests 


FASTAPI_URL= "http://127.0.0.1:8000"

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


##API helpers to get each and every endpoint

@st.cache_data(ttl=300) #ensures safety against mutations by creating a new copy of data (cached data) and holds data for 300 seconds
def get_departments_from_fastapi():
    """To GET the departments from the FASTAPI"""
    try:
        response_received = requests.get(f"{FASTAPI_URL}/departments", timeout=10)
        response_received.raise_for_status() # to check the http response status code
        return response_received.json().get("departments", [])
    except Exception as error:
        st.error(f"Failed to load Departments: {error}")
        return []

@st.cache_data(ttl=300)
def get_document_types_from_fastapi(department_name):
    """To GET the document types from the FASTAPI"""
    try:
        response_received = requests.get(f"{FASTAPI_URL}/document-types",params={"department": department_name}, timeout=10)
        response_received.raise_for_status()
        return response_received.json().get("document_types", [])
    except Exception as error:
        st.error(f"Failed to load Document types: {error}")
        return []
    
@st.cache_data(ttl=300)
def get_questions_from_fastapi(document_type):
    """To GET the questions for all the document type from the FASTAPI"""
    try:
        response_received = requests.get(f"{FASTAPI_URL}/questions",params={"document_type": document_type}, timeout=10)
        response_received.raise_for_status() 
        return response_received.json().get("questions", [])
    except Exception as error:
        st.error(f"Failed to load Questions: {error}")
        return []

@st.cache_data(ttl=600) # holds data for 600 seconds
def get_notionpage_urls_from_fastapi():
    """To GET all the generated pages url in notion pages from the FASTAPI"""
    try: 
        print("about to call api")
        response_received = requests.get(f"{FASTAPI_URL}/get_all_urls", timeout=30)
        response_received.raise_for_status()
        return response_received.json().get("pages")
    except Exception as error:
        st.error(f"Failed to load URLs from Notion pages: {error}")
        return []
    


#---------------------------------------------------
# Load the initial data from thee functions
#---------------------------------------------------

pages = get_notionpage_urls_from_fastapi()
departments = get_departments_from_fastapi()                           
department_names = [d["name"] for d in departments]  


# -------------------------------------------------
# Session State (this will get updated based on the documents which are published into notion as well as we will cache those documents so that there is no need to call the document again and again and waste api limits)
# -------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = pages

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
    selected_department = st.selectbox(
        "Department",
        department_names or ["(no departments found)"],
        label_visibility="collapsed"
    )

    st.subheader("Document")
    valid_dept = selected_department and selected_department != "(no departments found)"
    doc_types = get_document_types_from_fastapi(selected_department) if valid_dept else []
    document_names = [d["document_type"] for d in doc_types] #here document_type is the fastapi endpoint parameter

    st.selectbox(
        "Document",
        document_names or ["(no departments found)"],
        label_visibility="collapsed",
    )

    st.subheader("Generation History")

    history_container = st.container(height=350)
    with history_container:
        for h in st.session_state.history:
            st.markdown(f"<a href='{h.get('url','#')}'>{h.get('title','Untitled')}</a>", unsafe_allow_html=True)

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
