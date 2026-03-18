"""
ui/streamlit_uidemo.py

Unified Streamlit entry point for DOCFORGEHUB.

The sidebar contains the app-switcher buttons so the user can switch
between the two apps from anywhere without losing their place:

    📄 DocForgeHub  — AI document generation  (DocForgeHub FastAPI on port 8000)
    🤖 CiteRagLab   — RAG chat interface      (CiteRagLab FastAPI on port 8001)

The rest of the sidebar is rendered by whichever app is active.

Run with:
    streamlit run ui/streamlit_uidemo.py

Both apps share the same browser session.
State keys are namespaced — "dfh_" for DocForgeHub, "crl_" for CiteRagLab —
so they never collide.
"""

import sys
import os
import logging
import streamlit as st

# ── sys.path: ensure both ui/ and the project root are importable ─────────────
_UI_DIR       = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_UI_DIR, ".."))
for _path in [_UI_DIR, _PROJECT_ROOT]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

# ── Logging setup (done once here, before any module that logs) ───────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ui.streamlit_uidemo")

# ── Page config — must be the very first Streamlit call ──────────────────────
st.set_page_config(
    page_title="DocForge · CiteRagLab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger.info("🌐 streamlit_uidemo.py loaded")

# ── Active app state ──────────────────────────────────────────────────────────
if "active_app" not in st.session_state:
    st.session_state.active_app = "CiteRagLab"
    logger.info("active_app initialised to CiteRagLab")

# ── Sidebar: app-switcher buttons ─────────────────────────────────────────────
# These sit at the very top of the sidebar so they are always visible
# regardless of which app is currently active.
with st.sidebar:
    st.caption("SWITCH APP")
    doc_forge_button_clicked = st.button(
        "📄 DocForgeHub",
        key="nav_button_doc_forge",
        use_container_width=True,
        type="primary" if st.session_state.active_app == "DocForgeHub" else "secondary",
    )
    cite_rag_button_clicked = st.button(
        "🤖 CiteRagLab",
        key="nav_button_cite_rag",
        use_container_width=True,
        type="primary" if st.session_state.active_app == "CiteRagLab" else "secondary",
    )
    st.divider()

if doc_forge_button_clicked:
    logger.info("Navigation → DocForgeHub")
    st.session_state.active_app = "DocForgeHub"
    st.rerun()

if cite_rag_button_clicked:
    logger.info("Navigation → CiteRagLab")
    st.session_state.active_app = "CiteRagLab"
    st.rerun()

# ── Route to the selected app ─────────────────────────────────────────────────
logger.info("Rendering app: %s", st.session_state.active_app)

if st.session_state.active_app == "DocForgeHub":
    from doc_forge_ui import render_doc_forge_ui
    render_doc_forge_ui()
else:
    from cite_rag_lab_ui_rag import render_cite_rag_lab_ui
    render_cite_rag_lab_ui()