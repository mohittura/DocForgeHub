"""
ui/cite_rag_lab_ui_rag.py

CiteRagLab — RAG chat interface.

Built entirely with native Streamlit components.
No custom CSS. No unsafe_allow_html anywhere.

UI layout (matches wireframe):
  LEFT SIDEBAR
    ├── 🤖 CiteRagLab header
    ├── [＋ New Chat] button
    ├── [Search Chat] text input
    ├── "Your Chats" — scrollable session list via st.container
    └── Session ID caption footer

  MAIN AREA (tabs)
    💬 Chat      — st.chat_message bubbles + st.chat_input (only when a session is active)
    🔍 Inspector — retrieval debug panel (chunks + scores)
    📥 Ingest    — Notion ingestion trigger
    📊 Evaluation— RAGAS evaluation dashboard

Called from ui/streamlit_uidemo.py via render_cite_rag_lab_ui().
"""

import sys
import os
import uuid
import logging
import streamlit as st

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from api_helpers_rag import (
    call_chat,
    call_retrieval_debug,
    call_ingest_notion,
    call_run_evaluation,
    call_delete_session,
)

logger = logging.getLogger("ui.cite_rag_lab_ui_rag")


# ─────────────────────────────────────────────────────────────────────────────
#  Session state bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def _init_crl_session_state():
    """Initialise all CiteRagLab session-state keys once per browser session.

    A single default session is created here the very first time the app
    loads so the chat area is immediately usable.  After that, new sessions
    are only ever created when the user explicitly clicks ＋ New Chat.
    """
    if "crl_sessions" not in st.session_state:
        # Dict: session_id → {"title": str, "messages": list[dict]}
        st.session_state.crl_sessions = {}

    if "crl_active_session_id" not in st.session_state:
        st.session_state.crl_active_session_id = None

    # Create the one default session on first load only.
    # The guard on crl_sessions being empty ensures this never fires again
    # on subsequent reruns — not on tab switches, not on filter changes,
    # not on navigation between DocForgeHub and CiteRagLab.
    if not st.session_state.crl_sessions:
        default_id = str(uuid.uuid4())[:8]
        st.session_state.crl_sessions[default_id] = {
            "title":    "Chat 1",
            "messages": [],
        }
        st.session_state.crl_active_session_id = default_id
        logger.info("🆕 Default session created on first load — id=%s", default_id)

    if "crl_search_term" not in st.session_state:
        st.session_state.crl_search_term = ""

    if "crl_filters" not in st.session_state:
        st.session_state.crl_filters = {"industry": "", "doc_type": "", "version": ""}

    if "crl_ingest_running" not in st.session_state:
        st.session_state.crl_ingest_running = False

    if "crl_eval_running" not in st.session_state:
        st.session_state.crl_eval_running = False


# ─────────────────────────────────────────────────────────────────────────────
#  Session helpers
# ─────────────────────────────────────────────────────────────────────────────

def _create_new_session():
    """Create a new chat session, make it active, and return its ID."""
    session_id    = str(uuid.uuid4())[:8]
    chat_number   = len(st.session_state.crl_sessions) + 1
    session_title = f"Chat {chat_number}"
    st.session_state.crl_sessions[session_id] = {
        "title":    session_title,
        "messages": [],
    }
    st.session_state.crl_active_session_id = session_id
    logger.info("🆕 New session created — id=%s  title='%s'", session_id, session_title)
    return session_id


def _get_active_messages():
    """Return the message list for the currently active session, or []."""
    active_id = st.session_state.crl_active_session_id
    if active_id and active_id in st.session_state.crl_sessions:
        return st.session_state.crl_sessions[active_id]["messages"]
    return []


def _append_message(role, content, citations=None, pipeline_meta=None):
    """Append one message dict to the active session's message list."""
    messages = _get_active_messages()
    messages.append({
        "role":          role,
        "content":       content,
        "citations":     citations or [],
        "pipeline_meta": pipeline_meta or {},
    })
    logger.info(
        "   💬 Message appended — role=%s  length=%d chars  citations=%d",
        role, len(content), len(citations or []),
    )


def _get_filtered_sessions():
    """Return (session_id, session_data) pairs matching crl_search_term."""
    search_term = st.session_state.crl_search_term.lower().strip()
    return [
        (session_id, session_data)
        for session_id, session_data in st.session_state.crl_sessions.items()
        if not search_term
        or search_term in session_data["title"].lower()
        or any(search_term in msg["content"].lower() for msg in session_data["messages"])
    ]


# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def _render_sidebar():
    with st.sidebar:
        st.header("🤖 CiteRagLab")
        st.divider()

        # ── New Chat button ────────────────────────────────────────────────────
        if st.button("＋ New Chat", use_container_width=True, key="crl_new_chat_button"):
            _create_new_session()
            logger.info("New Chat button clicked")
            st.rerun()

        # ── Search Chat input ─────────────────────────────────────────────────
        st.text_input(
            "Search Chat",
            placeholder="Search chats…",
            key="crl_search_term",
        )

        # ── Your Chats list ────────────────────────────────────────────────────
        st.caption("YOUR CHATS")

        filtered_sessions = _get_filtered_sessions()

        if not filtered_sessions:
            st.caption("No chats yet.")
        else:
            chat_list_container = st.container(height=260)
            with chat_list_container:
                for session_id, session_data in filtered_sessions:
                    is_active_session = (session_id == st.session_state.crl_active_session_id)
                    session_title     = session_data["title"]
                    message_count     = len(session_data["messages"])
                    button_label      = f"{'▶ ' if is_active_session else ''}{session_title} ({message_count})"
                    button_type       = "primary" if is_active_session else "secondary"

                    if st.button(
                        button_label,
                        key=f"crl_select_session_{session_id}",
                        use_container_width=True,
                        type=button_type,
                    ):
                        logger.info(
                            "Session selected — id=%s  title='%s'",
                            session_id, session_title,
                        )
                        st.session_state.crl_active_session_id = session_id
                        st.rerun()

        # ── Session ID footer ──────────────────────────────────────────────────
        displayed_session_id = st.session_state.crl_active_session_id or "—"
        st.caption(f"Session ID: {displayed_session_id}")
        st.divider()

        # ── Delete current session ─────────────────────────────────────────────
        if st.session_state.crl_active_session_id:
            if st.button(
                "🗑️ Delete Session",
                use_container_width=True,
                key="crl_delete_session_button",
            ):
                session_id_to_delete = st.session_state.crl_active_session_id
                logger.info("Deleting session id=%s", session_id_to_delete)
                call_delete_session(session_id_to_delete)
                del st.session_state.crl_sessions[session_id_to_delete]
                st.session_state.crl_active_session_id = None
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  Tab: Chat
# ─────────────────────────────────────────────────────────────────────────────

def _render_chat_tab():
    """Main chat interface using st.chat_message and st.chat_input."""
    active_session_id = st.session_state.crl_active_session_id

    # No session active — this only happens after the user has deleted all sessions.
    # Never auto-create here; new sessions are created only via ＋ New Chat.
    if active_session_id is None or active_session_id not in st.session_state.crl_sessions:
        st.session_state.crl_active_session_id = None   # ensure consistent state
        st.info("All sessions deleted — click **＋ New Chat** in the sidebar to start a new conversation.")
        logger.info("🖥️  Chat tab rendered — no active session (all deleted)")
        return

    session_title = st.session_state.crl_sessions[active_session_id]["title"]
    logger.info(
        "🖥️  Rendering chat tab — session_id=%s  title='%s'",
        active_session_id, session_title,
    )

    st.subheader(f"CiteRagLab — {session_title}")

    # ── Metadata filter strip ────────────────────────────────────────────────
    filter_col_industry, filter_col_doctype, filter_col_version, filter_col_clear = st.columns([2, 2, 1, 1])
    with filter_col_industry:
        st.session_state.crl_filters["industry"] = st.text_input(
            "Industry",
            value=st.session_state.crl_filters["industry"],
            placeholder="Industry (e.g. Fintech)",
            key="crl_filter_industry",
        )
    with filter_col_doctype:
        st.session_state.crl_filters["doc_type"] = st.text_input(
            "Doc type",
            value=st.session_state.crl_filters["doc_type"],
            placeholder="Doc type (e.g. Policy)",
            key="crl_filter_doc_type",
        )
    with filter_col_version:
        st.session_state.crl_filters["version"] = st.text_input(
            "Version",
            value=st.session_state.crl_filters["version"],
            placeholder="Version",
            key="crl_filter_version",
        )
    with filter_col_clear:
        st.write("")   # vertical alignment spacer
        if st.button("✕ Clear filters", use_container_width=True, key="crl_clear_filters_button"):
            logger.info("Filters cleared")
            st.session_state.crl_filters = {"industry": "", "doc_type": "", "version": ""}
            st.rerun()

    st.divider()

    # ── Message history in a fixed-height scrollable container ───────────────
    # Keeping history inside a bounded container means the chat input below it
    # never scrolls off screen — it stays static at the bottom at all times.
    all_messages = _get_active_messages()

    messages_container = st.container(height=460, border=False)
    with messages_container:
        if not all_messages:
            st.info("🤖 Citter is ready — ask anything about your document library.")
        else:
            for message in all_messages:
                role          = message["role"]
                content       = message["content"]
                citations     = message.get("citations", [])
                pipeline_meta = message.get("pipeline_meta", {})

                avatar = "🤖" if role == "assistant" else "👤"
                with st.chat_message(role, avatar=avatar):
                    st.markdown(content)

                    # Citations shown as a collapsed expander under bot messages
                    if citations and role == "assistant":
                        with st.expander(f"📚 {len(citations)} source(s)", expanded=False):
                            for citation in citations:
                                title    = citation.get("title", "")
                                section  = citation.get("section", "")
                                doc_type = citation.get("doc_type", "")
                                score    = citation.get("score", 0)
                                index    = citation.get("index", "")
                                location = f"{title} → {section}" if section else title
                                st.markdown(
                                    f"**[{index}]** {location}  `{doc_type}`  score: `{score}`"
                                )

                    # Pipeline metadata shown as a caption under bot messages
                    if pipeline_meta and role == "assistant":
                        mode      = pipeline_meta.get("mode", "")
                        avg_score = pipeline_meta.get("avg_score", "")
                        rewritten = pipeline_meta.get("rewritten", "")
                        meta_parts = []
                        if mode:
                            meta_parts.append(f"mode: {mode}")
                        if avg_score:
                            meta_parts.append(f"score: {avg_score}")
                        if rewritten:
                            short_rewrite = (rewritten[:40] + "…") if len(rewritten) > 40 else rewritten
                            meta_parts.append(f'rewritten: "{short_rewrite}"')
                        if meta_parts:
                            st.caption("  ·  ".join(meta_parts))

    # ── Chat input — outside the scrollable container, always visible ─────────
    user_input = st.chat_input("Ask Citter anything…", key="crl_chat_input_box")

    if user_input and user_input.strip():
        user_query = user_input.strip()

        # Guard against duplicate submission on rerun
        if st.session_state.get("crl_last_submitted_query") == user_query:
            return
        st.session_state["crl_last_submitted_query"] = user_query

        logger.info(
            "💬 User message submitted — session_id=%s  query='%s…'",
            active_session_id, user_query[:60],
        )

        # Push user message immediately so it shows in the next render
        _append_message("user", user_query)

        # Only send non-empty filter values to the backend
        active_filters = {
            key: value
            for key, value in st.session_state.crl_filters.items()
            if value.strip()
        }

        with st.spinner("🤖 Citter is thinking…"):
            api_response = call_chat(
                session_id=active_session_id,
                message=user_query,
                filters=active_filters or None,
            )

        if api_response:
            bot_answer    = api_response.get("answer", "⚠️ No answer returned.")
            citations     = api_response.get("citations", [])
            pipeline_meta = {
                "mode":      api_response.get("mode", ""),
                "avg_score": api_response.get("avg_score", ""),
                "rewritten": api_response.get("rewritten", ""),
            }
            logger.info(
                "   ✅ Citter answered — mode=%s, avg_score=%s, citations=%d",
                pipeline_meta["mode"],
                pipeline_meta["avg_score"],
                len(citations),
            )
            _append_message(
                "assistant",
                bot_answer,
                citations=citations,
                pipeline_meta=pipeline_meta,
            )
        else:
            logger.error("   ❌ RAG backend returned None — is the API running on port 8001?")
            _append_message(
                "assistant",
                "⚠️ Could not reach the RAG backend. "
                "Is it running on port 8001?\n\n"
                "`uvicorn rag.api.main_rag:app --port 8001 --reload`",
            )

        # Update the session title from the first user message
        if len(all_messages) <= 2:
            new_title = (user_query[:30] + "…") if len(user_query) > 30 else user_query
            st.session_state.crl_sessions[active_session_id]["title"] = new_title
            logger.info("   📝 Session title updated to '%s'", new_title)

        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  Tab: Retrieval Inspector
# ─────────────────────────────────────────────────────────────────────────────

def _render_inspector_tab():
    """Retrieval inspector — shows raw chunks, similarity scores, and metadata."""
    st.subheader("🔍 Retrieval Inspector")
    st.caption(
        "Query the Milvus vector store directly and inspect retrieved chunks "
        "with cosine similarity scores."
    )

    inspector_query_col, inspector_topk_col, inspector_industry_col, inspector_doctype_col, inspector_search_col = st.columns([3, 1, 2, 2, 1])
    with inspector_query_col:
        inspector_query = st.text_input(
            "Search query",
            placeholder="Enter a search query…",
            key="crl_inspector_query",
        )
    with inspector_topk_col:
        inspector_top_k = st.number_input(
            "Top K",
            min_value=1,
            max_value=20,
            value=5,
            key="crl_inspector_top_k",
        )
    with inspector_industry_col:
        inspector_industry = st.text_input(
            "Industry filter",
            placeholder="Industry (optional)",
            key="crl_inspector_industry",
        )
    with inspector_doctype_col:
        inspector_doc_type = st.text_input(
            "Doc type filter",
            placeholder="Doc type (optional)",
            key="crl_inspector_doc_type",
        )
    with inspector_search_col:
        st.write("")   # vertical alignment spacer
        inspector_search_clicked = st.button(
            "Search",
            use_container_width=True,
            type="primary",
            key="crl_inspector_search_button",
        )

    if inspector_search_clicked and inspector_query.strip():
        logger.info(
            "🔍 Inspector search — query='%s…', top_k=%d, industry=%s, doc_type=%s",
            inspector_query[:60], inspector_top_k,
            inspector_industry or "(any)", inspector_doc_type or "(any)",
        )
        with st.spinner("Searching Milvus…"):
            retrieval_result = call_retrieval_debug(
                query=inspector_query.strip(),
                top_k=inspector_top_k,
                industry=inspector_industry,
                doc_type=inspector_doc_type,
            )
        if retrieval_result:
            retrieved_chunks = retrieval_result.get("chunks", [])
            average_score = (
                round(
                    sum(chunk.get("score", 0) for chunk in retrieved_chunks) / len(retrieved_chunks),
                    4,
                )
                if retrieved_chunks else 0
            )
            logger.info(
                "   ✅ Inspector — %d chunks returned, avg_score=%.4f",
                len(retrieved_chunks), average_score,
            )
            st.success(f"**{len(retrieved_chunks)} chunks** returned — avg score: `{average_score}`")

            for chunk_idx, chunk in enumerate(retrieved_chunks, start=1):
                chunk_score = chunk.get("score", 0)
                with st.expander(
                    f"[{chunk_idx}] {chunk.get('title', '?')} → {chunk.get('section', '?')}  "
                    f"(score: {chunk_score})",
                    expanded=(chunk_idx == 1),
                ):
                    chunk_text_col, chunk_meta_col = st.columns([2, 1])
                    with chunk_text_col:
                        st.code(chunk.get("chunk_text", ""), language=None)
                    with chunk_meta_col:
                        st.metric("Score", chunk_score)
                        st.text(f"Doc ID:   {chunk.get('doc_id', '')}")
                        st.text(f"Industry: {chunk.get('industry', '')}")
                        st.text(f"Type:     {chunk.get('doc_type', '')}")
                        st.text(f"Version:  {chunk.get('version', '')}")
                        st.text(f"Blocks:   {chunk.get('block_range', '')}")
        else:
            logger.error("   ❌ Retrieval debug call failed")
            st.error("❌ Retrieval debug call failed — is the API running on port 8001?")


# ─────────────────────────────────────────────────────────────────────────────
#  Tab: Ingest
# ─────────────────────────────────────────────────────────────────────────────

def _render_ingest_tab():
    """Notion ingestion panel."""
    st.subheader("📥 Notion Ingestion")
    st.caption(
        "Ingest Notion pages into the Milvus vector store. "
        "Milvus Lite persists data locally — no external service needed."
    )

    with st.expander("📄 Ingest a single page", expanded=False):
        single_page_col_left, single_page_col_right = st.columns(2)
        with single_page_col_left:
            single_page_id    = st.text_input("Page ID",  placeholder="Notion page UUID", key="crl_single_page_id")
            single_page_title = st.text_input("Title",    placeholder="Page title",       key="crl_single_page_title")
        with single_page_col_right:
            single_page_industry = st.text_input("Industry", placeholder="General",  key="crl_single_page_industry")
            single_page_doc_type = st.text_input("Doc type", placeholder="Document", key="crl_single_page_doc_type")
            single_page_version  = st.text_input("Version",  placeholder="1.0",      key="crl_single_page_version")

        if st.button("▶ Ingest this page", key="crl_ingest_single_page_button", type="primary"):
            if not single_page_id.strip():
                st.warning("⚠️ Page ID is required.")
            else:
                logger.info(
                    "📥 Single-page ingest triggered — page_id=%s  title=%r",
                    single_page_id, single_page_title,
                )
                with st.spinner("Ingesting page…"):
                    ingest_result = call_ingest_notion(
                        page_id=single_page_id.strip(),
                        title=single_page_title.strip(),
                        industry=single_page_industry or "General",
                        doc_type=single_page_doc_type or "Document",
                        version=single_page_version or "1.0",
                    )
                if ingest_result and ingest_result.get("status") == "ok":
                    chunks_inserted = ingest_result.get("chunks_inserted", 0)
                    logger.info("   ✅ Single-page ingest complete — %d chunks inserted", chunks_inserted)
                    st.success(f"✅ Inserted {chunks_inserted} chunks.")
                else:
                    logger.error("   ❌ Single-page ingest failed")
                    st.error("❌ Ingestion failed — check API logs.")

    st.divider()
    st.markdown("**Full re-ingest** — all pages under `NOTION_ROOT_PAGE_ID`")
    st.warning("⚠️ This may take several minutes depending on the number of pages.", icon="⏱️")

    if st.button(
        "🚀 Ingest All Pages",
        key="crl_ingest_all_pages_button",
        disabled=st.session_state.crl_ingest_running,
    ):
        logger.info("📥 Full Notion ingest triggered")
        st.session_state.crl_ingest_running = True
        with st.spinner("Ingesting all pages — this may take a while…"):
            full_ingest_result = call_ingest_notion()
        st.session_state.crl_ingest_running = False

        if full_ingest_result and full_ingest_result.get("status") == "ok":
            pages_processed = full_ingest_result.get("pages_processed", 0)
            chunks_inserted = full_ingest_result.get("chunks_inserted", 0)
            ingest_errors   = full_ingest_result.get("errors", [])
            logger.info(
                "   ✅ Full ingest complete — pages=%d, chunks=%d, errors=%d",
                pages_processed, chunks_inserted, len(ingest_errors),
            )
            st.success(
                f"✅ Done — {pages_processed} pages processed, "
                f"{chunks_inserted} chunks inserted. "
                f"Errors: {len(ingest_errors)}"
            )
            if ingest_errors:
                with st.expander("Show errors"):
                    for error_message in ingest_errors:
                        st.text(error_message)
        else:
            logger.error("   ❌ Full Notion ingest failed")
            st.error("❌ Ingestion failed — check API logs.")


# ─────────────────────────────────────────────────────────────────────────────
#  Tab: Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _render_evaluation_tab():
    """RAGAS evaluation dashboard."""
    st.subheader("📊 RAGAS Evaluation")
    st.caption(
        "Evaluate RAG quality across faithfulness, answer relevancy, "
        "context precision, and context recall."
    )

    st.markdown(
        "Enter your evaluation dataset below — one row per line, "
        "**question TAB ground_truth**:"
    )
    st.code("What is the incident response SLA?\t< 1 hour for P1 incidents", language="text")

    evaluation_dataset_raw = st.text_area(
        "Evaluation dataset (question \\t ground_truth per line)",
        height=140,
        key="crl_evaluation_dataset",
        placeholder="What is the security incident response time?\t< 1 hour for P1",
    )

    if st.button(
        "▶ Run RAGAS Evaluation",
        key="crl_run_evaluation_button",
        disabled=st.session_state.crl_eval_running,
        type="primary",
    ):
        if not evaluation_dataset_raw.strip():
            st.warning("Please enter at least one question → ground_truth pair.")
            return

        parsed_rows = [
            line.split("\t", 1)
            for line in evaluation_dataset_raw.strip().splitlines()
            if "\t" in line
        ]
        if not parsed_rows:
            st.warning(
                "Could not parse dataset — use a TAB character to separate "
                "question and ground truth on each line."
            )
            return

        questions     = [row[0].strip() for row in parsed_rows]
        ground_truths = [row[1].strip() for row in parsed_rows]

        logger.info("📊 RAGAS evaluation started — %d question(s)", len(questions))

        st.session_state.crl_eval_running = True
        generated_answers:  list[str]       = []
        retrieved_contexts: list[list[str]] = []

        progress_bar = st.progress(0, text="Generating answers…")
        for question_idx, question_text in enumerate(questions):
            logger.info(
                "   [%d/%d] Generating answer for: '%s…'",
                question_idx + 1, len(questions), question_text[:40],
            )
            chat_response = call_chat(
                session_id=f"eval_{question_idx}",
                message=question_text,
            )
            if chat_response:
                generated_answers.append(chat_response.get("answer", ""))
                retrieved_contexts.append([
                    citation.get("chunk_text", "")
                    for citation in chat_response.get("citations", [])
                ])
            else:
                generated_answers.append("")
                retrieved_contexts.append([])
            progress_bar.progress(
                (question_idx + 1) / len(questions),
                text=f"Processing {question_idx + 1}/{len(questions)}…",
            )

        with st.spinner("Running RAGAS metrics…"):
            evaluation_result = call_run_evaluation(
                questions=questions,
                answers=generated_answers,
                contexts=retrieved_contexts,
                ground_truths=ground_truths,
            )

        st.session_state.crl_eval_running = False

        if evaluation_result and evaluation_result.get("status") == "ok":
            scores = evaluation_result["scores"]
            logger.info("   ✅ Evaluation complete — scores=%s", scores)
            st.success("✅ Evaluation complete!")
            metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
            metric_col_1.metric("Faithfulness",      scores.get("faithfulness",      "—"))
            metric_col_2.metric("Answer Relevancy",  scores.get("answer_relevancy",  "—"))
            metric_col_3.metric("Context Precision", scores.get("context_precision", "—"))
            metric_col_4.metric("Context Recall",    scores.get("context_recall",    "—"))
        else:
            logger.error("   ❌ Evaluation failed: %s", evaluation_result)
            st.error(f"❌ Evaluation failed: {evaluation_result}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def render_cite_rag_lab_ui():
    """Called by streamlit_uidemo.py to render the full CiteRagLab UI."""
    _init_crl_session_state()
    _render_sidebar()

    tab_chat, tab_inspector, tab_ingest, tab_evaluation = st.tabs([
        "💬 Chat", "🔍 Inspector", "📥 Ingest", "📊 Evaluation",
    ])

    with tab_chat:
        _render_chat_tab()

    with tab_inspector:
        _render_inspector_tab()

    with tab_ingest:
        _render_ingest_tab()

    with tab_evaluation:
        _render_evaluation_tab()