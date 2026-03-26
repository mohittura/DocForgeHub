"""
ui/cite_rag_lab_ui_rag.py

CiteRagLab — RAG chat interface + StateCase ticket management.

Built entirely with native Streamlit components.
No custom CSS. No unsafe_allow_html anywhere.

UI layout:
  LEFT SIDEBAR
    ├── 🤖 CiteRagLab header
    ├── [＋ New Chat] button
    ├── [Search Chat] text input
    ├── "Your Chats" — scrollable session list
    └── Session ID caption footer

  MAIN AREA (tabs)
    💬 Chat        — st.chat_message bubbles + st.chat_input
    🎫 Tickets     — StateCase ticket board (list + create + update)
    🔍 Inspector   — retrieval debug panel (chunks + scores)
    📥 Ingest      — Notion ingestion trigger
    📊 Evaluation  — RAGAS evaluation dashboard
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
from api_helpers_statecase_rag import (
    call_statecase_chat,
    call_create_ticket,
    call_list_tickets,
    call_update_ticket,
)

logger = logging.getLogger("ui.cite_rag_lab_ui_rag")

# Leading substrings that identify a refusal / out-of-scope response
_REFUSAL_PREFIXES = (
    "I wasn't able to find relevant documents in the library for that question.",
    "I can only answer questions based on the documents in this library.",
    "The available documents do not contain sufficient information",
    "I can only compare documents that are present in this library",
    "I can only summarise documents that are present in this library",
)

PRIORITY_OPTIONS = ["Low", "Medium", "High", "Critical"]
STATUS_OPTIONS   = ["Not started", "In progress", "Done"]


# ─────────────────────────────────────────────────────────────────────────────
#  Session state bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def _init_crl_session_state():
    """Initialise all CiteRagLab session-state keys once per browser session."""
    if "crl_sessions" not in st.session_state:
        st.session_state.crl_sessions = {}

    if "crl_active_session_id" not in st.session_state:
        st.session_state.crl_active_session_id = None

    if not st.session_state.crl_sessions:
        default_id = str(uuid.uuid4())[:8]
        st.session_state.crl_sessions[default_id] = {
            "title":    "Chat 1",
            "messages": [],
        }
        st.session_state.crl_active_session_id = default_id
        logger.info("🆕 Default session created — id=%s", default_id)

    if "crl_search_term" not in st.session_state:
        st.session_state.crl_search_term = ""

    if "crl_filters" not in st.session_state:
        st.session_state.crl_filters = {"industry": "", "version": ""}

    if "crl_filter_version_counter" not in st.session_state:
        st.session_state.crl_filter_version_counter = 0

    if "crl_ingest_running" not in st.session_state:
        st.session_state.crl_ingest_running = False

    if "crl_eval_running" not in st.session_state:
        st.session_state.crl_eval_running = False

    # StateCase state keys
    if "sc_ticket_list" not in st.session_state:
        st.session_state.sc_ticket_list = []

    if "sc_ticket_list_loaded" not in st.session_state:
        st.session_state.sc_ticket_list_loaded = False

    if "sc_status_filter" not in st.session_state:
        st.session_state.sc_status_filter = "All"

    # Toggle: whether the chat tab uses the StateCase agent (with auto-ticketing)
    # vs the bare RAG pipeline (original behaviour).
    if "sc_agent_mode" not in st.session_state:
        st.session_state.sc_agent_mode = True


# ─────────────────────────────────────────────────────────────────────────────
#  Session helpers
# ─────────────────────────────────────────────────────────────────────────────

def _create_new_session():
    session_id    = str(uuid.uuid4())[:8]
    chat_number   = len(st.session_state.crl_sessions) + 1
    session_title = f"Chat {chat_number}"
    st.session_state.crl_sessions[session_id] = {
        "title":    session_title,
        "messages": [],
    }
    st.session_state.crl_active_session_id = session_id
    logger.info("🆕 New session — id=%s  title='%s'", session_id, session_title)
    return session_id


def _get_active_messages():
    active_id = st.session_state.crl_active_session_id
    if active_id and active_id in st.session_state.crl_sessions:
        return st.session_state.crl_sessions[active_id]["messages"]
    return []


def _append_message(role, content, citations=None, pipeline_meta=None, ticket_created=None):
    messages = _get_active_messages()
    messages.append({
        "role":           role,
        "content":        content,
        "citations":      citations or [],
        "pipeline_meta":  pipeline_meta or {},
        "ticket_created": ticket_created,
    })


def _get_filtered_sessions():
    search_term = st.session_state.crl_search_term.lower().strip()
    return [
        (sid, sdata)
        for sid, sdata in st.session_state.crl_sessions.items()
        if not search_term
        or search_term in sdata["title"].lower()
        or any(search_term in msg["content"].lower() for msg in sdata["messages"])
    ]


# ─────────────────────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def _render_sidebar():
    with st.sidebar:
        st.header("🤖 CiteRagLab")
        st.divider()

        if st.button("＋ New Chat", use_container_width=True, key="crl_new_chat_button"):
            _create_new_session()
            st.rerun()

        st.text_input("Search Chat", placeholder="Search chats…", key="crl_search_term")
        st.caption("YOUR CHATS")

        filtered_sessions = _get_filtered_sessions()
        if not filtered_sessions:
            st.caption("No chats yet.")
        else:
            chat_list_container = st.container(height=240)
            with chat_list_container:
                for session_id, session_data in filtered_sessions:
                    is_active = (session_id == st.session_state.crl_active_session_id)
                    label     = f"{'▶ ' if is_active else ''}{session_data['title']} ({len(session_data['messages'])})"
                    if st.button(
                        label,
                        key=f"crl_select_session_{session_id}",
                        use_container_width=True,
                        type="primary" if is_active else "secondary",
                    ):
                        st.session_state.crl_active_session_id = session_id
                        st.rerun()

        st.divider()

        # ── StateCase agent toggle ─────────────────────────────────────────────
        st.caption("STATECASE SETTINGS")
        st.session_state.sc_agent_mode = st.toggle(
            "🎫 Auto-ticketing",
            value=st.session_state.sc_agent_mode,
            key="sc_agent_mode_toggle",
            help="When enabled, unanswerable questions automatically create a StateCase ticket.",
        )

        st.divider()
        displayed_session_id = st.session_state.crl_active_session_id or "—"
        st.caption(f"Session ID: {displayed_session_id}")

        if st.session_state.crl_active_session_id:
            if st.button("🗑️ Delete Session", use_container_width=True, key="crl_delete_session_button"):
                sid = st.session_state.crl_active_session_id
                call_delete_session(sid)
                del st.session_state.crl_sessions[sid]
                st.session_state.crl_active_session_id = None
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  Tab: Chat
# ─────────────────────────────────────────────────────────────────────────────

def _render_chat_tab():
    active_session_id = st.session_state.crl_active_session_id

    if active_session_id is None or active_session_id not in st.session_state.crl_sessions:
        st.session_state.crl_active_session_id = None
        st.info("All sessions deleted — click **＋ New Chat** in the sidebar to start.")
        return

    session_title = st.session_state.crl_sessions[active_session_id]["title"]
    agent_mode    = st.session_state.sc_agent_mode

    col_title, col_badge = st.columns([5, 1])
    with col_title:
        st.subheader(f"CiteRagLab — {session_title}")
    with col_badge:
        if agent_mode:
            st.success("🎫 Auto-ticket ON", icon=None)

    # ── Filters ───────────────────────────────────────────────────────────────
    _fv = st.session_state.crl_filter_version_counter
    fc1, fc2, fc3 = st.columns([3, 2, 1])
    with fc1:
        st.session_state.crl_filters["industry"] = st.text_input(
            "Industry", placeholder="Industry (e.g. Cybersecurity)",
            key=f"crl_filter_industry_{_fv}",
        )
    with fc2:
        st.session_state.crl_filters["version"] = st.text_input(
            "Version", placeholder="Version (e.g. 1.0)",
            key=f"crl_filter_version_{_fv}",
        )
    with fc3:
        st.write("")
        if st.button("✕ Clear", use_container_width=True, key="crl_clear_filters_button"):
            st.session_state.crl_filters = {"industry": "", "version": ""}
            st.session_state.crl_filter_version_counter += 1
            st.rerun()

    st.divider()

    # ── Message history ────────────────────────────────────────────────────────
    all_messages      = _get_active_messages()
    messages_container = st.container(height=430, border=False)

    with messages_container:
        if not all_messages:
            st.info(
                "🤖 Citter is ready — ask anything about your document library.\n\n"
                + ("💡 *Auto-ticketing is ON — unanswerable questions will automatically create a StateCase ticket.*"
                   if agent_mode else "")
            )
        else:
            for msg in all_messages:
                role          = msg["role"]
                content       = msg["content"]
                citations     = msg.get("citations", [])
                pipeline_meta = msg.get("pipeline_meta", {})
                ticket_info   = msg.get("ticket_created")

                avatar = "🤖" if role == "assistant" else "👤"
                with st.chat_message(role, avatar=avatar):
                    st.markdown(content)

                    is_refusal = (
                        role == "assistant" and (
                            content.strip().startswith(_REFUSAL_PREFIXES)
                            or pipeline_meta.get("mode") in ("GREETING", "TICKET", "CLARIFY")
                        )
                    )

                    # Show ticket badge if a ticket was created this turn
                    if ticket_info and role == "assistant":
                        t_id  = ticket_info.get("ticket_id", "")
                        t_pri = ticket_info.get("priority", "")
                        t_url = ticket_info.get("url", "")
                        badge_text = f"🎫 {t_id} · Priority: {t_pri} · Status: {ticket_info.get('status', 'Not started')}"
                        if t_url:
                            badge_text += f" · [View in Notion]({t_url})"
                        st.info(badge_text)

                    # Citations
                    if citations and role == "assistant" and not is_refusal:
                        with st.expander(f"📚 {len(citations)} source(s)", expanded=False):
                            for c in citations:
                                loc = f"{c.get('title','')} → {c.get('section','')}" if c.get("section") else c.get("title","")
                                st.markdown(f"**[{c.get('index','')}]** {loc}  `{c.get('doc_type','')}`  score: `{c.get('score',0)}`")

                    # Pipeline metadata
                    if pipeline_meta and role == "assistant" and not is_refusal:
                        mode      = pipeline_meta.get("mode", "")
                        avg_score = pipeline_meta.get("avg_score", "")
                        rewritten = pipeline_meta.get("rewritten", "")
                        parts     = []
                        if mode:      parts.append(f"mode: {mode}")
                        if avg_score: parts.append(f"score: {avg_score}")
                        if rewritten: parts.append(f'rewritten: "{rewritten}"')
                        if parts:
                            st.caption("  ·  ".join(parts))

    # ── Chat input ────────────────────────────────────────────────────────────
    user_input = st.chat_input("Ask Citter anything…", key="crl_chat_input_box")

    if user_input and user_input.strip():
        user_query  = user_input.strip()
        _submit_key = (active_session_id, len(_get_active_messages()), user_query)
        if st.session_state.get("crl_last_submit_key") == _submit_key:
            return
        st.session_state["crl_last_submit_key"] = _submit_key

        _append_message("user", user_query)

        active_filters = {
            k: v.strip()
            for k, v in st.session_state.crl_filters.items()
            if v.strip()
        }

        with st.spinner("🤖 Citter is thinking…"):
            if agent_mode:
                # ── StateCase agent (auto-ticketing + memory) ─────────────────
                api_response = call_statecase_chat(
                    session_id=active_session_id,
                    message=user_query,
                    filters=active_filters or None,
                )
                if api_response:
                    bot_answer     = api_response.get("answer", "⚠️ No answer returned.")
                    citations      = api_response.get("citations", [])
                    ticket_created = api_response.get("ticket_created")
                    pipeline_meta  = {
                        **api_response.get("pipeline_meta", {}),
                        "trace_id": api_response.get("trace_id", ""),
                    }
                    _append_message(
                        "assistant", bot_answer,
                        citations=citations,
                        pipeline_meta=pipeline_meta,
                        ticket_created=ticket_created,
                    )
                    # Refresh ticket list if a new ticket was created
                    if ticket_created:
                        st.session_state.sc_ticket_list_loaded = False
                else:
                    _append_message(
                        "assistant",
                        "⚠️ Could not reach the RAG backend. Is it running on port 8001?\n\n"
                        "`uvicorn rag.api.main_rag:app --port 8001 --reload`",
                    )
            else:
                # ── Original bare RAG (no auto-ticketing) ─────────────────────
                api_response = call_chat(
                    session_id=active_session_id,
                    message=user_query,
                    filters=active_filters or None,
                )
                if api_response:
                    _prior = sum(1 for m in _get_active_messages() if m["role"] == "assistant")
                    _append_message(
                        "assistant",
                        api_response.get("answer", "⚠️ No answer returned."),
                        citations=api_response.get("citations", []),
                        pipeline_meta={
                            "mode":      api_response.get("mode", ""),
                            "avg_score": api_response.get("avg_score", ""),
                            "rewritten": api_response.get("rewritten", ""),
                            "turn":      _prior + 1,
                        },
                    )
                else:
                    _append_message(
                        "assistant",
                        "⚠️ Could not reach the RAG backend. Is it running on port 8001?\n\n"
                        "`uvicorn rag.api.main_rag:app --port 8001 --reload`",
                    )

        # Update session title from first message
        if len(all_messages) <= 2:
            new_title = (user_query[:30] + "…") if len(user_query) > 30 else user_query
            st.session_state.crl_sessions[active_session_id]["title"] = new_title

        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  Tab: StateCase Tickets
# ─────────────────────────────────────────────────────────────────────────────

def _render_tickets_tab():
    """StateCase ticket board — list, create, and update tickets."""
    st.subheader("🎫 StateCase Tickets")
    st.caption("View and manage support tickets raised from the chat or created manually.")

    # ── Sub-tabs: Board | Create ───────────────────────────────────────────────
    board_tab, create_tab = st.tabs(["📋 Ticket Board", "＋ Create Ticket"])

    # ─────────── BOARD ────────────────────────────────────────────────────────
    with board_tab:
        # Filter + Refresh controls
        fc1, fc2, fc3 = st.columns([2, 1, 1])
        with fc1:
            status_choice = st.selectbox(
                "Filter by Status",
                ["All"] + STATUS_OPTIONS,
                key="sc_status_filter_select",
            )
        with fc2:
            st.write("")
            st.write("")
            refresh_clicked = st.button(
                "🔄 Refresh",
                use_container_width=True,
                key="sc_refresh_tickets_button",
            )
        with fc3:
            st.write("")
            st.write("")
            if st.button("🗑️ Clear Cache", use_container_width=True, key="sc_clear_cache_button"):
                st.session_state.sc_ticket_list_loaded = False
                st.session_state.sc_ticket_list = []

        # Load tickets on first render or after refresh / ticket creation
        if refresh_clicked or not st.session_state.sc_ticket_list_loaded:
            with st.spinner("Loading tickets from Notion…"):
                result = call_list_tickets(
                    status_filter=None if status_choice == "All" else status_choice,
                    limit=100,
                )
            if result:
                st.session_state.sc_ticket_list        = result.get("tickets", [])
                st.session_state.sc_ticket_list_loaded = True
                st.session_state.sc_status_filter      = status_choice
            else:
                st.error("❌ Could not load tickets — is the API running on port 8001?")

        tickets = st.session_state.sc_ticket_list

        # Re-filter in-memory if filter changed without refresh
        if status_choice != "All":
            tickets = [t for t in tickets if t.get("status") == status_choice]

        if not tickets:
            st.info("No tickets found. Create one below or ask Citter an unanswerable question with auto-ticketing ON.")
        else:
            # ── Summary metrics ────────────────────────────────────────────────
            all_loaded = st.session_state.sc_ticket_list
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total",       len(all_loaded))
            m2.metric("Not started", sum(1 for t in all_loaded if t.get("status") == "Not started"))
            m3.metric("In progress", sum(1 for t in all_loaded if t.get("status") == "In progress"))
            m4.metric("Done",        sum(1 for t in all_loaded if t.get("status") == "Done"))

            st.divider()

            # ── Ticket cards ───────────────────────────────────────────────────
            for ticket in tickets:
                tid       = ticket.get("ticket_id",      "—")
                desc      = ticket.get("description",    "—")
                question  = ticket.get("question",       "")
                priority  = ticket.get("priority",       "Medium")
                status    = ticket.get("status",         "Not started")
                owner     = ticket.get("assigned_owner", "Unassigned")
                sources   = ticket.get("attempted_sources", "")
                page_id   = ticket.get("notion_page_id", "")
                t_url     = ticket.get("url", "")
                created   = ticket.get("created_time",  "")[:10]

                # Priority colour badge
                priority_icon = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Critical": "🔴"}.get(priority, "⚪")
                status_icon   = {"Not started": "📬", "In progress": "🔄", "Done": "✅"}.get(status, "📬")

                with st.expander(
                    f"{status_icon} {tid} — {desc[:70]}{'…' if len(desc)>70 else ''}  "
                    f"{priority_icon} {priority}",
                    expanded=False,
                ):
                    col_info, col_actions = st.columns([3, 2])

                    with col_info:
                        st.markdown(f"**Question:** {question}")
                        if sources and sources != "None":
                            st.markdown(f"**Attempted Sources:** {sources}")
                        st.caption(f"Assigned: {owner}  ·  Created: {created}")
                        if t_url:
                            st.markdown(f"[🔗 Open in Notion]({t_url})")

                    with col_actions:
                        st.markdown("**Update Ticket**")
                        new_status = st.selectbox(
                            "Status",
                            STATUS_OPTIONS,
                            index=STATUS_OPTIONS.index(status) if status in STATUS_OPTIONS else 0,
                            key=f"sc_status_{page_id}",
                        )
                        new_owner = st.text_input(
                            "Assigned Owner",
                            value=owner,
                            key=f"sc_owner_{page_id}",
                        )
                        new_priority = st.selectbox(
                            "Priority",
                            PRIORITY_OPTIONS,
                            index=PRIORITY_OPTIONS.index(priority) if priority in PRIORITY_OPTIONS else 1,
                            key=f"sc_priority_{page_id}",
                        )
                        if st.button(
                            "💾 Save Changes",
                            key=f"sc_save_{page_id}",
                            use_container_width=True,
                            type="primary",
                        ):
                            with st.spinner("Updating…"):
                                update_result = call_update_ticket(
                                    notion_page_id=page_id,
                                    status=new_status,
                                    assigned_owner=new_owner,
                                    priority=new_priority,
                                )
                            if update_result:
                                st.success(f"✅ {tid} updated — {new_status}")
                                # Refresh ticket list
                                st.session_state.sc_ticket_list_loaded = False
                                st.rerun()
                            else:
                                st.error("❌ Update failed — check API logs.")

    # ─────────── CREATE ───────────────────────────────────────────────────────
    with create_tab:
        st.markdown("### Create a ticket manually")
        st.caption(
            "Use this form to log questions or issues that the assistant couldn't resolve. "
            "Tickets are saved directly to the StateCase Notion database."
        )

        active_sid = st.session_state.crl_active_session_id or "manual"

        with st.form("sc_create_ticket_form", clear_on_submit=True):
            ct_question = st.text_area(
                "Question / Issue *",
                placeholder="What's the vendor escalation process for Tier-2 clients?",
                height=100,
                key="sc_ct_question",
            )
            ct_description = st.text_input(
                "Short description (optional)",
                placeholder="Leave blank to auto-derive from question",
                key="sc_ct_description",
            )
            ct_col1, ct_col2 = st.columns(2)
            with ct_col1:
                ct_priority = st.selectbox(
                    "Priority", PRIORITY_OPTIONS, index=1, key="sc_ct_priority"
                )
            with ct_col2:
                ct_owner = st.text_input(
                    "Assign to", value="Unassigned", key="sc_ct_owner"
                )
            ct_sources = st.text_input(
                "Attempted sources (comma-separated, optional)",
                placeholder="HR Policy v1.0, Vendor Handbook",
                key="sc_ct_sources",
            )
            ct_user_info = st.text_input(
                "User info (optional)",
                placeholder="Name, team, or any context",
                key="sc_ct_user_info",
            )

            submitted = st.form_submit_button("🎫 Create Ticket", type="primary", use_container_width=True)

        if submitted:
            if not ct_question.strip():
                st.warning("⚠️ Question is required.")
            else:
                attempted = [s.strip() for s in ct_sources.split(",") if s.strip()] if ct_sources else []
                with st.spinner("Creating ticket in Notion…"):
                    result = call_create_ticket(
                        question=ct_question.strip(),
                        session_id=active_sid,
                        description=ct_description.strip(),
                        priority=ct_priority,
                        assigned_owner=ct_owner.strip() or "Unassigned",
                        attempted_sources=attempted or None,
                        user_info=ct_user_info.strip(),
                    )
                if result and result.get("status") == "ok":
                    t = result["ticket"]
                    t_url = t.get("url", "")
                    st.success(
                        f"✅ Ticket **{t['ticket_id']}** created!"
                        + (f"  [View in Notion]({t_url})" if t_url else "")
                    )
                    # Mark ticket list stale so it refreshes next time
                    st.session_state.sc_ticket_list_loaded = False
                else:
                    st.error("❌ Ticket creation failed — check API logs.")


# ─────────────────────────────────────────────────────────────────────────────
#  Tab: Retrieval Inspector
# ─────────────────────────────────────────────────────────────────────────────

def _render_inspector_tab():
    st.subheader("🔍 Retrieval Inspector")
    st.caption("Query the Milvus vector store directly and inspect retrieved chunks with cosine similarity scores.")

    ic1, ic2, ic3, ic4 = st.columns([3, 1, 2, 1])
    with ic1:
        inspector_query = st.text_input("Search query", placeholder="Enter a search query…", key="crl_inspector_query")
    with ic2:
        inspector_top_k = st.number_input("Top K", min_value=1, max_value=20, value=5, key="crl_inspector_top_k")
    with ic3:
        inspector_industry = st.text_input("Industry filter", placeholder="Industry (optional)", key="crl_inspector_industry")
    with ic4:
        st.write("")
        search_clicked = st.button("Search", use_container_width=True, type="primary", key="crl_inspector_search_button")

    if search_clicked and inspector_query.strip():
        with st.spinner("Searching Milvus…"):
            retrieval_result = call_retrieval_debug(
                query=inspector_query.strip(),
                top_k=inspector_top_k,
                industry=inspector_industry,
            )
        if retrieval_result:
            retrieved_chunks = retrieval_result.get("chunks", [])
            avg_score = (
                round(sum(c.get("score", 0) for c in retrieved_chunks) / len(retrieved_chunks), 4)
                if retrieved_chunks else 0
            )
            st.success(f"**{len(retrieved_chunks)} chunks** returned — avg score: `{avg_score}`")
            for idx, chunk in enumerate(retrieved_chunks, 1):
                with st.expander(
                    f"[{idx}] {chunk.get('title','?')} → {chunk.get('section','?')}  (score: {chunk.get('score',0)})",
                    expanded=(idx == 1),
                ):
                    cc1, cc2 = st.columns([2, 1])
                    with cc1:
                        st.code(chunk.get("chunk_text", ""), language=None)
                    with cc2:
                        st.metric("Score", chunk.get("score", 0))
                        st.text(f"Doc ID:   {chunk.get('doc_id','')}")
                        st.text(f"Industry: {chunk.get('industry','')}")
                        st.text(f"Type:     {chunk.get('doc_type','')}")
                        st.text(f"Version:  {chunk.get('version','')}")
                        st.text(f"Blocks:   {chunk.get('block_range','')}")
        else:
            st.error("❌ Retrieval debug call failed — is the API running on port 8001?")


# ─────────────────────────────────────────────────────────────────────────────
#  Tab: Ingest
# ─────────────────────────────────────────────────────────────────────────────

def _render_ingest_tab():
    st.subheader("📥 Notion Ingestion")
    st.caption("Ingest Notion pages into the Milvus vector store.")
    st.markdown("**Full re-ingest** — all pages under `NOTION_ROOT_PAGE_ID`")
    st.warning("⚠️ This may take several minutes depending on the number of pages.", icon="⏱️")

    if st.button(
        "🚀 Ingest All Pages",
        key="crl_ingest_all_pages_button",
        disabled=st.session_state.crl_ingest_running,
    ):
        st.session_state.crl_ingest_running = True
        with st.spinner("Ingesting all pages…"):
            result = call_ingest_notion()
        st.session_state.crl_ingest_running = False

        if result and result.get("status") == "ok":
            st.success(
                f"✅ Done — {result.get('pages_processed',0)} pages, "
                f"{result.get('chunks_inserted',0)} chunks. "
                f"Errors: {len(result.get('errors',[]))}"
            )
            if result.get("errors"):
                with st.expander("Show errors"):
                    for e in result["errors"]:
                        st.text(e)
        else:
            st.error("❌ Ingestion failed — check API logs.")


# ─────────────────────────────────────────────────────────────────────────────
#  Tab: Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _render_evaluation_tab():
    st.subheader("📊 RAGAS Evaluation")
    st.caption("Evaluate RAG quality across faithfulness, answer relevancy, context precision, and context recall.")
    st.markdown("Enter your evaluation dataset below — one row per line, **question TAB ground_truth**:")
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
            st.warning("Could not parse dataset — use a TAB character to separate question and ground truth.")
            return

        questions     = [r[0].strip() for r in parsed_rows]
        ground_truths = [r[1].strip() for r in parsed_rows]

        st.session_state.crl_eval_running = True
        generated_answers:  list[str]       = []
        retrieved_contexts: list[list[str]] = []

        progress_bar = st.progress(0, text="Generating answers…")
        for i, q in enumerate(questions):
            resp = call_chat(session_id=f"eval_{i}", message=q)
            if resp:
                generated_answers.append(resp.get("answer", ""))
                retrieved_contexts.append([
                    c.get("chunk_text", "")
                    for c in resp.get("citations", [])
                    if c.get("chunk_text", "").strip()
                ])
            else:
                generated_answers.append("")
                retrieved_contexts.append([])
            progress_bar.progress((i + 1) / len(questions), text=f"Processing {i+1}/{len(questions)}…")

        with st.spinner("Running RAGAS metrics…"):
            eval_result = call_run_evaluation(
                questions=questions,
                answers=generated_answers,
                contexts=retrieved_contexts,
                ground_truths=ground_truths,
            )

        st.session_state.crl_eval_running = False

        if eval_result and eval_result.get("status") == "ok":
            scores = eval_result["scores"]
            st.success("✅ Evaluation complete!")
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Faithfulness",      scores.get("faithfulness",      "—"))
            mc2.metric("Answer Relevancy",  scores.get("answer_relevancy",  "—"))
            mc3.metric("Context Precision", scores.get("context_precision", "—"))
            mc4.metric("Context Recall",    scores.get("context_recall",    "—"))
        else:
            st.error(f"❌ Evaluation failed: {eval_result}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def render_cite_rag_lab_ui():
    """Called by streamlit_uidemo.py to render the full CiteRagLab UI."""
    _init_crl_session_state()
    _render_sidebar()

    tab_chat, tab_tickets, tab_inspector, tab_ingest, tab_evaluation = st.tabs([
        "💬 Chat", "🎫 Tickets", "🔍 Inspector", "📥 Ingest", "📊 Evaluation",
    ])

    with tab_chat:
        _render_chat_tab()

    with tab_tickets:
        _render_tickets_tab()

    with tab_inspector:
        _render_inspector_tab()

    with tab_ingest:
        _render_ingest_tab()

    with tab_evaluation:
        _render_evaluation_tab()