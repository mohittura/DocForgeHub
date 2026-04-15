"""
rag/pipeline/statecase_notion_rag.py

StateCase — Notion ticket client for CiteRagLab.

Wraps all Notion API calls for the StateCase Tickets database:
    https://www.notion.so/32d89db15e5b8051a212f5f983a90a0f

Database schema (from the image):
    Ticket ID        — title field (auto-generated e.g. "SC-0001")
    Description      — rich_text  — one-line summary of the question
    Question         — rich_text  — full user question
    Assigned Owner   — rich_text  — who should handle this
    Priority         — select     — Low / Medium / High / Critical
    User Info        — rich_text  — session_id + any user context
    Status           — select     — Open / In Progress / Resolved / Closed
    Attempted Sources— rich_text  — doc titles the RAG tried but couldn't answer from

Public API
──────────
    create_ticket(...)   — create a new row, return ticket dict
    update_ticket(...)   — update status / assigned owner / other fields
    get_ticket(...)      — fetch one ticket by Notion page_id
    list_tickets(...)    — query all tickets (with optional status filter)
    get_next_ticket_id() — atomic counter stored in Redis; falls back to timestamp

Idempotency
───────────
    Each ticket carries a dedup_key (SHA-256 of session_id + question).
    Before creating, we search Notion for a matching dedup_key stored in
    User Info.  If found we return the existing ticket instead of creating
    a duplicate.  This survives retries and double-clicks.

Rate limiting
─────────────
    All Notion calls go through _notion_call() copied from notion_loader_rag.py
    (0.35 s delay, 429 back-off, max 6 retries).
"""

import os
import time
import hashlib
import logging
from typing import Optional
from datetime import datetime, timezone
from notion_client import Client
from notion_client.errors import APIResponseError
from dotenv import load_dotenv
from nltk.corpus import stopwords

load_dotenv()

logger = logging.getLogger("rag.pipeline.statecase_notion_rag")

# ── Config ────────────────────────────────────────────────────────────────────
STATECASE_DB_ID = os.getenv(
    "STATECASE_DB_ID",
    "32d89db1-5e5b-8051-a212-f5f983a90a0f",   # from the Notion URL
)

REQUEST_DELAY_SEC = 0.35
MAX_RETRIES       = 6
BACKOFF_BASE_SEC  = 2.0
MAX_BACKOFF_SEC   = 64.0

# Priority options (must match Notion select options exactly)
PRIORITY_OPTIONS = ["Low", "Medium", "High", "Critical"]
STATUS_OPTIONS   = ["Open", "In Progress", "Resolved", "Closed"]

# ── Notion client ─────────────────────────────────────────────────────────────
_notion_client: Optional[Client] = None


def _get_client() -> Client:
    global _notion_client
    if _notion_client is None:
        api_key = os.getenv("NOTION_API_KEY")
        if not api_key:
            raise ValueError("NOTION_API_KEY is not set in environment / .env")
        _notion_client = Client(auth=api_key, notion_version="2022-06-28")
        logger.info("✅ StateCase Notion client initialised")
    return _notion_client


# ── Rate-limited call wrapper (mirrors notion_loader_rag._notion_call) ────────
def _notion_call(api_fn, **kwargs):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            result = api_fn(**kwargs)
            time.sleep(REQUEST_DELAY_SEC)
            return result
        except APIResponseError as api_err:
            if api_err.status != 429:
                raise
            retry_after = None
            if hasattr(api_err, "headers") and api_err.headers:
                raw = api_err.headers.get("Retry-After") or api_err.headers.get("retry-after")
                if raw:
                    try:
                        retry_after = float(raw)
                    except ValueError:
                        pass
            wait = retry_after if retry_after is not None else min(
                BACKOFF_BASE_SEC * (2 ** (attempt - 1)), MAX_BACKOFF_SEC
            )
            logger.warning("⚠️  Notion 429 (attempt %d/%d) — waiting %.1f s", attempt, MAX_RETRIES, wait)
            time.sleep(wait)
            if attempt == MAX_RETRIES:
                raise


# ── Dedup key ─────────────────────────────────────────────────────────────────
def _dedup_key(session_id: str, question: str) -> str:
    """
    Idempotency hash for ticket creation.

    Hashes QUESTION CONTENT ONLY (lowercased + stripped).
    session_id is intentionally excluded so the same question raised from a
    different session (new tab, new user) is still caught as a duplicate.
    session_id is still written to User Info for traceability.
    """
    return hashlib.sha256(question.strip().lower().encode()).hexdigest()[:16]


# ── Ticket ID counter ─────────────────────────────────────────────────────────
def _get_next_ticket_id() -> str:
    """
    Generate the next SC-XXXX ticket ID.
    Uses Redis INCR for atomicity; falls back to timestamp-based ID if Redis
    is unavailable.
    """
    try:
        import redis
        r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"), socket_connect_timeout=2)
        n = r.incr("statecase:ticket_counter")
        return f"SC-{int(n):04d}"
    except Exception:
        # Redis unavailable — use timestamp-based fallback
        ts = int(datetime.now(timezone.utc).timestamp()) % 100000
        return f"SC-{ts:05d}"


# ── Property builders ─────────────────────────────────────────────────────────
def _title_prop(text: str) -> dict:
    return {"title": [{"text": {"content": text[:2000]}}]}


def _rich_text_prop(text: str) -> dict:
    # Notion rich_text blocks max 2000 chars per element; we truncate safely
    return {"rich_text": [{"text": {"content": str(text)[:2000]}}]}


def _select_prop(value: str) -> dict:
    return {"select": {"name": value}}


def _status_prop(value: str) -> dict:
    # Notion "Status" property type is distinct from "Select" — it uses {"status": {"name": ...}}
    return {"status": {"name": value}}


# ── Property readers ──────────────────────────────────────────────────────────
def _read_title(props: dict, key: str) -> str:
    return "".join(
        rt.get("plain_text", "") for rt in props.get(key, {}).get("title", [])
    ).strip()


def _read_rich_text(props: dict, key: str) -> str:
    return "".join(
        rt.get("plain_text", "") for rt in props.get(key, {}).get("rich_text", [])
    ).strip()


def _read_select(props: dict, key: str) -> str:
    sel = props.get(key, {}).get("select") or {}
    return sel.get("name", "")


def _read_status(props: dict, key: str) -> str:
    """Read a Notion Status-type property (different from Select)."""
    st = props.get(key, {}).get("status") or {}
    return st.get("name", "")


def _page_to_ticket(page: dict) -> dict:
    """Convert a raw Notion page dict to a flat ticket dict."""
    props = page.get("properties", {})
    return {
        "notion_page_id":    page["id"],
        # Question is the title column in this database
        "ticket_id":         _read_rich_text(props, "Ticket ID"),
        "description":       _read_rich_text(props, "Description"),
        "question":          _read_title(props, "Question"),
        "assigned_owner":    _read_rich_text(props, "Assigned Owner"),
        "priority":          _read_select(props, "Priority"),
        "user_info":         _read_rich_text(props, "User Info"),
        # Status is a Notion Status type, not Select
        "status":            _read_status(props, "Status"),
        "attempted_sources": _read_rich_text(props, "Attempted Sources"),
        "created_time":      page.get("created_time", ""),
        "last_edited_time":  page.get("last_edited_time", ""),
        "url":               page.get("url", ""),
    }


# ═══════════════════════════════════════════════════════════════
#  Public API
# ═══════════════════════════════════════════════════════════════

def create_ticket(
    question: str,
    session_id: str,
    description: str = "",
    priority: str = "Medium",
    assigned_owner: str = "Unassigned",
    attempted_sources: list[str] | None = None,
    user_info: str = "",
    check_duplicate: bool = True,
) -> dict:
    """
    Create a new StateCase ticket in Notion.

    Parameters
    ──────────
    question          : the full user question that couldn't be answered
    session_id        : CiteRagLab session ID (stored in User Info)
    description       : one-line summary (auto-derived from question if empty)
    priority          : Low | Medium | High | Critical
    assigned_owner    : who will handle this ticket
    attempted_sources : list of doc titles the RAG tried
    user_info         : extra user context (session_id always prepended)
    check_duplicate   : if True, check for existing ticket before creating

    Returns a ticket dict with all fields including notion_page_id.
    """
    if priority not in PRIORITY_OPTIONS:
        logger.warning("⚠️  Invalid priority '%s' — defaulting to Medium", priority)
        priority = "Medium"

    # Auto-derive description from question if not provided
    if not description:
        description = question[:200] + ("…" if len(question) > 200 else "")

    # Build user_info string
    dedup = _dedup_key(session_id, question)
    full_user_info = f"session_id:{session_id} | dedup:{dedup}"
    if user_info:
        full_user_info += f" | {user_info}"

    # ── Idempotency check ─────────────────────────────────────────────────────
    if check_duplicate:
        # Layer 1: exact question-hash match (fast, hits Redis + Notion User Info)
        existing = _find_by_dedup(dedup)
        if existing:
            logger.info("♻️  Dedup hit (hash) — returning existing ticket_id=%s", existing["ticket_id"])
            return {**existing, "is_duplicate": True}

        # Layer 2: title substring match (Notion contains filter)
        similar = find_ticket_by_title(question[:120])
        if similar:
            logger.info("♻️  Dedup hit (title match) — returning existing ticket_id=%s", similar["ticket_id"])
            return {**similar, "is_duplicate": True}

        # Layer 3: key-term Notion search — searches ALL tickets, no 50-ticket
        # ceiling, no false positives from generic words. Extracts 2-3 distinctive
        # words from the question and requires ≥2 to match an existing ticket.
        key_match = _find_by_key_terms(question)
        if key_match:
            logger.info("♻️  Dedup hit (key terms) — returning existing ticket_id=%s", key_match["ticket_id"])
            return {**key_match, "is_duplicate": True}

    ticket_id = _get_next_ticket_id()
    sources_str = ", ".join(attempted_sources) if attempted_sources else "None"

    logger.info(
        "🎫 create_ticket — ticket_id=%s  session=%s  priority=%s",
        ticket_id, session_id, priority,
    )

    client = _get_client()
    props = {
        # Question is the title column — must use _title_prop
        "Question":          _title_prop(question),
        # Ticket ID is rich_text
        "Ticket ID":         _rich_text_prop(ticket_id),
        "Description":       _rich_text_prop(description),
        "Assigned Owner":    _rich_text_prop(assigned_owner),
        "Priority":          _select_prop(priority),
        "User Info":         _rich_text_prop(full_user_info),
        # Status is a Notion Status type — must use _status_prop, not _select_prop
        "Status":            _status_prop("Not started"),
        "Attempted Sources": _rich_text_prop(sources_str),
    }

    try:
        page = _notion_call(
            client.pages.create,
            parent={"database_id": _normalise_db_id(STATECASE_DB_ID)},
            properties=props,
        )
        ticket = _page_to_ticket(page)
        logger.info(
            "   ✅ Ticket created — ticket_id=%s  notion_page_id=%s",
            ticket["ticket_id"], ticket["notion_page_id"],
        )
        return {**ticket, "is_duplicate": False}
    except Exception as err:
        logger.error("   ❌ create_ticket failed: %s", err)
        raise


def update_ticket(
    notion_page_id: str,
    status: str | None = None,
    assigned_owner: str | None = None,
    priority: str | None = None,
    description: str | None = None,
) -> dict:
    """
    Update one or more fields on an existing ticket.

    Only fields that are not None are sent to Notion.
    Returns the updated ticket dict.
    """
    client = _get_client()
    props: dict = {}

    if status is not None:
        # Status is a Notion Status type — use _status_prop, not _select_prop
        props["Status"] = _status_prop(status)

    if assigned_owner is not None:
        props["Assigned Owner"] = _rich_text_prop(assigned_owner)

    if priority is not None:
        if priority not in PRIORITY_OPTIONS:
            raise ValueError(f"Invalid priority '{priority}'. Must be one of {PRIORITY_OPTIONS}")
        props["Priority"] = _select_prop(priority)

    if description is not None:
        props["Description"] = _rich_text_prop(description)

    if not props:
        logger.warning("⚠️  update_ticket called with no fields to update")
        return get_ticket(notion_page_id)

    logger.info(
        "✏️  update_ticket — page_id=%s  fields=%s",
        notion_page_id, list(props.keys()),
    )

    try:
        page = _notion_call(
            client.pages.update,
            page_id=notion_page_id,
            properties=props,
        )
        ticket = _page_to_ticket(page)
        logger.info("   ✅ Ticket updated — ticket_id=%s", ticket["ticket_id"])
        return ticket
    except Exception as err:
        logger.error("   ❌ update_ticket failed: %s", err)
        raise


def get_ticket(notion_page_id: str) -> dict:
    """Fetch a single ticket by its Notion page ID."""
    client = _get_client()
    try:
        page = _notion_call(client.pages.retrieve, page_id=notion_page_id)
        return _page_to_ticket(page)
    except Exception as err:
        logger.error("❌ get_ticket failed for page_id=%s: %s", notion_page_id, err)
        raise


def list_tickets(
    status_filter: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """
    Query all tickets from the StateCase database.

    Parameters
    ──────────
    status_filter : if provided, only return tickets with this status
    limit         : max rows to return (Notion page_size cap is 100)

    Returns list of ticket dicts sorted by created_time descending.
    """
    client = _get_client()
    db_id  = _normalise_db_id(STATECASE_DB_ID)

    body: dict = {"page_size": min(limit, 100), "sorts": [{"timestamp": "created_time", "direction": "descending"}]}
    if status_filter:
        # Status is a Notion "status" property type — must use {"status": ...} not {"select": ...}
        body["filter"] = {
            "property": "Status",
            "status":   {"equals": status_filter},
        }

    logger.info(
        "📋 list_tickets — status_filter=%s  limit=%d",
        status_filter or "(all)", limit,
    )

    try:
        resp = _notion_call(
            client.request,
            path=f"databases/{db_id}/query",
            method="POST",
            body=body,
        )
        tickets = [_page_to_ticket(page) for page in resp.get("results", [])]
        logger.info("   ✅ list_tickets — %d tickets returned", len(tickets))
        return tickets
    except Exception as err:
        logger.error("❌ list_tickets failed: %s", err)
        raise


# ── Internal helpers ──────────────────────────────────────────────────────────

def _normalise_db_id(raw_id: str) -> str:
    """Convert 32-char hex to dashed UUID format expected by Notion."""
    clean = raw_id.replace("-", "")
    if len(clean) == 32:
        return f"{clean[0:8]}-{clean[8:12]}-{clean[12:16]}-{clean[16:20]}-{clean[20:32]}"
    return raw_id


def _extract_key_terms(question: str) -> list[str]:
    """
    Extract up to 3 specific/distinctive words from a question for dedup matching.

    Strategy: Remove stopwords + filter by length (>4 chars).
    Longer words tend to be more specific (e.g., "termination" vs "policy").

    Uses:
    1. Common English stopwords (via nltk if available, else fallback list)
    2. Domain-specific HR/policy words (company, policy, employee, etc.)
    3. Length filter (words > 4 chars are more likely to be specific)

    Sorts by length descending, returns top 3.

    Examples:
      "policy for unlawful termination of an employee" → ["termination", "unlawful"]
      "Lack of policy for unlawful termination involving management misconduct"
                                                       → ["termination", "management", "misconduct"]
      "what is the shoes policy in the company"        → ["shoes"]
      "what is the pet policy of the company"          → ["policy"]  ← no specific terms left
    """
    # Try to load NLTK stopwords; fallback to minimal list if not available
    try:
        
        _common_stopwords = set(stopwords.words('english'))
    except (ImportError, LookupError):
        # Minimal fallback stopword list
        _common_stopwords = {
            "what", "is", "the", "a", "an", "of", "for", "in", "on", "at", "to",
            "and", "or", "are", "does", "do", "my", "our", "about", "can", "we",
            "how", "this", "that", "with", "its", "i", "by", "be", "was", "were",
            "has", "have", "had", "it", "if", "any", "all", "from", "get", "give",
            "will", "would", "could", "should", "who", "when", "where", "which",
            "there", "their", "they", "them", "than", "then", "into", "also",
        }

    # Domain-specific HR/policy words that are too generic to be distinctive
    _domain_stopwords = {
        "policy", "company", "employee", "employees", "workplace", "office",
        "work", "working", "human", "resources", "procedure", "process",
        "rules", "rule", "regulation", "guidelines", "guideline",
        "premises", "area", "building", "location", "place", "room", "floor",
        "site", "space", "zone", "section", "department", "team",
        "related", "concern", "matter", "issue", "question", "asked",
        "allow", "allowed", "permission", "permit", "require", "required",
        "applicable", "apply", "applies",
    }

    combined_stopwords = _common_stopwords | _domain_stopwords

    # Extract words > 4 chars (longer words tend to be more specific)
    words = [
        w.strip("?.,!:-") for w in question.lower().split()
        if w.strip("?.,!:-") not in combined_stopwords and len(w.strip("?.,!:-")) > 4
    ]
    words.sort(key=len, reverse=True)
    return words[:3]


def _find_by_key_terms(question: str) -> dict | None:
    """
    Layer 3 dedup: search Notion for matching specific key terms from the question.

    Why this beats word-overlap and the 50-ticket limit:
    - Searches Notion directly → covers ALL tickets, not just recent 50
    - Uses only specific terms (stopwords + domain-generic words removed)
      → avoids false positives from words like "related", "premises", "policy"
    - Threshold: ≥ 2 shared specific terms → conservative to avoid false positives
      e.g. "unlawful termination policy" and "unlawful termination procedures"
      share ["termination", "unlawful"] → correctly flagged as duplicate
      BUT "tiffin box policy" and "toy policy" share nothing → NOT a duplicate
    """
    terms = _extract_key_terms(question)
    if not terms:
        return None

    new_terms = set(terms)
    client    = _get_client()
    db_id     = _normalise_db_id(STATECASE_DB_ID)
    seen_ids  = set()
    candidates: list[dict] = []

    for term in terms:
        body = {
            "filter": {"property": "Question", "title": {"contains": term}},
            "page_size": 10,
        }
        try:
            resp = _notion_call(
                client.request,
                path=f"databases/{db_id}/query",
                method="POST",
                body=body,
            )
            for page in resp.get("results", []):
                pid = page["id"]
                if pid not in seen_ids:
                    seen_ids.add(pid)
                    candidates.append(_page_to_ticket(page))
        except Exception as err:
            logger.warning("⚠️  _find_by_key_terms term='%s': %s", term, err)

    # ≥2 shared specific terms = duplicate (conservative threshold to avoid false positives)
    # Even with expanded stopwords, requiring 2+ matches ensures we only catch
    # truly related questions, not just ones sharing a generic noun like "shoes" or "dress code"
    for ticket in candidates:
        existing_terms = set(_extract_key_terms(ticket.get("question", "")))
        shared = new_terms & existing_terms
        if len(shared) >= 2:
            logger.info(
                "♻️  Key-term dedup hit (shared=%s) — new='%s…' existing='%s…'",
                shared, question[:60], ticket.get("question", "")[:60],
            )
            return ticket

    return None


def _find_by_dedup(dedup: str) -> dict | None:
    """
    Search Notion for an existing ticket whose User Info contains the
    dedup key.  Returns the ticket dict if found, None otherwise.
    """
    client = _get_client()
    db_id  = _normalise_db_id(STATECASE_DB_ID)
    body   = {
        "filter": {
            "property": "User Info",
            "rich_text": {"contains": f"dedup:{dedup}"},
        },
        "page_size": 1,
    }
    try:
        resp = _notion_call(
            client.request,
            path=f"databases/{db_id}/query",
            method="POST",
            body=body,
        )
        results = resp.get("results", [])
        if results:
            return _page_to_ticket(results[0])
        return None
    except Exception as err:
        logger.warning("⚠️  _find_by_dedup search failed: %s", err)
        return None


def find_ticket_by_title(search_term: str) -> dict | None:
    """
    Search for a ticket by its Question (title) text or Ticket ID (e.g. "SC-0012").

    Used by update_support_ticket tool when the LLM supplies a human-readable
    name instead of a raw Notion page UUID.

    Returns the ticket dict (including notion_page_id) or None if not found.
    """
    client = _get_client()
    db_id  = _normalise_db_id(STATECASE_DB_ID)

    # Try SC-XXXX Ticket ID match first, then partial Question title match
    for filter_body in [
        {"property": "Ticket ID", "rich_text": {"contains": search_term}},
        {"property": "Question",  "title":     {"contains": search_term}},
    ]:
        body = {"filter": filter_body, "page_size": 1}
        try:
            resp = _notion_call(
                client.request,
                path=f"databases/{db_id}/query",
                method="POST",
                body=body,
            )
            results = resp.get("results", [])
            if results:
                ticket = _page_to_ticket(results[0])
                logger.info(
                    "🔍 find_ticket_by_title('%s') → ticket_id=%s  page_id=%s",
                    search_term, ticket["ticket_id"], ticket["notion_page_id"],
                )
                return ticket
        except Exception as err:
            logger.warning("⚠️  find_ticket_by_title search failed: %s", err)

    logger.info("🔍 find_ticket_by_title('%s') → not found", search_term)
    return None