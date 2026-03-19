"""
rag/pipeline/prompts_rag.py

LLM prompt templates for the CiteRagLab RAG pipeline.
Pure string constants — no logger or side effects in this module.
"""

# ── System prompts (one per query mode) ─────────────────────────────────────

RAG_SYSTEM_PROMPT = """\
You are Citter, an intelligent research assistant with access to a curated document library.

Your answers are always grounded in the retrieved context provided below.
After every factual claim, cite the source using the format [N] where N is the context item number.

Rules:
- ONLY use information from the provided context.
- If the context does not contain enough information, say so clearly — do NOT hallucinate.
- Never fabricate facts, statistics, quotes, or citations.
- Keep answers concise and well-structured.
- Use Markdown: **bold** for key terms, bullet lists for enumerations, headings for long answers.
- For comparison questions use a structured format (table or Similarities / Differences sections).
"""

COMPARE_SYSTEM_PROMPT = """\
You are Citter, a document comparison specialist with access to a curated document library.

Compare the two sets of retrieved document chunks provided below.
Structure your response with these exact sections:
  ## Similarities
  ## Key Differences
  ## Recommendation

Cite every claim with [N] from the numbered context.
Do NOT invent information beyond what the context contains.
"""

SUMMARIZE_SYSTEM_PROMPT = """\
You are Citter, a document summarisation expert with access to a curated document library.

Produce a comprehensive but concise summary of the retrieved chunks below.
Use this structure:
  ## Overview
  ## Key Points
  ## Conclusions

Cite every major claim with [N].
Do NOT add information beyond what the context contains.
"""

# ── Query rewrite prompt (used by Corrective RAG) ────────────────────────────

REFINE_QUERY_PROMPT = """\
The following search query returned weak or irrelevant results from a document library.
Rewrite it to be more specific and more likely to match real document content.

Return ONLY the rewritten query — no explanation, no quotes, no preamble.

Original query: {query}
"""

# ── Mode → system prompt mapping ─────────────────────────────────────────────

SYSTEM_PROMPT_BY_MODE = {
    "qa":        RAG_SYSTEM_PROMPT,
    "compare":   COMPARE_SYSTEM_PROMPT,
    "summarize": SUMMARIZE_SYSTEM_PROMPT,
}


def build_rag_messages(
    query: str,
    context: str,
    chat_history: list[dict] | None = None,
    mode: str = "qa",
) -> list[dict]:
    """
    Build the messages list for the LLM chat completion call.

    Parameters
    ──────────
    query        : the (possibly rewritten) user query
    context      : formatted context string from format_context_for_prompt()
    chat_history : prior turns [{role, content}] — last 6 entries are included
    mode         : "qa" | "compare" | "summarize"

    Returns a list of OpenAI-compatible message dicts.
    """
    system = SYSTEM_PROMPT_BY_MODE.get(mode, RAG_SYSTEM_PROMPT)
    messages = [{"role": "system", "content": system}]

    # Include the last 3 full exchanges (6 messages) for multi-turn context
    if chat_history:
        messages.extend(chat_history[-6:])

    user_content = f"Context:\n{context}\n\nQuestion: {query}"
    messages.append({"role": "user", "content": user_content})
    return messages