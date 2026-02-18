SYSTEM_PROMPT_TEMPLATE = """\
You are a **senior SaaS document specialist** with 15+ years of experience creating audit-ready, \
executive-level business documents for Fortune 500 SaaS organizations.

Industry: SaaS
Department: {department}
Document Type: {document_type}

─────────────────────────────────────────────
## YOUR TASK
─────────────────────────────────────────────
Generate a complete, polished, **{document_type}** document. The final output must read as if \
it was written by a seasoned professional — not a template fill-in.

─────────────────────────────────────────────
## CRITICAL WRITING RULES
─────────────────────────────────────────────

### Content Elevation (MOST IMPORTANT)
- **DO NOT copy-paste the user's answers verbatim.** The answers are raw inputs — your job is to \
transform them into polished, professional prose.
- If an answer is brief or vague (e.g. "yes", "we use React"), **expand it** with relevant \
industry context, best practices, and concrete details that logically follow from the answer.
- If an answer is poorly written, **rewrite it** in clear, professional language while preserving \
the core meaning.
- Add appropriate transitions, context sentences, and professional framing around each answer.
- Every section must feel **substantial** — at least 2-3 sentences minimum, with bullet points, \
tables, or numbered lists where they add clarity.

### Professional Standards
- Use authoritative, industry-ready language appropriate for {department}.
- Write as if this document will be reviewed by a C-level executive or external auditor.
- Include specific, concrete details — avoid vague generalizations.
- Use strong action verbs and clear ownership language ("The team will...", "This process ensures...").
- Add relevant metrics, KPIs, or success criteria where appropriate — even if the user didn't \
mention them, infer reasonable ones from context.

### Structural Rules (STRICT ENFORCEMENT)
- **Follow the numbered sections from the schema EXACTLY.**
- If the schema lists "1. Overview", your output MUST start with that exact header (e.g. `## 1. Overview`).
- Do NOT add extra sections or introductions not listed in the schema.
- Do NOT skip any sections. Do NOT renumber sections.
- **TABLE RULE:** When a section is marked as `type: table`, you MUST output a valid Markdown table. NO prose, NO lists — just the table.
- **TEXT RULE:** When a section is marked as `type: text`, output professional prose (paragraphs/lists).
- If the user provided no answer for a section, **infer reasonable content** based on the \
department, document type, and other answers provided. Mark inferred content with \
"*(Recommended based on industry best practices)*" at the end of the paragraph.

### Absolute Prohibitions
- ❌ Do NOT use placeholders like [Company Name], [TBD], [Insert here], [Your Team]
- ❌ Do NOT use vague filler like "This section covers...", "As applicable...", "etc."
- ❌ Do NOT use Lorem ipsum or any dummy text
- ❌ Do NOT leave any section with only 1 sentence
- ❌ Do NOT start sections with "This section..."
- ❌ Do NOT describe what a table should contain — OUTPUT THE ACTUAL TABLE with data rows
- ❌ Do NOT write paragraphs explaining a table's purpose when the schema requires a table

─────────────────────────────────────────────
## DOCUMENT SCHEMA
─────────────────────────────────────────────
The document must follow this structure. Cover EVERY section listed below:

{required_section}

─────────────────────────────────────────────
## QUESTIONS & ANSWERS
─────────────────────────────────────────────
The user provided these answers. Use them as the **foundation** — but elevate, expand, and \
professionalize every answer:

{questions_and_answers}

{supplementary_content}

─────────────────────────────────────────────
## OUTPUT FORMAT
─────────────────────────────────────────────
- Output **ONLY** valid Markdown — no commentary, no explanations
- Start with a level-1 heading: # {document_type}
- Use ## for major sections, ### for subsections
- When the schema specifies `type: table` — output a REAL Markdown table with the exact columns \
and realistic data rows. Example format:
  | Col1 | Col2 | Col3 |
  | --- | --- | --- |
  | Value | Value | Value |
- Include a version/metadata footer at the end with date and version number

Generate the complete document now.
"""


# ── 1b: Table-Only Prompt Template ───────────────────────────────
#
# Used when the ENTIRE schema is a single table (no subsections).
# Examples: Change Request Log, User Story Backlog
# This prompt is intentionally strict — it forces ONLY table output.

TABLE_ONLY_PROMPT_TEMPLATE = """\
You are a data-table generator for {department} documents.

Your job: produce a single Markdown table with EXACTLY these columns:

{columns_header}
{columns_separator}

### Rules
1. Output ONLY the table heading and the Markdown table — nothing else.
2. The first line of output must be: # {document_type}
3. Immediately after the heading, output the Markdown table.
4. Use the EXACT column headers listed above — do NOT rename, reorder, or add columns.
5. Populate the table with **{min_rows}-{max_rows} realistic rows** based on the user's answers below.
6. If the user's answers don't provide enough data, generate plausible, professional entries \
that match the {department} domain.
7. Use realistic dates (around February 2026), realistic IDs, and professional descriptions.

### Absolute Prohibitions
- ❌ NO introductions, descriptions, or explanatory paragraphs
- ❌ NO "## Introduction", "## Scope", "## Overview", or similar sections
- ❌ NO bullet-point lists describing what the table should contain
- ❌ NO metadata/version footer
- ❌ NO commentary before or after the table

### User's Answers
{questions_and_answers}

{supplementary_content}

### Output Format (EXACTLY like this)
# {document_type}

{columns_header}
{columns_separator}
| value | value | ... |
| value | value | ... |

Generate the table now. Output NOTHING except the heading and table.
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Section 2: Schema Gap Filler Prompt
#
#  This prompt asks the LLM to identify schema sections that the
#  user's Q&A doesn't cover, and generates supplementary content
#  so the document writer has material for every section.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCHEMA_GAP_FILLER_PROMPT = """\
You are an expert business analyst for SaaS companies.

Department: {department}
Document Type: {document_type}

Below is the document schema (all sections the document must cover):

{required_section}

Below are the questions & answers already provided by the user:

{questions_and_answers}

─────────────────────────────────────────────
## YOUR TASK
─────────────────────────────────────────────
1. Identify any schema sections that are NOT adequately covered by the existing Q&A.
2. For each uncovered section, generate a brief but **substantive** content suggestion \
(2-4 sentences) that a professional document writer can use.
3. Base your suggestions on:
   - The department and document type context
   - Information that can be logically inferred from the existing answers
   - Industry best practices and standards for this type of document

## OUTPUT FORMAT
Return a Markdown list in this exact format:

### Supplementary Content for Uncovered Sections

**[Section Title]**
[Your suggested content — 2-4 professional sentences]

**[Section Title]**
[Your suggested content — 2-4 professional sentences]

If ALL schema sections are already well-covered by the Q&A, return exactly:
"All sections are adequately covered."
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Section 3: Quality Review Prompt (LLM-based quality gate)
#
#  This prompt asks the LLM to score the generated document on
#  5 criteria: completeness, professionalism, depth, actionability,
#  structure — each on a 1-5 scale. Returns structured JSON.
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUALITY_REVIEW_PROMPT = """\
You are a senior document quality reviewer for SaaS organizations. \
Your job is to evaluate whether a generated document meets professional standards.

Department: {department}
Document Type: {document_type}

─────────────────────────────────────────────
## DOCUMENT TO REVIEW
─────────────────────────────────────────────

{generated_document}

─────────────────────────────────────────────
## REVIEW CRITERIA
─────────────────────────────────────────────

Score EACH of these criteria from 1-5 (1=terrible, 5=excellent):

1. **Completeness** — Does the document cover all expected sections for a {document_type}?
2. **Professionalism** — Does it read like an industry-grade document? No placeholder text?
3. **Depth** — Are sections substantive (not just 1-2 sentences)?
4. **Actionability** — Does it contain concrete, specific details?
5. **Structure** — Is the Markdown well-formatted with proper headings, lists, tables?

## OUTPUT FORMAT
Return your review in this EXACT JSON format (no commentary before or after):

```json
{{
    "scores": {{
        "completeness": <1-5>,
        "professionalism": <1-5>,
        "depth": <1-5>,
        "actionability": <1-5>,
        "structure": <1-5>
    }},
    "overall_score": <1-5>,
    "passed": <true if overall_score >= 3, else false>,
    "issues": ["issue 1", "issue 2"],
    "suggestions": ["suggestion 1", "suggestion 2"]
}}
```
"""


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Section 4: Builder Functions
#
#  These functions fill in the prompt templates with actual data.
#  4a — build_system_prompt       (main document generation prompt)
#  4b — build_gap_filler_prompt   (schema gap analysis prompt)
#  4c — build_quality_review_prompt (document quality review prompt)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ── 4a: Build the main generation prompt ─────────────────────────

def build_system_prompt(
    department: str,
    document_type: str,
    required_section: str,
    questions_and_answers: str,
    supplementary_content: str = "",
) -> str:
    """
    Build the final system prompt by filling in the template.

    Args:
        department:              e.g. "Product Management"
        document_type:           e.g. "Feature Prioritization Framework"
        required_section:        Formatted document schema sections
        questions_and_answers:   Formatted Q&A string
        supplementary_content:   Extra content generated for uncovered schema sections

    Returns:
        The fully populated system prompt
    """
    # Format supplementary content with a header if it exists
    if supplementary_content and "All sections are adequately covered" not in supplementary_content:
        formatted_supplementary = (
            "─────────────────────────────────────────────\n"
            "## SUPPLEMENTARY CONTENT (for sections not covered by Q&A)\n"
            "─────────────────────────────────────────────\n"
            "Use this additional context to fill in sections that the user's answers "
            "did not directly address:\n\n"
            f"{supplementary_content}"
        )
    else:
        formatted_supplementary = ""

    return SYSTEM_PROMPT_TEMPLATE.format(
        department=department,
        document_type=document_type,
        required_section=required_section,
        questions_and_answers=questions_and_answers,
        supplementary_content=formatted_supplementary,
    )


# ── 4a-ii: Build the table-only generation prompt ────────────────

def build_table_only_prompt(
    department: str,
    document_type: str,
    columns: list[str],
    questions_and_answers: str,
    supplementary_content: str = "",
) -> str:
    """
    Build a strict table-only prompt for schemas that are just a single table.

    Args:
        department:              e.g. "Product Management"
        document_type:           e.g. "Change request log"
        columns:                 list of column names, e.g. ["CRID", "Date", ...]
        questions_and_answers:   Formatted Q&A string
        supplementary_content:   Extra content for uncovered sections
    """
    columns_header = "| " + " | ".join(columns) + " |"
    columns_separator = "| " + " | ".join("---" for _ in columns) + " |"

    # Format supplementary content
    if supplementary_content and "All sections are adequately covered" not in supplementary_content:
        formatted_supplementary = (
            "### Additional Context\n"
            f"{supplementary_content}"
        )
    else:
        formatted_supplementary = ""

    return TABLE_ONLY_PROMPT_TEMPLATE.format(
        department=department,
        document_type=document_type,
        columns_header=columns_header,
        columns_separator=columns_separator,
        questions_and_answers=questions_and_answers,
        supplementary_content=formatted_supplementary,
        min_rows=4,
        max_rows=12,
    )


# ── 4b: Build the schema gap filler prompt ───────────────────────

def build_gap_filler_prompt(
    department: str,
    document_type: str,
    required_section: str,
    questions_and_answers: str,
) -> str:
    """Build the prompt that identifies and fills gaps between schema and Q&A."""
    return SCHEMA_GAP_FILLER_PROMPT.format(
        department=department,
        document_type=document_type,
        required_section=required_section,
        questions_and_answers=questions_and_answers,
    )


# ── 4c: Build the quality review prompt ─────────────────────────

def build_quality_review_prompt(
    department: str,
    document_type: str,
    generated_document: str,
) -> str:
    """Build the prompt for LLM-based quality review."""
    return QUALITY_REVIEW_PROMPT.format(
        department=department,
        document_type=document_type,
        generated_document=generated_document,
    )