"""
PDF generation from Markdown for DocForge Hub.

Uses ReportLab Platypus to render Markdown text into a professional A4 PDF.
All functions are pure (no Streamlit dependency) and can be reused by any
Python application that needs Markdown-to-PDF conversion.
"""

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


# ═══════════════════════════════════════════════════════════════
#  PDF colour palette
# ═══════════════════════════════════════════════════════════════

_PDF_BRAND  = colors.HexColor("#1E3A5F")
_PDF_ACCENT = colors.HexColor("#2E86AB")
_PDF_MID    = colors.HexColor("#8C9BAB")
_PDF_DARK   = colors.HexColor("#1A1A2E")
_PDF_TH_BG  = colors.HexColor("#2E86AB")
_PDF_ROW_A  = colors.HexColor("#F0F5FA")
_PDF_BORDER = colors.HexColor("#CBD5E1")


# ═══════════════════════════════════════════════════════════════
#  Style definitions
# ═══════════════════════════════════════════════════════════════

def _build_pdf_styles() -> dict:
    """Return a dict of named ParagraphStyles for document rendering."""
    base_kwargs = dict(fontName="Helvetica", textColor=_PDF_DARK, leading=14)
    return {
        "doc_title": ParagraphStyle(
            "DocTitle", fontName="Helvetica-Bold", fontSize=24,
            textColor=_PDF_BRAND, leading=30, spaceAfter=4, alignment=TA_CENTER,
        ),
        "doc_sub": ParagraphStyle(
            "DocSub", fontName="Helvetica", fontSize=10,
            textColor=_PDF_MID, leading=15, spaceAfter=18, alignment=TA_CENTER,
        ),
        "h1": ParagraphStyle(
            "H1", fontName="Helvetica-Bold", fontSize=16,
            textColor=_PDF_BRAND, leading=20, spaceBefore=20, spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "H2", fontName="Helvetica-Bold", fontSize=13,
            textColor=_PDF_ACCENT, leading=17, spaceBefore=14, spaceAfter=4,
        ),
        "h3": ParagraphStyle(
            "H3", fontName="Helvetica-BoldOblique", fontSize=11,
            textColor=_PDF_ACCENT, leading=15, spaceBefore=10, spaceAfter=3,
        ),
        "body": ParagraphStyle("Body", **base_kwargs, fontSize=10, spaceAfter=5),
        "bullet": ParagraphStyle("Bullet", **base_kwargs, fontSize=10, leftIndent=14, spaceAfter=3),
    }


# ═══════════════════════════════════════════════════════════════
#  Text cleaning & formatting
# ═══════════════════════════════════════════════════════════════

_UNICODE_REPLACEMENTS = {
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
}


def clean_text_for_pdf(text: str) -> str:
    """
    Replace Unicode characters that ReportLab can't render with ASCII equivalents,
    then drop anything still outside latin-1 to avoid black-box glyphs.
    """
    if not isinstance(text, str):
        text = str(text)
    for unicode_char, ascii_replacement in _UNICODE_REPLACEMENTS.items():
        text = text.replace(unicode_char, ascii_replacement)
    return text.encode("latin-1", "replace").decode("latin-1")


def _build_paragraph(text: str, style) -> Paragraph:
    """Convert Markdown inline formatting (**bold**, *italic*, `code`) to ReportLab XML."""
    text = clean_text_for_pdf(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*(.+?)\*",     r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`",       r"<font name='Courier'>\1</font>", text)
    return Paragraph(text, style)


# ═══════════════════════════════════════════════════════════════
#  Markdown table parsing & rendering
# ═══════════════════════════════════════════════════════════════

def parse_markdown_table(lines: list[str]) -> list[list[str]]:
    """
    Parse pipe-delimited Markdown table lines into a 2D list of cell strings.
    Skips separator rows (e.g. |---|---|).
    """
    rows = []
    for line_text in lines:
        line_text = line_text.strip()
        if re.match(r"^\|[\s\-\|:]+\|$", line_text):
            continue
        cells = [cell_text.strip() for cell_text in line_text.split("|") if cell_text.strip()]
        if cells:
            rows.append(cells)
    return rows


def build_reportlab_table(rows: list[list[str]], pdf_styles: dict) -> Table | None:
    """
    Convert parsed table rows into a styled ReportLab Table object.
    Returns None if the input is empty.
    """
    if not rows:
        return None
    max_cols = max(len(row_data) for row_data in rows)
    rows = [row_data + [""] * (max_cols - len(row_data)) for row_data in rows]
    header = [
        Paragraph(f"<b>{clean_text_for_pdf(header_cell)}</b>", pdf_styles["body"])
        for header_cell in rows[0]
    ]
    body = [
        [Paragraph(clean_text_for_pdf(body_cell), pdf_styles["body"]) for body_cell in row_data]
        for row_data in rows[1:]
    ]
    col_width = (A4[0] - 5 * cm) / max_cols
    pdf_table = Table([header] + body, colWidths=[col_width] * max_cols, repeatRows=1)
    pdf_table.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  _PDF_TH_BG),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, -1), 9),
        ("LEADING",       (0, 0), (-1, -1), 13),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [_PDF_ROW_A, colors.white]),
        ("GRID",          (0, 0), (-1, -1), 0.4, _PDF_BORDER),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 7),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 7),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return pdf_table


# ═══════════════════════════════════════════════════════════════
#  Main entry point: Markdown → PDF bytes
# ═══════════════════════════════════════════════════════════════

def generate_pdf_from_markdown(markdown_text: str, document_title: str = "") -> bytes:
    """Render Markdown text to a professional A4 PDF. Returns raw PDF bytes."""
    pdf_buffer = io.BytesIO()
    pdf_document = SimpleDocTemplate(
        pdf_buffer, pagesize=A4,
        leftMargin=2.5 * cm, rightMargin=2.5 * cm,
        topMargin=2.5 * cm, bottomMargin=2.5 * cm,
        title=document_title or "Document",
        author="DocForgeHub",
    )
    pdf_styles = _build_pdf_styles()
    story = []

    if document_title:
        story += [
            Spacer(1, 0.8 * cm),
            Paragraph(clean_text_for_pdf(document_title), pdf_styles["doc_title"]),
            HRFlowable(width="100%", thickness=2, color=_PDF_ACCENT, spaceBefore=4, spaceAfter=4),
            Paragraph("Generated by DocForgeHub", pdf_styles["doc_sub"]),
            Spacer(1, 0.8 * cm),
        ]

    lines = markdown_text.split("\n")
    line_idx = 0
    while line_idx < len(lines):
        current_line = lines[line_idx].strip()
        if not current_line:
            story.append(Spacer(1, 4))
            line_idx += 1
            continue
        if current_line.startswith("|"):
            table_markdown_lines = []
            while line_idx < len(lines) and lines[line_idx].strip().startswith("|"):
                table_markdown_lines.append(lines[line_idx])
                line_idx += 1
            pdf_table_obj = build_reportlab_table(parse_markdown_table(table_markdown_lines), pdf_styles)
            if pdf_table_obj:
                story += [Spacer(1, 6), pdf_table_obj, Spacer(1, 10)]
            continue
        if current_line.startswith("# "):
            story.append(_build_paragraph(current_line[2:].strip(), pdf_styles["h1"]))
            story.append(HRFlowable(width="100%", thickness=1, color=_PDF_BRAND, spaceAfter=4))
        elif current_line.startswith("## "):
            story.append(_build_paragraph(current_line[3:].strip(), pdf_styles["h2"]))
        elif current_line.startswith("### "):
            story.append(_build_paragraph(current_line[4:].strip(), pdf_styles["h3"]))
        elif re.match(r"^[\*\-\+]\s+", current_line):
            story.append(_build_paragraph("• " + re.sub(r"^[\*\-\+]\s+", "", current_line), pdf_styles["bullet"]))
        elif re.match(r"^\d+\.\s+", current_line):
            story.append(_build_paragraph(current_line, pdf_styles["body"]))
        else:
            story.append(_build_paragraph(current_line, pdf_styles["body"]))
        line_idx += 1

    pdf_document.build(story)
    return pdf_buffer.getvalue()


def build_safe_pdf_filename(title: str) -> str:
    """Convert a document title into a filesystem-safe PDF filename."""
    safe = re.sub(r"[^\w\s\-]", "", title).strip()
    safe = re.sub(r"\s+", "_", safe)
    return (safe or "document") + ".pdf"