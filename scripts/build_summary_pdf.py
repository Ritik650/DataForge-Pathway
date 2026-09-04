"""Render docs/CONCEPT_SUMMARY.md to a print-ready PDF.

Deliberately a small, explicit renderer rather than a markdown->HTML->PDF chain:
the summary is a fixed, known document and the only structures it uses are
headings, paragraphs with bold/italic/code spans, one table, a blockquote
equation and a rule. Committing the renderer keeps the PDF reproducible from
the markdown, so the two cannot drift.

Run: python scripts/build_summary_pdf.py
"""

import pathlib
import re
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, Frame, HRFlowable, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "CONCEPT_SUMMARY.md"
OUT = ROOT / "docs" / "concept-summary.pdf"

INK = colors.HexColor("#1a1a18")
PROSE = colors.HexColor("#2e2d28")
ACCENT = colors.HexColor("#b3401b")
MUTED = colors.HexColor("#6b6a63")
RULE = colors.HexColor("#d5d2c8")
WASH = colors.HexColor("#f6f5f1")

SERIF = "Times-Roman"
SERIF_B = "Times-Bold"
SERIF_I = "Times-Italic"
MONO = "Courier"

styles = {
    "title": ParagraphStyle("title", fontName=SERIF_B, fontSize=15.5, leading=18,
                            textColor=INK, spaceAfter=3),
    "sub": ParagraphStyle("sub", fontName=SERIF, fontSize=8.0, leading=10,
                          textColor=MUTED, spaceAfter=7),
    "body": ParagraphStyle("body", fontName=SERIF, fontSize=8.5, leading=11.0,
                           textColor=PROSE, alignment=TA_JUSTIFY, spaceAfter=5),
    "eq": ParagraphStyle("eq", fontName=MONO, fontSize=7.8, leading=10,
                         textColor=INK, leftIndent=12, spaceBefore=1, spaceAfter=5),
    "cell": ParagraphStyle("cell", fontName=SERIF, fontSize=7.1, leading=8.7,
                           textColor=PROSE),
    "cellh": ParagraphStyle("cellh", fontName=SERIF_B, fontSize=7.1, leading=8.7,
                            textColor=INK),
    "foot": ParagraphStyle("foot", fontName=SERIF_I, fontSize=7.4, leading=9.6,
                           textColor=MUTED, spaceBefore=4),
}


def inline(md: str) -> str:
    """Markdown spans -> reportlab markup. Order matters: code before bold."""
    md = md.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    md = re.sub(r"`([^`]+)`", rf'<font face="{MONO}" size="8">\1</font>', md)
    md = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", md)
    md = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", md)
    md = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", md)
    return md


def build():
    if not SRC.exists():
        sys.exit(f"missing {SRC}")
    lines = SRC.read_text(encoding="utf-8").splitlines()

    flow = []
    table_rows = []
    i = 0
    while i < len(lines):
        ln = lines[i].rstrip()

        if ln.startswith("# "):
            flow.append(Paragraph(inline(ln[2:]), styles["title"]))
        elif ln.startswith("**Concept:**"):
            flow.append(Paragraph(inline(ln), styles["sub"]))
        elif ln.startswith("---"):
            flow.append(HRFlowable(width="100%", thickness=0.6, color=RULE,
                                   spaceBefore=1, spaceAfter=5))
        elif ln.startswith("> "):
            flow.append(Paragraph(inline(ln[2:]), styles["eq"]))
        elif ln.startswith("|"):
            row = [c.strip() for c in ln.strip("|").split("|")]
            if not all(set(c) <= set("-: ") for c in row):
                table_rows.append(row)
            if i + 1 >= len(lines) or not lines[i + 1].startswith("|"):
                flow.append(make_table(table_rows))
                table_rows = []
        elif ln.startswith("*") and ln.endswith("*") and len(ln) > 2 and not ln.startswith("**"):
            flow.append(Paragraph(inline(ln.strip("*")), styles["foot"]))
        elif ln:
            # join wrapped paragraph lines
            buf = [ln]
            while i + 1 < len(lines) and lines[i + 1].strip() and \
                    not lines[i + 1].startswith(("|", ">", "#", "---")):
                i += 1
                buf.append(lines[i].rstrip())
            flow.append(Paragraph(inline(" ".join(buf)), styles["body"]))
        i += 1

    doc = BaseDocTemplate(str(OUT), pagesize=A4,
                          leftMargin=15 * mm, rightMargin=15 * mm,
                          topMargin=12 * mm, bottomMargin=11 * mm,
                          title="Where a fact lives — concept summary",
                          author="Ritik Yadav")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
    doc.addPageTemplates([PageTemplate(id="p", frames=[frame])])
    doc.build(flow)

    pages = count_pages(OUT)
    print(f"wrote {OUT} — {OUT.stat().st_size / 1024:.0f} KB, {pages} page(s)")
    if pages > 1:
        print("  WARNING: the brief asks for a ONE-page summary. Tighten the "
              "source or reduce leading before submitting.")
    return pages


def make_table(rows):
    if not rows:
        return Spacer(1, 0)
    data = [[Paragraph(inline(c), styles["cellh"] if r == 0 else styles["cell"])
             for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=[34 * mm, 44 * mm, 44 * mm, 58 * mm], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), WASH),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, RULE),
        ("LINEBELOW", (0, 1), (-1, -2), 0.3, colors.HexColor("#eae7de")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 2.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6),
    ]))
    return t


def count_pages(path):
    raw = path.read_bytes()
    return max(raw.count(b"/Type /Page\n"), raw.count(b"/Type/Page"), 1)


if __name__ == "__main__":
    build()
