"""Build the student-facing Teacher's Tips booklet from its Markdown source."""

from __future__ import annotations

import html
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "teachers_tips.md"
OUTPUT = ROOT / "docs" / "teachers_tips_alphazero.pdf"

NAVY = colors.HexColor("#102A43")
BLUE = colors.HexColor("#1F6FEB")
TEAL = colors.HexColor("#0E8A80")
CORAL = colors.HexColor("#E76F51")
INK = colors.HexColor("#243B53")
MUTED = colors.HexColor("#627D98")
PALE = colors.HexColor("#EAF4F4")
PAPER = colors.HexColor("#F8FAFC")


def inline(text: str) -> str:
    """Convert the small inline-Markdown subset used by the booklet."""

    text = html.escape(text, quote=False)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r'<font name="Courier">\1</font>', text)
    text = re.sub(
        r"\[(.+?)\]\((https?://.+?)\)",
        r'<link href="\2" color="#1F6FEB">\1</link>',
        text,
    )
    return text


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "CoverTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=30,
            leading=34,
            textColor=colors.white,
            alignment=TA_LEFT,
            spaceAfter=14,
        ),
        "cover_sub": ParagraphStyle(
            "CoverSub",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=13,
            leading=19,
            textColor=colors.HexColor("#D9EAF7"),
        ),
        "contents": ParagraphStyle(
            "Contents",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=INK,
        ),
        "chapter": ParagraphStyle(
            "Chapter",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=19,
            leading=22,
            textColor=NAVY,
        ),
        "heading": ParagraphStyle(
            "Heading",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=TEAL,
            spaceBefore=7,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.4,
            leading=13.4,
            textColor=INK,
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=12.8,
            textColor=INK,
        ),
        "callout": ParagraphStyle(
            "Callout",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=14,
            textColor=NAVY,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.2,
            leading=11,
            textColor=MUTED,
        ),
    }


def draw_page(canvas, document) -> None:
    width, height = A4
    page = canvas.getPageNumber()
    canvas.saveState()

    if page == 1:
        canvas.setFillColor(NAVY)
        canvas.rect(0, 0, width, height, fill=1, stroke=0)
        canvas.setStrokeColor(colors.HexColor("#284B63"))
        canvas.setLineWidth(0.6)
        cell = 16 * mm
        for index in range(9):
            canvas.line(width - 9 * cell, height - index * cell, width, height - index * cell)
            canvas.line(width - index * cell, height - 9 * cell, width - index * cell, height)
        canvas.setFillColor(TEAL)
        canvas.circle(width - 31 * mm, height - 38 * mm, 8 * mm, fill=1, stroke=0)
        canvas.setFillColor(CORAL)
        canvas.circle(width - 63 * mm, height - 70 * mm, 6 * mm, fill=1, stroke=0)
    else:
        canvas.setFillColor(PAPER)
        canvas.rect(0, 0, width, height, fill=1, stroke=0)
        canvas.setFillColor(NAVY)
        canvas.rect(0, height - 12 * mm, width, 12 * mm, fill=1, stroke=0)
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 7.5)
        canvas.drawString(18 * mm, height - 7.5 * mm, "TEACHER'S TIPS  /  ALPHAZERO")
        canvas.setStrokeColor(colors.HexColor("#D9E2EC"))
        canvas.line(18 * mm, 14 * mm, width - 18 * mm, 14 * mm)
        canvas.setFillColor(MUTED)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(18 * mm, 9.5 * mm, "Design for correctness, diagnosis, and fair evidence")
        canvas.drawRightString(width - 18 * mm, 9.5 * mm, str(page - 1))

    canvas.restoreState()


def chapter_header(number: str, title: str, style: ParagraphStyle) -> Table:
    badge = Paragraph(
        f'<font color="#FFFFFF"><b>{number}</b></font>',
        ParagraphStyle("Badge", fontName="Helvetica-Bold", fontSize=16, alignment=TA_CENTER),
    )
    table = Table(
        [[badge, Paragraph(inline(title), style)]],
        colWidths=[14 * mm, 148 * mm],
        rowHeights=[16 * mm],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), TEAL),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (1, 0), (1, 0), 8),
                ("BOX", (0, 0), (0, 0), 0, TEAL),
            ]
        )
    )
    return table


def callout(text: str, style: ParagraphStyle) -> Table:
    box = Table([[Paragraph(inline(text), style)]], colWidths=[158 * mm])
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE),
                ("BOX", (0, 0), (-1, -1), 0.7, TEAL),
                ("LINEBEFORE", (0, 0), (0, -1), 4, TEAL),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    return box


def parse_markdown(markdown: str, theme: dict[str, ParagraphStyle]) -> list:
    lines = markdown.splitlines()
    story = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            story.append(Paragraph(inline(" ".join(paragraph)), theme["body"]))
            paragraph.clear()

    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        if line.startswith("# "):
            index += 1
            continue
        if line.startswith("## "):
            flush_paragraph()
            match = re.match(r"##\s+(\d+)\.\s+(.+)", line)
            if match:
                story.append(PageBreak())
                story.append(chapter_header(match.group(1), match.group(2), theme["chapter"]))
                story.append(Spacer(1, 4 * mm))
            index += 1
            continue
        if line.startswith("### "):
            flush_paragraph()
            story.append(Paragraph(inline(line[4:]), theme["heading"]))
            index += 1
            continue
        if line.startswith("> "):
            flush_paragraph()
            quoted = []
            while index < len(lines) and lines[index].startswith("> "):
                quoted.append(lines[index][2:].strip())
                index += 1
            story.append(KeepTogether([callout(" ".join(quoted), theme["callout"]), Spacer(1, 3 * mm)]))
            continue
        if line.startswith("- ") or re.match(r"\d+\.\s", line):
            flush_paragraph()
            numbered = bool(re.match(r"\d+\.\s", line))
            items = []
            while index < len(lines):
                candidate = lines[index].strip()
                if numbered:
                    match = re.match(r"\d+\.\s+(.+)", candidate)
                    if not match:
                        break
                    content = match.group(1)
                else:
                    if not candidate.startswith("- "):
                        break
                    content = candidate[2:]
                index += 1
                while (
                    index < len(lines)
                    and lines[index].startswith("  ")
                    and lines[index].strip()
                ):
                    content += " " + lines[index].strip()
                    index += 1
                items.append(ListItem(Paragraph(inline(content), theme["bullet"]), leftIndent=10))
            story.append(
                ListFlowable(
                    items,
                    bulletType="1" if numbered else "bullet",
                    start="1" if numbered else "-",
                    leftIndent=15,
                    bulletFontName="Helvetica-Bold",
                    bulletFontSize=8,
                    bulletColor=TEAL,
                    spaceAfter=6,
                )
            )
            continue
        if not line:
            flush_paragraph()
        else:
            paragraph.append(line.strip())
        index += 1

    flush_paragraph()
    return story


def build() -> None:
    theme = styles()
    document = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        rightMargin=24 * mm,
        leftMargin=24 * mm,
        topMargin=21 * mm,
        bottomMargin=20 * mm,
        title="Teacher's Tips for Designing and Training Your AlphaZero Agent",
        author="Breakthrough Zero teaching project",
        subject="Practical advice for student AlphaZero projects",
    )

    story = [
        Spacer(1, 68 * mm),
        Paragraph("TEACHER'S TIPS", theme["cover_sub"]),
        Spacer(1, 4 * mm),
        Paragraph(
            "Designing and Training<br/>Your AlphaZero Agent",
            theme["cover_title"],
        ),
        HRFlowable(width="38%", thickness=4, color=CORAL, spaceBefore=5, spaceAfter=13),
        Paragraph(
            "A field guide to catching silent bugs, spending compute wisely, "
            "and turning experiments into evidence.",
            theme["cover_sub"],
        ),
        Spacer(1, 52 * mm),
        Paragraph(
            "Prepared for classroom use  /  Breakthrough Zero  /  2026",
            theme["cover_sub"],
        ),
        PageBreak(),
        Paragraph("Contents", theme["chapter"]),
        Spacer(1, 3 * mm),
    ]

    titles = [
        "Optimize for diagnosis, not training hours",
        "Choose a game and representation that expose mistakes",
        "Pick one value convention and make it impossible to violate",
        "Treat terminal states as a separate species",
        "Prove PUCT with a dummy network",
        "Optimize measured hot paths, not attractive ideas",
        "Save self-play as a reusable scientific instrument",
        "Use pretraining data as an offline laboratory",
        "Understand root noise before choosing its constants",
        "Treat the replay buffer as a control system",
        "Evaluate strength fairly and make Elo honest",
        "Expect self-play to make the agent worse",
        "Run research without building a pile of patches",
        "Stop/go checklist before an HPC run",
        "Further reading",
    ]
    rows = []
    for start in range(0, len(titles), 2):
        row = []
        for offset in (0, 1):
            number = start + offset + 1
            if number <= len(titles):
                row.append(
                    Paragraph(
                        f'<font color="#0E8A80"><b>{number:02d}</b></font>  '
                        f"{inline(titles[number - 1])}",
                        theme["contents"],
                    )
                )
            else:
                row.append("")
        rows.append(row)
    contents = Table(rows, colWidths=[80 * mm, 80 * mm], rowHeights=17 * mm)
    contents.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#D9E2EC")),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.extend(
        [
            contents,
            Spacer(1, 7 * mm),
            callout(
                "The central lesson: a system that reveals why it failed will "
                "usually improve faster than one given more blind compute.",
                theme["callout"],
            ),
        ]
    )
    story.extend(parse_markdown(SOURCE.read_text(encoding="utf-8"), theme))

    document.build(story, onFirstPage=draw_page, onLaterPages=draw_page)
    print(OUTPUT)


if __name__ == "__main__":
    build()
