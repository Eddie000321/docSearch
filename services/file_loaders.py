from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import pdfplumber
from docx import Document

from services.chunking import Segment


def load_txt(content: bytes) -> List[Segment]:
    text = content.decode("utf-8", errors="ignore")
    segments: List[Segment] = []
    pointer = 0
    paragraph_index = 1
    for match in re.finditer(r"(.+?)(\n\s*\n|$)", text, flags=re.DOTALL):
        paragraph = match.group(1)
        start = match.start(1)
        cleaned = paragraph.strip()
        if cleaned:
            segments.append(
                Segment(text=cleaned, label=f"Paragraph {paragraph_index}", start_offset=start)
            )
            paragraph_index += 1
    if not segments and text.strip():
        segments.append(Segment(text=text.strip(), label="Document", start_offset=0))
    return segments


def load_pdf(content: bytes, max_pages: int) -> Tuple[List[Segment], int]:
    segments: List[Segment] = []
    pdf = pdfplumber.open(io.BytesIO(content))
    try:
        total_pages = len(pdf.pages)
        page_count = min(total_pages, max_pages)
        offset = 0
        for index in range(page_count):
            page = pdf.pages[index]
            extracted = page.extract_text() or ""
            cleaned = extracted.strip()
            if cleaned:
                segments.append(
                    Segment(text=cleaned, label=f"Page {index + 1}", start_offset=offset)
                )
            offset += len(extracted)
        return segments, total_pages
    finally:
        pdf.close()


def load_docx(content: bytes) -> List[Segment]:
    document = Document(io.BytesIO(content))
    segments: List[Segment] = []
    offset = 0
    paragraph_index = 1
    current_heading = ""

    for paragraph in document.paragraphs:
        raw_text = paragraph.text or ""
        cleaned = raw_text.strip()
        if not cleaned:
            offset += len(raw_text)
            continue
        style_name = (paragraph.style.name or "").lower() if paragraph.style else ""
        if style_name.startswith("heading") or "title" in style_name:
            current_heading = cleaned
            label = f"Heading: {cleaned}"[:120]
        else:
            label = current_heading or f"Paragraph {paragraph_index}"
            paragraph_index += 1
        segments.append(Segment(text=cleaned, label=label, start_offset=offset))
        offset += len(raw_text) + 1
    return segments
