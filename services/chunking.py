from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from config import CHUNK_OVERLAP, CHUNK_SIZE


@dataclass
class Segment:
    text: str
    label: str
    start_offset: int


@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    text: str
    offset: int
    page_or_heading: str
    file_type: str


def build_chunks(
    doc_id: str,
    file_type: str,
    segments: Iterable[Segment],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    chunk_idx = 0

    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        pos = 0
        length = len(text)
        while pos < length:
            end = min(length, pos + chunk_size)
            chunk_text = text[pos:end].strip()
            if not chunk_text:
                break
            global_offset = segment.start_offset + pos
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    chunk_id=chunk_idx,
                    text=chunk_text,
                    offset=global_offset,
                    page_or_heading=segment.label,
                    file_type=file_type,
                )
            )
            chunk_idx += 1
            if end == length:
                break
            pos = max(0, end - chunk_overlap)
    return chunks
