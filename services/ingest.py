from __future__ import annotations

import uuid
from typing import Dict, List, Tuple

from config import (
    ALLOWED_EXTENSIONS,
    EMBEDDING_MODEL_NAME,
    MAX_FILE_BYTES,
    MAX_PDF_PAGES,
)
from services.chunking import Chunk, Segment, build_chunks
from services.embeddings import EmbeddingService
from services.file_loaders import load_docx, load_pdf, load_txt
from services.vector_store import VectorStore


class IngestionError(Exception):
    pass


def validate_extension(filename: str) -> str:
    suffix = filename.rsplit(".", 1)
    if len(suffix) != 2:
        raise IngestionError("File must have an extension")
    extension = suffix[1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise IngestionError(f"Unsupported file type: {extension}")
    return extension


def ensure_size_limit(file_size: int) -> None:
    if file_size > MAX_FILE_BYTES:
        raise IngestionError(
            f"File too large. Limit is {MAX_FILE_BYTES // (1024 * 1024)} MB"
        )


def load_segments(extension: str, content: bytes) -> Tuple[List[Segment], Dict[str, int]]:
    if extension == "txt":
        return load_txt(content), {}
    if extension == "pdf":
        segments, total_pages = load_pdf(content, MAX_PDF_PAGES)
        if total_pages > MAX_PDF_PAGES:
            raise IngestionError(
                f"PDF exceeds page limit of {MAX_PDF_PAGES}. Total pages: {total_pages}"
            )
        return segments, {"total_pages": total_pages}
    if extension == "docx":
        return load_docx(content), {}
    raise IngestionError(f"Unsupported extension: {extension}")


def ingest_file(
    filename: str,
    content: bytes,
    embedding_service: EmbeddingService,
    store: VectorStore,
) -> Dict:
    ensure_size_limit(len(content))
    extension = validate_extension(filename)
    segments, extra_meta = load_segments(extension, content)
    if not segments:
        raise IngestionError("No text content extracted from file")

    doc_uuid = str(uuid.uuid4())
    chunks = build_chunks(doc_uuid, extension, segments)
    if not chunks:
        raise IngestionError("Unable to produce chunks from document")

    embeddings = embedding_service.encode([chunk.text for chunk in chunks])
    metadata_records = [_chunk_to_metadata(chunk) for chunk in chunks]
    store.add(embeddings, metadata_records)

    return {
        "document_id": doc_uuid,
        "file_type": extension,
        "num_chunks": len(chunks),
        "embedding_model": EMBEDDING_MODEL_NAME,
        **extra_meta,
    }


def _chunk_to_metadata(chunk: Chunk) -> Dict:
    return {
        "doc_id": chunk.doc_id,
        "file_type": chunk.file_type,
        "chunk_id": chunk.chunk_id,
        "page_or_heading": chunk.page_or_heading,
        "offset": chunk.offset,
        "text": chunk.text,
    }
