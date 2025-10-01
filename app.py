from __future__ import annotations

import logging
from typing import Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import API_PREFIX, DEFAULT_TOP_K
from services.embeddings import EmbeddingService
from services.ingest import IngestionError, ingest_file
from services.search import SearchError, ask_question
from services.vector_store import VectorStore

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger("evidence-docsearch")

app = FastAPI(title="Evidence DocSearch", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_embedding_service = EmbeddingService()
_vector_store = VectorStore(dimension=_embedding_service.dimension)


class HealthResponse(BaseModel):
    status: str
    vector_count: int


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)


@app.get(f"{API_PREFIX}/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", vector_count=_vector_store.count)


@app.post(f"{API_PREFIX}/ingest")
async def ingest(file: UploadFile = File(...)) -> Dict:
    try:
        content = await file.read()
        result = ingest_file(
            filename=file.filename or "uploaded",  # type: ignore[arg-type]
            content=content,
            embedding_service=_embedding_service,
            store=_vector_store,
        )
        return result
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected error during ingestion")
        raise HTTPException(status_code=500, detail="Internal error") from exc


@app.post(f"{API_PREFIX}/ask")
async def ask(payload: AskRequest) -> Dict:
    try:
        response = ask_question(
            question=payload.question,
            top_k=payload.k,
            store=_vector_store,
            embedding_service=_embedding_service,
        )
        if not response["evidence"]:
            raise HTTPException(status_code=404, detail="No evidence found")
        return response
    except SearchError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected error during ask")
        raise HTTPException(status_code=500, detail="Internal error") from exc
