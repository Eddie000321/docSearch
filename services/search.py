from __future__ import annotations

from typing import Dict, List

from config import DEFAULT_TOP_K, EMBEDDING_MODEL_NAME
from services.embeddings import EmbeddingService
from services.vector_store import VectorStore


class SearchError(Exception):
    pass


def ask_question(
    question: str,
    store: VectorStore,
    embedding_service: EmbeddingService,
    top_k: int = DEFAULT_TOP_K,
) -> Dict:
    if not question.strip():
        raise SearchError("Question cannot be empty")
    if top_k <= 0:
        raise SearchError("k must be greater than zero")

    query_vector = embedding_service.encode_one(question)
    results = store.search(query_vector.reshape(1, -1), top_k)
    evidence: List[Dict] = []
    for item in results:
        evidence.append(
            {
                "doc_id": item["doc_id"],
                "file_type": item["file_type"],
                "page_or_heading": item["page_or_heading"],
                "score": item.get("score", 0.0),
                "text": item["text"],
            }
        )
    return {
        "answers": [],
        "evidence": evidence,
        "model_used": EMBEDDING_MODEL_NAME,
    }
