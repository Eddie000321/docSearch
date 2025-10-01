from __future__ import annotations

import hashlib
import os
from typing import Iterable, List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME


class EmbeddingService:
    """Encapsulates embedding model loading and encoding."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        self._use_fake = os.getenv("EVIDENCE_USE_FAKE_EMBEDDINGS") == "1"
        if self._use_fake:
            self._model = None
            self._dim = 384
        else:
            self._model = SentenceTransformer(model_name)
            self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        return self._dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, str):  # type: ignore[isinstance-second-argument-not-valid-type]
            raise TypeError("encode() expects a sequence of strings, not a single string")
        if self._use_fake:
            vectors = [self._fake_vector(text) for text in texts]
            return np.vstack(vectors).astype("float32")
        embeddings = self._model.encode(
            list(texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype("float32")

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def _fake_vector(self, text: str) -> np.ndarray:
        digest = hashlib.sha1(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big", signed=False)
        rng = np.random.default_rng(seed)
        vec = rng.normal(size=self._dim)
        norm = np.linalg.norm(vec) or 1.0
        return (vec / norm).astype("float32")


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms
