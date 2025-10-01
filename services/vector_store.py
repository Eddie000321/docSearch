from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Dict, Iterable, List

import faiss
import numpy as np

from config import INDEX_PATH, META_PATH


class VectorStore:
    def __init__(
        self,
        index_path: Path = INDEX_PATH,
        meta_path: Path = META_PATH,
        dimension: int = 384,
    ) -> None:
        self.index_path = index_path
        self.meta_path = meta_path
        self.dimension = dimension
        self._lock = Lock()
        self._metadata: List[Dict] = []
        self._load()

    def _load(self) -> None:
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            if self.index.d != self.dimension:
                raise ValueError(
                    f"Index dimension {self.index.d} does not match expected {self.dimension}"
                )
        else:
            self.index = faiss.IndexFlatIP(self.dimension)

        if self.meta_path.exists():
            with self.meta_path.open("r", encoding="utf-8") as fh:
                records = [json.loads(line) for line in fh if line.strip()]
            records.sort(key=lambda item: item["id"])
            self._metadata = records
        else:
            self.meta_path.touch()
            self._metadata = []
        self._next_id = len(self._metadata)

    def add(self, embeddings: np.ndarray, metadata: List[Dict]) -> None:
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype("float32")
        if embeddings.ndim != 2:
            raise ValueError("Embeddings should be 2D array")
        if embeddings.shape[0] != len(metadata):
            raise ValueError("Embeddings count and metadata length mismatch")

        with self._lock:
            ids = range(self._next_id, self._next_id + len(metadata))
            for meta, idx in zip(metadata, ids):
                meta["id"] = idx
                self._metadata.append(meta)
            self._next_id += len(metadata)
            self.index.add(embeddings)
            self._persist(embeddings_added=len(metadata))

    def search(self, query: np.ndarray, top_k: int = 4) -> List[Dict]:
        if self.count == 0:
            return []
        if query.ndim == 1:
            query = np.expand_dims(query, axis=0)
        if query.dtype != np.float32:
            query = query.astype("float32")
        scores, idxs = self.index.search(query, top_k)
        indices = idxs[0]
        score_values = scores[0]
        results: List[Dict] = []
        for index, score in zip(indices, score_values):
            if index == -1:
                continue
            meta = self._metadata[index].copy()
            meta["score"] = float(score)
            results.append(meta)
        return results

    def _persist(self, embeddings_added: int) -> None:
        faiss.write_index(self.index, str(self.index_path))
        # Append only the new metadata entries
        if embeddings_added <= 0:
            return
        new_records = self._metadata[-embeddings_added:]
        with self.meta_path.open("a", encoding="utf-8") as fh:
            for record in new_records:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    @property
    def count(self) -> int:
        return self.index.ntotal

    @property
    def metadata(self) -> List[Dict]:
        return self._metadata
