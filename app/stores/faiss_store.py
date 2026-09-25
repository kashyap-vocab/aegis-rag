"""FAISS dense store (D4).

``IndexFlatIP`` = exact inner-product search. With L2-normalised embeddings
this is exact cosine similarity: no approximation, no recall loss, and at this
corpus size also the fastest option. HNSW / IVF-PQ are a config switch once
the corpus reaches ~100k+ vectors.

Vectors are keyed by the SQLite docstore row id via ``IndexIDMap2``, so the
index and docstore can't drift out of alignment on re-ingest.
"""

from pathlib import Path

import faiss
import numpy as np

FILENAME = "faiss.index"


class FaissStore:
    def __init__(self, dim: int, index_type: str = "flat", hnsw_m: int = 32):
        if index_type == "flat":
            base = faiss.IndexFlatIP(dim)
        elif index_type == "hnsw":
            base = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        else:
            raise ValueError(f"unknown index_type {index_type!r}")
        self.index = faiss.IndexIDMap2(base)

    @classmethod
    def load(cls, directory: Path) -> "FaissStore":
        store = cls.__new__(cls)
        store.index = faiss.read_index(str(directory / FILENAME))
        return store

    @property
    def dim(self) -> int:
        return self.index.d

    def add(self, ids: np.ndarray, vectors: np.ndarray) -> None:
        self.index.add_with_ids(np.ascontiguousarray(vectors, dtype=np.float32), ids.astype(np.int64))

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        k = min(k, self.index.ntotal)
        if k == 0:
            return []
        scores, ids = self.index.search(np.ascontiguousarray(query.reshape(1, -1), dtype=np.float32), k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def save(self, directory: Path) -> None:
        faiss.write_index(self.index, str(directory / FILENAME))

    def __len__(self) -> int:
        return self.index.ntotal
