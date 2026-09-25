import json
from pathlib import Path

import bm25s

SUBDIR = "bm25"
IDS_FILE = "ids.json"


class BM25Store:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.retriever = bm25s.BM25(k1=k1, b=b)
        self.ids: list[int] = []

    @classmethod
    def load(cls, directory: Path) -> "BM25Store":
        store = cls.__new__(cls)
        store.retriever = bm25s.BM25.load(str(directory / SUBDIR))
        store.ids = json.loads((directory / SUBDIR / IDS_FILE).read_text())
        return store

    def add(self, ids: list[int], token_lists: list[list[str]]) -> None:
        self.ids = list(ids)
        self.retriever.index(token_lists, show_progress=False)

    def search(self, query_tokens: list[str], k: int) -> list[tuple[int, float]]:
        k = min(k, len(self.ids))
        if not query_tokens or k == 0:
            return []
        docs, scores = self.retriever.retrieve([query_tokens], k=k, show_progress=False, backend_selection="numpy")
        return [(self.ids[int(d)], float(s)) for d, s in zip(docs[0], scores[0]) if s > 0]

    def save(self, directory: Path) -> None:
        out = directory / SUBDIR
        self.retriever.save(str(out))
        (out / IDS_FILE).write_text(json.dumps(self.ids))

    def __len__(self) -> int:
        return len(self.ids)
