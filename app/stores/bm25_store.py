"""BM25 sparse store (D4, R3) using ``bm25s`` with our domain tokenizer.

Dense embeddings are good at paraphrase ("heat limit" ~ "thermal threshold")
but weak at exact identifiers; BM25 over the domain tokenizer catches
``Alpha-7-Tango`` / ``14.5`` / ``RDR_CAL_INIT`` literally. RRF fuses both.
"""

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
        # bm25s builds its index in one shot; the corpus is (re)indexed as a whole.
        self.ids = list(ids)
        self.retriever.index(token_lists, show_progress=False)

    def search(self, query_tokens: list[str], k: int) -> list[tuple[int, float]]:
        k = min(k, len(self.ids))
        if not query_tokens or k == 0:
            return []
        docs, scores = self.retriever.retrieve([query_tokens], k=k, show_progress=False)
        # bm25s pads with zero-score docs; a zero score means no term overlap at all.
        return [(self.ids[int(d)], float(s)) for d, s in zip(docs[0], scores[0]) if s > 0]

    def save(self, directory: Path) -> None:
        out = directory / SUBDIR
        self.retriever.save(str(out))
        (out / IDS_FILE).write_text(json.dumps(self.ids))

    def __len__(self) -> int:
        return len(self.ids)
