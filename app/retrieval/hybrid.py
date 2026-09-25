from pathlib import Path

from app.config import Settings
from app.ingest.tokenizer import tokenize
from app.models.embedder import Embedder
from app.retrieval.expand import expand_adjacent
from app.schemas import RetrievedChunk
from app.stores.bm25_store import BM25Store
from app.stores.docstore import DocStore
from app.stores.faiss_store import FaissStore


def rrf(rankings: dict[str, list[int]], k: int = 60) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranked_ids in rankings.values():
        for rank, doc_id in enumerate(ranked_ids, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


class IndexMismatchError(RuntimeError):
    pass


class HybridRetriever:
    def __init__(self, settings: Settings, embedder: Embedder, dense: FaissStore, sparse: BM25Store, docstore: DocStore):
        self.s = settings
        self.embedder = embedder
        self.dense = dense
        self.sparse = sparse
        self.docstore = docstore

    @classmethod
    def from_index_dir(cls, settings: Settings, embedder: Embedder, index_dir: Path | None = None) -> "HybridRetriever":
        index_dir = index_dir or settings.index_dir
        if not (index_dir / "faiss.index").exists():
            raise FileNotFoundError(f"no index at {index_dir} — run `uv run scripts/ingest.py`")
        docstore = DocStore.open(index_dir)
        meta = docstore.get_meta()
        if meta.get("embed_model") != settings.embed_model or meta.get("embed_dim") != embedder.dim:
            raise IndexMismatchError(
                f"index built with {meta.get('embed_model')} (dim {meta.get('embed_dim')}), "
                f"runtime uses {settings.embed_model} (dim {embedder.dim}) — re-run ingest"
            )
        return cls(settings, embedder, FaissStore.load(index_dir), BM25Store.load(index_dir), docstore)

    def dense_search(self, question: str) -> list[tuple[int, float]]:
        return self.dense.search(self.embedder.encode_query(question), self.s.dense_top_k)

    def sparse_search(self, question: str) -> list[tuple[int, float]]:
        return self.sparse.search(tokenize(question), self.s.sparse_top_k)

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        dense_hits = self.dense_search(question)
        sparse_hits = self.sparse_search(question)

        fused = rrf({"dense": [i for i, _ in dense_hits], "sparse": [i for i, _ in sparse_hits]}, self.s.rrf_k)
        top_ids = sorted(fused, key=fused.get, reverse=True)[: self.s.fusion_top_k]

        dense_info = {i: (r, sc) for r, (i, sc) in enumerate(dense_hits, start=1)}
        sparse_info = {i: (r, sc) for r, (i, sc) in enumerate(sparse_hits, start=1)}
        chunks = self.docstore.get(top_ids)

        hits = []
        for i in top_ids:
            d_rank, d_score = dense_info.get(i, (None, None))
            s_rank, s_score = sparse_info.get(i, (None, None))
            hits.append(
                RetrievedChunk(
                    id=i, chunk=chunks[i], rrf_score=fused[i],
                    dense_rank=d_rank, dense_score=d_score,
                    sparse_rank=s_rank, sparse_score=s_score,
                )
            )
        return expand_adjacent(hits, self.docstore, self.s.expand_neighbors)
