import pytest

from app.config import Settings, get_settings
from app.ingest.loader import load_eval_queries
from app.retrieval.expand import expand_adjacent
from app.retrieval.hybrid import HybridRetriever, IndexMismatchError, rrf
from app.schemas import Chunk, RetrievedChunk

MODELS_READY = get_settings().local_model_dir(get_settings().embed_model).exists()
needs_models = pytest.mark.skipif(not MODELS_READY, reason="run scripts/download_models.py first")


# --- pure logic (no models) --------------------------------------------------

def test_rrf_rewards_agreement_between_retrievers():
    fused = rrf({"dense": [1, 2, 3], "sparse": [2, 1, 4]}, k=60)
    assert fused[1] == pytest.approx(fused[2])  # symmetric ranks -> equal
    assert fused[1] > fused[3] and fused[1] > fused[4]  # in both lists beats in one
    assert fused[3] == pytest.approx(1 / 63)


def test_rrf_single_list_preserves_order():
    fused = rrf({"dense": [7, 8, 9]})
    assert sorted(fused, key=fused.get, reverse=True) == [7, 8, 9]


class _FakeDocstore:
    def __init__(self, chunks):
        self.chunks = chunks  # id -> Chunk

    def neighbors(self, doc_id, idx, n):
        return {i: c for i, c in self.chunks.items() if c.doc_id == doc_id and 0 < abs(c.chunk_idx - idx) <= n}


def _chunk(doc, idx):
    return Chunk(chunk_id=f"{doc}::{idx}", doc_id=doc, title="t", section="s", chunk_idx=idx, text="x", embed_text="x")


def test_expand_adds_same_doc_neighbours_once():
    store = _FakeDocstore({1: _chunk("A", 0), 2: _chunk("A", 1), 3: _chunk("A", 2), 4: _chunk("B", 1)})
    hits = [RetrievedChunk(id=2, chunk=store.chunks[2]), RetrievedChunk(id=3, chunk=store.chunks[3])]
    out = expand_adjacent(hits, store, n=1)
    assert [h.id for h in out] == [2, 3, 1]  # chunk 1 added once, B never
    assert out[-1].expanded_from == 2


def test_expand_disabled():
    hits = [RetrievedChunk(id=1, chunk=_chunk("A", 0))]
    assert expand_adjacent(hits, None, n=0) == hits


# --- index + retrieval (needs local models) ------------------------------------

@pytest.fixture(scope="module")
def built(tmp_path_factory):
    from app.ingest.indexer import build_index
    from app.models.embedder import Embedder

    settings = Settings(index_dir=tmp_path_factory.mktemp("idx") / "index")
    embedder = Embedder(settings)
    stats = build_index(settings, embedder)
    return settings, embedder, stats


@needs_models
def test_index_artifacts_are_aligned(built):
    settings, embedder, stats = built
    r = HybridRetriever.from_index_dir(settings, embedder)
    assert len(r.dense) == len(r.sparse) == r.docstore.count() == stats["chunks"] == 5
    assert r.docstore.get_meta()["embed_model"] == settings.embed_model


@needs_models
def test_hybrid_top1_matches_source_doc_for_official_answerable(built):
    settings, embedder, _ = built
    r = HybridRetriever.from_index_dir(settings, embedder)
    for q in load_eval_queries(settings.eval_csv):
        if q.is_answerable:
            assert r.retrieve(q.question)[0].chunk.doc_id == q.source_doc, q.question


@needs_models
def test_bm25_matches_exact_code(built):
    settings, embedder, _ = built
    r = HybridRetriever.from_index_dir(settings, embedder)
    top_id, _ = r.sparse_search("Alpha-7-Tango")[0]
    assert "Alpha-7-Tango" in r.docstore.get([top_id])[top_id].text


@needs_models
def test_mismatched_embedder_is_rejected(built):
    settings, embedder, _ = built
    other = settings.model_copy(update={"embed_model": "BAAI/bge-base-en-v1.5"})
    with pytest.raises(IndexMismatchError):
        HybridRetriever.from_index_dir(other, embedder)
