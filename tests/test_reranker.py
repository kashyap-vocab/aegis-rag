import numpy as np
import pytest

from app.config import Settings, get_settings
from app.ingest.loader import load_eval_queries

S = get_settings()
MODELS_READY = S.local_model_dir(S.rerank_model).exists() and S.local_model_dir(S.embed_model).exists()
pytestmark = pytest.mark.skipif(not MODELS_READY, reason="run scripts/download_models.py first")


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    from app.ingest.indexer import build_index
    from app.models.embedder import Embedder
    from app.models.reranker import Reranker
    from app.retrieval.hybrid import HybridRetriever

    settings = Settings(index_dir=tmp_path_factory.mktemp("idx") / "index")
    embedder = Embedder(settings)
    build_index(settings, embedder)
    return settings, HybridRetriever.from_index_dir(settings, embedder), Reranker(settings)


def test_scores_are_probabilities(pipeline):
    _, _, reranker = pipeline
    scores = reranker.score("What temperature triggers throttling?", ["exceeding 92°C throttling engages", "AES-256"])
    assert scores.shape == (2,) and np.all((scores >= 0) & (scores <= 1))
    assert scores[0] > scores[1]


def test_empty_candidates(pipeline):
    _, _, reranker = pipeline
    assert reranker.score("q", []).size == 0


def test_rerank_sorts_and_truncates(pipeline):
    settings, retriever, reranker = pipeline
    hits = retriever.retrieve("What is the authorization code to override the automated throttling?")
    ranked = reranker.rerank("What is the authorization code to override the automated throttling?", hits)
    assert len(ranked) == min(settings.rerank_top_n, len(hits))
    assert [h.rerank_score for h in ranked] == sorted((h.rerank_score for h in ranked), reverse=True)
    assert "Alpha-7-Tango" in ranked[0].chunk.text


def test_gate_threshold_on_official_set(pipeline):
    """Calibrated τ: every official answerable query passes; trap 5 is refused at the gate."""
    settings, retriever, reranker = pipeline
    tau = settings.rerank_threshold
    for q in load_eval_queries(settings.eval_csv):
        top = reranker.rerank(q.question, retriever.retrieve(q.question))[0]
        if q.is_answerable:
            assert top.rerank_score >= tau, q.question
            assert top.chunk.doc_id == q.source_doc
        elif q.query_id == "5":
            assert top.rerank_score < tau
