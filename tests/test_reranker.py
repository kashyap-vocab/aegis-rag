import numpy as np
import pytest

from app.ingest.loader import load_eval_queries
from tests.conftest import needs_models

pytestmark = needs_models


def test_scores_are_probabilities(reranker):
    scores = reranker.score("What temperature triggers throttling?", ["exceeding 92°C throttling engages", "AES-256"])
    assert scores.shape == (2,) and np.all((scores >= 0) & (scores <= 1))
    assert scores[0] > scores[1]


def test_empty_candidates(reranker):
    assert reranker.score("q", []).size == 0


def test_rerank_sorts_and_truncates(settings, retriever, reranker):
    hits = retriever.retrieve("What is the authorization code to override the automated throttling?")
    ranked = reranker.rerank("What is the authorization code to override the automated throttling?", hits)
    assert len(ranked) == min(settings.rerank_top_n, len(hits))
    assert [h.rerank_score for h in ranked] == sorted((h.rerank_score for h in ranked), reverse=True)
    assert "Alpha-7-Tango" in ranked[0].chunk.text


def test_gate_threshold_on_official_set(settings, retriever, reranker):
    tau = settings.rerank_threshold
    for q in load_eval_queries(settings.eval_csv):
        top = reranker.rerank(q.question, retriever.retrieve(q.question))[0]
        if q.is_answerable:
            assert top.rerank_score >= tau, q.question
            assert top.chunk.doc_id == q.source_doc
        elif q.query_id == "5":
            assert top.rerank_score < tau
