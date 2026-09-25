from dataclasses import dataclass

from app.schemas import RetrievedChunk


@dataclass
class GateDecision:
    passed: bool
    top_score: float | None
    threshold: float


def retrieval_gate(ranked: list[RetrievedChunk], threshold: float) -> GateDecision:
    if not ranked or ranked[0].rerank_score is None:
        return GateDecision(passed=False, top_score=None, threshold=threshold)
    top = ranked[0].rerank_score
    return GateDecision(passed=top >= threshold, top_score=top, threshold=threshold)
