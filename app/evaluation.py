import statistics
from dataclasses import asdict, dataclass

import numpy as np

from app.guardrails.grounding import critical_tokens
from app.ingest.loader import EvalQuery
from app.ingest.tokenizer import tokenize
from app.pipeline import RAGPipeline
from app.schemas import QueryResponse


@dataclass
class EvalRow:
    set: str
    query_id: str
    probe: str
    question: str
    expected: str
    answerable: bool
    answer: str
    refused: bool
    refusal_reason: str | None
    gate_score: float | None
    top_source: str | None
    expected_source: str | None
    retrieval_hit: bool | None
    correct: bool
    match_detail: str
    latency_ms: float


def answer_matches(expected: str, answer: str) -> tuple[bool, str]:
    exp, got = critical_tokens(expected), critical_tokens(answer)
    got_tokens = set(tokenize(answer, drop_stopwords=False))
    facts = [f"{n:g}" for n in exp["numbers"]] + sorted(exp["codes"] | exp["acronyms"])
    if facts:
        missing = [f"{n:g}" for n in exp["numbers"] if n not in got["numbers"]]
        missing += [t for t in sorted(exp["codes"] | exp["acronyms"]) if t not in got_tokens]
        return not missing, f"key facts {facts} missing {missing}" if missing else f"key facts {facts} ok"
    words = set(tokenize(expected))
    recall = len(words & got_tokens) / (len(words) or 1)
    return recall >= 0.5, f"content recall {recall:.2f}"


def evaluate_query(pipeline: RAGPipeline, q: EvalQuery, set_name: str) -> tuple[EvalRow, QueryResponse]:
    r = pipeline.answer(q.question)
    top = r.sources[0].doc_id if r.sources else None
    if q.is_answerable:
        ok, detail = (False, "wrongly refused") if r.refused else answer_matches(q.expected_answer, r.answer)
    else:
        ok, detail = r.refused, "refused" if r.refused else "answered a trap"
    row = EvalRow(
        set=set_name, query_id=q.query_id, probe=q.probe, question=q.question, expected=q.expected_answer,
        answerable=q.is_answerable, answer=r.answer, refused=r.refused, refusal_reason=r.refusal_reason,
        gate_score=None if r.gate_score is None else round(r.gate_score, 4),
        top_source=top, expected_source=q.source_doc,
        retrieval_hit=(top == q.source_doc) if q.is_answerable else None,
        correct=ok, match_detail=detail, latency_ms=r.latency_ms.get("total", 0.0),
    )
    return row, r


def summarize(rows: list[EvalRow]) -> dict:
    ans = [r for r in rows if r.answerable]
    trap = [r for r in rows if not r.answerable]
    lat = [r.latency_ms for r in rows if r.refusal_reason not in ("injection", "invalid")]

    def frac(xs) -> str:
        xs = list(xs)
        return f"{sum(xs)}/{len(xs)}"

    return {
        "n": len(rows),
        "overall_accuracy": frac(r.correct for r in rows),
        "answerable_accuracy": frac(r.correct for r in ans),
        "trap_refusal": frac(r.refused for r in trap),
        "false_refusals": frac(r.refused for r in ans),
        "retrieval_hit@1": frac(bool(r.retrieval_hit) for r in ans),
        "refusal_layers": {k: sum(r.refusal_reason == k for r in rows) for k in sorted({r.refusal_reason for r in rows if r.refusal_reason})},
        "latency_p50_ms": round(statistics.median(lat), 1) if lat else None,
        "latency_p95_ms": round(float(np.percentile(lat, 95)), 1) if lat else None,
    }


def rows_to_dicts(rows: list[EvalRow]) -> list[dict]:
    return [asdict(r) for r in rows]
