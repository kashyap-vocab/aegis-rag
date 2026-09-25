from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from app.schemas import QueryResponse

REGISTRY = CollectorRegistry()

REQUESTS = Counter(
    "aegis_requests_total", "Questions answered or refused", ["outcome", "reason"], registry=REGISTRY
)
STAGE_LATENCY = Histogram(
    "aegis_stage_latency_seconds", "Latency per pipeline stage", ["stage"], registry=REGISTRY,
    buckets=(0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 60),
)
GATE_SCORE = Histogram(
    "aegis_gate_top_rerank_score", "Top reranker score seen by the retrieval gate", registry=REGISTRY,
    buckets=(0.001, 0.0025, 0.005, 0.0075, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
)
GROUNDING_FAILURES = Counter(
    "aegis_grounding_failures_total", "Answers blocked by quote check or grounding verifier", ["layer"],
    registry=REGISTRY,
)
LLM_FALLBACKS = Counter("aegis_llm_fallback_total", "Generations served by the fallback model", registry=REGISTRY)
LLM_UP = Gauge("aegis_llm_up", "1 if the primary LLM backend is reachable", registry=REGISTRY)
INDEX_CHUNKS = Gauge("aegis_index_chunks", "Chunks in the live index", registry=REGISTRY)


def record(resp: QueryResponse, fallback_used: bool = False) -> None:
    REQUESTS.labels("refused" if resp.refused else "answered", resp.refusal_reason or "none").inc()
    for stage, ms in resp.latency_ms.items():
        STAGE_LATENCY.labels(stage).observe(ms / 1000)
    if resp.gate_score is not None:
        GATE_SCORE.observe(resp.gate_score)
    if resp.refusal_reason in ("quote_check", "grounding"):
        GROUNDING_FAILURES.labels(resp.refusal_reason).inc()
    if fallback_used:
        LLM_FALLBACKS.inc()
