import threading
import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from prometheus_client import make_asgi_app

from app.config import get_settings
from app.ingest.indexer import build_index
from app.observability import metrics
from app.observability.logging import configure_logging, get_logger
from app.pipeline import RAGPipeline
from app.retrieval.hybrid import HybridRetriever
from app.schemas import QueryRequest, QueryResponse

log = get_logger("aegis.api")
state: dict = {}
ingest_lock = threading.Lock()


def _llm_up(pipeline: RAGPipeline) -> bool:
    ping = getattr(pipeline.llm, "ping", None)
    return bool(ping()) if ping else True


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)
    t0 = time.perf_counter()
    pipeline = RAGPipeline.build(settings)
    state["pipeline"] = pipeline
    metrics.INDEX_CHUNKS.set(pipeline.retriever.docstore.count())
    metrics.LLM_UP.set(1 if _llm_up(pipeline) else 0)
    log.info("startup", load_s=round(time.perf_counter() - t0, 2), llm=pipeline.llm.name,
             tau=settings.rerank_threshold, chunks=pipeline.retriever.docstore.count())
    yield
    state.clear()


app = FastAPI(title="Operation Aegis — Offline RAG", version="1.0.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app(registry=metrics.REGISTRY))


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    structlog.contextvars.bind_contextvars(request_id=request_id)
    try:
        response = await call_next(request)
    finally:
        structlog.contextvars.clear_contextvars()
    response.headers["x-request-id"] = request_id
    return response


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, trace: bool = False) -> QueryResponse:
    pipeline: RAGPipeline = state["pipeline"]
    resp = await run_in_threadpool(pipeline.answer, req.question, True)
    fallback_used = bool(resp.trace and resp.trace.get("fallback_used"))
    metrics.record(resp, fallback_used)
    log.info(
        "query", refused=resp.refused, reason=resp.refusal_reason, gate_score=resp.gate_score,
        sources=[s.doc_id for s in resp.sources[:1]], fallback=fallback_used,
        total_ms=resp.latency_ms.get("total"), unsupported=resp.unsupported_tokens or None,
    )
    if not trace:
        resp.trace = None
    return resp


@app.get("/health")
async def health() -> dict:
    pipeline: RAGPipeline | None = state.get("pipeline")
    if pipeline is None:
        raise HTTPException(503, "pipeline not loaded")
    llm_ok = await run_in_threadpool(_llm_up, pipeline)
    metrics.LLM_UP.set(1 if llm_ok else 0)
    meta = pipeline.retriever.docstore.get_meta()
    return {
        "status": "ok" if llm_ok else "degraded",
        "llm": pipeline.llm.name, "llm_reachable": llm_ok,
        "fallback_llm": pipeline.fallback_llm.name if pipeline.fallback_llm else None,
        "index": {k: meta.get(k) for k in ("built_at", "chunks", "corpus_sha256", "embed_model", "embed_variant")},
        "tau": pipeline.s.rerank_threshold,
    }


@app.post("/ingest")
async def ingest() -> dict:
    if not ingest_lock.acquire(blocking=False):
        raise HTTPException(409, "ingest already running")
    try:
        pipeline: RAGPipeline = state["pipeline"]
        old = pipeline.retriever
        old.docstore.close()
        try:
            stats = await run_in_threadpool(build_index, pipeline.s, old.embedder)
        except Exception as exc:
            pipeline.retriever = HybridRetriever.from_index_dir(pipeline.s, old.embedder)
            log.error("ingest_failed", error=str(exc))
            raise HTTPException(500, f"ingest failed, previous index kept: {exc}") from exc
        pipeline.retriever = HybridRetriever.from_index_dir(pipeline.s, old.embedder)
        metrics.INDEX_CHUNKS.set(stats["chunks"])
        log.info("ingest", **stats)
        return stats
    finally:
        ingest_lock.release()


@app.get("/graph")
async def graph() -> dict:
    return {"mermaid": state["pipeline"].mermaid()}
