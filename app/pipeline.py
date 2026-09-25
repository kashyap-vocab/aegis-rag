import time
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import Settings
from app.generation.llm import Generation, LLMClient, build_llm
from app.guardrails.gate import retrieval_gate
from app.guardrails.grounding import check_grounding, quote_in_context
from app.guardrails.sanitize import sanitize_question
from app.models.embedder import Embedder
from app.models.reranker import Reranker
from app.retrieval.hybrid import HybridRetriever
from app.schemas import QueryResponse, RetrievedChunk, Source


class RAGState(TypedDict, total=False):
    question: str
    clean_question: str
    flags: list[str]
    candidates: list[RetrievedChunk]
    ranked: list[RetrievedChunk]
    context: list[RetrievedChunk]
    gate_score: float | None
    gate_passed: bool
    generation: Generation
    llm_used: str
    fallback_used: bool
    quote_ok: bool
    grounded: bool
    unsupported: list[str]
    refusal_reason: str | None
    latency: dict[str, float]


def _grounding_context(hits: list[RetrievedChunk]) -> str:
    return "\n".join(f"{h.chunk.doc_id}\n{h.chunk.embed_text}" for h in hits)


def _hit_trace(h: RetrievedChunk) -> dict:
    return {
        "chunk_id": h.chunk.chunk_id, "section": h.chunk.section,
        "rrf": round(h.rrf_score, 5), "dense_rank": h.dense_rank, "sparse_rank": h.sparse_rank,
        "expanded_from": h.expanded_from,
        "rerank": None if h.rerank_score is None else round(h.rerank_score, 5),
    }


def _timed(state: RAGState, stage: str, t0: float) -> dict[str, float]:
    latency = dict(state.get("latency", {}))
    latency[stage] = round((time.perf_counter() - t0) * 1000, 2)
    return latency


class RAGPipeline:
    def __init__(
        self,
        settings: Settings,
        retriever: HybridRetriever,
        reranker: Reranker,
        llm: LLMClient,
        fallback_llm: LLMClient | None = None,
    ):
        self.s = settings
        self.retriever = retriever
        self.reranker = reranker
        self.llm = llm
        self.fallback_llm = fallback_llm
        self.graph = self._build_graph()

    @classmethod
    def build(cls, settings: Settings) -> "RAGPipeline":
        embedder = Embedder(settings)
        reranker = Reranker(settings)
        fallback = None
        if settings.llm_fallback_model and settings.llm_provider != "stub":
            fallback = build_llm(settings.model_copy(update={"llm_model": settings.llm_fallback_model}))
        return cls(settings, HybridRetriever.from_index_dir(settings, embedder), reranker, build_llm(settings), fallback)

    def sanitize(self, state: RAGState) -> dict[str, Any]:
        t0 = time.perf_counter()
        q = sanitize_question(state["question"], self.s.max_question_chars)
        reason = None
        if not q.valid:
            reason = "invalid"
        elif "injection" in q.flags and self.s.refuse_on_injection:
            reason = "injection"
        return {"clean_question": q.text, "flags": q.flags, "refusal_reason": reason,
                "latency": _timed(state, "sanitize", t0)}

    def retrieve(self, state: RAGState) -> dict[str, Any]:
        t0 = time.perf_counter()
        candidates = self.retriever.retrieve(state["clean_question"])
        return {"candidates": candidates, "latency": _timed(state, "retrieve", t0)}

    def rerank(self, state: RAGState) -> dict[str, Any]:
        t0 = time.perf_counter()
        ranked = self.reranker.rerank(state["clean_question"], state["candidates"])
        return {"ranked": ranked, "latency": _timed(state, "rerank", t0)}

    def gate(self, state: RAGState) -> dict[str, Any]:
        decision = retrieval_gate(state["ranked"], self.s.rerank_threshold)
        if self.s.gate_enabled:
            context = [h for h in state["ranked"] if h.rerank_score >= self.s.rerank_threshold]
            reason = None if decision.passed else "gate"
        else:
            context, reason = state["ranked"], None
        return {"gate_score": decision.top_score, "gate_passed": decision.passed,
                "context": context, "refusal_reason": reason}

    def generate(self, state: RAGState) -> dict[str, Any]:
        t0 = time.perf_counter()
        gen = self.llm.generate(state["clean_question"], state["context"])
        used, fallback_used = self.llm.name, False
        if gen.error and self.fallback_llm is not None:
            gen = self.fallback_llm.generate(state["clean_question"], state["context"])
            used, fallback_used = self.fallback_llm.name, True
        reason = None
        if gen.error:
            reason = "llm_error"
        elif not gen.answerable or gen.answer.strip().rstrip(".").lower() == self.s.refusal_message.rstrip(".").lower():
            reason = "llm"
        return {"generation": gen, "llm_used": used, "fallback_used": fallback_used,
                "refusal_reason": reason, "latency": _timed(state, "generate", t0)}

    def verify(self, state: RAGState) -> dict[str, Any]:
        t0 = time.perf_counter()
        gen, ctx = state["generation"], _grounding_context(state["context"])
        quote_ok = quote_in_context(gen.supporting_quote, ctx)
        grounding = check_grounding(gen.answer, ctx)
        reason = None
        if self.s.quote_check_enabled and not quote_ok:
            reason = "quote_check"
        elif self.s.grounding_enabled and not grounding.grounded:
            reason = "grounding"
        return {"quote_ok": quote_ok, "grounded": grounding.grounded, "unsupported": grounding.unsupported,
                "refusal_reason": reason, "latency": _timed(state, "verify", t0)}

    @staticmethod
    def _route(next_node: str):
        def route(state: RAGState) -> str:
            return END if state.get("refusal_reason") else next_node
        return route

    def _build_graph(self):
        g = StateGraph(RAGState)
        g.add_node("sanitize", self.sanitize)
        g.add_node("retrieve", self.retrieve)
        g.add_node("rerank", self.rerank)
        g.add_node("gate", self.gate)
        g.add_node("generate", self.generate)
        g.add_node("verify", self.verify)
        g.add_edge(START, "sanitize")
        g.add_conditional_edges("sanitize", self._route("retrieve"), ["retrieve", END])
        g.add_edge("retrieve", "rerank")
        g.add_edge("rerank", "gate")
        g.add_conditional_edges("gate", self._route("generate"), ["generate", END])
        g.add_conditional_edges("generate", self._route("verify"), ["verify", END])
        g.add_edge("verify", END)
        return g.compile()

    def mermaid(self) -> str:
        return self.graph.get_graph().draw_mermaid()

    def answer(self, question: str, with_trace: bool = False) -> QueryResponse:
        state: RAGState = self.graph.invoke({"question": question, "latency": {}, "refusal_reason": None})
        latency = dict(state.get("latency", {}))
        latency["total"] = round(sum(latency.values()), 2)
        ranked = state.get("ranked", [])
        sources = [Source(doc_id=h.chunk.doc_id, section=h.chunk.section, rerank_score=h.rerank_score) for h in ranked]
        gen = state.get("generation")
        reason = state.get("refusal_reason")

        trace = None
        if with_trace:
            trace = {
                "llm": state.get("llm_used", self.llm.name),
                "fallback_used": state.get("fallback_used", False),
                "question": state.get("clean_question"),
                "flags": state.get("flags", []),
                "candidates": [_hit_trace(h) for h in state.get("candidates", [])],
                "reranked": [_hit_trace(h) for h in ranked],
                "gate": {"top_score": state.get("gate_score"), "threshold": self.s.rerank_threshold,
                         "passed": state.get("gate_passed")},
                "generation": gen.model_dump() if gen else None,
                "verify": {"quote_ok": state.get("quote_ok"), "unsupported": state.get("unsupported", [])},
            }

        if reason:
            return QueryResponse(
                answer=self.s.refusal_message, refused=True, refusal_reason=reason, sources=sources,
                gate_score=state.get("gate_score"), unsupported_tokens=state.get("unsupported", []),
                latency_ms=latency, trace=trace,
            )
        if gen.source_doc:
            sources.sort(key=lambda src: src.doc_id != gen.source_doc)
        return QueryResponse(
            answer=gen.answer.strip(), refused=False, supporting_quote=gen.supporting_quote, sources=sources,
            gate_score=state.get("gate_score"), latency_ms=latency, trace=trace,
        )
