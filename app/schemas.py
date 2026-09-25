"""Shared data models across ingest, retrieval, generation and the API."""

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str  # f"{doc_id}::{chunk_idx}"
    doc_id: str  # source filename, e.g. SOP_002_Cooling_System.md
    sop_number: str | None = None
    title: str
    section: str  # "Thermal Thresholds"
    chunk_idx: int  # position within the doc, used for adjacent expansion
    text: str  # raw chunk body (used for grounding checks)
    embed_text: str  # "title > section\n\ntext" (what gets embedded/indexed)


class RetrievedChunk(BaseModel):
    chunk: Chunk
    rrf_score: float = 0.0
    rerank_score: float | None = None


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class Source(BaseModel):
    doc_id: str
    section: str
    rerank_score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    refused: bool
    refusal_reason: str | None = None  # "gate" | "llm" | "grounding" | "quote_check"
    sources: list[Source] = []
    latency_ms: dict[str, float] = {}
