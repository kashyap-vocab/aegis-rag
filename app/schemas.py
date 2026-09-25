from pydantic import BaseModel, Field


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    sop_number: str | None = None
    title: str
    section: str
    chunk_idx: int
    text: str
    embed_text: str


class RetrievedChunk(BaseModel):
    id: int
    chunk: Chunk
    rrf_score: float = 0.0
    dense_rank: int | None = None
    dense_score: float | None = None
    sparse_rank: int | None = None
    sparse_score: float | None = None
    expanded_from: int | None = None
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
    refusal_reason: str | None = None
    supporting_quote: str | None = None
    sources: list[Source] = []
    gate_score: float | None = None
    unsupported_tokens: list[str] = []
    latency_ms: dict[str, float] = {}
    trace: dict | None = None
