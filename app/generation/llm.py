import json
from typing import Protocol

from pydantic import BaseModel, ValidationError

from app.schemas import RetrievedChunk


class Generation(BaseModel):

    answerable: bool
    answer: str
    supporting_quote: str = ""
    source_doc: str = ""
    raw: str | None = None
    error: str | None = None


class LLMClient(Protocol):
    name: str

    def generate(self, question: str, context: list[RetrievedChunk]) -> Generation: ...


def parse_generation(raw: str, refusal: str) -> Generation:
    text = raw.strip()
    start, end = text.find("{"), text.rfind("}")
    try:
        data = json.loads(text[start : end + 1])
        gen = Generation.model_validate({**data, "raw": raw})
    except (ValueError, ValidationError) as exc:
        return Generation(answerable=False, answer=refusal, raw=raw, error=f"unparseable output: {exc}"[:300])
    if gen.answerable and not gen.answer.strip():
        return gen.model_copy(update={"answerable": False, "answer": refusal, "error": "empty answer"})
    return gen


def build_llm(settings, reranker=None) -> LLMClient:
    provider = settings.llm_provider
    if provider == "stub":
        from app.generation.extractive_stub import ExtractiveStub

        return ExtractiveStub(settings)
    if provider == "ollama":
        from app.generation.ollama_client import OllamaClient

        return OllamaClient(settings)
    if provider == "openai_compat":
        from app.generation.openai_compat_client import OpenAICompatClient

        return OpenAICompatClient(settings)
    if provider == "transformers":
        from app.generation.hf_client import TransformersClient

        return TransformersClient(settings)
    raise ValueError(f"unknown llm_provider {provider!r}")
