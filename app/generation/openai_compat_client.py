import httpx

from app.config import Settings
from app.generation.llm import Generation, parse_generation
from app.generation.prompts import GENERATION_SCHEMA, build_messages
from app.schemas import RetrievedChunk


class OpenAICompatClient:
    def __init__(self, settings: Settings):
        self.s = settings
        self.name = f"openai_compat/{settings.llm_model}"
        self.http = httpx.Client(
            base_url=settings.llm_base_url.rstrip("/"),
            timeout=settings.llm_timeout_s,
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
        )

    def generate(self, question: str, context: list[RetrievedChunk]) -> Generation:
        payload = {
            "model": self.s.llm_model,
            "messages": build_messages(question, context, self.s.refusal_message),
            "temperature": self.s.llm_temperature,
            "seed": self.s.llm_seed,
            "max_tokens": self.s.llm_max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "sop_answer", "schema": GENERATION_SCHEMA, "strict": True},
            },
        }
        try:
            resp = self.http.post("/chat/completions", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            return Generation(answerable=False, answer=self.s.refusal_message, error=f"llm server error: {exc}")
        return parse_generation(resp.json()["choices"][0]["message"]["content"], self.s.refusal_message)
