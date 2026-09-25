import httpx

from app.config import Settings
from app.generation.llm import Generation, parse_generation
from app.generation.prompts import GENERATION_SCHEMA, build_messages
from app.schemas import RetrievedChunk


class OllamaClient:
    def __init__(self, settings: Settings):
        self.s = settings
        self.name = f"ollama/{settings.llm_model}"
        self.http = httpx.Client(base_url=settings.llm_base_url, timeout=settings.llm_timeout_s)

    def ping(self) -> bool:
        try:
            tags = self.http.get("/api/tags").json().get("models", [])
        except httpx.HTTPError:
            return False
        return any(m.get("name") == self.s.llm_model for m in tags)

    def generate(self, question: str, context: list[RetrievedChunk]) -> Generation:
        payload = {
            "model": self.s.llm_model,
            "messages": build_messages(question, context, self.s.refusal_message),
            "format": GENERATION_SCHEMA,
            "stream": False,
            "keep_alive": self.s.llm_keep_alive,
            "options": {
                "temperature": self.s.llm_temperature,
                "seed": self.s.llm_seed,
                "num_ctx": self.s.llm_num_ctx,
                "num_predict": self.s.llm_max_tokens,
            },
        }
        try:
            resp = self.http.post("/api/chat", json=payload)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            return Generation(answerable=False, answer=self.s.refusal_message, error=f"ollama error: {exc}")
        return parse_generation(resp.json()["message"]["content"], self.s.refusal_message)
