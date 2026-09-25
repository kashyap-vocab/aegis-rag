import re
import unicodedata
from dataclasses import dataclass, field

from app.schemas import RetrievedChunk

_INJECTION_PATTERNS = [
    r"\bignore\b.{0,40}\b(previous|prior|above|all|earlier)\b.{0,20}\b(instructions?|rules?|prompts?)\b",
    r"\b(disregard|forget|override)\b.{0,40}\b(instructions?|rules?|guidelines?|system prompt)\b",
    r"\b(system|developer)\s*prompt\b",
    r"\byou are now\b|\bact as\b|\bpretend (to be|you are)\b|\bjailbreak\b|\bDAN\b",
    r"\b(reveal|print|show|repeat)\b.{0,30}\b(instructions?|prompt|context window)\b",
    r"</?\s*(system|assistant|user|document|context)\s*>",
]
_INJECTION = re.compile("|".join(f"(?:{p})" for p in _INJECTION_PATTERNS), re.IGNORECASE | re.DOTALL)
_FENCE_TAG = re.compile(r"</?\s*(document|context|system|assistant|user)\b[^>]*>", re.IGNORECASE)


@dataclass
class SanitizedQuestion:
    text: str
    valid: bool = True
    flags: list[str] = field(default_factory=list)


def sanitize_question(raw: str, max_chars: int = 1000) -> SanitizedQuestion:
    text = unicodedata.normalize("NFKC", raw)
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    text = re.sub(r"\s+", " ", text).strip()

    flags = []
    if len(text) > max_chars:
        text = text[:max_chars]
        flags.append("truncated")
    if _INJECTION.search(text):
        flags.append("injection")
    if not re.search(r"[A-Za-z0-9]", text):
        return SanitizedQuestion(text=text, valid=False, flags=flags + ["empty"])
    return SanitizedQuestion(text=text, flags=flags)


def fence_context(hits: list[RetrievedChunk]) -> str:
    blocks = []
    for h in hits:
        body = _FENCE_TAG.sub("[removed-tag]", h.chunk.text)
        blocks.append(
            f'<document source="{h.chunk.doc_id}" section="{h.chunk.section}">\n{body}\n</document>'
        )
    return "\n\n".join(blocks)
