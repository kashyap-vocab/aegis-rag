import re
from dataclasses import dataclass, field

from app.ingest.tokenizer import normalize, tokenize

_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")
_CODE = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+")
_ACRONYM = re.compile(r"\b[A-Z]{2,}[0-9]*\b")


@dataclass
class GroundingResult:
    grounded: bool
    checked: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)


def _numbers(text: str) -> set[float]:
    return {float(n) for n in _NUMBER.findall(normalize(text))}


def critical_tokens(text: str) -> dict[str, set]:
    text = normalize(text)
    return {
        "numbers": _numbers(text),
        "codes": {c.lower() for c in _CODE.findall(text) if re.search(r"\d|_", c) or c.count("-") >= 2},
        "acronyms": {a.lower() for a in _ACRONYM.findall(text)},
    }


def check_grounding(answer: str, context: str) -> GroundingResult:
    ans = critical_tokens(answer)
    ctx_numbers = _numbers(context)
    ctx_tokens = set(tokenize(context, drop_stopwords=False))

    checked, unsupported = [], []
    for n in sorted(ans["numbers"]):
        label = f"{n:g}"
        checked.append(label)
        if n not in ctx_numbers:
            unsupported.append(label)
    for tok in sorted(ans["codes"] | ans["acronyms"]):
        checked.append(tok)
        if tok not in ctx_tokens:
            unsupported.append(tok)
    return GroundingResult(grounded=not unsupported, checked=checked, unsupported=unsupported)


def _canonical(text: str) -> str:
    text = normalize(text).lower().replace("`", "")
    text = re.sub(r"[\"'“”‘’]", "", text)
    return re.sub(r"\s+", " ", text).strip(" .")


def quote_in_context(quote: str, context: str, min_chars: int = 8) -> bool:
    q = _canonical(quote)
    return len(q) >= min_chars and q in _canonical(context)
