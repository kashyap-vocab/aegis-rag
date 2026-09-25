"""Domain-aware tokenizer for BM25 and grounding checks (R3).

Generic tokenizers split ``Alpha-7-Tango`` into ``alpha / 7 / tango`` and
``4.5V`` into ``4 / 5v`` — destroying exactly the tokens our answers depend on.
This tokenizer keeps compound identifiers whole *and* also emits their parts,
so both "Alpha-7-Tango" and "tango" match.
"""

import re

# A compound token: alphanumerics joined by - _ . : ° / (e.g. Alpha-7-Tango,
# 14.5, RDR_CAL_INIT, 92°C, AES-256, RS-232, 00:00).
_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.:°/][a-z0-9]+)*°?", re.IGNORECASE)
_PARTS = re.compile(r"[-_:/]")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")

STOPWORDS = frozenset(
    """a an and are as at be by for from has have if in into is it its of on or
    the this that to was were will with what which who whom when where why how
    does do did must should can could would during""".split()
)


def normalize(text: str) -> str:
    """Canonical form used before tokenizing (and by grounding checks)."""
    text = text.replace("–", "-").replace("—", "-")  # en/em dash
    text = text.replace("º", "°")  # masculine ordinal often mistyped for degree
    return _THOUSANDS.sub("", text)  # 5,000 -> 5000


def tokenize(text: str, *, drop_stopwords: bool = True, expand_parts: bool = True) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN.finditer(normalize(text)):
        tok = match.group(0).lower()
        if drop_stopwords and tok in STOPWORDS:
            continue
        tokens.append(tok)
        if expand_parts and _PARTS.search(tok):
            tokens.extend(p for p in _PARTS.split(tok) if p and p not in STOPWORDS)
    return tokens
