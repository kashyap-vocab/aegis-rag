import re

_TOKEN = re.compile(r"[a-z0-9]+(?:[-_.:°/][a-z0-9]+)*°?", re.IGNORECASE)
_PARTS = re.compile(r"[-_:/]")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")

STOPWORDS = frozenset(
    """a an and are as at be by for from has have if in into is it its of on or
    the this that to was were will with what which who whom when where why how
    does do did must should can could would during""".split()
)


def normalize(text: str) -> str:
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("º", "°")
    return _THOUSANDS.sub("", text)


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
