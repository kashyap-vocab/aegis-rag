import re
from collections.abc import Iterable

from app.ingest.parser import ParsedDoc
from app.ingest.tokenizer import tokenize
from app.schemas import Chunk

_STEP = re.compile(r"^\s*(?:\d+[.)]|[-*+])\s+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9`])")


def _n_tokens(text: str) -> int:
    return len(tokenize(text, drop_stopwords=False, expand_parts=False))


def _split_steps(body: str) -> list[str]:
    units: list[list[str]] = []
    for line in body.splitlines():
        if not line.strip():
            continue
        if _STEP.match(line) or not units:
            units.append([line.strip()])
        else:
            units[-1].append(line.strip())
    return [" ".join(u) for u in units]


def _split_step_sentences(step: str) -> list[str]:
    marker = m.group(0) if (m := _STEP.match(step)) else ""
    sentences = [s.strip() for s in _SENTENCE.split(step[len(marker):]) if s.strip()]
    if sentences and marker:
        sentences[0] = marker.strip() + " " + sentences[0]
    return sentences


def split_sentences(body: str) -> list[str]:
    return [s for step in _split_steps(body) for s in _split_step_sentences(step)]


def _split_units(body: str, max_tokens: int) -> list[str]:
    units: list[str] = []
    for step in _split_steps(body):
        if _n_tokens(step) <= max_tokens:
            units.append(step)
        else:
            units.extend(_split_step_sentences(step))
    return units


def _pack(units: Iterable[str], max_tokens: int, joiner: str = "\n") -> list[str]:
    packs, cur, cur_len = [], [], 0
    for unit in units:
        n = _n_tokens(unit)
        if cur and cur_len + n > max_tokens:
            packs.append(joiner.join(cur))
            cur, cur_len = [], 0
        cur.append(unit)
        cur_len += n
    if cur:
        packs.append(joiner.join(cur))
    return packs


def chunk_document(doc: ParsedDoc, max_tokens: int = 200) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in doc.sections:
        if _n_tokens(section.body) <= max_tokens:
            bodies = [section.body]
        else:
            bodies = _pack(_split_units(section.body, max_tokens), max_tokens)

        for body in bodies:
            idx = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::{idx}",
                    doc_id=doc.doc_id,
                    sop_number=doc.sop_number,
                    title=doc.title,
                    section=section.heading,
                    chunk_idx=idx,
                    text=body,
                    embed_text=f"{doc.title} > {section.heading}\n\n{body}",
                )
            )
    return chunks
