"""Structure-aware chunker (D6).

Strategy, in order:
1. One chunk per section (H2/H3) when it fits the token budget — sections are
   the natural semantic unit of an SOP.
2. Oversized sections are split on numbered steps, keeping unnumbered
   continuation lines attached to their step (SOP_001 step 3's "If voltage
   exceeds 5.0V ..." belongs to step 3, not to a new chunk).
3. A single step still over budget is split on sentences.
Steps/sentences are then greedily packed back up to the budget.

Every chunk is prefixed with "Title > Section" for embedding and BM25 so a
chunk like "Manual override requires authorization code Alpha-7-Tango." still
carries its "Main Engine Cooling System > Thermal Thresholds" context.
"""

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
            units[-1].append(line.strip())  # continuation of previous step
    return [" ".join(u) for u in units]


def _split_units(body: str, max_tokens: int) -> list[str]:
    units: list[str] = []
    for step in _split_steps(body):
        if _n_tokens(step) <= max_tokens:
            units.append(step)
            continue
        # Strip the "3. " marker first so the sentence splitter can't cut after it.
        marker = m.group(0) if (m := _STEP.match(step)) else ""
        sentences = [s.strip() for s in _SENTENCE.split(step[len(marker):]) if s.strip()]
        if sentences:
            sentences[0] = marker.strip() + " " + sentences[0] if marker else sentences[0]
        units.extend(sentences)
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
