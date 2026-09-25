"""Markdown SOP parser (D6).

Turns a Markdown file into a ``ParsedDoc``: the H1 title plus an ordered list of
H2/H3 sections with their body text. Purely structural and deterministic — no
LLM, no network — so re-ingesting the same file always yields the same output.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_SOP_NUMBER = re.compile(r"SOP[\s_-]*(\d+)", re.IGNORECASE)


@dataclass
class Section:
    heading: str
    level: int
    body: str


@dataclass
class ParsedDoc:
    doc_id: str
    title: str
    sop_number: str | None
    sections: list[Section] = field(default_factory=list)


def _sop_number(*candidates: str) -> str | None:
    for text in candidates:
        if m := _SOP_NUMBER.search(text):
            return m.group(1).zfill(3)
    return None


def parse_markdown(text: str, doc_id: str) -> ParsedDoc:
    title = Path(doc_id).stem.replace("_", " ")
    sections: list[Section] = []
    heading, level, buf = "Preamble", 1, []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append(Section(heading=heading, level=level, body=body))

    in_code = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        m = None if in_code else _HEADING.match(line)
        if not m:
            buf.append(line)
            continue
        hashes, name = m.groups()
        if len(hashes) == 1:
            title = name  # H1 is the document title, not a section
            continue
        flush()
        heading, level, buf = name, len(hashes), []
    flush()

    return ParsedDoc(
        doc_id=doc_id,
        title=title,
        sop_number=_sop_number(title, doc_id),
        sections=sections,
    )


def parse_file(path: Path) -> ParsedDoc:
    return parse_markdown(path.read_text(encoding="utf-8"), doc_id=path.name)
