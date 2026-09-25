"""Corpus + eval-set loading shared by ingest, benchmarks and evaluation."""

import csv
from dataclasses import dataclass
from pathlib import Path

from app.ingest.chunker import chunk_document
from app.ingest.parser import parse_file
from app.schemas import Chunk


def load_corpus(kb_dir: Path, max_tokens: int = 200) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        chunks.extend(chunk_document(parse_file(path), max_tokens=max_tokens))
    return chunks


@dataclass
class EvalQuery:
    query_id: str
    question: str
    expected_answer: str
    is_answerable: bool
    source_doc: str | None
    probe: str = "official"  # supplementary set: paraphrase | exact_token | trap


def load_eval_queries(csv_path: Path) -> list[EvalQuery]:
    with csv_path.open(encoding="utf-8", newline="") as f:
        return [
            EvalQuery(
                query_id=row["query_id"],
                question=row["question"],
                expected_answer=row["expected_answer"],
                is_answerable=row["is_answerable"].strip().lower() == "true",
                source_doc=None if row["source_doc"].strip() in {"", "None"} else row["source_doc"].strip(),
                probe=row.get("probe") or "official",
            )
            for row in csv.DictReader(f)
        ]
