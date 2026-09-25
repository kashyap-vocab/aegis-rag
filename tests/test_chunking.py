from pathlib import Path

import pytest

from app.config import get_settings
from app.ingest.chunker import chunk_document
from app.ingest.parser import parse_file, parse_markdown
from app.ingest.tokenizer import tokenize

KB = get_settings().knowledge_base_dir


@pytest.fixture(scope="module")
def chunks_by_doc():
    return {p.name: chunk_document(parse_file(p)) for p in sorted(KB.glob("*.md"))}


def test_parser_title_sections_and_sop_number():
    doc = parse_file(KB / "SOP_002_Cooling_System.md")
    assert doc.title == "SOP 002: Main Engine Cooling System"
    assert doc.sop_number == "002"
    assert [s.heading for s in doc.sections] == ["Maintenance Schedule", "Thermal Thresholds"]


def test_parser_ignores_headings_inside_code_blocks():
    doc = parse_markdown("# T\n## A\n```\n## not a heading\n```\ntext", "x.md")
    assert [s.heading for s in doc.sections] == ["A"]


def test_every_doc_produces_chunks(chunks_by_doc):
    assert len(chunks_by_doc) == 3
    assert all(chunks for chunks in chunks_by_doc.values())


def test_chunk_ids_and_indices_are_sequential(chunks_by_doc):
    for doc_id, chunks in chunks_by_doc.items():
        assert [c.chunk_idx for c in chunks] == list(range(len(chunks)))
        assert all(c.chunk_id == f"{doc_id}::{c.chunk_idx}" for c in chunks)


def test_embed_text_carries_title_and_section(chunks_by_doc):
    for chunks in chunks_by_doc.values():
        for c in chunks:
            assert c.embed_text.startswith(f"{c.title} > {c.section}")


@pytest.mark.parametrize(
    "doc_id, fact",
    [
        ("SOP_001_Radar_Calibration.md", "between 4.5V and 4.8V"),
        ("SOP_002_Cooling_System.md", "exceeding 92°C"),
        ("SOP_002_Cooling_System.md", "Alpha-7-Tango"),
        ("SOP_003_Comms_Protocol.md", "14.5 MHz"),
    ],
)
def test_answer_facts_survive_chunking_intact(chunks_by_doc, doc_id, fact):
    assert any(fact in c.text for c in chunks_by_doc[doc_id])


def test_override_code_shares_chunk_with_throttling_context(chunks_by_doc):
    chunk = next(c for c in chunks_by_doc["SOP_002_Cooling_System.md"] if "Alpha-7-Tango" in c.text)
    assert "throttling" in chunk.text


def test_small_budget_splits_on_steps_and_keeps_continuations():
    doc = parse_file(KB / "SOP_001_Radar_Calibration.md")
    chunks = chunk_document(doc, max_tokens=40)
    step3 = next(c for c in chunks if "RDR_CAL_INIT" in c.text)
    assert "exceeds 5.0V" in step3.text
    assert len(chunks) > len(doc.sections)
    assert all(not c.text.lstrip().startswith("If voltage") for c in chunks)


def test_oversized_step_splits_on_sentences_without_orphaning_marker():
    doc = parse_file(KB / "SOP_001_Radar_Calibration.md")
    chunks = chunk_document(doc, max_tokens=12)
    texts = [c.text for c in chunks]
    assert any(t.startswith("3. Run the command") for t in texts)
    assert all(t.strip() not in {"3.", "3"} for t in texts)
    assert any("4.5V and 4.8V" in t for t in texts)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("code Alpha-7-Tango.", {"alpha-7-tango", "alpha", "7", "tango"}),
        ("band 14.5 MHz", {"14.5", "mhz"}),
        ("between 4.5V and 4.8V.", {"4.5v", "4.8v"}),
        ("exceeding 92°C", {"92°c"}),
        ("Run `RDR_CAL_INIT`", {"rdr_cal_init", "rdr", "cal", "init"}),
        ("every 5,000 hours", {"5000"}),
        ("AES-256", {"aes-256", "aes", "256"}),
    ],
)
def test_tokenizer_keeps_domain_tokens(text, expected):
    assert expected <= set(tokenize(text))


def test_tokenizer_drops_stopwords():
    assert "the" not in tokenize("What is the code")
