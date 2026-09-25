import json

import httpx
import pytest

from app.config import Settings
from app.generation.llm import build_llm, parse_generation
from app.generation.ollama_client import OllamaClient
from app.generation.openai_compat_client import OpenAICompatClient
from app.generation.prompts import GENERATION_SCHEMA, build_messages
from app.schemas import Chunk, RetrievedChunk

REFUSAL = "Not found in documents."


def _ctx(text="Manual override requires authorization code Alpha-7-Tango."):
    c = Chunk(chunk_id="SOP_002_Cooling_System.md::1", doc_id="SOP_002_Cooling_System.md", title="SOP 002",
              section="Thermal Thresholds", chunk_idx=1, text=text, embed_text=text)
    return [RetrievedChunk(id=1, chunk=c, rerank_score=0.99)]


def test_parse_valid():
    g = parse_generation(json.dumps({"supporting_quote": "q", "source_doc": "d", "answerable": True, "answer": "A"}), REFUSAL)
    assert g.answerable and g.answer == "A" and g.error is None


def test_parse_tolerates_code_fences():
    raw = '```json\n{"supporting_quote": "", "source_doc": "", "answerable": false, "answer": "Not found in documents."}\n```'
    assert parse_generation(raw, REFUSAL).answerable is False


@pytest.mark.parametrize("raw", ["", "I think the answer is 42", '{"answer": "x"}', '{"answerable": "maybe", "answer": 1}'])
def test_parse_fails_closed(raw):
    g = parse_generation(raw, REFUSAL)
    assert g.answerable is False and g.answer == REFUSAL and g.error


def test_parse_empty_answer_is_refusal():
    g = parse_generation('{"supporting_quote":"","source_doc":"","answerable":true,"answer":"  "}', REFUSAL)
    assert g.answerable is False and g.error == "empty answer"


def test_messages_structure_and_fencing():
    msgs = build_messages("What is the code?", _ctx(), REFUSAL)
    assert msgs[0]["role"] == "system" and REFUSAL in msgs[0]["content"]
    assert [m["role"] for m in msgs[1:]] == ["user", "assistant", "user", "assistant", "user"]
    assert '<document source="SOP_002_Cooling_System.md"' in msgs[-1]["content"]
    assert msgs[-1]["content"].endswith("Question: What is the code?")
    for m in msgs[2::2]:
        assert set(json.loads(m["content"])) == set(GENERATION_SCHEMA["required"])


def test_schema_puts_evidence_before_answer():
    keys = list(GENERATION_SCHEMA["properties"])
    assert keys.index("supporting_quote") < keys.index("answerable") < keys.index("answer")


def test_few_shot_does_not_leak_eval_content():
    text = json.dumps(build_messages("q", [], REFUSAL)[:5]).lower()
    for term in ("mark-iv", "type-c", "alpha-7", "radar", "coolant", "throttl", "14.5"):
        assert term not in text


def _settings(**kw):
    return Settings(llm_model="m", **kw)


def test_ollama_request_and_parse():
    seen = {}

    def handler(request: httpx.Request):
        seen.update(json.loads(request.content))
        body = {"supporting_quote": "Manual override requires authorization code Alpha-7-Tango.",
                "source_doc": "SOP_002_Cooling_System.md", "answerable": True, "answer": "Alpha-7-Tango."}
        return httpx.Response(200, json={"message": {"content": json.dumps(body)}})

    client = OllamaClient(_settings())
    client.http = httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler))
    g = client.generate("code?", _ctx())
    assert g.answerable and g.answer == "Alpha-7-Tango."
    assert seen["format"] == GENERATION_SCHEMA and seen["stream"] is False
    assert seen["options"]["temperature"] == 0.0 and seen["options"]["seed"] == 42


def test_ollama_unreachable_fails_closed():
    def handler(request):
        raise httpx.ConnectError("down")

    client = OllamaClient(_settings())
    client.http = httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler))
    g = client.generate("code?", _ctx())
    assert g.answerable is False and g.answer == REFUSAL and "ollama error" in g.error


def test_openai_compat_uses_json_schema():
    seen = {}

    def handler(request: httpx.Request):
        seen.update(json.loads(request.content))
        body = {"supporting_quote": "", "source_doc": "", "answerable": False, "answer": REFUSAL}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(body)}}]})

    client = OpenAICompatClient(_settings(llm_provider="openai_compat"))
    client.http = httpx.Client(base_url="http://x", transport=httpx.MockTransport(handler))
    assert client.generate("q", _ctx()).answerable is False
    assert seen["response_format"]["json_schema"]["schema"] == GENERATION_SCHEMA


def test_build_llm_dispatch():
    assert build_llm(_settings(llm_provider="stub")).name == "extractive-stub"
    assert build_llm(_settings(llm_provider="ollama")).name == "ollama/m"
