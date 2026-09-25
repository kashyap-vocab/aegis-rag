import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.evaluation import evaluate_query, rows_to_dicts, summarize
from app.ingest.loader import load_eval_queries
from app.pipeline import RAGPipeline

CONFIGS = {
    "full (L1+L2+L3+L4)": {},
    "no gate (L1 off)": {"gate_enabled": False},
    "no quote check (L3 off)": {"quote_check_enabled": False},
    "no grounding (L4 off)": {"grounding_enabled": False},
    "LLM only (L1, L3, L4 off)": {"gate_enabled": False, "quote_check_enabled": False, "grounding_enabled": False},
}


class MemoLLM:
    def __init__(self, llm):
        self.llm, self.name, self.cache, self.calls = llm, llm.name, {}, 0

    def generate(self, question, context):
        key = (question, tuple(h.id for h in context))
        if key not in self.cache:
            self.calls += 1
            self.cache[key] = self.llm.generate(question, context)
        return self.cache[key]


def main() -> None:
    base = get_settings()
    built = RAGPipeline.build(base)
    llm = MemoLLM(built.llm)
    queries = [("official", q) for q in load_eval_queries(base.eval_csv)]
    queries += [("supplementary", q) for q in load_eval_queries(base.supplementary_eval_csv)]

    report = {"llm": llm.name, "configs": {}}
    for name, overrides in CONFIGS.items():
        settings = base.model_copy(update=overrides)
        pipe = RAGPipeline(settings, built.retriever, built.reranker, llm)
        rows = [evaluate_query(pipe, q, set_name)[0] for set_name, q in queries]
        report["configs"][name] = {
            "official": summarize([r for r in rows if r.set == "official"]),
            "all": summarize(rows),
            "rows": rows_to_dicts(rows),
        }
        print(f"done: {name}  (distinct LLM calls so far: {llm.calls})")

    keys = ["overall_accuracy", "answerable_accuracy", "trap_refusal", "false_refusals"]
    print(f"\nLLM: {llm.name}\n\n| config | set | " + " | ".join(keys) + " | refusal layers |\n|" + "---|" * (len(keys) + 3))
    for name, res in report["configs"].items():
        for set_name in ("official", "all"):
            sm = res[set_name]
            print(f"| {name} | {set_name} | " + " | ".join(str(sm[k]) for k in keys) + f" | {sm['refusal_layers']} |")

    tag = llm.name.replace("/", "_").replace(":", "_")
    out = base.results_dir / f"ablation_{tag}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
