import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.evaluation import evaluate_query, rows_to_dicts, summarize
from app.ingest.loader import load_eval_queries
from app.models import runtime
from app.pipeline import RAGPipeline


def main() -> None:
    s = get_settings()
    pipeline = RAGPipeline.build(s)
    tag = pipeline.llm.name.replace("/", "_").replace(":", "_")

    rows, stage_lat = [], {}
    for set_name, path in [("official", s.eval_csv), ("supplementary", s.supplementary_eval_csv)]:
        for q in load_eval_queries(path):
            row, resp = evaluate_query(pipeline, q, set_name)
            rows.append(row)
            for stage, ms in resp.latency_ms.items():
                stage_lat.setdefault(stage, []).append(ms)

    summary = {name: summarize([r for r in rows if r.set == name]) for name in ("official", "supplementary")}
    summary["all"] = summarize(rows)

    print(f"\nLLM backend: {pipeline.llm.name}   tau={s.rerank_threshold}   reranker={s.rerank_model}")
    keys = ["overall_accuracy", "answerable_accuracy", "trap_refusal", "false_refusals", "retrieval_hit@1", "latency_p50_ms", "latency_p95_ms"]
    print("\n| set | " + " | ".join(keys) + " | refusal layers |\n|" + "---|" * (len(keys) + 2))
    for name, sm in summary.items():
        print(f"| {name} | " + " | ".join(str(sm[k]) for k in keys) + f" | {sm['refusal_layers']} |")

    print("\n| set | id | answerable | refused (layer) | gate | correct | answer | detail |\n|---|---|---|---|---|---|---|---|")
    for r in rows:
        ans = r.answer if len(r.answer) <= 70 else r.answer[:67] + "..."
        print(f"| {r.set} | {r.query_id} | {r.answerable} | {r.refused} ({r.refusal_reason or '-'}) | {r.gate_score} | "
              f"{'PASS' if r.correct else 'FAIL'} | {ans} | {r.match_detail} |")

    stages = {k: round(statistics.median(v), 1) for k, v in stage_lat.items()}
    print(f"\nper-stage p50 latency (ms): {stages}")

    s.results_dir.mkdir(parents=True, exist_ok=True)
    (s.results_dir / f"eval_{tag}.json").write_text(json.dumps({
        "llm": pipeline.llm.name, "tau": s.rerank_threshold, "reranker": s.rerank_model,
        "runtime": runtime.describe(s.device, s.num_threads),
        "summary": summary, "stage_p50_ms": stages, "rows": rows_to_dicts(rows),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    with (s.results_dir / f"predictions_{tag}.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "question", "predicted_answer", "refused", "refusal_layer", "predicted_source_doc"])
        for r in rows:
            if r.set == "official":
                w.writerow([r.query_id, r.question, r.answer, r.refused, r.refusal_reason or "", r.top_source or ""])
    print(f"saved -> {s.results_dir / f'eval_{tag}.json'}")


if __name__ == "__main__":
    main()
