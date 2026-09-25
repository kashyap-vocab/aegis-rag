import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.evaluation import evaluate_query, rows_to_dicts
from app.ingest.loader import load_eval_queries
from app.pipeline import RAGPipeline


def frac(xs) -> str:
    xs = list(xs)
    return f"{sum(xs)}/{len(xs)}" if xs else "-"


def pct(xs) -> str:
    xs = list(xs)
    return f"{100 * sum(xs) / len(xs):.0f}%" if xs else "-"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="eval/extended_queries.csv")
    args = ap.parse_args()

    s = get_settings()
    pipeline = RAGPipeline.build(s)
    tag = pipeline.llm.name.replace("/", "_").replace(":", "_")
    queries = load_eval_queries(Path(args.csv))

    rows = []
    for i, q in enumerate(queries, 1):
        row, _ = evaluate_query(pipeline, q, "extended")
        rows.append(row)
        print(f"[{i:3d}/{len(queries)}] {q.query_id} {'PASS' if row.correct else 'FAIL'} "
              f"({row.refusal_reason or 'answered'}) {row.answer[:60]!r}", flush=True)

    ans = [r for r in rows if r.answerable]
    una = [r for r in rows if not r.answerable]
    lat = [r.latency_ms for r in rows if r.refusal_reason not in ("injection", "invalid")]
    summary = {
        "llm": pipeline.llm.name,
        "n": len(rows),
        "overall_accuracy": frac(r.correct for r in rows),
        "answerable_accuracy": frac(r.correct for r in ans),
        "unanswerable_refused": frac(r.refused for r in una),
        "false_refusals": frac(r.refused for r in ans),
        "hallucinated_answers_to_traps": frac(not r.refused for r in una),
        "retrieval_hit@1": frac(bool(r.retrieval_hit) for r in ans),
        "refusal_layers": {k: sum(r.refusal_reason == k for r in rows) for k in sorted({r.refusal_reason for r in rows if r.refusal_reason})},
        "latency_p50_ms": round(statistics.median(lat), 1) if lat else None,
    }
    by_probe = defaultdict(list)
    for r in rows:
        by_probe[(r.answerable, r.probe)].append(r)
    per_category = [
        {"answerable": a, "category": p, "n": len(rs), "correct": frac(r.correct for r in rs), "accuracy": pct(r.correct for r in rs),
         "refused_by": {k: sum(r.refusal_reason == k for r in rs) for k in sorted({r.refusal_reason for r in rs if r.refusal_reason})}}
        for (a, p), rs in sorted(by_probe.items(), key=lambda kv: (not kv[0][0], kv[0][1]))
    ]

    print("\n" + json.dumps(summary, indent=1))
    print("\n| answerable | category | n | correct | accuracy | refused by |\n|---|---|---|---|---|---|")
    for c in per_category:
        print(f"| {c['answerable']} | {c['category']} | {c['n']} | {c['correct']} | {c['accuracy']} | {c['refused_by'] or '-'} |")
    print("\nFailures:")
    for r in rows:
        if not r.correct:
            print(f"  {r.query_id} [{r.probe}] {r.question!r} -> {r.answer!r} ({r.refusal_reason or 'answered'}; {r.match_detail})")

    out = s.results_dir / f"extended_{tag}.json"
    out.write_text(json.dumps({"summary": summary, "per_category": per_category, "rows": rows_to_dicts(rows)},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
