import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.config import get_settings
from app.ingest.loader import load_eval_queries
from app.models import runtime
from app.models.embedder import Embedder
from app.models.reranker import Reranker
from app.retrieval.hybrid import HybridRetriever

MODELS = ["BAAI/bge-reranker-base", "cross-encoder/ms-marco-MiniLM-L6-v2"]
VARIANTS = ["torch-fp32", "onnx-fp32", "onnx-int8"]


def auroc(pos: list[float], neg: list[float]) -> float:
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5, help="timed repetitions per query")
    args = ap.parse_args()

    s = get_settings()
    retriever = HybridRetriever.from_index_dir(s, Embedder(s))
    queries = load_eval_queries(s.eval_csv) + load_eval_queries(s.supplementary_eval_csv)
    candidates = {q.query_id: retriever.retrieve(q.question) for q in queries}

    results, per_query, reference = [], [], {}
    for model in MODELS:
        if not s.local_model_dir(model).exists():
            print(f"skip {model}: not downloaded (scripts/download_models.py --extra-rerankers {model})")
            continue
        for variant in VARIANTS:
            name = f"{model.split('/')[-1]}|{variant}"
            t0 = time.perf_counter()
            rr = Reranker(s, model=model, variant=variant)
            load_s = time.perf_counter() - t0

            rr.rerank(queries[0].question, candidates[queries[0].query_id])
            lat, hits, rrs, top_scores, all_scores = [], [], [], {}, {}
            for q in queries:
                cands = candidates[q.query_id]
                for _ in range(args.runs):
                    t0 = time.perf_counter()
                    ranked = rr.rerank(q.question, cands, top_n=len(cands))
                    lat.append((time.perf_counter() - t0) * 1000)
                top_scores[q.query_id] = ranked[0].rerank_score
                all_scores[q.query_id] = {h.id: h.rerank_score for h in ranked}
                per_query.append({
                    "model": name, "id": q.query_id, "probe": q.probe, "answerable": q.is_answerable,
                    "top_doc": ranked[0].chunk.doc_id, "top_section": ranked[0].chunk.section,
                    "top_score": round(ranked[0].rerank_score, 4),
                })
                if q.is_answerable:
                    docs = list(dict.fromkeys(h.chunk.doc_id for h in ranked))
                    pos = docs.index(q.source_doc) + 1
                    hits.append(pos == 1)
                    rrs.append(1 / pos)

            if variant == "torch-fp32":
                reference[model] = all_scores
            drift = max(
                abs(sc - reference[model][qid][cid])
                for qid, d in all_scores.items() for cid, sc in d.items()
            )
            ans = [top_scores[q.query_id] for q in queries if q.is_answerable]
            trap = [top_scores[q.query_id] for q in queries if not q.is_answerable]
            results.append({
                "model": name,
                "load_s": round(load_s, 2),
                "rerank_p50_ms": round(statistics.median(lat), 1),
                "rerank_p95_ms": round(float(np.percentile(lat, 95)), 1),
                "hit@1": f"{sum(hits)}/{len(hits)}",
                "mrr": round(float(np.mean(rrs)), 3),
                "max_drift_vs_fp32": round(drift, 4),
                "auroc_ans_vs_trap": round(auroc(ans, trap), 3),
                "min_answerable_top": round(min(ans), 4),
                "max_trap_top": round(max(trap), 4),
            })
            print(f"done {name}")
            del rr

    cols = list(results[0])
    print("\n## Aggregate\n\n| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for r in results:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")

    shown = [r["model"] for r in results if "onnx" in r["model"]]
    table = {}
    for row in per_query:
        if row["model"] in shown:
            table.setdefault((row["id"], row["probe"], row["answerable"]), {})[row["model"]] = row["top_score"]
    print("\n## Top rerank score per query\n\n| id | probe | answerable | " + " | ".join(shown) + " |\n|" + "---|" * (3 + len(shown)))
    for (qid, probe, answerable), sc in table.items():
        print(f"| {qid} | {probe} | {answerable} | " + " | ".join(str(sc.get(m, '-')) for m in shown) + " |")

    out = s.results_dir / "reranker.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"runtime": runtime.describe(s.device, s.num_threads), "results": results, "per_query": per_query}, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
