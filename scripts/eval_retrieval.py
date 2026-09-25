"""Retrieval ablation (D4): dense-only vs BM25-only vs hybrid (RRF).

Reports, per query set (official / supplementary):
  hit@1, hit@3 (doc-level), MRR — answerable queries only
and a per-query table incl. trap queries, so we can see *what* each retriever
returns for questions the corpus can't answer (input for the refusal design).

    uv run scripts/eval_retrieval.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.config import get_settings
from app.ingest.loader import load_eval_queries
from app.models.embedder import Embedder
from app.retrieval.hybrid import HybridRetriever, rrf


def doc_ranking(retriever: HybridRetriever, ids: list[int]) -> list[str]:
    chunks = retriever.docstore.get(ids)
    ranked: list[str] = []
    for i in ids:
        if chunks[i].doc_id not in ranked:
            ranked.append(chunks[i].doc_id)
    return ranked


def run_modes(retriever: HybridRetriever, question: str) -> dict[str, list[str]]:
    dense = [i for i, _ in retriever.dense_search(question)]
    sparse = [i for i, _ in retriever.sparse_search(question)]
    fused = rrf({"dense": dense, "sparse": sparse}, retriever.s.rrf_k)
    hybrid = sorted(fused, key=fused.get, reverse=True)
    return {m: doc_ranking(retriever, ids) for m, ids in [("dense", dense), ("bm25", sparse), ("hybrid", hybrid)]}


def main() -> None:
    s = get_settings()
    retriever = HybridRetriever.from_index_dir(s, Embedder(s))
    report: dict = {"sets": {}, "queries": []}

    for set_name, path in [("official", s.eval_csv), ("supplementary", s.supplementary_eval_csv)]:
        queries = load_eval_queries(path)
        metrics = {m: {"hit@1": [], "hit@3": [], "rr": []} for m in ("dense", "bm25", "hybrid")}
        for q in queries:
            modes = run_modes(retriever, q.question)
            dense_top = retriever.dense_search(q.question)
            sparse_top = retriever.sparse_search(q.question)
            row = {
                "set": set_name, "id": q.query_id, "probe": q.probe, "answerable": q.is_answerable,
                "expected": q.source_doc,
                **{f"{m}_top1": (r[0] if r else None) for m, r in modes.items()},
                "dense_top_cos": round(dense_top[0][1], 3) if dense_top else None,
                "bm25_top_score": round(sparse_top[0][1], 3) if sparse_top else None,
            }
            report["queries"].append(row)
            if not q.is_answerable:
                continue
            for m, ranked in modes.items():
                pos = ranked.index(q.source_doc) + 1 if q.source_doc in ranked else None
                metrics[m]["hit@1"].append(pos == 1)
                metrics[m]["hit@3"].append(pos is not None and pos <= 3)
                metrics[m]["rr"].append(1 / pos if pos else 0.0)

        n = len(metrics["dense"]["rr"])
        report["sets"][set_name] = {
            m: {
                "n": n,
                "hit@1": f"{sum(v['hit@1'])}/{n}",
                "hit@3": f"{sum(v['hit@3'])}/{n}",
                "mrr": round(float(np.mean(v["rr"])), 3),
            }
            for m, v in metrics.items()
        }

    print("\n## Aggregate (answerable queries)\n\n| set | mode | hit@1 | hit@3 | MRR |\n|---|---|---|---|---|")
    for set_name, modes in report["sets"].items():
        for m, v in modes.items():
            print(f"| {set_name} | {m} | {v['hit@1']} | {v['hit@3']} | {v['mrr']} |")

    cols = ["set", "id", "probe", "expected", "dense_top1", "bm25_top1", "hybrid_top1", "dense_top_cos", "bm25_top_score"]
    print("\n## Per query\n\n| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for r in report["queries"]:
        cells = [str(r[c]).replace("SOP_00", "S").replace(".md", "") if r[c] is not None else "-" for c in cols]
        print("| " + " | ".join(cells) + " |")

    out = s.results_dir / "retrieval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
