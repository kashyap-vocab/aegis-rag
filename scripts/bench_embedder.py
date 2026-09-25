import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.config import get_settings
from app.ingest.loader import load_corpus, load_eval_queries
from app.models import runtime
from app.models.embedder import Embedder

VARIANTS = ["torch-fp32", "onnx-fp32", "onnx-int8"]


def pct(xs: list[float], p: float) -> float:
    return float(np.percentile(xs, p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    args = ap.parse_args()

    s = get_settings()
    chunks = load_corpus(s.knowledge_base_dir, s.chunk_max_tokens)
    passages = [c.embed_text for c in chunks]
    queries = [q for q in load_eval_queries(s.eval_csv) if q.is_answerable]
    throughput_batch = passages * 40

    print(json.dumps(runtime.describe(s.device, s.num_threads)))
    results, reference = [], None

    for variant in VARIANTS:
        t0 = time.perf_counter()
        emb = Embedder(s, variant=variant)
        load_s = time.perf_counter() - t0

        emb.encode_passages(passages)
        t0 = time.perf_counter()
        emb.encode_passages(throughput_batch)
        pass_per_s = len(throughput_batch) / (time.perf_counter() - t0)

        p_vecs = emb.encode_passages(passages)
        q_vecs = np.vstack([emb.encode_query(q.question, use_cache=False) for q in queries])

        lat = []
        for i in range(args.runs):
            t0 = time.perf_counter()
            emb.encode_query(queries[i % len(queries)].question, use_cache=False)
            lat.append((time.perf_counter() - t0) * 1000)

        hits, rr = 0, []
        for q, qv in zip(queries, q_vecs):
            ranked = [chunks[i].doc_id for i in np.argsort(-(p_vecs @ qv))]
            hits += ranked[0] == q.source_doc
            rr.append(1 / (ranked.index(q.source_doc) + 1))

        if reference is None:
            reference = (p_vecs, q_vecs)
        fidelity = float(
            np.mean(np.concatenate([(p_vecs * reference[0]).sum(1), (q_vecs * reference[1]).sum(1)]))
        )

        results.append(
            {
                "variant": variant,
                "load_s": round(load_s, 2),
                "passages_per_s": round(pass_per_s, 1),
                "query_p50_ms": round(statistics.median(lat), 2),
                "query_p95_ms": round(pct(lat, 95), 2),
                f"hit@1 (n={len(queries)})": f"{hits}/{len(queries)}",
                "mrr": round(float(np.mean(rr)), 3),
                "fidelity_vs_fp32": round(fidelity, 4),
            }
        )
        del emb

    cols = list(results[0])
    print("\n| " + " | ".join(cols) + " |\n|" + "---|" * len(cols))
    for r in results:
        print("| " + " | ".join(str(r[c]) for c in cols) + " |")

    out = s.results_dir / "embedder.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"runtime": runtime.describe(s.device, s.num_threads), "results": results}, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
