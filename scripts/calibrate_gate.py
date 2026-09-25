"""Sweep the retrieval-gate threshold τ over saved reranker scores (D8).

Reads docs/results/reranker.json (from bench_reranker.py) — no models needed.
For each τ: how many answerable queries would be wrongly refused (unrecoverable:
the LLM never sees them) vs. how many traps are refused before the LLM (a
trap that passes still meets the downstream grounding layers).

    uv run scripts/calibrate_gate.py [--model "bge-reranker-base|onnx-fp32"]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings

GRID = [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bge-reranker-base|onnx-fp32")
    args = ap.parse_args()

    s = get_settings()
    data = json.loads((s.results_dir / "reranker.json").read_text())
    rows = [r for r in data["per_query"] if r["model"] == args.model]
    ans = sorted(((r["top_score"], r["id"]) for r in rows if r["answerable"]))
    trap = sorted(((r["top_score"], r["id"]) for r in rows if not r["answerable"]))

    print(f"model: {args.model}   answerable n={len(ans)}   traps n={len(trap)}")
    print(f"lowest answerable: {ans[:3]}")
    print(f"traps: {trap}\n")
    print("| tau | answerable wrongly refused | traps refused at gate | trap ids refused |\n|---|---|---|---|")
    sweep = []
    for tau in GRID:
        fr = [i for sc, i in ans if sc < tau]
        tr = [i for sc, i in trap if sc < tau]
        sweep.append({"tau": tau, "false_refusals": fr, "traps_refused": tr})
        print(f"| {tau} | {len(fr)}/{len(ans)} {fr or ''} | {len(tr)}/{len(trap)} | {', '.join(tr)} |")

    out = s.results_dir / "gate_calibration.json"
    out.write_text(json.dumps({"model": args.model, "sweep": sweep}, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
