# Operation Aegis — Build Log

Running record of **what** was built at each stage, **why** (decision + requirement
it serves), **results/evidence**, and **issues found**. Source material for the
Kaggle writeup (`docs/WRITEUP.md`).

Requirements referenced below:
- **R1** Offline / air-gapped — no network at runtime; all artifacts packageable
- **R2** Hardware-agnostic — CPU baseline, GPU used automatically if present
- **R3** Exact-token accuracy — answers are codes/numbers (`Alpha-7-Tango`, `14.5 MHz`, `92°C`)
- **R4** Safety — confident refusal when the answer is absent
- **R5** Operational simplicity — few moving parts, auditable

Client evaluation focus: architectural choices, chunking logic, hallucination
prevention on adversarial queries, overall system design. Submission: Kaggle
notebook (internet OFF, CPU and GPU both configurable) + separate Writeup.

---

## Step 0 — Architecture review (2026-09-25)

**Input:** a reference diagram (Gemini-suggested, from "Designing a Production-Grade RAG
Architecture", Level Up Coding): Qdrant, bge-small, TF-IDF, RRF, adjacent chunks,
bge-reranker, LLM-generated metadata, orchestrator + tool plugin.

| Reference component | Decision | Why |
|---|---|---|
| Heading-aware extraction/chunking | Kept | SOP sections are the natural semantic unit |
| LLM "Generate Metadata" | Replaced by rule-based metadata | Deterministic, repeatable, no LLM at ingest (R1, R5) |
| TF-IDF sparse | Replaced by BM25 (`bm25s`) with a domain tokenizer | Exact-token matching of codes/numbers (R3) |
| Qdrant | Replaced by FAISS + SQLite docstore | In-process library, no server to secure; index is a single shippable file (R1, R5) |
| RRF fusion, adjacent chunks, cross-encoder reranker | Kept | Cheap, proven; reranker score doubles as refusal signal (R4) |
| Orchestrator + tool plugin | Replaced by a linear pipeline | Fixed QA flow; agent loops add latency, nondeterminism, hallucination surface (R4, R5) |
| *(missing)* Refusal/guardrail layer | Added | Core scoring criterion (R4) |
| *(missing)* Evaluation harness, citations, prompt sanitisation | Added | Evidence for every claim; provenance; injection hygiene |

**Key insight:** both trap queries retrieve the *correct* document ("Mark-IV Radar" →
SOP_001, "Type-C Marine coolant" → SOP_002). A retrieval-score threshold alone cannot
refuse them → refusal must be layered: retrieval gate → constrained generation →
grounding verifier → verbatim quote check.

**Other decisions:**
- **Frameworks:** no LangChain/LangGraph/LlamaIndex — linear ~6-stage pipeline doesn't need graph/agent state; own RRF/gate/grounding code is auditable; far smaller dependency tree for an air-gapped wheelhouse; no hidden network calls or LLM calls (R1, R5).
- **Hardware route:** CPU-first (dataset says "edge-optimized"; judges score design not speed; reproducible on any machine), GPU as config switch (`AEGIS_LLM_DEVICE`, ONNX provider auto-select).
- **LLM:** Qwen2.5-3B-Instruct (1.5B fallback). Ollama for Docker deployment; HF `transformers` backend for the Kaggle notebook; OpenAI-compatible client for vLLM on GPU servers — all behind one `LLMClient` interface.
- **Observability:** OpenTelemetry → Arize Phoenix (1 container, vs. Langfuse v3 needing Postgres+ClickHouse+Redis+MinIO), Prometheus + Grafana, structlog JSON logs. Optional compose profile.
- **Packaging:** `uv` (lockfile with hashes, offline sync); FastAPI; multi-stage Docker with models+index baked in, verified with `--network none`.

---

## Step 1 — Project scaffold

**What**
- `uv` project, Python 3.12 pinned, `uv.lock` for reproducible installs.
- Deps: `sentence-transformers[onnx]`, `faiss-cpu`, `bm25s`, `fastapi`, `uvicorn`, `pydantic-settings`, `pandas`, `structlog`; dev: `pytest`, `httpx`.
- `torch` pinned to the **CPU-only wheel index**.
- `app/config.py` — every tunable in one place, overridable via `AEGIS_*` env vars.
- `app/models/runtime.py` — detects ONNX Runtime execution providers at runtime: CUDA → DirectML → OpenVINO → CPU; threads default to physical cores.
- `app/schemas.py` — shared `Chunk` / request / response models.

**Why**
- uv lockfile + offline sync → reproducible air-gapped installs (R1). Faster CI/Docker builds.
- CPU torch index: default Linux torch pulls ~2 GB of CUDA libs we never use (R1 image size, R5).
- Config-driven + runtime provider detection → no hardware assumptions; moving to GPU is config, not code (R2).

**Results**
- Dev machine detection: `available=[AzureExecutionProvider, CPUExecutionProvider]`, `selected=[CPUExecutionProvider]`, `threads=6`.

**Issues**
- `.gitignore` pattern `models/` also ignored the code folder `app/models/` → fixed by anchoring (`/models/`, `/data/`).
- `pyproject.toml` sections were lost by a stale editor save; restored (CPU torch index, dev group, pytest config).

---

## Step 2 — Parsing, chunking, tokenizer

**What**
- `app/ingest/parser.py` — Markdown → title (H1), SOP number, ordered H2/H3 sections. Ignores `#` inside code fences.
- `app/ingest/chunker.py` — hierarchy: one chunk per section if within budget (200 tokens) → else split on numbered steps (unnumbered continuation lines stay with their step) → else split on sentences → greedy re-pack up to budget. Each chunk's `embed_text` is prefixed `Title > Section`.
- `app/ingest/tokenizer.py` — domain tokenizer for BM25 and grounding: keeps compound tokens whole **and** emits their parts; normalises `5,000 → 5000`, en/em dashes, `º → °`; small stopword list.
- Metadata per chunk: `chunk_id`, `doc_id`, `sop_number`, `title`, `section`, `chunk_idx` (for adjacent expansion).

**Why**
- SOP sections are self-contained semantic units; splitting mid-section fragments facts from their context (e.g. the override code must stay with "throttling" for query 4).
- Step-aware fallback future-proofs longer SOPs without breaking procedures apart.
- `Title > Section` prefix gives short chunks their context for both dense and BM25 retrieval.
- Generic tokenizers destroy `Alpha-7-Tango` / `4.5V` / `RDR_CAL_INIT` → BM25 couldn't match exact codes (R3).
- Rule-based metadata → deterministic re-ingest, no LLM (R1, R5).

**Results**
- 3 SOPs → **5 chunks** (one per section): SOP_001 Overview, SOP_001 Procedure, SOP_002 Maintenance Schedule, SOP_002 Thermal Thresholds, SOP_003 Protocol.
- All 4 answer facts (`4.5V and 4.8V`, `92°C`, `Alpha-7-Tango`, `14.5 MHz`) survive intact in a single chunk.
- Tokenizer examples: `Alpha-7-Tango → alpha-7-tango, alpha, 7, tango`; `4.8V. → 4.8v`; `92°C → 92°c`; `5,000 → 5000`.
- **20/20 unit tests pass** (`tests/test_chunking.py`).

**Issues**
- Bug caught by tests: sentence splitter cut after step marker `3.` producing an orphan chunk. Fix: strip marker, split sentences, re-attach marker. Regression test added.

---

## Step 3 — Model download + optimised embedder

**What**
- `scripts/download_models.py` — the **only networked step**. Downloads `BAAI/bge-small-en-v1.5` and `BAAI/bge-reranker-base` to `./models`, exports ONNX **from the safetensors weights ourselves** (not the repo's prebuilt ONNX), builds int8 dynamic-quantized variants (`avx2`), writes `models/MANIFEST.json` (SHA-256 of every file) for integrity verification after air-gap transfer.
- `app/models/embedder.py` — loads strictly from local files; variants `torch-fp32 | onnx-fp32 | onnx-int8`; ONNX session with `ORT_ENABLE_ALL` graph optimisation, configured threads and selected provider; bge query prefix on queries only; normalised outputs (cosine = inner product); `max_seq_length=256`; LRU query cache.
- `scripts/bench_embedder.py` → `docs/results/embedder.json`.

**Why**
- **bge-small-en-v1.5:** strong MTEB retrieval for its size (33M params, 384-d), fast on CPU (R2); bge-base/large is a one-line config swap.
- **ONNX Runtime:** removes PyTorch eager overhead, same runtime on CPU/CUDA/DirectML/OpenVINO (R2).
- **int8 quantization:** smaller + faster on CPU; adopted **only if accuracy holds** (evidence-gated).
- `quant_config` configurable (`avx2` portable to any x86-64 since ~2013; `avx512_vnni` newer Xeons; `arm64` for ARM edge boxes).
- Own ONNX export → reproducible artifact from pinned tool versions.
- Manifest → tamper/corruption detection on the air-gapped side (R1).

**Results** (dev machine: CPU only, 6 threads, fully offline `HF_HUB_OFFLINE=1`)

| Variant | Load (s) | Passages/s | Query p50 (ms) | Query p95 (ms) | hit@1 (n=4) | MRR | Fidelity vs fp32 |
|---|---|---|---|---|---|---|---|
| torch-fp32 | 0.24 | 44.0 | 33.81 | 42.74 | 4/4 | 1.000 | 1.0000 |
| onnx-fp32 | 0.93 | 50.0 | 12.82 | 14.42 | 4/4 | 1.000 | 1.0000 |
| **onnx-int8** | 0.44 | **58.7** | **12.53** | **13.97** | 4/4 | 1.000 | 0.9965 |

**Decision:** `onnx-int8` is the default.
- ~2.7× lower query latency than torch-fp32, identical retrieval (4/4, MRR 1.0).
- The ONNX switch gives most of the latency gain; int8 adds ~17% passage throughput and a ~4× smaller model file (matters for edge/air-gap transfer).
- hit@1 saturates on a 5-chunk corpus, so **fidelity** (mean cosine to fp32 embeddings) is reported: 0.9965 → negligible quantization drift.

**Issues / notes**
- Quantized filename is `model_quint8_<config>.onnx` (unsigned int8), not `qint8` → config fixed.
- ST 5.7 renamed `get_sentence_embedding_dimension` → `get_embedding_dimension` → updated.
- Reranker folder is 2.4 GB on disk (fp32 safetensors + fp32 ONNX + int8 ONNX). Runtime needs only one ONNX file → Docker/Kaggle packaging will ship just that.
- Reranker tokenizer printed an "incorrect regex pattern" warning from `transformers` — likely a false positive for XLM-R; to be verified in Step 5.

---

## Step 4 — Indexing (FAISS + BM25 + SQLite) and hybrid retrieval

**What**
- `app/stores/base.py` — `VectorStore` / `SparseStore` protocols (swap FAISS→Qdrant etc. without touching retrieval).
- `app/stores/faiss_store.py` — `IndexIDMap2(IndexFlatIP)`: exact inner-product search, vectors keyed by docstore row id; `hnsw` available via `AEGIS_FAISS_INDEX_TYPE`.
- `app/stores/bm25_store.py` — `bm25s` (k1=1.5, b=0.75) over the domain tokenizer; zero-score padding filtered out.
- `app/stores/docstore.py` — SQLite: `chunks` table (text + metadata, index on `(doc_id, chunk_idx)` for neighbour lookup) and `meta` table (index provenance).
- `app/ingest/indexer.py` — builds all three into `data/index.tmp/`, verifies `faiss == bm25 == chunks` counts, then swaps into `data/index/` (no half-written index is ever live). Records `corpus_sha256`, embed model/variant/dim, chunk budget, build time.
- `app/retrieval/hybrid.py` — dense top-k ‖ BM25 top-k → **RRF** (k=60) → top `fusion_top_k` → adjacent expansion. Every hit carries its dense rank/score, BM25 rank/score, fused score (full transparency for tracing/notebook).
- `HybridRetriever.from_index_dir` refuses to serve an index built with a different embedding model/dimension (`IndexMismatchError`).
- `app/retrieval/expand.py` — adds ±n same-document neighbours as extra candidates (`expanded_from` set); they compete in the reranker.
- `eval/supplementary_queries.csv` — **our own** 14-query robustness set (10 paraphrase/exact-token + 4 extra traps), clearly separate from the official set.
- `scripts/eval_retrieval.py` → `docs/results/retrieval.json` — dense vs BM25 vs hybrid ablation.

**Why**
- **Exact FAISS search:** corpus is tiny → exact is both fastest and lossless; approximate indexes only pay off at ~100k+ vectors (config switch).
- **IDs from the docstore:** FAISS, BM25 and SQLite share one id space → cannot silently drift apart.
- **RRF instead of weighted score blending:** cosine (0–1) and BM25 (unbounded) aren't comparable; RRF uses ranks only, needs no normalisation or tuning beyond one standard constant.
- **Atomic build + provenance + mismatch check:** operational safety for on-prem re-ingest (R5); corpus hash makes "which SOP version answered this?" auditable.
- **Supplementary set:** the official set has only 4 answerable queries, too few to distinguish retrieval strategies.

**Results** (fully offline; index build 0.3 s; artifacts: `faiss.index` 7.8 KB, `docstore.db` 24 KB, `bm25/` ~5 KB)

| Set | Mode | hit@1 | hit@3 | MRR |
|---|---|---|---|---|
| official (n=4) | dense | 4/4 | 4/4 | 1.000 |
| official (n=4) | bm25 | 4/4 | 4/4 | 1.000 |
| official (n=4) | **hybrid** | **4/4** | 4/4 | **1.000** |
| supplementary (n=10) | dense | 10/10 | 10/10 | 1.000 |
| supplementary (n=10) | bm25 | 9/10 | 9/10 | 0.900 |
| supplementary (n=10) | **hybrid** | **10/10** | 10/10 | **1.000** |

Findings:
1. **Hybrid is never worse than either retriever alone.** BM25 misses S5 ("above 5 volts" has no lexical overlap with "exceeds 5.0V"); dense recovers it.
2. **Retriever agreement is a strong signal.** Query 4 (override code): the correct chunk is #1 in both → RRF 0.0328 vs ≤0.0161 for all others. Dense alone separates it only weakly (cos 0.600 vs 0.575 for the runner-up); BM25 (`override`, `throttling`, `authorization`) makes it decisive.
3. **Trap queries score as high as or higher than answerable ones** — measured evidence for the layered-refusal design:

| Query | Type | Dense top cos | BM25 top score |
|---|---|---|---|
| 1 voltage range | answerable | 0.797 | 1.204 |
| 3 HF band | answerable | 0.588 | 1.467 |
| 4 override code | answerable | 0.600 | 2.796 |
| **5 Mark-IV range in bad weather** | **trap** | **0.682** | **2.780** |
| **6 Type-C coolant manufacturer** | **trap** | **0.728** | **3.379** |
| S14 tyre pressure (off-domain) | trap | 0.565 | — (no overlap) |

   In-domain traps share the entities of a real SOP ("Mark-IV Radar", "Type-C Marine coolant"), so every retrieval score is high. Only fully off-domain questions (S14) look weak at retrieval level. → Retrieval gate can only catch off-domain queries; in-domain traps **must** be caught by the answer-level layers (constrained generation, grounding, quote check). The reranker (Step 5) is tested next as a better-calibrated gate signal.

- **28/28 tests pass** (`tests/test_retrieval.py` adds: RRF maths, expansion de-dup/same-doc, index alignment, official top-1 = `source_doc`, BM25 exact code match, embedder-mismatch rejection).

**Issues / notes**
- With 5 chunks and `fusion_top_k=5`, expansion adds nothing on this corpus (every chunk is already a candidate); verified separately with `fusion_top_k=1` (Procedure hit → Overview neighbour added with `expanded_from`). It matters for larger/longer SOP sets.
- On Windows, the atomic swap (`rename`) would fail if another process holds the index open; the API's `/ingest` (Step 8) must close/reload handles around the swap.

---

## Step 5 — Reranker selection and refusal-gate calibration

**What**
- `app/models/reranker.py` — cross-encoder wrapper: local files only; `torch-fp32 | onnx-fp32 | onnx-int8`; ONNX session with graph optimisation, threads, provider selection; raw logits → our own sigmoid (identical scoring across models/variants); inputs stripped; reranks on `embed_text` (title > section + body) and returns top `rerank_top_n`.
- `scripts/download_models.py --extra-rerankers ...` — fetch additional cross-encoders for benchmarking (MiniLM downloaded, ONNX-exported, int8-quantized, added to manifest).
- `scripts/bench_reranker.py` → `docs/results/reranker.json` — 2 models × 3 variants on identical hybrid candidates, 20 queries (official + supplementary), 5 timed runs each.
- `scripts/calibrate_gate.py` → `docs/results/gate_calibration.json` — τ sweep over saved scores (no models needed; reproducible).
- `tests/test_reranker.py` — score range, empty input, sort/truncate, override-code chunk ranks first, official-set gate behaviour at the calibrated τ.

**Tokenizer warning check (open item from Step 3)**
`transformers` warned about an "incorrect regex pattern" in the bge-reranker tokenizer. Compared the fast tokenizer against the reference SentencePiece `XLMRobertaTokenizer` on all 3 SOPs, all official questions and a domain-token stress string: **identical token ids on all real texts**; the only difference is one extra `▁` token on an artificial string with trailing whitespace. Reranker inputs are `.strip()`-ed → no effect. Warning is a false positive for this data. (`sentencepiece` used only via `uv run --with` for this check; not a project dependency.)

**Results** (CPU only, 6 threads, offline; latency = rerank of all ~5 candidates for one query)

| Model / variant | Load (s) | p50 (ms) | p95 (ms) | hit@1 (n=14) | MRR | Max drift vs fp32 | AUROC ans-vs-trap | Max trap top score |
|---|---|---|---|---|---|---|---|---|
| bge-reranker-base / torch-fp32 | 1.8 | 805 | 1214 | 14/14 | 1.0 | 0 | 0.917 | 0.523 |
| **bge-reranker-base / onnx-fp32** | 11.1 | **617** | 780 | 14/14 | 1.0 | **0** | **0.917** | **0.523** |
| bge-reranker-base / onnx-int8 | 2.8 | 514 | 741 | 14/14 | 1.0 | **0.190** | 0.893 | 0.666 |
| MiniLM-L6-v2 / torch-fp32 | 0.6 | 123 | 146 | 14/14 | 1.0 | 0 | 0.857 | 0.848 |
| MiniLM-L6-v2 / onnx-fp32 | 0.5 | 102 | 145 | 14/14 | 1.0 | 0 | 0.857 | 0.848 |
| MiniLM-L6-v2 / onnx-int8 | 0.3 | 85 | 102 | 14/14 | 1.0 | 0.124 | 0.857 | 0.830 |

(AUROC = probability an answerable query's top score exceeds a trap's; 0.5 = useless, 1.0 = perfect separation.)

Top reranked score per query (bge-reranker-base onnx-fp32): answerable official 1–4: 0.936 / 1.000 / 0.871 / 0.988; **trap 5: 0.0039**; **trap 6: 0.523**; supplementary traps S11–S14: 0.089 / 0.021 / 0.023 / 0.001; lowest answerable paraphrases: S5 0.015, S3 0.085, S1 0.188.

**Findings**
1. **Ranking accuracy is saturated** — all six configurations put the correct document first for all 14 answerable queries. The choice therefore rests on gate quality, calibration and speed.
2. **The reranker is a far better refusal signal than retrieval scores.** Retrieval scores ranked traps *above* real questions (Step 4); bge-reranker reaches AUROC 0.917. It recognises that the radar procedure does not answer "maximum range in bad weather" (trap 5 → 0.0039) even though every retrieval score for that query was high.
3. **But no threshold separates everything.** Trap 6 (coolant manufacturer) scores 0.523 — the chunk *mentions* "Type-C Marine" coolant, so the cross-encoder judges it relevant. Meanwhile valid paraphrases can score low (S5 "above 5 volts" 0.015). A high τ would refuse real questions. → Confirms the layered design: the gate is a coarse pre-filter; answer-level layers (Step 6–7) must handle in-domain traps like query 6.
4. **int8 is fine for the embedder but NOT for the reranker.** Reranker int8 drifts scores by up to 0.19 and pushes traps *up* (trap 6: 0.52 → 0.67; S11: 0.09 → 0.28), degrading calibration of the very score the gate thresholds. Per the evidence-gated rule from Step 3, the reranker stays fp32.
5. **MiniLM is ~6× faster but a worse gate** (AUROC 0.857; trap 6 at 0.848, trap S12 at 0.61; valid paraphrase S4 at 0.35–0.47).

**Decision:** `BAAI/bge-reranker-base`, **ONNX fp32** (`rerank_quantized=False`).
- Best gate separation (AUROC 0.917) and best calibration (lowest trap scores).
- ONNX fp32 is numerically identical to torch (drift 0) and 23% faster (617 vs 805 ms p50).
- ~0.6 s/query on this low-power laptop CPU is acceptable: LLM generation (Step 7) will cost seconds, and on GPU (`AEGIS_DEVICE=cuda`) the cross-encoder cost becomes negligible.
- MiniLM-L6 ONNX stays available as a **low-latency profile** (`AEGIS_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2`, τ must be re-calibrated with `calibrate_gate.py --model ...`).

**Gate calibration (τ)**

| τ | Answerable wrongly refused (n=14) | Traps refused at gate (n=6) |
|---|---|---|
| 0.0025 | 0 | 1 (S14) |
| **0.005 – 0.0125** | **0** | **2 (S14, official 5)** |
| 0.02 | 1 (S5) | 2 |
| 0.05 | 1 (S5) | 4 |
| 0.1 | 2 (S5, S3) | 5 |
| 0.2 – 0.5 | 3 (S5, S3, S1) | 5 |

**Decision:** τ = **0.0075** (log-midpoint of the zero-false-refusal plateau between trap 5 at 0.0039 and paraphrase S5 at 0.015).
- Cost asymmetry: a false refusal at the gate is unrecoverable (the LLM never sees the question); a trap that passes still faces constrained generation, grounding and the quote check. So τ is set at the top of the safe plateau, not to maximise trap catches.
- Honest caveat: calibrated on 20 queries (14 answerable, 6 traps); the margin is thin (0.0039 vs 0.015). τ is config (`AEGIS_RERANK_THRESHOLD`) and must be re-run with `calibrate_gate.py` on a larger validation set before production.

- **32/32 tests pass.**

**Issues / notes**
- bge-reranker ONNX fp32 first load takes ~11 s (ORT graph optimisation of a 1.1 GB model at session creation). One-off at API startup; can be removed by saving the pre-optimised graph at download time (ORT `optimized_model_filepath`) — to do in Step 8 packaging.
- Windows console (cp1252) can't print `τ` → scripts print `tau`.
