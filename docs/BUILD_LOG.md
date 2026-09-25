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
