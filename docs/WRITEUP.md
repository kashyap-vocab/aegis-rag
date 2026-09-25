# Operation Aegis — Offline RAG with Layered Hallucination Control

**Code:** https://github.com/kashyap-vocab/aegis-rag · **Notebook:** https://www.kaggle.com/code/mandavakashyapsai/operation-aegis-offline-rag

A fully offline Retrieval-Augmented Generation system that answers questions about technical SOPs, and refuses when the answer isn't in them. It makes no cloud API calls. Every model is open-weight and runs locally: verified on a laptop CPU, on a Kaggle T4 with internet disabled, and in a Docker container with no network at all (tested in CI).

**Headline results, with Qwen2.5-3B-Instruct running locally:**

| | Official set (6) | Supplementary set (14) |
|---|---|---|
| Overall accuracy | **6 / 6** | 12 / 14 |
| Adversarial traps refused | **2 / 2** | **4 / 4** |
| False refusals of answerable questions | **0 / 4** | **0 / 10** |
| Retrieval hit@1 (correct source doc ranked first) | **4 / 4** | **10 / 10** |

The supplementary set is ours (paraphrases, exact-code queries, extra traps), built to stress what the official set can't. Both "misses" on it are keyword-only probes (`RDR_CAL_INIT`, `Alpha-7-Tango`) whose descriptive reference answers the automatic scorer can't match to the model's short answers (§6.3). A 100-query held-out evaluation is in §6.5.

---

## 1. System architecture

```
INGEST (offline CLI)
 knowledge_base/*.md → parser → structure-aware chunker → rule-based metadata
   ├─ bge-small-en-v1.5 (ONNX int8) → FAISS IndexFlatIP        (dense)
   ├─ domain tokenizer → BM25 (bm25s)                           (sparse)
   └─ SQLite docstore (text, metadata, index provenance)

QUERY (LangGraph StateGraph, served by FastAPI)
 sanitize ─► retrieve [dense ‖ BM25 → RRF → adjacent expansion] ─► rerank (bge-reranker-base)
    │                                                                  │
    └─(injection/invalid)─► REFUSE           (L1: top score < τ)─► REFUSE
                                                                       ▼
                         generate (Qwen2.5-3B, JSON-schema output, evidence-first)
                            │ (L2: answerable=false · model error → fallback model → refuse)
                            ▼
                         verify (L3: verbatim quote check · L4: numeric/code grounding) ─► ANSWER + citations
```

Every refusal returns the same message, `Not found in documents.`, and records which layer fired, for audit and evaluation.

| Layer | Choice | Why this choice / its advantage |
|---|---|---|
| Orchestration | **LangGraph** `StateGraph`: 6 nodes, conditional refusal edges | Each guardrail is an explicit, inspectable edge to `END`. The graph renders as a diagram (`GET /graph`), and new nodes (e.g. query rewriting) slot in without touching the others |
| Chunking | Heading → numbered step → sentence, rule-based metadata | SOP sections are self-contained units; deterministic, repeatable ingest with no LLM needed |
| Embeddings | `BAAI/bge-small-en-v1.5`, ONNX Runtime **int8** | Strong retrieval for 33M params; 2.7× faster than PyTorch with no retrieval loss (§4) |
| Vector DB | **FAISS** `IndexFlatIP` + SQLite docstore | In-process, no server to secure, index is a single shippable file; exact search = zero recall loss |
| Sparse | **BM25** (`bm25s`) with a domain tokenizer | Matches exact identifiers (`Alpha-7-Tango`, `14.5`, `RDR_CAL_INIT`) that embeddings blur |
| Fusion | **Reciprocal Rank Fusion** (k = 60) | Combines ranks, so cosine and BM25 scores never need normalising |
| Reranker | `BAAI/bge-reranker-base` cross-encoder, ONNX fp32 | Best trap/answer separation of the candidates tested (AUROC 0.917); its score drives the refusal gate |
| LLM | **Qwen2.5-3B-Instruct** (Ollama Q4_K_M locally; `transformers` / PyTorch on Kaggle) | Reliable instruction and JSON following at a size that fits edge hardware; 1.5B is 2× faster but less accurate (§6.2) |
| Serving | FastAPI + Prometheus + structlog, Docker Compose | Deployable on-prem as-is: health checks, hot re-ingest, metrics, JSON logs |
| Packaging | `uv` lockfile, offline wheel bundle, model SHA-256 manifest | Reproducible, pinned builds and verifiable artifacts for air-gapped transfer |

### Why LangGraph, and how much of it

The pipeline is a fixed QA flow, so LangGraph is used to express control flow, not to build an autonomous agent:
- **State:** a typed `RAGState` flows through `sanitize → retrieve → rerank → gate → generate → verify`.
- **Routing:** after each safety-relevant node, a router sends the request either onward or to `END` as a refusal.
- **Fallback lives inside the `generate` node:** primary model → smaller fallback model → fail-closed refusal.

Retrieval scoring, the gate and the verifiers are our own small functions (~50 lines each) rather than framework defaults. That keeps every decision that affects hallucination explicit, unit-tested and explainable. Generic abstractions would hide the prompts, retries and thresholds.

---

## 2. Offline RAG strategy

### 2.1 Ingestion and chunking

1. **Parse** each Markdown SOP into its title (H1), SOP number, and ordered H2/H3 sections. `#` lines inside code fences are ignored.
2. **Chunk** in order of preference:
   - one chunk per section if it fits the 200-token budget;
   - otherwise split on numbered steps, keeping unnumbered continuation lines with their step (e.g. the "If voltage exceeds 5.0V…" safety line stays with step 3);
   - otherwise split on sentences, re-attaching step markers.
3. **Prefix** each chunk with `Title > Section` for embedding and BM25, so a short chunk like *"Manual override requires authorization code Alpha-7-Tango."* still carries *"Main Engine Cooling System > Thermal Thresholds"*.
4. **Metadata** is rule-based: `chunk_id`, `doc_id`, `sop_number`, `section`, `chunk_idx` (used for adjacent expansion). It is deterministic and needs no LLM.

**Advantage:** every answer fact stays intact inside one chunk together with its context. The override code sits in the same chunk as "throttling", which query 4 depends on. The corpus produces 5 chunks, one per section.

**Domain tokenizer.** Used by BM25 and by the verifiers. It keeps compound tokens whole *and* emits their parts:
- `Alpha-7-Tango → alpha-7-tango, alpha, 7, tango`
- `4.8V. → 4.8v`, `92°C → 92°c`, `5,000 → 5000`

Generic tokenizers split these apart, and BM25 then can't match the codes an operator actually asks about.

### 2.2 Vector database configuration

- **FAISS `IndexIDMap2(IndexFlatIP)`** over L2-normalised vectors, i.e. exact cosine search. HNSW (`AEGIS_FAISS_INDEX_TYPE=hnsw`) is a config switch for corpora of ~100k+ vectors.
- **One id space:** FAISS, BM25 and SQLite share the same ids, so they can't drift apart.
- **Atomic builds:** the index is built into a temp directory, checked (`faiss == bm25 == chunks`), then swapped in, so a live API never sees a half-built index.
- **Provenance:** the docstore records the corpus SHA-256 and the embedder, and the API refuses to serve an index built with a different embedding model.

### 2.3 Retrieval: hybrid search, fusion, expansion, reranking

Dense top-10 and BM25 top-10 are fused with RRF, and the top 5 are expanded with ±1 same-document neighbours. The cross-encoder then rescores the candidates and keeps the top 3.

---

## 3. Hallucination control

The key finding that shaped the design came from measuring retrieval scores on the traps:

| Query | Type | Dense top cosine | BM25 top score | Reranker top score |
|---|---|---|---|---|
| 1 voltage range | answerable | 0.797 | 1.20 | 0.936 |
| 3 HF failover band | answerable | 0.588 | 1.47 | 0.871 |
| 4 override code | answerable | 0.600 | 2.80 | 0.988 |
| **5 Mark-IV range in bad weather** | **trap** | **0.682** | **2.78** | **0.004** |
| **6 Type-C coolant manufacturer** | **trap** | **0.728** | **3.38** | **0.523** |

**The traps score as high as the real questions, or higher, on every retrieval signal,** because they name real entities from the SOPs. A retrieval threshold alone can't refuse them. We therefore use four independent layers, each catching a different failure:

| Layer | Mechanism | Catches | Cost |
|---|---|---|---|
| **L0** input | NFKC normalisation, control-character stripping, length cap, narrow prompt-injection patterns (refused before retrieval) | "Ignore previous instructions…", fake `<system>` tags | 0.06 ms |
| **L1** retrieval gate | Refuse if the top cross-encoder score < τ = 0.0075; only chunks ≥ τ reach the LLM | Off-topic questions and questions no chunk addresses (trap 5) | 0 extra (reuses the reranker) |
| **L2** LLM judgement | JSON-schema output; *"a document merely mentioning the subject is not enough"* | In-domain traps whose entity exists but whose attribute doesn't (trap 6) | 1 LLM call |
| **L3** quote check | `supporting_quote` must appear verbatim in the context | Fabricated evidence | < 0.5 ms |
| **L4** grounding verifier | Every number, code and acronym in the answer must occur in the context (`5 == 5.0`, `5,000 == 5000`) | Hallucinated values such as `Bravo-9-Tango` or `95°C` | < 0.5 ms |

### Prompt design

- **Evidence first:** the schema orders `supporting_quote → source_doc → answerable → answer`. With constrained decoding, the model has to commit to a verbatim quote *before* deciding and answering, so it can't write an answer first and justify it afterwards.
- **"Mentioned ≠ answered"** is stated explicitly and shown in two few-shot examples. The examples use an invented hydraulic-pump SOP and a different attribute type, and a unit test asserts that no evaluation content leaks into the prompt.
- **Documents are data:** retrieved chunks are fenced as `<document source=… section=…>` blocks. Any fence-like tags inside a chunk are neutralised, so a poisoned document can't close its own block.
- **Deterministic decoding:** temperature 0, a fixed seed, and a 256-token cap.
- **Fail closed:** malformed JSON, an empty answer or an unreachable model always becomes a refusal, never a guess.

### Calibrating the gate (τ)

The threshold was swept over the cross-encoder scores of 20 queries:

| τ | Answerable wrongly refused (n = 14) | Traps refused at gate (n = 6) |
|---|---|---|
| 0.0025 | 0 | 1 |
| **0.005 – 0.0125** | **0** | **2 (incl. official trap 5)** |
| 0.02 | 1 | 2 |
| 0.1 | 2 | 5 |
| 0.2 – 0.5 | 3 | 5 |

We chose τ = 0.0075, the log-midpoint of the zero-false-refusal plateau, deliberately low because the two errors cost different amounts:
- a real question refused at the gate is lost for good, since the LLM never sees it;
- a trap that passes still meets L2–L4.

The honest caveat is that the calibration set has 20 queries; `scripts/calibrate_gate.py` re-runs it on any larger validation set.

---

## 4. Model selection and optimisation (with evidence)

The dev machine is a laptop with a Ryzen 5 5500U, CPU only, 6 threads, fully offline.

**Embedder variants.** Accuracy was gated by evidence: int8 was adopted only because retrieval didn't drop.

| Variant | Query p50 | Passages/s | hit@1 | Fidelity vs fp32 (mean cosine) |
|---|---|---|---|---|
| PyTorch fp32 | 33.8 ms | 44 | 4/4 | 1.0000 |
| ONNX fp32 | 12.8 ms | 50 | 4/4 | 1.0000 |
| **ONNX int8 (chosen)** | **12.5 ms** | **59** | 4/4 | 0.9965 |

**Rerankers.** 20 queries, all six configurations reach hit@1 = 14/14 on the answerable ones, so the choice turned on gate quality:

| Model / variant | p50 | AUROC answerable vs trap | Max trap score | Max score drift vs fp32 |
|---|---|---|---|---|
| **bge-reranker-base ONNX fp32 (chosen)** | 617 ms | **0.917** | **0.52** | 0 |
| bge-reranker-base ONNX int8 | 514 ms | 0.893 | 0.67 | 0.19 |
| MiniLM-L6 ONNX fp32 | 102 ms | 0.857 | 0.85 | 0 |
| MiniLM-L6 ONNX int8 | 85 ms | 0.857 | 0.83 | 0.12 |

**Findings**
- **int8 is right for the embedder but wrong for the reranker.** It shifts reranker scores by up to 0.19 and pushes traps *upward*, which damages the very score the gate thresholds.
- **MiniLM is 6× faster but a worse refusal signal.** It stays available as a low-latency profile (`AEGIS_RERANK_MODEL`, with τ re-calibrated).

**ONNX Runtime without `optimum`.** The embedder and reranker run through a ~40-line direct ONNX Runtime wrapper (tokenizer, session, CLS pooling or logits). Its output is numerically identical to PyTorch (cosine 1.00000; reranker scores identical to 4 d.p.). It drops a heavy dependency and version-coupling risk, and the ONNX execution provider (CPU, CUDA, DirectML or OpenVINO) is chosen at runtime from what the machine offers. The measurements here use CPU.

**Retrieval ablation** (answerable queries; supplementary set):

| Mode | Official hit@1 | Supplementary hit@1 | MRR |
|---|---|---|---|
| Dense only | 4/4 | 10/10 | 1.00 |
| BM25 only | 4/4 | 9/10 | 0.90 |
| **Hybrid (RRF)** | **4/4** | **10/10** | **1.00** |

- **Hybrid is never worse than either retriever alone.** BM25 misses the paraphrase "above 5 volts", which has no word overlap with "exceeds 5.0V", and dense retrieval recovers it.
- **Agreement between retrievers is decisive.** For query 4, the correct chunk is ranked #1 by both, giving an RRF score of 0.033, while every other chunk scores ≤ 0.016.

---

## 5. Evaluation pipeline and regression testing

- **Automated harness** (`scripts/evaluate.py`, shared with the notebook):
  - Scoring is deterministic, with no LLM-as-judge; a 3B judge would be less reliable than the system it grades.
  - An answer is correct if it contains every key fact of the reference (numbers, codes, acronyms, normalised). Otherwise it falls back to ≥ 50% content-word recall.
  - A trap is correct only if refused.
  - Retrieval is scored as hit@1 against `source_doc`.
- **Supplementary robustness set** (`eval/supplementary_queries.csv`): 14 queries, 10 answerable (paraphrase and exact-token probes) and 4 extra traps.
- **Guardrail ablation** (`scripts/ablate_guardrails.py`): the same questions run under different layer configurations, with LLM calls memoised so configurations differ only in which checks apply.
- **Benchmarks:** `bench_embedder.py`, `bench_reranker.py`, `calibrate_gate.py`, `eval_retrieval.py`. All results are versioned as JSON in `docs/results/`.
- **Regression tests:** 82 pytest tests. They cover:
  - chunk boundaries and facts surviving chunking;
  - the tokenizer;
  - RRF maths and expansion;
  - index alignment and embedder-mismatch rejection;
  - reranker ordering and the calibrated gate on the official set;
  - injection positives *and* legitimate negatives ("override" alone must not trigger);
  - grounded vs hallucinated tokens, and the quote check;
  - fail-closed JSON parsing, and mocked Ollama / OpenAI-compatible backends;
  - the FastAPI endpoints and metrics.

---

## 6. Results

### 6.1 End-to-end results (Qwen2.5-3B-Instruct Q4_K_M via Ollama, laptop CPU)

| Query | Result | Refused by | Answer |
|---|---|---|---|
| 1 acceptable voltage range | ✅ | — | 4.5V to 4.8V |
| 2 throttling temperature | ✅ | — | 92°C |
| 3 HF failover band | ✅ | — | 14.5 MHz |
| 4 override authorization code | ✅ | — | Alpha-7-Tango. |
| 5 Mark-IV radar range in bad weather (trap) | ✅ refused | **L1 gate** (0.004 < τ) | Not found in documents. |
| 6 Type-C coolant manufacturer (trap) | ✅ refused | **L2 LLM** (answerable = false) | Not found in documents. |

Per-stage p50 latency on the laptop CPU:

| Stage | Latency |
|---|---|
| sanitize | 0.1 ms |
| retrieve | 43 ms |
| rerank | 670 ms |
| generate | 7.4 s |
| verify | 0.3 ms |

### 6.2 Model and baseline comparison (all 20 queries)

| System | Overall | Traps refused | False refusals | p50 latency |
|---|---|---|---|---|
| Extractive baseline (no LLM) | 14/20 (official 5/6) | 2/6 | 2/14 | 0.7 s |
| Qwen2.5-1.5B-Instruct | 16/20 (official 5/6) | 6/6 | 3/14 | 3.8 s |
| **Qwen2.5-3B-Instruct** | **18/20 (official 6/6)** | **6/6** | **0/14** | 7.6 s |

**The no-LLM extractive baseline shows exactly what the generative model adds.** On trap 6 the baseline returns *"Coolant fluid (Type-C Marine) must be replaced every 5,000 operational hours."* That answer is fully grounded, with a verbatim quote and every token in the SOP, but it doesn't answer "who is the manufacturer". L3 and L4 check faithfulness, not relevance; only a model that understands the question can judge relevance. That is L2's job. The 1.5B model refuses every trap but also refuses 3 real questions (including official query 3), so it was rejected.

### 6.3 Guardrail ablation (Qwen2.5-3B)

| Configuration | Official | All 20 | Traps refused | False refusals |
|---|---|---|---|---|
| **Full (L1–L4)** | **6/6** | **18/20** | 6/6 | **0/14** |
| No retrieval gate (L1 off) | 5/6 | 17/20 | 6/6 | 2/14 |
| No quote check (L3 off) | 6/6 | 18/20 | 6/6 | 0/14 |
| No grounding verifier (L4 off) | 6/6 | 18/20 | 6/6 | 0/14 |
| LLM only (L1, L3, L4 off) | 5/6 | 17/20 | 6/6 | 2/14 |

- **The gate improves accuracy on real questions, not just safety.** Without it, low-relevance chunks reach the model, and it wrongly refuses two real questions, including official query 3.
- **L3 and L4 rarely need to act with the 3B model:** 0 times on these 20 queries, 2 times on the 100-query set (§6.5). They are near-zero-cost insurance for weaker or swapped models, for prompt drift and for distribution shift, and they are unit-tested against fabricated codes, values and quotes.
- **The two supplementary "misses" (S6, S7) come from the scorer.** The queries are bare keywords (`RDR_CAL_INIT`, `Alpha-7-Tango`), and the model answered with the command and the code. The reference texts were written as descriptions, so the key-fact matcher can't credit them.

### 6.4 Kaggle notebook run (internet disabled, Tesla T4)

This is the same code, run as an attached public notebook:
- **Models:** Qwen2.5-3B-Instruct loaded from Kaggle Models with `transformers`/PyTorch in fp16. The embedder and reranker load from an attached dataset. Dependencies install from an attached wheel dataset with `pip --no-index`.
- **Results reproduce the local Ollama run exactly:**

| Set | Overall | Answerable correct | Traps refused | False refusals | Retrieval hit@1 | p50 latency | p95 latency |
|---|---|---|---|---|---|---|---|
| Official | **6/6** | 4/4 | **2/2** | **0/4** | 4/4 | 3.4 s | 5.0 s |
| Supplementary | 12/14 | 8/10 | 4/4 | 0/10 | 10/10 | 3.4 s | 4.2 s |
| All | **18/20** | 12/14 | **6/6** | **0/14** | 14/14 | 3.4 s | 4.5 s |

- **Latency:** end-to-end p50 drops from 7.6 s (laptop CPU, 4-bit Ollama) to 3.4 s (T4, fp16), and the LLM loads in 16.7 s. Raw outputs are in `docs/results/kaggle/`.
- **Guardrail ablation on the T4 run:** the full configuration has 0/14 false refusals, while "no gate" and "LLM only" each have 1/14. This agrees with the CPU run: the gate improves answer quality by keeping noisy chunks away from the model.

**CPU and GPU give identical answers.** Same notebook, official set, LLM forced to CPU:

| Device | Official accuracy | Traps refused | p50 latency | p95 latency |
|---|---|---|---|---|
| Kaggle T4 GPU (fp16, `transformers`) | 6/6 | 2/2 | 3.4 s | 5.0 s |
| Kaggle CPU (2 vCPU, fp32, `transformers`) | 6/6 | 2/2 | 47.0 s | 55.4 s |
| Laptop CPU (Ryzen 5 5500U, 4-bit Q4_K_M via Ollama) | 6/6 | 2/2 | 8.1 s | 18.2 s |

The accuracy doesn't change with the hardware, only the latency does. On CPU-only edge hardware, 4-bit quantised serving (Ollama/llama.cpp) is about 6× faster than an unquantised fp32 model on a similar CPU budget. That is why Ollama with Q4_K_M is the deployment default, and `transformers` with fp16 is the GPU path.

**One engineering issue found only on Kaggle.** The first run failed with CUDA out-of-memory while loading the LLM. JAX, pulled in by `bm25s`'s optional top-k backend, pre-allocated 75% of the T4's memory. We fixed it by forcing `bm25s` onto its NumPy backend (which is also more deterministic) and setting `JAX_PLATFORMS=cpu`.

### 6.5 Extended held-out evaluation (100 queries)

To test generalisation we built a larger set after all tuning was frozen: 50 answerable questions (direct, paraphrase, reasoning, keyword-only) and 50 unanswerable ones (missing attribute, false premise, off-domain, near-miss number, adversarial). Full report: `docs/EXTENDED_EVAL.md`.

| System | Overall | Answerable correct | Unanswerable refused | Retrieval hit@1 | p50 latency | p95 latency |
|---|---|---|---|---|---|---|
| No-LLM extractive baseline | 52/100 | 37/50 | 15/50 | 50/50 | 0.44 s | 0.49 s |
| Qwen2.5-1.5B-Instruct | 76/100 | 33/50 | 43/50 | 50/50 | 2.85 s | 4.37 s |
| **Qwen2.5-3B-Instruct** | **91/100** | **43/50** | **48/50** | **50/50** | 4.91 s | 7.18 s |

| Category (Qwen2.5-3B) | Accuracy | Category (Qwen2.5-3B) | Accuracy |
|---|---|---|---|
| Direct | 18/18 (100%) | Missing attribute | 20/20 (100%) |
| Paraphrase | 14/14 (100%) | False premise | 9/10 (90%) |
| Reasoning | 7/11 (64%) | Off-domain | 10/10 (100%) |
| Keyword-only | 4/7 (57%) | Near-miss number | 4/5 (80%) |
| | | Adversarial | 5/5 (100%) |

- **The gate threshold generalises.** It was calibrated on 20 earlier queries. On the 50 unseen answerable questions it wrongly refused none, and it stopped 9/10 off-domain questions without an LLM call, 0.5 s vs 5.8 s for a generated answer.
- **Each layer covers a distinct category:** the gate handles off-domain questions, the LLM judgement handles in-domain missing-attribute and false-premise questions, and the injection filter handles override attempts (< 1 ms).
- **3B answers every direct and paraphrased question.** The 1.5B model is more conservative (10/18 direct questions answered), which is why it serves only as the fallback.

---

## 7. Offline deployment

- **Service:** FastAPI (`app/api/main.py`) exposes these endpoints:
  - `POST /query`, with `?trace=true` for the full per-stage trace;
  - `GET /health`, which reports LLM reachability, index provenance and τ;
  - `POST /ingest`, a hot rebuild with atomic swap that keeps the old index if the build fails;
  - `GET /metrics` (Prometheus);
  - `GET /graph`, the LangGraph diagram as Mermaid.

  Models load once at startup. Blocking inference runs in a thread pool.
- **Docker:** a multi-stage `Dockerfile`.
  - The builder runs `uv sync --frozen`, downloads the models, exports ONNX, builds the index, and strips unused weights.
  - The runtime image runs as a non-root user with `HF_HUB_OFFLINE=1` and a `HEALTHCHECK`.
  - `docker-compose.yml` runs `api` and `llm` (Ollama) on an `internal: true` network, so the LLM container has no route out. An `observability` profile adds Prometheus and Grafana.
  - **Verified in CI** (GitHub Actions, `.github/workflows/docker.yml`): the image builds from scratch, then runs with `docker run --network none`. Inside the container it passes `/health` (5 chunks indexed), a correct answer (`Alpha-7-Tango`), the trap refused at the gate, the injection refused, and the Prometheus metrics check. The test also confirms that outbound network access fails from inside the container.
- **Air-gapped delivery:**
  1. Build on a connected machine.
  2. `docker save` the images, and export the Ollama model volume.
  3. Verify `models/MANIFEST.json` (SHA-256 of every model file) after transfer.
  4. `docker load` on the target.

  The Python environment can also be installed from a wheelhouse with `uv sync --offline --frozen`.
- **Scaling:** the API is stateless apart from a read-only index, so it scales horizontally behind a load balancer.
  - The LLM is the bottleneck. Scale it separately: more Ollama replicas, or switch to vLLM with continuous batching on a GPU node.
  - Switching needs only config: `AEGIS_LLM_PROVIDER=openai_compat` with `AEGIS_LLM_BASE_URL`. The same prompt and JSON schema are used, and vLLM applies guided decoding.
- **Hardware profiles** are config, not code:
  - `AEGIS_DEVICE` picks the ONNX execution provider (CUDA → DirectML → OpenVINO → CPU, detected at runtime);
  - `AEGIS_LLM_DEVICE` and `AEGIS_LLM_DTYPE` do the same for the transformers backend;
  - `AEGIS_QUANT_CONFIG` switches int8 kernels between `avx2`, `avx512_vnni` and `arm64`.

## 8. Monitoring, logging and fallback

**Metrics** (Prometheus, `/metrics`; the Grafana dashboard is provisioned automatically):

| Metric | Why it matters |
|---|---|
| `aegis_requests_total{outcome, reason}` | Refusal rate *by layer*. A spike in `gate` refusals suggests index or corpus drift; a spike in `llm_error` means the model backend is failing |
| `aegis_stage_latency_seconds{stage}` | p50/p95 per stage; shows whether retrieval, reranking or generation regressed |
| `aegis_gate_top_rerank_score` (histogram) | Drift in the distribution of retrieval confidence; the early signal to re-calibrate τ |
| `aegis_grounding_failures_total{layer}` | L3/L4 blocks. They should be near zero with a healthy model, so any rise signals model or prompt regression |
| `aegis_llm_up`, `aegis_llm_fallback_total` | Backend health and how often the fallback model is serving |
| `aegis_index_chunks` | Confirms re-ingests loaded what was expected |

**Suggested alerts:** refusal rate > 2× its 7-day baseline; `aegis_llm_up == 0` for 2 minutes; p95 `generate` latency > SLO; any sustained `grounding_failures`.

**Logging:** `structlog` JSON lines to stdout, ready for Loki or ELK on-prem.
- Every request carries an `x-request-id`, echoed in the response header, and binds it to all its log lines.
- The `query` log records the refusal layer, gate score, top source, whether the fallback was used, total latency, and any unsupported tokens.
- Full per-stage traces are available on demand (`?trace=true`): each candidate's dense, BM25, RRF and rerank scores, the gate decision, the raw generation and the verifier output.
- **Not yet implemented:** OpenTelemetry spans exported to a self-hosted Arize Phoenix. That would give trace search across requests without changing the pipeline code.

**Fallback chain** (inside the LangGraph `generate` node):
1. **Primary LLM** (Qwen2.5-3B).
2. **Fallback LLM** (`AEGIS_LLM_FALLBACK_MODEL`, e.g. Qwen2.5-1.5B) on any backend error or unparseable output. Its answers pass through the same L3/L4 verifiers.
3. **Fail-closed refusal** (`llm_error`) with citations to the retrieved SOP sections, so an operator can still read the source directly.

The system never falls back to "answer without verification". `/health` reports `degraded` when the primary model is unreachable.

---

## 9. Limitations and next steps

- **Evaluation scale.** 6 official, 14 supplementary and 100 extended queries, all written against 3 short SOPs. τ should be re-validated on a labelled set drawn from real operator questions before production.
- **CPU generation latency** (~7 s with 3B on a laptop). A GPU, vLLM batching, or the 1.5B model with a stricter prompt are the options, each with its measured trade-off.
- **Next steps:**
  - OpenTelemetry → Phoenix tracing;
  - query rewriting for very short or keyword-only queries (the S6/S7 case);
  - a scheduled regression job that runs `evaluate.py` against every new model or prompt version before promotion.

## 10. Reproducibility and attribution

- **Local run:**
  ```
  uv sync
  uv run scripts/download_models.py
  uv run scripts/ingest.py
  uv run scripts/evaluate.py
  AEGIS_LLM_PROVIDER=ollama uv run scripts/evaluate.py
  uv run uvicorn app.api.main:app
  ```
- **Kaggle:** the notebook installs its dependencies from an attached wheel dataset (`--no-index`) and runs with internet disabled.
- **Open-source components:** FAISS, bm25s, ONNX Runtime, Hugging Face transformers and sentence-transformers, LangGraph, FastAPI, prometheus-client, structlog.
- **Models:** BAAI bge-small-en-v1.5, BAAI bge-reranker-base, Qwen2.5-Instruct (Alibaba Qwen team).
- Code was written with the help of an AI coding assistant; every design decision and measurement is documented in the repository's build log (`docs/BUILD_LOG.md`).

## 11. Submission requirements checklist

| Requirement | Where |
|---|---|
| System architecture, framework choice (LangGraph), agent workflow | §1: diagram, choice table, LangGraph `StateGraph` with conditional refusal edges and fallback chain |
| Chunking logic, embeddings, vector DB configuration | §2.1 – §2.3, §4 (FAISS `IndexFlatIP` + SQLite; bge-small-en-v1.5 transformer embeddings) |
| Prompt and retrieval logic for grounding; evaluation, regression tests, hallucination detection | §3 (4 layers, evidence-first prompt, calibrated gate, quote and grounding detectors), §5 (harness, ablation, 82 tests), §6 (results) |
| Public, reproducible notebook, end to end | [Kaggle notebook](https://www.kaggle.com/code/mandavakashyapsai/operation-aegis-offline-rag): chunking → indexing → hybrid retrieval → reranking → LangGraph → LLM → evaluation, with saved outputs |
| Open-weight LLM inference with PyTorch, plus transformer embeddings | Qwen2.5-3B-Instruct via Hugging Face `transformers` on PyTorch (T4 fp16 and CPU); bge-small-en-v1.5 and bge-reranker-base transformer encoders (ONNX Runtime, numerically identical to PyTorch) |
| No cloud APIs | Notebook runs with internet disabled; no external LLM API anywhere in the codebase |
| Scalable offline inference APIs and deployment | §7: FastAPI service, Docker/Compose on an internal network, air-gapped delivery, scaling path to vLLM; image verified in CI with `--network none` |
| Monitoring, logging and fallback | §8: Prometheus metrics and alerts, structured logs with request IDs, per-request traces, primary → fallback LLM → fail-closed refusal |
