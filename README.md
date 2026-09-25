# Operation Aegis — Offline RAG

Offline, air-gap-ready RAG over technical SOPs with layered hallucination control.
Local open-weight models only (bge-small embedder, bge-reranker, Qwen2.5-Instruct); no cloud APIs.

- Report / Kaggle writeup: [docs/WRITEUP.md](docs/WRITEUP.md)
- Build log (every decision, benchmark and issue): [docs/BUILD_LOG.md](docs/BUILD_LOG.md)
- Benchmark and evaluation outputs: [docs/results/](docs/results/)
- Kaggle notebook: [notebooks/aegis_rag.ipynb](notebooks/aegis_rag.ipynb)

## Quick start

```bash
uv sync
uv run scripts/download_models.py        # the only step that needs network
uv run scripts/ingest.py                 # FAISS + BM25 + SQLite index
uv run pytest                            # 82 tests
uv run scripts/evaluate.py               # extractive baseline (no LLM)
AEGIS_LLM_PROVIDER=ollama uv run scripts/evaluate.py
uv run uvicorn app.api.main:app          # POST /query, GET /health, /metrics, /graph
```

## Docker (on-prem)

```bash
docker compose up -d                                   # api + ollama on an internal network
docker compose exec llm ollama pull qwen2.5:3b-instruct-q4_K_M
docker compose --profile observability up -d           # + Prometheus :9090, Grafana :3000
```

All settings are environment variables prefixed `AEGIS_` (see `app/config.py`).
