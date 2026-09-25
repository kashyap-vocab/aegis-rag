FROM ghcr.io/astral-sh/uv:0.11 AS uv

FROM python:3.12-slim AS builder
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1 UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY app ./app
COPY scripts ./scripts
COPY knowledge_base ./knowledge_base
COPY evaluation_queries.csv ./
COPY eval ./eval
RUN uv run --no-sync python scripts/download_models.py \
 && uv run --no-sync python scripts/ingest.py \
 && find models -name "*.safetensors" -delete \
 && rm -f models/bge-reranker-base/onnx/model_quint8_*.onnx models/bge-small-en-v1.5/onnx/model.onnx

FROM python:3.12-slim AS runtime
RUN useradd --create-home --uid 10001 aegis
WORKDIR /app
COPY --from=builder --chown=aegis:aegis /app /app
ENV PATH=/app/.venv/bin:$PATH \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    AEGIS_LLM_PROVIDER=ollama \
    AEGIS_LLM_BASE_URL=http://llm:11434 \
    AEGIS_LLM_MODEL=qwen2.5:3b-instruct-q4_K_M \
    AEGIS_LLM_FALLBACK_MODEL=qwen2.5:1.5b-instruct-q4_K_M
USER aegis
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
