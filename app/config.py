"""Central configuration (D2).

Every tunable lives here and can be overridden with an ``AEGIS_`` env var
(e.g. ``AEGIS_RERANK_THRESHOLD=0.4``) so moving between hardware profiles or
deployment targets is a config change, never a code change.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AEGIS_", env_file=".env", extra="ignore")

    # --- Paths -------------------------------------------------------------
    knowledge_base_dir: Path = ROOT / "knowledge_base"
    index_dir: Path = ROOT / "data" / "index"
    models_dir: Path = ROOT / "models"
    eval_csv: Path = ROOT / "evaluation_queries.csv"

    # --- Runtime / hardware (D2) -------------------------------------------
    # "auto" picks the best available ONNX Runtime provider at startup.
    device: Literal["auto", "cuda", "dml", "openvino", "cpu"] = "auto"
    num_threads: int = 0  # 0 = physical core count

    # --- Chunking (D6) -----------------------------------------------------
    chunk_max_tokens: int = 200
    chunk_overlap_sentences: int = 0

    # --- Embeddings (D3) ---------------------------------------------------
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_backend: Literal["torch", "onnx", "openvino"] = "onnx"
    embed_quantized: bool = True  # int8; kept only if bench shows no hit@1 drop
    embed_max_seq_length: int = 256
    embed_batch_size: int = 32
    embed_query_prefix: str = "Represent this sentence for searching relevant passages: "
    query_cache_size: int = 1024

    # --- Retrieval (D4) ----------------------------------------------------
    dense_top_k: int = 10
    sparse_top_k: int = 10
    rrf_k: int = 60
    expand_neighbors: int = 1

    # --- Reranking + gate (D5, D8) ------------------------------------------
    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_top_n: int = 3
    rerank_threshold: float = 0.3  # τ — calibrated in step 5

    # --- Generation (D0) ---------------------------------------------------
    # ollama -> Docker deployment; transformers -> Kaggle notebook (internet off);
    # openai_compat -> vLLM on a GPU server; stub -> no LLM (extractive baseline).
    llm_provider: Literal["stub", "ollama", "openai_compat", "transformers"] = "stub"
    llm_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:3b-instruct-q4_K_M"  # HF id or local path when provider=transformers
    llm_device: Literal["auto", "cpu", "cuda"] = "auto"
    llm_temperature: float = 0.0
    llm_seed: int = 42
    llm_num_ctx: int = 4096
    llm_timeout_s: float = 120.0

    refusal_message: str = "Not found in documents."

    # --- Observability (D11) -----------------------------------------------
    otel_enabled: bool = False
    otel_endpoint: str = "http://localhost:6006/v1/traces"
    log_level: str = "INFO"
    log_json: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
