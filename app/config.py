from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AEGIS_", env_file=".env", extra="ignore")

    knowledge_base_dir: Path = ROOT / "knowledge_base"
    index_dir: Path = ROOT / "data" / "index"
    models_dir: Path = ROOT / "models"
    eval_csv: Path = ROOT / "evaluation_queries.csv"
    results_dir: Path = ROOT / "docs" / "results"
    supplementary_eval_csv: Path = ROOT / "eval" / "supplementary_queries.csv"

    device: Literal["auto", "cuda", "dml", "openvino", "cpu"] = "auto"
    num_threads: int = 0

    chunk_max_tokens: int = 200
    chunk_overlap_sentences: int = 0

    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_backend: Literal["torch", "onnx", "openvino"] = "onnx"
    embed_quantized: bool = True
    quant_config: Literal["avx2", "avx512", "avx512_vnni", "arm64"] = "avx2"
    embed_max_seq_length: int = 256
    embed_batch_size: int = 32
    embed_query_prefix: str = "Represent this sentence for searching relevant passages: "
    query_cache_size: int = 1024

    faiss_index_type: Literal["flat", "hnsw"] = "flat"
    dense_top_k: int = 10
    sparse_top_k: int = 10
    rrf_k: int = 60
    fusion_top_k: int = 5
    expand_neighbors: int = 1

    rerank_model: str = "BAAI/bge-reranker-base"
    rerank_quantized: bool = False
    rerank_max_length: int = 512
    rerank_top_n: int = 3
    rerank_threshold: float = 0.0075

    llm_provider: Literal["stub", "ollama", "openai_compat", "transformers"] = "stub"
    llm_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:3b-instruct-q4_K_M"
    llm_device: Literal["auto", "cpu", "cuda"] = "auto"
    llm_dtype: Literal["auto", "float32", "bfloat16", "float16"] = "auto"
    llm_temperature: float = 0.0
    llm_seed: int = 42
    llm_num_ctx: int = 4096
    llm_timeout_s: float = 120.0
    llm_max_tokens: int = 256
    llm_keep_alive: str = "30m"
    llm_fallback_model: str = ""
    llm_api_key: str = "not-needed"

    refusal_message: str = "Not found in documents."
    max_question_chars: int = 1000
    refuse_on_injection: bool = True
    gate_enabled: bool = True
    quote_check_enabled: bool = True
    grounding_enabled: bool = True

    log_level: str = "INFO"
    log_json: bool = True

    def local_model_dir(self, repo_id: str) -> Path:
        return self.models_dir / repo_id.split("/")[-1]

    def onnx_file(self, quantized: bool) -> str:
        return f"onnx/model_quint8_{self.quant_config}.onnx" if quantized else "onnx/model.onnx"


@lru_cache
def get_settings() -> Settings:
    return Settings()
