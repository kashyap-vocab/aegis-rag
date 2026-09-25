"""Dense embedder (D3): bge-small-en-v1.5 via sentence-transformers, optimised.

Loads strictly from the local ``models/`` directory — never from the network —
so the same code runs in Docker, on Kaggle with internet off, or air-gapped.

Speed levers (all config-driven): ONNX Runtime backend, int8 dynamic
quantization, capped sequence length, thread count, provider selection, and an
LRU cache for repeated queries.
"""

from functools import lru_cache
from typing import Literal

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import Settings
from app.models.runtime import resolve_threads, select_providers

Variant = Literal["torch-fp32", "onnx-fp32", "onnx-int8"]


def default_variant(settings: Settings) -> Variant:
    if settings.embed_backend == "torch":
        return "torch-fp32"
    return "onnx-int8" if settings.embed_quantized else "onnx-fp32"


def load_sentence_transformer(settings: Settings, variant: Variant) -> SentenceTransformer:
    path = settings.local_model_dir(settings.embed_model)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run `uv run scripts/download_models.py` first")
    threads = resolve_threads(settings.num_threads)

    if variant == "torch-fp32":
        import torch

        torch.set_num_threads(threads)
        return SentenceTransformer(str(path), backend="torch", device="cpu", local_files_only=True)

    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return SentenceTransformer(
        str(path),
        backend="onnx",
        local_files_only=True,
        model_kwargs={
            "file_name": settings.onnx_file(quantized=variant == "onnx-int8"),
            "provider": select_providers(settings.device)[0],
            "session_options": opts,
        },
    )


class Embedder:
    def __init__(self, settings: Settings, variant: Variant | None = None):
        self.settings = settings
        self.variant = variant or default_variant(settings)
        self.model = load_sentence_transformer(settings, self.variant)
        self.model.max_seq_length = settings.embed_max_seq_length
        self.dim = self.model.get_embedding_dimension()
        self._cached_query = lru_cache(maxsize=settings.query_cache_size)(self._encode_query)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        # bge-v1.5: passages get no instruction prefix.
        return self.model.encode(
            texts,
            batch_size=self.settings.embed_batch_size,
            normalize_embeddings=True,  # cosine == inner product -> FAISS IndexFlatIP
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

    def _encode_query(self, text: str) -> np.ndarray:
        vec = self.model.encode(
            [self.settings.embed_query_prefix + text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vec = vec.astype(np.float32)
        vec.setflags(write=False)  # cached arrays are shared; keep them immutable
        return vec

    def encode_query(self, text: str, use_cache: bool = True) -> np.ndarray:
        return self._cached_query(text) if use_cache else self._encode_query(text)
