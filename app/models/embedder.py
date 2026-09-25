from functools import lru_cache
from typing import Literal

import numpy as np

from app.config import Settings
from app.models.ort_session import OrtModel
from app.models.runtime import resolve_threads

Variant = Literal["torch-fp32", "onnx-fp32", "onnx-int8"]


def default_variant(settings: Settings) -> Variant:
    if settings.embed_backend == "torch":
        return "torch-fp32"
    return "onnx-int8" if settings.embed_quantized else "onnx-fp32"


class Embedder:
    def __init__(self, settings: Settings, variant: Variant | None = None):
        self.settings = settings
        self.variant = variant or default_variant(settings)
        path = settings.local_model_dir(settings.embed_model)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing — run `uv run scripts/download_models.py` first")

        if self.variant == "torch-fp32":
            import torch
            from sentence_transformers import SentenceTransformer

            torch.set_num_threads(resolve_threads(settings.num_threads))
            self._st = SentenceTransformer(str(path), backend="torch", device="cpu", local_files_only=True)
            self._st.max_seq_length = settings.embed_max_seq_length
            self._ort = None
            self.dim = self._st[0].auto_model.config.hidden_size
        else:
            self._st = None
            self._ort = OrtModel(
                path, settings.onnx_file(quantized=self.variant == "onnx-int8"),
                settings.device, settings.num_threads, settings.embed_max_seq_length,
            )
            self.dim = int(self._ort.session.get_outputs()[0].shape[-1])
        self._cached_query = lru_cache(maxsize=settings.query_cache_size)(self._encode_query)

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self._st is not None:
            vecs = self._st.encode(texts, batch_size=self.settings.embed_batch_size, normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False)
            return vecs.astype(np.float32)
        out = []
        bs = self.settings.embed_batch_size
        for i in range(0, len(texts), bs):
            cls = self._ort.run(texts[i : i + bs])[:, 0, :]
            out.append(cls / np.linalg.norm(cls, axis=1, keepdims=True))
        return np.vstack(out).astype(np.float32)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def _encode_query(self, text: str) -> np.ndarray:
        vec = self._encode([self.settings.embed_query_prefix + text])
        vec.setflags(write=False)
        return vec

    def encode_query(self, text: str, use_cache: bool = True) -> np.ndarray:
        return self._cached_query(text) if use_cache else self._encode_query(text)
