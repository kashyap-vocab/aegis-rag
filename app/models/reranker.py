from typing import Literal

import numpy as np

from app.config import Settings
from app.models.ort_session import OrtModel
from app.models.runtime import resolve_threads
from app.schemas import RetrievedChunk

Variant = Literal["torch-fp32", "onnx-fp32", "onnx-int8"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class _Identity:
    def __call__(self, x):
        return x


class Reranker:
    def __init__(self, settings: Settings, model: str | None = None, variant: Variant | None = None):
        self.settings = settings
        self.model_name = model or settings.rerank_model
        self.variant = variant or ("onnx-int8" if settings.rerank_quantized else "onnx-fp32")
        path = settings.local_model_dir(self.model_name)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing — run `uv run scripts/download_models.py` first")

        if self.variant == "torch-fp32":
            import torch
            from sentence_transformers import CrossEncoder

            torch.set_num_threads(resolve_threads(settings.num_threads))
            self._ce = CrossEncoder(str(path), max_length=settings.rerank_max_length, local_files_only=True,
                                    activation_fn=_Identity(), backend="torch", device="cpu")
            self._ort = None
        else:
            self._ce = None
            self._ort = OrtModel(
                path, settings.onnx_file(quantized=self.variant == "onnx-int8"),
                settings.device, settings.num_threads, settings.rerank_max_length,
            )

    def score(self, question: str, passages: list[str]) -> np.ndarray:
        if not passages:
            return np.array([], dtype=np.float32)
        q = question.strip()
        docs = [p.strip() for p in passages]
        if self._ce is not None:
            logits = self._ce.predict([(q, d) for d in docs], convert_to_numpy=True, show_progress_bar=False)
        else:
            logits = self._ort.run([q] * len(docs), docs)
        return _sigmoid(np.asarray(logits, dtype=np.float32).reshape(-1))

    def rerank(self, question: str, hits: list[RetrievedChunk], top_n: int | None = None) -> list[RetrievedChunk]:
        scores = self.score(question, [h.chunk.embed_text for h in hits])
        ranked = [h.model_copy(update={"rerank_score": float(s)}) for h, s in zip(hits, scores)]
        ranked.sort(key=lambda h: h.rerank_score, reverse=True)
        return ranked[: top_n or self.settings.rerank_top_n]
