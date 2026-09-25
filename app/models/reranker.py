"""Cross-encoder reranker (D5).

A bi-encoder embeds question and chunk separately; a cross-encoder reads them
*jointly* with full attention, so it can tell "manufacturer of the coolant"
apart from "replacement interval of the coolant". It is too slow to scan a
whole corpus, but cheap on the handful of fused candidates.

Scores are passed through a sigmoid -> [0, 1], which makes them usable as a
retrieval-confidence signal for the refusal gate (D8).

Same optimisation levers as the embedder: ONNX Runtime, int8, threads,
provider selection — and local files only.
"""

from typing import Literal

import numpy as np
from sentence_transformers import CrossEncoder

from app.config import Settings
from app.models.runtime import resolve_threads, select_providers
from app.schemas import RetrievedChunk

Variant = Literal["torch-fp32", "onnx-fp32", "onnx-int8"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class Reranker:
    def __init__(self, settings: Settings, model: str | None = None, variant: Variant | None = None):
        self.settings = settings
        self.model_name = model or settings.rerank_model
        self.variant = variant or ("onnx-int8" if settings.rerank_quantized else "onnx-fp32")
        path = settings.local_model_dir(self.model_name)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing — run `uv run scripts/download_models.py` first")
        threads = resolve_threads(settings.num_threads)

        if self.variant == "torch-fp32":
            import torch

            torch.set_num_threads(threads)
            kwargs = {"backend": "torch", "device": "cpu"}
        else:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = threads
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            kwargs = {
                "backend": "onnx",
                "model_kwargs": {
                    "file_name": settings.onnx_file(quantized=self.variant == "onnx-int8"),
                    "provider": select_providers(settings.device)[0],
                    "session_options": opts,
                },
            }
        # Raw logits out; we apply the sigmoid ourselves so every model/variant
        # is scored identically regardless of its default activation.
        self.model = CrossEncoder(
            str(path), max_length=settings.rerank_max_length, local_files_only=True,
            activation_fn=_Identity(), **kwargs,
        )

    def score(self, question: str, passages: list[str]) -> np.ndarray:
        if not passages:
            return np.array([], dtype=np.float32)
        pairs = [(question.strip(), p.strip()) for p in passages]
        logits = self.model.predict(pairs, convert_to_numpy=True, show_progress_bar=False)
        return _sigmoid(np.asarray(logits, dtype=np.float32).reshape(-1))

    def rerank(self, question: str, hits: list[RetrievedChunk], top_n: int | None = None) -> list[RetrievedChunk]:
        """Score candidates on their full ``embed_text`` (title > section + body), best first."""
        scores = self.score(question, [h.chunk.embed_text for h in hits])
        ranked = [h.model_copy(update={"rerank_score": float(s)}) for h, s in zip(hits, scores)]
        ranked.sort(key=lambda h: h.rerank_score, reverse=True)
        return ranked[: top_n or self.settings.rerank_top_n]


class _Identity:
    """Picklable no-op activation (torch.nn.Identity would pull torch into ONNX-only runs)."""

    def __call__(self, x):
        return x
