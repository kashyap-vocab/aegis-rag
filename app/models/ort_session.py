from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from app.models.runtime import resolve_threads, select_providers


class OrtModel:
    def __init__(self, model_dir: Path, onnx_file: str, device: str, num_threads: int, max_length: int):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = resolve_threads(num_threads)
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(model_dir / onnx_file), opts, providers=select_providers(device))
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.max_length = max_length

    def run(self, texts, text_pairs=None) -> np.ndarray:
        enc = self.tokenizer(
            texts, text_pairs, padding=True, truncation=True, max_length=self.max_length, return_tensors="np",
        )
        feed = {k: v.astype(np.int64) for k, v in enc.items() if k in self.input_names}
        if "token_type_ids" in self.input_names and "token_type_ids" not in feed:
            feed["token_type_ids"] = np.zeros_like(feed["input_ids"])
        return self.session.run(None, feed)[0]
