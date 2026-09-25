"""The ONLY step that needs network access (R1).

Downloads the embedder and reranker into ./models, exports ONNX if the repo
doesn't ship it, builds int8 dynamically-quantized variants, and writes a
SHA-256 manifest so the air-gapped side can verify integrity after transfer.

    uv run scripts/download_models.py
"""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import snapshot_download
from sentence_transformers import CrossEncoder, SentenceTransformer
from sentence_transformers.backend import export_dynamic_quantized_onnx_model

from app.config import get_settings

# Weights we never load: skip them to keep the offline bundle small.
IGNORE = ["*.h5", "*.msgpack", "*.ot", "*.onnx_data", "onnx/*", "openvino/*", "*.bin"]


def fetch(repo_id: str, target: Path, cls) -> None:
    print(f"==> {repo_id} -> {target}")
    snapshot_download(repo_id, local_dir=target, ignore_patterns=IGNORE)

    onnx = target / "onnx" / "model.onnx"
    if not onnx.exists():
        print("    exporting ONNX (fp32)")
        model = cls(str(target), backend="onnx", model_kwargs={"export": True})
        model.save_pretrained(str(target))


def quantize(target: Path, cls, config: str) -> None:
    out = target / "onnx" / f"model_quint8_{config}.onnx"
    if out.exists():
        print(f"    int8 ({config}) already present")
        return
    print(f"    quantizing int8 ({config})")
    model = cls(str(target), backend="onnx", model_kwargs={"file_name": "onnx/model.onnx"})
    export_dynamic_quantized_onnx_model(model, quantization_config=config, model_name_or_path=str(target))


def write_manifest(models_dir: Path) -> None:
    manifest = {
        str(p.relative_to(models_dir)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(models_dir.rglob("*"))
        if p.is_file() and p.name != "MANIFEST.json" and ".cache" not in p.parts
    }
    (models_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"==> manifest: {len(manifest)} files")


def main() -> None:
    s = get_settings()
    s.models_dir.mkdir(parents=True, exist_ok=True)
    for repo_id, cls in [(s.embed_model, SentenceTransformer), (s.rerank_model, CrossEncoder)]:
        target = s.local_model_dir(repo_id)
        fetch(repo_id, target, cls)
        quantize(target, cls, s.quant_config)
    write_manifest(s.models_dir)


if __name__ == "__main__":
    main()
