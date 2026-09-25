"""Hardware detection (D2): pick the best ONNX Runtime execution provider.

No hardware is assumed. We ask ONNX Runtime what is available and fall back
along a fixed preference order, always ending on CPU.
"""

import os

import onnxruntime as ort

_PREFERENCE = {
    "cuda": "CUDAExecutionProvider",
    "dml": "DmlExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "cpu": "CPUExecutionProvider",
}
_AUTO_ORDER = ["cuda", "dml", "openvino", "cpu"]


def available_providers() -> list[str]:
    return ort.get_available_providers()


def select_providers(device: str = "auto") -> list[str]:
    """Return an ordered provider list for ONNX Runtime sessions."""
    available = set(available_providers())
    order = _AUTO_ORDER if device == "auto" else [device, "cpu"]
    chosen = [_PREFERENCE[d] for d in order if _PREFERENCE[d] in available]
    if "CPUExecutionProvider" not in chosen:
        chosen.append("CPUExecutionProvider")
    return chosen


def resolve_threads(num_threads: int) -> int:
    """0 means 'physical cores' — hyperthreads rarely help matmul-bound inference."""
    if num_threads > 0:
        return num_threads
    logical = os.cpu_count() or 1
    return max(1, logical // 2)


def describe(device: str = "auto", num_threads: int = 0) -> dict:
    return {
        "onnxruntime": ort.__version__,
        "available": available_providers(),
        "selected": select_providers(device),
        "threads": resolve_threads(num_threads),
    }
