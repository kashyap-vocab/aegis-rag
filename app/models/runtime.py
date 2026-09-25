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
    available = set(available_providers())
    order = _AUTO_ORDER if device == "auto" else [device, "cpu"]
    chosen = [_PREFERENCE[d] for d in order if _PREFERENCE[d] in available]
    if "CPUExecutionProvider" not in chosen:
        chosen.append("CPUExecutionProvider")
    return chosen


def resolve_threads(num_threads: int) -> int:
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
