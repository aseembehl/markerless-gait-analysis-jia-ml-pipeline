"""Choice of compute device for the neural-network stages."""

from __future__ import annotations


def resolve_device(device: str, allow_mps: bool) -> str:
    """Turn ``"auto"`` into ``cuda``, ``mps`` (if ``allow_mps``) or ``cpu``.

    On Apple silicon, ViTPose runs ~3.7x faster on MPS than on the CPU with
    identical results, whereas torchvision's Faster R-CNN is far slower on MPS
    (ops fall back to the CPU), hence ``allow_mps`` per model.
    """
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if allow_mps and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
