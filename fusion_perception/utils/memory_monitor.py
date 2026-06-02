"""GPU and CPU memory monitoring. Logs warnings before OOM."""
from __future__ import annotations
from typing import Optional
from fusion_perception.utils.logging_setup import get_logger

logger = get_logger("memory_monitor")

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def log_gpu_memory(tag: str = "") -> dict:
    """Log current GPU memory usage. Returns dict with stats."""
    if not _TORCH_AVAILABLE or not torch.cuda.is_available():
        return {}

    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    free = total - reserved

    label = f"[{tag}] " if tag else ""
    logger.debug(
        f"{label}GPU: {allocated:.2f}GB alloc | "
        f"{reserved:.2f}GB reserved | "
        f"{free:.2f}GB free / {total:.2f}GB total"
    )

    if free < 1.0:
        logger.warning(f"{label}GPU memory low: {free:.2f}GB free — OOM risk")

    return {
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "free_gb": free,
        "total_gb": total,
    }


def clear_gpu_cache() -> None:
    """Free unused cached GPU memory."""
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("GPU cache cleared")
