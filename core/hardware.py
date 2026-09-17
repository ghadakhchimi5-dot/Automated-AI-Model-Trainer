"""Hardware detection and training budget selection."""

from __future__ import annotations

import os
from dataclasses import dataclass

import psutil

from .gpu_monitor import get_gpu_stats

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass(frozen=True)
class HardwareInfo:
    has_gpu: bool
    gpu_name: str | None
    vram_gb: float
    ram_gb: float
    cpu_count: int
    cuda_available: bool
    bf16_supported: bool

    @property
    def tier(self) -> str:
        if not self.has_gpu:
            return "cpu"
        if self.vram_gb < 8:
            return "low_gpu"
        if self.vram_gb < 24:
            return "mid_gpu"
        return "high_gpu"


def detect_hardware() -> HardwareInfo:
    """Detect CPU/GPU resources with safe fallbacks."""
    gpu = get_gpu_stats()
    cuda_available = False
    bf16_supported = False
    if torch is not None:
        try:
            cuda_available = bool(torch.cuda.is_available())
            bf16_supported = bool(cuda_available and torch.cuda.is_bf16_supported())
        except Exception:
            cuda_available = False
            bf16_supported = False

    vm = psutil.virtual_memory()
    return HardwareInfo(
        has_gpu=bool(gpu.get("has_gpu", False) and cuda_available),
        gpu_name=str(gpu.get("name")) if gpu.get("name") else None,
        vram_gb=float(gpu.get("vram_total_gb", 0.0) or 0.0),
        ram_gb=float(vm.total / (1024**3)),
        cpu_count=int(os.cpu_count() or 1),
        cuda_available=cuda_available,
        bf16_supported=bf16_supported,
    )


def auto_training_budget(hardware: HardwareInfo, dataset_size: int, task: str) -> dict[str, int | float | bool]:
    """Return conservative defaults so a run can start without user tuning.

    max_train_samples is left uncapped (== full dataset size) for every tier:
    weaker hardware gets fewer epochs / smaller batches instead of a smaller
    training set, so the whole dataset is always used.
    """
    n = max(1, int(dataset_size))
    if task == "image_classification":
        if hardware.tier == "cpu":
            return {"epochs": 2, "batch_size": 8, "max_train_samples": n, "freeze_backbone": True}
        if hardware.tier == "low_gpu":
            return {"epochs": 3, "batch_size": 16, "max_train_samples": n, "freeze_backbone": True}
        if hardware.tier == "mid_gpu":
            return {"epochs": 4, "batch_size": 24, "max_train_samples": n, "freeze_backbone": False}
        return {"epochs": 5, "batch_size": 32, "max_train_samples": n, "freeze_backbone": False}

    # text_qa
    if hardware.tier == "cpu":
        return {"epochs": 1, "batch_size": 1, "max_train_samples": n, "max_length": 384, "grad_accum": 8}
    if hardware.tier == "low_gpu":
        return {"epochs": 2, "batch_size": 1, "max_train_samples": n, "max_length": 512, "grad_accum": 8}
    if hardware.tier == "mid_gpu":
        return {"epochs": 3, "batch_size": 2, "max_train_samples": n, "max_length": 768, "grad_accum": 4}
    return {"epochs": 3, "batch_size": 4, "max_train_samples": n, "max_length": 1024, "grad_accum": 2}
