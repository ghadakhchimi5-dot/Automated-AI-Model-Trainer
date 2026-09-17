"""Small hardware monitor helpers used by the automatic pipeline."""

from __future__ import annotations

from typing import Any

import psutil

try:
    import torch
except Exception:  # pragma: no cover - optional dependency at import time
    torch = None  # type: ignore[assignment]

try:
    import pynvml
except Exception:  # pragma: no cover - optional dependency
    pynvml = None  # type: ignore[assignment]


def get_gpu_stats() -> dict[str, Any]:
    """Return GPU and VRAM metrics when an NVIDIA GPU is visible."""
    stats: dict[str, Any] = {
        "has_gpu": False,
        "name": None,
        "vram_used_gb": 0.0,
        "vram_total_gb": 0.0,
        "vram_pct": 0.0,
        "gpu_util_pct": 0.0,
        "gpu_temp_c": None,
    }

    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="ignore")
            stats.update(
                {
                    "has_gpu": True,
                    "name": str(name),
                    "vram_used_gb": mem.used / (1024**3),
                    "vram_total_gb": mem.total / (1024**3),
                    "vram_pct": (mem.used / mem.total) * 100 if mem.total else 0.0,
                    "gpu_util_pct": float(util.gpu),
                }
            )
            try:
                stats["gpu_temp_c"] = float(
                    pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                )
            except Exception:
                stats["gpu_temp_c"] = None
            return stats
        except Exception:
            pass
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    if torch is not None:
        try:
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                total = float(props.total_memory)
                used = float(max(torch.cuda.memory_allocated(0), torch.cuda.memory_reserved(0)))
                stats.update(
                    {
                        "has_gpu": True,
                        "name": str(props.name),
                        "vram_used_gb": used / (1024**3),
                        "vram_total_gb": total / (1024**3),
                        "vram_pct": (used / total) * 100 if total else 0.0,
                    }
                )
        except Exception:
            pass

    return stats


def get_cpu_stats() -> dict[str, float]:
    """Return CPU and RAM metrics."""
    vm = psutil.virtual_memory()
    return {
        "cpu_pct": psutil.cpu_percent(interval=0.05),
        "ram_used_gb": vm.used / (1024**3),
        "ram_total_gb": vm.total / (1024**3),
        "ram_pct": float(vm.percent),
    }
