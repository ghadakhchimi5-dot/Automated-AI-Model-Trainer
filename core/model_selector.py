"""Automatic model and training-strategy selection."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .hardware import HardwareInfo


@dataclass(frozen=True)
class TextModelPlan:
    task: str
    base_model_id: str
    model_params_b: float
    peft_method: str
    train_type: str
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    quantization: str
    load_in_4bit: bool
    load_in_8bit: bool
    torch_dtype: str
    learning_rate: float
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ImageModelPlan:
    task: str
    base_model_id: str
    freeze_backbone: bool
    learning_rate: float
    image_size: int
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def infer_params_b(model_id: str, default: float = 1.0) -> float:
    match = re.search(r"(\d+(?:\.\d+)?)\s*b", model_id.lower())
    if not match:
        return default
    try:
        return float(match.group(1))
    except Exception:
        return default


def _load_catalog(catalog_path: str | Path | None = None) -> list[dict[str, Any]]:
    candidates = []
    if catalog_path:
        candidates.append(Path(catalog_path))
    candidates.append(Path(__file__).resolve().parents[1] / "configs" / "models_catalog.yaml")
    for path in candidates:
        if path.exists():
            try:
                payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                models = payload.get("models", [])
                if isinstance(models, list):
                    return [m for m in models if isinstance(m, dict)]
            except Exception:
                continue
    return []


def _target_modules(model_id: str) -> list[str]:
    name = model_id.lower()
    if "qwen" in name or "llama" in name or "mistral" in name:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if "phi" in name:
        return ["q_proj", "v_proj", "dense"]
    if "gpt" in name:
        return ["c_attn", "c_proj"]
    if "smollm" in name:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def select_text_model(
    hardware: HardwareInfo,
    dataset_size: int,
    priority: str = "balanced",
    catalog_path: str | Path | None = None,
    override_model_id: str | None = None,
) -> TextModelPlan:
    """Choose an SLM and PEFT settings for local resources."""
    if override_model_id:
        base_model_id = override_model_id
        params_b = infer_params_b(base_model_id, default=1.0)
    else:
        catalog = _load_catalog(catalog_path)
        # Extra small fallback models. These are useful on CPU / very low VRAM.
        catalog = [
            {
                "id": "HuggingFaceTB/SmolLM2-135M-Instruct",
                "name": "SmolLM2 135M Instruct",
                "params_b": 0.135,
                "vram_min_gb": 1,
                "vram_qlora_gb": 1,
                "recommended_for": ["cpu", "demo"],
            },
            {
                "id": "HuggingFaceTB/SmolLM2-360M-Instruct",
                "name": "SmolLM2 360M Instruct",
                "params_b": 0.36,
                "vram_min_gb": 2,
                "vram_qlora_gb": 1,
                "recommended_for": ["low_vram", "demo"],
            },
        ] + catalog

        def is_compatible(m: dict[str, Any]) -> bool:
            params = float(m.get("params_b", infer_params_b(str(m.get("id", "")), 1.0)))
            if hardware.tier == "cpu":
                return params <= 0.6 or "cpu" in m.get("recommended_for", [])
            needed = float(m.get("vram_qlora_gb", m.get("vram_min_gb", 999)))
            return hardware.vram_gb >= needed

        compatible = [m for m in catalog if is_compatible(m)] or catalog[:1]
        if priority == "performance" and hardware.tier in {"mid_gpu", "high_gpu"}:
            selected = max(compatible, key=lambda m: float(m.get("params_b", 0.0)))
        elif priority == "memory":
            selected = min(compatible, key=lambda m: float(m.get("params_b", 999.0)))
        else:
            # Balanced: choose the largest model that still leaves a memory margin.
            selected = sorted(compatible, key=lambda m: float(m.get("params_b", 0.0)))[len(compatible) // 2]
        base_model_id = str(selected["id"])
        params_b = float(selected.get("params_b", infer_params_b(base_model_id, default=1.0)))

    # PEFT rules. They intentionally mirror the previous Auto-PEFT behavior but are CLI/UI agnostic.
    peft_method = "lora"
    rank = 16
    alpha = 32
    dropout = 0.05
    quantization = "none"
    explanation_parts: list[str] = []

    if hardware.tier == "cpu":
        rank, alpha, dropout = 4, 8, 0.1
        explanation_parts.append("CPU mode: selected a compact SLM and light LoRA rank to stay executable.")
    elif hardware.vram_gb < 8:
        if params_b > 1.5:
            peft_method, quantization, rank, alpha = "qlora", "4bit", 8, 16
            explanation_parts.append("Low VRAM: using QLoRA 4-bit to reduce memory.")
        else:
            rank, alpha = 8, 16
            explanation_parts.append("Low VRAM with compact model: using light LoRA.")
    elif hardware.vram_gb < 24:
        if params_b >= 7 or priority == "memory":
            peft_method, quantization, rank, alpha = "qlora", "4bit", 16, 32
            explanation_parts.append("Mid VRAM: using QLoRA for memory headroom.")
        elif priority == "performance":
            peft_method, rank, alpha = "lora", 32, 64
            explanation_parts.append("Mid VRAM and performance priority: using higher-rank LoRA.")
        else:
            rank, alpha = 16, 32
            explanation_parts.append("Mid VRAM: balanced LoRA settings.")
    else:
        if priority == "performance":
            rank, alpha = 64, 128
            explanation_parts.append("High VRAM: using high-rank LoRA for adaptation quality.")
        else:
            rank, alpha = 32, 64
            explanation_parts.append("High VRAM: using standard high-capacity LoRA.")

    if dataset_size < 200:
        rank = min(rank, 8)
        dropout = max(dropout, 0.1)
        explanation_parts.append("Small dataset: capped rank and increased dropout to reduce overfitting.")

    dtype = "float32"
    if hardware.has_gpu:
        dtype = "bfloat16" if hardware.bf16_supported else "float16"

    return TextModelPlan(
        task="text_qa",
        base_model_id=base_model_id,
        model_params_b=params_b,
        peft_method=peft_method,
        train_type="qlora" if peft_method == "qlora" else "lora",
        lora_rank=int(rank),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=_target_modules(base_model_id),
        quantization=quantization,
        load_in_4bit=quantization == "4bit" and hardware.has_gpu,
        load_in_8bit=quantization == "8bit" and hardware.has_gpu,
        torch_dtype=dtype,
        learning_rate=2e-4 if params_b <= 0.5 else 1e-4,
        explanation=" ".join(explanation_parts),
    )


def select_image_model(
    hardware: HardwareInfo,
    num_images: int,
    priority: str = "balanced",
    override_model_id: str | None = None,
) -> ImageModelPlan:
    """Choose an image classifier and fine-tuning strategy."""
    if override_model_id:
        model_id = override_model_id
    elif hardware.tier == "cpu":
        model_id = "microsoft/resnet-18"
    elif hardware.tier == "low_gpu":
        model_id = "microsoft/resnet-18"
    elif priority == "performance" and hardware.tier == "high_gpu":
        model_id = "google/vit-base-patch16-224-in21k"
    else:
        model_id = "microsoft/resnet-50"

    freeze = hardware.tier in {"cpu", "low_gpu"} or num_images < 300
    lr = 5e-4 if freeze else 2e-5
    explanation = (
        f"Selected {model_id}. "
        f"Backbone {'frozen' if freeze else 'trainable'} based on hardware tier={hardware.tier} and images={num_images}."
    )
    return ImageModelPlan(
        task="image_classification",
        base_model_id=model_id,
        freeze_backbone=freeze,
        learning_rate=lr,
        image_size=224,
        explanation=explanation,
    )
    
