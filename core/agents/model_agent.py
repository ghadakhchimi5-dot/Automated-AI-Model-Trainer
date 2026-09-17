"""Model Agent — selects the best model and PEFT strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Union

from ..hardware import HardwareInfo
from ..model_selector import (
    ImageModelPlan,
    TextModelPlan,
    select_image_model,
    select_text_model,
)

LogFn = Callable[[str], None]


@dataclass
class ModelDecision:
    plan: Union[TextModelPlan, ImageModelPlan]
    reasoning: str
    alternatives_considered: list[str]
    warnings: list[str]


class ModelAgent:
    """Chooses model architecture and PEFT strategy.

    Wraps select_text_model / select_image_model and adds transparent
    reasoning about the choices made and alternatives considered.
    """

    def select(
        self,
        task: str,
        hardware: HardwareInfo,
        dataset_size: int,
        priority: str = "balanced",
        override_model_id: str | None = None,
        log_fn: LogFn | None = None,
    ) -> ModelDecision:
        warnings: list[str] = []
        alternatives: list[str] = []

        if task == "text_qa":
            plan = select_text_model(
                hardware=hardware,
                dataset_size=dataset_size,
                priority=priority,
                override_model_id=override_model_id,
            )
            reasoning = self._text_reasoning(
                plan, hardware, dataset_size, priority, override_model_id, warnings, alternatives
            )
        else:
            plan = select_image_model(
                hardware=hardware,
                num_images=dataset_size,
                priority=priority,
                override_model_id=override_model_id,
            )
            reasoning = self._image_reasoning(
                plan, hardware, dataset_size, priority, override_model_id, warnings, alternatives
            )

        _log(log_fn, f"[ModelAgent] {reasoning}")
        for w in warnings:
            _log(log_fn, f"[ModelAgent] Warning: {w}")
        for a in alternatives:
            _log(log_fn, f"[ModelAgent] Alternative considered: {a}")

        return ModelDecision(
            plan=plan,
            reasoning=reasoning,
            alternatives_considered=alternatives,
            warnings=warnings,
        )

    def _text_reasoning(
        self,
        plan: TextModelPlan,
        hardware: HardwareInfo,
        dataset_size: int,
        priority: str,
        override: str | None,
        warnings: list[str],
        alternatives: list[str],
    ) -> str:
        parts: list[str] = []

        if override:
            parts.append(f"Model overridden by user: {plan.base_model_id}.")
        else:
            parts.append(
                f"Selected '{plan.base_model_id}' ({plan.model_params_b:.3g}B params) "
                f"for hardware tier '{hardware.tier}' ({hardware.vram_gb:.1f} GB VRAM, priority='{priority}')."
            )

        peft = plan.peft_method.upper()
        parts.append(
            f"PEFT: {peft} — rank={plan.lora_rank}, alpha={plan.lora_alpha}, dropout={plan.lora_dropout}."
        )

        if plan.quantization != "none":
            parts.append(
                f"Quantization '{plan.quantization}' applied to reduce GPU memory footprint."
            )

        if dataset_size < 200:
            parts.append(
                f"Small dataset ({dataset_size} examples): LoRA rank capped and dropout increased to resist overfitting."
            )

        if hardware.tier == "cpu":
            warnings.append(
                "CPU training will be very slow — expect hours even for small datasets. "
                "Use a GPU machine for practical training times."
            )
        elif hardware.tier == "low_gpu" and plan.model_params_b > 1.0:
            alternatives.append(
                "A smaller model (e.g. SmolLM2-135M) would be faster but less capable on this hardware."
            )

        parts.append(plan.explanation)
        return " ".join(parts)

    def _image_reasoning(
        self,
        plan: ImageModelPlan,
        hardware: HardwareInfo,
        dataset_size: int,
        priority: str,
        override: str | None,
        warnings: list[str],
        alternatives: list[str],
    ) -> str:
        parts: list[str] = []

        if override:
            parts.append(f"Model overridden by user: {plan.base_model_id}.")
        else:
            parts.append(
                f"Selected image model '{plan.base_model_id}' "
                f"for hardware tier '{hardware.tier}' ({hardware.vram_gb:.1f} GB VRAM)."
            )

        backbone_status = "frozen" if plan.freeze_backbone else "trainable"
        parts.append(
            f"Backbone is {backbone_status} (lr={plan.learning_rate:.0e})."
        )

        if plan.freeze_backbone and dataset_size < 300:
            parts.append(
                f"Backbone frozen because dataset has only {dataset_size} images "
                f"(threshold: 300 images for full fine-tuning)."
            )
            alternatives.append(
                "Provide 300+ images to enable full backbone fine-tuning for better accuracy."
            )
        elif plan.freeze_backbone and hardware.tier in {"cpu", "low_gpu"}:
            parts.append(
                f"Backbone frozen to conserve memory on '{hardware.tier}' hardware."
            )
            alternatives.append(
                "Use a GPU with ≥8 GB VRAM to enable full backbone fine-tuning."
            )

        if hardware.tier == "cpu":
            warnings.append(
                "Image training on CPU will be slow. "
                "Even frozen-backbone training on ResNet-18 can take tens of minutes per epoch."
            )

        parts.append(plan.explanation)
        return " ".join(parts)


def _log(log_fn: LogFn | None, msg: str) -> None:
    if log_fn:
        log_fn(msg)
    else:
        print(msg, flush=True)
