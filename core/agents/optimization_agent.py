"""Optimization Agent — selects and validates training hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

from ..hardware import HardwareInfo, auto_training_budget
from ..model_selector import ImageModelPlan, TextModelPlan

LogFn = Callable[[str], None]


@dataclass
class OptimizationPlan:
    epochs: int
    batch_size: int
    grad_accum: int
    max_length: int
    max_train_samples: int
    learning_rate: float
    reasoning: str
    warnings: list[str]
    overrides_applied: list[str]


class OptimizationAgent:
    """Selects and validates hyperparameters.

    Starts from auto_training_budget defaults, applies user overrides,
    then makes data-quality-driven adjustments (e.g. reducing epochs
    for noisy datasets or very small training sets).
    """

    def optimize(
        self,
        task: str,
        hardware: HardwareInfo,
        dataset_size: int,
        plan: Union[TextModelPlan, ImageModelPlan],
        quality_score: float = 80.0,
        user_epochs: int | None = None,
        user_batch_size: int | None = None,
        user_max_samples: int | None = None,
        user_max_length: int | None = None,
        log_fn: LogFn | None = None,
    ) -> OptimizationPlan:
        budget = auto_training_budget(hardware, dataset_size, task)
        warnings: list[str] = []
        overrides: list[str] = []
        reasoning_parts: list[str] = []

        epochs = int(budget["epochs"])
        batch_size = int(budget["batch_size"])
        grad_accum = int(budget.get("grad_accum", 1))
        max_length = int(budget.get("max_length", 512))
        max_train_samples = int(budget["max_train_samples"])
        lr = float(getattr(plan, "learning_rate", 2e-4))

        # Apply user overrides
        if user_epochs is not None:
            overrides.append(f"epochs {epochs}→{user_epochs}")
            epochs = int(user_epochs)
        if user_batch_size is not None:
            overrides.append(f"batch_size {batch_size}→{user_batch_size}")
            batch_size = int(user_batch_size)
        if user_max_samples is not None:
            overrides.append(f"max_train_samples {max_train_samples}→{user_max_samples}")
            max_train_samples = int(user_max_samples)
        if user_max_length is not None and task == "text_qa":
            overrides.append(f"max_length {max_length}→{user_max_length}")
            max_length = int(user_max_length)

        # Data-quality-driven adjustments (only when user hasn't overridden)
        if user_epochs is None:
            if quality_score < 40 and task == "text_qa":
                adjusted = max(1, epochs - 1)
                if adjusted != epochs:
                    warnings.append(
                        f"Low data quality score ({quality_score:.0f}/100): "
                        f"reducing epochs from {epochs} to {adjusted} to avoid memorising noise."
                    )
                    epochs = adjusted

            if dataset_size < 100 and task == "text_qa":
                adjusted = min(epochs, 2)
                if adjusted != epochs:
                    warnings.append(
                        f"Very small dataset ({dataset_size} examples): "
                        f"capping epochs at {adjusted} to prevent severe overfitting."
                    )
                    epochs = adjusted

        reasoning_parts.append(
            f"Base hyperparameters from hardware tier '{hardware.tier}': "
            f"epochs={epochs}, batch_size={batch_size}"
            + (f", grad_accum={grad_accum}, max_length={max_length}" if task == "text_qa" else "")
            + f", max_train_samples={max_train_samples}."
        )

        if overrides:
            reasoning_parts.append("User overrides: " + ", ".join(overrides) + ".")

        reasoning = " ".join(reasoning_parts)
        _log(log_fn, f"[OptimizationAgent] {reasoning}")
        for w in warnings:
            _log(log_fn, f"[OptimizationAgent] Warning: {w}")

        return OptimizationPlan(
            epochs=epochs,
            batch_size=batch_size,
            grad_accum=grad_accum,
            max_length=max_length,
            max_train_samples=max_train_samples,
            learning_rate=lr,
            reasoning=reasoning,
            warnings=warnings,
            overrides_applied=overrides,
        )


def _log(log_fn: LogFn | None, msg: str) -> None:
    if log_fn:
        log_fn(msg)
    else:
        print(msg, flush=True)
