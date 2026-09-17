"""Training Agent — runs training and monitors logs for errors and anomalies."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Union

from ..data_utils import ImageDatasetBundle, TextDatasetBundle
from ..image_trainer import ImageTrainingResult, train_image_classifier
from ..model_selector import ImageModelPlan, TextModelPlan
from ..text_trainer import TextTrainingResult, train_text_qa_model
from .optimization_agent import OptimizationPlan

LogFn = Callable[[str], None]


@dataclass
class TrainingHealth:
    status: str
    oom_detected: bool
    nan_loss_detected: bool
    final_loss: float | None
    loss_trend: str
    events: list[str]


@dataclass
class TrainingOutcome:
    result: Union[TextTrainingResult, ImageTrainingResult]
    health: TrainingHealth
    agent_summary: str


class _LogMonitor:
    """Wraps the user log_fn and watches for OOM, NaN, and loss values."""

    _OOM = [r"cuda out of memory", r"out of memory", r"\boom\b", r"cudaerroroutofmemory"]
    _NAN = [r"\bnan\b", r"loss.*nan", r"gradient.*nan", r"overflow"]

    def __init__(self, inner_fn: LogFn | None) -> None:
        self.inner_fn = inner_fn
        self.oom = False
        self.nan_loss = False
        self.losses: list[float] = []
        self.events: list[str] = []

    def __call__(self, msg: str) -> None:
        if self.inner_fn:
            self.inner_fn(msg)
        else:
            print(msg, flush=True)

        lower = msg.lower()

        if not self.oom and any(re.search(p, lower) for p in self._OOM):
            self.oom = True
            self.events.append(f"OOM: {msg[:120]}")

        if not self.nan_loss and any(re.search(p, lower) for p in self._NAN):
            self.nan_loss = True
            self.events.append(f"NaN: {msg[:120]}")

        m = re.search(r"['\"]?(?:train_)?loss['\"]?\s*[=:]\s*([\d.]+)", lower)
        if m:
            try:
                self.losses.append(float(m.group(1)))
            except ValueError:
                pass

    def loss_trend(self) -> str:
        if len(self.losses) < 4:
            return "unknown"
        mid = len(self.losses) // 2
        first = sum(self.losses[:mid]) / mid
        rest = len(self.losses) - mid
        second = sum(self.losses[mid:]) / rest
        delta = second - first
        if delta < -0.05:
            return "improving"
        if delta > 0.10:
            return "diverging"
        return "stable"


class TrainingAgent:
    """Runs training and monitors for errors, loss anomalies, and OOM.

    Wraps train_text_qa_model / train_image_classifier, intercepts all
    log messages, and builds a TrainingHealth report on completion.
    """

    def train(
        self,
        task: str,
        bundle: Union[TextDatasetBundle, ImageDatasetBundle],
        plan: Union[TextModelPlan, ImageModelPlan],
        optim: OptimizationPlan,
        run_dir: Path,
        goal: str,
        merge_model: bool = False,
        log_fn: LogFn | None = None,
    ) -> TrainingOutcome:
        monitor = _LogMonitor(inner_fn=log_fn)
        monitor(
            f"[TrainingAgent] Starting {task} training — "
            f"model={plan.base_model_id}, "
            f"epochs={optim.epochs}, batch={optim.batch_size}."
        )

        try:
            result = self._run(task, bundle, plan, optim, run_dir, goal, merge_model, monitor)
        except Exception as exc:
            err = str(exc).lower()
            if "cuda out of memory" in err or "out of memory" in err:
                monitor(
                    "[TrainingAgent] CUDA OOM caught. "
                    "Try: reduce batch_size, reduce max_length, or switch to QLoRA."
                )
            raise

        final_loss = result.metrics.get("eval_loss") or result.metrics.get("train_loss")
        health = TrainingHealth(
            status="success",
            oom_detected=monitor.oom,
            nan_loss_detected=monitor.nan_loss,
            final_loss=float(final_loss) if final_loss is not None else None,
            loss_trend=monitor.loss_trend(),
            events=monitor.events,
        )

        parts = [f"[TrainingAgent] Training completed."]
        if health.final_loss is not None:
            parts.append(f"Eval loss: {health.final_loss:.4f}.")
        if health.loss_trend != "unknown":
            parts.append(f"Loss trend: {health.loss_trend}.")
        if health.oom_detected:
            parts.append("Warning: OOM event detected during training — results may be incomplete.")
        if health.nan_loss_detected:
            parts.append("Warning: NaN/overflow detected — model may be unstable.")

        summary = " ".join(parts)
        monitor(summary)

        return TrainingOutcome(result=result, health=health, agent_summary=summary)

    def _run(
        self,
        task: str,
        bundle: Any,
        plan: Any,
        optim: OptimizationPlan,
        run_dir: Path,
        goal: str,
        merge_model: bool,
        monitor: _LogMonitor,
    ) -> Any:
        if task == "text_qa":
            return train_text_qa_model(
                dataset=bundle,
                plan=plan,
                run_dir=run_dir,
                goal=goal,
                epochs=optim.epochs,
                batch_size=optim.batch_size,
                grad_accum=optim.grad_accum,
                max_length=optim.max_length,
                merge_model=merge_model,
                log_fn=monitor,
            )
        return train_image_classifier(
            dataset=bundle,
            plan=plan,
            run_dir=run_dir,
            goal=goal,
            epochs=optim.epochs,
            batch_size=optim.batch_size,
            max_train_samples=optim.max_train_samples,
            log_fn=monitor,
        )
