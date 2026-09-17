"""Data Agent — detects task, prepares data, and assesses quality."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Union

from ..data_utils import (
    ImageDatasetBundle,
    TextDatasetBundle,
    detect_task,
    prepare_image_dataset,
    prepare_text_dataset,
)

LogFn = Callable[[str], None]


@dataclass
class DataAnalysis:
    task: str
    bundle: Union[TextDatasetBundle, ImageDatasetBundle]
    quality_score: float
    quality_label: str
    warnings: list[str]
    recommendations: list[str]
    reasoning: str
    report: dict[str, Any]


class DataAgent:
    """Detects the dataset type, prepares data, and assesses quality.

    Wraps detect_task + prepare_text_dataset / prepare_image_dataset.
    Returns a DataAnalysis with the prepared bundle and diagnostic info.
    """

    def run(
        self,
        data_path: Union[str, Path],
        goal: str,
        user_task: str = "auto",
        max_samples: int | None = None,
        run_dir: Path | None = None,
        log_fn: LogFn | None = None,
    ) -> DataAnalysis:
        path = Path(data_path)
        task = detect_task(path, user_task=user_task, goal=goal)

        warnings: list[str] = []
        recommendations: list[str] = []
        reasoning_parts: list[str] = []

        if user_task != "auto":
            reasoning_parts.append(f"Task overridden by user to '{task}'.")
        else:
            reasoning_parts.append(f"Task auto-detected as '{task}'.")

        if task == "text_qa":
            bundle, quality_score = self._prepare_text(
                path, goal, max_samples, warnings, recommendations, reasoning_parts, log_fn
            )
        else:
            if run_dir is None:
                raise ValueError("run_dir is required for image classification data preparation")
            bundle, quality_score = self._prepare_image(
                path, run_dir, warnings, recommendations, reasoning_parts
            )

        quality_label = _quality_label(quality_score)
        reasoning = " ".join(reasoning_parts)

        _log(log_fn, f"[DataAgent] {reasoning}")
        for w in warnings:
            _log(log_fn, f"[DataAgent] Warning: {w}")
        for r in recommendations:
            _log(log_fn, f"[DataAgent] Recommendation: {r}")

        return DataAnalysis(
            task=task,
            bundle=bundle,
            quality_score=quality_score,
            quality_label=quality_label,
            warnings=warnings,
            recommendations=recommendations,
            reasoning=reasoning,
            report=bundle.report,
        )

    def _prepare_text(
        self,
        path: Path,
        goal: str,
        max_samples: int | None,
        warnings: list[str],
        recommendations: list[str],
        reasoning_parts: list[str],
        log_fn: LogFn | None = None,
    ) -> tuple[TextDatasetBundle, float]:
        bundle = prepare_text_dataset(path, goal=goal, max_samples=max_samples, log_fn=log_fn)
        report = bundle.report
        quality_score = float(report.get("quality_score", 0.0))

        raw_rows = int(report.get("raw_rows", 0))
        train_rows = int(report.get("train_rows", 0))
        eval_rows = int(report.get("eval_rows", 0))
        duplicates = int(report.get("duplicates", 0))
        avg_resp = float(report.get("avg_response_chars", 0))

        reasoning_parts.append(
            f"Loaded {raw_rows} raw rows → {train_rows} train / {eval_rows} eval after cleaning. "
            f"Quality score: {quality_score:.0f}/100 ({_quality_label(quality_score)})."
        )

        eval_size_warning = report.get("eval_size_warning")
        if eval_size_warning:
            warnings.append(eval_size_warning)

        if bundle.generated_qa:
            warnings.append(
                "No answer column found — synthetic extractive QA pairs were auto-generated from text."
            )
            recommendations.append(
                "For better model quality, provide a dataset with explicit 'instruction'/'output' columns."
            )

        if duplicates > 0:
            warnings.append(f"{duplicates} duplicate rows detected and removed.")

        if train_rows < 50:
            warnings.append(
                f"Very small training set ({train_rows} rows). Fine-tuning results will be unreliable."
            )
            recommendations.append(
                "Aim for at least 200 training examples for stable fine-tuning."
            )
        elif train_rows < 200:
            recommendations.append(
                f"Training set is small ({train_rows} rows). More examples would improve generalization."
            )

        if avg_resp < 20:
            warnings.append(
                "Responses are very short on average (<20 chars). "
                "The model may learn to produce terse or unhelpful outputs."
            )

        return bundle, quality_score

    def _prepare_image(
        self,
        path: Path,
        run_dir: Path,
        warnings: list[str],
        recommendations: list[str],
        reasoning_parts: list[str],
    ) -> tuple[ImageDatasetBundle, float]:
        bundle = prepare_image_dataset(path, work_dir=run_dir / "prepared")
        report = bundle.report
        num_images = int(report.get("num_images", 0))
        num_classes = int(report.get("num_classes", 0))
        labels = report.get("labels", [])

        quality_score = _image_quality_score(num_images)

        reasoning_parts.append(
            f"Loaded {num_images} images across {num_classes} classes: {labels}. "
            f"Quality score: {quality_score:.0f}/100 ({_quality_label(quality_score)})."
        )

        if num_images < 100:
            warnings.append(
                f"Only {num_images} images total — very limited training data."
            )
            recommendations.append(
                "Aim for at least 100 images per class for reliable classification."
            )
        elif num_images < 500:
            recommendations.append(
                "Moderate dataset size. More images would improve accuracy and generalization."
            )

        if num_classes > 20:
            warnings.append(
                f"{num_classes} classes detected. Many-class problems are harder; "
                "consider grouping similar categories."
            )

        return bundle, quality_score


# ── helpers ──────────────────────────────────────────────────────────────────

def _quality_label(score: float) -> str:
    if score >= 80:
        return "excellent"
    if score >= 60:
        return "good"
    if score >= 40:
        return "fair"
    return "poor"


def _image_quality_score(num_images: int) -> float:
    if num_images >= 1000:
        return 90.0
    if num_images >= 500:
        return 75.0
    if num_images >= 200:
        return 55.0
    if num_images >= 50:
        return 35.0
    return 15.0


def _log(log_fn: LogFn | None, msg: str) -> None:
    if log_fn:
        log_fn(msg)
    else:
        print(msg, flush=True)
