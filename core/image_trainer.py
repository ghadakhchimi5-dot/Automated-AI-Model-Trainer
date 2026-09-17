"""Automated image-classification fine-tuning."""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from datasets import ClassLabel, Dataset, Features, Image
from transformers import AutoImageProcessor, AutoModelForImageClassification, Trainer, TrainingArguments

from .data_utils import IMAGE_EXTENSIONS, ImageDatasetBundle, ensure_dir
from .metrics_plots import plot_roc_curves
from .metrics_utils import compute_image_classification_metrics, save_metrics
from .model_selector import ImageModelPlan
from .packaging import write_json

LogFn = Callable[[str], None]


@dataclass(frozen=True)
class ImageTrainingResult:
    run_dir: Path
    model_dir: Path
    manifest_path: Path
    report_path: Path
    metrics: dict[str, Any]


def _log(log_fn: LogFn | None, message: str) -> None:
    if log_fn:
        log_fn(message)
    else:
        print(message, flush=True)


def _training_args_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(TrainingArguments.__init__)
    params = set(sig.parameters.keys())
    out = dict(kwargs)

    if "eval_strategy" in params:
        out["eval_strategy"] = out.pop("evaluation_strategy", "epoch")
    elif "evaluation_strategy" in params:
        out["evaluation_strategy"] = out.pop("evaluation_strategy", "epoch")
    else:
        out.pop("evaluation_strategy", None)

    return {k: v for k, v in out.items() if k in params}


def _freeze_backbone(model: Any) -> None:
    train_keywords = ["classifier", "score", "head", "fc", "pooler"]
    for name, param in model.named_parameters():
        param.requires_grad = any(key in name.lower() for key in train_keywords)


def _scalar_metric_subset(metrics: dict[str, Any]) -> dict[str, float]:
    subset: dict[str, float] = {}
    for key in ["accuracy", "top_5_accuracy", "precision", "recall", "f1", "macro_f1", "weighted_f1", "roc_auc", "pr_auc"]:
        value = metrics.get(key)
        if isinstance(value, (int, float)) and np.isfinite(value):
            subset[key] = float(value)
    return subset


def _scan_image_folder(bundle: ImageDatasetBundle) -> Dataset:
    """Build an image/label dataset by walking the class folders ourselves.

    We deliberately avoid ``load_dataset("imagefolder")`` here: its builder only
    emits a ``label`` column when every image sits at an identical directory depth
    and no ``metadata.*`` file exists in the tree. A single nested sub-folder or a
    stray metadata file makes it silently drop labels, which later surfaces as an
    opaque ``KeyError: 'label'``. ``bundle.labels`` was already resolved reliably
    in ``prepare_image_dataset``, so reuse it.
    """
    labels = list(bundle.labels)
    label2id = {label: i for i, label in enumerate(labels)}

    paths: list[str] = []
    label_ids: list[int] = []
    for label in labels:
        class_dir = bundle.data_dir / label
        for file in sorted(class_dir.rglob("*")):
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(str(file))
                label_ids.append(label2id[label])

    if not paths:
        raise ValueError(f"No images found under class folders in {bundle.data_dir}")

    features = Features({"image": Image(), "label": ClassLabel(names=labels)})
    return Dataset.from_dict({"image": paths, "label": label_ids}, features=features)


def _load_image_dataset(bundle: ImageDatasetBundle, max_train_samples: int | None = None):
    base = _scan_image_folder(bundle)

    if max_train_samples and len(base) > max_train_samples:
        base = base.shuffle(seed=42).select(range(max_train_samples))

    if len(base) < 4:
        return {"train": base, "test": base}

    try:
        return base.train_test_split(test_size=0.15, seed=42, stratify_by_column="label")
    except Exception:
        return base.train_test_split(test_size=0.15, seed=42)


def train_image_classifier(
    dataset: ImageDatasetBundle,
    plan: ImageModelPlan,
    run_dir: str | Path,
    goal: str,
    epochs: int,
    batch_size: int,
    max_train_samples: int | None = None,
    log_fn: LogFn | None = None,
) -> ImageTrainingResult:
    start = time.time()
    run_dir = ensure_dir(run_dir)
    model_dir = ensure_dir(run_dir / "image_model")

    _log(log_fn, f"Image classification training started: base={plan.base_model_id}")
    _log(log_fn, f"Images={dataset.num_images}, labels={dataset.labels}")
    _log(log_fn, f"Freeze backbone={plan.freeze_backbone}")

    split = _load_image_dataset(dataset, max_train_samples=max_train_samples)

    labels = split["train"].features["label"].names
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}

    processor = AutoImageProcessor.from_pretrained(plan.base_model_id)

    model = AutoModelForImageClassification.from_pretrained(
        plan.base_model_id,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    if plan.freeze_backbone:
        _freeze_backbone(model)

    def collate_fn(examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        images = [ex["image"].convert("RGB") for ex in examples]
        encoded = processor(images=images, return_tensors="pt")
        encoded["labels"] = torch.tensor([int(ex["label"]) for ex in examples], dtype=torch.long)
        return encoded

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits, labels_np = eval_pred
        full_metrics = compute_image_classification_metrics(labels_np, logits, label_names=labels)
        return _scalar_metric_subset(full_metrics)

    use_cuda = torch.cuda.is_available()

    training_args = TrainingArguments(
        **_training_args_kwargs(
            {
                "output_dir": str(run_dir / "trainer"),
                "overwrite_output_dir": True,
                "num_train_epochs": int(max(1, epochs)),
                "per_device_train_batch_size": int(max(1, batch_size)),
                "per_device_eval_batch_size": int(max(1, batch_size)),
                "learning_rate": float(plan.learning_rate),
                "logging_steps": 5,
                "save_strategy": "epoch",
                "save_total_limit": 1,
                "evaluation_strategy": "epoch",
                "report_to": [],
                "fp16": bool(use_cuda),
                "remove_unused_columns": False,
            }
        )
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=processor,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    train_output = trainer.train()
    eval_output = trainer.predict(split["test"], metric_key_prefix="eval")

    _log(log_fn, "Saving image classifier")
    model.save_pretrained(model_dir, safe_serialization=True)
    processor.save_pretrained(model_dir)

    runtime_s = time.time() - start
    eval_metrics = dict(eval_output.metrics)
    full_eval_metrics = compute_image_classification_metrics(
        eval_output.label_ids,
        eval_output.predictions,
        label_names=labels,
    )

    roc_curve_plot = plot_roc_curves(full_eval_metrics, run_dir / "roc_curve.png")

    metrics: dict[str, Any] = {
        "train_runtime_s": runtime_s,
        "train_loss": float(getattr(train_output, "training_loss", np.nan)),
        "eval_loss": float(eval_metrics.get("eval_loss", np.nan)),
        "train_rows": int(len(split["train"])),
        "eval_rows": int(len(split["test"])),
        **full_eval_metrics,
    }
    for key, value in eval_metrics.items():
        if key.startswith("eval_") and key not in metrics and isinstance(value, (int, float)):
            metrics[key] = float(value)
    if roc_curve_plot:
        metrics["roc_curve_plot"] = roc_curve_plot.name

    manifest = {
        "schema_version": 1,
        "task": "image_classification",
        "goal": goal,
        "base_model_id": plan.base_model_id,
        "model_type": "image_classifier",
        "model_dir": "image_model",
        "labels": labels,
        "plan": plan.to_dict(),
        "dataset_report": dataset.report,
        "metrics": metrics,
        "metrics_file": "metrics.json",
    }

    report = {
        "manifest": manifest,
        "metrics": metrics,
        "plan": plan.to_dict(),
        "dataset": dataset.report,
    }

    manifest_path = write_json(manifest, run_dir / "manifest.json")
    save_metrics(metrics, run_dir / "metrics.json")
    report_path = write_json(report, run_dir / "training_report.json")

    _log(
        log_fn,
        "Image training completed. "
        f"Accuracy={metrics.get('accuracy')} Macro-F1={metrics.get('macro_f1')}",
    )

    return ImageTrainingResult(
        run_dir=run_dir,
        model_dir=model_dir,
        manifest_path=manifest_path,
        report_path=report_path,
        metrics=metrics,
    )
