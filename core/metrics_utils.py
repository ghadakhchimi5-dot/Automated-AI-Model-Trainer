"""Shared metric helpers for post-training evaluation."""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    multilabel_confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    top_k_accuracy_score,
)
from sklearn.preprocessing import label_binarize

from .packaging import write_json


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _as_numpy(array_like: Any) -> np.ndarray | None:
    if array_like is None:
        return None
    if isinstance(array_like, tuple):
        if not array_like:
            return None
        array_like = array_like[0]
    array = np.asarray(array_like)
    if array.size == 0:
        return None
    return array


def _looks_like_probabilities(scores: np.ndarray) -> bool:
    if scores.ndim == 1:
        return bool(np.all((scores >= 0.0) & (scores <= 1.0)))
    if scores.ndim != 2:
        return False
    if not np.all((scores >= 0.0) & (scores <= 1.0)):
        return False
    row_sums = scores.sum(axis=1)
    return bool(np.allclose(row_sums, 1.0, atol=1e-3))


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.clip(exp_scores.sum(axis=-1, keepdims=True), a_min=1e-12, a_max=None)


def _sigmoid(scores: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-scores))


def _per_class_rows(
    precision: np.ndarray,
    recall: np.ndarray,
    f1: np.ndarray,
    support: np.ndarray,
    label_names: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, label_name in enumerate(label_names):
        rows.append(
            {
                "label": label_name,
                "precision": _safe_float(precision[idx]),
                "recall": _safe_float(recall[idx]),
                "f1": _safe_float(f1[idx]),
                "support": int(support[idx]),
            }
        )
    return rows


def _binary_auc_metrics(y_true: np.ndarray, positive_scores: np.ndarray, positive_label: str = "positive") -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if len(np.unique(y_true)) < 2:
        return metrics
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, positive_scores))
        fpr, tpr, _ = roc_curve(y_true, positive_scores)
        metrics["roc_curves"] = {positive_label: {"fpr": fpr.tolist(), "tpr": tpr.tolist()}}
    except Exception:
        pass
    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, positive_scores))
    except Exception:
        pass
    return metrics


def _multilabel_auc_metrics(y_true: np.ndarray, probabilities: np.ndarray, label_names: list[str]) -> dict[str, Any]:
    per_class_auc: list[dict[str, Any]] = []
    per_class_pr_auc: list[dict[str, Any]] = []

    for idx, label_name in enumerate(label_names):
        column_true = y_true[:, idx]
        column_prob = probabilities[:, idx]
        if len(np.unique(column_true)) < 2:
            continue
        try:
            per_class_auc.append({"label": label_name, "roc_auc": float(roc_auc_score(column_true, column_prob))})
        except Exception:
            pass
        try:
            per_class_pr_auc.append({"label": label_name, "pr_auc": float(average_precision_score(column_true, column_prob))})
        except Exception:
            pass

    metrics: dict[str, Any] = {}
    if per_class_auc:
        metrics["roc_auc"] = float(np.mean([row["roc_auc"] for row in per_class_auc]))
        metrics["per_class_roc_auc"] = per_class_auc
    if per_class_pr_auc:
        metrics["pr_auc"] = float(np.mean([row["pr_auc"] for row in per_class_pr_auc]))
        metrics["per_class_pr_auc"] = per_class_pr_auc
    return metrics


def _multiclass_auc_metrics(targets: np.ndarray, probabilities: np.ndarray, label_names: list[str]) -> dict[str, Any]:
    """One-vs-rest ROC AUC per class plus macro/micro averages and raw curve points."""
    num_classes = probabilities.shape[1]
    present_classes = set(np.unique(targets).tolist())
    per_class_auc: list[dict[str, Any]] = []
    curves: dict[str, Any] = {}

    for idx, label_name in enumerate(label_names):
        if idx not in present_classes:
            continue
        column_true = (targets == idx).astype(int)
        column_prob = probabilities[:, idx]
        if len(np.unique(column_true)) < 2:
            continue
        try:
            auc = float(roc_auc_score(column_true, column_prob))
        except Exception:
            continue
        fpr, tpr, _ = roc_curve(column_true, column_prob)
        per_class_auc.append({"label": label_name, "roc_auc": auc})
        curves[label_name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}

    metrics: dict[str, Any] = {}
    if not per_class_auc:
        return metrics

    metrics["per_class_roc_auc"] = per_class_auc
    metrics["roc_auc"] = float(np.mean([row["roc_auc"] for row in per_class_auc]))

    binarized = label_binarize(targets, classes=list(range(num_classes)))
    try:
        metrics["roc_auc_micro"] = float(roc_auc_score(binarized, probabilities, average="micro"))
        fpr_micro, tpr_micro, _ = roc_curve(binarized.ravel(), probabilities.ravel())
        curves["micro"] = {"fpr": fpr_micro.tolist(), "tpr": tpr_micro.tolist()}
    except Exception:
        pass

    metrics["roc_curves"] = curves
    return metrics


def compute_image_classification_metrics(
    y_true: Any,
    predictions: Any,
    label_names: list[str] | None = None,
    prediction_type: str = "auto",
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute robust image classification metrics for binary/multiclass/multilabel data."""
    targets = _as_numpy(y_true)
    scores = _as_numpy(predictions)

    if targets is None:
        return {"num_samples": 0, "skipped": "empty_targets"}
    if scores is None:
        return {"num_samples": int(len(targets)), "skipped": "empty_predictions"}

    metrics: dict[str, Any] = {"num_samples": int(len(targets))}
    multilabel = targets.ndim == 2 and targets.shape[1] > 1

    if multilabel:
        num_labels = int(targets.shape[1])
        names = label_names or [f"class_{idx}" for idx in range(num_labels)]
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)
        if scores.shape != targets.shape:
            return {
                "num_samples": int(len(targets)),
                "task_type": "multilabel",
                "skipped": "shape_mismatch",
            }

        if prediction_type == "labels":
            probabilities = None
            predicted = scores.astype(int)
        else:
            probabilities = scores if prediction_type == "probabilities" or _looks_like_probabilities(scores) else _sigmoid(scores)
            predicted = (probabilities >= threshold).astype(int)

        precision, recall, f1, support = precision_recall_fscore_support(
            targets,
            predicted,
            average=None,
            zero_division=0,
        )

        metrics.update(
            {
                "task_type": "multilabel",
                "accuracy": float(accuracy_score(targets, predicted)),
                "precision": float(precision_recall_fscore_support(targets, predicted, average="micro", zero_division=0)[0]),
                "recall": float(precision_recall_fscore_support(targets, predicted, average="micro", zero_division=0)[1]),
                "f1": float(precision_recall_fscore_support(targets, predicted, average="micro", zero_division=0)[2]),
                "macro_f1": float(f1_score(targets, predicted, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(targets, predicted, average="weighted", zero_division=0)),
                "per_class": _per_class_rows(precision, recall, f1, support, names),
                "confusion_matrix": multilabel_confusion_matrix(targets, predicted).tolist(),
                "confusion_matrix_labels": names,
            }
        )

        if probabilities is not None:
            metrics.update(_multilabel_auc_metrics(targets, probabilities, names))
        return metrics

    targets = targets.reshape(-1)

    if scores.ndim == 2:
        num_classes = int(scores.shape[1])
        names = label_names or [f"class_{idx}" for idx in range(num_classes)]
        probabilities = scores if prediction_type == "probabilities" or _looks_like_probabilities(scores) else _softmax(scores)
        predicted = np.argmax(probabilities, axis=-1)
    else:
        inferred_classes = int(len(label_names)) if label_names else int(max(np.max(targets), np.max(scores))) + 1
        num_classes = max(2, inferred_classes)
        names = label_names or [f"class_{idx}" for idx in range(num_classes)]
        if prediction_type == "labels":
            probabilities = None
            predicted = scores.astype(int)
        else:
            probabilities = scores if prediction_type == "probabilities" or _looks_like_probabilities(scores) else _sigmoid(scores)
            predicted = (probabilities >= threshold).astype(int)

    labels_idx = list(range(num_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predicted,
        labels=labels_idx,
        average=None,
        zero_division=0,
    )

    average_mode = "binary" if num_classes == 2 else "macro"
    averaged = precision_recall_fscore_support(
        targets,
        predicted,
        average=average_mode,
        zero_division=0,
    )

    metrics.update(
        {
            "task_type": "binary" if num_classes == 2 else "multiclass",
            "accuracy": float(accuracy_score(targets, predicted)),
            "precision": float(averaged[0]),
            "recall": float(averaged[1]),
            "f1": float(averaged[2]),
            "macro_f1": float(f1_score(targets, predicted, labels=labels_idx, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(targets, predicted, labels=labels_idx, average="weighted", zero_division=0)),
            "per_class": _per_class_rows(precision, recall, f1, support, names),
            "confusion_matrix": confusion_matrix(targets, predicted, labels=labels_idx).tolist(),
            "confusion_matrix_labels": names,
        }
    )

    if probabilities is not None and num_classes >= 5 and probabilities.ndim == 2:
        try:
            metrics["top_5_accuracy"] = float(
                top_k_accuracy_score(targets, probabilities, k=min(5, num_classes), labels=labels_idx)
            )
        except Exception:
            pass

    if probabilities is not None and num_classes == 2:
        positive_scores = probabilities[:, 1] if probabilities.ndim == 2 else probabilities.reshape(-1)
        positive_label = names[1] if len(names) > 1 else "positive"
        metrics.update(_binary_auc_metrics(targets, positive_scores, positive_label))

    if probabilities is not None and num_classes > 2 and probabilities.ndim == 2:
        metrics.update(_multiclass_auc_metrics(targets, probabilities, names))

    return metrics


def _normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def _token_f1(prediction: str, reference: str) -> float:
    pred_tokens = _normalize_text(prediction).split()
    ref_tokens = _normalize_text(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    overlap = Counter(pred_tokens) & Counter(ref_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    return (2 * precision * recall) / max(precision + recall, 1e-12)


def compute_slm_metrics(
    *,
    eval_loss: Any = None,
    logits: Any = None,
    labels: Any = None,
    generated_texts: list[str] | None = None,
    reference_texts: list[str] | None = None,
    log_fn: Any = None,
) -> dict[str, Any]:
    """Compute SLM metrics for causal-LM/instruction-tuning style evaluation."""
    metrics: dict[str, Any] = {}

    loss_value = _safe_float(eval_loss)
    metrics["eval_loss"] = loss_value
    if loss_value is not None:
        try:
            metrics["perplexity"] = float(math.exp(loss_value))
        except OverflowError:
            metrics["perplexity"] = None
    else:
        metrics["perplexity"] = None

    label_ids = _as_numpy(labels)
    prediction_scores = _as_numpy(logits)
    if label_ids is not None and prediction_scores is not None:
        if prediction_scores.ndim == label_ids.ndim + 1:
            # Causal LM: logits at position i are the model's prediction for the
            # token at position i+1. The Trainer does NOT pre-shift labels, so we
            # must shift here or every comparison is off by one token and accuracy
            # collapses toward 0 regardless of how good the model actually is.
            shift_logits = prediction_scores[..., :-1, :]
            shift_labels = label_ids[..., 1:]
            prediction_ids = np.argmax(shift_logits, axis=-1)
        else:
            # Already decoded/argmaxed token ids (e.g. from generate()); same length
            # as labels, so no shift is needed.
            shift_labels = label_ids
            prediction_ids = prediction_scores

        mask = shift_labels != -100
        valid_tokens = int(mask.sum())
        metrics["token_count"] = valid_tokens
        metrics["token_accuracy"] = (
            float((prediction_ids[mask] == shift_labels[mask]).mean()) if valid_tokens else None
        )

        debug_msg = (
            f"[compute_slm_metrics] logits.shape={prediction_scores.shape} "
            f"labels.shape={label_ids.shape} prediction_ids.shape={prediction_ids.shape} "
            f"mask.sum()={valid_tokens}"
        )
        if log_fn:
            log_fn(debug_msg)
        else:
            print(debug_msg, flush=True)

    if generated_texts is None or reference_texts is None:
        metrics["generation_metrics_skipped"] = "missing_references_or_generations"
        return metrics

    paired = list(zip(generated_texts, reference_texts))
    if not paired:
        metrics["generation_metrics_skipped"] = "empty_generation_sample"
        return metrics

    exact_match = [1.0 if _normalize_text(pred) == _normalize_text(ref) else 0.0 for pred, ref in paired]
    token_f1_values = [_token_f1(pred, ref) for pred, ref in paired]

    metrics.update(
        {
            "generation_sample_size": len(paired),
            "exact_match": float(np.mean(exact_match)),
            "token_f1": float(np.mean(token_f1_values)),
        }
    )
    return metrics


def infer_task_subtype(goal: str) -> str:
    """Detect a fine-grained text task from the goal string to select the right extra metrics."""
    goal_lower = goal.lower()
    if any(w in goal_lower for w in ["translat", "tradui", "traduire"]):
        return "translation"
    if any(w in goal_lower for w in ["summar", "résumé", "abstract", "condensed"]):
        return "summarization"
    if any(w in goal_lower for w in ["classif", "sentiment", "categor", "label", "intent"]):
        return "classification"
    return "generative_qa"


def compute_rouge_bleu(
    generated_texts: list[str],
    reference_texts: list[str],
    task_subtype: str = "generative_qa",
) -> dict[str, Any]:
    """Compute ROUGE and/or BLEU metrics using the evaluate library when relevant."""
    if not generated_texts or not reference_texts:
        return {}
    paired = list(zip(generated_texts, reference_texts))
    if not paired:
        return {}

    results: dict[str, Any] = {}

    if task_subtype in {"summarization", "generative_qa"}:
        try:
            import evaluate as hf_evaluate
            rouge = hf_evaluate.load("rouge")
            rouge_result = rouge.compute(
                predictions=[p for p, _ in paired],
                references=[r for _, r in paired],
            )
            results.update({
                "rouge1": round(float(rouge_result.get("rouge1", 0.0)), 4),
                "rouge2": round(float(rouge_result.get("rouge2", 0.0)), 4),
                "rougeL": round(float(rouge_result.get("rougeL", 0.0)), 4),
            })
        except Exception:
            pass

    if task_subtype == "translation":
        try:
            import evaluate as hf_evaluate
            bleu = hf_evaluate.load("sacrebleu")
            bleu_result = bleu.compute(
                predictions=[p for p, _ in paired],
                references=[[r] for _, r in paired],
            )
            results["bleu"] = round(float(bleu_result.get("score", 0.0)), 2)
        except Exception:
            pass

    return results


def save_metrics(metrics: dict[str, Any], path: str | Path) -> Path:
    return write_json(metrics, path)
