"""Evaluation Agent — interprets metrics and decides if retraining is needed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

LogFn = Callable[[str], None]

# Text QA thresholds
_TEXT_PERPLEXITY_WARN = 20.0
_TEXT_PERPLEXITY_RETRAIN = 50.0
_TEXT_TOKEN_ACCURACY_WARN = 0.50


# Image classification thresholds
_IMG_ACCURACY_WARN = 0.70
_IMG_ACCURACY_RETRAIN = 0.55


@dataclass
class EvaluationVerdict:
    verdict: str            # "pass" | "warn" | "suggest_retrain"
    score: float            # 0–100 quality score
    key_metrics: dict[str, Any]
    issues: list[str]
    recommendations: list[str]
    retrain_suggested: bool
    reasoning: str


class EvaluationAgent:
    """Interprets training metrics and recommends next actions.

    Wraps compute_slm_metrics / compute_image_classification_metrics results
    and applies threshold logic to produce a human-readable verdict.
    """

    def evaluate(
        self,
        metrics: dict[str, Any],
        task: str,
        dataset_report: dict[str, Any],
        plan_dict: dict[str, Any] | None = None,
        log_fn: LogFn | None = None,
    ) -> EvaluationVerdict:
        issues: list[str] = []
        recommendations: list[str] = []
        reasoning_parts: list[str] = []
        verdict = "pass"
        score = 100.0

        if task == "text_qa":
            verdict, score = self._eval_text(
                metrics, dataset_report, plan_dict or {}, issues, recommendations, reasoning_parts
            )
        else:
            verdict, score = self._eval_image(
                metrics, dataset_report, plan_dict or {}, issues, recommendations, reasoning_parts
            )

        score = max(0.0, min(100.0, score))
        retrain = verdict == "suggest_retrain"
        reasoning = " ".join(reasoning_parts)
        key_metrics = _extract_key_metrics(metrics, task)

        _log(log_fn, f"[EvaluationAgent] Verdict: {verdict.upper()} (score={score:.0f}/100). {reasoning}")
        for issue in issues:
            _log(log_fn, f"[EvaluationAgent] Issue: {issue}")
        for rec in recommendations:
            _log(log_fn, f"[EvaluationAgent] Recommendation: {rec}")
        if retrain:
            _log(log_fn, "[EvaluationAgent] Retraining is recommended before deploying this model.")

        return EvaluationVerdict(
            verdict=verdict,
            score=score,
            key_metrics=key_metrics,
            issues=issues,
            recommendations=recommendations,
            retrain_suggested=retrain,
            reasoning=reasoning,
        )

    def _eval_text(
        self,
        metrics: dict[str, Any],
        dataset_report: dict[str, Any],
        plan_dict: dict[str, Any],
        issues: list[str],
        recommendations: list[str],
        reasoning_parts: list[str],
    ) -> tuple[str, float]:
        score = 100.0
        verdict = "pass"

        perplexity = metrics.get("perplexity")
        token_accuracy = metrics.get("token_accuracy")
        generated_qa = bool(metrics.get("generated_qa", False))
        train_rows = int(dataset_report.get("train_rows", dataset_report.get("rows", 0)))

        reasoning_parts.append(f"Text QA: {train_rows} training rows.")

        if perplexity is not None:
            reasoning_parts.append(f"Perplexity={perplexity:.2f}.")
            if perplexity > _TEXT_PERPLEXITY_RETRAIN:
                score -= 40
                verdict = "suggest_retrain"
                issues.append(
                    f"High perplexity ({perplexity:.1f} > {_TEXT_PERPLEXITY_RETRAIN}): "
                    "model has not converged on the training data."
                )
                recommendations.append("Increase training epochs or use a higher LoRA rank.")
                recommendations.append("Check that your dataset has meaningful instruction/output pairs.")
            elif perplexity > _TEXT_PERPLEXITY_WARN:
                score -= 20
                verdict = _escalate(verdict, "warn")
                issues.append(
                    f"Moderate perplexity ({perplexity:.1f}): model is learning but has room to improve."
                )
                recommendations.append("Consider adding more diverse training examples or more epochs.")
            else:
                reasoning_parts.append("Perplexity is within acceptable range.")

        if token_accuracy is not None:
            reasoning_parts.append(f"Token accuracy={token_accuracy:.3f}.")

            if token_accuracy < _TEXT_TOKEN_ACCURACY_WARN:
                score -= 15
                verdict = _escalate(verdict, "warn")

                issues.append(
                    f"Low token accuracy ({token_accuracy:.1%} < {_TEXT_TOKEN_ACCURACY_WARN:.0%}): "
                    "generated answers predict less than half of the reference tokens correctly."
                )

                recommendations.append(
                    "Review training examples and ensure questions and answers are clearly aligned."
                )
            else:
                reasoning_parts.append(
                    f"Token accuracy is at or above the {_TEXT_TOKEN_ACCURACY_WARN:.0%} threshold — accepted."
                )

        if generated_qa:
            score -= 10
            reasoning_parts.append("Dataset used synthetic QA generation (no gold answers provided).")
            recommendations.append(
                "Replace synthetic QA with real human-written instruction/output pairs for better quality."
            )

        if train_rows < 100:
            score -= 15
            issues.append(f"Only {train_rows} training examples — very limited signal for fine-tuning.")

        return verdict, score

    def _eval_image(
        self,
        metrics: dict[str, Any],
        dataset_report: dict[str, Any],
        plan_dict: dict[str, Any],
        issues: list[str],
        recommendations: list[str],
        reasoning_parts: list[str],
    ) -> tuple[str, float]:
        score = 100.0
        verdict = "pass"

        accuracy = metrics.get("accuracy")
        f1 = metrics.get("f1") or metrics.get("macro_f1")
        num_images = int(dataset_report.get("num_images", 0))
        num_classes = int(dataset_report.get("num_classes", 2))

        reasoning_parts.append(
            f"Image classification: {num_images} images, {num_classes} classes."
        )

        if accuracy is not None:
            reasoning_parts.append(f"Accuracy={accuracy:.3f}.")
            if accuracy < _IMG_ACCURACY_RETRAIN:
                score -= 45
                verdict = "suggest_retrain"
                issues.append(
                    f"Low accuracy ({accuracy:.1%}) — model is barely above random chance "
                    f"for {num_classes} classes."
                )
                recommendations.append("Collect more labeled images (100+ per class).")
                if num_classes > 5:
                    recommendations.append(
                        "Consider reducing the number of classes by merging similar categories."
                    )
            elif accuracy < _IMG_ACCURACY_WARN:
                score -= 20
                verdict = _escalate(verdict, "warn")
                issues.append(
                    f"Moderate accuracy ({accuracy:.1%}): acceptable baseline but improvable."
                )
                recommendations.append("Add more diverse training images or increase training epochs.")
            else:
                reasoning_parts.append("Accuracy is good.")

        if f1 is not None:
            reasoning_parts.append(f"F1={f1:.3f}.")
            if f1 < 0.50:
                score -= 15
                issues.append(
                    f"Low F1 ({f1:.2f}): class imbalance or confusion between classes. "
                    "Check the confusion matrix in training_report.json."
                )
                recommendations.append(
                    "Balance class sizes or apply class-weighted loss for imbalanced datasets."
                )

        if num_images < 200:
            score -= 20
            issues.append(
                f"Small image dataset ({num_images} total). "
                "Results are unlikely to generalise well."
            )
            recommendations.append("Aim for at least 100 images per class.")

        return verdict, score


# ── helpers ──────────────────────────────────────────────────────────────────

def _escalate(current: str, new: str) -> str:
    order = {"pass": 0, "warn": 1, "suggest_retrain": 2}
    return new if order.get(new, 0) > order.get(current, 0) else current


def _extract_key_metrics(metrics: dict[str, Any], task: str) -> dict[str, Any]:
    if task == "text_qa":
        keys = [
            "eval_loss",
            "perplexity",
            "token_accuracy",
        ]
    else:
        keys = ["accuracy", "f1", "macro_f1", "weighted_f1", "precision", "recall"]
    return {k: metrics[k] for k in keys if k in metrics}


def _log(log_fn: LogFn | None, msg: str) -> None:
    if log_fn:
        log_fn(msg)
    else:
        print(msg, flush=True)
