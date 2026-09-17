"""Human-readable Markdown training report generator."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any


def _fmt(value: Any, decimals: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def _verdict_emoji(verdict: str) -> str:
    return {"pass": "PASS", "warn": "WARN", "suggest_retrain": "RETRAIN SUGGESTED"}.get(verdict, verdict.upper())


def generate_markdown_report(
    task: str,
    goal: str,
    hardware: dict[str, Any],
    plan: dict[str, Any],
    metrics: dict[str, Any],
    dataset_report: dict[str, Any],
    agent_diagnostics: dict[str, Any],
    run_dir: Path,
) -> Path:
    """Write a human-readable TRAINING_REPORT.md into run_dir and return its path."""
    lines: list[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines += [
        f"# GHD Auto Trainer — Training Report",
        f"",
        f"**Generated:** {now}  ",
        f"**Task:** {task}  ",
        f"**Goal:** {goal or '(not specified)'}  ",
        f"",
    ]

    # Hardware
    lines += [
        "## Hardware",
        f"| Property | Value |",
        f"|---|---|",
        f"| Tier | {hardware.get('tier', 'N/A')} |",
        f"| GPU | {hardware.get('gpu_name', 'None')} |",
        f"| VRAM | {_fmt(hardware.get('vram_gb', 0), 1)} GB |",
        f"| RAM | {_fmt(hardware.get('ram_gb', 0), 1)} GB |",
        f"",
    ]

    # Dataset
    lines += ["## Dataset"]
    if task == "text_qa":
        lines += [
            f"| Property | Value |",
            f"|---|---|",
            f"| Raw rows | {dataset_report.get('raw_rows', 'N/A')} |",
            f"| Training rows | {dataset_report.get('train_rows', 'N/A')} |",
            f"| Eval rows | {dataset_report.get('eval_rows', 'N/A')} |",
            f"| Quality score | {_fmt(dataset_report.get('quality_score'), 1)}/100 |",
            f"| Avg prompt length | {_fmt(dataset_report.get('avg_prompt_chars'), 0)} chars |",
            f"| Avg response length | {_fmt(dataset_report.get('avg_response_chars'), 0)} chars |",
            f"| Duplicates removed | {dataset_report.get('duplicates', 0)} |",
            f"",
        ]
    else:
        lines += [
            f"| Property | Value |",
            f"|---|---|",
            f"| Total images | {dataset_report.get('num_images', 'N/A')} |",
            f"| Number of classes | {dataset_report.get('num_classes', 'N/A')} |",
            f"| Classes | {', '.join(dataset_report.get('labels', []))} |",
            f"",
        ]

    # Model & PEFT
    lines += ["## Model & Training Strategy"]
    if task == "text_qa":
        lines += [
            f"| Property | Value |",
            f"|---|---|",
            f"| Base model | `{plan.get('base_model_id', 'N/A')}` |",
            f"| PEFT method | **{plan.get('peft_method', 'N/A').upper()}** |",
            f"| LoRA rank | {plan.get('lora_rank', 'N/A')} |",
            f"| LoRA alpha | {plan.get('lora_alpha', 'N/A')} |",
            f"| Quantization | {plan.get('quantization', 'none')} |",
            f"| Learning rate | {plan.get('learning_rate', 'N/A')} |",
            f"",
        ]
    else:
        lines += [
            f"| Property | Value |",
            f"|---|---|",
            f"| Base model | `{plan.get('base_model_id', 'N/A')}` |",
            f"| Backbone frozen | {plan.get('freeze_backbone', 'N/A')} |",
            f"| Learning rate | {plan.get('learning_rate', 'N/A')} |",
            f"",
        ]

    # Metrics
    lines += ["## Evaluation Metrics"]
    if task == "text_qa":
        lines += [
            f"| Metric | Value | Interpretation |",
            f"|---|---|---|",
            f"| Eval loss | {_fmt(metrics.get('eval_loss'))} | Lower is better |",
            f"| Perplexity | {_fmt(metrics.get('perplexity'), 2)} | <20 = good, >50 = retrain |",
            f"| Token accuracy | {_fmt(metrics.get('token_accuracy'), 4)} | >=0.5 = good, <0.5 = warn |",
            f"| Exact match | {_fmt(metrics.get('exact_match'), 4)} | % responses matched exactly |",
        ]
        if "rouge1" in metrics:
            lines += [
                f"| ROUGE-1 | {_fmt(metrics.get('rouge1'))} | Recall of unigrams |",
                f"| ROUGE-2 | {_fmt(metrics.get('rouge2'))} | Recall of bigrams |",
                f"| ROUGE-L | {_fmt(metrics.get('rougeL'))} | Longest common subsequence |",
            ]
        if "bleu" in metrics:
            lines += [f"| BLEU | {_fmt(metrics.get('bleu'), 2)} | Translation quality (0–100) |"]
        lines += [""]
    else:
        lines += [
            f"| Metric | Value | Interpretation |",
            f"|---|---|---|",
            f"| Accuracy | {_fmt(metrics.get('accuracy'), 4)} | >0.70 = good, <0.55 = retrain |",
            f"| Precision | {_fmt(metrics.get('precision'), 4)} | True positives / predicted |",
            f"| Recall | {_fmt(metrics.get('recall'), 4)} | True positives / actual |",
            f"| F1 | {_fmt(metrics.get('f1'), 4)} | Harmonic mean of P and R |",
            f"| Macro F1 | {_fmt(metrics.get('macro_f1'), 4)} | Unweighted avg across classes |",
            f"| Weighted F1 | {_fmt(metrics.get('weighted_f1'), 4)} | Class-size-weighted avg |",
        ]
        if "roc_auc" in metrics:
            lines += [f"| ROC AUC | {_fmt(metrics.get('roc_auc'), 4)} | One-vs-rest, macro-averaged; 0.5 = random, 1.0 = perfect |"]
        if "roc_auc_micro" in metrics:
            lines += [f"| ROC AUC (micro) | {_fmt(metrics.get('roc_auc_micro'), 4)} | Pooled across all classes |"]
        lines += [""]
        if metrics.get("roc_curve_plot"):
            lines += [f"![ROC Curve]({metrics['roc_curve_plot']})", ""]

    lines += [f"**Training time:** {_fmt(metrics.get('train_runtime_s', 0), 0)} seconds", ""]

    # Evaluation verdict
    diag_eval = agent_diagnostics.get("evaluation", {})
    verdict = diag_eval.get("verdict", "unknown")
    score = diag_eval.get("score", 0)
    lines += [
        "## Evaluation Verdict",
        f"**Result: {_verdict_emoji(verdict)}** (quality score: {_fmt(score, 0)}/100)",
        "",
    ]

    issues = diag_eval.get("issues", [])
    if issues:
        lines += ["### Issues Found", ""]
        for issue in issues:
            lines += [f"- {issue}"]
        lines += [""]

    recs = diag_eval.get("recommendations", [])
    if recs:
        lines += ["### Recommendations", ""]
        for rec in recs:
            lines += [f"- {rec}"]
        lines += [""]

    # Data agent warnings
    diag_data = agent_diagnostics.get("data", {})
    data_warnings = diag_data.get("warnings", [])
    if data_warnings:
        lines += ["### Data Warnings", ""]
        for w in data_warnings:
            lines += [f"- {w}"]
        lines += [""]

    # Training health
    diag_train = agent_diagnostics.get("training", {})
    lines += [
        "## Training Health",
        f"| Check | Result |",
        f"|---|---|",
        f"| OOM detected | {'YES — gradient checkpointing may help' if diag_train.get('oom_detected') else 'No'} |",
        f"| NaN loss | {'YES — check learning rate and data' if diag_train.get('nan_loss_detected') else 'No'} |",
        f"| Loss trend | {diag_train.get('loss_trend', 'unknown')} |",
        f"",
    ]

    # Next steps
    lines += ["## What To Do Next", ""]
    if verdict == "pass":
        lines += [
            "Your model passed evaluation. You can proceed to serving.",
            "",
            "- Download the model ZIP from the run directory.",
            "- Load the adapter with `core.inference` for text generation.",
            "",
        ]
    elif verdict == "warn":
        lines += [
            "Your model has acceptable quality but could improve. Consider:",
            "",
            "- Adding more diverse training examples.",
            "- Running more epochs (`--epochs N`).",
            "- Trying a slightly larger base model.",
            "",
        ]
    else:
        lines += [
            "Retraining is recommended. Prioritize these actions:",
            "",
        ]
        for rec in recs:
            lines += [f"1. {rec}"]
        lines += [
            "",
            "Then re-run the pipeline with adjustments, for example:",
            "```bash",
            "python auto_pipeline.py --data ./your_data.csv --goal \"your goal\" --epochs 5",
            "```",
            "",
        ]

    report_path = run_dir / "TRAINING_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
