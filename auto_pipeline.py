"""One-command automatic training pipeline for GHD.

Use from CLI:
    python auto_pipeline.py --task auto --data ./dataset.csv --goal "Create a support chatbot"

Use from Streamlit app:
    from auto_pipeline import run_automatic_training
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

# Pin BLAS/OpenMP thread pools to physical cores before torch/numpy import them.
# Hyperthreads contend for the same execution ports and slow matmul-heavy CPU
# training; torch.set_num_threads (in text_trainer) refines this at runtime.
try:
    import psutil

    _phys = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1
except Exception:
    _phys = os.cpu_count() or 1
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, str(int(max(1, _phys))))

from core.agents import (
    DataAgent,
    EvaluationAgent,
    ModelAgent,
    OptimizationAgent,
    TrainingAgent,
)
from core.data_utils import ensure_dir
from core.hardware import detect_hardware
from core.packaging import make_model_zip, read_json
from core.report_generator import generate_markdown_report

LogFn = Callable[[str], None]


@dataclass(frozen=True)
class AutoRunResult:
    task: str
    run_dir: Path
    zip_path: Path
    manifest: dict[str, Any]
    report: dict[str, Any]
    hardware: dict[str, Any]
    plan: dict[str, Any]
    metrics: dict[str, Any]
    agent_log: tuple[str, ...] = ()
    evaluation_verdict: str = "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "run_dir": str(self.run_dir),
            "zip_path": str(self.zip_path),
            "manifest": self.manifest,
            "report": self.report,
            "hardware": self.hardware,
            "plan": self.plan,
            "metrics": self.metrics,
            "agent_log": list(self.agent_log),
            "evaluation_verdict": self.evaluation_verdict,
        }


def _run_name(task: str) -> str:
    return f"run_{task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def _improvement_strategy(
    verdict: Any,
    current_epochs: int | None,
    current_priority: str,
    attempt: int,
) -> dict[str, Any]:
    """Return adjusted parameters for a retry attempt based on evaluation issues.

    Returns a dict of kwargs to override in the next run_automatic_training call.
    Returns empty dict when no improvement is possible (stop retrying).
    """
    issues = [i.lower() for i in verdict.issues]
    overrides: dict[str, Any] = {}

    if attempt >= 2:
        return {}  # Hard cap at 2 retry attempts

    high_perplexity = any("perplexity" in i for i in issues)
    low_f1 = any("token f1" in i or "f1" in i for i in issues)
    low_accuracy = any("accuracy" in i for i in issues)
    oom = verdict.key_metrics.get("oom_detected", False)

    if oom:
        # On OOM: switch to memory-priority (smaller model, lower rank)
        overrides["priority"] = "memory"
        return overrides

    if high_perplexity or low_f1:
        # Increase epochs by 2 for each attempt; also push priority to performance
        base_epochs = current_epochs or 3
        overrides["epochs"] = base_epochs + 2 * attempt
        if current_priority != "performance":
            overrides["priority"] = "performance"
        return overrides

    if low_accuracy:
        # More epochs and performance priority for image tasks
        base_epochs = current_epochs or 5
        overrides["epochs"] = base_epochs + 3 * attempt
        overrides["priority"] = "performance"
        return overrides

    # Generic fallback: add epochs
    overrides["epochs"] = (current_epochs or 3) + 2
    return overrides


def run_automatic_training(
    data_path: str | Path,
    goal: str,
    task: str = "auto",
    output_dir: str | Path = "output",
    priority: str = "balanced",
    override_model_id: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    max_samples: int | None = None,
    max_length: int | None = None,
    merge_text_model: bool = False,
    auto_improve: bool = True,
    log_fn: LogFn | None = None,
) -> AutoRunResult:
    """Run the full automatic pipeline and return paths/metrics.

    This function is intentionally synchronous: when it returns, the model package is ready.
    The pipeline uses five specialised agents:
      DataAgent → ModelAgent → OptimizationAgent → TrainingAgent → EvaluationAgent
    """
    agent_messages: list[str] = []

    def _log(msg: str) -> None:
        agent_messages.append(msg)
        if log_fn:
            log_fn(msg)
        else:
            print(msg, flush=True)

    data_path = Path(data_path)
    hardware = detect_hardware()
    _log(
        f"Hardware: tier={hardware.tier}, gpu={hardware.gpu_name}, "
        f"vram={hardware.vram_gb:.1f}GB, ram={hardware.ram_gb:.1f}GB"
    )

    # ── Step 1: Data Agent ────────────────────────────────────────────────────
    data_agent = DataAgent()
    # We need run_dir for image tasks but the name depends on task — detect task first
    # via a lightweight probe, then create the real run_dir.
    from core.data_utils import detect_task as _detect_task
    detected_task = _detect_task(data_path, user_task=task, goal=goal)
    run_dir = ensure_dir(Path(output_dir) / _run_name(detected_task))

    data_analysis = data_agent.run(
        data_path=data_path,
        goal=goal,
        user_task=task,
        max_samples=max_samples,
        run_dir=run_dir,
        log_fn=_log,
    )
    selected_task = data_analysis.task
    if selected_task not in {"text_qa", "image_classification"}:
        raise ValueError(f"Unsupported task: {selected_task}")

    # ── Step 2: Model Agent ───────────────────────────────────────────────────
    dataset_size = (
        len(data_analysis.bundle.all_rows)
        if selected_task == "text_qa"
        else data_analysis.bundle.num_images
    )
    model_agent = ModelAgent()
    model_decision = model_agent.select(
        task=selected_task,
        hardware=hardware,
        dataset_size=dataset_size,
        priority=priority,
        override_model_id=override_model_id,
        log_fn=_log,
    )
    plan = model_decision.plan

    # ── Step 3: Optimization Agent ────────────────────────────────────────────
    optim_agent = OptimizationAgent()
    optim_plan = optim_agent.optimize(
        task=selected_task,
        hardware=hardware,
        dataset_size=dataset_size,
        plan=plan,
        quality_score=data_analysis.quality_score,
        user_epochs=epochs,
        user_batch_size=batch_size,
        user_max_samples=max_samples,
        user_max_length=max_length,
        log_fn=_log,
    )

    # Clamp bundle to the final max_train_samples decided by the optimizer
    bundle = data_analysis.bundle
    if selected_task == "text_qa":
        from core.data_utils import prepare_text_dataset
        if optim_plan.max_train_samples < len(bundle.all_rows):
            bundle = prepare_text_dataset(
                data_path, goal=goal, max_samples=optim_plan.max_train_samples, log_fn=_log
            )

    # ── Step 4: Training Agent ────────────────────────────────────────────────
    training_agent = TrainingAgent()
    outcome = training_agent.train(
        task=selected_task,
        bundle=bundle,
        plan=plan,
        optim=optim_plan,
        run_dir=run_dir,
        goal=goal,
        merge_model=merge_text_model,
        log_fn=_log,
    )
    train_result = outcome.result

    # ── Step 5: Evaluation Agent ──────────────────────────────────────────────
    eval_agent = EvaluationAgent()
    verdict = eval_agent.evaluate(
        metrics=train_result.metrics,
        task=selected_task,
        dataset_report=data_analysis.report,
        plan_dict=plan.to_dict(),
        log_fn=_log,
    )

    # ── Packaging ─────────────────────────────────────────────────────────────
    zip_path = make_model_zip(train_result.run_dir)
    manifest = read_json(train_result.manifest_path)
    report = read_json(train_result.report_path)

    # Enrich report with agent diagnostics
    report["agent_diagnostics"] = {
        "data": {
            "quality_score": data_analysis.quality_score,
            "quality_label": data_analysis.quality_label,
            "warnings": data_analysis.warnings,
            "recommendations": data_analysis.recommendations,
        },
        "model": {
            "reasoning": model_decision.reasoning,
            "alternatives_considered": model_decision.alternatives_considered,
            "warnings": model_decision.warnings,
        },
        "optimization": {
            "reasoning": optim_plan.reasoning,
            "warnings": optim_plan.warnings,
            "overrides_applied": optim_plan.overrides_applied,
        },
        "training": {
            "status": outcome.health.status,
            "oom_detected": outcome.health.oom_detected,
            "nan_loss_detected": outcome.health.nan_loss_detected,
            "loss_trend": outcome.health.loss_trend,
            "events": outcome.health.events,
        },
        "evaluation": {
            "verdict": verdict.verdict,
            "score": verdict.score,
            "issues": verdict.issues,
            "recommendations": verdict.recommendations,
            "retrain_suggested": verdict.retrain_suggested,
        },
    }

    # Generate human-readable Markdown report
    try:
        md_path = generate_markdown_report(
            task=selected_task,
            goal=goal,
            hardware=asdict(hardware),
            plan=plan.to_dict(),
            metrics=train_result.metrics,
            dataset_report=data_analysis.report,
            agent_diagnostics=report["agent_diagnostics"],
            run_dir=train_result.run_dir,
        )
        _log(f"Markdown report saved to: {md_path}")
    except Exception as exc:
        _log(f"[warning] Could not generate Markdown report: {exc}")

    # ── Auto-improvement loop ─────────────────────────────────────────────────
    # When verdict is suggest_retrain and auto_improve is enabled, retry up to 2x
    # with automatically adjusted hyperparameters.
    if auto_improve and verdict.retrain_suggested:
        for attempt in range(1, 3):
            overrides = _improvement_strategy(verdict, epochs, priority, attempt)
            if not overrides:
                _log("[AutoImprove] No further improvement strategy available. Stopping.")
                break
            _log(
                f"[AutoImprove] Attempt {attempt}/2 — retrying with adjustments: {overrides}"
            )
            try:
                improved = run_automatic_training(
                    data_path=data_path,
                    goal=goal,
                    task=task,
                    output_dir=output_dir,
                    priority=overrides.get("priority", priority),
                    override_model_id=override_model_id,
                    epochs=overrides.get("epochs", epochs),
                    batch_size=batch_size,
                    max_samples=max_samples,
                    max_length=max_length,
                    merge_text_model=merge_text_model,
                    auto_improve=False,  # no recursive improvement
                    log_fn=log_fn,
                )
                # Accept the improved run if its verdict is better
                improved_order = {"pass": 2, "warn": 1, "suggest_retrain": 0}
                if improved_order.get(improved.evaluation_verdict, 0) > improved_order.get(verdict.verdict, 0):
                    _log(
                        f"[AutoImprove] Improved! New verdict: {improved.evaluation_verdict.upper()}"
                    )
                    return improved
                else:
                    _log(
                        f"[AutoImprove] Attempt {attempt} did not improve verdict "
                        f"({improved.evaluation_verdict}). Continuing..."
                    )
                    verdict_str = improved.evaluation_verdict
                    if verdict_str == "pass":
                        break
            except Exception as exc:
                _log(f"[AutoImprove] Retry attempt {attempt} failed: {exc}")
                break

    return AutoRunResult(
        task=selected_task,
        run_dir=train_result.run_dir,
        zip_path=zip_path,
        manifest=manifest,
        report=report,
        hardware=asdict(hardware),
        plan=plan.to_dict(),
        metrics=train_result.metrics,
        agent_log=tuple(agent_messages),
        evaluation_verdict=verdict.verdict,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automatic fine-tuning pipeline for text QA and image classification"
    )
    parser.add_argument("--data", required=True, help="Path to CSV/JSONL/TXT/zip/folder dataset")
    parser.add_argument("--goal", default="", help="User goal, e.g. 'create a customer support chatbot'")
    parser.add_argument("--task", choices=["auto", "text_qa", "image_classification"], default="auto")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--priority", choices=["balanced", "performance", "memory"], default="balanced")
    parser.add_argument("--model-id", default=None, help="Override the automatically selected base model")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None, help="Text max sequence length")
    parser.add_argument("--merge-text-model", action="store_true", help="Also save merged text model when possible")
    parser.add_argument("--no-auto-improve", action="store_true", help="Disable automatic retry when verdict is suggest_retrain")
    parser.add_argument("--json", action="store_true", help="Print final result as JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_automatic_training(
            data_path=args.data,
            goal=args.goal,
            task=args.task,
            output_dir=args.output,
            priority=args.priority,
            override_model_id=args.model_id,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            max_length=args.max_length,
            merge_text_model=args.merge_text_model,
            auto_improve=not args.no_auto_improve,
        )
        if args.json:
            print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        else:
            print("\nDONE")
            print(f"Task: {result.task}")
            print(f"Evaluation verdict: {result.evaluation_verdict.upper()}")
            print(f"Run directory: {result.run_dir}")
            print(f"Downloadable model zip: {result.zip_path}")
            print(f"Metrics: {json.dumps(result.metrics, indent=2, ensure_ascii=False)}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
