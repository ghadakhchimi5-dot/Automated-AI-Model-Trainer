"""Automated SLM fine-tuning for text QA/chat use cases."""

from __future__ import annotations

import inspect
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from .data_utils import TextDatasetBundle, ensure_dir, write_text_jsonl
from .metrics_utils import compute_rouge_bleu, compute_slm_metrics, infer_task_subtype, save_metrics
from .model_selector import TextModelPlan
from .packaging import write_json

LogFn = Callable[[str], None]


@dataclass(frozen=True)
class TextTrainingResult:
    run_dir: Path
    adapter_dir: Path
    merged_dir: Path | None
    tokenizer_dir: Path
    manifest_path: Path
    report_path: Path
    metrics: dict[str, Any]


def _log(log_fn: LogFn | None, message: str) -> None:
    if log_fn:
        log_fn(message)
    else:
        print(message, flush=True)


def _configure_cpu_threads(log_fn: LogFn | None = None) -> None:
    """Pin torch to physical cores for CPU training.

    Hyperthreads hurt matmul-heavy workloads (they contend for the same
    execution ports), and torch's default thread count is often the logical
    core count. No-op when CUDA is available.
    """
    if torch.cuda.is_available():
        return

    try:
        import psutil

        physical = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1
    except Exception:
        physical = os.cpu_count() or 1

    physical = int(max(1, physical))
    # OpenMP/MKL read these at first parallel-region entry; setdefault leaves
    # any user-provided value untouched.
    os.environ.setdefault("OMP_NUM_THREADS", str(physical))
    os.environ.setdefault("MKL_NUM_THREADS", str(physical))

    try:
        torch.set_num_threads(physical)
    except Exception as exc:  # pragma: no cover - platform dependent
        _log(log_fn, f"Could not set torch intra-op threads: {exc}")
        return

    try:
        torch.set_num_interop_threads(max(1, physical // 2))
    except Exception:
        # Can only be set once per process and only before parallel work starts.
        pass

    _log(log_fn, f"CPU threads: pinned torch to {physical} physical core(s)")


def _torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float16":
        return torch.float16
    return torch.float32


def _load_tokenizer(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    tokenizer.padding_side = "right"
    return tokenizer


def _chat_texts(tokenizer: Any, instruction: str, output: str) -> tuple[str, str]:
    messages_user = [{"role": "user", "content": instruction}]
    messages_full = messages_user + [{"role": "assistant", "content": output}]

    if getattr(tokenizer, "chat_template", None):
        try:
            prompt_text = tokenizer.apply_chat_template(
                messages_user,
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = tokenizer.apply_chat_template(
                messages_full,
                tokenize=False,
                add_generation_prompt=False,
            )
            return str(prompt_text), str(full_text)
        except Exception:
            pass

    prompt_text = f"### Instruction:\n{instruction}\n\n### Response:\n"
    full_text = f"{prompt_text}{output}{tokenizer.eos_token or ''}"
    return prompt_text, full_text


def _tokenize_dataset(df: pd.DataFrame, tokenizer: Any, max_length: int) -> Dataset:
    ds = Dataset.from_pandas(df[["instruction", "output"]].reset_index(drop=True))

    def preprocess(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []

        for instruction, output in zip(batch["instruction"], batch["output"]):
            prompt_text, full_text = _chat_texts(tokenizer, str(instruction), str(output))

            full = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            )

            prompt = tokenizer(
                prompt_text,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            )

            ids = list(full["input_ids"])
            mask = list(full["attention_mask"])
            lab = ids.copy()

            prompt_len = min(len(prompt["input_ids"]), len(lab))
            for i in range(prompt_len):
                lab[i] = -100

            if all(x == -100 for x in lab) and len(lab) > 0:
                tail = max(1, min(32, len(lab) // 4))
                lab[-tail:] = ids[-tail:]

            input_ids.append(ids)
            attention_mask.append(mask)
            labels.append(lab)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return ds.map(preprocess, batched=True, remove_columns=ds.column_names)


def _pack_tokenized_dataset(ds: Dataset, max_length: int) -> Dataset:
    """Greedily concatenate tokenized examples into <= max_length blocks.

    On CPU every pad token still costs a full forward+backward, so training on
    fixed-length packed blocks instead of one-short-sequence-per-step removes
    that waste and cuts the number of optimizer steps. Prompt-token masking is
    preserved because the per-example label lists (with -100 spans) are simply
    concatenated. Each example already ends with EOS from _chat_texts, which
    keeps a soft boundary between packed samples.
    """
    packed: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
    buf_ids: list[int] = []
    buf_mask: list[int] = []
    buf_labels: list[int] = []

    def flush() -> None:
        if not buf_ids:
            return
        packed["input_ids"].append(buf_ids.copy())
        packed["attention_mask"].append(buf_mask.copy())
        packed["labels"].append(buf_labels.copy())
        buf_ids.clear()
        buf_mask.clear()
        buf_labels.clear()

    for ids, mask, lab in zip(ds["input_ids"], ds["attention_mask"], ds["labels"]):
        if buf_ids and len(buf_ids) + len(ids) > max_length:
            flush()

        buf_ids.extend(ids)
        buf_mask.extend(mask)
        buf_labels.extend(lab)

        while len(buf_ids) >= max_length:
            rem_ids, buf_ids[max_length:] = buf_ids[max_length:], []
            rem_mask, buf_mask[max_length:] = buf_mask[max_length:], []
            rem_labels, buf_labels[max_length:] = buf_labels[max_length:], []
            flush()
            buf_ids.extend(rem_ids)
            buf_mask.extend(rem_mask)
            buf_labels.extend(rem_labels)

    flush()

    if not packed["input_ids"]:
        return ds
    return Dataset.from_dict(packed)


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


def _load_text_model(plan: TextModelPlan, log_fn: LogFn | None = None):
    dtype = _torch_dtype(plan.torch_dtype)
    device_map = "auto" if torch.cuda.is_available() else None
    quantization_config = None
    # Track whether quantization actually loaded (controls optimizer choice below)
    _quantization_active = False

    if plan.load_in_4bit or plan.load_in_8bit:
        try:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=plan.load_in_4bit,
                load_in_8bit=plan.load_in_8bit,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype,
            )
            _quantization_active = True
            _log(log_fn, f"Loading {plan.base_model_id} with quantization={plan.quantization}")
        except Exception as exc:
            _log(log_fn, f"bitsandbytes quantization unavailable; falling back to non-quantized load: {exc}")
            quantization_config = None

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
    }

    if device_map is not None:
        model_kwargs["device_map"] = device_map

    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(plan.base_model_id, **model_kwargs)

    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)

    # Expose flag so train_text_qa_model can pick the right optimizer
    model._bnb_quantization_active = _quantization_active

    if hasattr(model, "config"):
        model.config.use_cache = False

    return model


def _apply_lora(model: Any, plan: TextModelPlan) -> Any:
    lora_config = LoraConfig(
        r=plan.lora_rank,
        lora_alpha=plan.lora_alpha,
        lora_dropout=plan.lora_dropout,
        target_modules=plan.target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    return get_peft_model(model, lora_config)


def _scalar_slm_metric_subset(metrics: dict[str, Any]) -> dict[str, float]:
    subset: dict[str, float] = {}
    for key in ["token_accuracy"]:
        value = metrics.get(key)
        if isinstance(value, (int, float)) and np.isfinite(value):
            subset[key] = float(value)
    return subset


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cpu")


def _generate_eval_predictions(
    model: Any,
    tokenizer: Any,
    eval_df: pd.DataFrame,
    max_length: int,
    sample_size: int = 16,
) -> list[str]:
    if eval_df.empty or sample_size <= 0:
        return []

    prompt_rows = eval_df.head(sample_size)
    device = _model_device(model)
    max_new_tokens = max(32, min(128, max_length // 2 if max_length > 1 else 32))
    was_training = bool(getattr(model, "training", False))
    generated: list[str] = []

    model.eval()
    with torch.inference_mode():
        for _, row in prompt_rows.iterrows():
            prompt_text, _ = _chat_texts(tokenizer, str(row["instruction"]), "")
            encoded = tokenizer(
                prompt_text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            prompt_length = int(encoded["input_ids"].shape[1])
            output_ids = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            completion_ids = output_ids[0][prompt_length:]
            generated.append(tokenizer.decode(completion_ids, skip_special_tokens=True).strip())

    if was_training:
        model.train()
    return generated


def train_text_qa_model(
    dataset: TextDatasetBundle,
    plan: TextModelPlan,
    run_dir: str | Path,
    goal: str,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    max_length: int,
    merge_model: bool = False,
    log_fn: LogFn | None = None,
) -> TextTrainingResult:
    start = time.time()

    _configure_cpu_threads(log_fn)

    run_dir = ensure_dir(run_dir)
    data_dir = ensure_dir(run_dir / "data")
    adapter_dir = ensure_dir(run_dir / "adapter")
    tokenizer_dir = ensure_dir(run_dir / "tokenizer")
    merged_dir = run_dir / "merged_model"

    write_text_jsonl(dataset.train, data_dir / "train.jsonl")
    write_text_jsonl(dataset.eval, data_dir / "eval.jsonl")

    _log(log_fn, f"Text QA training started: base={plan.base_model_id}")
    _log(log_fn, f"PEFT={plan.peft_method}, rank={plan.lora_rank}, quantization={plan.quantization}")
    _log(log_fn, f"Rows: train={len(dataset.train)}, eval={len(dataset.eval)}")

    tokenizer = _load_tokenizer(plan.base_model_id)

    train_ds = _tokenize_dataset(dataset.train, tokenizer, max_length=max_length)
    eval_ds = _tokenize_dataset(dataset.eval, tokenizer, max_length=max_length)

    use_cuda = torch.cuda.is_available()

    # Sample packing (CPU only): the GPU path keeps one example per step so its
    # eval/convergence behaviour is unchanged; on CPU we pack to kill pad-token
    # compute. eval_ds stays unpacked so per-example metrics stay aligned.
    if not use_cuda:
        before = len(train_ds)
        train_ds = _pack_tokenized_dataset(train_ds, max_length)
        _log(
            log_fn,
            f"Sample packing: {before} examples -> {len(train_ds)} packed block(s) "
            f"of <= {max_length} tokens",
        )

    model = _load_text_model(plan, log_fn=log_fn)
    bnb_active = bool(getattr(model, "_bnb_quantization_active", False))
    model = _apply_lora(model, plan)

    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    fp16 = bool(use_cuda and plan.torch_dtype == "float16")
    bf16 = bool(use_cuda and plan.torch_dtype == "bfloat16")

    total_steps = math.ceil(len(train_ds) / max(1, batch_size * grad_accum)) * max(1, epochs)
    warmup_steps = max(0, min(100, int(total_steps * 0.05)))

    # Use paged_adamw_8bit only when bitsandbytes actually loaded; otherwise adamw_torch
    optimizer = "paged_adamw_8bit" if bnb_active else "adamw_torch"

    args_dict = _training_args_kwargs(
        {
            "output_dir": str(run_dir / "trainer"),
            "overwrite_output_dir": True,
            "num_train_epochs": int(max(1, epochs)),
            "per_device_train_batch_size": int(max(1, batch_size)),
            "per_device_eval_batch_size": int(max(1, batch_size)),
            "gradient_accumulation_steps": int(max(1, grad_accum)),
            "learning_rate": float(plan.learning_rate),
            "warmup_steps": warmup_steps,
            "logging_steps": 5,
            "save_strategy": "epoch",
            "save_total_limit": 1,
            "evaluation_strategy": "epoch",
            "report_to": [],
            "fp16": fp16,
            "bf16": bf16,
            "optim": optimizer,
            # On CPU there is abundant RAM, so trading a full forward recompute
            # for memory savings is pure waste; only checkpoint on GPU.
            "gradient_checkpointing": bool(use_cuda),
            "remove_unused_columns": False,
        }
    )

    training_args = TrainingArguments(**args_dict)

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits, labels_np = eval_pred
        full_metrics = compute_slm_metrics(logits=logits, labels=labels_np, log_fn=log_fn)
        return _scalar_slm_metric_subset(full_metrics)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    train_output = trainer.train()
    eval_output = trainer.predict(eval_ds, metric_key_prefix="eval")

    _log(log_fn, "Saving PEFT adapter and tokenizer")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(tokenizer_dir)

    merged_path: Path | None = None
    merge_error: str | None = None

    if merge_model:
        try:
            _log(log_fn, "Merging adapter into base model. This can take time and disk space.")
            merged = model.merge_and_unload() if isinstance(model, PeftModel) or hasattr(model, "merge_and_unload") else model
            ensure_dir(merged_dir)
            merged.save_pretrained(merged_dir, safe_serialization=True)
            tokenizer.save_pretrained(merged_dir)
            merged_path = merged_dir
        except Exception as exc:
            merge_error = str(exc)
            _log(log_fn, f"Merge skipped/failed: {merge_error}")

    runtime_s = time.time() - start

    eval_metrics = dict(eval_output.metrics)
    generation_sample_size = min(16, int(len(dataset.eval)))
    try:
        generated_texts = _generate_eval_predictions(
            model=model,
            tokenizer=tokenizer,
            eval_df=dataset.eval,
            max_length=max_length,
            sample_size=generation_sample_size,
        )
    except Exception as exc:
        generated_texts = []
        _log(log_fn, f"Generation-based eval metrics skipped: {exc}")
    reference_texts = dataset.eval["output"].astype(str).head(generation_sample_size).tolist()
    full_eval_metrics = compute_slm_metrics(
        eval_loss=eval_metrics.get("eval_loss"),
        logits=eval_output.predictions,
        labels=eval_output.label_ids,
        generated_texts=generated_texts,
        reference_texts=reference_texts,
        log_fn=log_fn,
    )

    task_subtype = infer_task_subtype(goal)
    if generated_texts:
        extra = compute_rouge_bleu(generated_texts, reference_texts, task_subtype=task_subtype)
        full_eval_metrics.update(extra)
    full_eval_metrics["task_subtype"] = task_subtype

    metrics: dict[str, Any] = {
        "train_runtime_s": runtime_s,
        "train_loss": float(getattr(train_output, "training_loss", np.nan)),
        "train_rows": int(len(dataset.train)),
        "eval_rows": int(len(dataset.eval)),
        "generated_qa": bool(dataset.generated_qa),
        "merge_error": merge_error,
        **full_eval_metrics,
    }
    for key, value in eval_metrics.items():
        if key.startswith("eval_") and key not in metrics and isinstance(value, (int, float)):
            metrics[key] = float(value)

    manifest = {
        "schema_version": 1,
        "task": "text_qa",
        "goal": goal,
        "base_model_id": plan.base_model_id,
        "model_type": "peft_adapter" if merged_path is None else "merged_or_adapter",
        "adapter_dir": "adapter",
        "tokenizer_dir": "tokenizer",
        "merged_model_dir": "merged_model" if merged_path else None,
        "plan": plan.to_dict(),
        "dataset_report": dataset.report,
        "metrics": metrics,
        "metrics_file": "metrics.json",
        "inference": {
            "max_new_tokens": 256,
            "temperature": 0.7,
            "top_p": 0.9,
        },
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
        "Text training completed. "
        f"Eval loss={metrics.get('eval_loss')} Perplexity={metrics.get('perplexity')} "
        f"Token accuracy={metrics.get('token_accuracy')}",
    )

    return TextTrainingResult(
        run_dir=run_dir,
        adapter_dir=adapter_dir,
        merged_dir=merged_path,
        tokenizer_dir=tokenizer_dir,
        manifest_path=manifest_path,
        report_path=report_path,
        metrics=metrics,
    )
