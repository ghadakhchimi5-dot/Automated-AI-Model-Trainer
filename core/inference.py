"""Inference helpers for generated model packages."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoImageProcessor, AutoModelForCausalLM, AutoModelForImageClassification, AutoTokenizer

from .packaging import read_json


@dataclass
class LoadedPackage:
    package_dir: Path
    manifest: dict[str, Any]
    model: Any
    processor_or_tokenizer: Any

    @property
    def task(self) -> str:
        return str(self.manifest.get("task", ""))


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


_NO_PREAMBLE_SYSTEM_PROMPT = (
    "Answer the user's question directly. Output only the answer itself: no greeting, "
    "no introductory phrase (e.g. 'Sure,', 'Great question!', 'Here is the answer:'), "
    "no restating the question, and no closing remarks."
)

_PREAMBLE_PATTERN = re.compile(
    r"^\s*"
    r"(?:(?:sure|okay|ok|certainly|of course|great question|good question|absolutely)[!,.\s]+)?"
    r"(?:(?:here'?s?|the following is|this is)\s+(?:is\s+)?(?:the\s+)?(?:answer|response)"
    r"(?:\s+to\s+(?:your|the)\s+question)?\s*[:\-]\s*)?",
    re.IGNORECASE,
)


# Conversation memory is folded into the single question block the model sees,
# because the model was fine-tuned on one-shot (instruction -> answer) pairs and
# never learned to attend to separate prior chat turns.
_MAX_HISTORY_TURNS = 6
_MAX_HISTORY_CHARS = 2000
_MAX_TURN_CHARS = 600


def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _recent_turns(history: list[dict[str, str]] | None) -> list[dict[str, str]]:
    turns = [
        {"role": msg["role"], "content": _clip(msg["content"], _MAX_TURN_CHARS)}
        for msg in (history or [])
        if msg.get("role") in ("user", "assistant") and msg.get("content")
    ][-_MAX_HISTORY_TURNS:]

    # Drop oldest turns until the rendered context fits the character budget.
    while turns and sum(len(t["content"]) for t in turns) > _MAX_HISTORY_CHARS:
        turns.pop(0)
    return turns


def _question_block(prompt: str, history: list[dict[str, str]] | None) -> str:
    """Build a single instruction containing the conversation context + question."""
    turns = _recent_turns(history)
    if not turns:
        return prompt

    transcript = "\n".join(
        f"{'User' if turn['role'] == 'user' else 'Assistant'}: {turn['content']}" for turn in turns
    )
    return (
        "Conversation so far:\n"
        f"{transcript}\n\n"
        "Using the conversation above for context, answer this question:\n"
        f"{prompt}"
    )


def _chat_prompt(tokenizer: Any, prompt: str, history: list[dict[str, str]] | None = None) -> str:
    question = _question_block(prompt, history)

    if getattr(tokenizer, "chat_template", None):
        try:
            return str(
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": _NO_PREAMBLE_SYSTEM_PROMPT},
                        {"role": "user", "content": question},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        except Exception:
            pass
        try:
            # Template may not support a system role (e.g. some Llama chat
            # templates); fold the instruction into the user turn instead.
            return str(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": f"{_NO_PREAMBLE_SYSTEM_PROMPT}\n\n{question}"}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        except Exception:
            pass

    return f"### Instruction:\n{_NO_PREAMBLE_SYSTEM_PROMPT}\n\n{question}\n\n### Response:\n"


def _strip_preamble(answer: str) -> str:
    """Remove a leading filler phrase the model prepends before the actual answer."""
    stripped = _PREAMBLE_PATTERN.sub("", answer, count=1).strip()
    return stripped or answer


def load_model_package(package_dir: str | Path) -> LoadedPackage:
    """Load a generated package for text or image inference."""
    root = Path(package_dir)
    manifest = read_json(root / "manifest.json")
    task = str(manifest.get("task"))

    if task == "text_qa":
        merged_dir = manifest.get("merged_model_dir")
        if merged_dir and (root / str(merged_dir)).exists():
            model_path = root / str(merged_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
        else:
            base_model_id = str(manifest["base_model_id"])
            tokenizer_dir = root / str(manifest.get("tokenizer_dir", "tokenizer"))
            adapter_dir = root / str(manifest.get("adapter_dir", "adapter"))
            tokenizer_source = tokenizer_dir if tokenizer_dir.exists() else base_model_id
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
            model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            model = PeftModel.from_pretrained(model, adapter_dir)
        model.eval()
        return LoadedPackage(root, manifest, model, tokenizer)

    if task == "image_classification":
        model_dir = root / str(manifest.get("model_dir", "image_model"))
        processor = AutoImageProcessor.from_pretrained(model_dir)
        model = AutoModelForImageClassification.from_pretrained(model_dir)
        model.to(_device())
        model.eval()
        return LoadedPackage(root, manifest, model, processor)

    raise ValueError(f"Unsupported package task: {task}")


def generate_text(
    package: LoadedPackage,
    prompt: str,
    history: list[dict[str, str]] | None = None,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> dict[str, Any]:
    if package.task != "text_qa":
        raise ValueError("Loaded package is not a text QA model")
    tokenizer = package.processor_or_tokenizer
    model = package.model
    input_text = _chat_prompt(tokenizer, prompt, history)
    inputs = tokenizer(input_text, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(0.01, temperature),
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        input_len = inputs["input_ids"].shape[-1]

    generated = output[0][input_len:]
    answer = tokenizer.decode(generated, skip_special_tokens=True).strip()
    answer = _strip_preamble(answer)
    return {"answer": answer, "raw": tokenizer.decode(output[0], skip_special_tokens=True)}


def classify_image(package: LoadedPackage, image: str | Path | Image.Image, top_k: int = 5) -> list[dict[str, Any]]:
    if package.task != "image_classification":
        raise ValueError("Loaded package is not an image classifier")
    processor = package.processor_or_tokenizer
    model = package.model
    if isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        pil_image = Image.open(image).convert("RGB")
    inputs = processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(_device()) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = logits.softmax(dim=-1)[0]
    top_k = max(1, min(int(top_k), probs.numel()))
    values, indices = torch.topk(probs, k=top_k)
    id2label = model.config.id2label
    results: list[dict[str, Any]] = []
    for score, idx in zip(values.tolist(), indices.tolist()):
        label = id2label.get(idx, id2label.get(str(idx), str(idx)))
        results.append({"label": str(label), "score": float(score)})
    return results
