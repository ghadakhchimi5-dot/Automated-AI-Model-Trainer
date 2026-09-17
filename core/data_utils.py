"""Dataset loading, task detection, cleaning, and conversion."""

from __future__ import annotations

import csv
import json
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
from uuid import uuid4

import pandas as pd
from sklearn.model_selection import train_test_split

TEXT_EXTENSIONS = {".txt", ".md", ".rst"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
PROMPT_COLUMNS = ["instruction", "question", "query", "prompt", "input", "text", "context", "document"]
RESPONSE_COLUMNS = ["output", "answer", "response", "target", "completion", "label"]


@dataclass(frozen=True)
class TextDatasetBundle:
    train: pd.DataFrame
    eval: pd.DataFrame
    all_rows: pd.DataFrame
    mapping: dict[str, str]
    generated_qa: bool
    report: dict[str, Any]


@dataclass(frozen=True)
class ImageDatasetBundle:
    data_dir: Path
    labels: list[str]
    num_images: int
    report: dict[str, Any]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _stream_to_path(uploaded_file: Any, dest: Path) -> Path:
    """Write an uploaded file to dest in chunks, avoiding a full in-memory buffer for large files."""
    ensure_dir(dest.parent)
    if hasattr(uploaded_file, "seek"):
        try:
            uploaded_file.seek(0)
        except Exception:
            pass
    with dest.open("wb") as f:
        if hasattr(uploaded_file, "read"):
            shutil.copyfileobj(uploaded_file, f, length=1024 * 1024)
        else:
            f.write(uploaded_file.getvalue())
    return dest


def save_uploaded_file(uploaded_file: Any, output_dir: str | Path) -> Path:
    """Persist a single Streamlit uploaded file to disk."""
    out_dir = ensure_dir(output_dir)
    filename = Path(getattr(uploaded_file, "name", "uploaded_data")).name
    return _stream_to_path(uploaded_file, out_dir / filename)


def save_uploaded_files(uploaded_files: list[Any], output_dir: str | Path) -> Path:
    """Persist one or more Streamlit uploaded files to disk.

    A single file is saved directly (identical to save_uploaded_file). Multiple files are
    merged into one shared batch folder: each zip is extracted into its own subfolder named
    after the zip (one zip = one class for image datasets), other files are copied in as-is
    so load_any_text_dataset can concatenate them.
    """
    if len(uploaded_files) == 1:
        return save_uploaded_file(uploaded_files[0], output_dir)

    batch_dir = ensure_dir(Path(output_dir) / f"batch_{uuid4().hex[:8]}")
    tmp_zip_dir = batch_dir / "_zips"
    for uploaded_file in uploaded_files:
        filename = Path(getattr(uploaded_file, "name", "uploaded_data")).name
        if filename.lower().endswith(".zip"):
            stem = Path(filename).stem or "class"
            class_dir = batch_dir / stem
            suffix = 1
            while class_dir.exists():
                class_dir = batch_dir / f"{stem}_{suffix}"
                suffix += 1
            tmp_zip = _stream_to_path(uploaded_file, tmp_zip_dir / filename)
            extract_zip(tmp_zip, class_dir)
        else:
            dest = batch_dir / filename
            suffix = 1
            while dest.exists():
                dest = batch_dir / f"{Path(filename).stem}_{suffix}{Path(filename).suffix}"
                suffix += 1
            _stream_to_path(uploaded_file, dest)
    if tmp_zip_dir.exists():
        shutil.rmtree(tmp_zip_dir, ignore_errors=True)
    return batch_dir


def detect_task(data_path: str | Path, user_task: str = "auto", goal: str = "") -> str:
    """Detect whether the run is text QA or image classification."""
    if user_task and user_task != "auto":
        return user_task
    p = Path(data_path)
    goal_lower = goal.lower()
    if any(word in goal_lower for word in ["image", "photo", "picture", "classify", "classification", "defect"]):
        if p.suffix.lower() in {".zip", ".csv"} or p.is_dir():
            return "image_classification"
    if p.is_dir():
        image_count = sum(1 for f in p.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS)
        text_count = sum(1 for f in p.rglob("*") if f.suffix.lower() in TEXT_EXTENSIONS)
        if image_count > max(3, text_count):
            return "image_classification"
        return "text_qa"
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p, "r") as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            image_count = sum(1 for n in names if Path(n).suffix.lower() in IMAGE_EXTENSIONS)
            text_count = sum(1 for n in names if Path(n).suffix.lower() in TEXT_EXTENSIONS)
        return "image_classification" if image_count > max(3, text_count) else "text_qa"
    return "text_qa"


def _read_json(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            return pd.DataFrame(payload["data"])
        if isinstance(payload.get("records"), list):
            return pd.DataFrame(payload["records"])
        return pd.DataFrame([payload])
    raise ValueError(f"Unsupported JSON payload in {path}")


def _read_jsonl(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)



def _read_tabular(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        try:
            return pd.read_csv(path)
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin-1")
    if suffix in {".tsv", ".tab"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".json":
        return _read_json(path)
    if suffix == ".jsonl":
        return _read_jsonl(path)
    raise ValueError(f"Unsupported tabular extension: {suffix}")


def _load_text_files(paths: Iterable[Path]) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for path in paths:
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1", errors="ignore")
        text = clean_text(text)
        if text:
            rows.append({"source": str(path), "text": text})
    return pd.DataFrame(rows)


def load_any_text_dataset(data_path: str | Path) -> pd.DataFrame:
    """Load CSV/JSON/JSONL/TXT/MD or a directory/zip containing those files."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.is_dir():
        tabular = [p for p in path.rglob("*") if p.suffix.lower() in {".csv", ".tsv", ".json", ".jsonl"}]
        if tabular:
            frames = [_read_tabular(p) for p in tabular]
            return pd.concat(frames, ignore_index=True)
        return _load_text_files(path.rglob("*"))

    if path.suffix.lower() == ".zip":
        extract_dir = path.parent / f"{path.stem}_extracted"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_zip(path, extract_dir)
        return load_any_text_dataset(extract_dir)

    if path.suffix.lower() in {".csv", ".tsv", ".json", ".jsonl"}:
        return _read_tabular(path)

    if path.suffix.lower() in TEXT_EXTENSIONS:
        return _load_text_files([path])

    raise ValueError(f"Unsupported text dataset: {path}")


def clean_text(text: Any) -> str:
    value = "" if text is None else str(text)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _find_column(df: pd.DataFrame, names: list[str]) -> str | None:
    lower_to_original = {str(c).lower().strip(): str(c) for c in df.columns}
    for name in names:
        if name in lower_to_original:
            return lower_to_original[name]
    for col_lower, original in lower_to_original.items():
        if any(name in col_lower for name in names):
            return original
    return None


def _normalise_messages(value: Any) -> tuple[str, str] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return None
    if not isinstance(value, list):
        return None
    user_parts: list[str] = []
    assistant_parts: list[str] = []
    for msg in value:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).lower()
        content = clean_text(msg.get("content", ""))
        if not content:
            continue
        if role in {"user", "human"}:
            user_parts.append(content)
        elif role in {"assistant", "gpt", "bot"}:
            assistant_parts.append(content)
    if user_parts and assistant_parts:
        return "\n".join(user_parts[-2:]), assistant_parts[-1]
    return None


def _keywords(text: str, max_words: int = 6) -> str:
    words = re.findall(r"[A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9_-]{2,}", text.lower())
    stop = {
        "the", "and", "for", "with", "from", "that", "this", "are", "was", "were", "you", "your",
        "les", "des", "pour", "avec", "dans", "une", "est", "que", "qui", "sur", "par", "aux",
    }
    counts: dict[str, int] = {}
    for w in words:
        if w not in stop:
            counts[w] = counts.get(w, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:max_words]
    return ", ".join(k for k, _ in ordered) or "this document"


def _chunk_text(text: str, min_chars: int = 180, max_chars: int = 1300) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    # Split first by sentence boundaries, then pack sentences into chunks.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
        else:
            if len(current) >= min_chars:
                chunks.append(current)
            current = sentence[:max_chars]
    if len(current) >= min_chars:
        chunks.append(current)
    if not chunks and len(text) >= 40:
        chunks.append(text[:max_chars])
    return chunks


def generate_synthetic_qa_from_text(df: pd.DataFrame, goal: str = "") -> pd.DataFrame:
    """Create local extractive QA pairs from text-only data.

    This is deterministic and offline. It is designed to bootstrap a chat SLM from documents
    when no answer column exists.
    """
    text_col = _find_column(df, ["text", "context", "document", "content", "body"]) or (str(df.columns[0]) if len(df.columns) else None)
    if text_col is None:
        return pd.DataFrame(columns=["instruction", "output", "source"])

    rows: list[dict[str, str]] = []
    for idx, row in df.iterrows():
        text = clean_text(row.get(text_col, ""))
        source = clean_text(row.get("source", f"row_{idx}"))
        for chunk_idx, chunk in enumerate(_chunk_text(text)):
            topic = _keywords(chunk)
            if goal:
                question = f"For the goal '{goal}', what should the assistant know about {topic}?"
            else:
                question = f"What does the source document say about {topic}?"
            rows.append({"instruction": question, "output": chunk, "source": f"{source}#chunk{chunk_idx}"})
    return pd.DataFrame(rows)


def to_instruction_output(df: pd.DataFrame, goal: str = "") -> tuple[pd.DataFrame, dict[str, str], bool]:
    """Convert many text formats to instruction/output rows."""
    if df.empty:
        raise ValueError("Text dataset is empty")

    cols_lower = {str(c).lower(): str(c) for c in df.columns}
    rows: list[dict[str, str]] = []
    if "messages" in cols_lower:
        msg_col = cols_lower["messages"]
        for _, row in df.iterrows():
            parsed = _normalise_messages(row.get(msg_col))
            if parsed:
                prompt, response = parsed
                rows.append({"instruction": prompt, "output": response})
        if rows:
            return pd.DataFrame(rows), {"prompt": "messages", "response": "messages"}, False

    prompt_col = _find_column(df, PROMPT_COLUMNS)
    response_col = _find_column(df, RESPONSE_COLUMNS)

    if prompt_col and response_col and prompt_col != response_col:
        for _, row in df.iterrows():
            prompt = clean_text(row.get(prompt_col, ""))
            response = clean_text(row.get(response_col, ""))
            extra_input = ""
            if "input" in cols_lower and cols_lower["input"] not in {prompt_col, response_col}:
                extra_input = clean_text(row.get(cols_lower["input"], ""))
            if extra_input:
                prompt = f"{prompt}\n{extra_input}".strip()
            if prompt and response:
                rows.append({"instruction": prompt, "output": response})
        return pd.DataFrame(rows), {"prompt": prompt_col, "response": response_col}, False

    generated = generate_synthetic_qa_from_text(df, goal=goal)
    return generated, {"prompt": "synthetic_question", "response": "synthetic_answer"}, True


def quality_report(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"rows": 0, "quality_score": 0.0, "avg_prompt_chars": 0.0, "avg_response_chars": 0.0}
    prompt_len = df["instruction"].astype(str).str.len()
    response_len = df["output"].astype(str).str.len()
    non_empty = ((prompt_len > 0) & (response_len > 0)).mean()
    # After filtering, all rows already pass min-length; measure response richness (>=50 chars)
    richness = (response_len >= 50).mean()
    diversity = min(1.0, df["instruction"].nunique() / max(1, len(df)))
    score = round((0.40 * non_empty + 0.40 * richness + 0.20 * diversity) * 100, 1)
    return {
        "rows": int(len(df)),
        "quality_score": score,
        "avg_prompt_chars": float(prompt_len.mean()),
        "avg_response_chars": float(response_len.mean()),
        "duplicates": int(df.duplicated(subset=["instruction", "output"]).sum()),
        "short_responses_pct": float(round((response_len < 50).mean() * 100, 1)),
    }


_CHITCHAT_EXAMPLES: list[dict[str, str]] = [
    {"instruction": "hello", "output": "Hello! How can I help you today?"},
    {"instruction": "hi", "output": "Hi there! What can I help you with?"},
    {"instruction": "hey", "output": "Hey! What would you like to know?"},
    {"instruction": "hi there", "output": "Hi! How can I assist you today?"},
    {"instruction": "good morning", "output": "Good morning! How can I help you today?"},
    {"instruction": "good evening", "output": "Good evening! How can I help you today?"},
    {"instruction": "how are you?", "output": "I'm doing well, thanks for asking! How can I help you today?"},
    {
        "instruction": "who are you?",
        "output": "I'm an assistant fine-tuned to help answer your questions. What would you like to know?",
    },
    {"instruction": "thanks", "output": "You're welcome! Let me know if you need anything else."},
    {"instruction": "thank you", "output": "You're welcome! Happy to help with anything else you need."},
    {"instruction": "bye", "output": "Goodbye! Feel free to come back if you have more questions."},
    {"instruction": "goodbye", "output": "Goodbye! Feel free to come back if you have more questions."},
]


def _augment_with_chitchat(train_df: pd.DataFrame, eval_df: pd.DataFrame, repeats: int = 3) -> pd.DataFrame:
    """Fold a few greeting/chit-chat examples into training data, repeated so a small
    LoRA fine-tune actually picks up the pattern instead of it being drowned out by the
    (usually much larger) domain-specific dataset. Skips any instruction already covered
    by the uploaded data so we don't fight a conflicting answer the user provided.
    """
    existing = {
        str(v).strip().lower()
        for v in pd.concat([train_df["instruction"], eval_df["instruction"]], ignore_index=True)
    }
    new_rows = [row for row in _CHITCHAT_EXAMPLES if row["instruction"].lower() not in existing]
    if not new_rows:
        return train_df
    chitchat_df = pd.DataFrame(new_rows * repeats)
    return pd.concat([train_df, chitchat_df], ignore_index=True)


def prepare_text_dataset(
    data_path: str | Path,
    goal: str = "",
    eval_ratio: float = 0.1,
    max_samples: int | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> TextDatasetBundle:
    """Load, clean, convert, deduplicate, and split a text dataset."""
    raw = load_any_text_dataset(data_path)
    converted, mapping, generated_qa = to_instruction_output(raw, goal=goal)
    if converted.empty:
        raise ValueError("No valid instruction/output examples found or generated")

    converted["instruction"] = converted["instruction"].map(clean_text)
    converted["output"] = converted["output"].map(clean_text)

    # Filter out rows that are too short to be useful training signal
    MIN_PROMPT_CHARS = 5
    MIN_RESPONSE_CHARS = 10
    # Cap extremely long rows to avoid OOM during tokenization (rough char budget)
    MAX_COMBINED_CHARS = 8000
    converted = converted[
        (converted["instruction"].str.len() >= MIN_PROMPT_CHARS)
        & (converted["output"].str.len() >= MIN_RESPONSE_CHARS)
        & ((converted["instruction"].str.len() + converted["output"].str.len()) <= MAX_COMBINED_CHARS)
    ]
    converted = converted.drop_duplicates(subset=["instruction", "output"]).reset_index(drop=True)
    if max_samples is not None and len(converted) > max_samples:
        converted = converted.sample(n=max_samples, random_state=42).reset_index(drop=True)

    MIN_EVAL_ROWS = 50
    eval_size_warning: str | None = None

    if len(converted) < 2:
        train_df = converted.copy()
        eval_df = converted.copy()
    else:
        n_total = len(converted)
        base_test_size = min(max(eval_ratio, 0.05), 0.3)
        # A fixed ratio starves the eval set on small datasets (e.g. 85 rows * 0.1
        # = 9 rows), which makes every eval metric noise. Grow the split toward a
        # minimum row count when the dataset can afford it, capping at half the
        # data so training isn't starved instead.
        target_eval_rows = min(MIN_EVAL_ROWS, n_total // 2)
        test_size = min(max(base_test_size, target_eval_rows / n_total), 0.5)
        if target_eval_rows < MIN_EVAL_ROWS:
            eval_size_warning = (
                f"Dataset has only {n_total} usable rows after cleaning, so the evaluation "
                f"split is capped at {target_eval_rows} rows (below the {MIN_EVAL_ROWS}-row "
                "floor needed for stable metrics). Treat eval metrics on this run as indicative, "
                "not conclusive — collect more data to get trustworthy numbers."
            )

        train_df, eval_df = train_test_split(converted, test_size=test_size, random_state=42, shuffle=True)
        train_df = train_df.reset_index(drop=True)
        eval_df = eval_df.reset_index(drop=True)

    train_df = _augment_with_chitchat(train_df, eval_df).reset_index(drop=True)

    report = quality_report(converted)
    report.update({"raw_rows": int(len(raw)), "train_rows": int(len(train_df)), "eval_rows": int(len(eval_df))})
    if eval_size_warning:
        report["eval_size_warning"] = eval_size_warning
        message = f"[prepare_text_dataset] Warning: {eval_size_warning}"
        if log_fn:
            log_fn(message)
        else:
            print(message, flush=True)
    return TextDatasetBundle(train=train_df, eval=eval_df, all_rows=converted, mapping=mapping, generated_qa=generated_qa, report=report)


def extract_zip(zip_path: str | Path, output_dir: str | Path) -> Path:
    """Extract a zip safely to output_dir."""
    zip_path = Path(zip_path)
    output = ensure_dir(output_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            target = output / member.filename
            if not str(target.resolve()).startswith(str(output.resolve())):
                raise ValueError(f"Unsafe path in zip: {member.filename}")
        zf.extractall(output)
    return output


def _find_image_root(path: Path) -> Path:
    """Return directory whose direct subdirectories are class folders."""
    if not path.is_dir():
        raise ValueError(f"Image root is not a directory: {path}")
    candidates = [path] + [p for p in path.rglob("*") if p.is_dir()]
    best = path
    best_count = -1
    for candidate in candidates:
        class_dirs = [d for d in candidate.iterdir() if d.is_dir()]
        image_count = sum(1 for d in class_dirs for f in d.rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS)
        if image_count > best_count:
            best = candidate
            best_count = image_count
    return best


def _copy_csv_image_dataset(csv_path: Path, output_dir: Path) -> Path:
    df = pd.read_csv(csv_path)
    image_col = _find_column(df, ["image", "image_path", "path", "file", "filename"])
    label_col = _find_column(df, ["label", "class", "category", "target"])
    if image_col is None or label_col is None:
        raise ValueError("CSV image dataset must contain image/path and label/class columns")
    root = ensure_dir(output_dir / "imagefolder")
    base = csv_path.parent
    for _, row in df.iterrows():
        src = Path(str(row[image_col]))
        if not src.is_absolute():
            src = base / src
        if not src.exists() or src.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(row[label_col]).strip()) or "unknown"
        label_dir = ensure_dir(root / label)
        dest = label_dir / src.name
        if dest.exists():
            dest = label_dir / f"{src.stem}_{abs(hash(str(src))) % 10_000}{src.suffix}"
        shutil.copy2(src, dest)
    return root


def prepare_image_dataset(data_path: str | Path, work_dir: str | Path) -> ImageDatasetBundle:
    """Prepare an imagefolder dataset from zip/folder/csv."""
    path = Path(data_path)
    work = ensure_dir(work_dir)
    if not path.exists():
        raise FileNotFoundError(f"Image dataset not found: {path}")

    if path.suffix.lower() == ".zip":
        extract_dir = work / f"{path.stem}_images"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_zip(path, extract_dir)
        root = _find_image_root(extract_dir)
    elif path.suffix.lower() == ".csv":
        root = _copy_csv_image_dataset(path, work)
    elif path.is_dir():
        root = _find_image_root(path)
    else:
        raise ValueError("Image classification requires a zip, folder, or CSV with image paths")

    labels = sorted([p.name for p in root.iterdir() if p.is_dir() and any(f.suffix.lower() in IMAGE_EXTENSIONS for f in p.rglob("*"))])
    if len(labels) < 2:
        raise ValueError("Image classification needs at least two class folders with images")
    num_images = sum(1 for label in labels for f in (root / label).rglob("*") if f.suffix.lower() in IMAGE_EXTENSIONS)
    return ImageDatasetBundle(
        data_dir=root,
        labels=labels,
        num_images=num_images,
        report={"num_images": num_images, "num_classes": len(labels), "labels": labels, "data_dir": str(root)},
    )


def write_text_jsonl(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps({"instruction": row["instruction"], "output": row["output"]}, ensure_ascii=False) + "\n")
    return path


def write_csv_report(rows: list[dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path
