"""Output package helpers."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

from .data_utils import ensure_dir


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def write_json(payload: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_model_zip(run_dir: str | Path, zip_name: str | None = None) -> Path:
    """Zip a run directory and return the zip file path."""
    run_dir = Path(run_dir).resolve()
    if zip_name is None:
        zip_name = f"{run_dir.name}_model"
    zip_base = run_dir.parent / zip_name
    zip_path = Path(shutil.make_archive(str(zip_base), "zip", root_dir=run_dir))
    return zip_path


def unpack_model_zip(zip_path: str | Path, output_dir: str | Path) -> Path:
    """Unpack a previously generated model package."""
    out = ensure_dir(output_dir)
    shutil.unpack_archive(str(zip_path), str(out), "zip")
    manifest = out / "manifest.json"
    if not manifest.exists():
        # Some archive tools include a top-level folder.
        nested = list(out.rglob("manifest.json"))
        if not nested:
            raise ValueError("No manifest.json found in model zip")
        return nested[0].parent
    return out
