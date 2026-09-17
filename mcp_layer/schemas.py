"""Typed I/O schemas for MCP tools.

Each model here mirrors an existing dataclass in core/agents/*.py or
core/data_utils.py. The only new concept is BundleRef: since MCP tool
payloads must be JSON, we cannot hand a live pandas.DataFrame (text) or
loaded tensors across the wire. Instead, a tool that produces a dataset
bundle persists it to disk under the run directory and returns paths;
the next tool reads those paths back in. Nothing about the underlying
dataset preparation logic changes — this is purely a transport boundary.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class BundleRef(BaseModel):
    """Disk pointer to a prepared dataset bundle, produced by prepare_dataset."""

    kind: Literal["text", "image"]

    # text_qa: paths to Parquet files written by the data_server
    train_path: str | None = None
    eval_path: str | None = None
    all_rows_path: str | None = None
    mapping: dict[str, str] | None = None
    generated_qa: bool | None = None

    # image_classification: already disk-backed in the original bundle
    data_dir: str | None = None
    labels: list[str] | None = None
    num_images: int | None = None


class PrepareDatasetInput(BaseModel):
    """Input for the prepare_dataset tool (Data Agent)."""

    data_path: str
    goal: str = ""
    user_task: str = "auto"
    max_samples: int | None = None
    run_dir: str = Field(
        ..., description="Directory to persist prepared dataset artifacts and run outputs."
    )


class PrepareDatasetOutput(BaseModel):
    """Output of the prepare_dataset tool (Data Agent). Mirrors DataAnalysis."""

    task: str
    quality_score: float
    quality_label: str
    warnings: list[str]
    recommendations: list[str]
    reasoning: str
    report: dict[str, Any]
    bundle_ref: BundleRef
