"""MCP server exposing the Data Agent as the `prepare_dataset` tool.

This wraps core.agents.data_agent.DataAgent.run() exactly as-is — no
dataset-preparation logic lives here. The only thing this module adds is
the MCP transport boundary: pandas DataFrames produced for text_qa
datasets are not JSON-serializable, so they are persisted to Parquet
files under <run_dir>/prepared/ and the tool returns paths instead of
the DataFrames themselves. Image datasets are already disk-backed by
prepare_image_dataset(), so no extra persistence is needed there.

Run standalone for manual testing:
    python -m mcp_layer.servers.data_server        # stdio MCP server
    mcp dev mcp_layer/servers/data_server.py        # MCP inspector UI
"""

from __future__ import annotations

import sys
from pathlib import Path

# Guarantee the project root is importable regardless of how this file is launched.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from mcp.server.fastmcp import Context, FastMCP  # noqa: E402

from core.agents import DataAgent  # noqa: E402
from core.data_utils import ImageDatasetBundle, TextDatasetBundle, ensure_dir  # noqa: E402
from mcp_layer.schemas import BundleRef, PrepareDatasetInput, PrepareDatasetOutput  # noqa: E402

mcp = FastMCP("ghd-data-agent")


@mcp.tool()
async def prepare_dataset(input: PrepareDatasetInput, ctx: Context) -> PrepareDatasetOutput:
    """Detect the task type, load and clean the dataset, and assess data quality.

    Wraps DataAgent.run() unchanged. For text_qa tasks, the resulting
    train/eval DataFrames are written to Parquet under <run_dir>/prepared/
    so they can be referenced by path in the tool's JSON response.
    """
    # DataAgent.run() takes a synchronous log_fn callback. We buffer the
    # messages it produces and forward them as MCP logging notifications
    # after the (fast, non-training) call completes, rather than plumbing
    # async calls through a sync callback.
    log_buffer: list[str] = []

    try:
        run_dir = ensure_dir(Path(input.run_dir))
        agent = DataAgent()
        analysis = agent.run(
            data_path=input.data_path,
            goal=input.goal,
            user_task=input.user_task,
            max_samples=input.max_samples,
            run_dir=run_dir,
            log_fn=log_buffer.append,
        )
    except Exception as exc:
        await ctx.error(f"[DataAgent] prepare_dataset failed: {exc}")
        raise

    for msg in log_buffer:
        await ctx.info(msg)

    bundle_ref = _persist_bundle(analysis.bundle, analysis.task, run_dir)
    await ctx.info(
        f"[DataAgent] prepare_dataset done — task={analysis.task}, "
        f"quality={analysis.quality_score:.0f}/100 ({analysis.quality_label})."
    )

    return PrepareDatasetOutput(
        task=analysis.task,
        quality_score=analysis.quality_score,
        quality_label=analysis.quality_label,
        warnings=analysis.warnings,
        recommendations=analysis.recommendations,
        reasoning=analysis.reasoning,
        report=analysis.report,
        bundle_ref=bundle_ref,
    )


def _persist_bundle(bundle, task: str, run_dir: Path) -> BundleRef:
    """Persist a dataset bundle to disk and return a JSON-safe reference to it."""
    if task == "text_qa":
        assert isinstance(bundle, TextDatasetBundle)
        prepared_dir = ensure_dir(run_dir / "prepared")
        train_path = prepared_dir / "train.parquet"
        eval_path = prepared_dir / "eval.parquet"
        all_rows_path = prepared_dir / "all_rows.parquet"
        bundle.train.to_parquet(train_path, index=False)
        bundle.eval.to_parquet(eval_path, index=False)
        bundle.all_rows.to_parquet(all_rows_path, index=False)
        return BundleRef(
            kind="text",
            train_path=str(train_path),
            eval_path=str(eval_path),
            all_rows_path=str(all_rows_path),
            mapping=bundle.mapping,
            generated_qa=bundle.generated_qa,
        )

    assert isinstance(bundle, ImageDatasetBundle)
    return BundleRef(
        kind="image",
        data_dir=str(bundle.data_dir),
        labels=bundle.labels,
        num_images=bundle.num_images,
    )


if __name__ == "__main__":
    mcp.run()
