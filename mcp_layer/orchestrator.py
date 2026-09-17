"""MCP client / orchestrator.

Phase 1 scope: launches the Data Agent MCP server as a subprocess,
discovers its tools, calls `prepare_dataset`, and rehydrates the result
back into a DataAnalysis object so it is a drop-in replacement for
calling DataAgent().run(...) directly. Phases 2+ will add the remaining
four servers (model, optimization, training, evaluation) and chain them
the same way auto_pipeline.run_automatic_training() does today.

This module is async internally (the MCP SDK is async) but exposes a
synchronous function so it can be called from Streamlit / the CLI
without changing their execution model.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import LoggingMessageNotificationParams

from core.agents.data_agent import DataAnalysis
from core.data_utils import ImageDatasetBundle, TextDatasetBundle
from mcp_layer.schemas import BundleRef, PrepareDatasetInput, PrepareDatasetOutput

LogFn = Callable[[str], None]

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _server_params(module: str) -> StdioServerParameters:
    """Build stdio launch parameters for one of our MCP servers.

    Uses the same Python interpreter running this process (sys.executable)
    so the server sees the same virtualenv/dependencies, and pins cwd to
    the project root so `core.*` and `mcp_layer.*` imports resolve.
    """
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", module],
        cwd=str(_PROJECT_ROOT),
    )


def _make_logging_callback(log_fn: LogFn | None):
    """Build an MCP logging_callback that forwards server log messages to log_fn."""

    async def _on_log(params: LoggingMessageNotificationParams) -> None:
        data = params.data
        text = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
        if log_fn:
            log_fn(text)
        else:
            print(text, flush=True)

    return _on_log


async def _call_tool(
    session: ClientSession,
    tool_name: str,
    arguments: dict[str, Any],
    log_fn: LogFn | None,
) -> dict[str, Any]:
    """Call one MCP tool, with discovery/validation and error handling.

    Raises RuntimeError with a clear message if the tool is missing or if
    the tool call itself reports an error, so failures behave the same way
    an in-process exception would in auto_pipeline.run_automatic_training().
    """
    tools = (await session.list_tools()).tools
    tool_names = [t.name for t in tools]
    if tool_name not in tool_names:
        raise RuntimeError(
            f"MCP tool '{tool_name}' was not discovered on this server (found: {tool_names})"
        )

    def _log(msg: str) -> None:
        if log_fn:
            log_fn(msg)
        else:
            print(msg, flush=True)

    _log(f"[MCP] Calling tool '{tool_name}'...")
    try:
        result = await session.call_tool(tool_name, arguments=arguments)
    except Exception as exc:
        _log(f"[MCP] Tool call '{tool_name}' raised an exception: {exc}")
        raise RuntimeError(f"MCP tool call '{tool_name}' failed: {exc}") from exc

    if result.isError:
        error_text = "; ".join(
            block.text for block in result.content if hasattr(block, "text")
        )
        _log(f"[MCP] Tool '{tool_name}' returned an error: {error_text}")
        raise RuntimeError(f"MCP tool '{tool_name}' returned an error: {error_text}")

    if result.structuredContent is not None:
        _log(f"[MCP] Tool '{tool_name}' completed.")
        return result.structuredContent

    # Fallback for servers/SDK versions that don't populate structuredContent:
    # parse the first text content block as JSON.
    text_blocks = [block.text for block in result.content if hasattr(block, "text")]
    if not text_blocks:
        raise RuntimeError(f"MCP tool '{tool_name}' returned no usable content.")
    _log(f"[MCP] Tool '{tool_name}' completed.")
    return json.loads(text_blocks[0])


def _rehydrate_bundle(ref: BundleRef):
    """Reconstruct a TextDatasetBundle / ImageDatasetBundle from a BundleRef."""
    if ref.kind == "text":
        train = pd.read_parquet(ref.train_path)
        evalset = pd.read_parquet(ref.eval_path)
        all_rows = pd.read_parquet(ref.all_rows_path)
        return TextDatasetBundle(
            train=train,
            eval=evalset,
            all_rows=all_rows,
            mapping=ref.mapping or {},
            generated_qa=bool(ref.generated_qa),
            report={},  # report is carried separately on DataAnalysis.report
        )
    return ImageDatasetBundle(
        data_dir=Path(ref.data_dir),
        labels=ref.labels or [],
        num_images=ref.num_images or 0,
        report={},
    )


async def _prepare_dataset_via_mcp_async(
    data_path: str | Path,
    goal: str,
    user_task: str,
    max_samples: int | None,
    run_dir: str | Path,
    log_fn: LogFn | None,
) -> DataAnalysis:
    params = _server_params("mcp_layer.servers.data_server")
    logging_callback = _make_logging_callback(log_fn)

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write, logging_callback=logging_callback) as session:
            await session.initialize()

            tool_input = PrepareDatasetInput(
                data_path=str(data_path),
                goal=goal,
                user_task=user_task,
                max_samples=max_samples,
                run_dir=str(run_dir),
            )
            raw_result = await _call_tool(
                session,
                "prepare_dataset",
                {"input": tool_input.model_dump()},
                log_fn,
            )

    output = PrepareDatasetOutput.model_validate(raw_result)
    bundle = _rehydrate_bundle(output.bundle_ref)
    # report was persisted inside output.report already; keep it on the bundle too
    # so downstream code that reads bundle.report (as the in-process path does) still works.
    bundle = _with_report(bundle, output.report)

    return DataAnalysis(
        task=output.task,
        bundle=bundle,
        quality_score=output.quality_score,
        quality_label=output.quality_label,
        warnings=output.warnings,
        recommendations=output.recommendations,
        reasoning=output.reasoning,
        report=output.report,
    )


def _with_report(bundle, report: dict[str, Any]):
    """Bundles are frozen dataclasses; return a copy with report populated."""
    from dataclasses import replace

    return replace(bundle, report=report)


def prepare_dataset_via_mcp(
    data_path: str | Path,
    goal: str,
    user_task: str = "auto",
    max_samples: int | None = None,
    run_dir: str | Path = "output/mcp_run",
    log_fn: LogFn | None = None,
) -> DataAnalysis:
    """Synchronous entry point: run the Data Agent through MCP and return a DataAnalysis.

    Drop-in replacement for DataAgent().run(...) — same return type — but the
    call is routed through a real MCP client/server round trip instead of an
    in-process function call.
    """
    return asyncio.run(
        _prepare_dataset_via_mcp_async(
            data_path=data_path,
            goal=goal,
            user_task=user_task,
            max_samples=max_samples,
            run_dir=run_dir,
            log_fn=log_fn,
        )
    )


if __name__ == "__main__":
    # Manual end-to-end smoke test:
    #   python -m mcp_layer.orchestrator
    sample = _PROJECT_ROOT / "sample_data" / "text_qa_sample.csv"
    analysis = prepare_dataset_via_mcp(
        data_path=sample,
        goal="Create a support chatbot",
        run_dir=_PROJECT_ROOT / "output" / "mcp_poc_run",
        log_fn=print,
    )
    print("\n--- DataAnalysis via MCP ---")
    print("task:", analysis.task)
    print("quality:", analysis.quality_score, analysis.quality_label)
    print("train rows:", len(analysis.bundle.train))
    print("eval rows:", len(analysis.bundle.eval))
