"""MCP (Model Context Protocol) coordination layer for GHD Auto Trainer.

This package is purely additive: it exposes the existing agents in
core/agents/ as MCP tools (one server process per agent) and provides an
MCP-client orchestrator that calls them in sequence. It does not change
any logic in core/ — it only adds a serialization boundary (disk handoff)
so agent inputs/outputs can travel as JSON over the MCP protocol.

The original in-process pipeline (auto_pipeline.run_automatic_training)
is untouched and remains the default.
"""
