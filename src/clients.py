"""
Gauntlet — API clients for local models and Claude Code CLI.

All models run through Claude Code CLI with identical tool access (config/mcp.json).
Built-in tools disabled. Only MCP tools (e.g. web search) if configured.

Designed by SK. Built by Claude.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
from pathlib import Path

import httpx

log = logging.getLogger("gauntlet")

ROOT = Path(__file__).resolve().parent.parent
MCP_CONFIG = str(ROOT / "config" / "mcp.json")


# ── model detection ────────────────────────────────────────────────────
_model_cache: dict[str, tuple[str, int | None]] = {}


async def _resolve_model(endpoint: str) -> tuple[str, int | None]:
    """Auto-detect model name and max context length via /v1/models."""
    if endpoint in _model_cache:
        return _model_cache[endpoint]

    base = endpoint.rsplit("/v1/", 1)[0]
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(f"{base}/v1/models")
        resp.raise_for_status()
    data = resp.json()
    model_info = data["data"][0]
    name = model_info["id"]
    max_len = model_info.get("max_model_len")
    _model_cache[endpoint] = (name, max_len)
    log.info("Auto-detected model: %s  max_context: %s", name, max_len)
    return name, max_len


# ── claude code CLI (shared) ──────────────────────────────────────────
def _build_claude_cmd(*, model: str, effort: str | None = None) -> list[str]:
    """Build Claude Code CLI command with consistent tool isolation."""
    cmd = [
        "claude", "-p",
        "--model", model,
        "--tools", "",
        "--mcp-config", MCP_CONFIG,
        "--strict-mcp-config",
        "--permission-mode", "bypassPermissions",
        "--output-format", "json",
    ]
    if effort:
        cmd.extend(["--effort", effort])
    return cmd


def _run_claude(prompt: str, cmd: list[str], *, timeout: int, env: dict | None = None) -> dict:
    """Execute Claude Code CLI and parse the response."""
    t0 = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(input=prompt, timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, 9)
        proc.wait()
        raise

    ms = (time.monotonic() - t0) * 1000

    if proc.returncode != 0:
        raise RuntimeError(f"Claude Code failed: {stderr[:500]}")

    data = json.loads(stdout)
    content = data.get("result", "")

    return {
        "content": content,
        "model": data.get("model") or cmd[cmd.index("--model") + 1],
        "duration_ms": round(ms, 1),
        "tokens": {
            "prompt": data.get("usage", {}).get("input_tokens"),
            "completion": data.get("usage", {}).get("output_tokens"),
        },
    }


# ── local model via claude code CLI ──────────────────────────────────
def call_local_via_claude_code(
    prompt: str,
    *,
    base_url: str,
    model_id: str,
    timeout: int = 900,
) -> dict:
    """Run a prompt through Claude Code CLI pointed at a local model."""
    cmd = _build_claude_cmd(model=model_id)
    env = {**os.environ, "ANTHROPIC_BASE_URL": base_url}
    return _run_claude(prompt, cmd, timeout=timeout, env=env)


async def call_local_via_claude_code_async(
    prompt: str,
    *,
    base_url: str,
    model_id: str,
    timeout: int = 900,
) -> dict:
    """Async wrapper around call_local_via_claude_code."""
    return await asyncio.to_thread(
        call_local_via_claude_code, prompt,
        base_url=base_url, model_id=model_id, timeout=timeout,
    )


# ── baseline + judge via claude code CLI ─────────────────────────────
def call_claude_code(
    prompt: str,
    *,
    model: str = "opus",
    effort: str = "max",
    timeout: int = 600,
) -> dict:
    """Run a prompt through Claude Code CLI (baselines and judge)."""
    cmd = _build_claude_cmd(model=model, effort=effort)
    return _run_claude(prompt, cmd, timeout=timeout)
