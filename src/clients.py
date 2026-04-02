"""
Gauntlet — API clients for local models and Claude Code CLI.

Two paths:
  - Local models: httpx to OpenAI-compatible /v1/chat/completions
  - Baseline + judge: Claude Code CLI subprocess (latest Opus)

Designed by SK. Built by Claude.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time

import httpx

log = logging.getLogger("gauntlet")


# ── local model (OpenAI-compatible) ────────────────────────────────────
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


async def _resolve_model_id(endpoint: str, model_id: str) -> str:
    """Resolve 'auto' to the actual model name."""
    if model_id != "auto":
        return model_id
    name, _ = await _resolve_model(endpoint)
    return name


async def call_local(
    endpoint: str,
    model_id: str,
    system: str,
    messages: list[dict],
    *,
    max_tokens: int | None = None,
    sampling: dict | None = None,
    timeout: float = 600,
) -> dict:
    """Send a chat completion to a local OpenAI-compatible endpoint."""
    model_id = await _resolve_model_id(endpoint, model_id)
    payload = {
        "model": model_id,
        "messages": [{"role": "system", "content": system}, *messages],
        "stream": False,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if sampling:
        payload.update(sampling)

    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(endpoint, json=payload)
        resp.raise_for_status()

    data = resp.json()
    ms = (time.monotonic() - t0) * 1000
    msg = data["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning") or ""
    model = data.get("model", model_id)
    usage = data.get("usage", {})

    result = {
        "content": content,
        "model": model,
        "duration_ms": round(ms, 1),
        "tokens": {
            "prompt": usage.get("prompt_tokens"),
            "completion": usage.get("completion_tokens"),
        },
    }
    if reasoning:
        result["reasoning"] = reasoning
    return result


# ── claude code CLI ────────────────────────────────────────────────────
def call_claude_code(
    prompt: str,
    *,
    model: str = "opus",
    effort: str = "max",
    timeout: int = 600,
) -> dict:
    """Run a prompt through Claude Code CLI."""
    t0 = time.monotonic()
    result = subprocess.run(
        [
            "claude", "-p",
            "--model", model,
            "--effort", effort,
            "--output-format", "json",
        ],
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    ms = (time.monotonic() - t0) * 1000

    if result.returncode != 0:
        raise RuntimeError(f"Claude Code failed: {result.stderr[:500]}")

    data = json.loads(result.stdout)
    content = data.get("result", "")

    return {
        "content": content,
        "model": data.get("model", model),
        "duration_ms": round(ms, 1),
        "tokens": {
            "prompt": data.get("usage", {}).get("input_tokens"),
            "completion": data.get("usage", {}).get("output_tokens"),
        },
    }


async def call_claude_code_async(
    prompt: str,
    *,
    timeout: int = 600,
) -> dict:
    """Async wrapper around call_claude_code."""
    return await asyncio.to_thread(call_claude_code, prompt, timeout=timeout)
