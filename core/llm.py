"""Anthropic client wrapper with prompt caching and structured JSON output."""
from __future__ import annotations
import json
import time
from typing import Any

import anthropic

from core.config import ANTHROPIC_API_KEY, LLM_MODEL


_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        if not ANTHROPIC_API_KEY:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client


def chat(
    system_prompt: str,
    user_message: str,
    *,
    model: str = LLM_MODEL,
    max_tokens: int = 4096,
    cache_system: bool = True,
    json_mode: bool = False,
    temperature: float = 0.3,
) -> str:
    """Single-turn LLM call. Returns the assistant text content."""
    client = get_client()

    system_blocks: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": system_prompt,
            **({"cache_control": {"type": "ephemeral"}} if cache_system else {}),
        }
    ]

    if json_mode:
        user_message = user_message + "\n\nRespond with valid JSON only — no markdown fences."

    t0 = time.time()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_blocks,  # type: ignore[arg-type]
        messages=[{"role": "user", "content": user_message}],
    )
    elapsed = time.time() - t0

    usage = response.usage
    _log_usage(usage, elapsed, model)

    return response.content[0].text  # type: ignore[index]


def chat_json(
    system_prompt: str,
    user_message: str,
    *,
    model: str = LLM_MODEL,
    max_tokens: int = 4096,
    cache_system: bool = True,
) -> dict[str, Any]:
    """Call LLM and parse the response as JSON. Retries once on parse failure."""
    raw = chat(
        system_prompt,
        user_message,
        model=model,
        max_tokens=max_tokens,
        cache_system=cache_system,
        json_mode=True,
    )
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # strip markdown fences and retry parse
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(cleaned)


def _log_usage(usage: Any, elapsed: float, model: str) -> None:
    try:
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        input_tok = getattr(usage, "input_tokens", 0) or 0
        output_tok = getattr(usage, "output_tokens", 0) or 0
        print(
            f"  [LLM] {model} | "
            f"in={input_tok} out={output_tok} "
            f"cache_read={cache_read} cache_write={cache_write} | "
            f"{elapsed:.1f}s"
        )
    except Exception:
        pass
