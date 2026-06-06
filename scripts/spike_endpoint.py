#!/usr/bin/env python
"""Step Zero — probe what the local mlx_vlm.server actually supports.

This dictates how the CRAG/HyDE nodes are written. Run BEFORE any RAG code:

    uv run python scripts/spike_endpoint.py

Checks, in order:
  1. Plain completion works at all.
  2. response_format={"type": "json_object"} — does the server honor it?
  3. tools=[...] (function calling) — does the server honor it?

The plan ALREADY assumes both (2) and (3) are unreliable and builds the grader
as a hand-parsed yes/no prompt. This script just makes the assumption evidence-based.
"""

import sys

from openai import OpenAI

from raglab.config import settings


def _client() -> OpenAI:
    return OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)


def check_basic(client: OpenAI) -> bool:
    print("1. Basic completion ... ", end="", flush=True)
    try:
        resp = client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": "Reply with exactly: pong"}],
            max_tokens=10,
            temperature=0,
        )
        text = (resp.choices[0].message.content or "").strip()
        print(f"OK -> {text!r}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"FAILED: {e}")
        return False


def check_json_mode(client: OpenAI) -> bool:
    print("2. response_format=json_object ... ", end="", flush=True)
    try:
        resp = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "user", "content": 'Return JSON: {"ok": true}. Only JSON.'}
            ],
            response_format={"type": "json_object"},
            max_tokens=30,
            temperature=0,
        )
        text = (resp.choices[0].message.content or "").strip()
        ok = text.startswith("{") and "ok" in text
        print(f"{'OK' if ok else 'IGNORED'} -> {text!r}")
        return ok
    except Exception as e:  # noqa: BLE001
        print(f"NOT SUPPORTED: {e}")
        return False


def check_tools(client: OpenAI) -> bool:
    print("3. tools / function-calling ... ", end="", flush=True)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "grade",
                "description": "Grade relevance",
                "parameters": {
                    "type": "object",
                    "properties": {"relevant": {"type": "boolean"}},
                    "required": ["relevant"],
                },
            },
        }
    ]
    try:
        resp = client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": "Is the sky blue? Call grade."}],
            tools=tools,
            tool_choice="auto",
            max_tokens=50,
            temperature=0,
        )
        msg = resp.choices[0].message
        called = bool(getattr(msg, "tool_calls", None))
        print(f"{'OK (tool_calls present)' if called else 'IGNORED (no tool_calls)'}")
        return called
    except Exception as e:  # noqa: BLE001
        print(f"NOT SUPPORTED: {e}")
        return False


def main() -> int:
    print(f"Endpoint: {settings.llm_base_url}  model: {settings.llm_model}\n")
    client = _client()
    if not check_basic(client):
        print("\nServer not reachable. Start it with:")
        print("  uv run python -m mlx_vlm.server "
              "--model mlx-community/gemma-4-12B-it-4bit --port 8085")
        return 1
    json_ok = check_json_mode(client)
    tools_ok = check_tools(client)
    print("\n--- Verdict ---")
    print(f"json_object supported : {json_ok}")
    print(f"function-calling      : {tools_ok}")
    if not tools_ok:
        print("=> Confirmed: build the CRAG grader as a hand-parsed yes/no prompt "
              "(no with_structured_output). This is the planned path.")
    else:
        print("=> Function-calling works, but the plan still uses the hand-parsed "
              "grader for portability/transparency.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
