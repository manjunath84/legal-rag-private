# src/raglab/llm.py
"""Thin wrapper around the local OpenAI-compatible endpoint.

Reuses Week 1's pattern: the `openai` SDK pointed at any `base_url`. Here it
points at the local mlx_vlm.server, so nothing leaves the machine at inference.
"""

from openai import OpenAI

from raglab.config import Settings, settings


def get_client(cfg: Settings = settings) -> OpenAI:
    return OpenAI(api_key=cfg.llm_api_key, base_url=cfg.llm_base_url)


def chat(
    messages: list[dict],
    cfg: Settings = settings,
    client: OpenAI | None = None,
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Send chat messages to the local model and return the text reply."""
    client = client or get_client(cfg)
    resp = client.chat.completions.create(
        model=cfg.llm_model,
        messages=messages,
        temperature=cfg.llm_temperature if temperature is None else temperature,
        max_tokens=cfg.llm_max_tokens if max_tokens is None else max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()
