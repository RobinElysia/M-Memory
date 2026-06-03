"""DeepSeek LLM adapter — production-grade LLM backend.

Uses the OpenAI-compatible DeepSeek API.
Requires ``openai>=1.0`` and ``DEEPSEEK_API_KEY`` environment variable.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from openai import OpenAI

from memory_system.interfaces import LLMAdapter

logger = logging.getLogger(__name__)


class DeepSeekAdapter(LLMAdapter):
    """OpenAI-compatible adapter for DeepSeek API.

    Reads API key from ``DEEPSEEK_API_KEY`` environment variable.
    Default model is ``deepseek-chat``.

    Args:
        api_key: Override for env var (mainly for testing).
        model: Model name.
        base_url: API base URL.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/v1",
    ) -> None:
        key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not key:
            raise ValueError(
                "DeepSeek API key required. Set DEEPSEEK_API_KEY env var "
                "or pass api_key=..."
            )
        self._client = OpenAI(api_key=key, base_url=base_url)
        self._model = model
        self.call_count: int = 0
        self.total_tokens: int = 0

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Single-shot completion via DeepSeek chat API."""
        self.call_count += 1
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", 0.0),
                max_tokens=kwargs.get("max_tokens", 1024),
            )
        except Exception as exc:
            logger.error("DeepSeek API error: %s", exc)
            raise RuntimeError(f"DeepSeek API error: {exc}") from exc

        usage = response.usage
        if usage:
            self.total_tokens += usage.total_tokens

        content = response.choices[0].message.content or ""
        logger.info(
            "DeepSeek call #%d tokens=%d len=%d preview=%.120s",
            self.call_count,
            usage.total_tokens if usage else 0,
            len(content),
            content,
        )
        return content

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Multi-turn chat completion."""
        self.call_count += 1
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=kwargs.get("temperature", 0.0),
                max_tokens=kwargs.get("max_tokens", 1024),
            )
        except Exception as exc:
            logger.error("DeepSeek chat error: %s", exc)
            raise RuntimeError(f"DeepSeek API error: {exc}") from exc

        usage = response.usage
        if usage:
            self.total_tokens += usage.total_tokens

        content = response.choices[0].message.content or ""
        logger.info(
            "DeepSeek chat #%d tokens=%d preview=%.120s",
            self.call_count,
            usage.total_tokens if usage else 0,
            content,
        )
        return content
