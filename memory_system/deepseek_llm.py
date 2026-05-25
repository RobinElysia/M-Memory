"""DeepSeek LLM adapter — production-grade LLM backend.

Uses the OpenAI-compatible DeepSeek API (deepseek-chat model).
Requires ``openai>=1.0`` (already listed in project dependencies).
"""

from __future__ import annotations

import logging
from typing import Any

from openai import OpenAI

from memory_system.interfaces import LLMAdapter

logger = logging.getLogger(__name__)


class DeepSeekAdapter(LLMAdapter):
    """OpenAI-compatible adapter for DeepSeek API.

    Args:
        api_key: DeepSeek API key (defaults to DEEPSEEK_API_KEY env var).
        model: Model name (default ``deepseek-chat``).
        base_url: API base URL.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com/v1",
    ) -> None:
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self.call_count: int = 0
        self.total_tokens: int = 0

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Send a single-shot completion via DeepSeek chat API."""
        self.call_count += 1
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", 0.0),
                max_tokens=kwargs.get("max_tokens", 1024),
            )
        except Exception as exc:
            logger.error("DeepSeek API call failed: %s", exc)
            raise RuntimeError(f"DeepSeek API error: {exc}") from exc

        usage = response.usage
        if usage:
            self.total_tokens += usage.total_tokens

        content = response.choices[0].message.content or ""
        logger.info(
            "DeepSeek complete call=%d tokens=%d len=%d preview=%s",
            self.call_count,
            usage.total_tokens if usage else 0,
            len(content),
            content[:120],
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
            logger.error("DeepSeek chat call failed: %s", exc)
            raise RuntimeError(f"DeepSeek API error: {exc}") from exc

        usage = response.usage
        if usage:
            self.total_tokens += usage.total_tokens

        content = response.choices[0].message.content or ""
        logger.info(
            "DeepSeek chat call=%d tokens=%d preview=%s",
            self.call_count,
            usage.total_tokens if usage else 0,
            content[:120],
        )
        return content
