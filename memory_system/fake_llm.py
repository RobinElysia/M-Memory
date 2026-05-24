"""Fake (deterministic) LLM adapter for testing.

Provides a controllable LLM backend that returns pre-scripted responses
or makes decisions based on simple heuristics.  Used in unit tests and
E2E scenario tests to ensure reproducibility.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from memory_system.interfaces import LLMAdapter

logger = logging.getLogger(__name__)


class FakeLLMAdapter(LLMAdapter):
    """Deterministic fake LLM for testing.

    Two modes:
    1. **Scripted**: Provide a list of ``(trigger_substring, response)`` pairs.
       The first matching trigger determines the response.
    2. **Heuristic**: When no script matches, returns a reasonable default
       (e.g. always "new" bucket, no conflicts).
    """

    def __init__(
        self,
        script: list[tuple[str, str]] | None = None,
        default_response: str = "{}",
    ) -> None:
        self._script: list[tuple[str, str]] = script or []
        self._default_response = default_response
        self.call_log: list[dict[str, Any]] = []

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Return a scripted or default response, logging the call."""
        response = self._default_response

        for trigger, resp in self._script:
            if trigger in prompt:
                response = resp
                break

        self.call_log.append(
            {
                "prompt_length": len(prompt),
                "prompt_preview": prompt[:200],
                "response": response,
            }
        )
        logger.info(
            "FakeLLM.complete prompt_len=%d response=%s",
            len(prompt),
            response[:100],
        )
        return response

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Delegates to :meth:`complete` using the last user message as prompt."""
        user_messages = [m["content"] for m in messages if m.get("role") == "user"]
        prompt = user_messages[-1] if user_messages else ""
        return self.complete(prompt, **kwargs)

    def add_script(self, trigger: str, response: str) -> None:
        """Append a script entry."""
        self._script.append((trigger, response))

    def clear_log(self) -> None:
        """Reset the call log."""
        self.call_log.clear()


def create_assignment_decision(
    primary_bucket: str,
    cross_links: list[dict[str, Any]] | None = None,
    reasoning: str = "scripted decision",
) -> str:
    """Helper to create a JSON assignment response string."""
    return json.dumps(
        {
            "primary_bucket": primary_bucket,
            "reasoning": reasoning,
            "cross_links": cross_links or [],
        }
    )


def create_conflict_response(
    conflicts: list[dict[str, Any]] | None = None,
) -> str:
    """Helper to create a JSON conflict detection response string."""
    return json.dumps({"conflicts": conflicts or []})
