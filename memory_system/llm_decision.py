"""LLM decision helpers — prompt builders and response parsers.

This module is responsible for constructing prompts sent to the LLM and
parsing structured JSON responses.  It does NOT call the LLM directly;
that is done through the :class:`LLMAdapter` interface.
"""

from __future__ import annotations

import json
import re
from typing import Any

from memory_system.models import Bucket, MemoryNode

# ═══════════════════════════════════════════════════════════════════════════════
# Prompt builders
# ═══════════════════════════════════════════════════════════════════════════════


def build_bucket_assignment_prompt(
    current_summary: str,
    candidates: list[tuple[Bucket, float]],
    nearby_count: int = 3,
) -> str:
    """Build the prompt for LLM-driven bucket assignment.

    Args:
        current_summary: The new node's summary text (A).
        candidates: Top-k candidate buckets with similarity scores.
        nearby_count: How many nearby node summaries to include per bucket.

    Returns:
        Prompt string ready for ``LLMAdapter.complete``.
    """
    if not candidates:
        return json.dumps(
            {
                "primary_bucket": "new",
                "reasoning": "No existing buckets — creating new bucket.",
                "cross_links": [],
            }
        )

    candidate_lines: list[str] = []
    for bucket, score in candidates:
        medoid_summary = bucket.medoid.summary if bucket.medoid else "(empty)"
        candidate_lines.append(
            f"- Bucket ID: {bucket.id}\n"
            f"  Medoid Summary: {medoid_summary}\n"
            f"  Similarity Score: {score:.4f}\n"
            f"  Node Count: {len(bucket.node_ids)}"
        )

    candidates_text = "\n".join(candidate_lines)

    return f"""You are a memory management assistant. Given a conversation summary and a list
of candidate buckets, decide which bucket the current node should be assigned to,
and whether to create cross-bucket links.

Current Node Summary: {current_summary}

Candidate Buckets:
{candidates_text}

Respond with a JSON object:
{{
  "primary_bucket": "<bucket_id or 'new'>",
  "reasoning": "<brief explanation>",
  "cross_links": [
    {{"bucket_id": "<bucket_id>", "weight": 0.0-1.0, "reason": "<why>"}}
  ]
}}

Rules:
- Choose the most semantically related bucket as primary_bucket.
- If no bucket fits well, use "new".
- For cross_links: only include buckets that are ALSO related but not the primary.
- Weight should reflect the degree of cross-relevance (0.0 = none, 1.0 = very high).
- Never create more than 3 cross_links."""


def build_conflict_detection_prompt(
    query: str,
    candidates: list[tuple[MemoryNode, float]],
) -> str:
    """Build the prompt for LLM-driven conflict detection.

    Args:
        query: The original search query.
        candidates: Top-N candidate nodes with their re-ranked scores.

    Returns:
        Prompt string ready for ``LLMAdapter.complete``.
    """
    candidate_texts: list[str] = []
    for i, (node, score) in enumerate(candidates):
        candidate_texts.append(
            f"[{i}] Content: {node.content}\n"
            f"    Timestamp: {node.timestamp}\n"
            f"    Confidence: {node.confidence:.2f}\n"
            f"    Score: {score:.4f}\n"
            f"    Already Stale: {node.is_stale}"
        )

    candidates_text = "\n".join(candidate_texts)

    return f"""You are a fact-checking assistant. Given a query and a list
of candidate memory nodes, identify any factual contradictions.

Query: {query}

Candidates:
{candidates_text}

Respond with a JSON object:
{{
  "conflicts": [
    {{"newer_id": "<node index>", "older_id": "<node index>", "reason": "<why they conflict>"}}
  ]
}}

Rules:
- A conflict exists when two nodes contain contradictory key facts.
- The NEWER node (higher timestamp) should be kept; the OLDER should be marked stale.
- If no contradictions exist, return an empty "conflicts" list.
- Only flag genuine factual contradictions, not normal topic evolution."""


# ═══════════════════════════════════════════════════════════════════════════════
# Response parsers
# ═══════════════════════════════════════════════════════════════════════════════


def parse_bucket_assignment_response(
    response: str,
) -> dict[str, Any]:
    """Parse the LLM's JSON response for bucket assignment.

    Args:
        response: Raw text response from the LLM.

    Returns:
        Dict with keys: ``primary_bucket`` (str), ``reasoning`` (str),
        ``cross_links`` (list of dict).

    Raises:
        ValueError: If the response cannot be parsed as valid JSON or is
            missing required fields.
    """
    # Try to extract JSON block from the response
    json_str = _extract_json(response)

    try:
        result: dict[str, Any] = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse LLM response as JSON: {response[:200]}") from exc

    if "primary_bucket" not in result:
        raise ValueError(
            f"LLM response missing 'primary_bucket' key: {response[:200]}"
        )

    result.setdefault("reasoning", "")
    result.setdefault("cross_links", [])

    # Validate cross_links
    for link in result["cross_links"]:
        if "bucket_id" not in link:
            raise ValueError(f"cross_link missing 'bucket_id': {link}")
        link.setdefault("weight", 0.5)
        link.setdefault("reason", "")

    return result


def parse_conflict_detection_response(
    response: str,
) -> list[dict[str, Any]]:
    """Parse the LLM's JSON response for conflict detection.

    Args:
        response: Raw text response from the LLM.

    Returns:
        List of conflict dicts, each with ``newer_id``, ``older_id``, ``reason``.

    Raises:
        ValueError: If the response cannot be parsed.
    """
    json_str = _extract_json(response)

    try:
        result: dict[str, Any] = json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse conflict detection response: {response[:200]}") from exc

    conflicts: list[dict[str, Any]] = result.get("conflicts", [])
    for c in conflicts:
        if "newer_id" not in c or "older_id" not in c:
            raise ValueError(f"Conflict entry missing required keys: {c}")
        c.setdefault("reason", "")
    return conflicts


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _extract_json(text: str) -> str:
    """Extract the first JSON object from text that may contain markdown fences."""
    # Try code-fence extraction first
    fence_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    match = re.search(fence_pattern, text)
    if match:
        return match.group(1).strip()

    # Try to find the outermost { ... }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return text.strip()
