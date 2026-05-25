# LLM 决策层 · llm_decision.py

> **源码位置**: [`memory_system/llm_decision.py`](https://github.com/RobinElysia/M-Memory/blob/main/memory_system/llm_decision.py)

构建 LLM prompt 并解析结构化 JSON 响应。本模块不调用 LLM——那是 `LLMAdapter` 的职责。

## 两个核心 Prompt

### 分桶决策 Prompt

`build_bucket_assignment_prompt(current_summary, candidates)` 构建如下 prompt：

```
You are a memory management assistant. Given a conversation summary
and a list of candidate buckets, decide which bucket...

Current Node Summary: {summary}

Candidate Buckets:
- Bucket ID: abc123
  Medoid Summary: Python编程技巧
  Similarity Score: 0.8234
  Node Count: 15
...

Respond with a JSON object:
{
  "primary_bucket": "<bucket_id or 'new'>",
  "reasoning": "<brief explanation>",
  "cross_links": [
    {"bucket_id": "...", "weight": 0.0-1.0, "reason": "..."}
  ]
}
```

### 矛盾检测 Prompt

`build_conflict_detection_prompt(query, candidates)` 列出所有候选节点的
内容、时间戳、置信度，要求 LLM 找出事实矛盾。

## JSON 提取

LLM 响应可能包裹在 Markdown 代码块中，`_extract_json()` 做两件事：
1. 尝试匹配 ` ```json ... ``` ` 代码块
2. 回退到查找最外层 `{ ... }`

```python
def _extract_json(text: str) -> str:
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        return fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 else text.strip()
```
