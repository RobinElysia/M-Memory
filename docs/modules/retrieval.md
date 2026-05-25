# 检索引擎 · retrieval.py

> **源码位置**: [`memory_system/retrieval.py`](https://github.com/RobinElysia/M-Memory/blob/main/memory_system/retrieval.py)

顶层 API，组合 VectorStore、GraphStore、BucketManager、LLMAdapter 四个组件，
提供 `ingest()`、`search()`、`resolve_conflicts()` 三个核心方法。

## 类结构

```python
class MemoryRetrievalEngineImpl(MemoryRetrievalEngine):
    def __init__(self, config, vector_store, graph_store, llm):
        ...
        self._bucket_manager = BucketManagerImpl(...)  # 引擎拥有桶管理器
        self._nodes: dict[str, MemoryNode] = {}         # 节点注册表
```

**设计要点**：BucketManager 由引擎内部创建（组合而非注入），
因为桶的创建/分配逻辑与检索强耦合，不应由外部控制。

---

## ingest() — 记忆摄入管线

```python
def ingest(self, summary: str, content: str, confidence: float = 1.0) -> str:
```

### 执行流程

**Step 1 — 向量化**：分别对 A（摘要）和 C（详情）调用 `vector_store.embed()`。
两个向量用途不同：
- `summary_vector` → 存入桶 Medoid 的向量索引，参与第一层粗筛
- `content_vector` → 存入独立的 content 索引，参与第二层精筛 + 冲突消解

**Step 2 — 候选桶查找**：调用 `bucket_manager.find_candidates(node)`，
计算新节点 A 向量与所有活跃桶 Medoid 向量的余弦相似度，保留 top_k 个。

**Step 3 — LLM 决策**：调用 `build_bucket_assignment_prompt()` 构造 prompt，
包含当前节点摘要 + 候选桶 Medoid 摘要 + 桶内最近节点的摘要作为上下文。
LLM 返回 JSON 格式的决策：

```json
{
  "primary_bucket": "<bucket_id 或 'new'>",
  "reasoning": "<决策理由>",
  "cross_links": [
    {"bucket_id": "...", "weight": 0.8, "reason": "..."}
  ]
}
```

**容错处理**：如果 LLM 返回的 JSON 解析失败，回退到创建新桶：
```python
try:
    decision = parse_bucket_assignment_response(response)
except ValueError:
    logger.warning("Failed to parse LLM assignment, creating new bucket")
    decision = {"primary_bucket": "new", ...}
```

**Step 4 — 物理分配**：
- 若 `primary_bucket == "new"` 或指定的桶不存在 → `bucket_manager.create_bucket(node)`
- 否则 → `bucket_manager.assign_to_bucket(node, bucket, cross_links)`

**Step 5 — 内容向量索引**：将 `content_vector` 以 `content:{node.id}` 为 key
存入向量存储，供后续搜索使用。

---

## search() — 双层检索管线

```python
def search(self, query: str, max_hops=None, weight_threshold=None) -> SearchResult:
```

### Layer 1: 桶级粗筛

```python
# 获取所有活跃（非休眠）桶
active_buckets = self._bucket_manager.get_active_buckets()

# 计算查询向量与每个桶 Medoid 的余弦相似度
for bucket in active_buckets:
    sim = self._cosine_sim(query_vector, bucket.medoid.vector)
    bucket_scores.append((bucket, sim))

# 保留 top_m 个桶
top_buckets = bucket_scores[:self._config.bucket.top_m]
```

**为什么用 Medoid 而不是遍历所有节点？**
因为桶的数量远小于节点数量（通常 B << N），
O(B·D) 的粗筛远快于 O(N·D) 的全量搜索。

### Layer 1: 桶内精筛 + 休眠唤醒

```python
# 对 top_m 桶内的每个节点计算相似度
for bucket, _ in top_buckets:
    for node_id in bucket.node_ids:
        node = self._nodes.get(node_id)
        sim = self._cosine_sim(query_vector, node.summary_vector)
        seed_nodes.append((node, sim))

# 截断到 top_p * m
seed_nodes = seed_nodes[:self._config.bucket.top_p * len(top_buckets)]
```

**休眠桶唤醒机制**（v0.1.0 新增）：
搜索前还检查所有休眠桶的 Medoid 向量，如果与查询的相似度超过阈值（0.5），
自动调用 `wake_bucket()` 将其恢复为活跃状态：

```python
for db in dormant_buckets:
    sim = self._cosine_sim(query_vector, db.medoid.vector)
    if sim > 0.5:
        self._bucket_manager.wake_bucket(db.id)
        active_buckets.append(db)
```

### Layer 1: 图联想扩展

```python
seed_ids = [n.id for n, _ in seed_nodes]
paths = self._graph_store.traverse(seed_ids, hops, threshold)
```

从种子节点出发，沿跨桶边 BFS 游走，发现关联桶中的相关节点。
**去重策略**：同一节点经多条路径可达时，只保留置信度最高的单一路径（不累加），
避免环路导致分数膨胀。

### Layer 2: 冲突消解

```python
resolved = self.resolve_conflicts(candidates, query)
```

详见 [冲突消解详解](../architecture/conflict-resolution.md)。

---

## resolve_conflicts() — 冲突消解管线

```python
def resolve_conflicts(self, candidates: list[MemoryNode], query: str) -> list[MemoryNode]:
```

### 加权重排序

使用三元组权重公式：

```
score = α · sim(C, query) + β / (1 + ΔT_days) + γ · confidence
```

| 参数 | 含义 | 默认值 | 作用 |
|------|------|--------|------|
| α | 语义相似度权重 | 0.5 | 内容与查询的相关性 |
| β | 时间新近度权重 | 0.3 | 新信息优先级 |
| γ | 来源置信度权重 | 0.2 | 信息来源的可靠性 |

`ΔT_days` 是节点创建到现在的天数。分母 `1 + ΔT_days` 确保新节点的时间分数趋近于 1，
旧节点趋近于 0（但永不为 0）。

### LLM 矛盾检测

将 top_n 个候选节点的内容、时间戳、置信度组装成 prompt，
让 LLM 判断是否存在事实矛盾：

```json
{
  "conflicts": [
    {"newer_id": "0", "older_id": "1", "reason": "地点已变更"}
  ]
}
```

### 过时标记

对 LLM 指出的旧节点：
```python
older_node.is_stale = True
older_node.confidence *= 0.1  # stale_mark_downgrade_factor
```

**关键设计**：旧信息**仅降权、不删除**。降权后它仍出现在检索结果中，
但排在末尾。用户仍可通过深层回溯找到它。这避免了信息丢失的风险。
