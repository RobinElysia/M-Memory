# 记忆系统实现架构 (Implementation Architecture)

> 本文档是 `ARCHITECTURE_DESIGN.md` 的实现视角转化，包含结构化接口定义、
> 数据模型、关键算法伪代码与完整参数表。所有实现必须严格参照本文档。

## 1. 核心数据模型

### 1.1 MemoryNode — 记忆节点

```python
@dataclass
class MemoryNode:
    id: str                          # 唯一标识 (UUID)
    summary: str                     # A — 总结性段落/关键词（用于粗筛）
    content: str                     # C — 待存详细内容（用于精确检索）
    summary_vector: np.ndarray       # A 的向量表示
    content_vector: np.ndarray       # C 的向量表示
    timestamp: float                 # 创建时间 (Unix timestamp)
    confidence: float                # 来源置信度 [0.0, 1.0]
    bucket_id: str                   # 所属主桶 ID（物理单桶存放）
    is_stale: bool = False           # 是否被标记为过时
```

**约束**：每个节点物理上只存在于一个主桶 (`bucket_id`)，跨桶关联仅通过边实现。

### 1.2 Bucket — 动态桶

```python
@dataclass
class Bucket:
    id: str                          # 唯一标识
    medoid: Optional[Medoid]         # 当前 Medoid（代表节点）
    node_ids: list[str]              # 桶内节点 ID 列表
    created_at: float                # 创建时间
    last_write_at: float             # 最后写入时间
    last_query_at: float             # 最后被查询时间
    is_dormant: bool = False         # 是否休眠
    version: int = 0                 # 版本号，Medoid 漂移时递增
```

### 1.3 Medoid — 桶代表

```python
@dataclass
class Medoid:
    node_id: str                     # 作为 Medoid 的实际节点 ID
    summary: str                     # 该节点的摘要文本 (A)
    vector: np.ndarray               # 该节点的摘要向量 (A 向量)
    version: int = 0                 # 与 Bucket.version 同步
```

**Medoid 计算规则**：选择桶内到所有其他节点的平均余弦距离最小的实际节点。
- 单节点桶：该节点自身即为 Medoid。
- 多节点桶：计算每个节点到其他所有节点的平均距离，取最小者。
- 确定性要求：禁止随机选择，距离相同时按 node_id 字典序取第一个。

### 1.4 Edge — 图边

```python
@dataclass
class Edge:
    id: str                          # 唯一标识
    source_id: str                   # 源节点 ID
    target_id: str                   # 目标节点 ID
    edge_type: EdgeType              # 边类型
    weight: float                    # 权重 [0.0, 1.0]
    created_at: float                # 创建时间

class EdgeType(Enum):
    TEMPORAL = "temporal"            # 时间边 — 按对话顺序连接
    INTRA_BUCKET = "intra_bucket"    # 桶内边 — 节点与其主桶 Medoid
    CROSS_BUCKET = "cross_bucket"    # 跨桶边 — LLM 判定关联的软连接
```

**跨桶边淘汰规则**：
- 每个节点最多 `max_out_degree` 条跨桶出边。
- 当出度已满且有更高权重的新连接时，淘汰权重最低的边。
- 权重相同的边，淘汰最早创建的（FIFO）。

## 2. 核心算法伪代码

### 2.1 分桶决策流程 (Bucket Assignment)

```
function assign_bucket(node_a: MemoryNode) -> Bucket:
    # Step 1: 粗筛候选桶
    candidates = []
    for bucket in active_buckets:
        sim = cosine_similarity(node_a.summary_vector, bucket.medoid.vector)
        candidates.append((bucket, sim))
    candidates.sort(by=sim, descending=True)
    candidates = candidates[:config.bucket.top_k]

    # Step 2: 构建 LLM 上下文
    context = {
        "current": node_a.summary,
        "candidates": [
            {
                "medoid_summary": bucket.medoid.summary,
                "nearby_summaries": get_top_nearby_summaries(bucket, n=3)
            }
            for bucket, _ in candidates
        ]
    }

    # Step 3: LLM 决策
    decision = llm.decide_assignment(context)
    # LLM 返回格式: {"primary_bucket": bucket_id, "cross_links": [
    #   {"bucket_id": ..., "weight": float}, ...]}

    # Step 4: 执行分配
    primary = find_bucket(decision.primary_bucket)
    if primary is None:
        primary = create_new_bucket(medoid_from(node_a))
    primary.add_node(node_a)

    # Step 5: 建立跨桶边 (软连接，不复制节点)
    for link in decision.cross_links:
        if link.bucket_id != primary.id:
            add_cross_bucket_edge(
                from_node=node_a,
                to_bucket_medoid=link.bucket_id,
                weight=link.weight
            )

    # Step 6: 更新 Medoid (仅主桶)
    update_medoid(primary)

    return primary
```

**LLM 决策 Prompt 模板**：
```
你是一个记忆管理助手。给定当前对话摘要和若干候选桶的 Medoid 摘要，
请决定将当前节点分配到哪个桶，以及是否建立跨桶关联。

当前节点摘要: {current_summary}

候选桶:
{candidate_list}

请返回 JSON:
{
  "primary_bucket": "<bucket_id 或 'new'>",
  "reasoning": "<简短理由>",
  "cross_links": [
    {"bucket_id": "...", "weight": 0.0-1.0, "reason": "..."}
  ]
}
```

### 2.2 图游走检索流程 (Graph Traversal Retrieval)

```
function search(query: str, max_hops: int, weight_threshold: float) -> SearchResult:
    # Step 1: 向量化查询
    query_vector = embed(query)

    # Step 2: 第一层 — 桶级粗筛
    bucket_scores = []
    for bucket in active_buckets:
        sim = cosine_similarity(query_vector, bucket.medoid.vector)
        bucket_scores.append((bucket, sim))
    top_buckets = bucket_scores.sort(descending)[:config.bucket.top_m]

    # Step 3: 第二层 — 桶内精筛
    seed_nodes = []
    for bucket, _ in top_buckets:
        for node_id in bucket.node_ids:
            node = get_node(node_id)
            sim = cosine_similarity(query_vector, node.summary_vector)
            seed_nodes.append((node, sim))
    seed_nodes.sort(by=sim, descending=True)
    seed_nodes = seed_nodes[:config.bucket.top_p * len(top_buckets)]

    # Step 4: 图联想扩展 — BFS/DFS 沿跨桶边游走
    visited = set()
    candidates = {}  # node_id -> best_path_score
    queue = deque()

    for node, score in seed_nodes:
        visited.add(node.id)
        candidates[node.id] = score  # 初始分数 = 语义相似度
        queue.append((node.id, score, 0))  # (node_id, path_score, depth)

    while queue:
        current_id, path_score, depth = queue.popleft()
        if depth >= max_hops:
            continue

        for edge in get_outgoing_cross_edges(current_id):
            if edge.weight < weight_threshold:
                continue
            neighbor_id = edge.target_id
            new_score = path_score * edge.weight  # 累积路径权重

            # 去重：只保留最优单路径 (取最高分数)
            if neighbor_id not in candidates or new_score > candidates[neighbor_id]:
                candidates[neighbor_id] = new_score
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, new_score, depth + 1))

    # Step 5: 排序、截断、返回
    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return SearchResult(
        nodes=[get_node(nid) for nid, score in sorted_candidates],
        scores=[score for nid, score in sorted_candidates]
    )
```

**去重策略**：同一节点经多条路径可达时，取置信度最高的单一路径分数，**不累加**，
避免环路导致的分数膨胀。

### 2.3 冲突消解流程 (Conflict Resolution)

```
function resolve_conflicts(candidates: list[MemoryNode], query: str) -> list[MemoryNode]:
    # Step 1: 加权重排序
    now = current_timestamp()
    scored = []
    for node in candidates:
        sim = cosine_similarity(node.content_vector, embed(query))
        time_factor = 1.0 / (1.0 + (now - node.timestamp) / 86400.0)  # 天为单位
        confidence = node.confidence
        score = (
            config.conflict.alpha * sim +
            config.conflict.beta * time_factor +
            config.conflict.gamma * confidence
        )
        scored.append((node, score))
    scored.sort(by=score, descending=True)
    top_candidates = scored[:config.conflict.top_n]

    # Step 2: LLM 冲突检查
    context = {
        "query": query,
        "candidates": [
            {
                "content": node.content,
                "timestamp": node.timestamp,
                "confidence": node.confidence,
                "score": score
            }
            for node, score in top_candidates
        ]
    }
    conflict_result = llm.detect_conflicts(context)
    # LLM 返回: {"conflicts": [{"newer_id": ..., "older_id": ..., "reason": "..."}]}

    # Step 3: 标记过时
    for conflict in conflict_result.conflicts:
        older_node = get_node(conflict.older_id)
        older_node.is_stale = True
        older_node.confidence *= config.conflict.stale_mark_downgrade_factor

    # Step 4: 返回重排序后的结果 (含降权)
    return [node for node, _ in top_candidates]
```

### 2.4 后台清理流程 (Periodic Cleanup)

```
async function cleanup_cycle():
    for bucket in all_buckets:
        # 1. 休眠检查
        if (now - bucket.last_write_at > dormancy_interval
            and now - bucket.last_query_at > dormancy_interval):
            bucket.is_dormant = True
            remove_from_active_index(bucket.medoid.vector)
            continue

        # 2. 桶内矛盾检测
        nodes = get_bucket_nodes(bucket)
        for i, node_a in enumerate(nodes):
            for node_b in nodes[i+1:]:
                sim = cosine_similarity(node_a.content_vector, node_b.content_vector)
                if sim > config.cleanup.node_similarity_for_contradiction:
                    # 检查关键事实矛盾 (LLM)
                    if llm.has_contradiction(node_a.content, node_b.content):
                        newer = max(node_a, node_b, key=lambda n: n.timestamp)
                        older = min(node_a, node_b, key=lambda n: n.timestamp)
                        older.is_stale = True
                        older.confidence *= config.conflict.stale_mark_downgrade_factor

        # 3. 话题漂移检查
        for node_id in bucket.node_ids:
            node = get_node(node_id)
            sim = cosine_similarity(node.content_vector, bucket.medoid.vector)
            if sim < config.cleanup.medoid_similarity_floor:
                mark_for_cold_storage(node)

        # 4. 桶分裂检查
        if len(bucket.node_ids) >= config.bucket.split_threshold:
            sub_clusters = detect_sub_clusters(bucket)
            if len(sub_clusters) > 1:
                split_bucket(bucket, sub_clusters)
```

## 3. 抽象接口定义

### 3.1 VectorStore

```python
class VectorStore(ABC):
    @abstractmethod
    def embed(self, text: str) -> np.ndarray: ...
    @abstractmethod
    def add(self, vectors: np.ndarray, metadata: list[dict]) -> list[str]: ...
    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[str, float]]: ...
    @abstractmethod
    def remove(self, ids: list[str]) -> None: ...
    @abstractmethod
    def count(self) -> int: ...
```

### 3.2 LLMAdapter

```python
class LLMAdapter(ABC):
    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str: ...
    @abstractmethod
    def chat(self, messages: list[dict[str, str]], **kwargs) -> str: ...
```

### 3.3 GraphStore

```python
class GraphStore(ABC):
    @abstractmethod
    def add_node(self, node_id: str, attributes: dict) -> None: ...
    @abstractmethod
    def add_edge(self, from_id: str, to_id: str, edge_type: EdgeType, weight: float) -> str: ...
    @abstractmethod
    def traverse(self, start_nodes: list[str], max_hops: int, weight_threshold: float) -> list[TraversalPath]: ...
    @abstractmethod
    def get_out_edges(self, node_id: str, edge_type: Optional[EdgeType] = None) -> list[Edge]: ...
    @abstractmethod
    def remove_edge(self, edge_id: str) -> None: ...
```

### 3.4 BucketManager

```python
class BucketManager(ABC):
    @abstractmethod
    def find_candidates(self, node_a: MemoryNode) -> list[tuple[Bucket, float]]: ...
    @abstractmethod
    def assign_to_bucket(self, node_a: MemoryNode, bucket: Bucket, cross_links: list[dict]) -> None: ...
    @abstractmethod
    def split_bucket(self, bucket: Bucket) -> list[Bucket]: ...
    @abstractmethod
    def dormancy_check(self) -> list[Bucket]: ...
    @abstractmethod
    def wake_bucket(self, bucket_id: str) -> Bucket: ...
```

### 3.5 MemoryRetrievalEngine

```python
class MemoryRetrievalEngine(ABC):
    @abstractmethod
    def search(self, query: str, max_hops: Optional[int] = None, 
               weight_threshold: Optional[float] = None) -> SearchResult: ...
    @abstractmethod
    def resolve_conflicts(self, candidates: list[MemoryNode], query: str) -> list[MemoryNode]: ...
```

## 4. 完整参数表

### 4.1 桶管理参数 (BucketConfig)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `top_k` | `int` | `3` | 分配时考虑的候选桶数 |
| `top_m` | `int` | `5` | 语义搜索时检索的桶数 |
| `top_p` | `int` | `10` | 每桶内检索的节点数 |
| `split_threshold` | `int` | `50` | 触发桶分裂检查的节点数下限 |
| `dormancy_interval_seconds` | `float` | `3600.0` | 无活动后判定休眠的时间窗口 |
| `cold_storage_similarity_threshold` | `float` | `0.3` | 节点与 Medoid 相似度低于此值则移入冷存储 |

### 4.2 图参数 (GraphConfig)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_out_degree` | `int` | `5` | 节点最大跨桶出度数 |
| `weight_decay_threshold` | `float` | `0.2` | 边权重低于此值则剪枝 |
| `max_hops` | `int` | `2` | 图游走最大跳数 |
| `edge_weight_threshold` | `float` | `0.5` | 游走时的边权重门槛 |

### 4.3 冲突消解参数 (ConflictConfig)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `alpha` | `float` | `0.5` | 语义相似度权重 |
| `beta` | `float` | `0.3` | 时间新近度权重 |
| `gamma` | `float` | `0.2` | 来源置信度权重 |
| `top_n` | `int` | `5` | 冲突检查前的截断数 |
| `stale_mark_downgrade_factor` | `float` | `0.1` | 过时标记的降权因子 |

### 4.4 清理参数 (CleanupConfig)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `interval_seconds` | `float` | `300.0` | 清理周期（秒） |
| `node_similarity_for_contradiction` | `float` | `0.85` | 矛盾检测的相似度阈值 |
| `medoid_similarity_floor` | `float` | `0.3` | 话题漂移的相似度下限 |

### 4.5 全局参数 (MemorySystemConfig)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `embedding_dim` | `int` | `1536` | 向量维度 |
| `log_level` | `str` | `"INFO"` | 日志级别 |

## 5. 边界条件处理

| 场景 | 处理方式 |
|------|----------|
| 桶只有一个节点时 Medoid 如何计算？ | 该节点自身即为 Medoid |
| 所有桶的相似度都低于阈值时？ | LLM 决定创建新桶或归入最相似桶 |
| 查询时活跃桶为空？ | 返回空结果，不报错 |
| 跨桶边出度已满且有更高权重边？ | 末位淘汰：移除权重最低/最早的边 |
| 节点经多条路径可达？ | 取最优单路径分数，不累加 |
| LLM 调用失败？ | 回退到纯向量相似度决策，记录 WARNING 日志 |
| 桶分裂后原 Medoid 归属？ | 按子簇重新计算各子桶 Medoid |
| 休眠桶被查询命中？ | 唤醒并恢复 Medoid 到活跃索引 |
