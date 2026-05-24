# 合同文档 (Contracts)

> 本文档整理所有抽象接口的方法签名、参数、返回值与副作用说明，
> 供人工审核与后续实现参考。内容从 `memory_system/interfaces.py` 的
> docstring 自动提取。

## VectorStore

向量嵌入与相似度搜索后端抽象。

| 方法 | 签名 | 说明 |
|------|------|------|
| `embed` | `(text: str) -> np.ndarray` | 将文本转为稠密向量 (shape `(dim,)`, dtype `float32`)。 |
| `add` | `(vectors: np.ndarray, metadata: list[dict]) -> list[str]` | 批量插入向量及元数据，返回 ID 列表。 |
| `search` | `(query_vector: np.ndarray, top_k: int) -> list[tuple[str, float]]` | 返回 top_k 最相似的 (id, score) 列表。 |
| `remove` | `(ids: list[str]) -> None` | 按 ID 删除向量。 |
| `count` | `() -> int` | 返回索引中向量总数。 |

## LLMAdapter

大语言模型后端抽象。

| 方法 | 签名 | 说明 |
|------|------|------|
| `complete` | `(prompt: str, **kwargs) -> str` | 单轮补全请求。 |
| `chat` | `(messages: list[dict[str, str]], **kwargs) -> str` | 多轮对话补全请求。 |

## GraphStore

图存储与遍历抽象。

| 方法 | 签名 | 说明 |
|------|------|------|
| `add_node` | `(node_id: str, attributes: dict) -> None` | 注册顶点。 |
| `add_edge` | `(from_id, to_id, edge_type, weight) -> str` | 创建有向边，返回边 ID。 |
| `traverse` | `(start_nodes, max_hops, weight_threshold) -> list[TraversalPath]` | 沿跨桶边游走，最优单路径去重。 |
| `get_out_edges` | `(node_id, edge_type?) -> list[Edge]` | 获取出边，可按类型过滤。 |
| `remove_edge` | `(edge_id: str) -> None` | 按 ID 删除边。 |
| `get_node_count` | `() -> int` | 顶点总数。 |
| `get_edge_count` | `(edge_type?) -> int` | 边总数，可按类型过滤。 |

## BucketManager

桶生命周期与节点分配抽象。

| 方法 | 签名 | 说明 |
|------|------|------|
| `find_candidates` | `(node_a: MemoryNode) -> list[tuple[Bucket, float]]` | 通过向量相似度找 top_k 候选桶。 |
| `assign_to_bucket` | `(node_a, bucket, cross_links) -> None` | 物理归入主桶并建立跨桶边，更新 Medoid。 |
| `create_bucket` | `(medoid_node: MemoryNode) -> Bucket` | 以给定节点为 Medoid 创建新桶。 |
| `split_bucket` | `(bucket: Bucket) -> list[Bucket]` | 基于子簇分裂桶，返回新桶列表。 |
| `dormancy_check` | `() -> list[Bucket]` | 标识并返回新标记为休眠的桶。 |
| `wake_bucket` | `(bucket_id: str) -> Bucket` | 唤醒休眠桶，恢复 Medoid 到活跃索引。 |
| `get_active_buckets` | `() -> list[Bucket]` | 获取所有非休眠桶。 |
| `get_all_buckets` | `() -> list[Bucket]` | 获取所有桶（含休眠）。 |

## MemoryRetrievalEngine

顶层检索与冲突消解 API。

| 方法 | 签名 | 说明 |
|------|------|------|
| `search` | `(query, max_hops?, weight_threshold?) -> SearchResult` | 完整双层检索管线（粗筛→精筛→图扩展→冲突消解）。 |
| `resolve_conflicts` | `(candidates, query) -> list[MemoryNode]` | 加权重排序 + LLM 矛盾检测 + 过时标记。 |
| `ingest` | `(summary, content, confidence?) -> str` | 完整摄入管线：嵌入→分桶→存储→返回节点 ID。 |

## 数据模型 (models.py)

| 类 | 关键字段 |
|----|----------|
| `MemoryNode` | `id`, `summary` (A), `content` (C), `summary_vector`, `content_vector`, `timestamp`, `confidence`, `bucket_id`, `is_stale` |
| `Bucket` | `id`, `medoid`, `node_ids`, `created_at`, `last_write_at`, `last_query_at`, `is_dormant`, `version` |
| `Medoid` | `node_id`, `summary`, `vector`, `version` |
| `Edge` | `id`, `source_id`, `target_id`, `edge_type`, `weight`, `created_at` |
| `EdgeType` | `TEMPORAL`, `INTRA_BUCKET`, `CROSS_BUCKET` |
| `TraversalPath` | `node_ids`, `total_weight`, `hops` |
| `SearchResult` | `nodes`, `scores` |
