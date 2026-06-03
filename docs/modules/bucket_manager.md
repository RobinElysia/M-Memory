# 桶管理器 · bucket_manager.py

> **源码位置**: [`memory_system/bucket_manager.py`](https://github.com/RobinElysia/M-Memory/blob/main/memory_system/bucket_manager.py)

管理桶的完整生命周期：创建、分配、分裂、休眠、唤醒。

## 核心数据结构

```python
class BucketManagerImpl(BucketManager):
    def __init__(self, config, vector_store, graph_store, llm):
        self._buckets: dict[str, Bucket] = {}   # 所有桶
        self._nodes: dict[str, MemoryNode] = {}  # 所有节点
```

桶管理器的向量索引存储 Medoid 向量（用于粗筛），graph_store 存储桶间边关系。

## Medoid 计算 — 确定性算法

Medoid 是桶内"最中心"的节点：到所有其他节点的平均余弦距离最小的实际节点。

```python
def _update_medoid(self, bucket: Bucket) -> None:
    best_node_id = bucket.node_ids[0]
    best_avg_dist = float("inf")

    for nid in bucket.node_ids:
        total_dist = 0.0
        for other_nid in bucket.node_ids:
            if other_nid == nid:
                continue
            sim = self._cosine_sim(node.summary_vector, other.summary_vector)
            total_dist += 1.0 - sim  # 余弦距离 = 1 - 相似度

        avg_dist = total_dist / count
        if avg_dist < best_avg_dist or (
            abs(avg_dist - best_avg_dist) < 1e-9 and nid < best_node_id
        ):
            best_avg_dist = avg_dist
            best_node_id = nid
```

**平局处理**：当两个节点的平均距离完全相同时（差值 < 1e-9），
按 node_id 字典序选择。这保证了 Medoid 计算的**绝对确定性**。

## 跨桶边出度淘汰

每个节点最多 `max_out_degree`（默认 5）条跨桶出边。
当达到上限且有更高权重的新边时，淘汰权重最低的现有边：

```python
def _add_cross_edge_with_eviction(self, from_id, to_id, weight):
    existing_cross = graph_store.get_out_edges(from_id, CROSS_BUCKET)
    if len(existing_cross) >= config.graph.max_out_degree:
        weakest = min(existing_cross, key=lambda e: e.weight)
        graph_store.remove_edge(weakest.id)
    graph_store.add_edge(from_id, to_id, CROSS_BUCKET, weight)
```

这防止了边的指数膨胀，确保图维持在可控规模。

## 休眠与唤醒

```python
def dormancy_check(self) -> list[Bucket]:
    for bucket in self._buckets.values():
        if (now - bucket.last_write_at > interval
            and now - bucket.last_query_at > interval):
            bucket.is_dormant = True
            vector_store.remove([f"medoid:{bucket.id}"])  # 移除出活跃索引
```

休眠桶的 Medoid 向量从向量索引中移除，后续搜索不再扫描该桶。
当查询命中休眠桶主题时，引擎自动调用 `wake_bucket()` 恢复。

## 桶分裂 — v0.2.0 实现

`split_bucket()` 现已实现基于谱聚类的子簇检测：

1. 构建桶内节点对的余弦相似度矩阵（|B| × |B|）
2. 计算归一化拉普拉斯矩阵：L = I - D^(-1/2) S D^(-1/2)
3. 提取 Fiedler 向量（第二小特征值对应向量）
4. 按 Fiedler 向量符号将节点分为两子桶
5. 各子桶独立计算 Medoid，注册到活跃索引

当桶内节点数 < `split_threshold`（默认 50）时不触发分裂。
