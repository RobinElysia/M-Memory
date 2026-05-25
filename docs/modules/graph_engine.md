# 图引擎 · graph_engine.py

> **源码位置**: [`memory_system/graph_engine.py`](https://github.com/RobinElysia/M-Memory/blob/main/memory_system/graph_engine.py)

基于 NetworkX `MultiDiGraph` 的图存储，支持平行边。

## 为什么用 MultiDiGraph？

两个节点之间可能存在多条不同类型的边。例如节点 A 和节点 B 之间可能同时有：

- `TEMPORAL` 边（时间顺序连接）
- `CROSS_BUCKET` 边（跨桶关联）

普通 `DiGraph` 只能存一条 A→B 的边，会被覆盖。`MultiDiGraph` 允许平行边，
用 `key` 区分。

## 遍历算法 — BFS + 最优单路径去重

```python
def traverse(self, start_nodes, max_hops, weight_threshold):
    best_score: dict[str, float] = {}  # 每个节点的最高累积权重
    queue = deque()                     # BFS 队列

    for start in start_nodes:
        best_score[start] = 1.0
        queue.append((start, 1.0, 0, [start]))

    while queue:
        current, cum_weight, depth, path = queue.popleft()
        if depth >= max_hops:
            continue

        for _, neighbor, _key, data in graph.out_edges(current, data=True, keys=True):
            # 只跟随 CROSS_BUCKET 边 且 权重大于阈值
            if data["edge_type"] != EdgeType.CROSS_BUCKET:
                continue
            if data["weight"] < weight_threshold:
                continue

            new_weight = cum_weight * data["weight"]

            # 最优单路径去重
            if neighbor not in best_score or new_weight > best_score[neighbor]:
                best_score[neighbor] = new_weight
                queue.append((neighbor, new_weight, depth + 1, path + [neighbor]))
```

**累积权重**使用乘积而非加法：`new_weight = cum_weight * edge_weight`。
这确保了多跳路径的权重严格递减，避免长路径获得不合理高分。

**去重策略**：同一节点经多条路径可达时，只保留 `best_score` 最高的路径。
这就是"最优单路径"——节点只通过最优路线进入候选集，不累加分数。

## 边类型约束

| 边类型 | 是否参与遍历 | 用途 |
|--------|------------|------|
| `TEMPORAL` | ❌ | 记录对话时间顺序，不参与检索 |
| `INTRA_BUCKET` | ❌ | 节点→Medoid 的归属标记，不参与检索 |
| `CROSS_BUCKET` | ✅ | 跨桶联想边，参与 BFS 扩展 |
