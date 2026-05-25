# 配置参数 · config.py

> **源码位置**: [`memory_system/config.py`](https://github.com/RobinElysia/M-Memory/blob/main/memory_system/config.py)

所有可调参数集中在 5 个 dataclass 中，有类型注解 + 默认值 + 中文说明。

## BucketConfig

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `top_k` | int | 3 | 新节点分配时考虑的候选桶数 |
| `top_m` | int | 5 | 搜索时检索的桶数 |
| `top_p` | int | 10 | 每桶内检索的节点数 |
| `split_threshold` | int | 50 | 触发桶分裂的节点数下限 |
| `dormancy_interval_seconds` | float | 3600 | 无活动后判定休眠的时间窗口 (秒) |
| `cold_storage_similarity_threshold` | float | 0.3 | 节点与 Medoid 相似度低于此值可能移入冷存储 |

## GraphConfig

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `max_out_degree` | int | 5 | 节点最大跨桶出度数 |
| `weight_decay_threshold` | float | 0.2 | 边权重低于此值则剪枝 |
| `max_hops` | int | 2 | 图游走最大跳数 |
| `edge_weight_threshold` | float | 0.5 | 游走时的边权重门槛 |

## ConflictConfig

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `alpha` | float | 0.5 | 重排序公式中语义相似度权重 |
| `beta` | float | 0.3 | 重排序公式中时间新近度权重 |
| `gamma` | float | 0.2 | 重排序公式中来源置信度权重 |
| `top_n` | int | 5 | 送入 LLM 做矛盾检测的候选数 |
| `stale_mark_downgrade_factor` | float | 0.1 | 过时标记后的置信度乘数 |

## CleanupConfig

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `interval_seconds` | float | 300 | 清理任务间隔 (秒) |
| `node_similarity_for_contradiction` | float | 0.85 | 矛盾扫描的内容相似度阈值 |
| `medoid_similarity_floor` | float | 0.3 | 话题漂移的相似度下限 |

## 使用

```python
from memory_system.config import MemorySystemConfig

config = MemorySystemConfig()
config.bucket.top_k = 5       # 更多候选桶
config.graph.max_hops = 3     # 更深的图游走
config.conflict.alpha = 0.7   # 更重视语义相关性

engine = MemoryRetrievalEngineImpl(
    config=config,
    ...
)
```
