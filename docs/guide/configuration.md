# 配置调优

## 完整配置

```python
from memory_system.config import MemorySystemConfig

config = MemorySystemConfig()

# 桶管理
config.bucket.top_k = 3           # 分配候选桶数
config.bucket.top_m = 5           # 搜索桶数
config.bucket.top_p = 10          # 每桶搜索节点数
config.bucket.split_threshold = 50
config.bucket.dormancy_interval_seconds = 3600

# 图
config.graph.max_out_degree = 5
config.graph.max_hops = 2
config.graph.edge_weight_threshold = 0.5

# 冲突消解
config.conflict.alpha = 0.5
config.conflict.beta = 0.3
config.conflict.gamma = 0.2
config.conflict.top_n = 5

# 清理
config.cleanup.interval_seconds = 300
config.cleanup.node_similarity_for_contradiction = 0.85
config.cleanup.medoid_similarity_floor = 0.3

# 全局
config.embedding_dim = 1536  # 必须与 VectorStore 一致
```

## 场景调优

### 低延迟场景（实时聊天）

```python
config.bucket.top_m = 2
config.bucket.top_p = 3
config.graph.max_hops = 1
```

### 高质量场景（知识管理）

```python
config.bucket.top_m = 10
config.bucket.top_p = 20
config.graph.max_hops = 3
config.graph.edge_weight_threshold = 0.3
```

### 长期记忆场景（跨会话）

```python
config.conflict.beta = 0.5       # 更重视时间
config.cleanup.interval_seconds = 600  # 更频繁清理
config.bucket.dormancy_interval_seconds = 86400  # 24h 才休眠
```
