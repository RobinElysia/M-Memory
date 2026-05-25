# 后台维护

后台 CleanupScheduler 以周期性任务方式运行，确保系统长期健康。

## 三项核心任务

### 1. 桶休眠

当桶在 `dormancy_interval_seconds`（默认 3600s = 1 小时）内
既无写入也无查询命中时，标记为休眠。其 Medoid 向量从活跃索引中移除。

**效果**：后续检索不再扫描休眠桶 → 降低检索开销。

**唤醒**：查询命中休眠桶主题时（相似度 > 0.5），自动唤醒。

### 2. 桶内矛盾扫描

遍历桶内所有节点对，对内容语义高度相似（cosine > 0.85）的节点对，
调用 LLM 判断是否存在事实矛盾。若存在，标记旧节点为过时。

**为什么限制在桶内？** 跨桶的节点内容差异大，不太可能描述同一事实。
桶内节点因主题聚集，更可能产生矛盾。

### 3. 话题漂移检测

当节点的内容向量与桶 Medoid 向量的相似度低于 `medoid_similarity_floor`（默认 0.3），
说明该节点的主题已经明显偏离桶的中心。将其标记为过时。

## 运行控制

```python
from memory_system.cleanup import CleanupScheduler

scheduler = CleanupScheduler(config, bucket_manager, vector_store, llm, nodes)

# 方式 1: daemon 线程（同步项目）
scheduler.start_sync()

# 方式 2: asyncio Task（异步项目）
await scheduler.start_async()

# 停止
scheduler.stop()
```
