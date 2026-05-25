# 后台清理 · cleanup.py

> **源码位置**: [`memory_system/cleanup.py`](https://github.com/RobinElysia/M-Memory/blob/main/memory_system/cleanup.py)

周期性的后台维护任务，确保记忆系统不会随时间推移而退化。

## 双模式运行

```python
class CleanupScheduler:
    def start_sync(self):   # daemon 线程，不阻塞
    async def start_async(self):  # asyncio Task
    def stop(self):         # 停止并等待
```

## 清理周期

每个清理周期执行三个扫描：

### 1. 桶休眠检查

```python
def dormancy_check(self):
    for bucket in all_buckets:
        if (now - bucket.last_write_at > interval
            and now - bucket.last_query_at > interval):
            bucket.is_dormant = True
            vector_store.remove([f"medoid:{bucket.id}"])
```

休眠的桶其 Medoid 向量被移出活跃索引，后续粗筛不再遍历它。
这减少了检索开销——你不会再在每个查询中扫描已经"死亡"的主题。

### 2. 桶内矛盾扫描

```python
def _scan_contradictions(self, bucket):
    for i in range(len(node_ids)):
        for j in range(i + 1, len(node_ids)):
            sim = cosine_sim(node_a.content_vector, node_b.content_vector)
            if sim < config.cleanup.node_similarity_for_contradiction:
                continue  # 语义不相似 → 不可能有矛盾
            # 语义非常相似 → 让 LLM 判断是否有事实矛盾
            prompt = build_conflict_detection_prompt(...)
            response = llm.complete(prompt)
            # 标记旧节点为过时
```

**为什么要先检查内容相似度？** 只有内容语义足够相似的节点才可能描述同一事实。
如果两段内容完全不相关（如"猫的饮食"和"狗的散步"），
没必要让 LLM 检查矛盾。`node_similarity_for_contradiction` 阈值（默认 0.85）
充当粗筛，减少不必要的 LLM 调用。

### 3. 话题漂移检测

```python
def _scan_topic_drift(self, bucket):
    for node_id in bucket.node_ids:
        sim = cosine_sim(node.content_vector, bucket.medoid.vector)
        if sim < config.cleanup.medoid_similarity_floor:
            node.is_stale = True  # 软删除（降权，不移除）
```

当一个节点的内容与桶 Medoid 的相似度过低时，说明话题已经漂移。
将其标记为过时（降权、保留），而非物理删除。
