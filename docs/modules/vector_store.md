# 向量存储 · vector_store.py

> **源码位置**: [`memory_system/vector_store.py`](https://github.com/RobinElysia/M-Memory/blob/main/memory_system/vector_store.py)

默认实现使用纯 NumPy + 余弦相似度，零外部依赖（不依赖 FAISS）。

## 确定性 hash 嵌入

当前默认的 `embed()` 方法使用 SHA-256 哈希生成确定性的伪向量，
**不具备语义理解能力**。设计目的是满足测试和 demo 场景的快速启动需求。

```python
def embed(self, text: str) -> NDArray[np.float32]:
    vec = np.zeros(self.dim, dtype=np.float32)
    for i, ch in enumerate(text.encode("utf-8")):
        h = hashlib.sha256(f"{i}:{ch}".encode()).digest()
        for j in range(min(self.dim, len(h) // 4)):
            val = int.from_bytes(h[j*4:(j+1)*4], "big")
            vec[j] += (val / 2**32) * 2.0 - 1.0  # 映射到 [-1, 1]
    # L2 归一化
    norm = np.linalg.norm(vec)
    return (vec / norm).astype(np.float32) if norm > 1e-8 else vec
```

**生产环境**：替换为实现了 `VectorStore` 接口的 FAISS / Chroma / OpenAI embedding 后端即可。

## 余弦相似度搜索

所有向量在存储时经过 L2 归一化，搜索时只需计算点积 = 余弦相似度：

```python
def search(self, query_vector, top_k):
    # 归一化查询向量
    q_norm = query_vector / np.linalg.norm(query_vector)
    # 点积 = 余弦相似度（因为存储向量已归一化）
    sims = np.dot(self._vectors, q_norm)
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [(self._metadata[i]["id"], float(sims[i])) for i in top_indices]
```

## 接口约束

所有实现必须严格遵守 `VectorStore` 接口（定义在 `interfaces.py`）：

| 方法 | 输入 | 输出 | 异常 |
|------|------|------|------|
| `embed(text)` | 字符串 | `(dim,)` float32 向量 | `RuntimeError` |
| `add(vectors, meta)` | `(n, dim)` 数组 + metadata 列表 | ID 列表 | `ValueError` (维度/数量不匹配) |
| `search(q_vec, k)` | `(dim,)` 查询向量 + k | `[(id, score), ...]` | `ValueError` (k≤0) |
| `remove(ids)` | ID 列表 | None | `KeyError` (ID 不存在) |
| `count()` | 无 | int | 无 |
