# 检索与查询指南

## search() 参数

```python
result = engine.search(
    query="用户的问题",
    max_hops=2,          # 可选：覆盖默认图扩展深度
    weight_threshold=0.5, # 可选：覆盖默认边权重阈值
)
```

## SearchResult

```python
@dataclass
class SearchResult:
    nodes: list[MemoryNode]   # 按相关性降序
    scores: list[float]       # 一一对应的最终分数
```

## 使用场景

### 精确查找

```python
result = engine.search("猫的疫苗什么时候打", max_hops=0)
```

设置 `max_hops=0` 禁用图扩展，只返回桶内直接匹配的结果。

### 广泛探索

```python
result = engine.search("宠物健康", max_hops=3, weight_threshold=0.3)
```

更多跳数 + 更低权重阈值 = 更广泛的跨主题联想。

### 检查过时信息

```python
result = engine.search("我住在哪里")
for node in result.nodes:
    if node.is_stale:
        print(f"[过时] {node.content} (置信度: {node.confidence:.2f})")
```

## 解读分数

分数由 Layer 1（相似度 + 图扩展）和 Layer 2（冲突消解）共同决定：

- **> 0.8**：高度相关，通常是桶内直接命中
- **0.5 - 0.8**：相关，可能是图扩展发现的关联内容
- **0.2 - 0.5**：弱相关，或已被降权的过时信息
- **< 0.2**：已被严重降权或仅微弱相关

## 性能建议

| 知识库规模 | top_m | top_p | max_hops |
|-----------|-------|-------|----------|
| < 100 条 | 5 | 20 | 2 |
| 100-1000 条 | 5 | 10 | 2 |
| 1000-10000 条 | 3 | 10 | 1 |
| > 10000 条 | 2 | 5 | 1 |
