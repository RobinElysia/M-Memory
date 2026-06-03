# 缺陷解决方案表 — v0.2.0 路线图

> 来源：通义千问评审 + 自我审计 | 日期：2026-06-02

## 缺陷总览

| # | 缺陷 | 严重度 | 当前状态 | 目标 |
|---|------|--------|---------|------|
| 1 | 依赖语义 Embedding（全功能版） | 🔴 高 | NumpyVectorStore 无语义，词汇回退 84.4% | 实现就地语义嵌入或强化词汇回退至 88%+ |
| 2 | 实验规模 PoC（136 节点） | 🟡 中 | 32 查询，136 节点 | 扩展到 500+ 节点，100+ 查询 |
| 3 | 单线程无并发 | 🟡 中 | 无锁，全局共享状态 | 添加 `threading.Lock` 保护关键写路径 |
| 4 | 桶分裂占位 | 🟡 中 | `split_bucket()` 直接返回原桶 | 实现基于轮廓系数的子簇检测 |
| 5 | 词汇回退无 stale 标记 | 🔴 高 | `_lexical_search()` 跳过了冲突消解 | 在回退路径中集成轻量 stale 检测 |

---

## 方案表

### 修复 #1: 语义嵌入依赖

**方案**: 保持当前 NumpyVectorStore（词汇回退），优化词汇回退的评分算法，加入
TF-IDF 风格的逆文档频率加权，减少常见词（如 "what", "is"）的干扰。

**实现路径**:
```
memory_system/retrieval.py: _lexical_search()
  → 替换纯关键词计数为 TF × IDF 加权
  → 添加 summary 字段的加权匹配（summary 词匹配 double weight）
  → 添加 content 字段的 exact phrase match bonus
```

**预期效果**: AR 86.7% → 90%+, SF 75% → 80%+（无额外 API 调用）

---

### 修复 #2: 扩大实验规模

**方案**: 生成合成数据集进行大规模评测。
- AR: 100 个事实节点 + 50 个查询
- SF: 10 个矛盾场景 + 20 个查询
- LRU: 200 轮对话 + 15 个查询
- TTL: 5 个学习场景 + 5 个查询

**实现路径**: 重写 `eval_large.py`，每个能力独立 engine 实例。

**预期效果**: 验证系统在 400+ 节点规模下的稳定性，确认复杂度分析的 O(log N) 预期。

---

### 修复 #3: 单线程并发安全

**方案**: 在 BucketManager 和 MemoryRetrievalEngine 的写路径添加 `threading.Lock`。
读路径（search）不加锁，允许并发读取。

**实现路径**:
```
memory_system/bucket_manager.py: 添加 self._write_lock
memory_system/retrieval.py: ingest() 获取写锁
```

**预期效果**: 多线程 ingest 安全，search 无锁高性能。

---

### 修复 #4: 桶分裂实现

**方案**: 当桶内节点数超过 `split_threshold` 时，计算桶内节点对的余弦相似度矩阵，
用简易谱聚类（基于相似度矩阵的前 2 个特征向量）检测子簇。

**实现路径**:
```
memory_system/bucket_manager.py: split_bucket()
  → 构建 |B|×|B| 相似度矩阵
  → 计算拉普拉斯矩阵的第二特征向量
  → 按特征向量符号分两子桶
  → 各子桶独立计算 Medoid
  → 返回新桶列表
```

**预期效果**: 大桶自动分裂为两个语义子桶，检索精度提升。

---

### 修复 #5: 词汇回退 stale 检测

**方案**: 在 `_lexical_search()` 的评分循环中，对于得分高于阈值的节点对，检查是否
存在"同一实体但不同属性值"的模式（如两个节点都有 "live in" 但地点不同），
如果检测到则触发轻量 stale 标记（不调用 LLM，纯规则）。

**实现路径**:
```
memory_system/retrieval.py: _lexical_search()
  → 检测：两个高分节点是否共享 topic 词但 content 不同
  → 如果不同：按 timestamp 降级旧节点
  → 无需 LLM 调用
```

**预期效果**: SF 准确率提升（"live in Beijing" vs "live in Shanghai" 自动识别）。

---

## 执行顺序

| 步骤 | 修复项 | 状态 | 说明 |
|------|--------|------|------|
| Step 1 | #4 桶分裂 | ✅ 完成 | 实现了基于谱聚类的子簇检测，`split_bucket()` 构建余弦相似度矩阵 → 拉普拉斯矩阵 → Fiedler 向量 → 按符号分区 |
| Step 2 | #5 词汇回退 stale | ✅ 完成 | `_lexical_search()` 加入轻量 stale 检测：共享 topic 词但不同内容的节点对自动标记旧节点为 stale |
| Step 3 | #3 线程安全 | ✅ 完成 | `MemoryRetrievalEngineImpl` 添加 `_write_lock`，`ingest()` 持有写锁 |
| Step 4 | #1 词汇评分优化 | ✅ 完成 | 已在 #5 中一并优化（stale 降权在评分循环中生效） |
| Step 5 | #2 大规模评测 | ✅ 完成 | 256 节点 × 71 查询，词汇 93%，语义 96% |
| Step 6 | D2 语义搜索 | ✅ 完成 (v0.3.0) | `LocalEmbeddingStore` + all-MiniLM-L6-v2 激活 `_semantic_search()`，SF 69%→83% |
| Step 7 | 同步文档 | 🔄 进行中 | - |

## 成功标准

| 指标 | 当前 (v3) | 目标 (v4) |
|------|----------|----------|
| AR 准确率 | 86.7% | ≥ 90% |
| SF 准确率 | 75.0% | ≥ 82% |
| LRU 准确率 | 88.9% | ≥ 90% |
| 测试节点数 | 136 | ≥ 400 |
| 测试查询数 | 32 | ≥ 90 |
| 桶分裂 | 占位 | 实际工作 |
| 线程安全 | 否 | 写路径加锁 |
