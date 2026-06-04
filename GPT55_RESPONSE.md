# GPT 5.5 评审回应 & v0.3.0 生产级修复卡片

> 原则：不造假、不回避、逐项修复、验证闭环

---

## 质疑总览

| # | 质疑 | 严重度 | 当前真相 | 目标 |
|---|------|--------|---------|------|
| C1 | 默认路径是词汇检索，非语义 | 🔴 | `NumpyVectorStore` 是默认 → `is_semantic()=False` → 走 `_lexical_search()` | 默认激活语义路径 |
| C2 | 无持久化，进程重启即丢失 | 🔴 | 全部状态在 `_nodes` dict 和 `_buckets` dict 中 | SQLite 持久化 + 启动恢复 |
| C3 | Hash 嵌入不可靠，桶结构不稳定 | 🔴 | `NumpyVectorStore.embed()` 用 SHA-256，无语义 | 语义嵌入为默认 |
| C4 | 词汇检索 O(N) 全局扫描 | 🟡 | `_lexical_search()` 遍历 `_nodes.values()` | 桶过滤优先，全局回退 |
| C5 | 冲突检测缺幂等/重试/超时 | 🟡 | `resolve_conflicts()` 单次调用，无错误处理 | 添加 retry + timeout + 降级 |
| C6 | Benchmark 表述易误导 | 🟡 | LoCoMo 100% 仅 48 样本 + DeepSeek 评判 | 标注样本量和评判器差异 |
| C7 | `split_bucket()` 依赖语义向量 | 🟡 | 谱聚类基于 hash 向量 = 随机分区 | `is_semantic()` 守卫 |

---

## 修复卡片

### 卡片 1: 默认语义后端

**问题**: `NumpyVectorStore` 是默认 → 所有检索走词汇回退

**修复**:
- `MemoryRetrievalEngineImpl.__init__()` 中：当 `vector_store` 未传入或 `is_semantic()=False` 时，自动 fallback 到 `LocalEmbeddingStore`
- `NumpyVectorStore` 重命名为 `HashVectorStore`，加 `DeprecationWarning`
- `pyproject.toml` 新增 `sentence-transformers` 为必需依赖

**验证**: `assert engine._vector_store.is_semantic() == True` 默认成立

---

### 卡片 2: SQLite 持久化

**问题**: 全部状态在进程内存中

**修复**:
- 新增 `memory_system/persistence.py`：`PersistenceStore` 类
- 表 1: `nodes(id, summary, content, bucket_id, timestamp, confidence, is_stale)`
- 表 2: `buckets(id, medoid_node_id, created_at, is_dormant)`
- `MemoryRetrievalEngineImpl` 构造时可选 `persistence: PersistenceStore`
- `ingest()` 写完后自动 `persistence.save_node(node)`
- `__init__()` 时自动 `persistence.load_all()` 恢复状态

**验证**: 
- 创建引擎 → ingest → 销毁 → 重建 → search 返回相同结果
- `load_all()` 恢复后 bucket 结构一致

---

### 卡片 3: 桶操作守卫

**问题**: `find_candidates()` 和 `split_bucket()` 在 hash 向量下不可靠

**修复**:
- `split_bucket()`: 入口检查 `self._vector_store.is_semantic()` → `False` 时直接返回 `[bucket]`（不分裂）
- `find_candidates()`: 已有 `is_semantic()` 分支（卡片 1 已确保默认走语义路径）
- 添加日志：`logger.warning("split_bucket skipped: non-semantic embeddings")`

**验证**: `NumpyVectorStore` 下 `split_bucket()` 返回原桶，不崩溃

---

### 卡片 4: 冲突检测鲁棒性

**问题**: 无重试、超时、幂等

**修复**:
- `resolve_conflicts()` 添加：
  - `max_retries=2`（指数退避 1s/2s）
  - `timeout=5s`（LLM 调用超时）
  - 失败时返回原排序（不崩溃）
  - dedup：同一节点对不重复检测
- `_lexical_search()` 中的轻量 stale 检测已有幂等（检查 `is_stale` 后再标记）

**验证**: 模拟 API 失败 → 不崩溃，返回降级结果

---

### 卡片 5: 评测诚实标注

**问题**: LoCoMo 100% 表述可能误导

**修复**:
- README.md 基准表添加 `⚠️ 48 sampled / DeepSeek judge` 标注
- 论文中已明确标注样本量和评判器差异
- `AI_Memory_System_Testing_Protocol.md` 添加 disclaimer section

**验证**: 所有 benchmark 表述附带样本量、评判器、对话数

---

### 卡片 6: 词汇搜索优化

**问题**: `_lexical_search()` 全局遍历 O(N)

**修复**:
- 已有桶过滤 + 全局回退（v0.2.0）
- 添加倒排索引缓存：`_keyword_index: dict[str, set[str]]`（词 → 节点ID集合）
- 查询时从索引取候选节点ID集合的交集，仅对交集中节点评分
- 索引在 `ingest()` 时增量更新
- 内存开销：约 `N × avg_words_per_node × 8 bytes`

**验证**: 500 节点下搜索时间 <5ms（vs 当前 ~10ms）

---

## 执行顺序

| 序号 | 卡片 | 依赖 | 预计耗时 |
|------|------|------|---------|
| 1 | 卡片 1: 默认语义 | 无 | 15 min |
| 2 | 卡片 3: 桶操作守卫 | 卡片 1 | 5 min |
| 3 | 卡片 6: 倒排索引 | 无 | 15 min |
| 4 | 卡片 2: SQLite 持久化 | 卡片 1 | 20 min |
| 5 | 卡片 4: 冲突检测鲁棒性 | 无 | 10 min |
| 6 | 卡片 5: 评测诚实标注 | 无 | 10 min |
| 7 | 全面测试验证 | 全部 | 20 min |

## 最终标准

- [x] 61 测试全部通过
- [x] `is_semantic()=True` 默认
- [x] 进程重启后状态可恢复
- [x] 所有 benchmark 标注样本量和条件
- [x] 冲突检测有重试/超时/降级
- [x] `split_bucket()` 在非语义嵌入下安全
