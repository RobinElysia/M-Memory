# GPT 5.5 Round 2 评审回应 & v0.3.0 修复卡片

## 质疑总览

| # | 质疑 | 严重度 | 状态 |
|---|------|--------|------|
| 1 | API key 硬编码 | 🔴 | ✅ 上轮已修复 |
| 2 | `_cosine_sim` 三处重复 | 🟡 | → 提取到 utils.py |
| 3 | 向量无持久化 | 🟡 | → 后续版本 |
| 4 | `persistence.py` 未接线 | 🔴 | → 引擎注入 persistence |
| 5 | `split_bucket` add_node bug | 🔴 | → 改为 update 现存节点 |
| 6 | 线程安全不完整 | 🟡 | → 读锁保护 find_candidates |
| 7 | 测试用已弃用类 | 🟡 | → 测试改用 HashVectorStore |
| 8 | JSON 解析脆弱 | 🔴 | → structured output 或增强容错 |
| 9 | 冲突检测用索引非ID | 🔴 | → 直接传 node ID |
| 10 | 停用词不一致 | 🟡 | → 统一定义 + 英文检测 |
| 11 | topic_words 过拟合 | 🟡 | → LLM judge 替代 + 可配置 |
| 12 | config.embedding_dim 不匹配 | 🔴 | → 启动校验 |
| 13 | CI 未覆盖语义路径 | 🟡 | → 后续CI更新 |
| 14 | version 不一致 | 🔴 | → 统一 0.3.0 |
| 15 | cli.py 缺失 | 🔴 | → 创建 + main.py 修复 |

---

## 修复卡片

### 卡片 2: 提取 _cosine_sim 到 utils

**修复**: 新建 `memory_system/utils.py`，包含 `cosine_sim()`。
删除 `bucket_manager.py`, `retrieval.py`, `cleanup.py` 中的重复定义。

### 卡片 4: 引擎接入 persistence

**修复**: `MemoryRetrievalEngineImpl.__init__` 接受 `persistence: PersistenceStore | None`。
`ingest()` 后自动保存 node + bucket。`__init__` 时自动 `load_all_nodes()` 恢复。

### 卡片 5: split_bucket 图节点处理

**修复**: `add_node` → 检查节点存在性，存在则 `update_node_attrs`。

### 卡片 8: LLM JSON 容错增强

**修复**: 新增 `llm_decision.py:parse_json_robust()` — 
尝试 1: 直接 json.loads；2: 提取 ```json ```；3: 找首尾 {}；4: 正则 key-value 提取。

### 卡片 9: 冲突检测传 node_id

**修复**: `build_conflict_detection_prompt` 不再用索引 `[{0}]`，改用 `[node_id:abc123]`。

### 卡片 10: 停用词统一

**修复**: `utils.py` 定义 `STOPWORDS: set[str]`，三处引用同一份。

### 卡片 12: 启动维度校验

**修复**: `MemoryRetrievalEngineImpl.__init__` 中检查 `config.embedding_dim == vector_store.dim`。

### 卡片 14: 版本统一 0.3.0

**修复**: `__init__.py` 和 `pyproject.toml` 统一为 `"0.3.0"`。

### 卡片 15: 创建 cli.py

**修复**: 创建 `memory_system/cli.py` 含 `main()` 入口。
