# 第 1 层 · 合同与骨架层 (Contracts & Skeleton)

## 目标
先定义抽象接口（合约），再谈实现，确保各组件可替换、可测试。

## 输入
- 第 0 层的 `docs/ARCHITECTURE.md`
- `memory_system/config.py`

## Agent 任务
1. **定义抽象基类/Protocol**：
   - `VectorStore`：包含 `embed(text) -> Vector`, `add(vectors, metadata)`, `search(query_vector, top_k)` 等方法。
   - `LLMAdapter`：`complete(prompt) -> str`, `chat(messages) -> str`，支持注入假 LLM。
   - `GraphStore`：`add_node(node_id, attributes)`, `add_edge(from_id, to_id, edge_type, weight)`, `traverse(start_nodes, max_hops, weight_threshold) -> list`。
   - `BucketManager`：`find_candidates(node_a) -> list[Bucket]`, `assign_to_bucket(node_a, bucket)`, `split_bucket(bucket) -> list[Bucket]`, `dormancy_check(bucket)`。
   - `MemoryRetrievalEngine`：`search(query, max_hops, ...) -> SearchResult`, `resolve_conflicts(candidates) -> list[C]`。
2. **编写合约**：为每个接口方法编写详细的 Google 风格 docstring，包括参数、返回、异常、副作用说明。
3. **项目骨架**：
   - 生成 `pyproject.toml`，包含包名、版本、Python 要求、依赖（如 `pydantic`, `numpy`, `faiss-cpu`, `openai` 等）、测试框架 `pytest`、lint 工具配置。
   - 创建包目录结构：`memory_system/` 下放置 `__init__.py`, `interfaces.py`, `models.py`，保留各模块空壳。

## 约束
- 所有函数签名必须带类型注解，禁止使用 `Any`（除非确有必要并注释原因）。
- 每个公共方法必须有完整 docstring。
- 抽象接口中不得出现具体实现（例如不能有 `import faiss` 在接口文件内）。

## 验收标准
- 运行 `mypy --strict memory_system/` 通过（针对接口和模型定义）。
- 随机抽检 3 个核心接口的 docstring，确认与架构描述一致。
- 项目能通过 `pip install -e .` 无错误安装（即使内部逻辑为空）。

## 交付物
- `memory_system/interfaces.py`
- `memory_system/models.py`
- `pyproject.toml`
- `docs/contracts.md`（自动从 docstring 或手工整理）

通过后，颁发 **Layer 1 Pass Token**，进入第 2 层。