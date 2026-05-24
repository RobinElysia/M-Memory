# 第 0 层 · 架构冻结层 (Architecture Freeze)

## 目标
确保 Agent 完全理解领域模型，生成可直接映射到代码的接口定义与参数清单，避免后续歪曲设计。

## 输入
- `ARCHITECTURE_DESIGN.md`（上文，必须完整读取）

## Agent 任务
1. **结构化接口定义**：将自然语言描述转化为类、方法签名、数据模型（Pydantic 或 dataclass）。
   - 核心实体：`MemoryNode(A, C)`, `Bucket`, `Medoid`, `Edge(时间边/桶内边/跨桶边)`
   - 关键算法伪代码（可直接写入 docstring 或注释）：
     - 分桶决策流程（含 LLM 调用步骤）
     - 图游走检索流程（含去重、最优路径选取）
     - 冲突消解流程（含重排序与过时标记）
     - 后台清理流程（周期性任务，含桶分裂与休眠逻辑）
2. **参数提取**：从描述中识别所有可配置参数，生成 `config.py` 骨架，明确类型与默认值。
   包括但不限于：`top_k`, `top_m`, `top_p`, `max_out_degree`, 权重 α/β/γ, 相似度阈值, 最大跳数, 边权重阈值, TOP-N, 冷存储阈值等。
3. **生成文档**：输出一份 `ARCHITECTURE.md`（不同于设计文档，此为实现视角），包含上述接口与参数表。

## 验收标准
- `ARCHITECTURE.md` 经人工审核，确认与 `ARCHITECTURE_DESIGN.md` 零二义性。
- Agent 能口头回答任意流程的边界条件（例如：“桶只有一个节点时 Medoid 如何计算？”答案应为：该节点自身即为 Medoid）。
- 参数定义完整，且全部有合理默认值。

## 交付物
- `memory_system/config.py`（只含参数定义，不含业务逻辑）
- `docs/ARCHITECTURE.md`

通过后，颁发 **Layer 0 Pass Token**，进入第 1 层。