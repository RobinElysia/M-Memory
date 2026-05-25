# AI Memory System Testing Protocol v2.0

> **执行者**: Reasonix Code (deepseek-v4-pro)
> **强制约束**: 本协议由 Reasonix 在 Harness 框架内严格遵循，不可跳过任一步骤。
> **AIGC 审查**: 所有输出须经人工审核，禁止完全依赖生成内容作为最终结论。

---

## 0. 前置声明

本协议设计的测试方案要求 Agent 在 **无人类干预** 的条件下完成以下全部步骤。
凡涉及 LLM 调用的环节，须记录完整的 prompt、response、token 消耗。
所有测试结果须附上**可复现证据**（命令行输出、日志摘要、API 返回截断）。

**降低 AIGC 标记措施**：
- 所有文档中的示例输出必须来自真实运行结果，禁止伪造
- 所有性能数据必须来自实际测量，禁止估算
- 所有结论必须附原始数据引用路径

---

## 1. 环境准备

### 1.1 依赖验证
```bash
uv run python -c "import memory_system; print(memory_system.__version__)"
```

### 1.2 LLM 接入验证
- 适配器: `memory_system.deepseek_llm.DeepSeekAdapter`
- API Key: 由用户提供（不写入文档）
- Model: `deepseek-chat`
- 验证方法: 发送单条 `complete("Hello, respond with 'OK'")` 并确认返回 OK

### 1.3 向量存储验证
- 默认后端: `NumpyVectorStore`
- 替换后端: `FAISSVectorStore`（待实现）
- 验证方法: embed → add → search → remove 全流程通过

---

## 2. 核心功能测试矩阵

### 2.1 摄入测试 (Ingestion)

| 测试ID | 场景 | 输入 | 预期 |
|--------|------|------|------|
| ING-01 | 单条摄入 | summary="AI研究", content="Transformer架构..." | 返回有效 node_id，创建 1 个桶 |
| ING-02 | 同主题摄入 | 连续 5 条 AI 相关 | 全部归入同一桶 |
| ING-03 | 跨主题摄入 | 3 条 AI + 3 条 Cooking | 形成至少 2 个桶 |
| ING-04 | 空摘要 | summary="" | 应能正常处理（零向量） |
| ING-05 | 超长内容 | content=10000 字中文 | 不应崩溃，记录 token 消耗 |

### 2.2 检索测试 (Search)

| 测试ID | 场景 | 查询 | 预期 |
|--------|------|------|------|
| SCH-01 | 精确匹配 | "Transformer架构" | 返回相关节点，top1 分数 > 0.5 |
| SCH-02 | 语义搜索 | "注意力机制" | 返回 Transformer 节点（即使词不匹配） |
| SCH-03 | 跨桶联想 | "AI和烹饪" | 通过图扩展返回两个桶的节点 |
| SCH-04 | 空库搜索 | 任意查询 | 返回空结果，不报错 |
| SCH-05 | 休眠桶唤醒 | 查询已休眠桶的主题 | 自动唤醒并返回结果 |

### 2.3 冲突消解测试 (Conflict)

| 测试ID | 场景 | 输入 | 预期 |
|--------|------|------|------|
| CNF-01 | 直接矛盾 | "北京" → "上海" | 旧信息标记 stale，新信息优先 |
| CNF-02 | 无矛盾 | 两条独立信息 | 均不标记 stale |
| CNF-03 | 多重矛盾 | A→B→C 三次变更 | 仅最新 (C) 非 stale，其余降权 |

### 2.4 后台维护测试 (Cleanup)

| 测试ID | 场景 | 操作 | 预期 |
|--------|------|------|------|
| CLN-01 | 桶休眠 | 老化桶时间戳 + dormancy_check() | 桶标记 dormant，Medoid 移除 |
| CLN-02 | 桶唤醒 | wake_bucket() | 恢复活跃状态 |
| CLN-03 | 话题漂移 | 插入语义远离桶 Medoid 的节点 | 节点标记 stale |

---

## 3. 性能基准

| 指标 | 目标 | 测量方法 |
|------|------|---------|
| ingest 延迟 | < 2s（含 1 次 LLM 调用） | `time.perf_counter()` |
| search 延迟 | < 500ms（不含 LLM） | `time.perf_counter()` |
| search 延迟（含冲突消解） | < 3s（含 1 次 LLM 调用） | `time.perf_counter()` |
| 内存占用（100 节点） | < 50MB | `sys.getsizeof()` 估算 |
| LLM 调用次数/ingest | 1 | 计数器 |
| LLM 调用次数/search | 1（仅冲突消解） | 计数器 |

---

## 4. Harness 工程框架

### 4.1 目录结构
```
harness/
├── __init__.py
├── runner.py          # 测试执行器
├── metrics.py         # 指标采集
├── reporter.py        # 报告生成
└── scenarios/
    ├── ingestion.py   # 摄入场景
    ├── search.py      # 检索场景
    ├── conflict.py    # 冲突场景
    └── cleanup.py     # 维护场景
```

### 4.2 Runner 规范
- 每个测试场景独立运行
- 失败不阻塞后续场景
- 自动采集：耗时、token 消耗、LLM 调用次数、异常堆栈
- 输出 JSON + Markdown 两种格式报告

### 4.3 报告格式
```markdown
## 测试报告 — {timestamp}

### 摘要
- 总测试数: N
- 通过: P / 失败: F / 跳过: S
- 总耗时: X.XXs
- 总 token 消耗: T

### 详细结果
| ID | 场景 | 状态 | 耗时 | LLM调用 | 备注 |
|----|------|------|------|---------|------|
| ING-01 | 单条摄入 | PASS | 0.8s | 1 | - |
```

---

## 5. 强制要求（Agent 必须遵守）

1. **阅读 README.md** — 在开始测试前重新读取项目 README，确认理解所有 API
2. **真实执行** — 所有测试必须实际运行代码，不得模拟结果
3. **记录原始输出** — 每个测试用例附上 `stdout`/`stderr` 截断
4. **LLM 调用透明** — 每次 LLM 调用记录 prompt 长度 + response 长度 + token 数
5. **不做 AIGC 伪造** — 报告中的数字必须是测量值，不是估算值
6. **错误不掩盖** — 测试失败时记录完整堆栈，不跳过不隐藏
