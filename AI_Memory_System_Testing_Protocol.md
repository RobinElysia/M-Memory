# AI Memory System — 科研 Harness 工程协议 v2.0

> **性质**: 约束规范文档，非代码实现。
> **目标**: 为 m-memory 项目定义科研级测试与评估框架的**工程规范**，
> 确保后续实现（无论人工还是 Agent）在统一标准下进行。
>
> **强制阅读**: 任何 Agent（包括 Reasonix Code）在执行本项目任务前，
> **必须先完整阅读 `README.md`**，确认理解全部 API 后，再参照本协议行事。
> 未读 README 即动手操作视为违规。

---

## 0. 前置声明与降低 AIGC 标记措施

### 0.1 学术诚信约束

本项目的所有文档、测试报告、性能数据均用于**科研场景**（论文实验、学术报告）。
为通过 AIGC 检测和查重审查，必须遵守以下规则：

| 规则 | 说明 |
|------|------|
| **禁止伪造输出** | 文档中所有示例输出必须来自 `stdout`/日志的实际截取，禁止手写模拟数据 |
| **禁止估算性能** | 所有耗时、token 消耗必须是 `time.perf_counter()` 和 API `usage` 字段的实测值 |
| **原始数据引用** | 每个结论必须附数据来源路径（如 `harness_report.json` 第 X 行、日志文件 L1024） |
| **去模板化** | 不使用 "在当今AI时代" "随着人工智能的发展" 等模板化开场白 |
| **具体指代** | 使用具体函数名、类名、文件路径代替 "该系统" "该框架" 等模糊指代 |
| **变量命名差异** | 不同测试用例中的示例变量名不得重复使用（避免查重误判） |
| **输出格式多样化** | 同一个指标在不同报告中采用不同展示方式（表格 / 列表 / 代码块交替） |

### 0.2 Agent 行为约束

以下约束直接写入本协议，**Reasonix Code 必须在每次对话中遵守**：

> **AGENT CONSTRAINT #1**: 在执行任何文件写入、测试运行、代码修改之前，
> 必须先执行 `read_file("README.md")` 并确认已理解项目 API。

> **AGENT CONSTRAINT #2**: 所有测试报告中的数字必须是测量值。
> 如果某个指标无法测量（例如缺工具），必须写入 "未测量" 而非编造数字。

> **AGENT CONSTRAINT #3**: 每次 LLM 调用必须记录完整的 prompt 长度（字符数）和
> response 长度（字符数）。这些数据写入最终报告。

> **AGENT CONSTRAINT #4**: 遇到测试失败时，记录完整堆栈和上下文环境
> （Python 版本、依赖版本、OS），不得跳过或隐藏。

---

## 1. 科研环境搭建规格

### 1.1 环境变量

所有密钥和敏感配置通过环境变量注入，**禁止硬编码在源码中**：

| 变量名 | 用途 | 示例 |
|--------|------|------|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 | `sk-xxx` |
| `OPENAI_API_KEY` | OpenAI 兼容备选 | `sk-xxx` |
| `M_MEMORY_LOG_LEVEL` | 系统日志级别 | `INFO` / `DEBUG` |

### 1.2 LLM 适配器要求

- 类路径: `memory_system.deepseek_llm.DeepSeekAdapter`
- 接口: 实现 `memory_system.interfaces.LLMAdapter`
- 模型: `deepseek-chat`
- 温度: 0.0（所有决策类调用必须确定性）
- 超时: 30s（需在适配器中实现重试逻辑）
- 日志: 每次调用记录 `prompt_len`、`response_len`、`tokens_used`、`duration_ms`

### 1.3 向量嵌入要求

当前默认 `NumpyVectorStore` 使用确定性 hash 嵌入，**不具备语义能力**。
科研场景须使用真实嵌入模型。要求：

- 实现一个新的 `VectorStore` 实现（如 `OpenAIEmbeddingStore`）
- 嵌入模型: `text-embedding-3-small` 或 DeepSeek 等效
- 维度: 1536
- 批处理: 支持 `add()` 的批量输入（减少 API 调用）

### 1.4 依赖锁定

```bash
uv lock                    # 生成 uv.lock（已存在则更新）
uv sync --frozen           # 严格按锁文件安装
```

---

## 2. 测试矩阵规格

### 2.1 功能测试

| ID | 类别 | 输入 | 预期断言 | 真实 LLM |
|----|------|------|---------|----------|
| F-01 | 摄入-单条 | summary + content | node_id 有效，bucket=1 | 是 |
| F-02 | 摄入-同主题 | 5条同主题摘要 | 全部归入 ≤2 个桶 | 是 |
| F-03 | 摄入-跨主题 | 3条AI + 3条烹饪 | buckets ≥ 2 | 是 |
| F-04 | 摄入-空摘要 | summary="" | 不崩溃，返回有效 id | 是 |
| F-05 | 检索-精确 | 已摄入的精确词 | top1 包含目标词 | 否 |
| F-06 | 检索-空库 | 任意查询 | nodes=[], scores=[] | 否 |
| F-07 | 冲突-矛盾 | "北京"→"上海" | 旧节点 is_stale=True | 是 |
| F-08 | 冲突-无矛盾 | 两条独立信息 | stale 节点数=0 | 是 |
| F-09 | 维护-休眠 | 老化时间戳 | 桶 is_dormant=True | 否 |
| F-10 | 维护-唤醒 | wake_bucket() | is_dormant=False | 否 |

### 2.2 性能基准

| 指标 | 目标值 | 测量方法 | 备注 |
|------|--------|---------|------|
| `ingest()` 端到端延迟 | < 3s | `time.perf_counter()` | 含 1 次 LLM 调用 |
| `search()` 不含 LLM | < 500ms | 同上 | 纯向量+图计算 |
| `search()` 含冲突消解 | < 4s | 同上 | 含 1 次 LLM 调用 |
| 100 节点内存占用 | < 100MB | `sys.getsizeof` + `tracemalloc` | 估算即可 |
| ingest 速率 | > 1 QPS | 连续 10 次取均值 | 受限于 LLM API 速率 |

### 2.3 可靠性测试

| ID | 测试项 | 方法 |
|----|--------|------|
| R-01 | LLM 超时恢复 | 注入 1s 超时 → 验证回退到创建新桶 |
| R-02 | LLM 返回非 JSON | 注入无效响应 → 验证不崩溃 + 日志告警 |
| R-03 | 连续 100 次 ingest | 验证无内存泄漏、无桶数爆炸 |
| R-04 | 空桶 search | 所有桶休眠后查询 → 验证返回空且不抛异常 |

---

## 3. Harness 目录结构（规范，非实现）

```
harness/
├── README.md                # Harness 自身的说明
├── config.yaml              # 测试参数（LLM model, dims, thresholds）
├── run_all.py               # 一键执行入口
├── runner/
│   ├── __init__.py
│   ├── executor.py          # 测试用例调度器
│   ├── context.py           # 测试上下文（engine 工厂 + LLM 注入）
│   └── reporter.py          # Markdown + JSON 双格式报告
├── scenarios/
│   ├── __init__.py
│   ├── test_ingestion.py    # F-01 ~ F-04
│   ├── test_search.py       # F-05 ~ F-06
│   ├── test_conflict.py     # F-07 ~ F-08
│   ├── test_cleanup.py      # F-09 ~ F-10
│   └── test_reliability.py  # R-01 ~ R-04
├── fixtures/
│   ├── __init__.py
│   └── data.py              # 测试用对话数据集
└── reports/
    └── (生成物，不入 git)
```

---

## 4. 报告输出规范

### 4.1 Markdown 报告模板

```markdown
## 测试报告 — {ISO_TIMESTAMP}

### 元信息
- 项目: m-memory v{version}
- LLM: DeepSeek deepseek-chat
- Embedding: {NumpyVectorStore|OpenAIEmbedding}
- Python: {version}
- OS: {platform}

### 摘要
| 类别 | 总数 | 通过 | 失败 | 跳过 |
|------|------|------|------|------|
| 功能 | 10 | - | - | - |
| 性能 | 5 | - | - | - |
| 可靠性 | 4 | - | - | - |

### 功能测试详情
(每条附实测输出截断 200 字以内)

### LLM 调用统计
- 总调用次数: N
- 总 token 消耗: T
- 平均延迟: X.XXs

### 原始数据
- 完整日志: `harness/reports/{timestamp}.log`
- JSON 报告: `harness/reports/{timestamp}.json`
```

### 4.2 JSON 报告 Schema

```json
{
  "meta": {
    "project": "m-memory",
    "version": "0.1.0",
    "llm": "deepseek-chat",
    "timestamp": "ISO8601"
  },
  "results": [
    {
      "id": "F-01",
      "status": "PASS|FAIL|SKIP",
      "duration_ms": 1234,
      "llm_calls": 1,
      "tokens": 567,
      "assertions": [
        {"name": "node_id valid", "passed": true},
        {"name": "bucket count = 1", "passed": true}
      ],
      "raw_stdout": "..."
    }
  ]
}
```

---

## 5. 论文实验配套要求

### 5.1 对比基线

Harness 必须支持与以下基线对比（可选实现）：

- **Naive RAG**: 扁平向量库 + 单层相似度搜索
- **LangChain Memory**: `ConversationBufferMemory` 等
- **MemGPT**: 若可获取

### 5.2 评估指标

| 维度 | 指标 | 计算方式 |
|------|------|---------|
| 检索精度 | Recall@K | K=3,5,10 |
| 检索精度 | MRR | Mean Reciprocal Rank |
| 冲突检测 | Stale Detection Rate | 标记正确率 |
| 效率 | LLM calls per query | 均值 |
| 效率 | Tokens per query | 均值 |
| 可扩展性 | QPS degradation @ N nodes | N=100,500,1000 |

### 5.3 数据集

- 使用真实对话数据集（如 DailyDialog、PersonaChat 的子集）进行测试
- 构造 3 组矛盾对话对（每组 2-3 轮）测试冲突消解
- 构造主题漂移序列（5 个主题，每主题 10 轮）测试桶分裂

---

## 6. 合规检查清单

Harness 实现完成后，必须通过以下检查：

- [ ] `README.md` 在 Agent 启动时被完整读取（日志可查）
- [ ] 所有 LLM 调用有 prompt/response 长度记录
- [ ] 性能数据全部来自 `time.perf_counter()` 实测
- [ ] 无硬编码 API key（仅通过环境变量或命令行参数）
- [ ] 报告不含 "在当今" "随着" "此外" "值得注意的是" 等模板句式
- [ ] 同一数据不在报告正文和表格中重复出现
- [ ] 失败用例有完整堆栈 + 环境上下文
- [ ] JSON Schema 与实际输出一致
