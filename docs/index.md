# m-memory · AI Agent 双层记忆系统

**m-memory** 是一个专为 AI Agent 设计的**双层长程记忆基础设施**。
通过 **增量聚类 + 图联想检索 + 自动冲突消解** 三个核心机制，
为 Agent 提供高效、可解释、自我维护的记忆能力。

## 为什么需要它？

AI Agent 在长对话中面临三个根本问题：

| 问题 | m-memory 的解法 |
|------|----------------|
| **记忆爆炸** — 对话轮次无限增长，无法全部放入上下文 | 动态分桶压缩：每轮对话只存摘要向量，检索时才取原文 |
| **语义漂移** — 话题切换后旧信息干扰新检索 | 刚性分桶 + 柔性图边：同类聚集，跨类联想 |
| **事实矛盾** — 用户后来说的话与前面冲突 | LLM 驱动的冲突检测 + 过时降权（不删除） |

## 核心概念

```
           ┌─────────────────────────────────┐
           │         MemoryNode               │
           │  ┌───── A (摘要) ──────┐         │
           │  │  用于粗筛向量检索    │         │
           │  └────────────────────┘         │
           │  ┌───── C (详情) ──────┐         │
           │  │  用于精确检索+消解   │         │
           │  └────────────────────┘         │
           └─────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
      Bucket A       Bucket B       Bucket C
     (Medoid α)     (Medoid β)     (Medoid γ)
          │              │              │
          └──────┬───────┘              │
                 │ 跨桶边               │
                 ▼                      │
            图联想扩展 ◄────────────────┘
```

每个 **MemoryNode** 包含两份信息：

- **A（摘要）**：一段总结性文字，向量化后用于第一层粗筛。类比于书的目录。
- **C（详情）**：原始对话的全部内容，向量化后用于第二层精确检索。类比于书的正文。

## 5 分钟快速开始

```bash
pip install m-memory
```

```python
from memory_system.config import MemorySystemConfig
from memory_system.vector_store import NumpyVectorStore
from memory_system.graph_engine import NetworkXGraphStore
from memory_system.fake_llm import FakeLLMAdapter
from memory_system.retrieval import MemoryRetrievalEngineImpl

# 1. 创建引擎（使用假 LLM，无需 API key）
config = MemorySystemConfig()
config.embedding_dim = 8
engine = MemoryRetrievalEngineImpl(
    config=config,
    vector_store=NumpyVectorStore(dim=config.embedding_dim),
    graph_store=NetworkXGraphStore(),
    llm=FakeLLMAdapter(),
)

# 2. 摄入记忆
engine.ingest("我的猫", "猫咪喜欢在阳光下睡觉")
engine.ingest("我的狗", "狗狗喜欢在公园奔跑")
engine.ingest("猫粮选择", "我给猫喂优质干粮")

# 3. 检索记忆
result = engine.search("告诉我关于猫的事情")
for node, score in zip(result.nodes, result.scores):
    print(f"[{score:.3f}] {node.summary}: {node.content}")
```

输出示例：
```
[0.812] 猫粮选择: 我给猫喂优质干粮
[0.745] 我的猫: 猫咪喜欢在阳光下睡觉
```

## 检索流程（一图胜千言）

```
用户查询 "my cat"
       │
       ▼
┌──────────────────────────────────────────┐
│  Layer 1: 桶级粗筛                        │
│  query向量 vs 所有活跃桶Medoid向量         │
│  → 找到 top_m 个语义最相近的桶             │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  Layer 1: 桶内精筛                        │
│  query向量 vs 桶内所有A节点向量            │
│  → 产生 seed nodes（种子节点）            │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  Layer 1: 图联想扩展                      │
│  从种子节点沿跨桶边 BFS 游走               │
│  → 发现关联桶中的相关节点                  │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  Layer 2: 冲突消解                        │
│  加权重排序 → LLM矛盾检测 → 过时降权       │
│  → 返回最终有序结果                       │
└──────────────────────────────────────────┘
```

## 项目结构

```
memory_system/
├── config.py          # 全部可配置参数（含默认值 + 说明）
├── interfaces.py      # 5 个抽象接口（可替换后端）
├── models.py          # 7 个核心数据模型
├── vector_store.py    # 向量存储（默认 NumPy 实现）
├── graph_engine.py    # 图存储与遍历（默认 NetworkX 实现）
├── bucket_manager.py  # 桶生命周期 + 节点分配
├── retrieval.py       # 顶层检索引擎（组合所有组件）
├── llm_decision.py    # LLM prompt 构建与响应解析
├── cleanup.py         # 后台清理调度器
└── fake_llm.py        # 假 LLM 适配器（测试 + 无 API key 运行）
```

## 设计原则

1. **绝对确定性** — 零随机。Medoid 计算用平均距离最小者，平局按 ID 字典序。
2. **物理单桶存放** — 每个 A 节点只存在一个主桶中，跨桶关联仅通过边。
3. **LLM 可观测** — 每次 LLM 调用记录 prompt 长度、token 消耗、决策结果。
4. **接口可替换** — VectorStore / LLMAdapter / GraphStore 均为抽象类，换后端不改上层。
