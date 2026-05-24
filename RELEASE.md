# m-memory v0.1.0

> 双层记忆系统：增量聚类 + 图检索 + 冲突消解，专为 AI Agent 长程记忆设计。

## 核心能力

| 特性 | 说明 |
|------|------|
| **双层检索** | Layer 1 桶级粗筛 → 图联想扩展 → Layer 2 冲突消解 |
| **动态分桶** | LLM 驱动的增量聚类，Medoid 确定性计算，无随机 |
| **柔性图边** | 时间边 / 桶内边 / 跨桶边，最优单路径去重，出度淘汰 |
| **冲突消解** | `α·sim + β/(1+ΔT) + γ·confidence` 重排序 + LLM 矛盾检测 + 过时降权 |
| **后台维护** | 桶休眠/唤醒、话题漂移检测、周期矛盾扫描 |
| **可替换后端** | 抽象接口 VectorStore / LLMAdapter / GraphStore，默认 NumPy + NetworkX |

## 安装

```bash
pip install dist/m_memory-0.1.0-py3-none-any.whl
```

## 5 分钟快速开始

```python
from memory_system.config import MemorySystemConfig
from memory_system.vector_store import NumpyVectorStore
from memory_system.graph_engine import NetworkXGraphStore
from memory_system.fake_llm import FakeLLMAdapter
from memory_system.retrieval import MemoryRetrievalEngineImpl

engine = MemoryRetrievalEngineImpl(
    config=MemorySystemConfig(),
    vector_store=NumpyVectorStore(dim=1536),
    graph_store=NetworkXGraphStore(),
    llm=FakeLLMAdapter(),
)

# 摄入记忆
engine.ingest("my cat", "My cat loves sleeping in the sun")
engine.ingest("my dog", "My dog enjoys running at the park")

# 检索
result = engine.search("tell me about my cat")
for node, score in zip(result.nodes, result.scores):
    print(f"[{score:.3f}] {node.summary}: {node.content}")
```

## 质量

| 门禁 | 状态 |
|------|------|
| `ruff check` | 0 errors |
| `mypy --strict` | 0 errors |
| `pytest` (61 tests) | 100% passed |
| 代码覆盖率 | 89% |
| Python | ≥ 3.11 |

## 依赖

`pydantic` · `numpy` · `faiss-cpu` · `openai` · `structlog` · `networkx`
