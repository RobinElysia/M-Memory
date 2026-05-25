# 快速开始

## 安装

```bash
pip install m-memory
```

## 最小可运行示例

```python
from memory_system.config import MemorySystemConfig
from memory_system.vector_store import NumpyVectorStore
from memory_system.graph_engine import NetworkXGraphStore
from memory_system.fake_llm import FakeLLMAdapter
from memory_system.retrieval import MemoryRetrievalEngineImpl

config = MemorySystemConfig()
config.embedding_dim = 8

engine = MemoryRetrievalEngineImpl(
    config=config,
    vector_store=NumpyVectorStore(dim=8),
    graph_store=NetworkXGraphStore(),
    llm=FakeLLMAdapter(),
)

# 写入记忆
engine.ingest("猫的饮食习惯", "我的猫每天吃两顿干粮")
engine.ingest("猫的健康", "猫需要定期打疫苗")

# 查询记忆
result = engine.search("猫吃什么")
for node, score in zip(result.nodes, result.scores):
    print(f"[{score:.3f}] {node.summary}: {node.content}")
```

## 换用真实 LLM

将 `FakeLLMAdapter` 替换为 OpenAI 适配器：

```python
from openai import OpenAI
from memory_system.interfaces import LLMAdapter

class OpenAIAdapter(LLMAdapter):
    def __init__(self):
        self.client = OpenAI()  # 使用 OPENAI_API_KEY 环境变量

    def complete(self, prompt: str, **kwargs) -> str:
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return resp.choices[0].message.content

    def chat(self, messages, **kwargs):
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            **kwargs
        )
        return resp.choices[0].message.content

engine = MemoryRetrievalEngineImpl(
    config=config,
    vector_store=NumpyVectorStore(dim=1536),
    graph_store=NetworkXGraphStore(),
    llm=OpenAIAdapter(),
)
```

## 换用 FAISS 向量存储

```python
from memory_system.interfaces import VectorStore

class FAISSVectorStore(VectorStore):
    # 实现 embed / add / search / remove / count 方法
    ...

engine = MemoryRetrievalEngineImpl(
    config=config,
    vector_store=FAISSVectorStore(dim=1536),
    graph_store=NetworkXGraphStore(),
    llm=your_llm_adapter,
)
```
