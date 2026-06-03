# m-memory — Dual-Layer Memory System for AI Agents

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-89%25-brightgreen)](.)

A dual-layer memory system for AI agents with **incremental clustering**,
**graph-based retrieval**, and **automatic conflict resolution**.

## Quick Start (5 minutes)

```bash
pip install m-memory
```

```python
from memory_system.config import MemorySystemConfig
from memory_system.vector_store import NumpyVectorStore
from memory_system.graph_engine import NetworkXGraphStore
from memory_system.fake_llm import FakeLLMAdapter
from memory_system.retrieval import MemoryRetrievalEngineImpl

# 1. Create the engine
config = MemorySystemConfig()
config.embedding_dim = 8  # small dim for demo
engine = MemoryRetrievalEngineImpl(
    config=config,
    vector_store=NumpyVectorStore(dim=config.embedding_dim),
    graph_store=NetworkXGraphStore(),
    llm=FakeLLMAdapter(),
)

# 2. Ingest memories
id1 = engine.ingest("my cat", "My cat loves sleeping in the sun")
id2 = engine.ingest("my dog", "My dog enjoys running at the park")
id3 = engine.ingest("cat food", "I feed my cat premium dry food")

# 3. Search memories
result = engine.search("tell me about my cat")
for node, score in zip(result.nodes, result.scores):
    print(f"[{score:.3f}] {node.summary}: {node.content}")
```

**Output example:**
```
[0.812] cat food: I feed my cat premium dry food
[0.745] my cat: My cat loves sleeping in the sun
```

## Core Concepts

| Concept | Description |
|---------|-------------|
| **MemoryNode** | One dialogue turn: **A** (summary for screening) + **C** (content for retrieval) |
| **Bucket** | Dynamic cluster with a **Medoid** (representative node) |
| **Medoid** | The node closest to all others in its bucket — used for coarse search |
| **Cross-bucket Edge** | Soft link between related buckets — no physical duplication |
| **Conflict Resolution** | Detects contradictions, marks old info as stale (not deleted) |

## Architecture

```
Query → [Layer 1: Bucket coarse screen] → [In-bucket fine search]
         → [Graph associative expansion] → [Layer 2: Conflict resolution]
```

See [`ARCHITECTURE_DESIGN.md`](ARCHITECTURE_DESIGN.md) for the full design spec.

## API Reference

- [`MemoryRetrievalEngine`](docs/api/retrieval.md) — `ingest()`, `search()`, `resolve_conflicts()`
- [`BucketManager`](docs/api/bucket_manager.md) — bucket lifecycle and assignment
- [`VectorStore`](docs/api/vector_store.md) — embedding and similarity search
- [`GraphStore`](docs/api/graph_engine.md) — graph traversal and edge management

## Semantic Mode (v0.3.0)

Replace `NumpyVectorStore` with `LocalEmbeddingStore` to activate the full
two-layer semantic pipeline:

```python
from memory_system.local_embedding import LocalEmbeddingStore

engine = MemoryRetrievalEngineImpl(
    config=config,
    vector_store=LocalEmbeddingStore(dim=384),  # all-MiniLM-L6-v2, semantic
    graph_store=NetworkXGraphStore(),
    llm=DeepSeekAdapter(),
)
# is_semantic() == True → _semantic_search() activates
# Layer 1: Medoid screening + graph expansion
# Layer 2: LLM conflict resolution + stale marking
```

## Benchmarks (v7 — LLM-as-Judge on LoCoMo)

| System | LoCoMo (LLM-as-Judge) |
|--------|----------------------|
| Full-Context (upper bound) | 87.5% |
| MIRIX | 85.4% |
| **m-memory** | **100%** (48/48 sampled) |
| Zep | 75.1% |
| Mem0 | 66.9% |
| A-Mem | 48.4% |

> ⚠️ 48 questions sampled from 1 conversation; full 1,986-question run pending.
> DeepSeek-chat judge (SOTA uses GPT-4o judge).

## Production LLM

Replace `FakeLLMAdapter` with the DeepSeek adapter:

```python
from memory_system.deepseek_llm import DeepSeekAdapter
from memory_system.embedding_store import OpenAIEmbeddingStore

# Keys from environment variables (never hardcode)
llm = DeepSeekAdapter()           # reads DEEPSEEK_API_KEY
store = OpenAIEmbeddingStore()    # reads OPENAI_API_KEY, real embeddings

engine = MemoryRetrievalEngineImpl(
    config=config,
    vector_store=store,        # semantic search
    graph_store=NetworkXGraphStore(),
    llm=llm,
)
```

When `vector_store.is_semantic() == False` (e.g., NumpyVectorStore),
the engine automatically falls back to lexical keyword search.

## Research Harness

This project includes a **research testing protocol** for academic experiments.
Before running any Agent-driven tests, **read these files in order**:

| # | File | Purpose |
|---|------|---------|
| 1 | [`README.md`](README.md) | Project overview + API (you are here) |
| 2 | [`AI_Memory_System_Testing_Protocol.md`](AI_Memory_System_Testing_Protocol.md) | Harness engineering specs + anti-AIGC rules |

> ⚠️ **Agent Constraint**: Any AI agent (including Reasonix Code) MUST read
> `README.md` before writing code or running tests. See Protocol §0.2.

## Development

```bash
# Clone and install with dev dependencies
git clone https://github.com/RobinElysia/M-Memory.git
cd M-Memory
pip install -e ".[dev]"

# Run tests
pytest tests/

# Type check
mypy --strict memory_system/

# Lint
ruff check memory_system/
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full development harness guide.

## License

MIT — see `pyproject.toml`.
