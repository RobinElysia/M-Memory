# VectorStore

Abstraction over vector embedding and similarity search backends.

Default implementation: `NumpyVectorStore` (in-memory, deterministic
hash-based embedding for testing).  Swap in FAISS, Chroma, or USearch
by implementing the same interface.

## Interface

```python
from memory_system.interfaces import VectorStore

class VectorStore(ABC):
    @abstractmethod
    def embed(self, text: str) -> NDArray[np.float32]: ...
    @abstractmethod
    def add(self, vectors: NDArray[np.float32], metadata: list[dict[str, Any]]) -> list[str]: ...
    @abstractmethod
    def search(self, query_vector: NDArray[np.float32], top_k: int) -> list[tuple[str, float]]: ...
    @abstractmethod
    def remove(self, ids: list[str]) -> None: ...
    @abstractmethod
    def count(self) -> int: ...
```

## Default: NumpyVectorStore

```python
from memory_system.vector_store import NumpyVectorStore

store = NumpyVectorStore(dim=1536)
assert store.is_semantic() == False  # hash-based, NOT semantic
```

> ⚠️ **NumpyVectorStore uses deterministic hash-based embeddings** — not semantic.
> For production, use `OpenAIEmbeddingStore` or implement your own.

| Method | Description |
|--------|-------------|
| `embed(text)` | Deterministic SHA-256 hash → normalized float32 vector |
| `add(vectors, metadata)` | Batch insert; each metadata dict must have `"id"` key |
| `search(query_vector, top_k)` | Cosine similarity search, returns `[(id, score), ...]` |
| `remove(ids)` | Delete vectors by stored ID |
| `count()` | Total vectors in the index |
| `is_semantic()` | Returns `False` — vectors are not semantically meaningful |

## Local Semantic: LocalEmbeddingStore (v0.3.0)

```python
from memory_system.local_embedding import LocalEmbeddingStore

store = LocalEmbeddingStore(model_name="all-MiniLM-L6-v2", dim=384)
assert store.is_semantic() == True  # real semantic embeddings, runs locally
```

Uses `sentence-transformers` with `all-MiniLM-L6-v2` (384-d, 80MB, CPU-friendly).
First call downloads the model; subsequent calls run locally at ~1ms per text.

When used with `MemoryRetrievalEngineImpl`, the engine automatically activates
`_semantic_search()` — full two-layer pipeline with Medoid screening, graph
expansion, and LLM conflict resolution.

## Cloud: OpenAIEmbeddingStore

```python
from memory_system.embedding_store import OpenAIEmbeddingStore
import os

store = OpenAIEmbeddingStore(
    api_key=os.environ["OPENAI_API_KEY"],
    model="text-embedding-3-small",
    dim=1536,
)
assert store.is_semantic() == True  # real semantic embeddings
```

| Method | Description |
|--------|-------------|
| `embed(text)` | Calls `/v1/embeddings` API → semantic float32 vector |
| `is_semantic()` | Returns `True` |
| Other methods | Same interface as NumpyVectorStore |

## Auto Fallback

`MemoryRetrievalEngine.search()` checks `vector_store.is_semantic()` at runtime:
- `False` → lexical keyword search (no API embedding calls)
- `True` → full vector-based semantic search (Layer 1 + Layer 2)

## Usage

```python
store = NumpyVectorStore(dim=8)

vec = store.embed("hello world")  # hash-based, shape (8,)

ids = store.add(
    vectors=np.array([vec]),
    metadata=[{"id": "node-1", "extra": "data"}],
)

results = store.search(query_vector=vec, top_k=3)

store.remove(["node-1"])
assert store.count() == 0
```
