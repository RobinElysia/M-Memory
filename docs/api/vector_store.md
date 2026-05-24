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
```

| Method | Description |
|--------|-------------|
| `embed(text)` | Deterministic hash-based embedding → normalized float32 vector |
| `add(vectors, metadata)` | Batch insert; each metadata dict must have `"id"` key |
| `search(query_vector, top_k)` | Cosine similarity search, returns `[(id, score), ...]` |
| `remove(ids)` | Delete vectors by stored ID |
| `count()` | Total vectors in the index |

## Usage

```python
store = NumpyVectorStore(dim=8)

# Embed text
vec = store.embed("hello world")  # shape (8,), dtype float32

# Add vectors
ids = store.add(
    vectors=np.array([vec]),
    metadata=[{"id": "node-1", "extra": "data"}],
)

# Search
results = store.search(query_vector=vec, top_k=3)
# → [("node-1", 1.0), ...]

# Remove
store.remove(["node-1"])
assert store.count() == 0
```
