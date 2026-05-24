# Changelog

All notable changes to the m-memory project.

## [0.1.0] — Initial Release

### Added
- Dual-layer memory retrieval: bucket-level coarse screening + fine-grained semantic search
- Incremental clustering with LLM-driven bucket assignment
- Graph-based associative retrieval with cross-bucket edges (BFS traversal, best-single-path dedup)
- Conflict resolution pipeline: weighted re-ranking + LLM contradiction detection + stale marking
- Background cleanup scheduler: dormancy checks, contradiction scanning, topic-drift detection
- Deterministic Medoid computation (no randomness)
- Configurable parameters (`MemorySystemConfig`) with sensible defaults
- Abstract interfaces: `VectorStore`, `LLMAdapter`, `GraphStore`, `BucketManager`, `MemoryRetrievalEngine`
- NumPy-based `VectorStore` (default, hash-based deterministic embedding)
- NetworkX-based `GraphStore` (MultiDiGraph for parallel cross-bucket edges)
- `FakeLLMAdapter` for deterministic testing
- 58 unit tests + 3 E2E scenario tests (89% coverage)
- Full mypy strict + ruff lint compliance
