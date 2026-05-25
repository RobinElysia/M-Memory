# API 参考（自动生成）

以下文档从 Python 源码 docstring 自动生成，始终与代码同步。

## 核心接口

::: memory_system.interfaces.VectorStore
    options:
      members:
        - embed
        - add
        - search
        - remove
        - count

::: memory_system.interfaces.LLMAdapter
    options:
      members:
        - complete
        - chat

::: memory_system.interfaces.GraphStore
    options:
      members:
        - add_node
        - add_edge
        - traverse
        - get_out_edges
        - remove_edge

::: memory_system.interfaces.BucketManager
    options:
      members:
        - find_candidates
        - assign_to_bucket
        - create_bucket
        - split_bucket
        - dormancy_check
        - wake_bucket

::: memory_system.interfaces.MemoryRetrievalEngine
    options:
      members:
        - search
        - resolve_conflicts
        - ingest

## 实现类

::: memory_system.retrieval.MemoryRetrievalEngineImpl
    options:
      members:
        - ingest
        - search
        - resolve_conflicts

::: memory_system.bucket_manager.BucketManagerImpl

::: memory_system.graph_engine.NetworkXGraphStore

::: memory_system.vector_store.NumpyVectorStore

::: memory_system.cleanup.CleanupScheduler

## 数据模型

::: memory_system.models.MemoryNode

::: memory_system.models.Bucket

::: memory_system.models.Medoid

::: memory_system.models.Edge

::: memory_system.models.SearchResult

## 工具函数

::: memory_system.llm_decision
    options:
      members:
        - build_bucket_assignment_prompt
        - build_conflict_detection_prompt
        - parse_bucket_assignment_response
        - parse_conflict_detection_response

::: memory_system.fake_llm
    options:
      members:
        - FakeLLMAdapter
        - create_assignment_decision
        - create_conflict_response
