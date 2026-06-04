"""CLI entry point for m-memory."""

import sys

from memory_system import __version__


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] in ("--help", "-h"):
        print(f"m-memory v{__version__}")
        print("Usage: m-memory [command]")
        print("Commands:")
        print("  version   Print version and exit")
        print("  test      Run quick self-test")
    elif args[0] == "version":
        print(f"m-memory v{__version__}")
    elif args[0] == "test":
        from memory_system.config import MemorySystemConfig
        from memory_system.fake_llm import FakeLLMAdapter
        from memory_system.graph_engine import NetworkXGraphStore
        from memory_system.retrieval import MemoryRetrievalEngineImpl
        from memory_system.vector_store import HashVectorStore

        config = MemorySystemConfig()
        config.embedding_dim = 1536
        engine = MemoryRetrievalEngineImpl(
            config=config,
            vector_store=HashVectorStore(dim=1536),
            graph_store=NetworkXGraphStore(),
            llm=FakeLLMAdapter(),
        )
        _ = engine.ingest("test fact", "The sky is blue", confidence=0.9)
        result = engine.search("sky")
        found = any("blue" in n.content for n in result.nodes)
        if found:
            print("Self-test PASSED")
        else:
            print("Self-test FAILED — check logs")
            sys.exit(1)
    else:
        print(f"Unknown command: {args[0]}")
        sys.exit(1)
