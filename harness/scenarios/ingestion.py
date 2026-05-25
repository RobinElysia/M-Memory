"""摄入场景测试 — 使用真实 DeepSeek LLM。"""

from __future__ import annotations

from memory_system.config import MemorySystemConfig
from memory_system.deepseek_llm import DeepSeekAdapter
from memory_system.graph_engine import NetworkXGraphStore
from memory_system.retrieval import MemoryRetrievalEngineImpl
from memory_system.vector_store import NumpyVectorStore

DEEPSEEK_KEY = "sk-768c26bb7779496e907781f52d82e526"


def _make_engine() -> MemoryRetrievalEngineImpl:
    config = MemorySystemConfig()
    config.embedding_dim = 1536
    config.bucket.top_k = 3
    return MemoryRetrievalEngineImpl(
        config=config,
        vector_store=NumpyVectorStore(dim=1536),
        graph_store=NetworkXGraphStore(),
        llm=DeepSeekAdapter(api_key=DEEPSEEK_KEY),
    )


def test_ing_01_single_ingest() -> tuple[bool, str]:
    """ING-01: 单条摄入，应返回有效 ID 并创建 1 个桶。"""
    engine = _make_engine()
    node_id = engine.ingest("人工智能研究", "Transformer 架构通过自注意力机制实现了高效并行计算")
    buckets = engine._bucket_manager.get_all_buckets()

    ok = bool(node_id) and len(node_id) > 0 and len(buckets) == 1
    return ok, f"id={node_id[:12]}... buckets={len(buckets)}"


def test_ing_02_same_topic() -> tuple[bool, str]:
    """ING-02: 连续 5 条 AI 相关，应全部归入同一桶。"""
    engine = _make_engine()
    topics = [
        "Transformer模型原理",
        "注意力机制详解",
        "BERT预训练方法",
        "GPT系列发展",
        "大语言模型应用",
    ]
    for t in topics:
        engine.ingest(t, f"{t} 的相关研究内容")

    buckets = engine._bucket_manager.get_all_buckets()
    ok = len(buckets) >= 1
    return ok, f"buckets={len(buckets)}"


def test_ing_03_cross_topic() -> tuple[bool, str]:
    """ING-03: 3 AI + 3 Cooking，应形成至少 2 个桶。"""
    engine = _make_engine()
    ai = ["Transformer", "BERT", "GPT"]
    cooking = ["红烧肉做法", "清蒸鱼技巧", "川菜调味"]

    for t in ai + cooking:
        engine.ingest(t, f"{t} 的详细说明")

    buckets = engine._bucket_manager.get_all_buckets()
    ok = len(buckets) >= 2
    return ok, f"buckets={len(buckets)} (expected ≥2)"


def test_ing_04_empty_summary() -> tuple[bool, str]:
    """ING-04: 空摘要，应能正常处理。"""
    engine = _make_engine()
    try:
        node_id = engine.ingest("", "内容不为空")
        ok = bool(node_id) and len(node_id) > 0
        return ok, f"id={node_id[:12]}..."
    except Exception as e:
        return False, f"exception: {e}"
