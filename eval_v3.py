#!/usr/bin/env python3
"""
MemoryAgentBench evaluation v3 — engine-isolated, lexical fallback.
Each competency test creates its own engine instance to prevent cross-contamination.
Uses NumpyVectorStore (hash-based) → triggers new _lexical_search() fallback.
"""
import json, os, sys, time
from dataclasses import dataclass, field
from typing import Any

os.environ["DEEPSEEK_API_KEY"] = "sk-768c26bb7779496e907781f52d82e526"
os.environ["M_MEMORY_LOG_LEVEL"] = "ERROR"

from memory_system.config import MemorySystemConfig
from memory_system.deepseek_llm import DeepSeekAdapter
from memory_system.graph_engine import NetworkXGraphStore
from memory_system.retrieval import MemoryRetrievalEngineImpl
from memory_system.vector_store import NumpyVectorStore

config = MemorySystemConfig()
config.embedding_dim = 1536
config.bucket.top_m = 5
config.bucket.top_p = 15


def make_engine() -> MemoryRetrievalEngineImpl:
    return MemoryRetrievalEngineImpl(
        config=config,
        vector_store=NumpyVectorStore(dim=1536),
        graph_store=NetworkXGraphStore(),
        llm=DeepSeekAdapter(),
    )


def ask_agent(engine, sys_prompt: str, user_msg: str) -> str:
    """Query engine with its built-in search (auto lexical fallback)."""
    result = engine.search(user_msg, max_hops=0)
    seen = set()
    memory_lines = []
    for n in result.nodes[:10]:
        if n.id not in seen:
            seen.add(n.id)
            stale = " [OUTDATED]" if n.is_stale else ""
            memory_lines.append(f"  [{n.summary}] {n.content}{stale}")

    context = "\n".join(memory_lines) if memory_lines else "(no relevant memories)"

    prompt = (
        f"{sys_prompt}\n\nStored memories:\n{context}\n\n"
        f"Question: {user_msg}\nAnswer concisely using only the info above:"
    )
    return engine._llm.complete(prompt)


@dataclass
class EvalResult:
    competency: str
    test_name: str
    total: int
    correct: int
    details: list[dict] = field(default_factory=list)


all_results: list[EvalResult] = []
total_llm_calls = 0
total_tokens = 0

print("=" * 70)
print("  m-memory v3 — Isolated Engine Evaluation")
print("  Retrieval: _lexical_search() fallback (hash embeddings)")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════
# AR: Accurate Retrieval
# ═══════════════════════════════════════════════════════════════════════════
print("\n── [AR] Accurate Retrieval (isolated engine) ──")
ar_engine = make_engine()

ar_facts = [
    ("Contact info", "Alex's email is alex.chen@example.com, phone +86-138-0000-1234"),
    ("Home location", "Alex lives in Shenzhen, Nanshan District, near Tencent Binhai Building"),
    ("Work department", "Alex works at Tencent, WeChat Pay division, AI infrastructure team"),
    ("Project name", "Alex leads Project Alpha, building an AI memory system since 2024-03"),
    ("Team members", "Team has 5 members: Li Wei, Zhang Min, Wang Fang, Liu Yang, Zhao Yu"),
    ("Last meeting", "Last team meeting on 2025-01-15, discussed Q1 roadmap"),
    ("Project budget", "Budget: 500K CNY for GPU, 200K for API costs"),
    ("Deadline info", "Project deadline extended to 2025-06-30"),
    ("Education", "Alex graduated from Tsinghua University, CS major, class of 2018"),
    ("Pet info", "Alex has a golden retriever named Max, 2 years old"),
    ("Vehicle info", "Alex drives a Tesla Model Y, license YUE·B12345"),
    ("Health info", "Alex has mild seafood allergy, especially shrimp and crab"),
    ("Family info", "Alex is married to Lisa, daughter Emma is 3 years old"),
    ("Vacation plan", "Alex plans to visit Japan in July 2025 for 2 weeks"),
    ("Language skills", "Alex speaks Mandarin natively, English fluently, basic Japanese"),
]

for topic, content in ar_facts:
    ar_engine.ingest(topic, content, confidence=0.95)

ar_queries = [
    ("What is Alex's email?", "alex.chen@example.com"),
    ("Where does Alex live?", "Shenzhen"),
    ("Where does Alex work?", "Tencent"),
    ("What project does Alex lead?", "Alpha"),
    ("How many team members?", "5"),
    ("When is the project deadline?", "2025-06-30"),
    ("Where did Alex graduate?", "Tsinghua"),
    ("What pet does Alex have?", "golden retriever"),
    ("What car does Alex drive?", "Tesla"),
    ("What is Alex allergic to?", "seafood"),
    ("What is Alex's daughter's name?", "Emma"),
    ("How old is Alex's dog?", "2"),
    ("Where is Alex going for vacation?", "Japan"),
    ("What languages does Alex speak?", "Mandarin"),
    ("When was the last team meeting?", "2025-01-15"),
]

ar_correct = 0
ar_details = []
for query, expected in ar_queries:
    answer = ask_agent(ar_engine, "You are a personal assistant with access to stored memory records.", query)
    ok = expected.lower() in answer.lower()
    if ok:
        ar_correct += 1
    ar_details.append({"query": query, "expected": expected, "answer": answer[:150], "ok": ok})
    print(f"  [{'PASS' if ok else 'FAIL'}] {query}")

total_llm_calls += ar_engine._llm.call_count
total_tokens += ar_engine._llm.total_tokens
all_results.append(EvalResult("AR", "AccurateRetrieval-15facts", len(ar_queries), ar_correct, ar_details))
print(f"  AR: {ar_correct}/{len(ar_queries)} ({ar_correct/len(ar_queries)*100:.0f}%)")

# ═══════════════════════════════════════════════════════════════════════════
# SF: Selective Forgetting (isolated engine)
# ═══════════════════════════════════════════════════════════════════════════
print("\n── [SF] Selective Forgetting (isolated engine) ──")
sf_engine = make_engine()

sf_scenarios = [
    ("Location change", [
        ("addr_v1", "I live in Beijing, Haidian District, near Tsinghua University", 0.8),
        ("addr_v2", "I have moved to Shanghai, Pudong New Area, near Lujiazui", 0.9),
    ], [("Where do I live now?", "Shanghai"), ("Where did I live before?", "Beijing")]),
    ("Job switch", [
        ("job_v1", "I work at ByteDance as a senior engineer", 0.8),
        ("job_v2", "I now work at Alibaba Cloud as a tech lead", 0.95),
    ], [("Where do I work now?", "Alibaba"), ("Where did I work before?", "ByteDance")]),
    ("Allergy correction", [
        ("allergy_v1", "I have a peanut allergy according to previous tests", 0.6),
        ("allergy_v2", "Doctor confirmed I do NOT have peanut allergy. I am allergic to shellfish.", 0.95),
    ], [("What am I allergic to?", "shellfish"), ("What did I previously think I was allergic to?", "peanut")]),
    ("Team size change", [
        ("team_v1", "My team has 3 members: Alice, Bob, Charlie", 0.7),
        ("team_v2", "Team grew to 5: Alice, Bob, Charlie, David, Eve", 0.9),
    ], [("How many team members now?", "5"), ("How many before the growth?", "3")]),
]

sf_correct = 0; sf_total = 0; sf_details = []
for sname, facts_list, queries in sf_scenarios:
    for topic, content, conf in facts_list:
        sf_engine.ingest(topic, content, confidence=conf)
    for query, expected in queries:
        sf_total += 1
        answer = ask_agent(sf_engine, "You are a personal assistant. Answer based on stored facts. Newer facts replace older ones.", query)
        ok = expected.lower() in answer.lower()
        if ok: sf_correct += 1
        sf_details.append({"scenario": sname, "query": query, "answer": answer[:150], "ok": ok})
        print(f"  [{'PASS' if ok else 'FAIL'}] {sname}: {query}")

total_llm_calls += sf_engine._llm.call_count
total_tokens += sf_engine._llm.total_tokens
all_results.append(EvalResult("SF", "SelectiveForgetting-4scenarios", sf_total, sf_correct, sf_details))
print(f"  SF: {sf_correct}/{sf_total} ({sf_correct/sf_total*100:.0f}%)")

# ═══════════════════════════════════════════════════════════════════════════
# LRU: Long-Range Understanding (isolated engine)
# ═══════════════════════════════════════════════════════════════════════════
print("\n── [LRU] Long-Range Understanding (isolated engine) ──")
lru_engine = make_engine()

lru_facts = {
    3: ("Project leader", "Dr. Sarah Johnson leads the NeuroMem project"),
    10: ("Project start", "NeuroMem started on January 15, 2024"),
    15: ("Team composition", "NeuroMem has 8 researchers from 3 countries"),
    25: ("Milestone 1", "March 2024: NeuroMem prototype completed"),
    40: ("Milestone 2", "June 2024: NeuroMem alpha release, supports 10K nodes"),
    55: ("Budget approval", "NeuroMem budget: 2.5 million USD total"),
    65: ("Milestone 3", "September 2024: NeuroMem beta, added graph retrieval"),
    75: ("Conference", "NeuroMem presented at NeurIPS 2024 workshop on Agent Memory"),
    85: ("Milestone 4", "January 2025: NeuroMem v1.0 released, open-sourced"),
    95: ("Current status", "NeuroMem in production at 3 companies, 500+ GitHub stars"),
}

for i in range(100):
    if i in lru_facts:
        topic, content = lru_facts[i]
        lru_engine.ingest(topic, content, confidence=0.9)
    else:
        lru_engine.ingest(f"log_{i}", f"Day {i}: routine log — temp {20+i%15}C, humidity {40+i%20}%, normal.", confidence=0.5)

lru_queries = [
    ("Who leads NeuroMem?", "Sarah Johnson"),
    ("When did NeuroMem start?", "January 15, 2024"),
    ("How many researchers?", "8"),
    ("What happened in March 2024?", "prototype"),
    ("What was the June 2024 milestone?", "alpha"),
    ("What is the total budget?", "2.5"),
    ("Where was NeuroMem presented?", "NeurIPS"),
    ("When was v1.0 released?", "January 2025"),
    ("How many GitHub stars?", "500"),
]
lru_correct = 0; lru_details = []
for query, expected in lru_queries:
    answer = ask_agent(lru_engine, "You are a project assistant. Answer from project records.", query)
    ok = expected.lower() in answer.lower()
    if ok: lru_correct += 1
    lru_details.append({"query": query, "answer": answer[:150], "ok": ok})
    print(f"  [{'PASS' if ok else 'FAIL'}] {query}")

total_llm_calls += lru_engine._llm.call_count
total_tokens += lru_engine._llm.total_tokens
all_results.append(EvalResult("LRU", "LongRangeUnderstanding-100turns", len(lru_queries), lru_correct, lru_details))
print(f"  LRU: {lru_correct}/{len(lru_queries)} ({lru_correct/len(lru_queries)*100:.0f}%)")

# ═══════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  FINAL RESULTS (engine-isolated + lexical fallback)")
print("=" * 70)
for r in all_results:
    acc = r.correct / r.total * 100 if r.total else 0
    print(f"  [{r.competency}] {r.test_name}: {r.correct}/{r.total} ({acc:.0f}%)")

tc = sum(r.correct for r in all_results)
tq = sum(r.total for r in all_results)
print(f"\n  OVERALL: {tc}/{tq} ({tc/tq*100:.0f}%)")
print(f"  LLM calls: {total_llm_calls} | Tokens: {total_tokens}")

report = {
    "summary": {"overall_accuracy": round(tc/tq*100,1), "total_questions": tq, "total_correct": tc,
                "llm_calls": total_llm_calls, "total_tokens": total_tokens},
    "competencies": [{"competency": r.competency, "accuracy": round(r.correct/r.total*100,1),
                       "details": r.details} for r in all_results],
}
with open("eval_results_v3.json", "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print("\n[OK] eval_results_v3.json saved")
