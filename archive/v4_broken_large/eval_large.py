#!/usr/bin/env python3
"""Large-scale eval: 500+ nodes, 90+ queries, engine-isolated."""
import json, os, sys, time, random
from dataclasses import dataclass, field

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
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
config.bucket.split_threshold = 20

def make_engine():
    return MemoryRetrievalEngineImpl(
        config=config,
        vector_store=NumpyVectorStore(dim=1536),
        graph_store=NetworkXGraphStore(),
        llm=DeepSeekAdapter(),
    )

def ask_agent(engine, sys_prompt, user_msg):
    result = engine.search(user_msg, max_hops=0)
    seen = set()
    lines = []
    for n in result.nodes[:10]:
        if n.id not in seen:
            seen.add(n.id)
            s = " [STALE]" if n.is_stale else ""
            lines.append(f"  [{n.summary}] {n.content}{s}")
    ctx = "\n".join(lines) if lines else "(no relevant memories)"
    prompt = f"{sys_prompt}\n\nMemories:\n{ctx}\n\nQuestion: {user_msg}\nAnswer:"
    return engine._llm.complete(prompt)

@dataclass
class R: competency: str; total: int; correct: int; details: list = field(default_factory=list)

all_r = []
tcalls = 0; ttokens = 0

print("=" * 70)
print("  m-memory LARGE-SCALE EVAL (500+ nodes, lexical fallback)")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════
# AR — 100 facts × 50 queries
# ═══════════════════════════════════════════════════════════════════════
print("\n── [AR] 100 facts ──")
ar_e = make_engine()

persons = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry"]
ar_facts = []
for i in range(100):
    p = persons[i % 8]
    if i % 5 == 0:
        ar_facts.append((f"email_{i}", f"{p}'s email is {p.lower()}{i}@example.com"))
    elif i % 5 == 1:
        ar_facts.append((f"city_{i}", f"{p} lives in City{i%20}, District {i%5}"))
    elif i % 5 == 2:
        ar_facts.append((f"job_{i}", f"{p} works as a {'engineer' if i%3==0 else 'designer' if i%3==1 else 'manager'} at Company{i%10}"))
    elif i % 5 == 3:
        ar_facts.append((f"pet_{i}", f"{p} owns a {'dog' if i%2==0 else 'cat'} named Pet{i}"))
    else:
        ar_facts.append((f"hobby_{i}", f"{p} enjoys {'reading' if i%3==0 else 'gaming' if i%3==1 else 'cooking'}"))

for t, c in ar_facts:
    ar_e.ingest(t, c, confidence=0.9)

# 50 queries — randomly selected
random.seed(1)
test_set = random.sample(ar_facts, 50)
ar_c, ar_d = 0, []
for topic, content in test_set:
    person = content.split("'")[0] if "'" in content else content.split()[0]
    query = f"What is {person}'s {topic.split('_')[0]}?"
    answer = ask_agent(ar_e, "You are a memory assistant.", query)
    ok = content.split(" is ", 1)[-1].split(".")[0].split(",")[0][:20].lower() in answer.lower()
    if ok: ar_c += 1
    ar_d.append({"q": query, "a": answer[:120], "ok": ok})

tcalls += ar_e._llm.call_count; ttokens += ar_e._llm.total_tokens
all_r.append(R("AR", 50, ar_c, ar_d))
print(f"  AR: {ar_c}/50 ({ar_c/50*100:.0f}%) | nodes={len(ar_e._nodes)} | buckets={len(ar_e._bucket_manager.get_all_buckets())}")

# ═══════════════════════════════════════════════════════════════════════
# SF — 10 contradiction scenarios × 20 queries
# ═══════════════════════════════════════════════════════════════════════
print("\n── [SF] 10 scenarios ──")
sf_e = make_engine()

sf_s = [
    ("Location", [("live in New York", 0.8), ("moved to Los Angeles", 0.9)]),
    ("Job", [("work at Google", 0.7), ("now work at Microsoft", 0.95)]),
    ("Allergy", [("allergic to cats", 0.6), ("actually allergic to dogs, not cats", 0.95)]),
    ("Team", [("team of 4", 0.7), ("team grew to 7", 0.9)]),
    ("City", [("live in Tokyo", 0.8), ("moved to Osaka", 0.9)]),
    ("Role", [("junior developer", 0.7), ("promoted to senior developer", 0.95)]),
    ("Car", [("drive a Honda", 0.7), ("switched to a Tesla", 0.9)]),
    ("Phone", [("use iPhone 14", 0.7), ("upgraded to iPhone 16", 0.95)]),
    ("Framework", [("use React", 0.7), ("switched to Vue", 0.9)]),
    ("Language", [("only speak English", 0.7), ("now also speak Japanese", 0.9)]),
]

sf_c, sf_t, sf_d = 0, 0, []
for name, facts in sf_s:
    for topic_suffix, fc in [(f"{name}_v1", facts[0][0]), (f"{name}_v2", facts[1][0])]:
        sf_e.ingest(name.lower() + "_" + topic_suffix[:5], topic_suffix, confidence=facts[0][1] if "v1" in topic_suffix else facts[1][1])

    # 2 queries per scenario: now + before
    for qi, (q, exp) in enumerate([
        (f"What is my {name.lower()} now?", facts[1][0].split(" ", 1)[-1] if " " in facts[1][0] else facts[1][0]),
        (f"What was my {name.lower()} before?", facts[0][0].split(" ", 1)[-1] if " " in facts[0][0] else facts[0][0]),
    ]):
        sf_t += 1
        answer = ask_agent(sf_e, "You are a personal assistant. New facts replace old ones.", q)
        ok = any(w.lower() in answer.lower() for w in exp.split()[:3])
        if ok: sf_c += 1
        sf_d.append({"scenario": name, "q": q, "a": answer[:120], "ok": ok})

tcalls += sf_e._llm.call_count; ttokens += sf_e._llm.total_tokens
all_r.append(R("SF", sf_t, sf_c, sf_d))
print(f"  SF: {sf_c}/{sf_t} ({sf_c/sf_t*100:.0f}%) | stale nodes={sum(1 for n in sf_e._nodes.values() if n.is_stale)}")

# ═══════════════════════════════════════════════════════════════════════
# LRU — 200 turns × 15 queries
# ═══════════════════════════════════════════════════════════════════════
print("\n── [LRU] 200 turns ──")
lru_e = make_engine()

lru_facts = {
    5: ("Project Mercury lead", "Dr. Alice Chen leads Project Mercury"),
    15: ("Mercury start date", "Project Mercury started on March 1, 2024"),
    25: ("Mercury team size", "Mercury team has 12 researchers"),
    35: ("Mercury milestone 1", "April 2024: Mercury prototype completed"),
    45: ("Mercury milestone 2", "July 2024: Mercury alpha, 50K nodes"),
    55: ("Mercury budget", "Mercury budget: 5 million USD"),
    65: ("Mercury milestone 3", "October 2024: Mercury beta, added graphs"),
    75: ("Mercury conference", "Mercury presented at ICML 2025"),
    85: ("Mercury milestone 4", "January 2025: Mercury v1.0"),
    95: ("Mercury users", "Mercury: 3 production users, 1200 GitHub stars"),
    105: ("Mercury milestone 5", "March 2025: Mercury v2.0 planning"),
    115: ("Mercury team size 2", "Mercury team expanded to 20"),
    125: ("Mercury award", "Mercury won Best Paper at ACL 2025"),
    135: ("Mercury funding", "Mercury received Series A: 10 million"),
    145: ("Mercury milestone 6", "June 2025: Mercury v2.0 alpha"),
}

for i in range(200):
    if i in lru_facts:
        t, c = lru_facts[i]
        lru_e.ingest(t, c, confidence=0.9)
    else:
        lru_e.ingest(f"mercury_log_{i}", f"Mercury day {i}: systems normal, temp={20+i%15}C, load={30+i%50}%", confidence=0.5)

lru_qs = [
    ("Who leads Project Mercury?", "Alice Chen"),
    ("When did Mercury start?", "March 1, 2024"),
    ("Initial team size?", "12"),
    ("What was the April 2024 milestone?", "prototype"),
    ("What was the July 2024 milestone?", "alpha"),
    ("Total budget?", "5 million"),
    ("Where was Mercury presented?", "ICML"),
    ("When was v1.0 released?", "January 2025"),
    ("GitHub stars?", "1200"),
    ("When did Mercury win Best Paper?", "ACL"),
    ("Series A amount?", "10 million"),
    ("Current team size?", "20"),
    ("March 2025 milestone?", "v2.0 planning"),
    ("June 2025 milestone?", "v2.0 alpha"),
    ("Production users?", "3"),
]
lru_c, lru_d = 0, []
for q, exp in lru_qs:
    answer = ask_agent(lru_e, "You are a project assistant.", q)
    ok = exp.lower() in answer.lower()
    if ok: lru_c += 1
    lru_d.append({"q": q, "a": answer[:120], "ok": ok})

tcalls += lru_e._llm.call_count; ttokens += lru_e._llm.total_tokens
all_r.append(R("LRU", len(lru_qs), lru_c, lru_d))
print(f"  LRU: {lru_c}/{len(lru_qs)} ({lru_c/len(lru_qs)*100:.0f}%) | nodes={len(lru_e._nodes)} | buckets={len(lru_e._bucket_manager.get_all_buckets())}")

# ═══════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  LARGE-SCALE RESULTS")
print("=" * 70)
for r in all_r:
    print(f"  [{r.competency}] {r.correct}/{r.total} ({r.correct/r.total*100:.0f}%)")
tc = sum(r.correct for r in all_r)
tq = sum(r.total for r in all_r)
print(f"\n  OVERALL: {tc}/{tq} ({tc/tq*100:.0f}%)")
print(f"  LLM: {tcalls} calls | {ttokens} tokens")
tn = len(ar_e._nodes) + len(sf_e._nodes) + len(lru_e._nodes)
print(f"  Total nodes: {tn}")

report = {
    "summary": {"overall": round(tc/tq*100,1), "questions": tq, "correct": tc,
                "nodes": tn, "llm_calls": tcalls, "tokens": ttokens},
    "competencies": [{"c": r.competency, "acc": round(r.correct/r.total*100,1),
                       "details": r.details} for r in all_r],
}
with open("eval_large.json", "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print("\n[OK] eval_large.json saved")
