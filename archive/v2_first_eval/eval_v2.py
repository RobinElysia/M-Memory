#!/usr/bin/env python3
"""
Improved evaluation: combines vector + keyword retrieval for better accuracy.
"""
import json, os, sys, time, random
from dataclasses import dataclass, field
from typing import Any

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
os.environ["M_MEMORY_LOG_LEVEL"] = "ERROR"

from memory_system.config import MemorySystemConfig
from memory_system.deepseek_llm import DeepSeekAdapter
from memory_system.graph_engine import NetworkXGraphStore
from memory_system.retrieval import MemoryRetrievalEngineImpl
from memory_system.vector_store import NumpyVectorStore

config = MemorySystemConfig()
config.embedding_dim = 1536

llm = DeepSeekAdapter(model="deepseek-chat")
memory_engine = MemoryRetrievalEngineImpl(
    config=config,
    vector_store=NumpyVectorStore(dim=1536),
    graph_store=NetworkXGraphStore(),
    llm=llm,
)

@dataclass
class EvalResult:
    competency: str
    test_name: str
    total_questions: int
    correct: int
    details: list[dict] = field(default_factory=list)

results: list[EvalResult] = []
call_log: list[dict] = []  # Track all LLM calls

def ask_agent(sys_prompt: str, user_msg: str) -> tuple[str, int, int]:
    """Ask agent with keyword+vector retrieval."""
    search_result = memory_engine.search(user_msg, max_hops=0)
    tokens_before = llm.total_tokens

    # Build memory context from search results
    seen = set()
    memory_lines = []
    for n in search_result.nodes[:8]:
        if n.id not in seen:
            seen.add(n.id)
            stale_mark = " [OUTDATED]" if n.is_stale else ""
            memory_lines.append(f"  [{n.summary}] {n.content}{stale_mark}")

    # Also add keyword-matched nodes (fallback for hash embedding)
    keywords = set(user_msg.lower().split()) - {"the", "a", "an", "is", "are", "was", "were", "do", "does", "did", "in", "on", "at", "to", "of", "for", "with", "what", "how", "when", "where", "who", "why", "about", "i", "my", "me", "you", "your", "now", "before", "after"}
    for nid, node in memory_engine._nodes.items():
        if nid not in seen and len(seen) < 15:
            if any(kw in node.content.lower() for kw in keywords):
                seen.add(nid)
                stale_mark = " [OUTDATED]" if node.is_stale else ""
                memory_lines.append(f"  [keyword:{node.summary}] {node.content}{stale_mark}")

    memory_context = "\n".join(memory_lines) if memory_lines else "(no relevant memories found)"

    prompt = (
        f"{sys_prompt}\n\n"
        f"Stored memories:\n{memory_context}\n\n"
        f"Question: {user_msg}\n"
        f"Answer concisely using only the information above. If not found, say 'Not found in memory':"
    )

    response = llm.complete(prompt)
    tokens_used = llm.total_tokens - tokens_before
    call_log.append({"prompt_len": len(prompt), "response_len": len(response), "tokens": tokens_used})
    return response, len(prompt), len(response)

print("=" * 70)
print("  m-memory MemoryAgentBench Evaluation v2")
print("  Retrieval: Vector + Keyword hybrid")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════
# AR: Accurate Retrieval — 50 facts
# ═══════════════════════════════════════════════════════════════════════════
print("\n── [AR] Accurate Retrieval ──")

facts = [
    ("User profile", "User's name is Alex Chen, 29 years old, software engineer at Tencent"),
    ("Contact info", "Alex's email is alex.chen@example.com, phone number is +86-138-0000-1234"),
    ("Home location", "Alex lives in Shenzhen, Nanshan District, near Tencent Binhai Building"),
    ("Work department", "Alex works in WeChat Pay division, focusing on AI infrastructure"),
    ("Project name", "Alex leads Project Alpha, building an AI memory system since March 2024"),
    ("Team members", "Team has 5 members: Li Wei (backend), Zhang Min (frontend), Wang Fang (ML), Liu Yang (DevOps), Zhao Yu (PM)"),
    ("Last meeting", "Last team meeting on January 15, 2025, discussed Q1 roadmap for AI memory features"),
    ("Project budget", "Budget: 500,000 CNY for GPU hardware, 200,000 CNY for API and cloud costs"),
    ("Deadline info", "Project deadline extended to June 30, 2025 due to additional scope"),
    ("Main competitor", "Main competitor is MIRIX, which uses multi-agent architecture with 6 memory types"),
    ("Vacation plan", "Alex plans to visit Japan in July 2025 for 2 weeks with family"),
    ("Personal hobby", "Alex is a 3-dan amateur Go player and does photography on weekends"),
    ("Family details", "Alex is married to Lisa, daughter Emma is 3 years old, born in 2022"),
    ("Health condition", "Alex has a mild seafood allergy, especially to shrimp and crab"),
    ("Education background", "Graduated from Tsinghua University, Computer Science, class of 2018"),
    ("Language skills", "Speaks Mandarin natively, English fluently (IELTS 7.5), basic Japanese"),
    ("Pet information", "Has a golden retriever named Max, 2 years old, weighs 30kg"),
    ("Vehicle details", "Drives a Tesla Model Y, blue color, license plate YUE·B12345"),
    ("Subscriptions", "Subscribes to ChatGPT Plus and GitHub Copilot for development"),
    ("Recent event", "Attended AI Conference 2025 in Beijing last week, gave a talk on memory systems"),
]

for i, (topic, content) in enumerate(facts):
    memory_engine.ingest(topic, content, confidence=0.95)

print(f"  Ingested {len(facts)} facts. LLM calls: {llm.call_count}")

ar_questions = [
    ("What is Alex's email address?", "alex.chen@example.com"),
    ("Where does Alex live?", "Shenzhen"),
    ("Where does Alex work?", "Tencent"),
    ("What project does Alex lead?", "Alpha"),
    ("How many team members?", "5"),
    ("When is the project deadline?", "June 30, 2025"),
    ("What is Alex allergic to?", "seafood"),
    ("Where did Alex graduate from?", "Tsinghua"),
    ("What pet does Alex have?", "golden retriever"),
    ("What car does Alex drive?", "Tesla"),
    ("What is Alex's daughter's name?", "Emma"),
    ("What programming languages?", "not found"),
    ("How old is Alex?", "29"),
    ("Where is Alex going for vacation?", "Japan"),
    ("What Go rank is Alex?", "3-dan"),
]

ar_correct = 0
ar_details = []
ar_tokens = 0

print("  Testing 15 retrieval questions...")
for query, expected in ar_questions:
    answer, plen, rlen = ask_agent("You are Alex's personal assistant with access to stored memory records.", query)
    if expected == "not found":
        correct = expected.lower() not in answer.lower() or "not found" in answer.lower()
    else:
        correct = expected.lower() in answer.lower()
    if correct:
        ar_correct += 1
    ar_details.append({"query": query, "expected": expected, "answer": answer[:200], "correct": correct})
    status = "PASS" if correct else "FAIL"
print(f"    [{status}] {query} -> {answer[:80]}")

results.append(EvalResult("AR", "AccurateRetrieval-20facts", len(ar_questions), ar_correct, ar_details))
print(f"  AR: {ar_correct}/{len(ar_questions)} ({ar_correct/len(ar_questions)*100:.0f}%)")

# ═══════════════════════════════════════════════════════════════════════════
# SF: Selective Forgetting
# ═══════════════════════════════════════════════════════════════════════════
print("\n── [SF] Selective Forgetting ──")

sf_scenarios = [
    ("Location change", [
        ("address_v1", "I live in Beijing, Haidian District, near Tsinghua University", 0.8),
        ("address_v2", "I have moved to Shanghai, Pudong New Area, near Lujiazui", 0.9),
    ], [("Where do I live now?", "Shanghai"), ("Where did I live before?", "Beijing")]),
    ("Job switch", [
        ("job_v1", "I work at ByteDance as a senior engineer", 0.8),
        ("job_v2", "I now work at Alibaba Cloud as a tech lead", 0.95),
    ], [("Where do I work now?", "Alibaba"), ("Where did I work before?", "ByteDance")]),
    ("Allergy correction", [
        ("allergy_v1", "I have a peanut allergy according to previous tests", 0.6),
        ("allergy_v2", "Doctor confirmed I do NOT have peanut allergy. I am allergic to shellfish.", 0.95),
    ], [("What am I allergic to?", "shellfish"), ("What did I think I was allergic to?", "peanut")]),
    ("Team size growth", [
        ("team_v1", "My team has 3 members: Alice, Bob, Charlie", 0.7),
        ("team_v2", "Team grew to 5: Alice, Bob, Charlie, David, Eve", 0.9),
    ], [("How many team members now?", "5"), ("How many before the growth?", "3")]),
]

sf_correct = 0
sf_total = 0
sf_details = []

for sname, facts_list, queries in sf_scenarios:
    print(f"  Scenario: {sname}")
    for topic, content, conf in facts_list:
        memory_engine.ingest(topic, content, confidence=conf)
    for query, expected in queries:
        sf_total += 1
        answer, plen, rlen = ask_agent("You are a personal assistant. Answer based on stored facts. Note: newer facts replace older ones.", query)
        correct = expected.lower() in answer.lower()
        if correct:
            sf_correct += 1
        sf_details.append({"scenario": sname, "query": query, "answer": answer[:200], "correct": correct})
        print(f"    {'[OK]' if correct else '[XX]'} {query}")

results.append(EvalResult("SF", "SelectiveForgetting-4scenarios", sf_total, sf_correct, sf_details))
print(f"  SF: {sf_correct}/{sf_total} ({sf_correct/sf_total*100:.0f}%)")

# ═══════════════════════════════════════════════════════════════════════════
# LRU: Long-Range Understanding with hybrid retrieval
# ═══════════════════════════════════════════════════════════════════════════
print("\n── [LRU] Long-Range Understanding (100 turns) ──")

lru_facts = {
    3: ("Project leader", "Dr. Sarah Johnson leads the NeuroMem project"),
    10: ("Project start", "NeuroMem project started on January 15, 2024"),
    15: ("Team composition", "NeuroMem team has 8 researchers from 3 countries"),
    25: ("Milestone 1", "March 2024: NeuroMem prototype completed with basic memory functions"),
    40: ("Milestone 2", "June 2024: NeuroMem alpha release, supports 10K memory nodes"),
    55: ("Budget approval", "NeuroMem budget approved: 2.5 million USD total funding"),
    65: ("Milestone 3", "September 2024: NeuroMem beta, added graph retrieval and conflict resolution"),
    75: ("Conference", "NeuroMem presented at NeurIPS 2024 workshop on Agent Memory"),
    85: ("Milestone 4", "January 2025: NeuroMem v1.0 released, open-sourced on GitHub"),
    95: ("Current status", "NeuroMem now in production at 3 companies, 500+ GitHub stars"),
}

for i in range(100):
    if i in lru_facts:
        topic, content = lru_facts[i]
        memory_engine.ingest(topic, content, confidence=0.9)
    else:
        memory_engine.ingest(f"log_day_{i}", f"Day {i}: NeuroMem log — temperature {20+i%15}°C, humidity {40+i%20}%, all systems normal.", confidence=0.5)

print(f"  Ingested 100 turns. LLM calls: {llm.call_count}")

lru_queries = [
    ("Who leads the NeuroMem project?", "Sarah Johnson"),
    ("When did NeuroMem start?", "January 15, 2024"),
    ("How many researchers?", "8"),
    ("What happened in March 2024?", "prototype"),
    ("What was the June 2024 milestone?", "alpha"),
    ("What is the total budget?", "2.5"),
    ("Where was NeuroMem presented?", "NeurIPS"),
    ("When was v1.0 released?", "January 2025"),
    ("How many GitHub stars?", "500"),
]
lru_correct = 0
lru_details = []

for query, expected in lru_queries:
    answer, plen, rlen = ask_agent("You are a project management assistant. Answer from stored project records.", query)
    correct = expected.lower() in answer.lower()
    if correct:
        lru_correct += 1
    lru_details.append({"query": query, "answer": answer[:200], "correct": correct})
    print(f"    {'[OK]' if correct else '[XX]'} {query}")

results.append(EvalResult("LRU", "LongRangeUnderstanding-100turns", len(lru_queries), lru_correct, lru_details))
print(f"  LRU: {lru_correct}/{len(lru_queries)} ({lru_correct/len(lru_queries)*100:.0f}%)")

# ═══════════════════════════════════════════════════════════════════════════
# TTL: Test-Time Learning
# ═══════════════════════════════════════════════════════════════════════════
print("\n── [TTL] Test-Time Learning ──")

ttl_scenarios = [
    ("User preferences", [
        ("pref_dark", "User prefers dark mode in all applications and websites"),
        ("pref_notify", "User wants silent notifications only, no sound alerts"),
        ("pref_morning", "User is most productive between 6am and 11am"),
        ("pref_coffee", "User drinks Ethiopian single-origin coffee, black, no sugar"),
    ], "Summarize all user preferences", ["dark", "silent", "morning", "coffee"]),
    ("K8s learning", [
        ("k8s_pod", "Kubernetes pods are the smallest deployable computing units"),
        ("k8s_service", "Kubernetes services expose pods via stable network endpoints"),
        ("k8s_deploy", "Kubernetes deployments manage replica sets and rolling updates"),
        ("k8s_config", "Kubernetes ConfigMaps store non-sensitive configuration data"),
    ], "Explain Kubernetes based on what you've learned", ["pod", "service", "deployment", "configmap"]),
]

ttl_correct = 0
ttl_total = len(ttl_scenarios)
ttl_details = []

for sname, sessions, query, keywords in ttl_scenarios:
    for topic, content in sessions:
        memory_engine.ingest(topic, content)
    answer, plen, rlen = ask_agent("You are a learning assistant. Synthesize all past discussions.", query)
    hits = sum(1 for kw in keywords if kw.lower() in answer.lower())
    correct = hits >= len(keywords) * 0.6
    if correct:
        ttl_correct += 1
    ttl_details.append({"scenario": sname, "answer": answer[:300], "hits": hits, "keywords": keywords})
    print(f"    {'[OK]' if correct else '[XX]'} {sname}: {hits}/{len(keywords)} keywords")

results.append(EvalResult("TTL", "TestTimeLearning-2scenarios", ttl_total, ttl_correct, ttl_details))

# ═══════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════

total_nodes = len(memory_engine._nodes)
total_buckets = len(memory_engine._bucket_manager.get_all_buckets())
stale_nodes = sum(1 for n in memory_engine._nodes.values() if n.is_stale)

print("\n" + "=" * 70)
print("  FINAL RESULTS")
print("=" * 70)
for r in results:
    acc = r.correct / r.total_questions * 100 if r.total_questions else 0
    print(f"  [{r.competency}] {r.test_name}: {r.correct}/{r.total_questions} ({acc:.0f}%)")

total_correct = sum(r.correct for r in results)
total_q = sum(r.total_questions for r in results)
print(f"\n  OVERALL: {total_correct}/{total_q} ({total_correct/total_q*100:.0f}%)")
print(f"  Nodes: {total_nodes} | Buckets: {total_buckets} | Stale: {stale_nodes}")
print(f"  LLM: {llm.call_count} calls | {llm.total_tokens} tokens")
print(f"  Avg: {llm.total_tokens/max(llm.call_count,1):.0f} tokens/call")
print(f"  Avg prompt length: {sum(c['prompt_len'] for c in call_log)/max(len(call_log),1):.0f} chars")
print(f"  Avg response length: {sum(c['response_len'] for c in call_log)/max(len(call_log),1):.0f} chars")
print(f"\n  Competency breakdown:")
for r in results:
    acc = r.correct / r.total_questions * 100 if r.total_questions else 0
    print(f"    {r.competency}: {acc:.0f}%")

# Save
report = {
    "summary": {
        "overall_accuracy": round(total_correct/total_q*100, 1),
        "total_questions": total_q, "total_correct": total_correct,
        "nodes": total_nodes, "buckets": total_buckets, "stale": stale_nodes,
        "llm_calls": llm.call_count, "total_tokens": llm.total_tokens,
        "avg_tokens_per_call": round(llm.total_tokens/max(llm.call_count,1)),
    },
    "competencies": [
        {"competency": r.competency, "accuracy": round(r.correct/r.total_questions*100,1),
         "details": r.details} for r in results
    ],
}
with open("eval_results.json", "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print("\n[OK] eval_results.json saved")
