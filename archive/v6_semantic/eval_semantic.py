#!/usr/bin/env python3
"""Semantic eval: LocalEmbeddingStore → _semantic_search() activates fully."""
import json, os, time, random
os.environ["DEEPSEEK_API_KEY"] = "sk-768c26bb7779496e907781f52d82e526"
os.environ["M_MEMORY_LOG_LEVEL"] = "ERROR"

from memory_system.config import MemorySystemConfig
from memory_system.deepseek_llm import DeepSeekAdapter
from memory_system.graph_engine import NetworkXGraphStore
from memory_system.retrieval import MemoryRetrievalEngineImpl
from memory_system.local_embedding import LocalEmbeddingStore

config = MemorySystemConfig()
config.embedding_dim = 384  # all-MiniLM-L6-v2
config.bucket.top_m = 5
config.bucket.top_p = 15
config.bucket.top_k = 3
config.graph.max_hops = 1  # enable graph expansion!

print("Loading LocalEmbeddingStore (all-MiniLM-L6-v2)...")
VECTOR_STORE = LocalEmbeddingStore(dim=384)
print("Model loaded.\n")

def make_engine():
    return MemoryRetrievalEngineImpl(
        config=config, vector_store=VECTOR_STORE,
        graph_store=NetworkXGraphStore(), llm=DeepSeekAdapter(),
    )

def ask(engine, sys_prompt, user_msg):
    result = engine.search(user_msg, max_hops=config.graph.max_hops)
    seen = set(); lines = []
    for n in result.nodes[:10]:
        if n.id not in seen:
            seen.add(n.id)
            s = " [STALE]" if n.is_stale else ""
            lines.append(f"  [{n.summary}] {n.content}{s}")
    ctx = "\n".join(lines) if lines else "(no relevant memories)"
    return engine._llm.complete(
        f"{sys_prompt}\n\nMemories:\n{ctx}\n\nQuestion: {user_msg}\nAnswer concisely:"
    )

def check(answer, expected_words, min_hits=1):
    al = answer.lower()
    return sum(1 for w in expected_words if w.lower() in al) >= min_hits

class R:
    def __init__(self,c,t): self.competency=c; self.total=t; self.correct=0; self.details=[]

all_r = []; tcalls = 0; ttokens = 0
random.seed(42)

print("=" * 70)
print("  SEMANTIC EVAL — LocalEmbeddingStore + _semantic_search()")
print("  is_semantic() =", VECTOR_STORE.is_semantic())
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════
# AR: 30 facts — test semantic retrieval
# ═══════════════════════════════════════════════════════════════════════════
print("\n── [AR] 30 facts (semantic) ──")
ar_e = make_engine()

ar_data = [
    ("Alice email", "Alice Wang's email is alice.wang@example.com",
     "What is Alice Wang's email address?", ["alice.wang@example.com"]),
    ("Alice Shenzhen home", "Alice lives in Shenzhen, Nanshan District",
     "Where does Alice reside?", ["shenzhen"]),
    ("Alice Tencent job", "Alice works as a senior engineer at Tencent",
     "What is Alice's occupation?", ["tencent", "engineer"]),
    ("Alice golden retriever", "Alice owns a golden retriever named Max",
     "Does Alice have any animals?", ["golden retriever", "max"]),
    ("Alice Tsinghua grad", "Alice graduated from Tsinghua University in 2018",
     "Which university did Alice attend?", ["tsinghua"]),

    ("Bob email", "Bob Chen's email is bob.chen@example.com",
     "What is Bob Chen's contact email?", ["bob.chen@example.com"]),
    ("Bob Beijing home", "Bob lives in Beijing, Haidian District",
     "In which city does Bob live?", ["beijing"]),
    ("Bob ByteDance job", "Bob works at ByteDance as a product manager",
     "What company employs Bob?", ["bytedance"]),
    ("Bob Siamese cat", "Bob owns a Siamese cat named Luna",
     "What kind of pet does Bob have?", ["siamese", "cat", "luna"]),
    ("Bob PKU grad", "Bob graduated from Peking University in 2019",
     "Where did Bob graduate from?", ["peking"]),

    ("Charlie email", "Charlie Li's email is charlie.li@example.com",
     "Charlie Li's email address?", ["charlie.li@example.com"]),
    ("Charlie Shanghai home", "Charlie lives in Shanghai, Pudong District",
     "Charlie's city of residence?", ["shanghai"]),
    ("Charlie Alibaba job", "Charlie works at Alibaba as a data scientist",
     "Charlie's employer?", ["alibaba"]),
    ("Charlie parrot Kiwi", "Charlie owns an African Grey parrot named Kiwi",
     "Charlie's pet?", ["parrot", "kiwi"]),
    ("Charlie Fudan grad", "Charlie graduated from Fudan University in 2020",
     "Charlie's alma mater?", ["fudan"]),

    ("Diana email", "Diana Zhang's email is diana.zhang@example.com",
     "Diana Zhang's email address?", ["diana.zhang@example.com"]),
    ("Diana Guangzhou home", "Diana lives in Guangzhou, Tianhe District",
     "Diana's city?", ["guangzhou"]),
    ("Diana Huawei job", "Diana works at Huawei as a hardware engineer",
     "Diana's workplace?", ["huawei"]),
    ("Diana hamster", "Diana owns a dwarf hamster named Peanut",
     "Diana's pet?", ["hamster", "peanut"]),
    ("Diana ZJU grad", "Diana graduated from Zhejiang University in 2017",
     "Diana's university?", ["zhejiang"]),

    ("Eve email", "Eve Liu's email is eve.liu@example.com",
     "Eve Liu's email?", ["eve.liu@example.com"]),
    ("Eve Chengdu home", "Eve lives in Chengdu, Wuhou District",
     "Eve's location?", ["chengdu"]),
    ("Eve Meituan job", "Eve works at Meituan as a backend developer",
     "Eve's company?", ["meituan"]),
    ("Eve rabbit Snowball", "Eve owns a Holland Lop rabbit named Snowball",
     "Eve's animal?", ["rabbit", "snowball"]),
    ("Eve NJU grad", "Eve graduated from Nanjing University in 2018",
     "Eve's school?", ["nanjing"]),

    ("Frank email", "Frank Zhao's email is frank.zhao@example.com",
     "Frank Zhao's email?", ["frank.zhao@example.com"]),
    ("Frank Wuhan home", "Frank lives in Wuhan, Hongshan District",
     "Frank's city?", ["wuhan"]),
    ("Frank Xiaomi job", "Frank works at Xiaomi as an iOS developer",
     "Frank's employer?", ["xiaomi"]),
    ("Frank turtle Sheldon", "Frank owns a red-eared slider turtle named Sheldon",
     "Frank's pet?", ["turtle", "sheldon"]),
    ("Frank HUST grad", "Frank graduated from Huazhong University of Science and Technology in 2016",
     "Frank's university?", ["huazhong"]),
]

for summary, content, _, _ in ar_data:
    ar_e.ingest(summary, content, confidence=0.95)

ar_r = R("AR", len(ar_data))
for summary, content, query, expected_kw in ar_data:
    answer = ask(ar_e, "You are a personal memory assistant.", query)
    ok = check(answer, expected_kw)
    if ok: ar_r.correct += 1
    ar_r.details.append({"q": query, "a": answer[:120], "ok": ok})
tcalls += ar_e._llm.call_count; ttokens += ar_e._llm.total_tokens
all_r.append(ar_r)
print(f"  AR: {ar_r.correct}/{ar_r.total} ({ar_r.correct/ar_r.total*100:.0f}%) | buckets={len(ar_e._bucket_manager.get_all_buckets())}")

# ═══════════════════════════════════════════════════════════════════════════
# SF: 6 scenarios — test conflict resolution with real semantic search
# ═══════════════════════════════════════════════════════════════════════════
print("\n── [SF] 6 scenarios (semantic) ──")
sf_e = make_engine()

sf_data = [
    ("location_change",
     [("I live in Beijing Haidian", "I currently live in Beijing, Haidian District", 0.8),
      ("I moved to Shanghai Pudong", "I moved to Shanghai, Pudong New Area last month", 0.9)],
     [("Where do I live now?", ["shanghai"]), ("Where did I live previously?", ["beijing"])]),
    ("job_switch",
     [("working at Google", "I work at Google as a software engineer", 0.7),
      ("now working at Microsoft", "I switched to Microsoft as a principal engineer", 0.95)],
     [("Where do I work now?", ["microsoft"]), ("Where did I work previously?", ["google"])]),
    ("allergy_correction",
     [("allergic to peanuts", "I am allergic to peanuts according to old tests", 0.6),
      ("actually allergic to shellfish", "Doctor confirmed I am NOT allergic to peanuts. I am allergic to shellfish instead.", 0.95)],
     [("What am I allergic to now?", ["shellfish"]), ("What did I think I was allergic to?", ["peanut"])]),
    ("team_growth",
     [("team of 5 people", "My engineering team has 5 members", 0.7),
      ("team expanded to 12", "My team grew to 12 members after recent hiring", 0.9)],
     [("How many team members now?", ["12"]), ("How many before?", ["5"])]),
    ("city_relocation",
     [("living in Tokyo Shinjuku", "I live in Tokyo, Shinjuku area", 0.8),
      ("relocated to Osaka Umeda", "I relocated to Osaka, Umeda district for work", 0.9)],
     [("Which city do I live in now?", ["osaka"]), ("Which city did I live in previously?", ["tokyo"])]),
    ("car_upgrade",
     [("driving Honda Civic 2019", "I drive a Honda Civic 2019 model", 0.7),
      ("switched to Tesla Model 3", "I sold my Honda and bought a Tesla Model 3", 0.9)],
     [("What car do I drive now?", ["tesla"]), ("What car did I drive previously?", ["honda"])]),
]

sf_r = R("SF", len(sf_data) * 2)
for sname, facts, queries in sf_data:
    for summary, content, conf in facts:
        sf_e.ingest(summary, content, confidence=conf)
    for query, expected_kw in queries:
        answer = ask(sf_e, "You are a personal assistant. Newer info replaces older.", query)
        ok = check(answer, expected_kw)
        if ok: sf_r.correct += 1
        sf_r.details.append({"scenario": sname, "q": query, "a": answer[:150], "ok": ok})

tcalls += sf_e._llm.call_count; ttokens += sf_e._llm.total_tokens
stale = sum(1 for n in sf_e._nodes.values() if n.is_stale)
all_r.append(sf_r)
print(f"  SF: {sf_r.correct}/{sf_r.total} ({sf_r.correct/sf_r.total*100:.0f}%) | stale={stale}")

# ═══════════════════════════════════════════════════════════════════════════
# LRU: 100 turns semantic
# ═══════════════════════════════════════════════════════════════════════════
print("\n── [LRU] 100 turns (semantic) ──")
lru_e = make_engine()

lru_key = {
    5:  ("Mercury project leader Dr Alice Chen", "Dr. Alice Chen leads Project Mercury as principal investigator"),
    15: ("Mercury started March 2024", "Project Mercury officially started on March 1, 2024"),
    25: ("Mercury team 12 researchers", "Project Mercury started with 12 researchers across 3 laboratories"),
    35: ("Mercury prototype April 2024", "April 2024 milestone: Mercury prototype completed with basic memory storage"),
    45: ("Mercury alpha July 2024 50K nodes", "July 2024: Mercury alpha shipped, supporting 50K memory nodes"),
    55: ("Mercury budget 5 million USD", "Project Mercury total budget: 5 million USD over 3 years"),
    65: ("Mercury beta graph October 2024", "October 2024: Mercury beta added graph retrieval features"),
    75: ("Mercury ICML 2025 presentation", "Project Mercury was presented at ICML 2025 in Vancouver"),
    85: ("Mercury v1.0 January 2025 open source", "January 2025: Mercury v1.0 released as open-source"),
    95: ("Mercury 3 users 1200 stars", "Mercury has 3 production users and 1200 GitHub stars"),
}

for i in range(100):
    if i in lru_key:
        s, c = lru_key[i]
        lru_e.ingest(s, c, confidence=0.95)
    else:
        lru_e.ingest(f"Mercury day {i} status", f"Day {i}: systems normal, {100+i*10} req/min", confidence=0.5)

lru_qs = [
    ("Who leads Project Mercury?", ["alice chen"]),
    ("When did Mercury officially start?", ["march", "2024"]),
    ("How many researchers on Mercury initially?", ["12"]),
    ("What milestone in April 2024?", ["prototype"]),
    ("What milestone in July 2024?", ["alpha", "50k"]),
    ("Total budget of Mercury?", ["5 million"]),
    ("Where was Mercury presented in 2025?", ["icml"]),
    ("When did v1.0 launch?", ["january 2025"]),
    ("GitHub stars?", ["1200"]),
    ("How many production users?", ["3"]),
]
lru_r = R("LRU", len(lru_qs))
for q, kw in lru_qs:
    a = ask(lru_e, "You are a project management assistant.", q)
    ok = check(a, kw)
    if ok: lru_r.correct += 1
    lru_r.details.append({"q":q,"a":a[:150],"ok":ok})

tcalls += lru_e._llm.call_count; ttokens += lru_e._llm.total_tokens
all_r.append(lru_r)
print(f"  LRU: {lru_r.correct}/{lru_r.total} ({lru_r.correct/lru_r.total*100:.0f}%)")

# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SEMANTIC EVAL RESULTS")
print("=" * 70)
for r in all_r:
    print(f"  [{r.competency}] {r.correct}/{r.total} ({r.correct/r.total*100:.0f}%)")
tc = sum(r.correct for r in all_r); tq = sum(r.total for r in all_r)
print(f"\n  OVERALL: {tc}/{tq} ({tc/tq*100:.0f}%)")
print(f"  LLM: {tcalls} calls | {ttokens} tokens | mode: {'SEMANTIC' if VECTOR_STORE.is_semantic() else 'LEXICAL'}")
print(f"  Vector store: LocalEmbeddingStore, {VECTOR_STORE.call_count} embedding calls")

report = {"overall_pct": round(tc/tq*100,1), "questions": tq, "correct": tc,
          "llm_calls": tcalls, "tokens": ttokens, "embedding_calls": VECTOR_STORE.call_count,
          "mode": "semantic" if VECTOR_STORE.is_semantic() else "lexical"}
with open("eval_semantic.json","w") as f:
    json.dump(report, f, indent=2)
print("\n[OK] eval_semantic.json saved")
