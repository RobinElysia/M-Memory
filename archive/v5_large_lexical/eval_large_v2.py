#!/usr/bin/env python3
"""Large-scale eval v2: proper summaries, hand-crafted queries, robust matching."""
import json, os, sys, time, random

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

def make_engine():
    return MemoryRetrievalEngineImpl(
        config=config, vector_store=NumpyVectorStore(dim=1536),
        graph_store=NetworkXGraphStore(), llm=DeepSeekAdapter(),
    )

def ask(engine, sys_prompt, user_msg):
    result = engine.search(user_msg, max_hops=0)
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
    """Lenient check: at least min_hits of expected words appear in answer."""
    answer_lower = answer.lower()
    hits = sum(1 for w in expected_words if w.lower() in answer_lower)
    return hits >= min_hits

class R:
    def __init__(self, c, t): self.competency=c; self.total=t; self.correct=0; self.details=[]

all_r = []; tcalls = 0; ttokens = 0
random.seed(42)

print("=" * 70)
print("  M-Memory Large-Scale Eval v2 (proper summaries)")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════
# AR: 80 facts × 40 queries, proper summaries
# ═══════════════════════════════════════════════════════════════════════════
print("\n── [AR] 80 facts ──")
ar_e = make_engine()

# Fact templates with semantic summaries
people = ["Alice Wang", "Bob Chen", "Charlie Li", "Diana Zhang", "Eve Liu",
          "Frank Zhao", "Grace Wu", "Henry Sun"]
ar_data = [
    # (semantic_summary, content, query, expected_keywords)
    ("Alice email address", "Alice Wang's email is alice.wang@example.com",
     "What is Alice Wang's email?", ["alice.wang@example.com"]),
    ("Alice lives in Shenzhen", "Alice lives in Shenzhen, Nanshan District",
     "Where does Alice live?", ["shenzhen", "nanshan"]),
    ("Alice works at Tencent", "Alice works at Tencent as a senior engineer",
     "Where does Alice work?", ["tencent"]),
    ("Alice's pet dog Max", "Alice has a golden retriever named Max",
     "What pet does Alice have?", ["golden retriever"]),
    ("Alice graduated from Tsinghua", "Alice graduated from Tsinghua University in 2018",
     "Where did Alice graduate from?", ["tsinghua"]),

    ("Bob email address", "Bob Chen's email is bob.chen@example.com",
     "What is Bob Chen's email?", ["bob.chen@example.com"]),
    ("Bob lives in Beijing", "Bob lives in Beijing, Haidian District",
     "Where does Bob live?", ["beijing"]),
    ("Bob works at ByteDance", "Bob works at ByteDance as a product manager",
     "Where does Bob work?", ["bytedance"]),
    ("Bob's cat Luna", "Bob has a Siamese cat named Luna",
     "What pet does Bob have?", ["siamese", "cat", "luna"]),
    ("Bob graduated from PKU", "Bob graduated from Peking University in 2019",
     "Where did Bob graduate from?", ["peking"]),

    ("Charlie email address", "Charlie Li's email is charlie.li@example.com",
     "What is Charlie Li's email?", ["charlie.li@example.com"]),
    ("Charlie lives in Shanghai", "Charlie lives in Shanghai, Pudong District",
     "Where does Charlie live?", ["shanghai"]),
    ("Charlie works at Alibaba", "Charlie works at Alibaba as a data scientist",
     "Where does Charlie work?", ["alibaba"]),
    ("Charlie's parrot Kiwi", "Charlie has an African Grey parrot named Kiwi",
     "What pet does Charlie have?", ["parrot", "kiwi"]),
    ("Charlie graduated from Fudan", "Charlie graduated from Fudan University in 2020",
     "Where did Charlie graduate from?", ["fudan"]),

    ("Diana email address", "Diana Zhang's email is diana.zhang@example.com",
     "What is Diana Zhang's email?", ["diana.zhang@example.com"]),
    ("Diana lives in Guangzhou", "Diana lives in Guangzhou, Tianhe District",
     "Where does Diana live?", ["guangzhou"]),
    ("Diana works at Huawei", "Diana works at Huawei as a hardware engineer",
     "Where does Diana work?", ["huawei"]),
    ("Diana's hamster Peanut", "Diana has a dwarf hamster named Peanut",
     "What pet does Diana have?", ["hamster", "peanut"]),
    ("Diana graduated from ZJU", "Diana graduated from Zhejiang University in 2017",
     "Where did Diana graduate from?", ["zhejiang"]),

    ("Eve email address", "Eve Liu's email is eve.liu@example.com",
     "What is Eve Liu's email?", ["eve.liu@example.com"]),
    ("Eve lives in Chengdu", "Eve lives in Chengdu, Wuhou District",
     "Where does Eve live?", ["chengdu"]),
    ("Eve works at Meituan", "Eve works at Meituan as a backend developer",
     "Where does Eve work?", ["meituan"]),
    ("Eve's rabbit Snowball", "Eve has a Holland Lop rabbit named Snowball",
     "What pet does Eve have?", ["rabbit", "snowball"]),
    ("Eve graduated from NJU", "Eve graduated from Nanjing University in 2018",
     "Where did Eve graduate from?", ["nanjing"]),

    ("Frank email address", "Frank Zhao's email is frank.zhao@example.com",
     "What is Frank Zhao's email?", ["frank.zhao@example.com"]),
    ("Frank lives in Wuhan", "Frank lives in Wuhan, Hongshan District",
     "Where does Frank live?", ["wuhan"]),
    ("Frank works at Xiaomi", "Frank works at Xiaomi as an iOS developer",
     "Where does Frank work?", ["xiaomi"]),
    ("Frank's turtle Sheldon", "Frank has a red-eared slider turtle named Sheldon",
     "What pet does Frank have?", ["turtle", "sheldon"]),
    ("Frank graduated from HUST", "Frank graduated from Huazhong University of Science and Technology in 2016",
     "Where did Frank graduate from?", ["huazhong"]),

    ("Grace email address", "Grace Wu's email is grace.wu@example.com",
     "What is Grace Wu's email?", ["grace.wu@example.com"]),
    ("Grace lives in Hangzhou", "Grace lives in Hangzhou, Xihu District",
     "Where does Grace live?", ["hangzhou"]),
    ("Grace works at NetEase", "Grace works at NetEase as a game designer",
     "Where does Grace work?", ["netease"]),
    ("Grace's snake Medusa", "Grace has a ball python named Medusa",
     "What pet does Grace have?", ["python", "snake", "medusa"]),
    ("Grace graduated from SJTU", "Grace graduated from Shanghai Jiao Tong University in 2019",
     "Where did Grace graduate from?", ["jiao tong"]),

    ("Henry email address", "Henry Sun's email is henry.sun@example.com",
     "What is Henry Sun's email?", ["henry.sun@example.com"]),
    ("Henry lives in Xian", "Henry lives in Xian, Yanta District",
     "Where does Henry live?", ["xian"]),
    ("Henry works at Baidu", "Henry works at Baidu as an NLP researcher",
     "Where does Henry work?", ["baidu"]),
    ("Henry's dog Coco", "Henry has a corgi named Coco",
     "What pet does Henry have?", ["corgi", "coco"]),
    ("Henry graduated from USTC", "Henry graduated from University of Science and Technology of China in 2017",
     "Where did Henry graduate from?", ["ustc", "science and technology"]),
]

# Ingest all 40 facts (5 per person × 8 people)
for summary, content, _, _ in ar_data:
    ar_e.ingest(summary, content, confidence=0.95)

ar_r = R("AR", len(ar_data))
for summary, content, query, expected_kw in ar_data:
    answer = ask(ar_e, "You are a personal memory assistant.", query)
    ok = check(answer, expected_kw, min_hits=1)
    if ok: ar_r.correct += 1
    ar_r.details.append({"q": query, "a": answer[:120], "ok": ok})
tcalls += ar_e._llm.call_count; ttokens += ar_e._llm.total_tokens
all_r.append(ar_r)
print(f"  AR: {ar_r.correct}/{ar_r.total} ({ar_r.correct/ar_r.total*100:.0f}%) | nodes={len(ar_e._nodes)} | buckets={len(ar_e._bucket_manager.get_all_buckets())}")

# ═══════════════════════════════════════════════════════════════════════════
# SF: 8 contradiction scenarios × 16 queries, proper summaries
# ═══════════════════════════════════════════════════════════════════════════
print("\n── [SF] 8 scenarios ──")
sf_e = make_engine()

sf_data = [
    # Each: [(summary1, content1, conf1), (summary2, content2, conf2)],
    #        [(query_now, expected_kw_now), (query_before, expected_kw_before)]
    ("location_address_change",
     [("I live in Beijing", "I currently live in Beijing, Haidian District", 0.8),
      ("I moved to Shanghai", "I moved to Shanghai, Pudong New Area last month", 0.9)],
     [("Where do I live now?", ["shanghai"]), ("Where did I live before?", ["beijing"])]),
    ("job_company_switch",
     [("working at Google", "I work at Google as a software engineer", 0.7),
      ("working at Microsoft now", "I switched to Microsoft as a principal engineer", 0.95)],
     [("Where do I work now?", ["microsoft"]), ("Where did I work before?", ["google"])]),
    ("food_allergy_correction",
     [("allergic to peanuts", "I am allergic to peanuts according to old tests", 0.6),
      ("actually allergic to shellfish", "Doctor confirmed I am NOT allergic to peanuts. I am allergic to shellfish instead.", 0.95)],
     [("What am I allergic to?", ["shellfish"]), ("What did I think I was allergic to before?", ["peanut"])]),
    ("team_size_growth",
     [("team has 5 people", "My engineering team has 5 members", 0.7),
      ("team expanded to 12 people", "My team grew to 12 members after recent hiring", 0.9)],
     [("How many team members now?", ["12"]), ("How many before the expansion?", ["5"])]),
    ("city_relocation_osaka",
     [("living in Tokyo", "I live in Tokyo, Shinjuku area", 0.8),
      ("moved to Osaka", "I relocated to Osaka, Umeda district for work", 0.9)],
     [("Which city do I live in now?", ["osaka"]), ("Which city did I live in before?", ["tokyo"])]),
    ("car_upgrade_tesla",
     [("driving a Honda Civic", "I drive a Honda Civic 2019 model", 0.7),
      ("switched to a Tesla Model 3", "I sold my Honda and bought a Tesla Model 3", 0.9)],
     [("What car do I drive now?", ["tesla"]), ("What car did I drive before?", ["honda"])]),
    ("phone_upgrade_iphone",
     [("using iPhone 13", "I have been using an iPhone 13 for two years", 0.7),
      ("upgraded to iPhone 16 Pro", "I just upgraded to the iPhone 16 Pro Max", 0.95)],
     [("What phone do I have now?", ["16"]), ("What phone did I have before?", ["13"])]),
    ("framework_switch_vue",
     [("using React for frontend", "I use React.js for all my frontend projects", 0.7),
      ("switched to Vue now", "I switched from React to Vue 3 for better performance", 0.9)],
     [("What framework do I use now?", ["vue"]), ("What framework did I use before?", ["react"])]),
]

sf_r = R("SF", len(sf_data) * 2)
for scenario_name, facts, queries in sf_data:
    for summary, content, conf in facts:
        sf_e.ingest(summary, content, confidence=conf)
    for query, expected_kw in queries:
        answer = ask(sf_e, "You are a personal assistant. Newer information replaces older information.", query)
        ok = check(answer, expected_kw)
        if ok: sf_r.correct += 1
        sf_r.details.append({"scenario": scenario_name, "q": query, "a": answer[:150], "ok": ok})

tcalls += sf_e._llm.call_count; ttokens += sf_e._llm.total_tokens
stale_count = sum(1 for n in sf_e._nodes.values() if n.is_stale)
all_r.append(sf_r)
print(f"  SF: {sf_r.correct}/{sf_r.total} ({sf_r.correct/sf_r.total*100:.0f}%) | stale nodes={stale_count}")

# ═══════════════════════════════════════════════════════════════════════════
# LRU: 200 turns, proper summaries for key facts
# ═══════════════════════════════════════════════════════════════════════════
print("\n── [LRU] 200 turns ──")
lru_e = make_engine()

lru_key_facts = {
    5:  ("Project Mercury leader", "Dr. Alice Chen is the lead researcher of Project Mercury"),
    15: ("Project Mercury start", "Project Mercury officially started on March 1, 2024"),
    25: ("Mercury team size initial", "Project Mercury started with 12 researchers across 3 labs"),
    35: ("Mercury prototype completed", "April 2024 milestone: Mercury prototype completed with basic memory storage"),
    45: ("Mercury alpha release", "July 2024: Mercury alpha shipped, supporting 50K memory nodes per instance"),
    55: ("Mercury budget allocation", "Project Mercury total budget: 5 million USD over 3 years"),
    65: ("Mercury beta with graph", "October 2024: Mercury beta added graph retrieval and cross-bucket edges"),
    75: ("Mercury ICML presentation", "Project Mercury was presented at ICML 2025 in Vancouver as an oral paper"),
    85: ("Mercury v1.0 launch", "January 2025: Mercury v1.0 released as open-source on GitHub"),
    95: ("Mercury adoption numbers", "Mercury has 3 production users and accumulated 1200 GitHub stars by March 2025"),
    105:("Mercury v2 planning", "March 2025: Mercury v2.0 planning started with community feedback integration"),
    115:("Mercury team expansion", "Mercury team expanded from 12 to 20 researchers after Series A funding"),
    125:("Mercury ACL award", "Project Mercury won Best Paper Award at ACL 2025"),
    135:("Mercury Series A funding", "Mercury received Series A funding of 10 million USD led by Sequoia Capital"),
    145:("Mercury v2 alpha release", "June 2025: Mercury v2.0 alpha released with automatic bucket splitting"),
}

for i in range(200):
    if i in lru_key_facts:
        summary, content = lru_key_facts[i]
        lru_e.ingest(summary, content, confidence=0.95)
    else:
        lru_e.ingest(
            f"Mercury project day {i} update",
            f"Day {i} status: systems operating normally, processing {100+i*10} requests per minute",
            confidence=0.5,
        )

lru_queries = [
    ("Who leads Project Mercury?", ["alice chen"]),
    ("When did Project Mercury officially start?", ["march 1", "2024"]),
    ("How many researchers started on Mercury?", ["12"]),
    ("What milestone happened in April 2024?", ["prototype"]),
    ("What was the July 2024 milestone?", ["alpha", "50k"]),
    ("What is the total budget of Project Mercury?", ["5 million"]),
    ("Where was Mercury presented in 2025?", ["icml", "vancouver"]),
    ("When did Mercury v1.0 launch?", ["january 2025"]),
    ("How many GitHub stars does Mercury have?", ["1200"]),
    ("When did Mercury win Best Paper?", ["acl 2025"]),
    ("How much Series A funding did Mercury receive?", ["10 million"]),
    ("How large is the Mercury team now?", ["20"]),
    ("What was the March 2025 milestone?", ["v2.0 planning"]),
    ("What was the June 2025 milestone?", ["v2.0 alpha"]),
    ("How many production users does Mercury have?", ["3"]),
]

lru_r = R("LRU", len(lru_queries))
for query, expected_kw in lru_queries:
    answer = ask(lru_e, "You are a project management assistant with access to Project Mercury records.", query)
    ok = check(answer, expected_kw, min_hits=1)
    if ok: lru_r.correct += 1
    lru_r.details.append({"q": query, "a": answer[:150], "ok": ok})

tcalls += lru_e._llm.call_count; ttokens += lru_e._llm.total_tokens
all_r.append(lru_r)
print(f"  LRU: {lru_r.correct}/{lru_r.total} ({lru_r.correct/lru_r.total*100:.0f}%) | nodes={len(lru_e._nodes)} | buckets={len(lru_e._bucket_manager.get_all_buckets())}")

# ═══════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  FINAL RESULTS — Large Scale v2")
print("=" * 70)
for r in all_r:
    acc = r.correct / r.total * 100 if r.total else 0
    print(f"  [{r.competency}] {r.correct}/{r.total} ({acc:.0f}%)")

tc = sum(r.correct for r in all_r)
tq = sum(r.total for r in all_r)
tn = len(ar_e._nodes) + len(sf_e._nodes) + len(lru_e._nodes)
print(f"\n  OVERALL: {tc}/{tq} ({tc/tq*100:.0f}%)")
print(f"  Nodes: {tn} | LLM calls: {tcalls} | Tokens: {ttokens} | Avg token/call: {ttokens/max(tcalls,1):.0f}")

report = {
    "summary": {"overall_pct": round(tc/tq*100,1), "questions": tq, "correct": tc,
                "nodes": tn, "llm_calls": tcalls, "total_tokens": ttokens,
                "avg_tokens_per_call": round(ttokens/max(tcalls,1))},
    "competencies": [
        {"competency": r.competency, "total": r.total, "correct": r.correct,
         "accuracy_pct": round(r.correct/r.total*100,1) if r.total else 0,
         "details": r.details[:5]}  # first 5 for brevity
        for r in all_r
    ],
}
with open("eval_large_v2.json", "w") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print("\n[OK] eval_large_v2.json saved")
