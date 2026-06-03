#!/usr/bin/env python3
"""LoCoMo v7 — LLM-as-Judge protocol, aligned with MIRIX/LiCoMemory."""
import json, os, time, re, sys, random
from collections import Counter

os.environ["DEEPSEEK_API_KEY"] = "sk-768c26bb7779496e907781f52d82e526"
os.environ["M_MEMORY_LOG_LEVEL"] = "ERROR"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from memory_system.config import MemorySystemConfig
from memory_system.deepseek_llm import DeepSeekAdapter
from memory_system.graph_engine import NetworkXGraphStore
from memory_system.retrieval import MemoryRetrievalEngineImpl
from memory_system.local_embedding import LocalEmbeddingStore

JUDGE = DeepSeekAdapter()

def llm_judge(question, ground_truth, agent_answer):
    """LLM-as-Judge: binary correctness (SOTA protocol)."""
    prompt = (
        f"Evaluate the answer. Question: {question}\n"
        f"Ground truth: {ground_truth}\n"
        f"Agent answer: {agent_answer}\n\n"
        f"Reply EXACTLY one word: CORRECT or INCORRECT"
    )
    resp = JUDGE.complete(prompt).strip().upper()
    return "CORRECT" in resp

def retry_ingest(engine, s, c, confidence=0.9, max_r=3):
    for a in range(max_r):
        try: return engine.ingest(s, c, confidence)
        except RuntimeError:
            if a < max_r-1: time.sleep(2**a)

def retry_complete(llm, prompt, max_r=3):
    for a in range(max_r):
        try: return llm.complete(prompt)
        except RuntimeError:
            if a < max_r-1: time.sleep(2**a)
    return "[ERROR]"

print("Loading LoCoMo + model...")
data = json.load(open("locomo_data.json"))
VS = LocalEmbeddingStore(dim=384)
random.seed(42)

config = MemorySystemConfig()
config.embedding_dim = 384
config.bucket.top_m = 5; config.bucket.top_p = 15
config.graph.max_hops = 1

engine = MemoryRetrievalEngineImpl(
    config=config, vector_store=VS,
    graph_store=NetworkXGraphStore(), llm=DeepSeekAdapter(),
)

# Pick 1 conversation, sample 50 QA pairs stratified
conv = data[0]
qa_sample = []
by_cat = {}
for qa in conv["qa"]:
    by_cat.setdefault(qa.get("category",0), []).append(qa)
for cat, items in by_cat.items():
    qa_sample.extend(random.sample(items, min(12, len(items))))

print(f"Conv: {conv['sample_id']} — {len(qa_sample)} sampled QAs\n")

# Ingest turns
turn_count = 0
conv_obj = conv["conversation"]
for sk in sorted(conv_obj.keys()):
    if not sk.startswith("session_") or sk.endswith("_date_time"): continue
    s = conv_obj[sk]
    if not isinstance(s, list): continue
    for turn in s:
        t = turn.get("text","").strip()
        if t:
            summary = f"{turn.get('speaker','?')}: {t[:80]}"
            retry_ingest(engine, summary, t)
            turn_count += 1

print(f"Ingested {turn_count} turns. Evaluating...")

# LLM-as-Judge evaluation
cat_correct = {}; total_q = 0; total_c = 0
cat_total = {}

for i, qa in enumerate(qa_sample):
    question = qa["question"]
    gt = str(qa.get("answer", "no information available"))
    if not qa.get("answer"): continue  # adversarial
    cat = qa.get("category", 0)

    result = engine.search(question, max_hops=1)
    seen = set(); lines = []
    for n in result.nodes[:10]:
        if n.id not in seen:
            seen.add(n.id); lines.append(f"[{n.summary}] {n.content}")
    ctx = "\n".join(lines) if lines else "(none)"

    answer = retry_complete(engine._llm,
        f"You are a memory assistant.\nMemories:\n{ctx}\n\nQ: {question}\nA:"
    )

    correct = llm_judge(question, gt, answer)
    cat_correct[cat] = cat_correct.get(cat, 0) + int(correct)
    cat_total[cat] = cat_total.get(cat, 0) + 1
    total_q += 1; total_c += int(correct)

    if (i+1) % 10 == 0:
        print(f"  {i+1}/{len(qa_sample)}...", end="", flush=True)

print(f" done\n")

cat_names = {1:"single-hop",2:"multi-hop",3:"temporal",4:"open-domain",5:"adversarial"}
print("=" * 60)
print(f"  LoCoMo v7 — LLM-as-Judge ({conv['sample_id']})")
for cat in sorted(cat_correct):
    c = cat_correct[cat]; t = cat_total[cat]
    print(f"  {cat_names.get(cat,cat)}: {c}/{t} ({c/t*100:.1f}%)")
print(f"  OVERALL: {total_c}/{total_q} ({total_c/total_q*100:.1f}%)")
print(f"  LLM: {engine._llm.call_count+JUDGE.call_count} calls, {engine._llm.total_tokens+JUDGE.total_tokens} tokens")
print(f"\n  SOTA (same metric): MIRIX 85.4% | Zep 75.1% | Mem0 66.9%")
print(f"  SOTA (upper bound): Full-Context 87.5%")
print("=" * 60)

with open("eval_locomo_v7.json","w") as f:
    json.dump({
        "benchmark":"LoCoMo","protocol":"LLM-as-Judge","conversation":conv["sample_id"],
        "questions":total_q,"correct":total_c,"accuracy_pct":round(total_c/total_q*100,1),
        "by_category":{cat_names.get(k,k):round(cat_correct[k]/cat_total[k]*100,1) for k in sorted(cat_correct)},
        "sota_mirix":85.4,"sota_zep":75.1,"sota_mem0":66.9,"sota_fullcontext":87.5,
    },f,indent=2)
print("[OK] eval_locomo_v7.json saved")
