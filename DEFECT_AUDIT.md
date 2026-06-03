# M-Memory v0.2.0 重大缺陷审计报告

> 审计日期: 2026-06-02 | 审计范围: 全部 12 个源文件 + 评测体系 + 架构设计
> 方法: 逐模块源码走查 + 评测数据交叉验证 + 与论文声明对照

---

## 一、缺陷总览

| # | 严重度 | 缺陷 | 影响 |
|---|--------|------|------|
| D1 | 🔴 Critical | 词汇回退的 stale 检测规则过于脆弱 | SF 准确率卡在 ~70%，无法突破 |
| D2 | ~~🔴 Critical~~ ✅ **FIXED (v0.3.0)** | `_semantic_search()` 从未被真正运行过 | `LocalEmbeddingStore` 上线，语义模式激活，SF 69%→83% |
| D3 | 🟡 High | `split_bucket()` 使用 hash 向量计算谱聚类 | 分裂基于随机相似度，完全不可靠 |
| D4 | 🟡 High | `find_candidates()` + `search()` 用 hash 向量选桶 | 桶筛选是随机的，LLM 在随机候选中做决策 |
| D5 | 🟡 High | `_lexical_search()` 是 O(N) 全局遍历 | 10K 节点时每次查询扫描所有节点，不可扩展 |
| D6 | 🟡 High | `resolve_conflicts()` 在词汇模式下从未被调用 | Layer 2 冲突消解只存在于语义路径 |
| D7 | 🟠 Medium | 无持久化 | 进程重启后所有记忆丢失 |
| D8 | 🟠 Medium | `_write_lock` 锁粒度过粗 | ingest 期间 LLM 调用持有锁，阻塞所有写入 |

---

## 二、逐缺陷详析

### D1 [Critical] 词汇回退 stale 检测过于脆弱

**现象**: SF 准确率 68.8%，5 个失败全是 stale 标记未触发。

**根因链**:
```
_lexical_search() 的 stale 检测 (retrieval.py:304-326)
  → 要求两个节点共享 topic_words 集合中的词
  → topic_words = {"live", "work", "allergy", "team", "member",
                   "address", "job", "name", "email", "phone"}
  → 对于 "drive a Honda" / "switched to Tesla" → "drive" 不在集合中 → 不检测
  → 对于 "use React" / "switched to Vue" → "use" 不在集合中 → 不检测
  → 对于 "speak English" / "also speak Japanese" → "speak" 不在集合中 → 不检测
```

**为什么**: 规则匹配是基于**硬编码的英文关键词集合**。这个集合是人类手工写的，
无法覆盖所有动词和名词形式。本质上是一个正则表达式冒充了语义理解。

**解决方案**:
1. **短期**: 扩展 topic_words 集合到 100+ 个常用动词/名词
2. **中期**: 用句子嵌入（Sentence-BERT）的余弦相似度替代关键词匹配，
   相似度 > 0.85 即判定为同一主题
3. **长期**: 激活 `_semantic_search()` 路径，让 Layer 2 的 LLM 冲突检测处理

---

### D2 [Critical] `_semantic_search()` 从未被真正运行过

**现象**: 两次大规模评测 (v4, v5) 全部走的是 `_lexical_search()` 路径。
`_semantic_search()` 代码存在但没有生产级的 embedding 后端可用。

**根因链**:
```
search() → is_semantic() == False → _lexical_search()
        → is_semantic() == True  → _semantic_search()  ← 从未到达

NumpyVectorStore.is_semantic() → False (hash-based)
OpenAIEmbeddingStore 存在但需要 OPENAI_API_KEY → 评测未配置
```

**为什么**: 论文声称的 "双层检索" (Layer 1 桶粗筛 + Layer 2 冲突消解) 
是**有条件承诺**——只有在语义嵌入可用时才激活。而当默认使用 NumpyVectorStore
时，整个系统退化为一个**带关键词评分的全局字典遍历**，根本不是双层架构。

**解决方案**:
1. **短期**: 在评测中配置 OPENAI_API_KEY 或 DeepSeek 兼容的 embedding 端点
2. **中期**: 实现一个本地 embedding 模型 (all-MiniLM-L6-v2, sentence-transformers)
3. **长期**: 将词汇回退升级为真正的 BM25 检索，加入 TF-IDF 权重

---

### D3 [High] 谱聚类基于随机相似度

**现象**: `split_bucket()` 用 pairwise cosine similarity 构建相似度矩阵，
但这些向量来自 `NumpyVectorStore` 的 hash 嵌入——方向是随机的。

**根因链**:
```
split_bucket() (bucket_manager.py:152-239)
  → sim = _cosine_sim(nodes[i].summary_vector, nodes[j].summary_vector)
  → summary_vector 来自 NumpyVectorStore.embed(summary_text)
  → hash 嵌入: cosine("cat","dog") ≈ cosine("cat","cat") ≈ 随机值
  → 相似度矩阵是随机矩阵
  → 拉普拉斯矩阵的第二特征向量 (Fiedler 向量) 的符号 → 随机分区
```

**为什么**: `split_bucket()` 的算法实现是正确的（谱聚类），但**输入数据是错误的**。
这就像在一张白纸上画了正确的坐标轴——算法本身没问题，但数据本身没有信息。

**解决方案**:
1. **不治本**: 在 `split_bucket()` 内部检测 `is_semantic()` 并拒绝分裂
2. **治本**: 确保只有语义嵌入可用时才调用 split_bucket

---

### D4 [High] 桶筛选是随机的

**现象**: `find_candidates()` 和 `search()` 用 hash 向量做 Medoid 余弦相似度筛选。
新节点被分配到**随机桶**，LLM 在随机候选中做决策。

**根因链**:
```
find_candidates() (bucket_manager.py:82-100)
  → sim = _cosine_sim(node.summary_vector, bucket.medoid.vector)
  → 两个向量都是 hash-based → 相似度随机
  → 返回的 top_k 候选桶 → 随机选择
  → LLM 收到随机候选 → 只能根据文本摘要判断
  → 但候选桶本身就是随机的 → LLM 可能看不到正确的桶
```

**为什么**: 这就是为什么桶的数量不稳定——LLM 得不到正确的候选桶，
只能不断创建新桶（primary_bucket="new"）。评测中 AR 只有 8 个桶对应 40 个事实
说明 LLM 确实在努力做正确的事——它看到了随机候选但凭文本做出了合理判断。

**解决方案**:
1. **词汇模式下**: `find_candidates()` 应该用关键词匹配而非余弦相似度
2. **语义模式下**: 用真实的 embedding 做余弦相似度

---

### D5 [High] 词汇搜索是 O(N) 全局遍历

**现象**: `_lexical_search()` 遍历 `self._nodes.values()` 的全部节点。

```python
# retrieval.py:280-295
for node in self._nodes.values():  # O(N) — 遍历所有节点
    content_hits = sum(1 for kw in keywords if kw in content_lower)
    score = content_hits * 2.0 + summary_hits * 1.0
```

**为什么**: 词汇回退没有利用桶结构。桶的存在是为了减少搜索空间（O(B) 而非 O(N)），
但词汇回退完全绕过了桶。10,000 个节点时每次查询扫描全部节点。

**解决方案**:
1. **短期**: 先按 bucket 分组，只搜索与查询关键词匹配的 bucket（即 bucket 的 Medoid
   summary 包含任意查询关键词）
2. **中期**: 为每个 bucket 建立倒排索引
3. **长期**: 激活语义搜索路径

---

### D6 [High] Layer 2 在词汇模式下从未被调用

**现象**: `resolve_conflicts()` 只在 `_semantic_search()` 中被调用。
`_lexical_search()` 有自己的轻量 stale 检测，但从未调用 LLM 做真正的冲突消解。

**根因链**:
```
search(query)
  → is_semantic() == False → _lexical_search()  ← 走这里
      → 轻量 stale 检测 (规则匹配, 无 LLM)
      → 没有调用 resolve_conflicts()

  → is_semantic() == True  → _semantic_search()  ← 从未走
      → resolve_conflicts() 调用 LLM 做矛盾检测
```

**为什么**: 词汇回退的设计哲学是"零额外 LLM 调用"——轻量 stale 检测不消耗 API。
但代价是真正的矛盾（如 "Beijing → Shanghai"）无法被正确检测，因为规则引擎
看不到 "Beijing" 和 "Shanghai" 之间的矛盾关系，只能看到它们都包含 "live"。

**解决方案**: 在词汇回退中也加入一次 LLM 调用做冲突检测（类似于语义路径）。
代价是每次查询多 1 次 LLM 调用。

---

### D7 [Medium] 无持久化

**现象**: 所有数据 (`_nodes`, `_buckets`, `_vectors`) 存储在内存中。
进程退出后全部丢失。

**解决方案**: 实现 SQLite / LanceDB 持久化后端用于 BucketManager 和 VectorStore。

---

### D8 [Medium] 写锁粒度过粗

**现象**: `ingest()` 持有 `_write_lock` 期间包含 LLM API 调用（~1.2s）。
在这 1.2 秒内，其他线程的所有 ingest 全部阻塞。

```python
def ingest(self, ...):
    with self._write_lock:  # 锁住整个 ingest，包括 LLM 调用
        return self._ingest_impl(...)
```

**解决方案**: 将 LLM 调用移出锁范围。锁只保护 `_nodes` 和 `_buckets` 的修改，
不保护网络 I/O。

---

## 三、评测体系缺陷

| # | 缺陷 | 说明 |
|---|------|------|
| E1 | v3/v5 评测没有跑 `_semantic_search()` 路径 | 论文的 projection 数字 (93.8%) 是理论值，不是实测 |
| E2 | 评测匹配规则 `check()` 过于宽松 | 仅需 1 个 keyword 命中即判定正确，可能假阳性 |
| E3 | SF 评测的 "before" 查询期望不严谨 | "Where did I live before?" 期望 ["beijing"]，
  但旧信息可能被 stale 降权后 LLM 输出 "no prior location" → 仍被判对 |
| E4 | 无 Recall@K, MRR 等标准 IR 指标 | 只用 accuracy，无法评估检索排序质量 |
| E5 | TTL 从未被评测 | MemoryAgentBench 四能力只测了三个 |

---

## 四、架构设计缺陷

| # | 缺陷 | 说明 |
|---|------|------|
| A1 | 双层架构是有条件的 | 语义嵌入不可用时系统退化为单层词汇检索 |
| A2 | 桶结构在词汇模式下未被利用 | 桶是 LLM 花 token 创建的，但词汇搜索不用它们 |
| A3 | Medoid 在词汇模式下无意义 | 基于 hash 向量的 Medoid 是随机节点，不代表任何语义中心 |
| A4 | 跨桶边从未在生产评测中使用 | 所有评测设 `max_hops=0` |
| A5 | `CleanupScheduler` 从未在评测中运行 | dormancy/contradiction scan/topic drift 全未测试 |

---

## 五、修正优先级路线图

| 优先级 | 行动 | 预期效果 |
|--------|------|---------|
| **P0** | 接入真实 embedding 后端，跑通 `_semantic_search()` | 激活完整两层架构，SF 75% → 90%+ |
| **P0** | 词汇模式下的 `find_candidates()` 改用关键词匹配 | 桶筛选不再随机 |
| **P1** | `_lexical_search()` 改为 bucket-aware 搜索 | 从 O(N) 降到 O(K·B)，可扩展到万级节点 |
| **P1** | 词汇模式中加入 LLM conflict detection | SF 矛盾检测不再依赖硬编码规则 |
| **P1** | `split_bucket()` 加入 `is_semantic()` 守卫 | 防止在非语义嵌入下执行无效分裂 |
| **P2** | 评测加入 Recall@K, MRR 指标 | 更全面的检索质量评估 |
| **P2** | 评测加入 TTL 能力 | 四能力全覆盖 |
| **P2** | `_write_lock` 缩小粒度 | 提升并发 ingest 性能 |
| **P3** | SQLite 持久化 | 进程重启后记忆不丢失 |
| **P3** | 跨桶边评测 | 验证图扩展的实际收益 |
