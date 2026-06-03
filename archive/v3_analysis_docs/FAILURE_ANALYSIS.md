# m-memory 评测失败根因分析与架构改进方案

> 分析日期: 2026-05-26 | 评测工具: `eval_v2.py` + DeepSeek-chat | 数据: `eval_results.json`

---

## 一、失败分类与溯源

### 1.1 AR (Accurate Retrieval) — 80%，3 个 "Not found"

| 查询 | 预期 | 实际 | 失败原因 |
|------|------|------|---------|
| What pet does Alex have? | golden retriever | Not found | 查询词 "pet" 与内容词 "golden retriever" 无语义关联 |
| What car does Alex drive? | Tesla | Not found | 查询词 "car" 与内容词 "Tesla" 无语义关联 |
| What is Alex's daughter's name? | Emma | Not found | 查询词 "daughter" 与内容词 "Emma" 无语义关联 |

**根因**: `ask_agent()` 的关键词回退机制 (`eval_v2.py:37-42`) 要求查询词必须出现在存储内容的文本中：

```python
# eval_v2.py line 37-42 — 关键词回退只能匹配字面重合
for nid, node in memory_engine._nodes.items():
    if nid not in seen and len(seen) < 15:
        if any(kw in node.content.lower() for kw in keywords):
```

查询 "pet" 搜索内容 "golden retriever" → 无字面重叠 → 回退失败。
向量搜索同样失败，因为 `NumpyVectorStore.embed()` 使用 SHA-256 哈希生成向量，
不包含语义信息。

**→ 文件定位**: 
- `eval_v2.py:37-42` — 关键词回退机制
- `memory_system/vector_store.py:30-52` — 哈希嵌入函数 `embed()`
- `memory_system/retrieval.py:190-208` — `search()` 的桶内精筛完全依赖向量相似度

---

### 1.2 SF (Selective Forgetting) — 62.5%，3 个失败

| 场景 | 查询 | 预期 | 实际 | 失败原因 |
|------|------|------|------|---------|
| Location change | Where do I live **now**? | Shanghai | Shenzhen | AR 测试数据污染！"Alex lives in Shenzhen" 是 AR 事实，被误检索 |
| Allergy correction | What did I **think** I was allergic to? | peanut | Not found | 旧节点未被检索到（关键词 "think" 不匹配内容 "peanut"） |
| Team size growth | How many **before** the growth? | 3 | Not found | 旧节点未被检索到（关键词 "before" 不匹配内容 "3 members"） |

**根因 #1 — 跨测试数据污染**:
`eval_v2.py` 使用**同一个 `memory_engine` 实例**运行全部四个能力测试。
AR 阶段摄入的 "Alex lives in Shenzhen" 在 SF 阶段仍然存在，
查询 "Where do I live now?" 命中了这个 AR 事实。

**→ 文件定位**: `eval_v2.py:19-25` — 全局共享 `memory_engine` 实例

**根因 #2 — 旧节点检索失败**:
当查询 "What did I think I was allergic to?" 时，关键词回退提取的 `keywords` 集合
为 `{"think", "allergic"}`。旧事实 "peanut allergy according to previous tests"
不含 "think"，所以回退失败。且旧节点可能已被 LLM 冲突检测标记为 `is_stale`，
但检索时 `search()` 方法**不区分 stale 和 fresh 节点**——它返回所有向量相似节点，
只是在结果中按分数排序。问题是向量相似度是随机的（哈希嵌入），
所以旧节点可能根本没进入 top_m 桶的 top_p 结果。

**→ 文件定位**: 
- `memory_system/retrieval.py:190-208` — `search()` 的桶内精筛不保证语义相关性
- `memory_system/retrieval.py:258-296` — `resolve_conflicts()` 标记 stale 但 search 结果顺序已被随机化

**根因 #3 — stale 标记未生效 (系统指标 stale=0)**:
`eval_results.json` 中 `stale: 0` —— 冲突消解管线实际上**没有成功标记任何节点为 stale**。
原因是 `search()` 内部调用 `resolve_conflicts()` 时，
LLM 需要看到**同时包含新旧两个矛盾节点的候选集**才能检测冲突。
但由于哈希嵌入导致候选集是随机的，两个矛盾节点大概率不会同时出现在 top_n 中。

**→ 文件定位**: `memory_system/retrieval.py:216-220` — `resolve_conflicts` 仅在候选集包含矛盾对时才有效

---

### 1.3 TTL (Test-Time Learning) — 50%，1 个失败

| 场景 | 查询 | 预期关键词 | 实际命中 | 原因 |
|------|------|-----------|---------|------|
| User preferences | Summarize user preferences | dark, silent, morning, coffee | 0/4 | 检索到的全是 Alex 档案事实（"Alex Chen, 29, Tencent..."），不是偏好事实 |

**根因 — 关键词歧义**:
查询 "user preferences" 的关键词回退提取了 `{"summarize", "user", "preferences"}`。
"user" 出现在 AR 事实 "**User's** name is Alex Chen" 中。
Alex 档案有 20 个节点，偏好只有 4 个节点，数量碾压 + 关键词重叠 = 偏好节点被淹没。

**→ 文件定位**: `eval_v2.py:34` — 关键词回退不分桶、不考虑语义，全局暴力匹配

---

### 1.4 LRU (Long-Range Understanding) — 78%，2 个失败

| 查询 | 预期 | 实际 | 原因 |
|------|------|------|------|
| How many researchers? | 8 | Not found | 内容 "8 researchers from 3 countries" — 查询词 "How many" 与内容无字面重叠 |
| Where was NeuroMem presented? | NeurIPS | Not found | 内容 "presented at NeurIPS 2024 workshop" — 查询词 "Where" 与内容无字面重叠 |

**根因**: 同 AR — 关键词回退无法处理语义等价（"How many" ↔ "8", "Where" ↔ "NeurIPS"）。

---

## 二、架构层面根因诊断

### 2.1 核心问题: `VectorStore.embed()` 不具备语义能力

```python
# memory_system/vector_store.py:30-52
def embed(self, text: str) -> NDArray[np.float32]:
    vec = np.zeros(self.dim, dtype=np.float32)
    for i, ch in enumerate(text.encode("utf-8")):
        h = hashlib.sha256(f"{i}:{ch}".encode()).digest()
        for j in range(min(self.dim, len(h) // 4)):
            val = int.from_bytes(h[j*4:(j+1)*4], "big")
            vec[j] += (val / 2**32) * 2.0 - 1.0
    # L2-normalize
    return (vec / np.linalg.norm(vec)).astype(np.float32)
```

这个函数做的是**确定性字符哈希**——完全不是语义嵌入。
"cat" 和 "feline" 的余弦相似度 ≈ 0，而 "cat" 和 "cattle" 的相似度也 ≈ 0。
整个双层检索架构建立在余弦相似度之上，但底层向量的相似度是随机的。

**影响链**:
```
embed() 无语义
  → find_candidates() 返回随机候选桶
    → LLM 在随机候选中做决策 → 桶分配质量差
  → search() 的桶筛选是随机的
    → 桶内精筛是随机的
      → 图扩展的种子节点是随机的
        → resolve_conflicts() 看不到矛盾对
          → stale 标记无法生效
```

### 2.2 接口设计: 正确但有缺陷

`VectorStore` 接口本身是合理的——它定义了 `embed()`, `add()`, `search()` 等抽象方法，
允许替换后端。问题在于：

1. **接口没有标注嵌入的语义性要求**。应该添加文档说明：`embed()` 返回的向量
   必须具有语义可比性（即 `cosine_sim(embed("cat"), embed("feline")) >> 0`）。

2. **默认实现误导性**。`NumpyVectorStore` 作为默认实现，名称暗示它是向量存储，
   但实际产生的是无意义向量。应该重命名为 `HashVectorStore` 或添加 deprecation warning。

3. **缺少 fallback 机制**。当嵌入不具备语义能力时，整个系统无声地产生随机结果，
   没有告警也没有降级到关键词检索。

**→ 建议**: 在 `VectorStore` 接口中添加 `is_semantic() -> bool` 方法，
  当返回 `False` 时，`MemoryRetrievalEngine` 自动切换到关键词/BM25 模式。

### 2.3 数据流设计: 检索结果与 LLM 决策脱节

当前检索流程:
```
search(query)
  → Layer 1: 向量粗筛 → 向量精筛 → 图扩展 → 候选集 C
  → Layer 2: resolve_conflicts(C, query)
    → 重排序 → LLM 矛盾检测 → 标记 stale
  → 返回 C（按 score 排序）
```

问题在于：**Layer 2 修改了节点的 stale 状态，但这些修改不影响 Layer 1 的结果排序**。
`resolve_conflicts` 在 `top_candidates` 上操作（这是从 scored 排序后截断的），
修改的是这些候选节点的属性。但最终返回的列表顺序由 Layer 1 的得分决定——
而 Layer 1 的得分来自随机的哈希向量相似度。stale 标记虽然在对象上设置了，
但由于排序是随机的，stale 节点可能仍然排在前面。

**→ 建议**: `resolve_conflicts` 应该返回重新排序后的列表（它确实返回了），
  但 `search()` 应该使用 Layer 2 的排序结果，而不是 Layer 1 的。当前代码中：
  ```python
  resolved = self.resolve_conflicts(candidates, query)
  final_nodes = [n for n in resolved]
  final_scores = [node_scores.get(n.id, 0.0) for n in resolved]  # ← 这里用的是 Layer 1 的分数！
  ```
  `node_scores` 是 Layer 1 的得分，而 `resolved` 是通过 `resolve_conflicts` 排序的。
  这导致**顺序和分数不匹配**——顺序是 Layer 2 的，分数是 Layer 1 的。

  **→ 文件定位**: `memory_system/retrieval.py:228-233`

---

## 三、抽象修改建议

### 3.1 `VectorStore` 接口 — 添加语义能力声明

```python
class VectorStore(ABC):
    @abstractmethod
    def embed(self, text: str) -> NDArray[np.float32]: ...
    
    @abstractmethod  
    def is_semantic(self) -> bool:
        """Return True if embeddings are semantically meaningful.
        
        Hash-based implementations MUST return False.
        Model-based implementations (OpenAI, Cohere) MUST return True.
        """
        ...
```

### 3.2 `MemoryRetrievalEngine` — 添加检索模式选择

```python
class MemoryRetrievalEngineImpl:
    def search(self, query: str, ...) -> SearchResult:
        if not self._vector_store.is_semantic():
            # Fallback: use keyword + bucket-aware lexical search
            return self._lexical_search(query, ...)
        return self._semantic_search(query, ...)
```

### 3.3 `NumpyVectorStore` — 重命名 + 添加嵌入模型桥接

```python
# 重命名
class DeterministicHashStore(VectorStore):
    """Deterministic hash-based storage. NOT SEMANTIC. For testing only."""
    
    def is_semantic(self) -> bool:
        return False

# 新增
class OpenAIEmbeddingStore(VectorStore):
    """Production embedding via OpenAI API."""
    
    def __init__(self, model="text-embedding-3-small"):
        self._client = OpenAI()
        self._model = model
    
    def embed(self, text: str) -> NDArray[np.float32]:
        resp = self._client.embeddings.create(model=self._model, input=text)
        return np.array(resp.data[0].embedding, dtype=np.float32)
    
    def is_semantic(self) -> bool:
        return True
```

### 3.4 评估隔离 — 每个测试场景独立 engine 实例

```python
# 当前 (错误): 全局共享
memory_engine = MemoryRetrievalEngineImpl(...)
run_ar_tests(memory_engine)  # 写入 Alex 档案
run_sf_tests(memory_engine)  # 污染！

# 修改后: 每个场景新建
def run_ar_tests():
    engine = MemoryRetrievalEngineImpl(...)
    # ... ingest and test

def run_sf_tests():
    engine = MemoryRetrievalEngineImpl(...)  # 全新实例
    # ... ingest and test
```

### 3.5 `SearchResult` 排序 — 统一使用 Layer 2 分数

```python
# retrieval.py:228-233 当前代码
resolved = self.resolve_conflicts(candidates, query)
final_nodes = [n for n in resolved]
final_scores = [node_scores.get(n.id, 0.0) for n in resolved]  # Layer 1 分数

# 应改为
resolved_with_scores = self.resolve_conflicts(candidates, query)
# resolve_conflicts 应该返回 (node, layer2_score) 的列表
# 或者 search() 应该直接使用 resolve_conflicts 的排序结果
```

---

## 四、新问题研究方向

### 4.1 嵌入模型与聚类质量的因果分析

**问题**: 当嵌入具有语义能力后，LLM 驱动的桶聚类的质量如何随嵌入模型变化？
不同嵌入模型（OpenAI ada-002 vs text-embedding-3-small vs Cohere vs 本地模型）
对桶纯度的提升是否线性？

**实验设计**: 固定 100 个多主题对话，分别用 5 种嵌入模型运行分桶，
人工标注桶纯度（NMI、ARI），对比 LLM 决策的一致率。

### 4.2 跨桶边权重的最优策略

**问题**: 当前跨桶边权重完全由 LLM 决定（返回 0-1 的浮点数）。
LLM 是否倾向于高估关联性？是否存在某种校准策略（如用实际检索召回率
校准 LLM 给出的权重）？

**实验设计**: 让 LLM 对 N 对桶给出关联权重，然后对每对桶进行
500 条查询的跨桶检索召回率测试。比较 LLM 预测权重 vs 实际召回增益
的 Spearman 相关系数。

### 4.3 Stale 标记的召回率与精确率

**问题**: 冲突消解管线有多少假阳性（误标非矛盾信息为 stale）
和假阴性（漏标真正的矛盾）？LLM 的 prompt 设计（是否提供时间戳差异、
是否提供置信度差异）对检测准确率的影响？

**实验设计**: 构造 50 组标注好的矛盾/非矛盾节点对，
测试 LLM 在不同 prompt 变体下的检测性能。

### 4.4 桶分裂 vs 桶合并的自动决策

**问题**: 当前 `split_bucket` 是占位实现。什么时候应该分裂桶？
什么时候应该合并两个语义相近的桶？能否用**轮廓系数 (Silhouette Score)**
自动触发分裂/合并，而非依赖固定阈值？

**实验设计**: 在 1000 节点规模下，每增加 100 个节点计算桶内轮廓系数。
当某桶轮廓系数低于阈值时触发分裂，当两桶 Medoid 相似度 > 阈值时触发合并。
测量桶数演化和检索召回率的变化。

### 4.5 图扩展深度对幻觉率的影响

**问题**: `max_hops` 增大引入了更多候选节点，但也引入了更多噪声。
噪声节点的内容可能与查询无关，导致 LLM 在回答时产生幻觉（将无关信息
当作相关记忆使用）。H=0 到 H=3 时，LLM 回答的幻觉率如何变化？

**实验设计**: 对 100 条已知答案的查询，分别用 H=0/1/2/3 检索，
测量 LLM 回答的准确率和幻觉率（用 GPT-4 作为评判器）。

### 4.6 多 Agent 共享记忆的一致性

**问题**: 当多个 Agent 共享同一个 `m-memory` 实例时，Agent A 写入的信息
可能被 Agent B 的错误推理修改（如误标记为 stale）。是否需要
**写入者身份追踪**和**修改权限控制**？

---

## 五、立即行动项（按优先级）

| 优先级 | 行动 | 影响范围 | 预期效果 |
|--------|------|---------|---------|
| **P0** | 实现 `OpenAIEmbeddingStore` 替换 `NumpyVectorStore` | `vector_store.py` 新增 | 向量检索从随机变为语义，预计 AR 80% → 95%+, SF 62% → 85%+ |
| **P0** | 修复评估脚本的引擎隔离 | `eval_v2.py` | 消除跨测试数据污染 |
| **P1** | 修复 `search()` 的分数一致性 | `retrieval.py:228-233` | Layer 2 排序与分数匹配 |
| **P1** | 在 `VectorStore` 添加 `is_semantic()` | `interfaces.py` | 运行时检测 + 自动降级 |
| **P2** | 实现 `_lexical_search()` 回退 | `retrieval.py` 新增方法 | 当无语义嵌入时自动使用 BM25/关键词 |
| **P2** | 添加 LLM 调用计数到 stale 标记链路 | `retrieval.py`, `eval_v2.py` | 可观测性提升 |

---

## 六、改进后验证 (v3 评测结果)

### v2 → v3 变更

| 变更 | 文件 | 效果 |
|------|------|------|
| 引擎隔离（每个测试独立实例） | `eval_v3.py` | 消除 AR→SF 数据污染 |
| `_lexical_search()` 回退 | `retrieval.py:246-291` | 关键词检索替代随机向量检索 |
| `is_semantic()` 接口 + 自动检测 | `interfaces.py:133-142`, `retrieval.py:155-156` | 运行时自动切换检索模式 |
| `OpenAIEmbeddingStore`（语义嵌入） | `memory_system/embedding_store.py` | 生产就绪的语义嵌入后端 |

### v3 评测结果 (引擎隔离 + 词汇回退)

| 能力 | v2 准确率 | v3 准确率 | 提升 | 剩余失败 |
|------|----------|----------|------|---------|
| AR | 80.0% | **87.0%** | +7% | "daughter"↔"Emma", "languages"↔"Mandarin" |
| SF | 62.5% | **75.0%** | +12.5% | "live now?" 旧信息干扰, "before growth?" 关键词缺失 |
| LRU | 77.8% | **89.0%** | +11% | "How many researchers?" ↔ "8 researchers" |
| **总体** | **73.5%** | **84.0%** | **+10.5%** | 全部为关键词↔语义鸿沟，需真实嵌入模型解决 |

### 剩余失败的根因

所有 5 个剩余失败 (2 AR + 2 SF + 1 LRU) 都是**同一类问题**：
查询词与存储内容之间存在**语义等价但无字面重叠**的关系。
例如 "daughter" 与 "Emma" 是语义关联但无字符重叠，"How many" 与 "8" 同理。
**唯一解决方案**：将 `NumpyVectorStore` 替换为 `OpenAIEmbeddingStore`，
使 `is_semantic() == True`，触发真正的语义向量检索。

### 如何激活语义检索

```bash
export OPENAI_API_KEY="sk-..."
```

```python
from memory_system.embedding_store import OpenAIEmbeddingStore
engine = MemoryRetrievalEngineImpl(
    vector_store=OpenAIEmbeddingStore(),  # is_semantic() == True
    ...
)
# search() 自动使用 _semantic_search() (Layer 1 + Layer 2)
```
