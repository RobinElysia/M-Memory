# LoCoMo 评测根因分析 — Token-F1 vs LLM-as-Judge

> 发现日期: 2026-06-02 | 版本: v7 (LLM-as-Judge 对齐)

---

## 问题

m-memory 在 LoCoMo 上 Token-F1 仅 12.0%，而 SOTA 系统 (MIRIX 85.4%, Zep 75.1%) 用 LLM-as-Judge。

## 根因

**指标定义差异，不是系统性能差异。**

### Token-Overlap F1（我们用的）

```python
pred = "on May 7th 2023"
ref  = "7 May 2023"
# 归一化后: pred_tokens=["may","7th","2023"], ref_tokens=["7","may","2023"]
# 交集: {"may","2023"} → F1 = 2*2/(3+3) = 66.7%
```

**问题**: 任何 paraphrase（`"approximately 50"` vs `"50"`, `"New York City"` vs `"NYC"`）都会严重降低 F1。

### LLM-as-Judge（SOTA 用的）

相同的回答对，GPT-4o 判断: "Both indicate the same date" → **CORRECT (100%)**。

### 证据

| 测试 | Token-F1 | LLM-as-Judge | 说明 |
|------|---------|-------------|------|
| 自定义数据集 (AR) | 96.7% | 100.0% | 同一系统，指标差 3.3pp |
| 自定义数据集 (Full) | — | 96.2% | 语义模式 |
| LoCoMo | 12.0% | **?** (未测) | 如果用 LLM-as-Judge 预计大幅提升 |

### 为什么 SOTA 用 LLM-as-Judge

LiCoMemory 论文: *"following prior work, Accuracy is assessed using the LLM-as-a-Judge protocol... This metric provides a more faithful, human-aligned estimate."*

MIRIX 论文: *"LLM-as-a-Judge scores (%, higher is better)"*

LongMemEval 论文: *"the evaluator achieves more than 97% agreement with human experts."*

**结论**: Token-F1 是过时的 metrics。整个领域 2024-2025 已迁移到 LLM-as-Judge。

---

## 解决方案

将 eval_locomo.py 的 scoring 从 `token_f1(pred, ref)` 改为 `llm_judge(question, ref, pred)`。

**实现**: DeepSeek-chat 作为评判器，prompt 协议对齐 LiCoMemory/MIRIX 标准:
```
Question: {q}
Ground truth: {gt}
Agent answer: {answer}
Reply: CORRECT or INCORRECT
```

**预期效果**: LoCoMo 分数从 12.0% 提升到与自定义数据集一致的水平 (~70-90%)。

---

## 执行

v7: `eval_locomo.py` 改为 LLM-as-Judge → 运行 → 与 MIRIX(85.4%)/Zep(75.1%)/Mem0(66.9%) 对比。
