# E2E Scenario Test Report

## 概述

3 个端到端场景测试全部通过（61 个测试总计通过）。所有测试均使用
**假嵌入模型**（确定性 hash）+ **假 LLM**（脚本化响应），确保快速且可重复。

## 场景 1: 主题漂移与桶分裂 ✅

**测试**: `TestTopicDriftAndBucketSplit::test_topic_drift_forms_separate_buckets`

**步骤**:
1. 摄入 10 轮"猫"主题对话
2. 摄入 5 轮"狗"主题对话
3. 摄入 1 轮"猫"主题对话

**预期行为**:
- 至少形成 2 个桶（猫桶 + 狗桶）
- 回到猫话题时，正确分配回猫桶
- 猫桶应包含 11 个节点

**实际结果**: ✅ 通过
- 2 个桶正确形成
- 猫桶含 11 个节点
- 狗桶含 5 个节点
- 第 11 轮猫对话正确分配回猫桶

**可视化**:
```
Bucket: cat (11 nodes)          Bucket: dog (5 nodes)
├── n1: cat topic 0             ├── n11: dog topic 0
├── n2: cat topic 1             ├── n12: dog topic 1
├── ...                         ├── ...
└── n11: cat is purring again   └── n15: dog topic 4
```

---

## 场景 2: 矛盾信息与冲突消解 ✅

**测试**: `TestConflictResolution::test_contradiction_resolution`

**步骤**:
1. 创建节点 "I live in Beijing" (timestamp=1000, confidence=0.8)
2. 创建节点 "I moved to Shanghai" (timestamp=2000, confidence=0.9)
3. LLM 检测到矛盾
4. 执行冲突消解管线

**预期行为**:
- 北京节点被标记为 `is_stale=True`
- 上海节点保持非过时状态
- 上海节点置信度高于北京节点
- 两个节点均未删除

**实际结果**: ✅ 通过
- `beijing.is_stale = True`
- `shanghai.is_stale = False`
- `shanghai.confidence (0.9) > beijing.confidence (0.08)` (降权后)
- 两个节点均在结果中保留

**可视化**:
```
Results (after conflict resolution):
[0] Shanghai (score=0.63, confidence=0.9,  stale=false) ← 优先
[1] Beijing  (score=0.53, confidence=0.08, stale=true)  ← 降权保留
```

---

## 场景 3: 桶休眠与唤醒 ✅

**测试**: `TestBucketDormancy::test_dormancy_and_wake`

**步骤**:
1. 创建桶并摄入 3 个节点
2. 人工老化桶（last_write_at=0, last_query_at=0）
3. 运行休眠检查
4. 手动唤醒桶
5. 执行搜索

**预期行为**:
- 休眠检查后桶被标记为 `is_dormant=True`
- Medoid 向量移出活跃索引
- 唤醒后桶恢复激活状态
- 搜索能正常返回结果

**实际结果**: ✅ 通过
- 休眠后活跃桶数量 = 0
- 唤醒后活跃桶数量 = 1
- 搜索返回 ≥1 个结果

**可视化**:
```
Before dormancy check:     After dormancy check:      After wake:
Active index: [bucket_1]   Active index: []           Active index: [bucket_1]
                           bucket_1: dormant=true     bucket_1: dormant=false
```

---

## 测试统计

| 指标 | 数值 |
|------|------|
| E2E 场景数 | 3 |
| 全部测试数 | 61 |
| 通过 | 61 |
| 失败 | 0 |
| 代码覆盖率 | 89% |
| mypy strict | ✅ |
| ruff check | ✅ |
