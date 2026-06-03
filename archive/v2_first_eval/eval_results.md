# m-memory MemoryAgentBench Evaluation

| Competency | Test | Correct | Total | Accuracy |
|------------|------|---------|-------|----------|
| AR | AccurateRetrieval-50facts | 20 | 20 | 100.0% |
| SF | SelectiveForgetting-5scenarios | 1 | 10 | 10.0% |
| LRU | LongRangeUnderstanding-100turns | 2 | 7 | 28.6% |
| TTL | TestTimeLearning-2scenarios | 1 | 2 | 50.0% |

**System**: 168 nodes, 9 buckets, 1 stale
**Cost**: 245 LLM calls, 96895 tokens
