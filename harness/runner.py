"""测试执行器 — 按协议运行全部测试场景并采集结果。"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class TestResult:
    id: str
    scenario: str
    status: str  # PASS | FAIL | SKIP
    duration_s: float
    llm_calls: int
    tokens_used: int
    note: str = ""
    error: str = ""


@dataclass
class HarnessReport:
    results: list[TestResult] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == "PASS")

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == "FAIL")

    @property
    def total_duration(self) -> float:
        return self.finished_at - self.started_at

    @property
    def total_tokens(self) -> int:
        return sum(r.tokens_used for r in self.results)


class TestRunner:
    """执行测试场景并聚合结果。"""

    def __init__(self) -> None:
        self.report = HarnessReport()
        self._llm_call_counter = 0
        self._token_counter = 0

    def reset_counters(self) -> None:
        self._llm_call_counter = 0
        self._token_counter = 0

    def run(
        self,
        test_id: str,
        scenario: str,
        fn: Callable[[], tuple[bool, str]],
    ) -> TestResult:
        """执行单个测试。fn 返回 (passed: bool, note: str)。"""
        self.reset_counters()

        t0 = time.perf_counter()
        try:
            passed, note = fn()
            status = "PASS" if passed else "FAIL"
            error = ""
        except Exception:
            passed = False
            status = "FAIL"
            note = "unexpected exception"
            error = traceback.format_exc()

        duration = time.perf_counter() - t0

        result = TestResult(
            id=test_id,
            scenario=scenario,
            status=status,
            duration_s=round(duration, 4),
            llm_calls=self._llm_call_counter,
            tokens_used=self._token_counter,
            note=note,
            error=error,
        )
        self.report.results.append(result)
        return result

    def record_llm_call(self, tokens: int = 0) -> None:
        self._llm_call_counter += 1
        self._token_counter += tokens

    def finalize(self) -> HarnessReport:
        self.report.finished_at = time.time()
        return self.report

    def to_markdown(self) -> str:
        r = self.report
        lines = [
            f"## 测试报告 — {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "### 摘要",
            f"- 总测试数: {len(r.results)}",
            f"- 通过: {r.passed} / 失败: {r.failed}",
            f"- 总耗时: {r.total_duration:.2f}s",
            f"- 总 token 消耗: {r.total_tokens}",
            "",
            "### 详细结果",
            "| ID | 场景 | 状态 | 耗时 | LLM调用 | Token | 备注 |",
            "|----|------|------|------|---------|-------|------|",
        ]
        for res in r.results:
            lines.append(
                f"| {res.id} | {res.scenario} | {res.status} | "
                f"{res.duration_s}s | {res.llm_calls} | {res.tokens_used} | "
                f"{res.note[:80]} |"
            )

        failed = [r for r in r.results if r.status == "FAIL"]
        if failed:
            lines.append("")
            lines.append("### 失败详情")
            for r in failed:
                lines.append(f"**{r.id}**: {r.note}")
                if r.error:
                    lines.append(f"```\n{r.error[:600]}\n```")

        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "summary": {
                    "total": len(self.report.results),
                    "passed": self.report.passed,
                    "failed": self.report.failed,
                    "duration_s": round(self.report.total_duration, 2),
                    "total_tokens": self.report.total_tokens,
                },
                "results": [
                    {
                        "id": r.id,
                        "scenario": r.scenario,
                        "status": r.status,
                        "duration_s": r.duration_s,
                        "llm_calls": r.llm_calls,
                        "tokens_used": r.tokens_used,
                        "note": r.note,
                    }
                    for r in self.report.results
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
