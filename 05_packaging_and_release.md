# 第 5 层 · 打包与发布层 (Packaging & Release)

## 目标
产出一个可分发、高质量的 pip 包，并通过所有质量门禁。

## 输入
- 整个仓库代码、文档、测试

## Agent 任务
1. **完善 pyproject.toml**：
   - 确定版本号（从 `0.1.0` 开始或询问用户）
   - 填写元数据：作者、许可证、关键词（agent memory, knowledge graph, RAG, long-term memory）
   - 配置入口点（如有 CLI 命令）
2. **生成 CHANGELOG.md**：基于 git 提交历史（如果尚未初始化，则创建一个简洁模板）。
3. **配置 CI**：生成 `.github/workflows/ci.yml`（或类似），包含步骤：
   - `pip install -e .[dev]`
   - `ruff check` 或 `black --check`
   - `mypy --strict`
   - `pytest`
   - 安全扫描（可选，如 `pip-audit`）
4. **最终自查清单**：Agent 执行并确认：
   - `pip install -e .` 成功
   - `pytest` 全绿
   - `mypy` 无错误
   - 代码格式一致
   - 依赖无已知高危漏洞

## 验收标准
- CI 脚本在本地能够顺利执行全部步骤。
- 构建源码分发包（`python -m build`）无报错，并可安装测试。
- Agent 生成 `RELEASE_CHECKLIST.md` 报告清单完成情况。

## 交付物
- 更新的 `pyproject.toml`
- `CHANGELOG.md`
- `.github/workflows/ci.yml`
- `RELEASE_CHECKLIST.md`

完成此层后，仓库即为 **v1.0 就绪**，可推送至 PyPI（需人工操作或进一步自动化）。