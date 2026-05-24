# 第 4 层 · 文档与范例层 (Docs & Examples)

## 目标
让任何开发者能在 5 分钟内安装、运行最小示例，并理解 API。

## 输入
- 完整代码库
- 测试通过证明（Layer 3 token）

## Agent 任务
1. **编写 README.md**：
   - 顶部徽章（build, coverage, python version）
   - 项目简介（一句话）
   - 安装命令（`pip install memory-system` 或 `pip install -e .`）
   - 5 分钟快速开始：一个可复制运行的代码块，演示添加 3 轮对话并检索
   - 基本概念说明（A、C、桶、Medoid、跨桶边等，引用 `ARCHITECTURE_DESIGN.md` 简要介绍）
   - API 参考大纲（链接到自动生成文档）
   - 贡献指南链接
2. **生成示例**：提供 `examples/basic_usage.py` 或 Jupyter Notebook，代码可执行，注释清晰，输出友好。
3. **API 文档**：使用 Sphinx 或 mkdocs 生成静态文档骨架，从 docstring 自动提取。最少生成一个 `docs/api/` 索引页。
4. **贡献指南**：撰写 `CONTRIBUTING.md`，说明如何使用本 Harness 继续开发。

## 约束
- 所有示例代码必须能直接运行，无额外手工配置（如 API key 需使用假适配器提示）。
- 文档中的所有 Python 代码块必须通过 `pytest --doctest-modules` 或 `doctest` 验证。
- 语言中英文不限，但必须准确。

## 验收标准
- 按照 `README.md` 的快速开始步骤执行，5 分钟内跑通示例。
- 运行 `pytest --doctest-modules` 无失败。
- 生成的 API 文档链接可访问，至少包含 3 个核心类的说明。

## 交付物
- `README.md`
- `examples/` 目录
- `docs/` 下 API 文档
- `CONTRIBUTING.md`

通过后，颁发 **Layer 4 Pass Token**，进入最后一层。