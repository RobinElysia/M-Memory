# Contributing to m-memory

## How This Harness Works

This project uses a **layered agent-driven development harness** to build the
`m-memory` Python package. Each layer has defined deliverables and acceptance
criteria. Work through layers sequentially — you must pass all checks in the
current layer before proceeding.

## Layer Structure

| Layer | File | Focus |
|-------|------|-------|
| 0 | `00_architecture_freeze.md` | Freeze architecture → config + interface stubs |
| 1 | `01_contracts_and_skeleton.md` | Define abstract contracts + project skeleton |
| 2 | `02_core_implementation.md` | TDD implementation of all modules |
| 3 | `03_integration_and_e2e.md` | End-to-end scenario tests |
| 4 | `04_docs_and_examples.md` | Documentation, examples, developer experience |
| 5 | `05_packaging_and_release.md` | Packaging, CI, release readiness |

## Development Workflow

```bash
# 1. Clone and install in editable mode with dev dependencies
git clone https://github.com/RobinElysia/M-Memory.git
cd M-Memory
pip install -e ".[dev]"

# 2. Run all quality gates
ruff check memory_system/        # Lint
mypy --strict memory_system/     # Type check
pytest tests/                    # Unit + E2E tests

# 3. Generate coverage report
pytest tests/ --cov=memory_system --cov-report=html
```

## Code Guidelines

- **No randomness**: Medoid computation, bucket assignment — everything is deterministic.
- **Single physical placement**: Every MemoryNode lives in exactly one bucket.
- **LLM calls must be observable**: Use structured logging for every LLM interaction.
- **Graph atomicity**: Node/edge mutations should be reversible on failure.
- **Interfaces over implementations**: Code against `VectorStore`, `LLMAdapter`, etc., not concrete classes.

## Running Example

```bash
python examples/basic_usage.py
```

## Quality Gates (CI)

```bash
ruff check memory_system/ && mypy --strict memory_system/ && pytest tests/
```

All three must pass with zero errors before merging.
