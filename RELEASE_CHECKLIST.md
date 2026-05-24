# Release Checklist — v0.1.0

## Quality Gates

- [x] `ruff check memory_system/` — 0 errors
- [x] `mypy --strict memory_system/` — 0 errors
- [x] `pytest tests/` — 61/61 passed
- [x] Coverage ≥ 80% (actual: 89%)

## Package Integrity

- [x] `pip install -e .` succeeds
- [x] `pip install -e ".[dev]"` succeeds
- [x] `python -c "import memory_system; print(memory_system.__version__)"` prints `0.1.0`

## Documentation

- [x] `README.md` — install + quickstart + API reference
- [x] `CONTRIBUTING.md` — development harness guide
- [x] `docs/api/index.md` — core API documentation
- [x] `docs/ARCHITECTURE.md` — implementation architecture
- [x] `docs/contracts.md` — interface contracts
- [x] `ARCHITECTURE_DESIGN.md` — design spec (unchanged)
- [x] `E2E_REPORT.md` — scenario test report
- [x] `CHANGELOG.md` — version history

## CI

- [x] `.github/workflows/ci.yml` — lint + typecheck + test pipeline

## Example

- [x] `examples/basic_usage.py` runs without errors

## Pre-release Check

- [x] All layer files (00-05) addressed
- [x] No TODO/FIXME in core modules
- [x] No `random` usage (deterministic guarantee)
- [x] No physical node duplication (single bucket_id per node)

## Post-release (manual)

- [ ] Tag `v0.1.0` on main
- [ ] Build `python -m build` and verify dist/
- [ ] Publish to PyPI via `twine upload dist/*`
