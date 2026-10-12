# Contributing to NAIR Research Harness

Thank you for contributing to the Neural AI Induction and Reuse (NAIR) project. This repository is dedicated to scientific reproducibility, rigorous compute accounting, and transparent leakage tracking.

---

## 1. Research Integrity Principles

1. **No Silent Overwrites**: Never overwrite existing experimental result files in `results/`. Always create versioned or tagged subdirectories.
2. **Pre-Register Hypotheses & Gates**: Define success thresholds in `docs/experimental_protocol.md` before analyzing test set metrics.
3. **Leakage Isolation**: When introducing new task families or splits, verify truth-table signatures over the full state domain to prevent functional leakage.
4. **Honest Compute Accounting**: Report total parameters, active parameters per step, and forward/backward FLOP estimates separately from peak memory.

---

## 2. Development & Testing Workflow

1. **Environment Setup**:
   ```bash
   python -m venv .venv
   pip install -e '.[dev]'
   ```

2. **Running Tests**:
   All PRs must maintain 100% test pass rate across the full suite:
   ```bash
   pytest -v
   ```

3. **Adding New Architectures**:
   - Subclass `nn.Module` in `src/ssnn/models.py`.
   - Implement parameter and FLOP estimation logic in `src/ssnn/accounting.py`.
   - Add unit tests covering output shapes, finite values, and gradient flow in `tests/test_models.py`.

---

## 3. Commit Message Conventions

We adhere to the Conventional Commits specification:
- `feat`: A new model architecture, task family, or evaluation slice.
- `fix`: A bug fix (e.g. padding handling, device transfer).
- `docs`: Documentation updates, ledger entries, or protocol additions.
- `test`: Adding or expanding unit tests.
- `refactor`: Code refactoring without behavior modification.
- `perf`: Performance optimization.
