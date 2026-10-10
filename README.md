# Self-Structuring Neural Networks — Research Harness

This is a small, auditable starting point for testing whether learned computational units generalize to held-out compositions. It is **not** evidence of novelty or compute efficiency.

## First experiment

A deterministic task family defines transformations over a finite state space. Each example supplies an operation sequence and an input state; the target is the result of executing the sequence. Some ordered operation pairs are held out from training and used to test compositional generalization.

Included models:
- `MonolithicMLP`: fixed-depth feed-forward baseline; receives a program encoding and input.
- `SharedStepModel`: applies one shared learned transformation conditioned on each operation token.
- `ExplicitStepModel`: operation-specific learned transformation modules (an upper-bound-style modular control, but roles are not guaranteed to match ground-truth operations).
- `LearnedLibraryModel`: shared bank of learned units with a learned operation-to-unit router.

The learned-library model is a research candidate, not a claimed novel architecture. The task family is intentionally small and synthetic.

## Setup

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e '.[dev]'
pytest -q
```

## Run

```bash
python scripts/run_experiment.py --family affine --steps 500 --seeds 0
```

Use `--device cpu` to force CPU or `--device cuda` if CUDA is available. The script prints and saves the actual device, model parameter count, an estimated matrix-operation FLOP count, wall time, and category-level exact accuracy. The FLOP estimate counts linear-layer matrix operations for forward/backward passes and is explicitly incomplete: it does not count elementwise ops, optimizer updates, data generation, or evaluation.

## Scientific cautions

- Held-out token pairs are not automatically functionally novel; transformations can be equivalent. The generator reports pair holdouts, but stronger studies should additionally filter programs by functional equivalence over the complete finite input state space.
- A modular model may have a different inductive bias from a monolithic baseline. Match parameters and optimization budgets before interpreting results.
- One seed and a toy benchmark are a smoke/pilot only. Run multiple seeds and at least one second task family before making claims.
- Lower peak memory is not lower training FLOPs. Report training cost, inference cost, parameter count, and quality separately.
