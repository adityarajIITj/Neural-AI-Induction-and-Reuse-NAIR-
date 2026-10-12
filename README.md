# Neural AI Induction and Reuse (NAIR) — Research Harness

A transparent, auditable experimental testbed for evaluating whether learned computational units generalize to held-out compositions, with rigorous truth-table signature tracking and compute accounting.

---

## 1. Research Question

> **Can learned computational units compose on genuinely held-out functions and combinations without joint retraining, and does any observed advantage survive fair parameter, compute, and inference controls?**

---

## 2. Implemented Model Architectures

The harness compares six model paradigms under strictly documented parameter and FLOP conventions:

| Architecture | Paradigm | Parameter Role |
| :--- | :--- | :--- |
| **`MonolithicMLP`** | Fixed Feed-Forward | Sequence-level monolithic control receiving the full zero-masked program encoding and state. |
| **`SharedStepModel`** | Recurrent Shared Step | Single non-linear transformation unit reused across every program step, conditioned on operation tokens. |
| **`ExplicitStepModel`** | Supervised Modular Control | Dedicated module per operation ($M = n_{ops}$); serves as an empirical upper-bound modular control. |
| **`LearnedLibraryModel`** | Discrete Modular Bank | Bank of $M = \lceil n_{ops} / 2 \rceil$ reusable units with straight-through discrete learned routing logits. |
| **`SoftLibraryModel`** | Continuous Mixture of Experts | Continuous soft mixture with softmax routing probabilities ([Rosenbaum et al., 2018](docs/novelty_matrix.md)). |
| **`HyperNetStep`** | Dynamic Parameter Generation | Operation embeddings generate the linear transformation weights dynamically ([Ha et al., 2016](docs/novelty_matrix.md)). |

---

## 3. Deterministic Task Families

1. **`affine` ($S=11$, 12 ops):**
   - Bijections over $\mathbb{Z}_{11}$: rotations, reflections, multipliers, and offsets.
   - Ground truth affine permutation group $AGL(1, \mathbb{Z}_{11})$ containing 110 distinct functions.
2. **`permutation` ($S=16$, 12 ops):**
   - Non-linear 4-bit cryptographic permutations: PRESENT cipher S-box, inverse S-box, bit reversals, bitwise XORs, adjacent bit swaps, and cyclic shifts.
   - Spans **1,232 distinct functional signatures** at depth 3, establishing a collision-resistant composition benchmark.

---

## 4. Truth-Table Signature & Leakage Isolation

A held-out token pair is not necessarily a held-out function. In finite state spaces, sequences can collapse to identical mappings:
$$\text{signature}(P) = (F(0), F(1), \dots, F(S-1))$$

To prevent spurious generalization claims, our split partitions test sequences into:
- **Strict Functional Holdouts**: Sequences containing held-out pairs whose truth table was **never seen in training**.
- **Functionally Leaked Compositions**: Sequences containing held-out pairs whose mathematical truth table **was already present in training**.

---

## 5. Setup & Quickstart

```bash
# Setup virtual environment
python -m venv .venv
# Activate (Windows: .venv\Scripts\Activate.ps1 | Linux: source .venv/bin/activate)
pip install -e '.[dev]'

# Run the full test suite
pytest -v
```

### Running Experiments
```bash
# Single pilot on affine family
python scripts/run_experiment.py --family affine --steps 500 --seeds 0

# Multi-seed pilot on permutation family
python scripts/run_experiment.py --family permutation --steps 500 --seeds 0 1 2

# Aggregate and compare all pilot results
python scripts/aggregate_pilot_results.py
```

---

## 6. Project Documentation

- [`docs/task_generator_validation.md`](docs/task_generator_validation.md): Mathematical proofs, collision rates, and leakage audit.
- [`docs/experimental_protocol.md`](docs/experimental_protocol.md): Pre-registered success criteria and evaluation gates.
- [`docs/novelty_matrix.md`](docs/novelty_matrix.md): Mechanism-level comparison to prior art (NMN, HyperNetworks, Soft Modular Networks, DreamCoder).
- [`docs/research_ledger.md`](docs/research_ledger.md): Persistent ledger of confirmed findings and open questions.
- [`docs/decision_memo.md`](docs/decision_memo.md): Phase 4 decision memo analyzing pilot outcomes.

---

## 7. License

Licensed under the [MIT License](LICENSE).
