# Frozen Experimental Protocol: Compositional Generalization Pilot

**Protocol Status:** FROZEN prior to pilot execution  
**Date:** 2026-10-10  
**Authors:** Research Harness Engineering  

---

## 1. Primary Research Question

> **Can learned computational units compose on genuinely held-out functions and combinations without being jointly retrained for each held-out composition, and does any observed advantage survive fair parameter, compute, and inference accounting?**

---

## 2. Model Architectures & Parameter Accounting

We evaluate six model families under transparent, documented capacity accounting:

1. **`MonolithicMLP` (Fixed Baseline):** Receives concatenated one-hot representation of full program sequence (padded slots explicitly zeroed out) and input state.
2. **`SharedStepModel` (Recurrent Shared Baseline):** Applies a single learned non-linear transformation conditioned on operation tokens sequentially.
3. **`ExplicitModularModel` (Upper Bound Modular Control):** Uses operation-specific dedicated units ($M = n_{ops}$); routing is deterministic and ground-truth indexed.
4. **`LearnedLibraryModel` (Candidate Modular Library):** Bank of $M = \lceil n_{ops} / 2 \rceil$ units with straight-through discrete learned routing logits.
5. **`SoftLibraryModel` (Prior Art Baseline — Rosenbaum et al. 2018):** Continuous mixture of units with softmax routing probabilities.
6. **`HyperNetStep` (Prior Art Baseline — Ha et al. 2016):** Operation embeddings dynamically generate the weights of the recurrent transformation step.

### Capacity & FLOP Accounting Convention
- **Total Parameters:** Sum of all trainable parameters.
- **Active Parameters:** Parameters utilized during a single forward evaluation step (e.g., in `Explicit`, only 1 unit is active per step despite storing 12 units).
- **Matrix-Op FLOP Convention:** Standard convention: 1 multiply-accumulate = 2 FLOPs. Training FLOPs = $3 \times \text{Forward FLOPs}$ (1 forward + 2 backward). Elementwise operations, activations, and routing overhead are explicitly noted as omitted.

---

## 3. Evaluation Protocol & Quantitative Gates

### 3.1 Evaluation Slices
For every run, models are evaluated on (see [`docs/glossary.md`](glossary.md) for formal definitions):
1. `seen_depth3`: Depth-3 sequences composed of seen pairs.
2. `unseen_pair_depth3`: Depth-3 sequences containing held-out pairs.
3. `strict_functional_holdout`: Depth-3 sequences containing held-out pairs whose mathematical truth table was **strictly absent from the training set**.
4. `functional_leaked`: Depth-3 sequences containing held-out pairs whose truth table **was already present in the training set**.
5. `depth4_extrapolation`: Length-4 sequences testing extrapolation depth beyond training length 2.
6. `val_seen_pair` and `val_heldout_pair`: Validation set slices (used exclusively for tuning).

### 3.2 Pre-Registered Success & Decision Gates
The experimental results are evaluated against the following pre-registered scientific criteria:

| Gate | Metric / Criterion | Threshold / Success Standard | Failure Interpretation |
| :--- | :--- | :--- | :--- |
| **G0: Correctness** | Test suite | 100% of unit & leakage tests pass | Halt execution; fix scaffold |
| **G1: Primitive Learning** | Training accuracy on primitives | $> 95\%$ on depth-1 tasks | Model capacity or optimization failure |
| **G2: Non-Triviality** | Chance baseline margin | Seen accuracy $> 5 \times$ random chance ($1/S \approx 9.1\%$) | Model failed to induce task rules |
| **G3: Honest Generalization** | Strict Holdout vs Leaked | If Leaked Acc $\gg$ Strict Holdout Acc, model exploits functional memorization | Acknowledge functional leakage confound |
| **G4: Compositional Advantage** | Candidate vs Baselines | Candidate `Library` outperforms `MLP` & `Shared` on strict holdout under matched compute | If `MLP` or `Shared` matches/dominates, reject modular library hypothesis |
| **G5: Stability Across Seeds** | 3-seed variance | Standard deviation of strict holdout accuracy $< 10\%$ | Inconclusive due to optimization instability |

---

## 4. Execution Plan
- Pilot runs executed across multiple seeds ($S \in \{0, 1, 2\}$) for both `affine` ($S=11$) and `permutation` ($S=16$) task families.
- All per-run configurations and raw predictions persisted as machine-readable JSON artifacts in `results/<run_id>/`.
