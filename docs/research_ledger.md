# Research Ledger

**Date:** 2026-10-10  
**Project:** Self-Structuring Neural Networks (SSNN)  

---

## 1. What is Known (Empirically Confirmed)

1. **Scaffold Trustworthiness:**
   - Primitive operations for `affine` ($S=11$) and `permutation` ($S=16$) have been mathematically verified against independent reference implementations.
   - Program execution follows deterministic left-to-right diagrammatic order: $y = f_{op_1}(f_{op_0}(x))$.
   - `MonolithicMLP` padding bug identified and resolved: padded positions are now zero-masked rather than clamped to operation $n_{ops}-1$.
   - Test suite expanded from 5 to 13 tests covering shapes, gradients, optimizer updates, reproducibility, padding, and leakage invariants.
2. **Task Generator & Split Leakage:**
   - In `affine`, the finite state space induces severe functional collapse at depth 3: all 1,728 programs produce only 110 distinct truth tables ($AGL(1, \mathbb{Z}_{11})$).
   - A naive token-pair holdout results in **66.1% functional leakage** into the test set.
   - Genuine generalization must be evaluated on **Strict Functional Holdouts** (programs whose truth tables never occurred in training).
   - The newly introduced `permutation` family with non-linear S-boxes yields 1,232 distinct signatures at depth 3, establishing a collision-resistant benchmark.
3. **Explicit Modular Performance:**
   - `ExplicitModularModel` trivially solves modular compositions when supplied with ground-truth operation-to-module indices ($100\%$ accuracy). However, it uses $M=12$ separate modules and has no unsupervised role discovery.

---

## 2. What is Uncertain (Under Active Investigation)

1. **Can `LearnedLibraryModel` Induce Sub-units Unsupervised?**
   - With straight-through discrete routing ($M=6$ units for 12 ops), does the model discover functional clustering (e.g., grouping related affine or permutation transformations), or does it suffer from routing collapse?
2. **Does Modular Library Outperform Fair Parameter-Matched Baselines?**
   - At matched capacity (parameters and training FLOPs), does `LearnedLibraryModel` achieve higher generalization on strict functional holdouts than `MonolithicMLP` or `SharedStepModel`?
3. **Behavior Under Depth Extrapolation:**
   - When tested on depth-4 sequences having seen only depth-1 and depth-2 training, does modular reuse generalize better than monolithic memory?

---

## 3. What is Still to Verify (Roadmap)

- [ ] Execute multi-seed pilot runs ($S \in \{0, 1, 2\}$) across both `affine` and `permutation` families.
- [ ] Measure exact test accuracies across all evaluation categories (`seen_depth3`, `unseen_pair_depth3`, `strict_functional_holdout`, `functional_leaked`, `depth4_extrapolation`).
- [ ] Record complete FLOP and parameter ledger across all runs.
- [ ] Evaluate pre-registered Gates G0 through G5.
- [ ] Author the Phase 4 Decision Memo (`docs/decision_memo.md`) determining whether structural growth (Phase 5) is warranted.
