# Task Generator Validation Report

**Date:** 2026-10-10  
**Status:** Validated  
**Component:** `src/ssnn/tasks.py`

---

## 1. Executive Summary

This report establishes the mathematical validity, execution order correctness, and functional leakage characteristics of the synthetic composition tasks used in the SSNN research harness.

A central premise of the Implementation Plan is:
> *"A held-out token pair is not necessarily a held-out function. Different operation sequences can produce exactly the same mapping over the finite state space."*

Our mathematical audit confirms this finding quantitatively:
- In the initial **`affine`** family ($S=11$), the affine permutation group $AGL(1, \mathbb{Z}_{11})$ contains only $10 \times 11 = 110$ distinct bijections.
- Consequently, at depth 3 (1,728 programs), all 1,728 programs collapse into the exact same 110 functional signatures.
- Under a naive token-pair holdout, **66.1% of "unseen" test programs compute functions already present in the training set**.
- To prevent spurious claims of compositional generalization, we implemented full truth-table signature partitioning: `signature(program) = (F(0), F(1), ..., F(S-1))`, strictly isolating **Strict Functional Holdouts** (functions never seen in training) from **Functionally Leaked** compositions.
- We further engineered a second family, **`permutation`** ($S=16$), which includes non-linear S-box transformations (PRESENT block cipher), bit reversals, bitwise XORs, and cyclic rotations. This family yields **1,232 distinct functional signatures** at depth 3, drastically reducing collision rates and establishing task diversity.

---

## 2. Independent Reference Implementations & Execution Order

### 2.1 Primitive Operation Verification
Every primitive operation in both families was independently verified against mathematical reference definitions:
- **`affine` ($S=11$, 12 ops):** verified against $\lambda v: (a v + b) \pmod{11}$. All 12 operations have $\gcd(a, 11) = 1$, ensuring valid permutations on $\mathbb{Z}_{11}$.
- **`permutation` ($S=16$, 12 ops):** verified against explicit bitwise and S-box specifications. All 12 generators produce valid bijections on $\{0, \dots, 15\}$.

Automated tests in `tests/test_tasks.py::test_all_primitives_match_independent_reference` verify that every row in `family.tables` matches `reference_run` bit-for-bit.

### 2.2 Program Execution Order
The harness adopts **diagrammatic left-to-right order**:
For program $P = (op_0, op_1)$, $x$ is first transformed by $op_0$, and the resulting intermediate state is transformed by $op_1$:
$$y = f_{op_1}(f_{op_0}(x))$$

Hand-worked verification (`tests/test_tasks.py::test_execution_order_hand_worked_examples`):
- For $x = 3$, $op_0(v) = (v + 1) \pmod{11}$, $op_4(v) = (2 v) \pmod{11}$:
  - $(op_0, op_4)$: $op_0(3) = 4 \implies op_4(4) = 8$.
  - $(op_4, op_0)$: $op_4(3) = 6 \implies op_0(6) = 7$.
  - Both models and ground-truth execute in this exact sequence.

---

## 3. Truth-Table Signatures and Leakage Audit

For each program $P$, its signature is defined as:
$$\text{sig}(P) = (F(0), F(1), \dots, F(S-1)) \in \{0, \dots, S-1\}^S$$

### 3.1 Affine Family ($S=11$)
| Program Depth | Total Programs | Unique Signatures | Colliding Programs | Collision Rate |
| :--- | :--- | :--- | :--- | :--- |
| **Depth 1** | 12 | 12 | 0 | 0.0% |
| **Depth 2** | 144 | 73 | 115 | 79.9% |
| **Depth 3** | 1,728 | 110 | 1,728 | 100.0% |

**Leakage under Naive Pair Split (Seed 0, 25% heldout pairs):**
- Training programs: 120 (12 primitives + 108 depth-2 pairs) covering 63 unique functional signatures.
- Naive "Unseen" Depth-3 programs: 756 programs.
  - **Functionally Leaked (signature already in train):** 500 / 756 (**66.1%**)
  - **Strict Functional Holdout (signature absent from train):** 256 / 756 (**33.9%**)

### 3.2 Permutation Family ($S=16$)
| Program Depth | Total Programs | Unique Signatures | Distinct Functional Space |
| :--- | :--- | :--- | :--- |
| **Depth 1** | 12 | 12 | 100.0% |
| **Depth 2** | 144 | 127 | 88.2% |
| **Depth 3** | 1,728 | 1,232 | 71.3% |

By utilizing non-linear cryptographic S-box substitutions and bit permutations, the functional space spans over 1,200 unique functions at depth 3, providing a benchmark resistant to trivial group collapse.

---

## 4. Auditable Split Architecture

The updated `make_split` implements strict partition guarantees:
1. **Validation & Test Disjointness:** Held-out pairs are split into validation pairs ($V_H$) and test pairs ($T_H$) such that $V_H \cap T_H = \emptyset$. Hyperparameter selection uses validation pairs only.
2. **Train Disjointness:** Training programs contain no pairs from $V_H$ or $T_H$.
3. **Strict Separation of Evaluation Categories:**
   - `seen_depth3`: Compositions whose consecutive pairs were seen in training.
   - `unseen_pair_depth3`: Compositions containing held-out pairs.
   - `strict_functional_holdout`: Compositions containing held-out pairs whose truth table was **never present in training**.
   - `functional_leaked`: Compositions containing held-out pairs whose truth table **was already present in training**.
   - `depth4_extrapolation`: Length-4 sequences testing depth generalization.

Unit tests in `tests/test_leakage_and_splits.py` enforce these invariants and will fail if leakage occurs.
