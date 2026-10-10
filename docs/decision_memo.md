# Phase 4 Research Gate — Decision Memo

**Date:** 2026-10-10  
**Authors:** Research Harness Engineering  
**Gate Decision:** **REJECT IMMEDIATE STRUCTURAL GROWTH; PIVOT CANDIDATE TO CONTINUOUS MODULAR ADAPTATION (HYPERNETWORKS / INTERFACE CONSTRAINTS)**

---

## 1. What Did We Learn?

1. **Functional Leakage is Massive in Finite-State Tasks:**
   In the `affine` family, 66.1% of naive "held-out pair" test cases shared functional signatures with training programs. Evaluating on naive splits produces an artificial 3%–10% performance boost across all architectures. True compositional generalization must be judged on **Strict Functional Holdouts**.
2. **Modular Inductive Bias is Necessary for Composition:**
   Standard architectures (`MonolithicMLP` and `SharedStepModel`) completely fail to induce composition (scoring near random chance: 5%–10% on both `affine` and `permutation`). Meanwhile, explicit modular routing (`ExplicitModularModel`) achieves 99.4%–100.0% accuracy on strict holdouts and depth-4 sequences.
3. **Discrete Unit Routing (`LearnedLibraryModel`) Underperforms Prior Art:**
   The candidate discrete library (with straight-through routing) achieves modest above-chance generalization (15.3% on affine, 17.7% on permutation). However, it is **dominated by existing prior-art baselines**:
   - `SoftLibraryModel` (Rosenbaum et al. 2018) achieves 24.8% on affine and 20.7% on permutation.
   - `HyperNetStep` (Ha et al. 2016) achieves **82.2% on affine** and **48.6% on permutation** at matched parameters and fewer FLOPs.

---

## 2. What is the Closest Prior Art?

- **HyperNetworks (Ha et al. 2016):** An embedding generates transformation weights conditioned on operation tokens. This baseline outperforms discrete modular routing by **+66.9%** on `affine` and **+30.9%** on `permutation`.
- **Soft Modular Networks (Rosenbaum et al. 2018; Andreas et al. 2016):** Soft continuous routing avoids discrete optimization collapse and consistently surpasses hard straight-through routing.

---

## 3. What Result is Genuinely Unexplained by the Baselines?

**None.**  
The current candidate `LearnedLibraryModel` produces no phenomenon that is not already better explained and achieved by established prior art (`HyperNetworks` and `Soft Modular Networks`). 
Claiming that a discrete learned library is a superior path to compositional generalization is unsupported by the empirical evidence under parameter-matched and compute-matched controls.

---

## 4. What Should Be Abandoned?

- **Abandon automatic structural growth (Phase 5 / Phase 6) on the current discrete library:**  
  Expanding, modifying, or pruning units in a library that already underperforms standard continuous baselines would simply add unconstrained complexity without addressing the core optimization bottleneck (discrete routing collapse).
- **Abandon claims of novel architecture for discrete token-routed MLPs.**

---

## 5. What is the Smallest Justified Next Experiment?

Before any structural creation/growth rule can be considered:
1. **Investigate Hybrid Modular-HyperNetwork Formulation (Condition C3/C4 from Phase 2):**  
   Test whether imposing an explicit low-rank algebraic interface or compatibility projection (as in NEO / Local Module Composition) allows the discrete library to match `HyperNetStep`.
2. **Test Restricted / Local Updates (Condition C2):**  
   Evaluate whether local non-backprop updates (e.g. layer-wise target loss or Forward-Forward) offer compute savings without losing the compositionality seen in `HyperNetStep`.
