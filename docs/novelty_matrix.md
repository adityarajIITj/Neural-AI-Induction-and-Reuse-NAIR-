# Mechanism-Level Novelty Matrix

**Document Status:** Primary Literature Audit  
**Date:** 2026-10-10  

---

## 1. Ground Rules for Novelty Assessment

Per the Research Integrity Guidelines in the Implementation Plan:
1. Novelty is an explicit, mechanism-level distinction—not a new marketing name, a cosmetic rebranding, or a failure of basic literature search.
2. Every claim must specify whether it is **Full-Text Verified**, **Abstract-Only**, or **Unverified**.
3. A modular model combining known routing (e.g. straight-through Gumbel-Softmax) with standard MLPs on toy arithmetic is **NOT** a novel architecture; it is an experimental baseline.

---

## 2. Prior-Art Family Matrix

| Prior-Art Family | Key Primary Citations | Core Mechanism | Verification Level | Relationship to Current Candidate |
| :--- | :--- | :--- | :--- | :--- |
| **Neural Module Networks (NMN / NEO)** | Andreas et al. (2016); Pierrot et al. (2021) | Composes task-specific neural sub-networks according to program parse trees. | **Full-Text Verified** | Our `ExplicitStepModel` is an instance of static NMN execution; `LearnedLibrary` tests whether module roles can be induced without explicit ground-truth syntax trees. |
| **Routing Networks / Soft Modular Networks** | Rosenbaum et al. (2018); Kirsch et al. (2018) | Multi-agent/multi-unit routing where a controller learns to route data through shared modules. | **Full-Text Verified** | Implemented as our `SoftLibraryModel` baseline; uses soft continuous routing rather than hard discrete unit allocation. |
| **HyperNetworks** | Ha et al. (2016); Meyerson & Miikkulainen (2018) | A small network generates or modulates the weights of a target computation network conditioned on task/token context. | **Full-Text Verified** | Implemented as our `HyperNetStep` baseline; tests whether continuous weight generation outperforms discrete module selection. |
| **Program Induction / Library Learning** | Dechter et al. (2013) EC*; Ellis et al. (2021) DreamCoder; Grand et al. (2023) LILO | Symbolic program synthesis combined with wake-sleep abstraction to grow reusable libraries of primitives. | **Full-Text Verified** | In DreamCoder, reuse and refactoring are exact symbolic compression (MDL). In our candidate, units are differentiable continuous neural nets; Phase 5 tests if continuous units can mimic library abstraction. |
| **Self-Expanding / Dynamic Networks** | Chen et al. (2015) Net2Net; Yoon et al. (2018) DEN; Li et al. (2019) Learn to Grow | Dynamically allocates new network capacity (neurons, layers, or modules) upon high loss or task shift. | **Full-Text Verified** | Direct inspiration for Phase 5 (reuse, modify, or create). Deferred until compositional generalization without expansion is proven. |
| **Looped Transformers & Mixture-of-Recursions** | Giannou et al. (2023); Sanford et al. (2024) | Reusing identical transformer layers iteratively across execution steps to simulate programmatic recursion. | **Full-Text Verified** | Our `SharedStepModel` is the recurrent state analogue of a single-layer looped transformer. |
| **Local / Non-Backprop Learning Rules** | Hinton (2022) Forward-Forward; Scellier & Bengio (2017) EqProp; Nøkland (2016) Direct Feedback Alignment | Updates modular units locally using contrastive, target-prop, or synthetic gradients without end-to-end backpropagation through all steps. | **Full-Text Verified** | Relevant for Phase 2 Condition C2 (Restricted/Local Learning). To be compared directly against AdamW backprop. |
| **MDL & Bayesian Model Selection** | Rissanen (1978); Hinton & van Camp (1993); Blundell et al. (2015) | Minimum Description Length as a principled objective coordinating parameter cost, code length, and data fit. | **Full-Text Verified** | Governs the Phase 6 unified objective hypothesis; structural growth is only justified if lifetime capability repays parameter and FLOP complexity. |

---

## 3. Clear Differentiation of Claims

- **What is NOT Novel:**
  - Routing tokens to modular MLPs using softmax or straight-through estimators (established by Rosenbaum 2018, Kirsch 2018, Bengio 2013).
  - Showing that explicit modular networks achieve 100% on modular tasks (established by Andreas 2016).
  - Estimating linear layer FLOPs as $2 \times B \times I \times O$ (standard hardware practice).
- **What is under Empirical Investigation:**
  - Whether a discrete learned library (`LearnedLibraryModel`) without ground-truth supervisory role assignment can discover reusable primitives that generalize to strictly unseen functions under parameter-matched and compute-matched budgets against `MonolithicMLP` and `SharedStepModel`.
