# Research Harness Architecture

This document describes the high-level architecture of the NAIR experimental harness, from synthetic task generation to split isolation and evaluation.

---

## 1. Pipeline Overview

```mermaid
flowchart TD
    subgraph Data Pipeline
        TF[Task Family: affine / permutation] --> MP[make_split: Token Pairs]
        MP --> SIG[Truth-Table Signature Analysis]
        SIG --> TRAIN[Train Programs: Primitives + Seen Pairs]
        SIG --> VAL[Validation Sets: Disjoint Pairs]
        SIG --> TEST_STRICT[Strict Functional Holdout: Novel Truth Tables]
        SIG --> TEST_LEAKED[Functionally Leaked: Repeated Truth Tables]
        SIG --> TEST_D4[Depth-4 Extrapolation: Long Sequences]
    end

    subgraph Model Architectures
        M_MLP[MonolithicMLP]
        M_SHARED[SharedStepModel]
        M_EXP[ExplicitStepModel]
        M_LIB[LearnedLibraryModel - ST]
        M_SOFT[SoftLibraryModel - MoE]
        M_HYP[HyperNetStep - Dynamic]
    end

    subgraph Accounting & Evaluation
        TRAIN --> FIT[Training Loop: AdamW, Fixed Seeds]
        M_MLP & M_SHARED & M_EXP & M_LIB & M_SOFT & M_HYP --> FIT
        FIT --> METRICS[Per-Slice Accuracy & Program Exact Match]
        FIT --> ACCT[Active Parameters & Matrix FLOPs Accounting]
        METRICS & ACCT --> RES[Machine-Readable results/metrics.json]
    end
```

---

## 2. Key Modules

1. **`ssnn.tasks`**: Deterministic bijections, truth-table signatures, program execution, and leakage isolation.
2. **`ssnn.models`**: Implementations of monolithic, recurrent, modular, and hypernetwork baselines.
3. **`ssnn.accounting`**: Transparent accounting of total parameters, active parameters per step, and forward/backward FLOPs.
4. **`ssnn.train`**: Reproducible training loop and multi-slice evaluation with example-level and exact-program accuracy.
