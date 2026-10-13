# Glossary of Compositional Terms

This glossary defines standard terminology used across the NAIR research harness and protocol documents.

---

### 1. Truth-Table Signature
For a deterministic program $P$ executed over a finite state space $\mathbb{Z}_S$, its truth-table signature is the complete input-output tuple:
$$\text{sig}(P) = (F(0), F(1), \dots, F(S-1)) \in \{0, \dots, S-1\}^S$$
Two syntactically distinct programs $P_1 \neq P_2$ are **functionally equivalent** if and only if $\text{sig}(P_1) = \text{sig}(P_2)$.

### 2. Strict Functional Holdout
A test program sequence containing held-out operation pairs whose resulting truth-table signature $\text{sig}(P)$ is **strictly absent** from all training program signatures:
$$\text{sig}(P) \notin \{\text{sig}(P_{\text{train}}) \mid P_{\text{train}} \in \mathcal{D}_{\text{train}}\}$$
This isolates genuine compositional inference from associative pattern recall.

### 3. Functionally Leaked Composition
A test program containing held-out token pairs whose resulting truth table $\text{sig}(P)$ was **already produced** by some alternative program in the training set. Evaluating on leaked sets produces inflated generalization metrics.

### 4. Diagrammatic Execution Order
Left-to-right sequential composition order:
$$y = f_{op_k}(\dots f_{op_1}(f_{op_0}(x))\dots)$$
Operation $op_0$ is evaluated first on input $x$, followed by $op_1$, etc.

### 5. Active Parameters per Step
The number of model parameters actively participating in forward computation for a single execution step. For example, in an explicit modular network with 12 modules, only 1 module (2,251 parameters) is active during any single step, despite storing 27,012 parameters in memory.
