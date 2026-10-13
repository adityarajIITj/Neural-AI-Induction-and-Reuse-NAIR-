"""Deterministic finite-state composition tasks with truth-table signatures and functional leakage tracking."""
from __future__ import annotations
from dataclasses import dataclass, field
from itertools import product
from typing import Dict, Any, List, Tuple, Set
import numpy as np

# --- Independent Mathematical Reference Implementations ---

def affine_ref(op_idx: int, x: np.ndarray, n_states: int) -> np.ndarray:
    """Independent reference implementation of affine transformations on Z_p."""
    x = np.asarray(x, dtype=np.int64)
    ref_funcs = [
        lambda v: (v + 1) % n_states,
        lambda v: (v + 2) % n_states,
        lambda v: (v - 1) % n_states,
        lambda v: (-v) % n_states,
        lambda v: (2 * v) % n_states,
        lambda v: (3 * v) % n_states,
        lambda v: (v + 3) % n_states,
        lambda v: (v + 5) % n_states,
        lambda v: (2 * v + 1) % n_states,
        lambda v: (2 * v + 3) % n_states,
        lambda v: (3 * v + 1) % n_states,
        lambda v: (3 * v + 2) % n_states,
    ]
    if op_idx < 0 or op_idx >= len(ref_funcs):
        raise ValueError(f"Invalid affine op_idx {op_idx}")
    return ref_funcs[op_idx](x)

# PRESENT 4-bit S-box lookup table
_PRESENT_SBOX = np.array([12, 5, 6, 11, 9, 0, 10, 13, 3, 14, 15, 8, 4, 7, 1, 2], dtype=np.int64)
_PRESENT_INV_SBOX = np.zeros(16, dtype=np.int64)
for _i, _v in enumerate(_PRESENT_SBOX):
    _PRESENT_INV_SBOX[_v] = _i

def permutation_ref(op_idx: int, x: np.ndarray, n_states: int = 16) -> np.ndarray:
    """Independent reference implementation of 4-bit non-linear permutation generators."""
    x = np.asarray(x, dtype=np.int64)
    if n_states != 16:
        raise ValueError("permutation family currently requires n_states=16 (4-bit state space)")
    
    # 12 non-linear / permutation generators:
    # 0: cyclic shift right 1
    # 1: cyclic shift right 3
    # 2: cyclic shift right 7
    # 3: bitwise NOT (15 - x)
    # 4: bitwise XOR with 1
    # 5: bitwise XOR with 5
    # 6: bit reversal of 4-bit integer
    # 7: swap adjacent bits ((x & 10) >> 1 | (x & 5) << 1)
    # 8: PRESENT S-box
    # 9: Inverse PRESENT S-box
    # 10: Multiplicative mod 16: (3 * x) % 16
    # 11: Affine mod 16: (5 * x + 1) % 16
    ref_funcs = [
        lambda v: (v + 1) % 16,
        lambda v: (v + 3) % 16,
        lambda v: (v + 7) % 16,
        lambda v: 15 - v,
        lambda v: v ^ 1,
        lambda v: v ^ 5,
        lambda v: np.array([0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15], dtype=np.int64)[v],
        lambda v: ((v & 10) >> 1) | ((v & 5) << 1),
        lambda v: _PRESENT_SBOX[v],
        lambda v: _PRESENT_INV_SBOX[v],
        lambda v: (3 * v) % 16,
        lambda v: (5 * v + 1) % 16,
    ]
    if op_idx < 0 or op_idx >= len(ref_funcs):
        raise ValueError(f"Invalid permutation op_idx {op_idx}")
    return ref_funcs[op_idx](x)


@dataclass(frozen=True)
class Family:
    """Represents a finite-state transformation family over a bounded integer domain.

    Attributes:
        name: Identifier of the task family ('affine' or 'permutation').
        n_states: Total discrete cardinality of the state domain (S).
        n_ops: Number of primitive operations available in the vocabulary.
        tables: Lookup tensor of shape [n_ops, n_states] mapping (op, state) -> next_state.
    """
    name: str
    n_states: int
    n_ops: int
    tables: np.ndarray  # [operation, state] -> state

    def run(self, program: Tuple[int, ...] | List[int], x: np.ndarray | List[int] | int) -> np.ndarray:
        """Execute a sequence of operation tokens sequentially from left to right."""
        if isinstance(x, int):
            val = int(x)
            for op in program:
                val = int(self.tables[int(op), val])
            return np.array([val], dtype=np.int64)
        y = np.asarray(x, dtype=np.int64).copy()
        for op in program:
            y = self.tables[int(op), y]
        return y

    def signature(self, program: Tuple[int, ...] | List[int]) -> Tuple[int, ...]:
        """Compute full truth-table signature: (F(0), F(1), ..., F(S-1))."""
        inputs = np.arange(self.n_states, dtype=np.int64)
        outputs = self.run(program, inputs)
        return tuple(int(v) for v in outputs)

    def reference_run(self, op_idx: int, x: np.ndarray) -> np.ndarray:
        """Run independent reference implementation for operation op_idx."""
        if self.name == "affine":
            return affine_ref(op_idx, x, self.n_states)
        elif self.name == "permutation":
            return permutation_ref(op_idx, x, self.n_states)
        else:
            raise ValueError(f"Unknown family {self.name}")

    def verify_primitives(self) -> bool:
        """Verify all primitive tables against their independent reference."""
        x = np.arange(self.n_states, dtype=np.int64)
        for op in range(self.n_ops):
            ref_out = self.reference_run(op, x)
            table_out = self.tables[op, x]
            if not np.array_equal(ref_out, table_out):
                return False
        return True

    def describe(self) -> str:
        """Return a formatted string describing the task family properties."""
        all_bijective = all(len(np.unique(self.tables[op])) == self.n_states for op in range(self.n_ops))
        return (
            f"Family(name='{self.name}', n_states={self.n_states}, n_ops={self.n_ops}, "
            f"all_bijective={all_bijective})"
        )

    def get_signature_table(self) -> np.ndarray:
        """Return truth-table signatures for all primitive (depth-1) operations [n_ops, n_states]."""
        return np.stack([np.array(self.signature((op,)), dtype=np.int64) for op in range(self.n_ops)])


@dataclass
class Split:
    """Encapsulates partitioned program sets ensuring strict leakage separation.

    Separates token-pair holdouts from true functional holdouts, and provides
    distinct sets for training, validation, and multi-slice evaluation.
    """
    train_programs: list[tuple[int, ...]]
    seen_test_programs: list[tuple[int, ...]]
    unseen_test_programs: list[tuple[int, ...]]
    heldout_pairs: set[tuple[int, int]]
    val_seen_programs: list[tuple[int, ...]] = field(default_factory=list)
    val_heldout_programs: list[tuple[int, ...]] = field(default_factory=list)
    strict_functional_holdout_programs: list[tuple[int, ...]] = field(default_factory=list)
    functional_leaked_programs: list[tuple[int, ...]] = field(default_factory=list)
    depth4_extrapolation_programs: list[tuple[int, ...]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def make_family(name: str = "affine", n_states: int = 11) -> Family:
    """Construct a validated Family instance ('affine' or 'permutation')."""
    if name == "affine":
        x = np.arange(n_states, dtype=np.int64)
        funcs = [
            (x + 1) % n_states, (x + 2) % n_states, (x - 1) % n_states,
            (-x) % n_states, (2 * x) % n_states, (3 * x) % n_states,
            (x + 3) % n_states, (x + 5) % n_states,
            (2 * x + 1) % n_states, (2 * x + 3) % n_states,
            (3 * x + 1) % n_states, (3 * x + 2) % n_states,
        ]
        tables = np.stack([f % n_states for f in funcs])
        return Family(name=name, n_states=n_states, n_ops=len(tables), tables=tables)

    elif name == "permutation":
        n_states = 16
        x = np.arange(n_states, dtype=np.int64)
        bit_rev = np.array([0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15], dtype=np.int64)
        swap_adj = ((x & 10) >> 1) | ((x & 5) << 1)
        funcs = [
            (x + 1) % 16,
            (x + 3) % 16,
            (x + 7) % 16,
            15 - x,
            x ^ 1,
            x ^ 5,
            bit_rev,
            swap_adj,
            _PRESENT_SBOX,
            _PRESENT_INV_SBOX,
            (3 * x) % 16,
            (5 * x + 1) % 16,
        ]
        tables = np.stack(funcs)
        return Family(name=name, n_states=n_states, n_ops=len(tables), tables=tables)

    else:
        raise ValueError(f"Unknown family: {name}")


def make_split(
    family: Family,
    seed: int = 0,
    heldout_fraction: float = 0.25,
    val_fraction: float = 0.333,
    max_d3_eval: int = 800,
    max_d4_eval: int = 400,
) -> Split:
    """Generate auditable data splits with separate validation, token-pair holdout,

    and strict functional holdout tracking.
    """
    rng = np.random.default_rng(seed)
    all_pairs = list(product(range(family.n_ops), repeat=2))
    rng.shuffle(all_pairs)

    total_heldout_count = max(2, int(len(all_pairs) * heldout_fraction))
    val_heldout_count = max(1, int(total_heldout_count * val_fraction))
    test_heldout_count = total_heldout_count - val_heldout_count

    test_heldout_pairs = set(all_pairs[:test_heldout_count])
    val_heldout_pairs = set(all_pairs[test_heldout_count:total_heldout_count])
    all_heldout_pairs = test_heldout_pairs | val_heldout_pairs

    train_pairs = [p for p in all_pairs if p not in all_heldout_pairs]

    # Training programs: all depth 1 primitives + non-heldout pairs
    train_programs = [(i,) for i in range(family.n_ops)] + train_pairs

    # Truth-table signatures of all training programs
    train_signatures = {family.signature(p) for p in train_programs}

    # Validation programs
    # Seen-pair validation: sample from train pairs
    val_seen = [train_pairs[i] for i in rng.choice(len(train_pairs), size=min(16, len(train_pairs)), replace=False)]
    val_heldout = list(val_heldout_pairs)

    # Test candidate programs at depth 3
    # Candidate 1: Seen pairs composed to depth 3
    seen_d3_all = [
        (i, j, k) for i, j, k in product(range(family.n_ops), repeat=3)
        if (i, j) not in all_heldout_pairs and (j, k) not in all_heldout_pairs
    ]
    # Candidate 2: Contains at least one test heldout pair
    unseen_d3_all = [
        (i, j, k) for i, j, k in product(range(family.n_ops), repeat=3)
        if (i, j) in test_heldout_pairs or (j, k) in test_heldout_pairs
    ]

    # Subsample if pool is large, while keeping reproducible order
    rng.shuffle(seen_d3_all)
    rng.shuffle(unseen_d3_all)
    seen_test_programs = seen_d3_all[:max_d3_eval]
    unseen_test_candidates = unseen_d3_all[:max_d3_eval]

    # Partition unseen test candidates into strict functional holdout vs functionally leaked
    strict_functional_holdout: list[tuple[int, ...]] = []
    functional_leaked: list[tuple[int, ...]] = []

    for p in unseen_test_candidates:
        sig = family.signature(p)
        if sig in train_signatures:
            functional_leaked.append(p)
        else:
            strict_functional_holdout.append(p)

    # Depth 4 extrapolation programs (composed beyond training depth 2)
    d4_all = list(product(range(family.n_ops), repeat=4))
    rng.shuffle(d4_all)
    depth4_extrapolation = d4_all[:max_d4_eval]

    stats = {
        "family": family.name,
        "n_ops": family.n_ops,
        "n_states": family.n_states,
        "n_train_programs": len(train_programs),
        "n_train_unique_signatures": len(train_signatures),
        "n_test_heldout_pairs": len(test_heldout_pairs),
        "n_val_heldout_pairs": len(val_heldout_pairs),
        "n_seen_test_programs": len(seen_test_programs),
        "n_unseen_test_candidates": len(unseen_test_candidates),
        "n_strict_functional_holdout": len(strict_functional_holdout),
        "n_functional_leaked": len(functional_leaked),
        "strict_functional_holdout_rate": len(strict_functional_holdout) / max(1, len(unseen_test_candidates)),
        "functional_leakage_rate": len(functional_leaked) / max(1, len(unseen_test_candidates)),
    }

    return Split(
        train_programs=train_programs,
        seen_test_programs=seen_test_programs,
        unseen_test_programs=unseen_test_candidates,
        heldout_pairs=all_heldout_pairs,
        val_seen_programs=val_seen,
        val_heldout_programs=val_heldout,
        strict_functional_holdout_programs=strict_functional_holdout,
        functional_leaked_programs=functional_leaked,
        depth4_extrapolation_programs=depth4_extrapolation,
        stats=stats,
    )


def sample_examples(
    family: Family,
    programs: List[Tuple[int, ...]],
    per_program: int = 16,
    seed: int = 0,
    exhaustive: bool = False,
) -> List[Tuple[Tuple[int, ...], int, int]]:
    """Sample input-output pairs. If exhaustive is True or per_program >= n_states,

    evaluates all states in order for deterministic truth-table evaluation.
    """
    rng = np.random.default_rng(seed)
    rows: List[Tuple[Tuple[int, ...], int, int]] = []

    if exhaustive:
        xs = np.arange(family.n_states, dtype=np.int64)
        for p in programs:
            ys = family.run(p, xs)
            for x, y in zip(xs, ys):
                rows.append((tuple(p), int(x), int(y)))
    else:
        for p in programs:
            xs = rng.integers(0, family.n_states, size=per_program)
            ys = family.run(p, xs)
            for x, y in zip(xs, ys):
                rows.append((tuple(p), int(x), int(y)))
        rng.shuffle(rows)

    return rows


def batchify(rows, n_ops: int, n_states: int, max_depth: int = 4):
    import torch
    # Use actual max program length in rows if greater than default max_depth
    actual_max = max((len(r[0]) for r in rows), default=max_depth)
    seq_len = max(max_depth, actual_max)

    prog = torch.full((len(rows), seq_len), n_ops, dtype=torch.long)  # n_ops is PAD token
    x = torch.tensor([r[1] for r in rows], dtype=torch.long)
    y = torch.tensor([r[2] for r in rows], dtype=torch.long)
    lengths = torch.tensor([len(r[0]) for r in rows], dtype=torch.long)

    for i, (p, _, _) in enumerate(rows):
        prog[i, :len(p)] = torch.tensor(p, dtype=torch.long)

    return prog, x, y, lengths
