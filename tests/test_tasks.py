import numpy as np
import pytest
from ssnn.tasks import make_family, make_split, sample_examples, affine_ref, permutation_ref

def test_ground_truth_execution_is_deterministic():
    f = make_family("affine", 11)
    x = np.arange(f.n_states)
    assert np.array_equal(f.run((0,), x), f.tables[0])
    assert np.array_equal(f.run((0, 1), x), f.tables[1, f.tables[0, x]])

def test_all_primitives_match_independent_reference():
    # Affine family
    f_aff = make_family("affine", 11)
    assert f_aff.verify_primitives()
    x = np.arange(11)
    for op in range(f_aff.n_ops):
        assert np.array_equal(f_aff.tables[op], affine_ref(op, x, 11))

    # Permutation family
    f_perm = make_family("permutation", 16)
    assert f_perm.verify_primitives()
    x16 = np.arange(16)
    for op in range(f_perm.n_ops):
        assert np.array_equal(f_perm.tables[op], permutation_ref(op, x16, 16))

def test_execution_order_hand_worked_examples():
    f = make_family("affine", 11)
    # op 0: x + 1 mod 11
    # op 4: 2*x mod 11
    # Program (0, 4) executes op 0 then op 4: (2 * (x + 1)) % 11
    # For x = 3: (3 + 1) = 4 -> 2 * 4 = 8
    assert f.run((0, 4), [3])[0] == 8

    # Program (4, 0) executes op 4 then op 0: (2*x + 1) % 11
    # For x = 3: 2 * 3 = 6 -> 6 + 1 = 7
    assert f.run((4, 0), [3])[0] == 7
    assert f.run((0, 4), [3])[0] != f.run((4, 0), [3])[0]

def test_truth_table_signatures_and_equivalence():
    f = make_family("affine", 11)
    # op 0: x + 1; op 1: x + 2.
    # Program (0, 0): (x + 1) + 1 = x + 2.
    # Program (1,): x + 2.
    # Hand-worked test: (0, 0) and (1,) MUST have identical signatures!
    sig_00 = f.signature((0, 0))
    sig_1 = f.signature((1,))
    assert sig_00 == sig_1

    # op 6: x + 3. Program (0, 1) is (x + 1) + 2 = x + 3.
    assert f.signature((0, 1)) == f.signature((6,))

def test_sampling_is_reproducible():
    f = make_family("affine", 11)
    s = make_split(f, seed=3)
    a = sample_examples(f, s.train_programs[:3], per_program=5, seed=10)
    b = sample_examples(f, s.train_programs[:3], per_program=5, seed=10)
    assert a == b

def test_family_describe():
    f_aff = make_family("affine", 11)
    desc_aff = f_aff.describe()
    assert "affine" in desc_aff
    assert "all_bijective=True" in desc_aff

    f_perm = make_family("permutation", 16)
    desc_perm = f_perm.describe()
    assert "permutation" in desc_perm
    assert "all_bijective=True" in desc_perm

def test_permutation_bijections_have_full_cardinality():
    f_perm = make_family("permutation", 16)
    for op in range(f_perm.n_ops):
        unique_outputs = set(f_perm.tables[op])
        assert len(unique_outputs) == 16, f"Op {op} is not a full bijection over {16} states"

def test_family_get_signature_table():
    f = make_family("affine", 11)
    sig_table = f.get_signature_table()
    assert sig_table.shape == (f.n_ops, f.n_states)
    assert np.array_equal(sig_table, f.tables)
