import numpy as np
import pytest
from ssnn.tasks import make_family, make_split

def test_leakage_checks_and_disjoint_sets():
    for fam_name, states in [("affine", 11), ("permutation", 16)]:
        f = make_family(fam_name, states)
        s = make_split(f, seed=42)
        
        # 1. Validation and test pairs must be disjoint
        val_pairs = set(s.val_heldout_programs)
        test_pairs = s.heldout_pairs - val_pairs
        assert val_pairs.isdisjoint(test_pairs)
        
        # 2. No training program contains any heldout pair
        for p in s.train_programs:
            if len(p) == 2:
                assert p not in s.heldout_pairs, f"Training program {p} contains heldout pair!"
                
        # 3. Validation set programs are not in test set
        val_set = set(s.val_seen_programs) | set(s.val_heldout_programs)
        test_set = set(s.seen_test_programs) | set(s.unseen_test_programs)
        assert val_set.isdisjoint(test_set)
        
        # 4. Strict functional holdout programs MUST NOT share signatures with training programs
        train_sigs = {f.signature(p) for p in s.train_programs}
        for p in s.strict_functional_holdout_programs:
            sig = f.signature(p)
            assert sig not in train_sigs, f"Leakage detected! Program {p} signature {sig} is present in train!"
            
        # 5. Functional leaked programs MUST share signatures with training programs
        for p in s.functional_leaked_programs:
            sig = f.signature(p)
            assert sig in train_sigs, f"Program {p} in leaked set was expected to share a train signature"

def test_strict_functional_holdout_is_non_empty():
    # Both families must produce a non-empty set of genuine functional holdout programs
    for fam_name, states in [("affine", 11), ("permutation", 16)]:
        f = make_family(fam_name, states)
        s = make_split(f, seed=0)
        assert len(s.strict_functional_holdout_programs) > 0, f"No strict functional holdouts found in {fam_name}"
        assert s.stats["n_strict_functional_holdout"] > 0
