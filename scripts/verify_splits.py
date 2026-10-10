from itertools import product
import numpy as np

def test_split_design():
    print("Testing split design...")
    # Let us check how many pairs for train, val, test
    n_ops = 12
    n_states = 11
    x = np.arange(n_states)
    
    # Affine tables
    funcs = [
        (x + 1) % n_states, (x + 2) % n_states, (x - 1) % n_states,
        (-x) % n_states, (2*x) % n_states, (3*x) % n_states,
        (x + 3) % n_states, (x + 5) % n_states,
        (2*x + 1) % n_states, (2*x + 3) % n_states,
        (3*x + 1) % n_states, (3*x + 2) % n_states,
    ]
    tables = np.stack(funcs)
    
    def run(p, state):
        y = np.asarray(state, dtype=np.int64).copy()
        for op in p:
            y = tables[int(op), y]
        return tuple(int(v) for v in y)

    rng = np.random.default_rng(0)
    all_pairs = list(product(range(n_ops), repeat=2))
    rng.shuffle(all_pairs)
    
    # 25% heldout = 36 pairs
    # Split heldout pairs: 12 for validation, 24 for test
    heldout_test = set(all_pairs[:24])
    heldout_val = set(all_pairs[24:36])
    heldout_all = heldout_test | heldout_val
    train_pairs = [p for p in all_pairs if p not in heldout_all]
    
    train_programs = [(i,) for i in range(n_ops)] + train_pairs
    train_sigs = {run(p, x) for p in train_programs}
    
    print(f"Train programs: {len(train_programs)}, unique signatures: {len(train_sigs)}")
    print(f"Val heldout pairs: {len(heldout_val)}, Test heldout pairs: {len(heldout_test)}")
    
    # Generate test depth 3 programs containing test heldout pairs
    test_d3 = [p for p in product(range(n_ops), repeat=3) if (p[0], p[1]) in heldout_test or (p[1], p[2]) in heldout_test]
    
    strict_functional = [p for p in test_d3 if run(p, x) not in train_sigs]
    functional_leaked = [p for p in test_d3 if run(p, x) in train_sigs]
    
    print(f"Test depth 3 unseen-pair programs: {len(test_d3)}")
    print(f"  Strict functional holdouts: {len(strict_functional)} ({len(strict_functional)/len(test_d3)*100:.1f}%)")
    print(f"  Functionally leaked: {len(functional_leaked)} ({len(functional_leaked)/len(test_d3)*100:.1f}%)")

if __name__ == "__main__":
    test_split_design()
