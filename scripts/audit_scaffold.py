import numpy as np
from itertools import product
from ssnn.tasks import make_family, make_split

def audit_family(family_name="affine", n_states=11):
    f = make_family(family_name, n_states)
    x = np.arange(f.n_states)
    print(f"=== Auditing Family: {f.name} (n_ops={f.n_ops}, n_states={f.n_states}) ===")
    
    # Analyze signatures by depth
    for d in range(1, 4):
        sigs = {}
        for p in product(range(f.n_ops), repeat=d):
            sig = tuple(int(v) for v in f.run(p, x))
            sigs.setdefault(sig, []).append(p)
        print(f"Depth {d}: {f.n_ops**d} total programs -> {len(sigs)} unique functional signatures")
        # Example of collision
        collisions = [progs for progs in sigs.values() if len(progs) > 1]
        print(f"  Colliding programs: {sum(len(c) for c in collisions)} across {len(collisions)} distinct signatures")

    # Analyze split
    split = make_split(f, seed=0)
    train_sigs = {tuple(int(v) for v in f.run(p, x)): p for p in split.train_programs}
    seen_test_sigs = {tuple(int(v) for v in f.run(p, x)): p for p in split.seen_test_programs}
    unseen_test_sigs = {tuple(int(v) for v in f.run(p, x)): p for p in split.unseen_test_programs}

    print("\n=== Split Leakage Analysis (Seed 0) ===")
    print(f"Train programs: {len(split.train_programs)}, unique signatures: {len(train_sigs)}")
    print(f"Seen test programs: {len(split.seen_test_programs)}, unique signatures: {len(seen_test_sigs)}")
    print(f"Unseen test programs: {len(split.unseen_test_programs)}, unique signatures: {len(unseen_test_sigs)}")

    # Check functional leakage
    unseen_leaked_sigs = set(unseen_test_sigs.keys()) & set(train_sigs.keys())
    seen_leaked_sigs = set(seen_test_sigs.keys()) & set(train_sigs.keys())
    
    strict_unseen_progs = [p for p in split.unseen_test_programs if tuple(int(v) for v in f.run(p, x)) not in train_sigs]
    leaked_unseen_progs = [p for p in split.unseen_test_programs if tuple(int(v) for v in f.run(p, x)) in train_sigs]

    print(f"Unseen test signatures already present in Train: {len(unseen_leaked_sigs)} / {len(unseen_test_sigs)} ({len(unseen_leaked_sigs)/len(unseen_test_sigs)*100:.1f}%)")
    print(f"Seen test signatures already present in Train: {len(seen_leaked_sigs)} / {len(seen_test_sigs)} ({len(seen_leaked_sigs)/len(seen_test_sigs)*100:.1f}%)")
    print(f"Unseen test programs breakdown:")
    print(f"  Strict Functional Holdout: {len(strict_unseen_progs)} / {len(split.unseen_test_programs)} ({len(strict_unseen_progs)/len(split.unseen_test_programs)*100:.1f}%)")
    print(f"  Functionally Leaked: {len(leaked_unseen_progs)} / {len(split.unseen_test_programs)} ({len(leaked_unseen_progs)/len(split.unseen_test_programs)*100:.1f}%)")

if __name__ == "__main__":
    audit_family()
