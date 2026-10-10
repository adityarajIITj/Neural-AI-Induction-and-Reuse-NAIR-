from itertools import product
import numpy as np

n_states = 16
x = np.arange(n_states, dtype=np.int64)

# 12 non-linear / permutation operations on 4-bit state space (0..15)
bit_rev = np.array([0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15], dtype=np.int64)
swap_adj = ((x & 10) >> 1) | ((x & 5) << 1)
sbox = np.array([12, 5, 6, 11, 9, 0, 10, 13, 3, 14, 15, 8, 4, 7, 1, 2], dtype=np.int64)
inv_sbox = np.zeros(16, dtype=np.int64)
for i, v in enumerate(sbox): inv_sbox[v] = i
mul3 = (3 * x) % 16
mul5_1 = (5 * x + 1) % 16

funcs = [
    (x + 1) % 16, (x + 3) % 16, (x + 7) % 16,
    15 - x, x ^ 1, x ^ 5,
    bit_rev, swap_adj,
    sbox, inv_sbox,
    mul3, mul5_1
]
tables = np.stack(funcs)

print("All are bijections:", all(len(np.unique(t)) == 16 for t in tables))

for d in range(1, 4):
    sigs = set()
    total = 0
    for p in product(range(12), repeat=d):
        total += 1
        y = x.copy()
        for op in p:
            y = tables[op, y]
        sigs.add(tuple(y))
    print(f"Depth {d}: {total} programs -> {len(sigs)} unique signatures ({len(sigs)/total*100:.1f}%)")
