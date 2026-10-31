"""Build the 3x3 sliding puzzle (8-puzzle) level banks used by envs/slidingpuzzle_env.py.

Breadth-first search from the solved board (tiles 1..8 row-major, blank bottom-right) enumerates all 9!/2 = 181,440
solvable boards with their exact optimal solution length (max 31). Boards are split by optimal length:
  train  all boards with optimal length <= --max_train_moves (default 28: 180,457 boards)
  test   all boards with optimal length >  --max_train_moves (default: 983 boards, lengths 29-31)
Both odd and even lengths are included. Within a split, boards are in BFS order (sorted by optimal length; ties in
the deterministic BFS discovery order), so the level index is stable.
Files: <out>/<split>.npz with puzzle [N,3,3] int8 (0 = blank), moves [N] int16 (optimal length).
Usage (repo root): python data_scripts/build_slidingpuzzle_banks.py [--out data/slidingpuzzle/3x3] [--max_train_moves 28]
"""
import argparse
import os
from collections import deque

import numpy as np

N = 3
ap = argparse.ArgumentParser()
ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'slidingpuzzle', f'{N}x{N}'))
ap.add_argument('--max_train_moves', type=int, default=28)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

goal = tuple(range(1, N * N)) + (0,)
nbrs = [[c for c, ok in ((i - N, i >= N), (i + 1, i % N < N - 1), (i + N, i < N * (N - 1)), (i - 1, i % N > 0)) if ok]
        for i in range(N * N)]
dist = {goal: 0}
queue = deque([goal])
while queue:
    s = queue.popleft()
    b = s.index(0)
    for c in nbrs[b]:
        t = list(s)
        t[b], t[c] = t[c], 0
        t = tuple(t)
        if t not in dist:
            dist[t] = dist[s] + 1
            queue.append(t)
puzzle = np.array(list(dist), np.int8).reshape(-1, N, N)
moves = np.array(list(dist.values()), np.int16)
print('boards', len(moves), 'max optimal length', moves.max())

splits = {'train': moves <= args.max_train_moves, 'test': moves > args.max_train_moves}
for name, mask in splits.items():
    np.savez_compressed(f'{args.out}/{name}.npz', puzzle=puzzle[mask], moves=moves[mask])
    m = moves[mask]
    print(f'{name:6s} n={mask.sum():7d} moves {m.min()}-{m.max()} mean {m.mean():.2f}')
