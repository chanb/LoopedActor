"""Build the sliding puzzle level banks used by envs/slidingpuzzle_env.py (goal: tiles 1..N^2-1 row-major, blank bottom-right).

--grid 3 (8-puzzle): breadth-first search from the solved board enumerates all 9!/2 = 181,440 solvable boards with their
exact optimal solution length (max 31). Boards are split by optimal length:
  train  all boards with optimal length <= --max_train_moves (default 28: 180,457 boards)
  test   all boards with optimal length >  --max_train_moves (default: 983 boards, lengths 29-31)
Within a split, boards are in BFS order (sorted by optimal length; ties in the deterministic BFS discovery order).

--grid 4 (15-puzzle, 16!/2 ~ 1.05e13 boards, so the banks are samples; uses data_scripts/slidingpuzzle4.c, compiled with gcc):
  train  2M boards with optimal length <= 60, balanced in length: every board with length <= 14 (61,865; fewer than the
         per-length share) and the rest split evenly over lengths 15..60 (42,133-42,134 each). Lengths 15..26 are drawn
         without replacement from exact BFS layers; lengths 27..60 are uniform random boards solved optimally (IDA*) and
         kept while their length has quota left. Either way each board is uniform among the boards of its length.
  test   100k boards with optimal length > 60, uniform over all such boards (lengths in their natural proportions:
         mostly 61-64, about 1-2% at 69-70). About 4M random boards are solved; the run takes several hours on 48 threads.
Within a split, boards are sorted by optimal length (stable), so the level index is stable.

Both odd and even lengths are included. Files: <out>/<split>.npz with puzzle [N,G,G] int8 (0 = blank), moves [N] int16
(optimal length).
Usage (repo root): python data_scripts/build_slidingpuzzle_banks.py [--grid 3|4] [--out data/slidingpuzzle/<G>x<G>]
"""
import argparse
import os
import subprocess
from collections import deque

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--grid', type=int, default=3, choices=[3, 4])
ap.add_argument('--out', default='')
ap.add_argument('--max_train_moves', type=int, default=28, help='--grid 3 only')
args = ap.parse_args()
N = args.grid
args.out = args.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'slidingpuzzle', f'{N}x{N}')
os.makedirs(args.out, exist_ok=True)


def save(name, puzzle, moves):
    order = np.argsort(moves, kind='stable')
    np.savez_compressed(f'{args.out}/{name}.npz', puzzle=puzzle[order], moves=moves[order])
    print(f'{name:6s} n={len(moves):8d} moves {moves.min()}-{moves.max()} mean {moves.mean():.2f}', flush=True)


def build_3x3():
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
    for name, mask in (('train', moves <= args.max_train_moves), ('test', moves > args.max_train_moves)):
        save(name, puzzle[mask], moves[mask])


def decode(boards):
    """u64 boards (nibble i = tile at cell i) -> [B, 4, 4] int8."""
    boards = np.asarray(boards, np.uint64)
    return ((boards[:, None] >> (4 * np.arange(16, dtype=np.uint64))) & np.uint64(15)).astype(np.int8).reshape(-1, 4, 4)


def build_4x4(train_size=2_000_000, test_size=100_000, max_train=60, bfs_depth=26):
    work = os.path.join(args.out, 'build')  # compiled tool, cached pattern databases, BFS layers, sampler outputs
    os.makedirs(work, exist_ok=True)
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'slidingpuzzle4.c')
    subprocess.run(['gcc', '-O3', '-march=native', '-fopenmp', src, '-o', 'sp4'], cwd=work, check=True)

    def sample(name, seed, quota, max_total):
        with open(f'{work}/{name}_quota.txt', 'w') as f:
            f.write(' '.join(map(str, quota)))
        with open(f'{work}/{name}.txt', 'w') as f:
            subprocess.run(['./sp4', 'sample', str(seed), f'{name}_quota.txt', str(max_total)], cwd=work, stdout=f, check=True)
        rows = np.loadtxt(f'{work}/{name}.txt', dtype=str, ndmin=2)
        return decode([int(h, 16) for h in rows[:, 0]]), rows[:, 1].astype(np.int16)

    subprocess.run(['./sp4', 'bfs', str(bfs_depth), '.'], cwd=work, check=True)
    layers = [np.array([sum((i + 1) << (4 * i) for i in range(15))], np.uint64)]  # d=0: the solved board
    layers += [np.fromfile(f'{work}/d{d:02d}.u64', np.uint64) for d in range(1, bfs_depth + 1)]
    small = [d for d in range(max_train + 1) if d < len(layers) and len(layers[d]) * (max_train + 1 - d) <= train_size - sum(len(l) for l in layers[:d])]
    n_small = sum(len(layers[d]) for d in small)
    rest = list(range(len(small), max_train + 1))
    share = [(train_size - n_small) // len(rest) + (i < (train_size - n_small) % len(rest)) for i in range(len(rest))]
    rng = np.random.RandomState(0)
    boards, moves = [layers[d] for d in small], [np.full(len(layers[d]), d, np.int16) for d in small]
    for d, k in zip(rest, share):
        if d <= bfs_depth:
            boards.append(rng.choice(layers[d], k, replace=False))
            moves.append(np.full(k, d, np.int16))
    quota = [0] * 81
    for d, k in zip(rest, share):
        if d > bfs_depth:
            quota[d] = k
    train_puzzle, train_moves = sample('train', 1, quota, 0)
    save('train', np.concatenate([decode(np.concatenate(boards)), train_puzzle]), np.concatenate(moves + [train_moves]))
    test_puzzle, test_moves = sample('test', 2, [0] * (max_train + 1) + [-1] * (80 - max_train), test_size)
    save('test', test_puzzle, test_moves)


build_3x3() if N == 3 else build_4x4()
