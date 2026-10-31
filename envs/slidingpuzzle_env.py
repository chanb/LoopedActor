"""Sliding tile puzzle (8- / 15-puzzle) environment in JAX, with a sparse reward.

Native port of jumanji's SlidingTilePuzzle (jumanji.environments.logic.sliding_tile_puzzle) as used by
Stoix (env=jumanji/slidingtile*: RandomWalkGenerator + SparseRewardFn, time_limit 40). An N x N board
holds tiles 1..N^2-1 and one blank (0); the solved board is 1..N^2-1 in row-major order with the blank
in the bottom-right corner.

Instances are generated procedurally (no level bank), as jumanji's RandomWalkGenerator does: starting
from the solved board, the blank makes `num_random_moves` uniformly random valid moves (immediate
reversals allowed, so the scramble can be shallower than its length and occasionally solved).
Training resets (`reset`) use `num_random_moves` (or, with `min_random_moves` > 0, a length drawn per reset
from [min_random_moves, num_random_moves] with p(k) ∝ k^-random_moves_zipf; 0 = uniform, < 0 favours long scrambles), evaluation resets (`eval_reset`) always use the fixed
`eval_num_random_moves` (default: the same, as in Stoix; raise it to evaluate on deeper scrambles).
With `train_split` / `eval_split` set (env id slidingpuzzle-<N>x<N>-<train>-<eval>), resets instead draw a board uniformly
from a level bank <DATA_DIR>/<N>x<N>/<split>.npz (puzzle [L,N,N] int8, moves [L] int16 optimal length), built by
data_scripts/build_slidingpuzzle_banks.py (3x3: train = optimal length <= 28, test = > 28). No scrambling is done then.

Observation: per-cell one-hot of the tile id (channel k = tile k, channel 0 = blank), cell-major flat
vector of size N*N*N^2 (obs_channels N^2; the actor reshapes it to [N*N, N^2] tokens). No goal vector
(goal_size 0): the solved board is fixed.
Actions (readout head, as in jumanji): move the blank 0 up, 1 right, 2 down, 3 left (the neighbouring
tile slides into the blank). A move off the board is a no-op.
Reward (sparse, jumanji SparseRewardFn): 1 when the board is solved (episode ends), 0 otherwise.
Episode length set in the config (Stoix time_limit 40).
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from ml_collections import config_dict

from envs.utils import State

DATA_DIR = os.environ.get('SLIDINGPUZZLE_DATA_DIR', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'slidingpuzzle'))
MOVES = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]], jnp.int32)  # blank moves: up, right, down, left
BLANK = 0


@struct.dataclass
class SlidingPuzzleData:
    puzzle: jax.Array  # [N, N] int32 tile ids, 0 = blank
    blank: jax.Array   # [2] int32 (row, col) of the blank


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        grid_size=3,               # N: 3 = 8-puzzle, 4 = 15-puzzle
        episode_length=40,
        eval_episode_length=40,    # eval_reset (eval/test_*) episodes; set by make_env (envs/utils.py)
        num_random_moves=10,       # training scramble length
        min_random_moves=0,        # > 0: each training reset draws its length from [this, num_random_moves] (0 = fixed)
        random_moves_zipf=0.0,     # min_random_moves > 0: p(length k) ∝ k^-this (0 = uniform; > 0 favours short, < 0 long scrambles)
        eval_num_random_moves=10,  # evaluation scramble length
        train_split='',            # non-empty: training resets draw from this level bank instead of scrambling
        eval_split='',             # non-empty: evaluation resets draw from this level bank instead of scrambling
        step_penalty=0.0,          # 0 = sparse reward (jumanji SparseRewardFn)
        solve_reward=1.0,
    )


def solved_puzzle(grid_size):
    return jnp.arange(1, grid_size ** 2 + 1, dtype=jnp.int32).at[-1].set(BLANK).reshape(grid_size, grid_size)


class SlidingPuzzleEnv:
    def __init__(self, config: config_dict.ConfigDict = default_config()):
        self._config = config
        self.m = self.n = config.grid_size
        self.grid_size = self.m * self.n
        self.obs_channels = self.grid_size  # one-hot over the N^2 tile ids
        self.action_head = 'readout'
        self.solved = solved_puzzle(self.m)
        self.train_bank = load_bank(self.m, config.train_split) if config.train_split else None
        self.eval_bank = load_bank(self.m, config.eval_split) if config.eval_split else None

    @property
    def action_size(self): return 4

    @property
    def observation_size(self): return self.grid_size * self.obs_channels

    @property
    def goal_size(self): return 0

    def _obs(self, data: SlidingPuzzleData):
        return jax.nn.one_hot(data.puzzle.reshape(-1), self.obs_channels, dtype=jnp.float32).reshape(-1)

    def move(self, puzzle, blank, action):
        """Move the blank by MOVES[action] (no-op off the board). Returns (puzzle, blank, moved)."""
        target = blank + MOVES[action]
        ok = jnp.all((target >= 0) & (target < self.m))
        target = jnp.where(ok, target, blank)
        tile = puzzle[target[0], target[1]]
        new_puzzle = puzzle.at[blank[0], blank[1]].set(tile).at[target[0], target[1]].set(BLANK)
        return jnp.where(ok, new_puzzle, puzzle), target, ok

    def scramble(self, rng, num_moves, length=None):
        """Random walk of the blank from the solved board (jumanji RandomWalkGenerator). `length` (traced,
        <= num_moves) stops the walk early; None walks all num_moves steps."""
        def body(carry, x):
            (puzzle, blank), (i, key) = carry, x
            valid = jnp.all((blank + MOVES >= 0) & (blank + MOVES < self.m), axis=-1)
            action = jax.random.choice(key, 4, p=valid / valid.sum())
            new_puzzle, new_blank, _ = self.move(puzzle, blank, action)
            keep = i < (num_moves if length is None else length)
            return (jnp.where(keep, new_puzzle, puzzle), jnp.where(keep, new_blank, blank)), None
        start = (self.solved, jnp.array([self.m - 1, self.m - 1], jnp.int32))
        (puzzle, blank), _ = jax.lax.scan(body, start, (jnp.arange(num_moves), jax.random.split(rng, num_moves)))
        return SlidingPuzzleData(puzzle=puzzle, blank=blank)

    def _make_state(self, rng, data: SlidingPuzzleData):
        return State(
            data=data, obs=self._obs(data),
            reward=jnp.array(0.0, jnp.float32), done=jnp.array(0.0, jnp.float32),
            metrics={'success': 0.0, 'reward': 0.0},
            info={'rng': rng, 'target_goal': jnp.zeros((0,), jnp.float32)},
        )

    def _reset_from_bank(self, rng, bank):
        rng, key = jax.random.split(rng)
        puzzle = bank['puzzle'][jax.random.randint(key, (), 0, bank['puzzle'].shape[0])]
        b = jnp.argmin(puzzle.reshape(-1))  # the blank (0) is the only zero
        return self._make_state(rng, SlidingPuzzleData(puzzle=puzzle, blank=jnp.stack([b // self.m, b % self.m])))

    def reset(self, rng: jax.Array) -> State:
        if self.train_bank is not None:
            return self._reset_from_bank(rng, self.train_bank)
        rng, key = jax.random.split(rng)
        cfg = self._config
        if cfg.min_random_moves <= 0:
            return self._make_state(rng, self.scramble(key, cfg.num_random_moves))
        rng, key_len = jax.random.split(rng)
        lengths = jnp.arange(cfg.min_random_moves, cfg.num_random_moves + 1)
        length = lengths[jax.random.categorical(key_len, -cfg.random_moves_zipf * jnp.log(lengths))]
        return self._make_state(rng, self.scramble(key, cfg.num_random_moves, length))

    def eval_reset(self, rng: jax.Array) -> State:
        if self.eval_bank is not None:
            return self._reset_from_bank(rng, self.eval_bank)
        rng, key = jax.random.split(rng)
        return self._make_state(rng, self.scramble(key, self._config.eval_num_random_moves))

    def step(self, state: State, action: jax.Array) -> State:
        d = state.data
        puzzle, blank, _ = self.move(d.puzzle, d.blank, action)
        success = jnp.all(puzzle == self.solved).astype(jnp.float32)
        cfg = self._config
        reward = cfg.solve_reward * success - cfg.step_penalty
        new_data = SlidingPuzzleData(puzzle=puzzle, blank=blank)
        return State(
            data=new_data, obs=self._obs(new_data), reward=reward, done=success,
            metrics={**state.metrics, 'success': success, 'reward': reward},
            info={**state.info, 'target_goal': state.info['target_goal']},
        )


def load_bank(grid_size, split):
    z = np.load(os.path.join(DATA_DIR, f'{grid_size}x{grid_size}', f'{split}.npz'))
    return dict(puzzle=jnp.asarray(z['puzzle'].astype(np.int32)), moves=jnp.asarray(z['moves'].astype(np.int32)))


def render_ascii(puzzle):
    p = np.asarray(puzzle)
    width = len(str(p.size - 1))
    return '\n'.join(' '.join('.'.rjust(width) if v == BLANK else str(v).rjust(width) for v in row) for row in p)
