"""Sliding puzzle scramble tests (CPU, a few seconds)."""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.slidingpuzzle_env import SlidingPuzzleEnv, default_config  # noqa: E402


def _env(**kw):
    config = default_config()
    config.grid_size = 4
    config.update(kw)
    return SlidingPuzzleEnv(config)


def _manhattan(env, puzzle):
    """Total Manhattan distance of the tiles from their solved cells, [B]."""
    p = np.asarray(puzzle).reshape(len(puzzle), -1)
    tile = np.where(p == 0, env.grid_size, p) - 1  # solved cell index of each tile (blank -> last cell)
    cell = np.arange(env.grid_size)
    d = np.abs(tile // env.m - cell // env.m) + np.abs(tile % env.m - cell % env.m)
    return np.where(p == 0, 0, d).sum(-1)


def _boards(env, reset, n=512):
    return jax.vmap(getattr(env, reset))(jax.random.split(jax.random.PRNGKey(0), n)).data.puzzle


def test_fixed_length_unchanged():
    # min_random_moves = 0 (default): reset is the plain num_random_moves walk, as before.
    env = _env(num_random_moves=50)
    rng, key = jax.random.split(jax.random.PRNGKey(3))
    assert jnp.array_equal(env.reset(jax.random.PRNGKey(3)).data.puzzle, env.scramble(key, 50).puzzle)


def test_eval_reset_stays_fixed_length():
    env = _env(num_random_moves=200, min_random_moves=1, eval_num_random_moves=200)
    rng, key = jax.random.split(jax.random.PRNGKey(3))
    assert jnp.array_equal(env.eval_reset(jax.random.PRNGKey(3)).data.puzzle, env.scramble(key, 200).puzzle)


def test_mixed_scramble_lengths():
    mixed = _env(num_random_moves=200, min_random_moves=1, eval_num_random_moves=200)
    fixed = _env(num_random_moves=200)
    d_mixed, d_fixed = _manhattan(mixed, _boards(mixed, 'reset')), _manhattan(fixed, _boards(fixed, 'reset'))
    d_eval = _manhattan(mixed, _boards(mixed, 'eval_reset'))
    assert (d_mixed <= 4).mean() > 0.02 and (d_fixed <= 4).mean() < 0.005  # short scrambles appear only when mixed
    assert d_mixed.mean() < 0.85 * d_fixed.mean()
    assert np.array_equal(d_eval, d_fixed)  # eval resets: same fixed-length walks as without mixing


def test_zipf_scramble_lengths():
    # random_moves_zipf = s > 0: p(k) ∝ k^-s on [min, num_random_moves] puts more mass on short scrambles than uniform.
    uniform = _env(num_random_moves=200, min_random_moves=1, eval_num_random_moves=200)
    zipf = _env(num_random_moves=200, min_random_moves=1, eval_num_random_moves=200, random_moves_zipf=1.0)
    d_uniform, d_zipf = _manhattan(uniform, _boards(uniform, 'reset')), _manhattan(zipf, _boards(zipf, 'reset'))
    assert (d_zipf <= 4).mean() > 2 * (d_uniform <= 4).mean() and d_zipf.mean() < 0.8 * d_uniform.mean()
    assert np.array_equal(_manhattan(zipf, _boards(zipf, 'eval_reset')), _manhattan(uniform, _boards(uniform, 'eval_reset')))


def test_negative_zipf_favours_long_scrambles():
    # random_moves_zipf = -s < 0: p(k) ∝ k^s puts more mass on long scrambles than uniform, and make_env accepts it.
    from types import SimpleNamespace
    from envs.utils import make_env
    _, cfg = make_env(SimpleNamespace(env_id='slidingpuzzle-4x4', slidingpuzzle_random_moves_zipf=-1.0))
    assert cfg.random_moves_zipf == -1.0
    uniform = _env(num_random_moves=200, min_random_moves=1, eval_num_random_moves=200)
    ramp = _env(num_random_moves=200, min_random_moves=1, eval_num_random_moves=200, random_moves_zipf=-1.0)
    d_uniform, d_ramp = _manhattan(uniform, _boards(uniform, 'reset')), _manhattan(ramp, _boards(ramp, 'reset'))
    assert (d_ramp <= 4).mean() < 0.5 * (d_uniform <= 4).mean() and d_ramp.mean() > 1.1 * d_uniform.mean()
