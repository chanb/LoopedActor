"""optimal_horizon: an unsolved episode terminates (reward 0, not a truncation) after the optimal number of steps."""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs import lightsout_env, slidingpuzzle_env  # noqa: E402
from utils.wrapper import EpisodeWrapper  # noqa: E402


def _sp(on):
    c = slidingpuzzle_env.default_config()
    c.update(grid_size=4, train_split='train', eval_split='test', optimal_horizon=on)
    return EpisodeWrapper(slidingpuzzle_env.SlidingPuzzleEnv(c), 100)


def _lo(on, m=5, n=4):
    c = lightsout_env.default_config()
    c.update(m=m, n=n, optimal_horizon=on)
    return lightsout_env.LightsOutEnv(c)


def _run(env, state, actions):
    out = []
    for a in actions:
        state = env.step(state, jnp.asarray(a))
        out.append((float(state.done), float(state.reward), float(state.info.get('truncation', 0.0))))
    return out


def _sp_reset(env, opt):
    for i in range(5000):  # a train board (lengths 0..60) of this optimal length, blank not in the top row (0 = up moves it)
        s = env.reset(jax.random.PRNGKey(i))
        if int(s.data.opt) == opt and int(s.data.blank[0]) > 0:
            return s
    raise AssertionError(opt)


def test_slidingpuzzle_cut_at_optimal_length():
    env = _sp(True)
    s = _sp_reset(env, 40)
    up_down = [0, 2] * 20  # back to the start board every 2 steps, never solved
    out = _run(env, s, up_down + [0])
    assert all(d == 0 for d, _, _ in out[:39])
    assert out[39] == (1.0, 0.0, 0.0)  # terminated at t = opt, not truncated


def test_slidingpuzzle_off_unchanged():
    env = _sp(False)
    out = _run(env, _sp_reset(env, 40), [0, 2] * 25)
    assert all(d == 0 for d, _, _ in out)


def test_slidingpuzzle_optimal_solve_rewarded():
    env = _sp(True)
    bank = env.env.train_bank
    idx = int(jnp.argmax(bank['moves'] >= 1))  # the shortest unsolved board (sorted bank; length 0 = the solved board)
    puzzle = bank['puzzle'][idx]
    b = int(jnp.argmin(puzzle.reshape(-1)))
    s = env.reset(jax.random.PRNGKey(0))
    s = s.replace(data=s.data.replace(puzzle=puzzle, blank=jnp.array([b // 4, b % 4]), opt=bank['moves'][idx]))
    opt = int(bank['moves'][idx])
    # breadth-first over action sequences of length opt (opt = 1)
    seqs = [list(q) for q in np.ndindex(*([4] * opt))]
    solved = [q for q in seqs if _run(env, s, q)[-1][:2] == (1.0, 1.0)]
    assert solved, opt


def test_slidingpuzzle_needs_bank():
    c = slidingpuzzle_env.default_config()
    c.optimal_horizon = True
    with pytest.raises(AssertionError):
        slidingpuzzle_env.SlidingPuzzleEnv(c)


def test_lightsout_cut_and_optimal_solve():
    env = EpisodeWrapper(_lo(True), 20)
    key = jax.random.PRNGKey(0)
    # the presses that generated the goal (as reset_with_dist draws them)
    order = np.asarray(jax.random.permutation(jax.random.split(key)[1], 20))
    s = env.env.reset_with_dist(key, jnp.asarray(3))
    s = s.replace(info={**s.info, 'steps': jnp.zeros((), jnp.int32), 'truncation': jnp.zeros(())})
    assert _run(env, s, order[:3])[-1] == (1.0, 1.0, 0.0)  # optimal: solved and rewarded
    out = _run(env, s, [order[3]] * 4)                     # a press outside the solution, 3 times: never solved
    assert [d for d, _, _ in out[:3]] == [0.0, 0.0, 1.0] and out[2][1:] == (0.0, 0.0)
    assert all(d == 0 for d, _, _ in _run(EpisodeWrapper(_lo(False), 20), s, [order[3]] * 4))


def test_lightsout_needs_invertible_press():
    _lo(True, 3, 3)
    with pytest.raises(AssertionError):
        _lo(True, 4, 4)


def test_slidingpuzzle_bank_never_starts_solved():
    # The 4x4 train bank holds the solved board (length 0); by-length resets drew it 1 time in 61.
    env = _sp(False).env
    opt = jax.vmap(env.reset)(jax.random.split(jax.random.PRNGKey(0), 4096)).data.opt
    assert int(opt.min()) >= 1
