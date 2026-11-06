"""MinAtar (gymnax) env tests (CPU, a few seconds)."""
import os
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.minatar_env import GAMES  # noqa: E402
from envs.utils import make_env  # noqa: E402
from utils.wrapper import wrap_env  # noqa: E402


@pytest.mark.parametrize('game', GAMES)
def test_rollout_with_wrappers(game):
    env_class, config = make_env(SimpleNamespace(env_id=f'minatar-{game}', minatar_episode_length=50))
    raw = env_class(config=config)
    assert raw.m == raw.n == 10 and raw.goal_size == 0 and raw.observation_size == 100 * raw.obs_channels
    env = wrap_env(raw, config.episode_length)
    state = env.reset(jax.random.split(jax.random.PRNGKey(0), 8))
    assert state.obs.shape == (8, raw.observation_size) and state.info['target_goal'].shape == (8, 0)

    def body(s, key):
        a = jax.random.randint(key, (8,), 0, raw.action_size)
        s = env.step(s, a)
        return s, (s.reward, s.done, s.info['truncation'])

    _, (r, d, t) = jax.jit(lambda s: jax.lax.scan(body, s, jax.random.split(jax.random.PRNGKey(1), 120)))(state)
    assert jnp.isfinite(r).all()
    assert (t.sum(0) >= 1).all() or (d.sum(0) >= 2).all()  # every env ends at least once in 120 steps (limit 50)
    assert ((t == 1) <= (d == 1)).all()  # a truncation is a done


def test_time_out_is_truncation_not_terminal():
    env_class, config = make_env(SimpleNamespace(env_id='minatar-freeway', minatar_episode_length=5))
    env = wrap_env(env_class(config=config), config.episode_length, train=False)
    s = env.reset(jax.random.split(jax.random.PRNGKey(0), 4))
    for _ in range(5):
        s = env.step(s, jnp.zeros(4, jnp.int32))
    assert (s.done == 1).all() and (s.info['truncation'] == 1).all()  # Freeway never terminates early
