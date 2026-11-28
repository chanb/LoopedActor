"""LightsOutEnv tests: initial != goal even on non-invertible grids (CPU, fast)."""
import jax
import jax.numpy as jnp
import pytest

from envs.lightsout_env import LightsOutEnv, default_config


@pytest.mark.parametrize('m,n', [(4, 4), (5, 5), (5, 4)])
def test_reset_never_produces_initial_equal_goal(m, n):
    cfg = default_config()
    cfg.m, cfg.n = m, n
    env = LightsOutEnv(cfg)
    keys = jax.random.split(jax.random.PRNGKey(0), 4096)
    for fn in (env.reset, env.eval_reset):
        st = jax.jit(jax.vmap(fn))(keys)
        same = jnp.all(st.obs == st.info['target_goal'], axis=-1)
        assert not bool(jnp.any(same)), f'{m}x{n} produced {int(jnp.sum(same))} already-solved resets'
