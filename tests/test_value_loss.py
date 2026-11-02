"""ppo.value_loss: the expectile value loss, and its deep-supervision form over passes (CPU, instant)."""
import os
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ppo import value_loss  # noqa: E402


def test_deep_value_loss_averages_each_state_over_its_passes():
    # values [T=3, B=2]; state 0 supervises pass 1 only, state 1 passes 1..3. Masked passes add no loss and no gradient.
    v = jnp.array([[1.0, 0.0], [1e6, 2.0], [1e6, 4.0]])
    target = jnp.array([0.0, 1.0])
    mask = jnp.arange(1, 4)[:, None] <= jnp.array([1, 3])
    want = 0.5 * jnp.mean(jnp.array([0.5 * 1.0, 0.5 * (1.0 + 1.0 + 9.0) / 3]))
    assert jnp.allclose(value_loss(v, target, 0.5, mask), want)
    assert jnp.allclose(value_loss(v[:1], target, 0.5, mask[:1]), value_loss(v[0], target, 0.5))
    g = jax.grad(lambda v: value_loss(v, target, 0.5, mask))(v)
    assert jnp.all(g[1:, 0] == 0.0) and jnp.all(g[:, 1] != 0.0)
