"""halt_objective='score' (ppo.score_halt_loss): grad = -mean_B w grad log p(c|s), w = pi_c(a|s) G - d pi_(c+1)(a|s) G,
log p(c|s) = sum_(n<c) log(1 - h_n) + log h_c, so the gradient reaches every decision up to c."""
import os
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ppo import score_halt_loss  # noqa: E402


def test_score_halt_loss_by_hand():
    T, B, d = 4, 3, 0.9
    lp_a = jax.nn.log_sigmoid(jax.random.normal(jax.random.PRNGKey(0), (T, B)))  # log pi_n(a|s) by pass
    z = jax.random.normal(jax.random.PRNGKey(1), (T, B))  # halt logits by pass
    G = jnp.array([1.0, -2.0, 0.5])
    c = jnp.array([1, 3, 4])  # c = T: the cap, no pass c + 1

    def log_p_c(z):  # sum_(n<c) log(1 - h_n) + log h_c, with no h_T term
        n = jnp.arange(1, T + 1)[:, None]
        return jnp.sum(jnp.where(n < c, jax.nn.log_sigmoid(-z), jnp.where((n == c) & (n < T), jax.nn.log_sigmoid(z), 0.0)), 0)

    pi = jnp.exp(lp_a)
    w = jnp.array([pi[0, 0] * G[0] - d * pi[1, 0] * G[0], pi[2, 1] * G[1] - d * pi[3, 1] * G[1], 0.0])
    g = jax.grad(lambda z: score_halt_loss(lp_a, log_p_c(z), G, c, d))(z)
    h = jax.nn.sigmoid(z)
    # d log p(c|s) / d z_n: -h_n for n < c (continue), 1 - h_n at n = c (halt)
    want = jnp.zeros((T, B))
    want = want.at[0, 0].set(1 - h[0, 0])
    want = want.at[:2, 1].set(-h[:2, 1]).at[2, 1].set(1 - h[2, 1])
    want = -want * w / B  # c = T adds 0
    assert jnp.allclose(g, want, atol=1e-6)
    assert float(jnp.abs(g[:2, 1]).min()) > 0  # the continue decisions before c get gradient
    assert float(jnp.abs(jax.grad(lambda x: score_halt_loss(x, log_p_c(z), G, c, d))(lp_a)).sum()) == 0.0  # w is fixed
