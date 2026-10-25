"""stable_softmax tests: the Softmax Collapse / Cancellation fixes of Barron et al. (arXiv 2610.31642) (CPU, fast)."""
import os
import sys
from decimal import Decimal as D, getcontext

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.stable_softmax import Categorical, kl, log_softmax  # noqa: E402

Z = jnp.array([9.0, -9.0, -9.0], jnp.float32)  # margin 17.3 > the fp32 cliff 16.6


def test_log_softmax_no_collapse():
    # jax.nn.log_softmax rounds log p_0 and its gradient to 0 here; the fix keeps -2 e^-18 and a zero-sum gradient.
    assert float(jax.nn.log_softmax(Z)[0]) == 0.0
    assert np.isclose(float(log_softmax(Z)[0]), -2 * np.exp(-18.0), rtol=1e-5)
    g = jax.grad(lambda z: log_softmax(z)[0])(Z)
    assert np.isclose(float(g[0]), 2 * np.exp(-18.0), rtol=1e-5) and abs(float(g.sum())) < 1e-12
    z = jax.random.normal(jax.random.PRNGKey(0), (5, 7))
    assert np.allclose(log_softmax(z), jax.nn.log_softmax(z), atol=1e-6)


def test_categorical_log_prob_no_collapse():
    d = Categorical(logits=Z)
    g = jax.grad(lambda z: Categorical(logits=z).log_prob(jnp.array(0)))(Z)
    assert float(g[0]) > 0 and abs(float(g.sum())) < 1e-12
    assert np.isclose(float(d.entropy()), float(-jnp.sum(jnp.exp(log_softmax(Z)) * log_softmax(Z))))


def test_kl_exact_at_equal_logits():
    z = 3.0 * jax.random.normal(jax.random.PRNGKey(0), (64, 4))
    assert bool(jnp.all(kl(z, z) == 0.0))


def _kl_exact(a, b):
    """KL(softmax(a) || softmax(b)) per row in 50-digit decimal arithmetic."""
    getcontext().prec = 50
    out = []
    for ra, rb in zip(np.float64(a), np.float64(b)):
        ea, eb = [D(float(x)).exp() for x in ra], [D(float(x)).exp() for x in rb]
        sa, sb = sum(ea), sum(eb)
        out.append(float(sum(x / sa * ((x / sa).ln() - (y / sb).ln()) for x, y in zip(ea, eb))))
    return np.array(out)


def test_kl_matches_exact_for_close_logits():
    # KL(softmax(z) || softmax(z + e)) down to ~1e-15, far below the ~1e-7 rounding floor of sum q (log q - log p).
    zq = jnp.float32(3.0 * np.random.default_rng(0).standard_normal((64, 4)))
    for eps in (1e-5, 1e-2, 1.0):
        zp = zq + jnp.float32(eps * np.random.default_rng(1).standard_normal((64, 4)))
        assert np.allclose(np.asarray(kl(zq, zp)), _kl_exact(zq, zp), rtol=1e-5, atol=0)
