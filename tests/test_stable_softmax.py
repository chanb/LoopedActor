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


def test_kl_large_and_masked_logits_no_overflow():
    # Masked off-board logits (-1e9, e.g. action_head='neighbour' on pass 1 vs uniform) and large separations.
    zq_mask = jnp.array([[1.0, 2.0, -1e9, -1e9]], jnp.float32)
    zp_unif = jnp.zeros((1, 4), jnp.float32)
    val_fwd = kl(zq_mask, zp_unif)
    # Exact KL(softmax([1, 2]) || [0.25, 0.25, 0.25, 0.25]) = KL(softmax([1, 2]) || [0.5, 0.5]) + ln 2
    p12 = np.exp([1.0, 2.0]) / np.sum(np.exp([1.0, 2.0]))
    want_fwd = float(np.sum(p12 * (np.log(p12) - np.log(0.25))))
    assert bool(jnp.all(jnp.isfinite(val_fwd))) and np.isclose(float(val_fwd[0]), want_fwd, rtol=1e-5)

    val_rev = kl(zp_unif, zq_mask)
    assert bool(jnp.all(jnp.isfinite(val_rev))) and float(val_rev[0]) > 1e8

    # Large finite separation (delta = +-100 > fp32 exp overflow 88.7)
    zq_far = jnp.array([[100.0, 0.0], [0.0, -100.0]], jnp.float32)
    zp_far = jnp.array([[0.0, 100.0], [0.0, -200.0]], jnp.float32)
    val_far = kl(zq_far, zp_far)
    assert bool(jnp.all(jnp.isfinite(val_far)))
    assert np.isclose(float(val_far[0]), 100.0, rtol=1e-5)
    assert np.isclose(float(val_far[1]), 0.0, atol=1e-6)

