"""The halting head (CPU, tiny model): h_n = sigmoid(w . rms_norm(h_n) + b), no lookahead."""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.fprm import FPRMConfig, rms_norm  # noqa: E402
from models.fprm_mem_halt import FPRMMemHaltActorValue  # noqa: E402

CFG = FPRMConfig(d_model=16, num_heads=2)
T = 4


def _log_pc(halt_logit, min_iters):
    """log p(c = n | s) [T, B] of a yes/no halt at passes min_iters..T-1 (c = T is the cap)."""
    n = jnp.arange(1, T + 1)[:, None]
    log_h = jnp.where(n == T, 0.0, jnp.where(n < min_iters, -1e9, jax.nn.log_sigmoid(halt_logit)))
    log_not_h = jnp.where(n == T, -1e9, jnp.where(n < min_iters, 0.0, jax.nn.log_sigmoid(-halt_logit)))
    survive = jnp.concatenate([jnp.zeros_like(log_not_h[:1]), jnp.cumsum(log_not_h[:-1], axis=0)], axis=0)
    return log_h + survive


@pytest.mark.parametrize('min_iters', [1, 2])
def test_linear_head(min_iters):
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 18))
    net = FPRMMemHaltActorValue(m=3, n=3, config=CFG, max_think_iters=T, min_think_iters=min_iters)
    params = net.init(jax.random.PRNGKey(0), x)['params']
    assert set(params['halt_linear']) == {'kernel', 'bias'} and not any(k.startswith('halt_alpha') for k in params)
    params = dict(params, halt_linear={'kernel': jax.random.normal(jax.random.PRNGKey(2), (16, 1)), 'bias': jnp.array([-1.0])})
    _, inter = net.apply({'params': params}, x, mutable=['intermediates'])
    out = {k: v[0] for k, v in inter['intermediates'].items() if isinstance(v, tuple)}

    # pass n decides from its own latent h_n (no lookahead)
    logit = (rms_norm(out['readout_latent_t'], eps=CFG.rms_norm_eps) @ params['halt_linear']['kernel'])[..., 0] - 1.0
    log_pc = _log_pc(logit, min_iters)
    np.testing.assert_allclose(out['halt_entropy'], -jnp.sum(jnp.exp(log_pc) * log_pc, axis=0), rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(out['fp_iterations'], jnp.argmax(log_pc, axis=0) + 1)
    for c in range(min_iters, T + 1):
        _, inter = net.apply({'params': params}, x, mutable=['intermediates'], halt_steps=jnp.full((3,), c))
        np.testing.assert_allclose(inter['intermediates']['halt_log_prob'][0], log_pc[c - 1], rtol=1e-4, atol=1e-4)


def test_init_half():
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 18))
    net = FPRMMemHaltActorValue(m=3, n=3, config=CFG, max_think_iters=T, min_think_iters=1)
    params = net.init(jax.random.PRNGKey(0), x)['params']
    _, inter = net.apply({'params': params}, x, mutable=['intermediates'])
    np.testing.assert_allclose(inter['intermediates']['halt_prob'][0][:-1], 0.5, rtol=1e-6)
