"""halt_head='sigmoid' (CPU, tiny model): h_n = sigmoid(beta_{n+1} - alpha log KL_{n+1}), the kl_only-prev_latent head."""
import math
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
T, EPS = 4, 1e-10


def _kl_branch_log_pc(halt_logit, min_iters):
    """log p(c = n | s) [T, B], copied from kl_only-prev_latent:models/fprm_mem_halt.py halt_log_probs."""
    n = jnp.arange(1, T + 1)[:, None]
    log_h = jnp.where(n == T, 0.0, jnp.where(n < min_iters, -1e9, jax.nn.log_sigmoid(halt_logit)))
    log_not_h = jnp.where(n == T, -1e9, jnp.where(n < min_iters, 0.0, jax.nn.log_sigmoid(-halt_logit)))
    survive = jnp.concatenate([jnp.zeros_like(log_not_h[:1]), jnp.cumsum(log_not_h[:-1], axis=0)], axis=0)
    return log_h + survive


@pytest.mark.parametrize('kw', [dict(), dict(halt_beta_head=True), dict(halt_threshold_fixed=True),
                                dict(halt_alpha_min=0.3, min_think_iters=2)])
def test_matches_kl_branch(kw):
    kw = dict(dict(min_think_iters=1), **kw)
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 18))
    net = FPRMMemHaltActorValue(m=3, n=3, config=CFG, max_think_iters=T, halt_head='sigmoid', **kw)
    params = net.init(jax.random.PRNGKey(0), x)['params']
    # Move alpha and beta away from their init so the test sees them.
    params = dict(params, halt_alpha_raw=jnp.float32(0.7), halt_beta={
        'kernel': jax.random.normal(jax.random.PRNGKey(2), (16, 1)), 'bias': jnp.array([-4.0])})
    _, inter = net.apply({'params': params}, x, mutable=['intermediates'])
    out = {k: v[0] for k, v in inter['intermediates'].items()}

    alpha = kw.get('halt_alpha_min', 0.0) + jax.nn.softplus(0.7)
    if kw.get('halt_threshold_fixed'):
        beta = alpha * math.log(1e-3)
    elif kw.get('halt_beta_head'):
        beta = (rms_norm(out['readout_latent_t'], eps=CFG.rms_norm_eps) @ params['halt_beta']['kernel'])[..., 0] - 4.0
    else:
        beta = -4.0
    beta = jnp.broadcast_to(beta, (T, 3))
    shift = lambda a: jnp.concatenate([a[1:], a[-1:]], axis=0)  # row n: pass n+1 (lookahead)
    logit = shift(beta) - alpha * jnp.log(shift(out['policy_kl_t']) + EPS)
    log_pc = _kl_branch_log_pc(logit, kw['min_think_iters'])

    np.testing.assert_allclose(out['halt_entropy'], -jnp.sum(jnp.exp(log_pc) * log_pc, axis=0), rtol=1e-4, atol=1e-5)
    np.testing.assert_array_equal(out['fp_iterations'], jnp.argmax(log_pc, axis=0) + 1)
    m = kw['min_think_iters']
    for c in range(m, T + 1):  # the replayed log p(c | s) of every reachable c; decisions at passes m..min(c, T - 1)
        path = (1 << min(c, T - 1)) - (1 << (m - 1))
        _, inter = net.apply({'params': params}, x, mutable=['intermediates'],
                             halt_steps=jnp.full((3,), c), halt_path=jnp.full((3,), path))
        np.testing.assert_allclose(inter['intermediates']['halt_log_prob'][0], log_pc[c - 1], rtol=1e-4, atol=1e-4)


def test_init_half_at_halt_kl():
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 18))
    net = FPRMMemHaltActorValue(m=3, n=3, config=CFG, max_think_iters=T, halt_head='sigmoid', min_think_iters=1, halt_kl=1e-2)
    params = net.init(jax.random.PRNGKey(0), x)['params']
    _, inter = net.apply({'params': params}, x, mutable=['intermediates'])
    kl = inter['intermediates']['policy_kl_t'][0][1:]
    h = inter['intermediates']['halt_prob'][0][:-1]
    np.testing.assert_allclose(h, jax.nn.sigmoid(math.log(1e-2) - jnp.log(kl + EPS)), rtol=1e-5)
