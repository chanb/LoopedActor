"""Simplicial embeddings (SEM): D = before the actor/value heads, E = on the thought feedback."""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.fprm import FPRMConfig, rms_norm, simplicial  # noqa: E402
from models.fprm_mem_halt import FPRMMemHaltActorValue  # noqa: E402

CFG = FPRMConfig(d_model=16, num_heads=2)
X = jax.random.normal(jax.random.PRNGKey(1), (3, 18))


def test_simplicial_groups_are_distributions():
    y = simplicial(jax.random.normal(jax.random.PRNGKey(0), (5, 16)), groups=4, tau=0.5)
    assert y.shape == (5, 16) and bool(jnp.all(y >= 0))
    np.testing.assert_allclose(y.reshape(5, 4, 4).sum(-1), 1.0, rtol=1e-6)


def _run(**kw):
    net = FPRMMemHaltActorValue(m=3, n=3, config=CFG, max_think_iters=3, halting=False, action_head='readout', **kw)
    params = net.init(jax.random.PRNGKey(0), X)['params']
    (_, value), inter = net.apply({'params': params}, X, mutable=['intermediates'])
    return params, value, {k: v[0] for k, v in inter['intermediates'].items()}


def test_sem_readout_feeds_the_heads():
    params, value, out = _run(sem_readout=True, sem_groups=4, sem_tau=0.5)
    h = out['readout_latent_t'][-1]
    feat = simplicial(rms_norm(h, CFG.rms_norm_eps), 4, 0.5)
    want = feat @ params['value_head']['kernel'][:, 0] + params['value_head']['bias'][0]
    np.testing.assert_allclose(value, want, rtol=1e-5, atol=1e-6)


def test_sem_thought_changes_only_passes_after_the_first():
    _, _, base = _run()
    _, _, sem = _run(sem_thought=True, sem_groups=4)
    np.testing.assert_allclose(base['readout_latent_t'][0], sem['readout_latent_t'][0], rtol=1e-6)
    assert not np.allclose(base['readout_latent_t'][1], sem['readout_latent_t'][1])
