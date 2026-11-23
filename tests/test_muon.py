"""optimizer='muon': Muon on the core's 2-D kernels (actor and critic), Adam on every other param (ppo.make_optimizer)."""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ppo import Args, make_optimizer  # noqa: E402

LR = 1e-2
SHAPES = {  # as the fprm_mem actor-critic (d = 128 cut to 16)
    'core': {'block_0': {'self_attn': {'qkv_proj': {'kernel': (16, 48)}, 'o_proj': {'kernel': (16, 16)}},
                         'mlp': {'down_proj': {'kernel': (64, 16)}}, 'norm': {'scale': (16,)}},
             'grid_depthwise_conv': {'kernel': (3, 3, 1, 16)}},
    'critic': {'core': {'block_0': {'mlp': {'down_proj': {'kernel': (64, 16)}}}}, 'value_head': {'kernel': (16, 1)}},
    'actor_head': {'kernel': (16, 1)}, 'halt_linear': {'kernel': (16, 1), 'bias': (1,)},
    'bot': (1, 16), 'sep': (1, 16), 'cell_projection': {'kernel': (11, 16)},
}
MUON = ['core/block_0/self_attn/qkv_proj/kernel', 'core/block_0/self_attn/o_proj/kernel',
        'core/block_0/mlp/down_proj/kernel', 'critic/core/block_0/mlp/down_proj/kernel']


def _flat(tree):
    return {'/'.join(k.key for k in path): v for path, v in jax.tree_util.tree_flatten_with_path(tree)[0]}


@pytest.mark.parametrize('kw', [dict(), dict(halt_adam_b1=0.5), dict(weight_decay=1e-3)])
def test_muon_on_core_kernels_adam_elsewhere(kw):
    params = {'params': jax.tree_util.tree_map(jnp.ones, SHAPES, is_leaf=lambda s: isinstance(s, tuple))}
    leaves, treedef = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(jax.random.PRNGKey(0), len(leaves))
    grads = jax.tree_util.tree_unflatten(treedef, [jax.random.normal(k, x.shape) for k, x in zip(keys, leaves)])
    tx = make_optimizer(Args(optimizer='muon', learning_rate=LR, anneal_lr=False, max_grad_norm=0.0, **kw), params)
    updates = _flat(tx.update(grads, tx.init(params), params)[0]['params'])
    for name, u in updates.items():
        if name in MUON:  # orthogonalized: every singular value ~ the same, times 0.2 sqrt(max fan) (consistent RMS)
            s = np.linalg.svd(np.asarray(u), compute_uv=False) / (LR * 0.2 * np.sqrt(max(u.shape)))
            assert np.all((s > 0.5) & (s < 1.3)), (name, s)
        else:  # Adam's first step: |update| = LR per entry (up to the weight decay)
            assert np.allclose(np.abs(np.asarray(u)), LR, rtol=1e-2), name
    assert set(MUON) < set(updates)


def test_adam_unchanged():
    params = {'params': {'core': {'k': {'kernel': jnp.ones((4, 4))}}}}
    tx = make_optimizer(Args(learning_rate=LR, anneal_lr=False, max_grad_norm=0.0), params)
    u = tx.update({'params': {'core': {'k': {'kernel': jnp.arange(16.0).reshape(4, 4) - 7.5}}}}, tx.init(params), params)[0]
    assert np.allclose(np.abs(np.asarray(u['params']['core']['k']['kernel'])), LR, rtol=1e-2)  # Adam, not Muon
