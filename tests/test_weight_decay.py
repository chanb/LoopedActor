"""weight_decay: decoupled decay on the trunk only, never on the halting head (ppo.make_optimizer)."""
import os
import sys

import jax.numpy as jnp
import optax
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ppo import Args, make_optimizer  # noqa: E402

LR, WD = 0.1, 0.5


@pytest.mark.parametrize('kw', [
    dict(halt_head='convtest'), dict(halt_head='convtest', halt_precondition=False),
    dict(halt_head='distance'), dict(halt_head='distance', halt_adam_b1=0.5),
])
def test_decay_reaches_the_trunk_only(kw):
    params = {'params': {'core': {'w': jnp.ones((3,))}, 'halt_hidden': {'kernel': jnp.ones((2,))},
                         'halt_out': {'kernel': jnp.ones((2,))}, 'theta0': jnp.float32(1.0), 'theta1': jnp.float32(1.0)}}
    tx = make_optimizer(Args(learning_rate=LR, weight_decay=WD, **kw), params)
    zeros = {'params': {k: optax.tree_utils.tree_zeros_like(v) for k, v in params['params'].items()}}
    new = optax.apply_updates(params, tx.update(zeros, tx.init(params), params)[0])['params']
    assert jnp.allclose(new['core']['w'], 1.0 - LR * WD)  # zero gradient: the decay alone
    for k in ('halt_hidden', 'halt_out'):
        assert jnp.all(new[k]['kernel'] == 1.0), k
    assert float(new['theta0']) == 1.0 and float(new['theta1']) == 1.0
