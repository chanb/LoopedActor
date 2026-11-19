"""halt_update_period=p: the trunk updates every step, the halting head only on steps p, 2p, ... (ppo.make_optimizer)."""
import os
import sys

import jax.numpy as jnp
import optax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ppo import Args, make_optimizer  # noqa: E402


def test_halt_head_moves_every_pth_step_only():
    params = {'params': {'core': {'w': jnp.ones((3,))}, 'halt_linear': {'bias': jnp.ones((1,))}}}
    tx = make_optimizer(Args(learning_rate=0.1, halt_update_period=3), params)
    grads = {'params': {'core': {'w': jnp.ones((3,))}, 'halt_linear': {'bias': jnp.ones((1,))}}}
    state, moved = tx.init(params), []
    for _ in range(6):
        old = params
        upd, state = tx.update(grads, state, params)
        params = optax.apply_updates(params, upd)
        assert not jnp.allclose(params['params']['core']['w'], old['params']['core']['w'])  # trunk: every step
        moved.append(bool(jnp.any(params['params']['halt_linear']['bias'] != old['params']['halt_linear']['bias'])))
    assert moved == [False, False, True, False, False, True]
