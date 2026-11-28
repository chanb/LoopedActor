"""Exclusive self attention (XSA, arXiv:2603.09078) tests (CPU, a few seconds)."""
import dataclasses
import os
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.fprm import FPRMAttention, FPRMConfig, exclusive_self_attention  # noqa: E402
from models.fprm_mem_halt import FPRMMemHaltActorValue  # noqa: E402

CFG = FPRMConfig(d_model=16, num_heads=2)


def test_xsa_matches_paper_formula_and_is_orthogonal_to_own_value():
    y, v = (jax.random.normal(k, (3, 5, 2, 8)) for k in jax.random.split(jax.random.PRNGKey(0)))
    z = exclusive_self_attention(y, v)
    want = y - jnp.sum(y * v, -1, keepdims=True) * v / jnp.sum(v * v, -1, keepdims=True)
    assert jnp.allclose(z, want, atol=1e-5)
    assert jnp.allclose(jnp.sum(z * v, -1), 0.0, atol=1e-5)  # per head, per token
    assert bool(jnp.all(jnp.isfinite(exclusive_self_attention(y, jnp.zeros_like(v)))))  # zero v_i: no NaN


def _attn_out(cfg, attn_mask=None):
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 6, cfg.d_model))
    mask = jnp.ones((2, 6), bool)
    attn = FPRMAttention(cfg)
    params = attn.init(jax.random.PRNGKey(0), x, mask, attn_mask)['params']
    return attn, params, x, mask


def test_fprm_attention_xsa_output_orthogonal_before_o_proj():
    cfg = dataclasses.replace(CFG, xsa=True)
    causal = jnp.broadcast_to(jnp.tril(jnp.ones((6, 6), bool)), (2, 6, 6))
    for attn_mask in (None, causal):  # fused path and masked-einsum path
        attn, params, x, mask = _attn_out(cfg, attn_mask)
        out = attn.apply({'params': params}, x, mask, attn_mask)
        H, nh = cfg.d_model, cfg.num_heads
        v = (x @ params['qkv_proj']['kernel']).reshape(2, 6, 3 * nh, H // nh)[:, :, 2 * nh:]
        # undo o_proj (square, invertible at init) to recover the per-head pre-projection output
        pre = jnp.linalg.solve(params['o_proj']['kernel'].T, out.reshape(-1, H).T).T.reshape(2, 6, nh, H // nh)
        assert jnp.allclose(jnp.sum(pre * v, -1), 0.0, atol=1e-3)


def test_xsa_flag_same_params_different_output():
    attn, params, x, mask = _attn_out(CFG)
    attn_x, params_x, _, _ = _attn_out(dataclasses.replace(CFG, xsa=True))
    assert jax.tree.all(jax.tree.map(lambda a, b: a.shape == b.shape and bool(jnp.all(a == b)), params, params_x))
    assert not jnp.allclose(attn.apply({'params': params}, x, mask), attn_x.apply({'params': params}, x, mask))


def test_mem_halt_xsa_runs_and_changes_output():
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 18))
    outs = []
    for xsa in (False, True):
        net = FPRMMemHaltActorValue(m=3, n=3, config=dataclasses.replace(CFG, xsa=xsa), max_think_iters=4)
        params = net.init(jax.random.PRNGKey(0), x)['params']
        outs.append(jax.tree.leaves(net.apply({'params': params}, x)))
    assert all(bool(jnp.all(jnp.isfinite(o))) for o in outs[1] if jnp.issubdtype(o.dtype, jnp.floating))
    assert any(not jnp.allclose(a, b) for a, b in zip(*outs) if jnp.issubdtype(a.dtype, jnp.floating))


def test_mem_halt_xsa_and_rope_together():
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 18))
    net = FPRMMemHaltActorValue(m=3, n=3, config=dataclasses.replace(CFG, xsa=True), max_think_iters=4, cell_rope=True)
    params = net.init(jax.random.PRNGKey(0), x)['params']
    out = net.apply({'params': params}, x)
    assert all(bool(jnp.all(jnp.isfinite(o))) for o in jax.tree.leaves(out) if jnp.issubdtype(o.dtype, jnp.floating))
