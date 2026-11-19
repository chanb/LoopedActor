import jax
import jax.numpy as jnp

from models.fprm import fused_attention


def test_fused_attention_matches_masked_softmax():
    # fused_attention over the first kv_len keys == the masked einsum softmax it replaced (odd lengths included)
    B, T, S, h, d = 3, 5, 9, 2, 8
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    q, k, v = (jax.random.normal(kk, (B, n, h, d)) for kk, n in ((k1, T), (k2, S), (k3, S)))
    for kv_len in (T, 7, S):
        valid = jnp.arange(S) < kv_len
        logits = jnp.where(valid, jnp.einsum('bqhd,bkhd->bhqk', q, k) * 0.3, -jnp.inf)
        want = jnp.einsum('bhqk,bkhd->bqhd', jax.nn.softmax(logits, axis=-1), v)
        assert jnp.allclose(fused_attention(q, k, v, 0.3, kv_len), want, atol=1e-5)
