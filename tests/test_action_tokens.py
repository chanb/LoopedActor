"""action_head='action_tokens': |A| extra input tokens, logit a read off token a (CPU, a few seconds)."""
import os
import sys

import jax
import jax.numpy as jnp
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.fprm import FPRMConfig  # noqa: E402
from models.fprm_mem_halt import FPRMMemHaltActorValue  # noqa: E402
from models.fprm_thinker import FPRMThinkerActorValue  # noqa: E402
from models.transformer_baseline import TransformerActorValue  # noqa: E402

CFG = FPRMConfig(d_model=16, num_heads=2)
M, N, C, A = 4, 5, 3, 6
NETS = {
    'fprm': lambda tok: FPRMThinkerActorValue(m=M, n=N, config=CFG, max_think_iters=3, halt_kl=-1.0, output_dim_1=A,
                                              obs_channels=C, action_head='action_tokens', tokenizer=tok),
    'multi_block': lambda tok: TransformerActorValue(m=M, n=N, config=CFG, num_blocks=2, output_dim_1=A, obs_channels=C,
                                                    action_head='action_tokens', tokenizer=tok),
    'fprm_mem_halt': lambda tok: FPRMMemHaltActorValue(m=M, n=N, config=CFG, max_think_iters=3, output_dim_1=A,
                                                      obs_channels=C, action_head='action_tokens', tokenizer=tok),
}


@pytest.mark.parametrize('arch', NETS)
@pytest.mark.parametrize('tok', ['cell', 'conv'])
def test_logit_a_reads_token_a(arch, tok):
    net = NETS[arch](tok)
    x = jax.random.normal(jax.random.PRNGKey(1), (2, M * N * C))
    params = net.init(jax.random.PRNGKey(0), x)['params']
    assert params['action_embed'].shape == (A, CFG.d_model)
    logits = net.apply({'params': params}, x)[0].logits
    assert logits.shape == (2, A) and bool(jnp.all(jnp.isfinite(logits)))
    perm = jnp.array([3, 1, 2, 0, 4, 5])  # swap action tokens 0 and 3
    swapped = net.apply({'params': {**params, 'action_embed': params['action_embed'][perm]}}, x)[0].logits
    assert jnp.allclose(swapped, logits[:, perm], atol=1e-5)
    assert not jnp.allclose(logits[:, 0], logits[:, 3])  # the tokens differ, so the logits do
