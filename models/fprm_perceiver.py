"""FPRM actor-critic with a Perceiver AR-style latent chain of thought.

Follows Perceiver AR (Hawthorne et al. 2022, arXiv:2202.07765) - queries only at the latent positions, keys/values
over the inputs and the latents - with these modifications:

  1. The latents are generated autoregressively as continuous vectors (no tokens are sampled): the output at the
     newest latent position is fed back as the next latent.
  2. The queries start at a learned <BOT> latent (not at the last input token), and the query, key and value
     sets all grow by one position per generated latent.
  3. A learned latent modal embedding is added to every generated latent when it is fed back.
  4. The (state) cell tokens are read by *every* layer (Perceiver AR proper reads them only in its first,
     cross-attention layer): at each layer the latents attend to [cells, latents so far]. The cells are never
     processed themselves - their keys/values at each layer are that layer's projections of their fixed input
     embedding - so they are computed once per decision.

    latents  x_0 = BOT,  x_i = latent_embed + out_{i-1}   (rms_norm(out_{i-1}) if `thought_norm`)
    pass i:  out_i = Blocks(x_i; attending to cells and x_0..x_i)      (causal over the latents)

The blocks are the FPRM core's (FPRMBlock: pre-norm attention + SwiGLU with the coupled a1/b1 residual scaling,
and the a2/b2 input mixing h = a2 * z0 + b2 * x applied to every position, cells included); no grid convolution
(the latents are not a grid). The core is evaluated functionally on the parameters of an `FPRMCore` submodule
(`fprm_cot_thinker._core_decode`), so every pass costs one latent token: the cell keys/values are computed once
and the latent keys/values are cached.

Readout of pass i, from the newest latent out_i:
    - value and the 'readout' action head: Dense on rms_norm(out_i).
    - 'per_cell' action head (pointer): out_i forms one query per logit slot, logit (c, k) =
      <rms_norm(cell token c), q_k> / sqrt(d_model) (the cells have no outputs).

Halting is the policy-KL rule of the looped thinker (stop after pass i once KL(pi_i || pi_{i-1}) < `halt_kl` and
i >= `min_think_iters`, or at `max_think_iters`; the first pass has no previous policy, KL = inf), with halted
examples frozen inside the differentiable scan. The final readout is sown as 'logits'/'value' with the same 'fp_*'
diagnostics as the other thinkers, so it is interchangeable in the PPO loss.

Supports the observation tokenizers of models/tokenizers.py. No parameter shape depends on m, n or the iteration
cap.
"""

import dataclasses
import math

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from models.fprm import (
    FPRMConfig,
    FPRMCore,
    STOP_ACTIVE,
    STOP_CONVERGED,
    STOP_ITERATION_CAP,
    STOP_NONFINITE,
    make_latent_init,
    rms_norm,
)
from models.fprm_cot_thinker import _core_decode, _qkv, _residual_scales
from models.fprm_thinker import tokenize
from models.tokenizers import InputGridConv, ObsTokenizer, check_tokenizer


@flax.struct.dataclass
class PerceiverState:
    token: jax.Array  # [B, H] newest latent output, fed back as the next latent
    logits: jax.Array  # [B, A] readout of the last executed pass
    value: jax.Array  # [B]
    residual: jax.Array  # [B] float32, policy KL at the last executed pass
    iterations: jax.Array  # [B] int32, executed passes
    active: jax.Array  # [B] bool
    nonfinite: jax.Array  # [B] bool
    stop_reason: jax.Array  # [B] int32


def cell_kv(params, cells, z0, cfg: FPRMConfig):
    """Per-layer keys/values of the (never processed) cell tokens: layer l's projection of the cells' input
    embedding after the core's input mixing. Returns [(k, v)] * num_layers, each [B, P, nh, hd]."""
    _, _, a2, b2 = _residual_scales(params, cfg)
    h = rms_norm(a2 * z0 + b2 * cells, cfg.rms_norm_eps)
    out = []
    for i in range(cfg.num_layers):
        _, k, v = _qkv(params[f'block_{i}']['self_attn'], h, cfg)
        out.append((k, v))
    return out


class FPRMPerceiverActorValue(nn.Module):
    """Perceiver AR-style latent-CoT actor-critic over FPRM blocks (see the module docstring)."""

    m: int = 3
    n: int = 3
    config: FPRMConfig = FPRMConfig()
    max_think_iters: int = 8
    min_think_iters: int = 2
    # Stop thinking once KL(pi_i || pi_{i-1}) between consecutive readouts drops below this (0 = always max_think_iters).
    halt_kl: float = 1e-3
    output_dim_1: int = 9   # action logits (= m * n buttons)
    output_dim_2: int = 1   # value
    obs_channels: int = 0          # see FPRMThinkerActorValue
    action_head: str = 'per_cell'  # see FPRMThinkerActorValue
    remat: bool = True             # rematerialize each pass in the backward pass
    thought_norm: bool = False     # rms_norm each generated latent before adding the modal embedding
    tokenizer: str = 'cell'        # see FPRMThinkerActorValue
    tokenizer_conv_features: int = 64
    # Residual depthwise 3x3 conv over the cell embeddings (tokenizer='cell'; see models/tokenizers.InputGridConv).
    input_grid_conv: bool = False

    @property
    def core_config(self) -> FPRMConfig:
        return dataclasses.replace(self.config, use_grid_conv=False)  # the latents are not a grid

    def setup(self):
        assert self.action_head in ('per_cell', 'readout'), self.action_head
        assert self.max_think_iters >= 1, self.max_think_iters
        check_tokenizer(self.tokenizer, self.action_head, self.obs_channels)
        if self.input_grid_conv:
            assert self.tokenizer == 'cell', "input_grid_conv needs tokenizer='cell' (one token per grid cell)"
            self.input_conv = InputGridConv(self.m, self.n)
        H = self.config.d_model
        self.cell_projection = nn.Dense(H)
        if self.tokenizer != 'cell':
            self.obs_tokenizer = ObsTokenizer(
                self.tokenizer, self.m, self.n, self.obs_channels, H, self.tokenizer_conv_features,
            )
        self.bot = self.param('bot', nn.initializers.normal(stddev=0.02), (1, H))
        # Modal embedding added to every generated latent fed back into the model.
        self.latent_embed = self.param('latent_embed', nn.initializers.normal(stddev=0.02), (1, H))
        # Parameters only; evaluated functionally (`cell_kv`, `_core_decode`).
        self.core = FPRMCore(self.core_config)
        if self.action_head == 'per_cell':
            assert self.output_dim_1 % (self.m * self.n) == 0, (self.output_dim_1, self.m, self.n)
            # One query per logit slot (k = output_dim_1 // (m*n)), matched against every cell token.
            self.actor_head = nn.Dense(self.output_dim_1 // (self.m * self.n) * H, bias_init=nn.initializers.zeros)
        else:
            self.actor_head = nn.Dense(self.output_dim_1, bias_init=nn.initializers.zeros)
        self.value_head = nn.Dense(self.output_dim_2, bias_init=nn.initializers.zeros)
        self.latent_init = make_latent_init(self.config)

    def __call__(self, x, normalizer_params=None):
        if normalizer_params is not None:
            x = (x - normalizer_params.mean) / (normalizer_params.std)

        cfg = self.core_config
        input_shape = x.shape
        x_flat = x.reshape(-1, input_shape[-1])
        B = x_flat.shape[0]
        H = cfg.d_model
        T = self.max_think_iters  # latent slots: BOT + T - 1 generated latents
        per_cell = self.action_head == 'per_cell'

        if self.tokenizer == 'cell':
            cells = self.cell_projection(tokenize(x_flat, self.m, self.n, self.obs_channels))
        else:
            cells = self.obs_tokenizer(x_flat)  # [B, P, H]
        if self.input_grid_conv:
            cells = self.input_conv(cells)
        z0 = self.latent_init.astype(cells.dtype)

        if self.is_initializing():
            # Create the core / head parameters (the forward pass below is functional).
            dummy = jnp.zeros((1, 2, H), cells.dtype)
            self.core(dummy, dummy, jnp.ones((1, 2), bool), (1, 2), prefix_len=0)
            self.actor_head(jnp.zeros((1, H)))
            self.value_head(jnp.zeros((1, H)))
        params = self.variables['params']
        core_p, actor_p, value_p = params['core'], params['actor_head'], params['value_head']

        prefix_cache = cell_kv(core_p, cells, z0, cfg)
        cell_latents = rms_norm(cells, eps=cfg.rms_norm_eps)  # pointer keys for the per-cell head

        def readout(out):
            last_latent = rms_norm(out, eps=cfg.rms_norm_eps)
            actor_out = last_latent @ actor_p['kernel'] + actor_p['bias']
            if per_cell:
                queries = actor_out.reshape(B, -1, H)  # [B, k, H]
                logits = jnp.einsum('bch,bkh->bck', cell_latents, queries) / math.sqrt(H)
                logits = logits.reshape(B, -1)  # [B, m*n*k], cell-major
            else:
                logits = actor_out
            assert logits.shape[-1] == self.output_dim_1, (logits.shape, self.output_dim_1)
            value = jnp.squeeze(last_latent @ value_p['kernel'] + value_p['bias'], axis=-1)
            return logits, value

        def update(st, out):
            """Readout of a new pass, policy-KL halting, and freezing of halted examples."""
            logits, value = readout(out)
            iterations = st.iterations + 1
            lp_new = jax.nn.log_softmax(logits, axis=-1)
            lp_prev = jax.nn.log_softmax(st.logits, axis=-1)
            kl = jnp.sum(jnp.exp(lp_new) * (lp_new - lp_prev), axis=-1)
            kl = jnp.where(st.iterations == 0, jnp.inf, kl)  # no previous policy on the first pass
            kl = jax.lax.stop_gradient(kl).astype(jnp.float32)

            # A nonfinite pass keeps the last finite readout.
            finite = jnp.all(jnp.isfinite(logits), axis=-1) & jnp.isfinite(value) & jnp.all(jnp.isfinite(out), axis=-1)
            nonfinite = st.nonfinite | ~finite

            # Stop priority: nonfinite > converged > iteration_cap.
            converged = (iterations >= self.min_think_iters) & (kl < self.halt_kl)
            capped = iterations >= self.max_think_iters
            should_stop = nonfinite | converged | capped
            reason = jnp.full_like(st.stop_reason, STOP_ACTIVE)
            reason = jnp.where(capped, STOP_ITERATION_CAP, reason)
            reason = jnp.where(converged, STOP_CONVERGED, reason)
            reason = jnp.where(nonfinite, STOP_NONFINITE, reason)

            stepped = PerceiverState(
                token=jnp.where(finite[:, None], out, st.token),
                logits=jnp.where(finite[:, None], logits, st.logits),
                value=jnp.where(finite, value, st.value),
                residual=kl,
                iterations=iterations,
                active=~should_stop,
                nonfinite=nonfinite,
                stop_reason=reason,
            )

            # Freeze examples that had already halted before this pass.
            def sel(new, old):
                mask = st.active.reshape((-1,) + (1,) * (new.ndim - 1))
                return jnp.where(mask, new, old)

            return jax.tree_util.tree_map(sel, stepped, st)

        state = PerceiverState(
            token=jnp.zeros((B, H), cells.dtype),
            logits=jnp.zeros((B, self.output_dim_1), cells.dtype),
            value=jnp.zeros((B,), cells.dtype),
            residual=jnp.full((B,), jnp.inf, jnp.float32),
            iterations=jnp.zeros((B,), jnp.int32),
            active=jnp.ones((B,), bool),
            nonfinite=jnp.zeros((B,), bool),
            stop_reason=jnp.full((B,), STOP_ACTIVE, jnp.int32),
        )
        z0_token = jnp.broadcast_to(z0[None], (B, H))
        bot = jnp.broadcast_to(self.bot, (B, H))
        latent_embed = self.latent_embed
        kv_shape = (cfg.num_layers, B, T, cfg.num_heads, H // cfg.num_heads)
        buffers = (jnp.zeros(kv_shape, cells.dtype), jnp.zeros(kv_shape, cells.dtype))

        def body(carry, slot):
            st, (buf_k, buf_v) = carry
            fed_back = rms_norm(st.token, eps=cfg.rms_norm_eps) if self.thought_norm else st.token
            latent = jnp.where(slot == 0, bot, fed_back + latent_embed)
            out, buf_k, buf_v = _core_decode(core_p, z0_token, latent, prefix_cache, buf_k, buf_v, slot, cfg)
            return (update(st, out), (buf_k, buf_v)), None

        if self.remat:
            # Rematerialize each pass in the backward pass.
            body = jax.checkpoint(body, prevent_cse=False)
        (state, _), _ = jax.lax.scan(body, (state, buffers), jnp.arange(T))

        logits = state.logits.reshape(input_shape[:-1] + (-1,))
        value = state.value.reshape(input_shape[:-1])
        self.sow('intermediates', 'logits', logits)
        self.sow('intermediates', 'value', value)

        # Diagnostics, retrievable via apply(..., mutable=['intermediates']).
        # `state.residual` holds each example's policy KL at its halt pass.
        self.sow('intermediates', 'fp_iterations', state.iterations)
        self.sow('intermediates', 'fp_residual', state.residual)
        self.sow('intermediates', 'fp_converged', state.stop_reason == STOP_CONVERGED)
        self.sow('intermediates', 'fp_stop_reason', state.stop_reason)

        return distrax.Categorical(logits=logits), value
