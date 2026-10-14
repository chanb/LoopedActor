"""FPRM with a thought memory: the looped board of `fprm` plus a chain of thought tokens.

Every pass refines the board exactly as `fprm` does - the cells carry a latent z that goes through one FPRM core
call per pass, with the grid convolution on z and the cell embeddings re-injected - and additionally runs one
*thought* token alongside the cells. The thoughts of earlier passes stay available as a memory of keys/values:

    pass i:  tokens  [z_cells, t_i]                      (P cells + 1 thought, attending to each other)
             memory  K/V of t_1 .. t_{i-1}, per layer     (read by the cells and by t_i)
             [z_cells, out_i] = Core([z_cells, t_i]; memory)
             write t_i's per-layer K/V into the memory
    t_1 = BOT (learned),  t_i = latent_embed + out_{i-1}   (rms_norm(out_{i-1}) first if `thought_norm`)

So thinking writes back into the board (cells attend to all thoughts so far) and the board is refined every pass,
while the thoughts give a memory of earlier passes that fprm's single latent does not have. With no thought tokens
this reduces to fprm's loop over the cells.

Memory semantics: a thought's keys/values are those computed in the pass that wrote it (they are not recomputed when
the cells change later), Transformer-XL style; the memory has max_think_iters - 1 slots, preallocated and masked
until written.

Readout of pass i: value and the 'readout' action head from rms_norm(out_i) (the newest thought); the 'per_cell'
action head from the refined cell latents (one Dense per cell, as in fprm). Halting is the policy-KL rule of the
other thinkers (KL(pi_i || pi_{i-1}) < `halt_kl` once i >= `min_think_iters`, or the cap; the first pass has no
previous policy), with halted examples frozen inside the differentiable scan; the final readout and the 'fp_*'
diagnostics are sown as by the other thinkers, so it is interchangeable in the PPO loss.

`value_at_max_iters`: halting only fixes the policy readout (logits of the halt pass); the cells and thoughts of a
halted example keep evolving for all max_think_iters passes (up to its halt pass the trajectory is the one above), and
the value is read out from the last pass. The scan runs every pass for every example either way, so this costs nothing.

Cost per pass: one core call over P + 1 tokens plus the memory keys - about one fprm iteration. Each pass is
rematerialized in the backward pass (`remat`). Supports the observation tokenizers of models/tokenizers.py; no
parameter shape depends on m, n or the iteration cap.
"""

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
from models.fprm_cot_thinker import _attn_softmax, _mixed_precision_input, _qkv, _residual_scales, _swiglu
from models.fprm_thinker import tokenize
from models.tokenizers import ObsTokenizer, check_tokenizer, core_config, token_grid_shape


@flax.struct.dataclass
class MemState:
    z: jax.Array  # [B, P, H] cell latents (refined every pass)
    token: jax.Array  # [B, H] newest thought output, fed back as the next thought
    logits: jax.Array  # [B, A] readout of the last executed pass
    value: jax.Array  # [B]
    value_last: jax.Array  # [B] value readout of the newest pass (not frozen at halting; value_at_max_iters)
    residual: jax.Array  # [B] float32, policy KL at the last executed pass
    iterations: jax.Array  # [B] int32, executed passes
    active: jax.Array  # [B] bool
    nonfinite: jax.Array  # [B] bool
    stop_reason: jax.Array  # [B] int32


def mem_core(params, z, x, mem_k, mem_v, mem_valid, grid_shape, num_cells, cfg: FPRMConfig):
    """One FPRM core call over the P + 1 current tokens [cells, thought] that also attend to the thought memory.

    z: [B, P + 1, H] latents (cells: carried latent; thought: latent_init); x: [B, P + 1, H] inputs (cell embeddings;
    thought input). mem_k / mem_v: [L, B, S, nh, hd]; mem_valid: [S] bool (written slots). The grid convolution acts
    on the cells' latent only. Returns (h [B, P + 1, H], per-layer [(k, v)] of the thought token, each [B, nh, hd]).
    """
    B, N, H = x.shape
    if cfg.use_grid_conv:
        R, C = grid_shape
        grid = jax.lax.conv_general_dilated(
            z[:, :num_cells].reshape(B, R, C, H), params['grid_depthwise_conv']['kernel'], window_strides=(1, 1),
            padding='SAME', dimension_numbers=('NHWC', 'HWIO', 'NHWC'), feature_group_count=H,
        )
        if cfg.conv_bias:
            grid = grid + params['grid_depthwise_conv']['bias']
        z = jnp.concatenate([grid.reshape(B, num_cells, H), z[:, num_cells:]], axis=1)
    a1, b1, a2, b2 = _residual_scales(params, cfg)
    h, a1, b1 = _mixed_precision_input(a2 * z + b2 * x, a1, b1, cfg)
    scale = 1.0 / (math.sqrt(H // cfg.num_heads) * cfg.softmax_temp)
    thought_kv = []
    for i in range(cfg.num_layers):
        p = params[f'block_{i}']
        q, k, v = _qkv(p['self_attn'], rms_norm(h, cfg.rms_norm_eps), cfg)  # [B, N, nh, hd]
        logits_cur = jnp.einsum('bqhd,bkhd->bhqk', q, k) * scale
        logits_mem = jnp.einsum('bqhd,bkhd->bhqk', q, mem_k[i]) * scale
        logits_mem = jnp.where(mem_valid[None, None, None, :], logits_mem, -jnp.inf)
        probs = _attn_softmax(jnp.concatenate([logits_cur, logits_mem], axis=-1))
        out = (jnp.einsum('bhqk,bkhd->bqhd', probs[..., :N], v)
               + jnp.einsum('bhqk,bkhd->bqhd', probs[..., N:], mem_v[i]))
        out = out.reshape(B, N, H) @ p['self_attn']['o_proj']['kernel'].astype(h.dtype)
        h = a1 * h + b1 * out
        h = a1 * h + b1 * _swiglu(p['mlp'], rms_norm(h, cfg.rms_norm_eps))
        thought_kv.append((k[:, -1], v[:, -1]))
    return h.astype(x.dtype), thought_kv


class FPRMMemActorValue(nn.Module):
    """fprm's looped board plus a memory of thought tokens (see the module docstring)."""

    m: int = 3
    n: int = 3
    config: FPRMConfig = FPRMConfig()
    max_think_iters: int = 8
    min_think_iters: int = 2
    # Stop thinking once KL(pi_i || pi_{i-1}) between consecutive readouts is <= this (0 = only at a fixed point,
    # < 0 = always max_think_iters).
    halt_kl: float = 1e-3
    output_dim_1: int = 9   # action logits (= m * n buttons)
    output_dim_2: int = 1   # value
    obs_channels: int = 0          # see FPRMThinkerActorValue
    action_head: str = 'per_cell'  # see FPRMThinkerActorValue
    remat: bool = True             # rematerialize each pass in the backward pass
    thought_norm: bool = False     # rms_norm each thought output before adding the modal embedding
    value_at_max_iters: bool = False  # value from the last (max_think_iters-th) pass's latent instead of the halt pass's
    tokenizer: str = 'cell'        # see FPRMThinkerActorValue
    tokenizer_conv_features: int = 64

    def setup(self):
        assert self.action_head in ('per_cell', 'readout'), self.action_head
        assert self.max_think_iters >= 1, self.max_think_iters
        check_tokenizer(self.tokenizer, self.action_head, self.obs_channels)
        H = self.config.d_model
        self.cell_projection = nn.Dense(H)
        if self.tokenizer != 'cell':
            self.obs_tokenizer = ObsTokenizer(
                self.tokenizer, self.m, self.n, self.obs_channels, H, self.tokenizer_conv_features,
            )
        self.bot = self.param('bot', nn.initializers.normal(stddev=0.02), (1, H))
        # Modal embedding added to every thought fed back into the model.
        self.latent_embed = self.param('latent_embed', nn.initializers.normal(stddev=0.02), (1, H))
        # Parameters only; evaluated functionally (`mem_core`). Grid convolution for one-token-per-cell inputs.
        self.core = FPRMCore(core_config(self.config, self.tokenizer))
        if self.action_head == 'per_cell':
            assert self.output_dim_1 % (self.m * self.n) == 0, (self.output_dim_1, self.m, self.n)
        # per_cell: k = output_dim_1 // (m*n) logits per cell latent (as fprm); readout: logits from the newest thought.
        self.actor_head = nn.Dense(
            self.output_dim_1 // (self.m * self.n) if self.action_head == 'per_cell' else self.output_dim_1,
            bias_init=nn.initializers.zeros,
        )
        self.value_head = nn.Dense(self.output_dim_2, bias_init=nn.initializers.zeros)
        self.latent_init = make_latent_init(self.config)

    def __call__(self, x, normalizer_params=None):
        if normalizer_params is not None:
            x = (x - normalizer_params.mean) / (normalizer_params.std)

        cfg = core_config(self.config, self.tokenizer)
        input_shape = x.shape
        x_flat = x.reshape(-1, input_shape[-1])
        B = x_flat.shape[0]
        H = cfg.d_model
        T = self.max_think_iters
        S = max(T - 1, 1)  # memory slots (the last pass's thought is never read back; >= 1 for static shapes)
        per_cell = self.action_head == 'per_cell'

        if self.tokenizer == 'cell':
            cells = self.cell_projection(tokenize(x_flat, self.m, self.n, self.obs_channels))
        else:
            cells = self.obs_tokenizer(x_flat)  # [B, P, H]
        P = cells.shape[1]
        grid_shape = token_grid_shape(self.tokenizer, self.m, self.n, x_flat.shape[-1], self.tokenizer_conv_features)
        z0 = self.latent_init.astype(cells.dtype)

        if self.is_initializing():
            # Create the core / head parameters (the forward pass below is functional).
            seq = jnp.concatenate([cells[:1], jnp.zeros((1, 1, H), cells.dtype)], axis=1)
            self.core(seq, seq, jnp.ones(seq.shape[:2], bool), grid_shape, prefix_len=0, suffix_len=1)
            self.actor_head(jnp.zeros((1, H)))
            self.value_head(jnp.zeros((1, H)))
        params = self.variables['params']
        core_p, actor_p, value_p = params['core'], params['actor_head'], params['value_head']

        def readout(z_cells, out):
            last_latent = rms_norm(out, eps=cfg.rms_norm_eps)
            if per_cell:
                cell_latents = rms_norm(z_cells, eps=cfg.rms_norm_eps)
                logits = (cell_latents @ actor_p['kernel'] + actor_p['bias']).reshape(B, -1)  # [B, m*n*k], cell-major
            else:
                logits = last_latent @ actor_p['kernel'] + actor_p['bias']
            assert logits.shape[-1] == self.output_dim_1, (logits.shape, self.output_dim_1)
            value = jnp.squeeze(last_latent @ value_p['kernel'] + value_p['bias'], axis=-1)
            return logits, value

        def update(st, z_cells, out):
            """Readout of a new pass, policy-KL halting, and freezing of halted examples."""
            logits, value = readout(z_cells, out)
            iterations = st.iterations + 1
            lp_new = jax.nn.log_softmax(logits, axis=-1)
            lp_prev = jax.nn.log_softmax(st.logits, axis=-1)
            kl = jnp.sum(jnp.exp(lp_new) * (lp_new - lp_prev), axis=-1)
            kl = jnp.where(st.iterations == 0, jnp.inf, kl)  # no previous policy on the first pass
            kl = jax.lax.stop_gradient(kl).astype(jnp.float32)

            # A nonfinite pass keeps the last finite state.
            finite = (jnp.all(jnp.isfinite(logits), axis=-1) & jnp.isfinite(value)
                      & jnp.all(jnp.isfinite(out), axis=-1) & jnp.all(jnp.isfinite(z_cells), axis=(-2, -1)))
            nonfinite = st.nonfinite | ~finite

            # Stop priority: nonfinite > converged > iteration_cap.
            # KL <= halt_kl halts: halt_kl = 0 halts only at a fixed point (consecutive policies equal up to rounding);
            # halt_kl < 0 never halts.
            converged = (iterations >= self.min_think_iters) & (kl <= self.halt_kl) & (self.halt_kl >= 0)
            capped = iterations >= self.max_think_iters
            should_stop = nonfinite | converged | capped
            reason = jnp.full_like(st.stop_reason, STOP_ACTIVE)
            reason = jnp.where(capped, STOP_ITERATION_CAP, reason)
            reason = jnp.where(converged, STOP_CONVERGED, reason)
            reason = jnp.where(nonfinite, STOP_NONFINITE, reason)

            stepped = MemState(
                z=jnp.where(finite[:, None, None], z_cells, st.z),
                token=jnp.where(finite[:, None], out, st.token),
                logits=jnp.where(finite[:, None], logits, st.logits),
                value=jnp.where(finite, value, st.value),
                value_last=jnp.where(finite, value, st.value_last),
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

            new_st = jax.tree_util.tree_map(sel, stepped, st)
            if self.value_at_max_iters:
                # Halting fixes only the policy readout: the trajectory (cells, thought) and the value keep going.
                new_st = new_st.replace(z=stepped.z, token=stepped.token, value_last=stepped.value_last)
            return new_st

        state = MemState(
            z=jnp.broadcast_to(z0[None, None], cells.shape),  # cell latents start at latent_init, as in fprm
            token=jnp.zeros((B, H), cells.dtype),
            logits=jnp.zeros((B, self.output_dim_1), cells.dtype),
            value=jnp.zeros((B,), cells.dtype),
            value_last=jnp.zeros((B,), cells.dtype),
            residual=jnp.full((B,), jnp.inf, jnp.float32),
            iterations=jnp.zeros((B,), jnp.int32),
            active=jnp.ones((B,), bool),
            nonfinite=jnp.zeros((B,), bool),
            stop_reason=jnp.full((B,), STOP_ACTIVE, jnp.int32),
        )
        bot = jnp.broadcast_to(self.bot, (B, H))
        latent_embed = self.latent_embed
        z0_thought = jnp.broadcast_to(z0[None, None], (B, 1, H))
        kv_shape = (cfg.num_layers, B, S, cfg.num_heads, H // cfg.num_heads)
        memory = (jnp.zeros(kv_shape, cfg.compute_dtype), jnp.zeros(kv_shape, cfg.compute_dtype))

        def body(carry, i):
            # Pass i + 1: refine the cells and run thought t_{i+1}, reading the thoughts written by passes 1..i.
            st, (mem_k, mem_v) = carry
            fed_back = rms_norm(st.token, eps=cfg.rms_norm_eps) if self.thought_norm else st.token
            thought = jnp.where(i == 0, bot, fed_back + latent_embed)
            z_in = jnp.concatenate([st.z, z0_thought], axis=1)
            x_in = jnp.concatenate([cells, thought[:, None]], axis=1)
            h, thought_kv = mem_core(core_p, z_in, x_in, mem_k, mem_v, jnp.arange(S) < i, grid_shape, P, cfg)
            # Write this thought's per-layer K/V into slot i (a no-op past the last read slot).
            write = jnp.arange(S) == i
            mem_k = jnp.where(write[None, None, :, None, None], jnp.stack([k for k, _ in thought_kv])[:, :, None], mem_k)
            mem_v = jnp.where(write[None, None, :, None, None], jnp.stack([v for _, v in thought_kv])[:, :, None], mem_v)
            return (update(st, h[:, :P], h[:, P]), (mem_k, mem_v)), None

        if self.remat:
            # Rematerialize each pass in the backward pass (activation memory of one pass, as in fprm).
            body = jax.checkpoint(body, prevent_cse=False)
        (state, _), _ = jax.lax.scan(body, (state, memory), jnp.arange(T))

        logits = state.logits.reshape(input_shape[:-1] + (-1,))
        value = (state.value_last if self.value_at_max_iters else state.value).reshape(input_shape[:-1])
        self.sow('intermediates', 'logits', logits)
        self.sow('intermediates', 'value', value)

        # Diagnostics, retrievable via apply(..., mutable=['intermediates']).
        # `state.residual` holds each example's policy KL at its halt pass.
        self.sow('intermediates', 'fp_iterations', state.iterations)
        self.sow('intermediates', 'fp_residual', state.residual)
        self.sow('intermediates', 'fp_converged', state.stop_reason == STOP_CONVERGED)
        self.sow('intermediates', 'fp_stop_reason', state.stop_reason)

        return distrax.Categorical(logits=logits), value
