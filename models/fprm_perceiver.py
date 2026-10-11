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
     embedding - so they are computed once per decision (or once per pass with `step_grid_conv`).

    latents  x_0 = BOT,  x_i = latent_embed + out_{i-1}   (rms_norm(out_{i-1}) if `thought_norm`)
    pass i:  out_i = Blocks(x_i; attending to cells and x_0..x_i)      (causal over the latents)

Initial latents (`num_initial_latents` K, default 1): BOT is K learned latents [BOT_1, ..., BOT_K] that attend to
the cells and to each other bidirectionally. They are all processed in pass 1, which reads out once from the last
one (and counts as one think iteration); its output is fed back as the first generated latent. Every later latent
attends to all K of them (causal generation after the initial block).

Looped board encoder (`cell_encoder_iters` N > 0): before thinking, the cells go through their own weight-tied
num_layers-layer FPRM core (`cell_core`: separate weights, bidirectional over the cells, grid convolution on its
latent) N times, exactly fprm's loop with input injection, and latent layer l cross-attends to keys/values of the
last iteration's layer-l output (layer-aligned):

    z_0 = x,  z_j = cell_core(z_{j-1}, x)  (j = 1..N, rematerialized)      c_l = layer-l output of iteration N
    latent layer l attends to [K/V_l(c_l), latents so far];  the per-cell pointer reads c_L

Computed once per decision (N cell-core passes, the cost of N fprm iterations). N = 1 is a single encoder pass.

Per-step grid convolution (`step_grid_conv`, tokenizer='cell'): the cells carry a state across passes and every
pass applies a weight-tied residual depthwise 3x3 convolution to it before the latent reads it,

    c_0 = x,  c_i = a * c_{i-1} + (1 - a) * DWConv3x3(c_{i-1})      pass i (i = 1, 2, ...) attends to K/V(c_i)

with the FPRM core's style of residual scaling: a = sigmoid(alpha_logit), learnable per channel, initialized at
config.alpha_1_init (plain c + DWConv(c) if config.residual_scaling is off). Pass i sees every cell's i-hop
neighbourhood; the per-cell pointer head of pass i reads c_i. The cell keys/values
are then recomputed every pass (one depthwise conv + the per-layer K/V projections of the cells); the latent
keys/values stay cached. The cell update is linear and independent of the latents.

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
from models.fprm_cot_thinker import _core_decode, _qkv, _residual_scales, _swiglu
from models.fprm_thinker import tokenize
from models.tokenizers import InputGridConv, ObsTokenizer, check_tokenizer, core_config, scaled_residual, token_grid_shape


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


def encoder_layers(params, z, x, grid_shape, cfg: FPRMConfig):
    """One FPRMCore(z, input_tokens=x) call over the cell tokens alone (bidirectional), returning every layer's output
    [h_1, ..., h_L] (each [B, P, H]); h_L equals the core's output. `grid_shape` (R, C) with R * C = P."""
    B, P, H = x.shape
    if cfg.use_grid_conv:
        R, C = grid_shape
        grid = jax.lax.conv_general_dilated(
            z.reshape(B, R, C, H), params['grid_depthwise_conv']['kernel'], window_strides=(1, 1), padding='SAME',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'), feature_group_count=H,
        )
        if cfg.conv_bias:
            grid = grid + params['grid_depthwise_conv']['bias']
        z = grid.reshape(B, P, H)
    a1, b1, a2, b2 = _residual_scales(params, cfg)
    h = a2 * z + b2 * x
    scale = 1.0 / (math.sqrt(H // cfg.num_heads) * cfg.softmax_temp)
    outs = []
    for i in range(cfg.num_layers):
        p = params[f'block_{i}']
        q, k, v = _qkv(p['self_attn'], rms_norm(h, cfg.rms_norm_eps), cfg)
        attn = jax.nn.softmax(jnp.einsum('bqhd,bkhd->bhqk', q, k) * scale, axis=-1)
        out = jnp.einsum('bhqk,bkhd->bqhd', attn, v).reshape(B, P, H) @ p['self_attn']['o_proj']['kernel']
        h = a1 * h + b1 * out
        h = a1 * h + b1 * _swiglu(p['mlp'], rms_norm(h, cfg.rms_norm_eps))
        outs.append(h)
    return outs


def latent_prefill(params, z0, latents, prefix_cache, cfg: FPRMConfig):
    """The core on K latent tokens [B, K, H] that attend to the cached cell prefix and to each other bidirectionally.
    Returns (outputs [B, K, H], per-layer [(k, v)] each [B, K, nh, hd]) - the latents' cache entries."""
    B, K, H = latents.shape
    a1, b1, a2, b2 = _residual_scales(params, cfg)
    h = a2 * z0 + b2 * latents
    scale = 1.0 / (math.sqrt(H // cfg.num_heads) * cfg.softmax_temp)
    cache = []
    for i in range(cfg.num_layers):
        p = params[f'block_{i}']
        q, k, v = _qkv(p['self_attn'], rms_norm(h, cfg.rms_norm_eps), cfg)  # [B, K, nh, hd]
        k_pre, v_pre = prefix_cache[i]
        keys = jnp.concatenate([k_pre, k], axis=1)
        values = jnp.concatenate([v_pre, v], axis=1)
        attn = jax.nn.softmax(jnp.einsum('bqhd,bkhd->bhqk', q, keys) * scale, axis=-1)
        out = jnp.einsum('bhqk,bkhd->bqhd', attn, values).reshape(B, K, H) @ p['self_attn']['o_proj']['kernel']
        h = a1 * h + b1 * out
        h = a1 * h + b1 * _swiglu(p['mlp'], rms_norm(h, cfg.rms_norm_eps))
        cache.append((k, v))
    return h, cache


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
    # input_grid_conv depth: 1 = a single linear depthwise conv; D > 1 = D ConvNeXt-style blocks (models/tokenizers.py).
    input_grid_conv_depth: int = 1
    # Apply a weight-tied residual depthwise 3x3 conv to the cells at every pass (pass i reads the cells convolved i
    # times; tokenizer='cell'). False = every pass reads the same cells.
    step_grid_conv: bool = False
    # Number of learned initial latents (BOT_1..BOT_K, bidirectional among themselves, all in pass 1; 1 = a single BOT).
    num_initial_latents: int = 1
    # Looped board encoder: number of weight-tied cell-core iterations before thinking (0 = off, every layer reads the
    # cell embeddings). Latent layer l reads the last iteration's layer-l output.
    cell_encoder_iters: int = 0

    @property
    def core_config(self) -> FPRMConfig:
        return dataclasses.replace(self.config, use_grid_conv=False)  # the latents are not a grid

    def setup(self):
        assert self.action_head in ('per_cell', 'readout'), self.action_head
        assert self.max_think_iters >= 1, self.max_think_iters
        check_tokenizer(self.tokenizer, self.action_head, self.obs_channels)
        if self.input_grid_conv:
            assert self.tokenizer == 'cell', "input_grid_conv needs tokenizer='cell' (one token per grid cell)"
            self.input_conv = InputGridConv(self.m, self.n, depth=self.input_grid_conv_depth, config=self.config,
                                            remat=self.remat)
        H = self.config.d_model
        self.cell_projection = nn.Dense(H)
        if self.tokenizer != 'cell':
            self.obs_tokenizer = ObsTokenizer(
                self.tokenizer, self.m, self.n, self.obs_channels, H, self.tokenizer_conv_features,
            )
        assert self.num_initial_latents >= 1, self.num_initial_latents
        self.bot = self.param('bot', nn.initializers.normal(stddev=0.02), (self.num_initial_latents, H))
        # Modal embedding added to every generated latent fed back into the model.
        self.latent_embed = self.param('latent_embed', nn.initializers.normal(stddev=0.02), (1, H))
        # Parameters only; evaluated functionally (`cell_kv`, `_core_decode`).
        self.core = FPRMCore(self.core_config)
        assert self.cell_encoder_iters >= 0, self.cell_encoder_iters
        if self.cell_encoder_iters > 0:
            assert not self.step_grid_conv, 'cell_encoder_iters and step_grid_conv cannot be combined'
            # Board encoder, separate from the latent core; grid convolution for one-token-per-cell inputs.
            # Parameters only; evaluated functionally (`encoder_layers`).
            self.cell_core = FPRMCore(core_config(self.config, self.tokenizer))
        if self.step_grid_conv:
            assert self.tokenizer == 'cell', "step_grid_conv needs tokenizer='cell' (one token per grid cell)"
            # Parameters only; applied functionally inside the pass scan (`step_conv` below).
            # Scaled residual (the FPRM core's residual scaling) unless config.residual_scaling is off.
            self.step_conv = InputGridConv(self.m, self.n, residual_scaling=self.config.residual_scaling,
                                           alpha_init=self.config.alpha_1_init)
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
        T = self.max_think_iters  # passes: pass 1 = the K initial latents, passes 2..T = one generated latent each
        K = self.num_initial_latents
        S = K + T - 1             # latent cache slots
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
            if self.step_grid_conv:
                self.step_conv(cells)
            if self.cell_encoder_iters > 0:
                grid_shape = token_grid_shape(self.tokenizer, self.m, self.n, x_flat.shape[-1], self.tokenizer_conv_features)
                self.cell_core(cells, cells, jnp.ones(cells.shape[:2], bool), grid_shape, prefix_len=0)
            self.actor_head(jnp.zeros((1, H)))
            self.value_head(jnp.zeros((1, H)))
        params = self.variables['params']
        core_p, actor_p, value_p = params['core'], params['actor_head'], params['value_head']

        if self.step_grid_conv:
            conv_kernel = params['step_conv']['conv']['kernel']  # [3, 3, 1, H]
            alpha_logit = params['step_conv'].get('alpha_logit')     # [H] (residual_scaling) or None

            def step_conv(c):
                """One step of the [B, m*n, H] cell state (InputGridConv as a pure function): a * c + (1 - a) * DWConv3x3(c)
                (or c + DWConv3x3(c) without residual scaling)."""
                g = jax.lax.conv_general_dilated(
                    c.reshape(B, self.m, self.n, H), conv_kernel, window_strides=(1, 1), padding='SAME',
                    dimension_numbers=('NHWC', 'HWIO', 'NHWC'), feature_group_count=H,
                )
                g = g.reshape(c.shape)
                return c + g if alpha_logit is None else scaled_residual(alpha_logit, c, g)
        elif self.cell_encoder_iters > 0:
            # Looped board encoder (cells only, independent of the latents), once per decision.
            enc_cfg = core_config(self.config, self.tokenizer)
            grid_shape = token_grid_shape(self.tokenizer, self.m, self.n, x_flat.shape[-1], self.tokenizer_conv_features)
            cell_p = params['cell_core']

            def encoder_iter(z, _):
                return encoder_layers(cell_p, z, cells, grid_shape, enc_cfg)[-1], None

            z = cells
            if self.cell_encoder_iters > 1:
                # Iterations 1..N-1, each rematerialized in the backward pass (as fprm's loop).
                it = jax.checkpoint(encoder_iter, prevent_cse=False) if self.remat else encoder_iter
                z, _ = jax.lax.scan(it, z, None, length=self.cell_encoder_iters - 1)
            board = encoder_layers(cell_p, z, cells, grid_shape, enc_cfg)  # iteration N, every layer's output
            fixed_prefix_cache = [_qkv(core_p[f'block_{i}']['self_attn'], rms_norm(board[i], cfg.rms_norm_eps), cfg)[1:]
                                  for i in range(cfg.num_layers)]
            fixed_cell_latents = rms_norm(board[-1], eps=cfg.rms_norm_eps)
        else:
            # Fixed cells: their keys/values and pointer keys are computed once per decision.
            fixed_prefix_cache = cell_kv(core_p, cells, z0, cfg)
            fixed_cell_latents = rms_norm(cells, eps=cfg.rms_norm_eps)

        def readout(out, cell_latents):
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

        def update(st, out, cell_latents):
            """Readout of a new pass, policy-KL halting, and freezing of halted examples."""
            logits, value = readout(out, cell_latents)
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
        latent_embed = self.latent_embed

        def pass_cells(c):
            """(cell state, prefix K/V, pointer keys) read by the next pass."""
            if self.step_grid_conv:
                c = step_conv(c)  # pass i reads the cells convolved i times
                return c, cell_kv(core_p, c, z0, cfg), rms_norm(c, eps=cfg.rms_norm_eps)
            return c, fixed_prefix_cache, fixed_cell_latents

        # Pass 1: the K initial latents, bidirectional among themselves; read out from the last one.
        c, prefix_cache, cell_latents = pass_cells(cells)
        initial = jnp.broadcast_to(self.bot[None], (B, K, H))
        init_out, init_kv = latent_prefill(core_p, z0, initial, prefix_cache, cfg)
        kv_shape = (cfg.num_layers, B, S, cfg.num_heads, H // cfg.num_heads)
        buf_k = jnp.zeros(kv_shape, cells.dtype).at[:, :, :K].set(jnp.stack([k for k, _ in init_kv]))
        buf_v = jnp.zeros(kv_shape, cells.dtype).at[:, :, :K].set(jnp.stack([v for _, v in init_kv]))
        state = update(state, init_out[:, -1], cell_latents)

        def body(carry, slot):
            # Passes 2..T: one generated latent each, written to cache slot `slot` (K .. S - 1).
            st, c, (buf_k, buf_v) = carry
            c, prefix_cache, cell_latents = pass_cells(c)
            fed_back = rms_norm(st.token, eps=cfg.rms_norm_eps) if self.thought_norm else st.token
            out, buf_k, buf_v = _core_decode(core_p, z0_token, fed_back + latent_embed, prefix_cache,
                                             buf_k, buf_v, slot, cfg)
            return (update(st, out, cell_latents), c, (buf_k, buf_v)), None

        if T > 1:
            if self.remat:
                # Rematerialize each pass in the backward pass.
                body = jax.checkpoint(body, prevent_cse=False)
            (state, _, _), _ = jax.lax.scan(body, (state, c, (buf_k, buf_v)), jnp.arange(K, S))

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
