"""FPRM actor-critic with implicit chain-of-thought (continuous thought tokens).

Variant of `fprm_thinker.FPRMThinkerActorValue`: instead of feeding the last
latent back into the core (z_{i+1} = core(z_i, x)), every think pass appends
the core's output at the last position as the next token (Coconut-style
continuous chain-of-thought):

    sequence:  [CELL_00, ..., CELL_(m-1,n-1), BOT, THOUGHT_1, ..., THOUGHT_{i-1}]
    pass i:    z_i = core(z0, sequence)          (causal over BOT / thoughts)
               THOUGHT_i = z_i[last position]          (rms_norm'ed if `thought_norm`)

Attention is prefix-causal: the cell tokens attend to each other
bidirectionally, while BOT and every thought attend to the cells and to the
earlier positions only. The causal mask gives the thoughts their order (no
position embedding is needed), and the cell latents never depend on the
thoughts.

The actor/value readout of pass i is decoded from the last position's latent
(the token that would be appended next):
    - value and the 'readout' action head: Dense on rms_norm(z_last).
    - 'per_cell' action head: the cell latents do not see the thoughts, so the
      last latent forms one query per logit slot and cell logit (c, k) is
      <rms_norm(z_cell_c), q_k> / sqrt(d_model).

KV caching: because of the causal mask, earlier positions never change. Pass 1
runs the core over [cells, BOT] once and keeps every layer's keys/values;
pass i > 1 runs the core on the single new token only, attending to those
cached prefix keys/values plus a small buffer of the thought keys/values. The
core is evaluated functionally (`_core_prefill` / `_core_decode`) on the
parameters of an `FPRMCore` submodule, so the parameter tree and
initialization are those of `FPRMCore`. The prefix cache is a constant of the
scan (not carried), so the backward pass stores it once.

Halting is the policy-KL rule of the looped thinker: an example stops after
pass i once KL(pi_i || pi_{i-1}) < `halt_kl` (and i >= `min_think_iters`), or
at `max_think_iters`. The first pass has no previous policy (KL = inf).
Halted examples are frozen inside the differentiable scan, so their gradient
path ends at the pass where they halted. The final readout is sown as
'logits'/'value', together with the same 'fp_*' diagnostics as the looped
thinker, so the two are interchangeable in the PPO loss.

No parameter shape depends on m, n, or the iteration cap.
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
from models.fprm_thinker import tokenize
from models.tokenizers import InputGridConv, ObsTokenizer, check_tokenizer, core_config, num_tokens, token_grid_shape


# ---------------------------------------------------------------------------
# Functional FPRMCore (same math as models/fprm.py, on FPRMCore's parameters)
# ---------------------------------------------------------------------------

def _residual_scales(params, cfg: FPRMConfig):
    """(a1, b1, a2, b2) of FPRMCore's coupled residual scaling."""
    if not cfg.residual_scaling:
        return 1.0, 1.0, 1.0, 1.0
    a1 = jax.nn.sigmoid(params['alpha_1_logit'])
    a2 = jax.nn.sigmoid(params['alpha_2_logit'])
    two_l = 2 * cfg.num_layers
    b2 = 1.0 - a2 * a1**two_l
    b1 = b2 * (1.0 - a1) / (1.0 - a1**two_l + 1e-5)
    return a1, b1, a2, b2


def _qkv(params, x, cfg: FPRMConfig):
    """[..., H] -> q, k, v of shape [..., num_heads, head_dim]."""
    H = x.shape[-1]
    qkv = x @ params['qkv_proj']['kernel']
    qkv = qkv.reshape(x.shape[:-1] + (3 * cfg.num_heads, H // cfg.num_heads))
    q, k, v = jnp.split(qkv, 3, axis=-2)
    return q, k, v


def _swiglu(params, x):
    gate, up = jnp.split(x @ params['gate_up_proj']['kernel'], 2, axis=-1)
    return (jax.nn.silu(gate) * up) @ params['down_proj']['kernel']


def _core_prefill(params, z, tokens, grid_shape, num_cells, cfg: FPRMConfig):
    """FPRMCore over [cells, BOT] with BOT attending to everything and the cells
    not attending to BOT. Returns (output [B, L, H], per-layer (k, v) [B, L, nh, hd])."""
    B, L, H = tokens.shape
    R, C = grid_shape
    if cfg.use_grid_conv:
        grid = z[:, :num_cells].reshape(B, R, C, H)
        grid = jax.lax.conv_general_dilated(
            grid, params['grid_depthwise_conv']['kernel'],
            window_strides=(1, 1), padding='SAME',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            feature_group_count=H,
        )
        if cfg.conv_bias:
            grid = grid + params['grid_depthwise_conv']['bias']
        z = jnp.concatenate([grid.reshape(B, num_cells, H), z[:, num_cells:]], axis=1)
    a1, b1, a2, b2 = _residual_scales(params, cfg)
    h = a2 * z + b2 * tokens

    pos = jnp.arange(L)
    attn_mask = (pos[None, :] < num_cells) | (pos[None, :] <= pos[:, None])  # [L, L]
    scale = 1.0 / (math.sqrt(H // cfg.num_heads) * cfg.softmax_temp)
    cache = []
    for i in range(cfg.num_layers):
        p = params[f'block_{i}']
        q, k, v = _qkv(p['self_attn'], rms_norm(h, cfg.rms_norm_eps), cfg)
        logits = jnp.einsum('bqhd,bkhd->bhqk', q, k) * scale
        logits = jnp.where(attn_mask[None, None], logits, -jnp.inf)
        out = jnp.einsum('bhqk,bkhd->bqhd', jax.nn.softmax(logits, axis=-1), v)
        out = out.reshape(B, L, H) @ p['self_attn']['o_proj']['kernel']
        h = a1 * h + b1 * out
        h = a1 * h + b1 * _swiglu(p['mlp'], rms_norm(h, cfg.rms_norm_eps))
        cache.append((k, v))
    return h, cache


def _core_decode(params, z, token, prefix_cache, buf_k, buf_v, slot, cfg: FPRMConfig):
    """FPRMCore on one new token at thought slot `slot`, attending to the cached
    prefix and to thought slots <= `slot`.

    z, token: [B, H]; buf_k/buf_v: [num_layers, B, S, nh, hd].
    Returns (output [B, H], buf_k, buf_v) with this token's keys/values written.
    """
    B, H = token.shape
    S = buf_k.shape[2]
    a1, b1, a2, b2 = _residual_scales(params, cfg)
    h = a2 * z + b2 * token
    slot_valid = jnp.arange(S) <= slot  # [S]
    scale = 1.0 / (math.sqrt(H // cfg.num_heads) * cfg.softmax_temp)
    for i in range(cfg.num_layers):
        p = params[f'block_{i}']
        q, k, v = _qkv(p['self_attn'], rms_norm(h, cfg.rms_norm_eps), cfg)  # [B, nh, hd]
        buf_k = buf_k.at[i, :, slot].set(k)
        buf_v = buf_v.at[i, :, slot].set(v)
        k_pre, v_pre = prefix_cache[i]
        logits_pre = jnp.einsum('bhd,bkhd->bhk', q, k_pre) * scale
        logits_buf = jnp.einsum('bhd,bkhd->bhk', q, buf_k[i]) * scale
        logits_buf = jnp.where(slot_valid[None, None], logits_buf, -jnp.inf)
        probs = jax.nn.softmax(jnp.concatenate([logits_pre, logits_buf], axis=-1), axis=-1)
        P = k_pre.shape[1]
        out = (
            jnp.einsum('bhk,bkhd->bhd', probs[..., :P], v_pre)
            + jnp.einsum('bhk,bkhd->bhd', probs[..., P:], buf_v[i])
        )
        out = out.reshape(B, H) @ p['self_attn']['o_proj']['kernel']
        h = a1 * h + b1 * out
        h = a1 * h + b1 * _swiglu(p['mlp'], rms_norm(h, cfg.rms_norm_eps))
    return h, buf_k, buf_v


# ---------------------------------------------------------------------------
# Actor-critic
# ---------------------------------------------------------------------------

@flax.struct.dataclass
class CoTState:
    token: jax.Array  # [B, H] last position's latent, appended as the next thought
    logits: jax.Array  # [B, A] readout of the last executed pass
    value: jax.Array  # [B]
    residual: jax.Array  # [B] float32, policy KL at the last executed pass
    iterations: jax.Array  # [B] int32, executed passes
    active: jax.Array  # [B] bool
    nonfinite: jax.Array  # [B] bool
    stop_reason: jax.Array  # [B] int32


class FPRMCoTThinkerActorValue(nn.Module):
    """FPRM actor-critic that thinks by appending continuous thought tokens."""

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
    # RMS-normalize each thought before appending it (no parameters), matching the scale of the
    # normalized readout instead of feeding back the raw core output.
    thought_norm: bool = False
    tokenizer: str = 'cell'        # see FPRMThinkerActorValue
    tokenizer_conv_features: int = 64
    # Residual depthwise 3x3 conv over the cell embeddings (tokenizer='cell'; see models/tokenizers.InputGridConv).
    input_grid_conv: bool = False

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
        # Beginning-of-thought token: the last position of the first pass.
        self.bot = self.param(
            'bot', nn.initializers.normal(stddev=0.02), (1, 1, H)
        )
        # Parameters only; evaluated functionally with KV caching.
        self.core = FPRMCore(core_config(self.config, self.tokenizer))
        if self.action_head == 'per_cell':
            assert self.output_dim_1 % (self.m * self.n) == 0, (self.output_dim_1, self.m, self.n)
            # One query per logit slot (k = output_dim_1 // (m*n)), matched against every cell latent.
            self.actor_head = nn.Dense(
                self.output_dim_1 // (self.m * self.n) * H,
                bias_init=nn.initializers.zeros,
            )
        else:
            self.actor_head = nn.Dense(self.output_dim_1, bias_init=nn.initializers.zeros)
        self.value_head = nn.Dense(
            self.output_dim_2, bias_init=nn.initializers.zeros
        )
        self.latent_init = make_latent_init(self.config)

    def __call__(self, x, normalizer_params=None):
        if normalizer_params is not None:
            x = (x - normalizer_params.mean) / (normalizer_params.std)

        cfg = core_config(self.config, self.tokenizer)
        input_shape = x.shape
        x_flat = x.reshape(-1, input_shape[-1])
        B = x_flat.shape[0]
        H = cfg.d_model
        S = self.max_think_iters - 1  # thought slots (the last pass's output is never read back)
        # Input tokens (cells for tokenizer='cell'), bidirectional before BOT.
        num_cells = num_tokens(
            self.tokenizer, self.m, self.n, x_flat.shape[-1], self.tokenizer_conv_features
        )
        grid_shape = token_grid_shape(
            self.tokenizer, self.m, self.n, x_flat.shape[-1], self.tokenizer_conv_features
        )
        per_cell = self.action_head == 'per_cell'

        if self.tokenizer == 'cell':
            cell_features = tokenize(x_flat, self.m, self.n, self.obs_channels)
            cell_tokens = self.cell_projection(cell_features)
        else:
            cell_tokens = self.obs_tokenizer(x_flat)
        if self.input_grid_conv:
            cell_tokens = self.input_conv(cell_tokens)
        tokens = jnp.concatenate(
            [cell_tokens, jnp.broadcast_to(self.bot, (B, 1, H))], axis=1
        )  # [B, num_cells + 1, H]
        z0 = jnp.broadcast_to(
            self.latent_init[None, None, :].astype(tokens.dtype), tokens.shape
        )

        if self.is_initializing():
            # Create the core / head parameters (the forward pass below is functional).
            self.core(z0, tokens, jnp.ones(tokens.shape[:2], bool), grid_shape,
                      prefix_len=0, suffix_len=1)
            self.actor_head(jnp.zeros((1, H)))
            self.value_head(jnp.zeros((1, H)))
        params = self.variables['params']
        core_p, actor_p, value_p = params['core'], params['actor_head'], params['value_head']

        # Pass 1: [cells, BOT]; keeps every layer's keys/values.
        z, prefix_cache = _core_prefill(
            core_p, z0, tokens, grid_shape, num_cells, cfg
        )
        # Cell latents never see the thoughts: decoded once for the per-cell head.
        cell_latents = rms_norm(z[:, :num_cells], eps=cfg.rms_norm_eps)

        def readout(z_last):
            last_latent = rms_norm(z_last, eps=cfg.rms_norm_eps)
            actor_out = last_latent @ actor_p['kernel'] + actor_p['bias']
            if per_cell:
                queries = actor_out.reshape(B, -1, H)  # [B, k, H]
                logits = jnp.einsum('bch,bkh->bck', cell_latents, queries) / math.sqrt(H)
                logits = logits.reshape(B, -1)  # [B, m*n*k], cell-major
            else:
                logits = actor_out  # [B, output_dim_1]
            assert logits.shape[-1] == self.output_dim_1, (
                logits.shape, self.output_dim_1
            )
            value = jnp.squeeze(last_latent @ value_p['kernel'] + value_p['bias'], axis=-1)
            return logits, value

        def update(st, z_last):
            """Readout of a new pass, policy-KL halting, and freezing of halted examples."""
            logits, value = readout(z_last)
            iterations = st.iterations + 1

            # Policy KL to the previous pass (no previous policy on the first pass).
            lp_new = jax.nn.log_softmax(logits, axis=-1)
            lp_prev = jax.nn.log_softmax(st.logits, axis=-1)
            kl = jnp.sum(jnp.exp(lp_new) * (lp_new - lp_prev), axis=-1)
            kl = jnp.where(st.iterations == 0, jnp.inf, kl)
            kl = jax.lax.stop_gradient(kl).astype(jnp.float32)

            # A nonfinite pass keeps the last finite readout.
            finite = (
                jnp.all(jnp.isfinite(logits), axis=-1)
                & jnp.isfinite(value)
                & jnp.all(jnp.isfinite(z_last), axis=-1)
            )
            nonfinite = st.nonfinite | ~finite

            # Stop priority: nonfinite > converged > iteration_cap.
            converged = (iterations >= self.min_think_iters) & (kl < self.halt_kl)
            capped = iterations >= self.max_think_iters
            should_stop = nonfinite | converged | capped
            reason = jnp.full_like(st.stop_reason, STOP_ACTIVE)
            reason = jnp.where(capped, STOP_ITERATION_CAP, reason)
            reason = jnp.where(converged, STOP_CONVERGED, reason)
            reason = jnp.where(nonfinite, STOP_NONFINITE, reason)

            stepped = CoTState(
                token=jnp.where(finite[:, None], z_last, st.token),
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

        state = CoTState(
            token=jnp.zeros((B, H), tokens.dtype),
            logits=jnp.zeros((B, self.output_dim_1), tokens.dtype),
            value=jnp.zeros((B,), tokens.dtype),
            residual=jnp.full((B,), jnp.inf, jnp.float32),
            iterations=jnp.zeros((B,), jnp.int32),
            active=jnp.ones((B,), bool),
            nonfinite=jnp.zeros((B,), bool),
            stop_reason=jnp.full((B,), STOP_ACTIVE, jnp.int32),
        )
        state = update(state, z[:, num_cells])

        if S > 0:
            # Passes 2..max_think_iters: one new token each, written to thought slot `slot`.
            z0_token = jnp.broadcast_to(self.latent_init[None].astype(tokens.dtype), (B, H))
            kv_shape = (cfg.num_layers, B, S, cfg.num_heads, H // cfg.num_heads)
            buffers = (jnp.zeros(kv_shape, tokens.dtype), jnp.zeros(kv_shape, tokens.dtype))

            def body(carry, slot):
                st, (buf_k, buf_v) = carry
                thought = rms_norm(st.token, eps=cfg.rms_norm_eps) if self.thought_norm else st.token
                z_last, buf_k, buf_v = _core_decode(
                    core_p, z0_token, thought, prefix_cache, buf_k, buf_v, slot, cfg
                )
                return (update(st, z_last), (buf_k, buf_v)), None

            if self.remat:
                # Rematerialize each pass in the backward pass.
                body = jax.checkpoint(body, prevent_cse=False)
            (state, _), _ = jax.lax.scan(body, (state, buffers), jnp.arange(S))

        logits = state.logits.reshape(input_shape[:-1] + (-1,))
        value = state.value.reshape(input_shape[:-1])
        self.sow('intermediates', 'logits', logits)
        self.sow('intermediates', 'value', value)

        # Diagnostics, retrievable via apply(..., mutable=['intermediates']).
        # `state.residual` holds each example's policy KL at its halt pass.
        self.sow('intermediates', 'fp_iterations', state.iterations)
        self.sow('intermediates', 'fp_residual', state.residual)
        self.sow(
            'intermediates', 'fp_converged',
            state.stop_reason == STOP_CONVERGED,
        )
        self.sow('intermediates', 'fp_stop_reason', state.stop_reason)

        return distrax.Categorical(logits=logits), value
