"""RiM-style thinker: every thinking step appends a whole re-fed board block to the sequence.

On an m x n board (Sokoban: 100 cells) the sequence grows by one block of m*n + 1 tokens per step:

    block 1:  [x_1 ... x_P, BOT]                              the cell embeddings and a learned BOT token
    block s:  [GridConv(cell outputs of block s-1), SEP]      s >= 2: the P output latents of the previous block,
                                                              re-fed through a grid convolution, and a learned SEP

The FPRM core runs over the sequence with a block-causal mask: every token attends to all tokens of its own block
(bidirectionally) and to every earlier block. Step s produces the outputs of block s: P new cell latents (re-fed as
block s + 1) and the latent at its last position (BOT for s = 1, SEP after). Earlier blocks never change under the
block-causal mask, so their per-layer keys/values are cached: a step runs the core on its own P + 1 tokens only,
attending to the cache (about one fprm iteration plus attention over the growing cache). The cache is carried
through a rematerialized scan, which keeps one copy per step for the backward pass, so training memory grows
~quadratically with max_think_iters.

Re-feeding: the P output latents are passed through a residual depthwise 3x3 convolution over the board, with the
FPRM core's style of per-channel scaled residual (a * h + (1 - a) * DWConv(h), plain h + DWConv(h) without
residual scaling), before they enter the next block. Every block enters the core through its input mixing
a2 * z0 + b2 * tokens (z0 = the fixed latent init); the core's own grid convolution is off (its input would be
constant).

Halting (`halt_criterion`):
    'latent': stop after step s >= `min_think_iters` once the last-position latent has converged,
              ||sep_s - sep_{s-1}|| / ||sep_s|| < `halt_residual_thresh` (step 2 compares SEP with the BOT output);
    'kl':     the policy-KL rule of the other thinkers, KL(pi_s || pi_{s-1}) < `halt_kl`.
The first step has no previous latent / policy (residual = inf); the cap is `max_think_iters`. All steps are
computed and each example's readout is taken at its halting step (later steps get no gradient - the same result as
freezing halted examples); a nonfinite step keeps the last finite readout.

Readout of step s: value and the 'readout' action head from rms_norm(sep_s); the 'per_cell' action head from the
block's P cell latents. The final readout and the 'fp_*' diagnostics (fp_residual = the halting residual at the halt
step) are sown as by the other thinkers. Each step is rematerialized in the backward pass (`remat`). Needs the
'cell' tokenizer.
"""

import dataclasses
import math

import distrax
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
from models.tokenizers import InputGridConv, scaled_residual

HALT_CRITERIA = ('latent', 'kl')


def block_core(params, block, cache_k, cache_v, cache_valid, z0, cfg: FPRMConfig):
    """The core on one block [B, N, H] (bidirectional within the block) attending to the cached earlier blocks.

    cache_k / cache_v: [L, B, C, nh, hd]; cache_valid: [C] bool. Returns (h [B, N, H], per-layer [(k, v)] of the block,
    each [B, N, nh, hd])."""
    B, N, H = block.shape
    a1, b1, a2, b2 = _residual_scales(params, cfg)
    h, a1, b1 = _mixed_precision_input(a2 * z0 + b2 * block, a1, b1, cfg)
    scale = 1.0 / (math.sqrt(H // cfg.num_heads) * cfg.softmax_temp)
    kv = []
    for i in range(cfg.num_layers):
        p = params[f'block_{i}']
        q, k, v = _qkv(p['self_attn'], rms_norm(h, cfg.rms_norm_eps), cfg)
        logits_cache = jnp.einsum('bqhd,bkhd->bhqk', q, cache_k[i]) * scale
        logits_cache = jnp.where(cache_valid[None, None, None, :], logits_cache, -jnp.inf)
        logits_self = jnp.einsum('bqhd,bkhd->bhqk', q, k) * scale
        probs = _attn_softmax(jnp.concatenate([logits_cache, logits_self], axis=-1))
        C = cache_k.shape[2]
        out = (jnp.einsum('bhqk,bkhd->bqhd', probs[..., :C], cache_v[i])
               + jnp.einsum('bhqk,bkhd->bqhd', probs[..., C:], v))
        out = out.reshape(B, N, H) @ p['self_attn']['o_proj']['kernel'].astype(h.dtype)
        h = a1 * h + b1 * out
        h = a1 * h + b1 * _swiglu(p['mlp'], rms_norm(h, cfg.rms_norm_eps))
        kv.append((k, v))
    return h.astype(block.dtype), kv


def select_halting(logits, values, residuals, min_iters, max_iters, thresh):
    """Halting over precomputed per-step readouts (logits [T, B, A], values [T, B]) and halting residuals
    (residuals [T, B], inf at step 1): stop at the first step >= min_iters with residual < thresh, or the cap; a
    nonfinite step keeps the last finite readout. Returns (logits [B, A], value [B], iterations, residual, reason)."""
    T, B, A = logits.shape

    def step(st, inp):
        lg, v, res = inp
        logits_prev, value_prev, it, active, nonfinite, reason, residual = st
        it_new = it + 1
        finite = jnp.all(jnp.isfinite(lg), -1) & jnp.isfinite(v) & ~jnp.isnan(res)
        nonfinite_new = nonfinite | ~finite
        converged = (it_new >= min_iters) & (res < thresh)
        capped = it_new >= max_iters
        stop = nonfinite_new | converged | capped
        r = jnp.full_like(reason, STOP_ACTIVE)
        r = jnp.where(capped, STOP_ITERATION_CAP, r)
        r = jnp.where(converged, STOP_CONVERGED, r)
        r = jnp.where(nonfinite_new, STOP_NONFINITE, r)
        new = (jnp.where(finite[:, None], lg, logits_prev), jnp.where(finite, v, value_prev), it_new, ~stop,
               nonfinite_new, r, res)
        new = jax.tree_util.tree_map(
            lambda a, b: jnp.where(active.reshape((-1,) + (1,) * (a.ndim - 1)), a, b), new, st)
        return new, None

    init = (jnp.zeros((B, A), logits.dtype), jnp.zeros((B,), values.dtype), jnp.zeros((B,), jnp.int32),
            jnp.ones((B,), bool), jnp.zeros((B,), bool), jnp.full((B,), STOP_ACTIVE, jnp.int32),
            jnp.full((B,), jnp.inf, jnp.float32))
    (lg, v, it, _, _, reason, residual), _ = jax.lax.scan(step, init, (logits, values, residuals))
    return lg, v, it, residual, reason


class FPRMRiMActorValue(nn.Module):
    """RiM-style thinker (see the module docstring)."""

    m: int = 3
    n: int = 3
    config: FPRMConfig = FPRMConfig()
    max_think_iters: int = 8
    min_think_iters: int = 2
    halt_criterion: str = 'latent'      # 'latent' (SEP latent converged) | 'kl' (policy KL)
    halt_residual_thresh: float = 0.1   # 'latent': ||sep_s - sep_{s-1}|| / ||sep_s|| below this halts (0 = never)
    halt_kl: float = 1e-3               # 'kl': KL(pi_s || pi_{s-1}) below this halts (0 = never)
    output_dim_1: int = 9
    output_dim_2: int = 1
    obs_channels: int = 0
    action_head: str = 'per_cell'
    remat: bool = True

    @property
    def core_config(self) -> FPRMConfig:
        return dataclasses.replace(self.config, use_grid_conv=False)  # re-fed latents get their own grid conv

    def setup(self):
        assert self.action_head in ('per_cell', 'readout'), self.action_head
        assert self.halt_criterion in HALT_CRITERIA, self.halt_criterion
        assert self.max_think_iters >= 1, self.max_think_iters
        H = self.config.d_model
        self.cell_projection = nn.Dense(H)
        self.bot = self.param('bot', nn.initializers.normal(stddev=0.02), (1, H))
        self.sep = self.param('sep', nn.initializers.normal(stddev=0.02), (1, H))
        # Parameters only; evaluated functionally (`block_core`, the re-feed conv below).
        self.core = FPRMCore(self.core_config)
        self.refeed_conv = InputGridConv(self.m, self.n, residual_scaling=self.config.residual_scaling,
                                         alpha_init=self.config.alpha_1_init)
        if self.action_head == 'per_cell':
            assert self.output_dim_1 % (self.m * self.n) == 0, (self.output_dim_1, self.m, self.n)
        self.actor_head = nn.Dense(
            self.output_dim_1 // (self.m * self.n) if self.action_head == 'per_cell' else self.output_dim_1,
            bias_init=nn.initializers.zeros,
        )
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
        P = self.m * self.n
        N = P + 1             # tokens per block
        T = self.max_think_iters
        per_cell = self.action_head == 'per_cell'

        cells = self.cell_projection(tokenize(x_flat, self.m, self.n, self.obs_channels))  # [B, P, H]
        z0 = self.latent_init.astype(cells.dtype)

        if self.is_initializing():
            dummy = jnp.zeros((1, N, H), cells.dtype)
            self.core(dummy, dummy, jnp.ones((1, N), bool), (1, N), prefix_len=0)
            self.refeed_conv(cells)
            self.actor_head(jnp.zeros((1, H)))
            self.value_head(jnp.zeros((1, H)))
        params = self.variables['params']
        core_p, actor_p, value_p = params['core'], params['actor_head'], params['value_head']
        conv_kernel = params['refeed_conv']['conv']['kernel']
        alpha_logit = params['refeed_conv'].get('alpha_logit')

        def refeed(c):
            """Grid convolution of the P output latents before they are fed back (InputGridConv as a function)."""
            g = jax.lax.conv_general_dilated(
                c.reshape(B, self.m, self.n, H), conv_kernel, window_strides=(1, 1), padding='SAME',
                dimension_numbers=('NHWC', 'HWIO', 'NHWC'), feature_group_count=H,
            ).reshape(c.shape)
            return c + g if alpha_logit is None else scaled_residual(alpha_logit, c, g)

        def readout(cell_out, last):
            last_latent = rms_norm(last, eps=cfg.rms_norm_eps)
            if per_cell:
                logits = (rms_norm(cell_out, cfg.rms_norm_eps) @ actor_p['kernel'] + actor_p['bias']).reshape(B, -1)
            else:
                logits = last_latent @ actor_p['kernel'] + actor_p['bias']
            value = jnp.squeeze(last_latent @ value_p['kernel'] + value_p['bias'], axis=-1)
            return logits, value

        bot = jnp.broadcast_to(self.bot, (B, H))
        sep = jnp.broadcast_to(self.sep, (B, H))
        C = max(T - 1, 1) * N  # cache positions (the last block is never read back)
        kv_shape = (cfg.num_layers, B, C, cfg.num_heads, H // cfg.num_heads)
        cache = (jnp.zeros(kv_shape, cfg.compute_dtype), jnp.zeros(kv_shape, cfg.compute_dtype))

        def run_block(prev_cells, cache_, s):
            """Step s + 1: block [cells, BOT] (s = 0) or [refeed(previous cell outputs), SEP]; returns the block's
            outputs and the cache with its keys/values written at positions [s * N, (s + 1) * N)."""
            ck, cv = cache_
            block_cells = jnp.where(s == 0, cells, refeed(prev_cells))
            block = jnp.concatenate([block_cells, jnp.where(s == 0, bot, sep)[:, None]], axis=1)
            h, kv = block_core(core_p, block, ck, cv, jnp.arange(C) < s * N, z0, cfg)
            # The last block (s = T - 1) is never read back and the cache holds only T - 1 blocks: skip its write.
            start = jnp.minimum(s, C // N - 1) * N
            new_k = jax.lax.dynamic_update_slice_in_dim(ck, jnp.stack([k for k, _ in kv]), start, axis=2)
            new_v = jax.lax.dynamic_update_slice_in_dim(cv, jnp.stack([v for _, v in kv]), start, axis=2)
            keep = s * N >= C
            return h[:, :P], h[:, P], (jnp.where(keep, ck, new_k), jnp.where(keep, cv, new_v))

        def first_step_inf(res, s):
            return jax.lax.stop_gradient(jnp.where(s == 0, jnp.inf, res)).astype(jnp.float32)

        if self.halt_criterion == 'latent':
            def step(carry, s):
                prev_cells, prev_last, cache_ = carry
                cell_out, last, cache_ = run_block(prev_cells, cache_, s)
                logits, value = readout(cell_out, last)
                res = jnp.linalg.norm(last - prev_last, axis=-1) / (jnp.linalg.norm(last, axis=-1) + cfg.eps)
                return (cell_out, last, cache_), (logits, value, first_step_inf(res, s))
            init = (cells, jnp.zeros((B, H), cells.dtype), cache)
            thresh = self.halt_residual_thresh
        else:
            def step(carry, s):
                prev_cells, prev_logits, cache_ = carry
                cell_out, last, cache_ = run_block(prev_cells, cache_, s)
                logits, value = readout(cell_out, last)
                lp_new, lp_prev = jax.nn.log_softmax(logits, -1), jax.nn.log_softmax(prev_logits, -1)
                res = jnp.sum(jnp.exp(lp_new) * (lp_new - lp_prev), -1)
                return (cell_out, logits, cache_), (logits, value, first_step_inf(res, s))
            init = (cells, jnp.zeros((B, self.output_dim_1), cells.dtype), cache)
            thresh = self.halt_kl

        if self.remat:
            # Rematerialize each step in the backward pass. The carried cache is kept once per step for the backward
            # pass, so training memory grows ~quadratically with max_think_iters (Sokoban, minibatch 1024: ~11 GiB at
            # 4 steps, ~25 GiB at 8); use more minibatches per rollout for long chains.
            step = jax.checkpoint(step, prevent_cse=False)
        _, (logits_seq, values_seq, res_seq) = jax.lax.scan(step, init, jnp.arange(T))

        logits, value, iterations, residual, reason = select_halting(
            logits_seq, values_seq, res_seq, self.min_think_iters, T, thresh)
        logits = logits.reshape(input_shape[:-1] + (-1,))
        value = value.reshape(input_shape[:-1])
        self.sow('intermediates', 'logits', logits)
        self.sow('intermediates', 'value', value)
        self.sow('intermediates', 'fp_iterations', iterations)
        self.sow('intermediates', 'fp_residual', residual)
        self.sow('intermediates', 'fp_converged', reason == STOP_CONVERGED)
        self.sow('intermediates', 'fp_stop_reason', reason)
        return distrax.Categorical(logits=logits), value
