"""Latent-exchange thinker: an autoregressive chain of thought plus latents that read the board and write back to it.

The inductive bias of `fprm` that the Perceiver-style models lack is that the reasoning state takes part in the
board computation: every step refines the board itself, and the action is read from a token that both attends to
and is attended by the cells. This model keeps that bias without cell-to-cell attention, while generating thoughts
autoregressively as in fprm_cot / fprm_perceiver:

    thought t_1 = BOT (learned), t_s = latent_embed + out_{s-1}   (rms_norm(out_{s-1}) first if `thought_norm`)

Per thinking step s (one weight-tied step, `num_layers` untied sublayers, the FPRM core's coupled residual scaling and
input injection - cells re-inject their embeddings x, persistent latents their initial values L0, the thought its
input t_s), with A = [t_s, persistent latents] and M = the thoughts of steps 1..s-1 (memory):

    A <- A + Attn(q = A, kv = [cells, A, M])          the thought and latents read the board and reason together
    A <- A + SwiGLU(rms_norm(A))
    c <- c + Attn(q = c, kv = [A, M])                  write back (cells read the latents and every thought so far)
    c <- c + SwiGLU(rms_norm(DWConv3x3(c)))            local board update
    out_s = the thought's output; the memory keeps t_s's per-layer states (Transformer-XL style: not recomputed)

Attention costs O(P * (K + s)) per step instead of fprm's O(P^2) (P cells, K persistent latents).

Persistent latents (`latent_mode`) - a workspace carried and refined across steps:
    'learned': `num_latents` learned latents, attending to all cells (0 = thoughts only).
    'patch':   one latent per `patch` x `patch` patch of the board, initialized from the mean of its cells'
               embeddings (so each latent starts with a place on the board). With `patch_local` (default), a patch
               latent reads and writes only its own patch's cells; the thoughts always read and write every cell, and
               all latents and thoughts attend to each other.

Readout of step s: value and the 'readout' action head from the newest thought out_s; the 'per_cell' action head from
the refined cells (one Dense per cell, as in fprm). Halting is the policy-KL rule of the other thinkers: all
max_think_iters steps are computed and each example's readout is taken at its halting step (KL(pi_s || pi_{s-1}) <
`halt_kl` once s >= `min_think_iters`, or the cap; a nonfinite step keeps the last finite readout), so steps after
an example's halt get no gradient - the same result as freezing halted examples. The final readout and the 'fp_*'
diagnostics are sown as by the other thinkers. Each step is rematerialized in the backward pass (`remat`). Needs the
'cell' tokenizer (the local update is a grid convolution).
"""

import math

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from models.fprm import (
    FPRMConfig,
    FPRMSwiGLU,
    make_latent_init,
    STOP_ACTIVE,
    STOP_CONVERGED,
    STOP_ITERATION_CAP,
    STOP_NONFINITE,
    pytorch_depthwise_conv_init,
    rms_norm,
)
from models.fprm_thinker import tokenize

LATENT_MODES = ('learned', 'patch')


def _logit(p):
    return math.log(p / (1.0 - p))


def patch_assignment(m, n, patch):
    """[num_patches, m*n] one-hot (row-major cells) of each cell's p x p patch."""
    pr, pc = -(-m // patch), -(-n // patch)
    assign = np.zeros((pr * pc, m * n), np.float32)
    for r in range(m):
        for c in range(n):
            assign[(r // patch) * pc + c // patch, r * n + c] = 1.0
    return assign


def exchange_masks(m, n, latent_mode, num_latents, patch, patch_local):
    """(persistent latent -> cell [K, P], cell -> persistent latent [P, K]) attention masks. The thought tokens are
    global (always attend to / are attended by every cell) and are not part of these masks."""
    P = m * n
    if latent_mode == 'learned':
        lat_cell = np.ones((num_latents, P), bool)
    elif not patch_local:
        lat_cell = np.ones((patch_assignment(m, n, patch).shape[0], P), bool)
    else:
        lat_cell = patch_assignment(m, n, patch).astype(bool)
    return lat_cell, lat_cell.T.copy()


def _attend(q_in, kv_in, mask, cfg: FPRMConfig, name):
    """Multi-head attention (bias-free projections, FPRM style): q from q_in [B, Q, H], k/v from kv_in [B, K, H],
    mask [Q, K] or [B, Q, K] bool. Defined inside a compact module (creates `{name}_q`, `{name}_kv`, `{name}_o`)."""
    B, Q, H = q_in.shape
    nh, hd = cfg.num_heads, H // cfg.num_heads
    init = nn.initializers.truncated_normal(1.0 / math.sqrt(H))
    q = nn.Dense(H, use_bias=False, kernel_init=init, name=f'{name}_q')(q_in).reshape(B, Q, nh, hd)
    kv = nn.Dense(2 * H, use_bias=False, kernel_init=init, name=f'{name}_kv')(kv_in)
    k, v = jnp.split(kv.reshape(B, kv_in.shape[1], 2 * nh, hd), 2, axis=-2)
    scores = jnp.einsum('bqhd,bkhd->bhqk', q, k) / (math.sqrt(hd) * cfg.softmax_temp)
    mask = mask if mask.ndim == 3 else mask[None]
    scores = jnp.where(mask[:, None], scores, -jnp.inf)
    out = jnp.einsum('bhqk,bkhd->bqhd', jax.nn.softmax(scores, axis=-1), v).reshape(B, Q, H)
    return nn.Dense(H, use_bias=False, kernel_init=init, name=f'{name}_o')(out)


class ExchangeLayer(nn.Module):
    """One sublayer. A = [thought, persistent latents] read [cells, A, thought memory] + MLP; then the cells read
    [A, thought memory] + local conv/MLP update. Returns the new (c, A) and the thought's normalized states at the
    two attention inputs (what later steps read from the memory)."""

    m: int
    n: int
    config: FPRMConfig
    lat_cell_mask: tuple  # [K, P] static, nested tuples of bools
    cell_lat_mask: tuple  # [P, K]

    @nn.compact
    def __call__(self, c, A, scales, mem_read, mem_write, mem_valid):
        cfg = self.config
        a1c, b1c, a1l, b1l = scales
        B, P, H = c.shape
        nA, S = A.shape[1], mem_read.shape[1]
        K = nA - 1
        lat_cell = jnp.asarray(self.lat_cell_mask, bool).reshape(K, P)
        cell_lat = jnp.asarray(self.cell_lat_mask, bool).reshape(P, K)
        valid = jnp.broadcast_to(mem_valid[None, :], (1, S))
        # Thought + latents read the board, each other and the thought memory.
        A_n = rms_norm(A, cfg.rms_norm_eps)
        read_state = A_n[:, 0]
        kv = jnp.concatenate([rms_norm(c, cfg.rms_norm_eps), A_n, mem_read], axis=1)
        to_cells = jnp.concatenate([jnp.ones((1, P), bool), lat_cell], axis=0)                  # [nA, P]
        mask = jnp.concatenate([to_cells, jnp.ones((nA, nA), bool), jnp.broadcast_to(valid, (nA, S))], axis=1)
        A = a1l * A + b1l * _attend(A_n, kv, mask, cfg, 'latent_attn')
        A = a1l * A + b1l * FPRMSwiGLU(cfg, name='latent_mlp')(rms_norm(A, cfg.rms_norm_eps))
        # Cells read the thought, their latents and the thought memory (write-back).
        A_n = rms_norm(A, cfg.rms_norm_eps)
        write_state = A_n[:, 0]
        to_lat = jnp.concatenate([jnp.ones((P, 1), bool), cell_lat], axis=1)                    # [P, nA]
        mask = jnp.concatenate([to_lat, jnp.broadcast_to(valid, (P, S))], axis=1)
        c = a1c * c + b1c * _attend(rms_norm(c, cfg.rms_norm_eps), jnp.concatenate([A_n, mem_write], axis=1),
                                    mask, cfg, 'write')
        # Local board update.
        conv = nn.Conv(H, (3, 3), padding='SAME', feature_group_count=H, use_bias=False,
                       kernel_init=pytorch_depthwise_conv_init, name='cell_dwconv')
        y = conv(c.reshape(B, self.m, self.n, H)).reshape(B, P, H)
        c = a1c * c + b1c * FPRMSwiGLU(cfg, name='cell_mlp')(rms_norm(y, cfg.rms_norm_eps))
        return c, A, read_state, write_state


class ExchangeStep(nn.Module):
    """One weight-tied thinking step (input injection, num_layers ExchangeLayers, readout heads), scanned over.

    carry = (cells c, persistent latents L, newest thought output, cell inputs x, initial latents L0, thought memory
    (read [num_layers, B, S, H], write [num_layers, B, S, H])); xs = the step index s - 1."""

    m: int
    n: int
    config: FPRMConfig
    lat_cell_mask: tuple
    cell_lat_mask: tuple
    output_dim_1: int
    output_dim_2: int
    per_cell: bool
    thought_norm: bool

    def _scales(self, name):
        cfg = self.config
        H = cfg.d_model
        if not cfg.residual_scaling:
            return 1.0, 1.0, 1.0, 1.0
        a1 = jax.nn.sigmoid(self.param(f'{name}_alpha_1_logit', nn.initializers.constant(_logit(cfg.alpha_1_init)), (H,)))
        a2 = jax.nn.sigmoid(self.param(f'{name}_alpha_2_logit', nn.initializers.constant(_logit(cfg.alpha_2_init)), (H,)))
        two_l = 2 * cfg.num_layers
        b2 = 1.0 - a2 * a1 ** two_l
        b1 = b2 * (1.0 - a1) / (1.0 - a1 ** two_l + 1e-5)
        return a1, b1, a2, b2

    @nn.compact
    def __call__(self, carry, idx):
        c, L, token, x, L0, z0, (mem_read, mem_write) = carry
        cfg = self.config
        H = cfg.d_model
        B = c.shape[0]
        S = mem_read.shape[2]
        bot = self.param('bot', nn.initializers.normal(stddev=0.02), (1, H))
        latent_embed = self.param('latent_embed', nn.initializers.normal(stddev=0.02), (1, H))
        a1c, b1c, a2c, b2c = self._scales('cell')
        a1l, b1l, a2l, b2l = self._scales('latent')
        # Thought input: BOT at the first step, else the fed-back output + modal embedding.
        fed_back = rms_norm(token, cfg.rms_norm_eps) if self.thought_norm else token
        thought = jnp.where(idx == 0, jnp.broadcast_to(bot, (B, H)), fed_back + latent_embed)
        # Input injection (as fprm's h = a2 * z + b2 * x): cells and persistent latents are carried; the thought
        # starts from the fixed latent init.
        c = a2c * c + b2c * x
        A = jnp.concatenate([(a2l * z0 + b2l * thought)[:, None], a2l * L + b2l * L0], axis=1)
        mem_valid = jnp.arange(S) < idx
        reads, writes = [], []
        for i in range(cfg.num_layers):
            c, A, rs, ws = ExchangeLayer(self.m, self.n, cfg, self.lat_cell_mask, self.cell_lat_mask, name=f'layer_{i}')(
                c, A, (a1c, b1c, a1l, b1l), mem_read[i], mem_write[i], mem_valid)
            reads.append(rs)
            writes.append(ws)
        # Write this thought's per-layer states into memory slot idx (a no-op past the last read slot).
        slot = (jnp.arange(S) == idx)[None, None, :, None]
        mem_read = jnp.where(slot, jnp.stack(reads)[:, :, None], mem_read)
        mem_write = jnp.where(slot, jnp.stack(writes)[:, :, None], mem_write)
        out = A[:, 0]
        readout = rms_norm(out, cfg.rms_norm_eps)
        if self.per_cell:
            k = self.output_dim_1 // (self.m * self.n)
            logits = nn.Dense(k, bias_init=nn.initializers.zeros, name='actor_head')(rms_norm(c, cfg.rms_norm_eps))
            logits = logits.reshape(B, -1)
        else:
            logits = nn.Dense(self.output_dim_1, bias_init=nn.initializers.zeros, name='actor_head')(readout)
        value = jnp.squeeze(nn.Dense(self.output_dim_2, bias_init=nn.initializers.zeros, name='value_head')(readout), -1)
        return (c, A[:, 1:], out, x, L0, z0, (mem_read, mem_write)), (logits, value)


def select_halting(logits, values, min_iters, max_iters, halt_kl):
    """Policy-KL halting over precomputed per-step readouts (logits [T, B, A], values [T, B]); the same state machine
    as the other thinkers' in-scan `update` (first step KL = inf; nonfinite steps keep the last finite readout).
    Returns (logits [B, A], value [B], iterations, residual, stop_reason)."""
    T, B, A = logits.shape

    def step(st, inp):
        lg, v = inp
        logits_prev, value_prev, it, active, nonfinite, reason, residual = st
        it_new = it + 1
        lp_new, lp_prev = jax.nn.log_softmax(lg, -1), jax.nn.log_softmax(logits_prev, -1)
        kl = jnp.sum(jnp.exp(lp_new) * (lp_new - lp_prev), -1)
        kl = jax.lax.stop_gradient(jnp.where(it == 0, jnp.inf, kl)).astype(jnp.float32)
        finite = jnp.all(jnp.isfinite(lg), -1) & jnp.isfinite(v)
        nonfinite_new = nonfinite | ~finite
        converged = (it_new >= min_iters) & (kl < halt_kl)
        capped = it_new >= max_iters
        stop = nonfinite_new | converged | capped
        r = jnp.full_like(reason, STOP_ACTIVE)
        r = jnp.where(capped, STOP_ITERATION_CAP, r)
        r = jnp.where(converged, STOP_CONVERGED, r)
        r = jnp.where(nonfinite_new, STOP_NONFINITE, r)
        new = (jnp.where(finite[:, None], lg, logits_prev), jnp.where(finite, v, value_prev), it_new, ~stop,
               nonfinite_new, r, kl)
        new = jax.tree_util.tree_map(
            lambda a, b: jnp.where(active.reshape((-1,) + (1,) * (a.ndim - 1)), a, b), new, st)
        return new, None

    init = (jnp.zeros((B, A), logits.dtype), jnp.zeros((B,), values.dtype), jnp.zeros((B,), jnp.int32),
            jnp.ones((B,), bool), jnp.zeros((B,), bool), jnp.full((B,), STOP_ACTIVE, jnp.int32),
            jnp.full((B,), jnp.inf, jnp.float32))
    (lg, v, it, _, _, reason, residual), _ = jax.lax.scan(step, init, (logits, values))
    return lg, v, it, residual, reason


class FPRMExchangeActorValue(nn.Module):
    """Latent-exchange thinker (see the module docstring)."""

    m: int = 3
    n: int = 3
    config: FPRMConfig = FPRMConfig()
    max_think_iters: int = 8
    min_think_iters: int = 2
    halt_kl: float = 1e-3
    output_dim_1: int = 9
    output_dim_2: int = 1
    obs_channels: int = 0
    action_head: str = 'per_cell'
    remat: bool = True
    thought_norm: bool = False     # rms_norm each thought output before adding the modal embedding
    latent_mode: str = 'learned'   # 'learned' | 'patch' (persistent latents beside the thought chain)
    num_latents: int = 16          # latent_mode='learned': number of persistent latents (0 = thoughts only)
    patch: int = 2                 # latent_mode='patch': patch side length
    patch_local: bool = True       # latent_mode='patch': patch latents read / write only their own patch

    @nn.compact
    def __call__(self, x, normalizer_params=None):
        assert self.action_head in ('per_cell', 'readout'), self.action_head
        assert self.latent_mode in LATENT_MODES, self.latent_mode
        assert self.max_think_iters >= 1, self.max_think_iters
        if normalizer_params is not None:
            x = (x - normalizer_params.mean) / (normalizer_params.std)
        cfg = self.config
        H = cfg.d_model
        input_shape = x.shape
        x_flat = x.reshape(-1, input_shape[-1])
        B = x_flat.shape[0]
        P = self.m * self.n
        per_cell = self.action_head == 'per_cell'
        if per_cell:
            assert self.output_dim_1 % P == 0, (self.output_dim_1, self.m, self.n)

        cells = nn.Dense(H, name='cell_projection')(tokenize(x_flat, self.m, self.n, self.obs_channels))  # [B, P, H]
        if self.latent_mode == 'learned':
            latents0 = self.param('latents', nn.initializers.normal(stddev=0.02), (self.num_latents, H))
            L0 = jnp.broadcast_to(latents0[None], (B, self.num_latents, H))
        else:
            assign = jnp.asarray(patch_assignment(self.m, self.n, self.patch))
            pooled = jnp.einsum('kp,bph->bkh', assign / assign.sum(1, keepdims=True), cells)  # mean of each patch
            patch_embed = self.param('patch_embed', nn.initializers.normal(stddev=0.02), (1, H))
            L0 = nn.Dense(H, name='patch_proj')(pooled) + patch_embed

        lat_cell, cell_lat = exchange_masks(self.m, self.n, self.latent_mode, self.num_latents, self.patch, self.patch_local)
        to_tuple = lambda a: tuple(tuple(bool(v) for v in row) for row in a)
        step_args = (self.m, self.n, cfg, to_tuple(lat_cell), to_tuple(cell_lat), self.output_dim_1, self.output_dim_2,
                     per_cell, self.thought_norm)
        T = self.max_think_iters
        S = max(T - 1, 1)  # thought memory slots (the last step's thought is never read back)
        mem = jnp.zeros((cfg.num_layers, B, S, H), cells.dtype)
        z0 = jnp.broadcast_to(make_latent_init(cfg).astype(cells.dtype)[None], (B, H))
        # Cells start at their embeddings, persistent latents at L0.
        carry = (cells, L0, jnp.zeros((B, H), cells.dtype), cells, L0, z0, (mem, mem))
        if self.is_initializing():
            # nn.scan cannot create parameters at init; run one step instead.
            _, (lg, v) = ExchangeStep(*step_args, name='step')(carry, jnp.int32(0))
            logits_seq = jnp.broadcast_to(lg, (T,) + lg.shape)
            values_seq = jnp.broadcast_to(v, (T,) + v.shape)
        else:
            step_cls = nn.remat(ExchangeStep, prevent_cse=False) if self.remat else ExchangeStep
            scan = nn.scan(step_cls, variable_broadcast='params', split_rngs={'params': False}, length=T)
            _, (logits_seq, values_seq) = scan(*step_args, name='step')(carry, jnp.arange(T))

        logits, value, iterations, residual, reason = select_halting(
            logits_seq, values_seq, self.min_think_iters, self.max_think_iters, self.halt_kl)
        logits = logits.reshape(input_shape[:-1] + (-1,))
        value = value.reshape(input_shape[:-1])
        self.sow('intermediates', 'logits', logits)
        self.sow('intermediates', 'value', value)
        self.sow('intermediates', 'fp_iterations', iterations)
        self.sow('intermediates', 'fp_residual', residual)
        self.sow('intermediates', 'fp_converged', reason == STOP_CONVERGED)
        self.sow('intermediates', 'fp_stop_reason', reason)
        return distrax.Categorical(logits=logits), value
