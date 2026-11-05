"""FPRM with a thought memory (the `fprm_mem` and `fprm_mem_halt` architectures), with or without learned halting.

Trunk. Every pass refines the board exactly as `fprm` does - the cells carry a latent z that goes through one FPRM core
call per pass, with the grid convolution on z and the cell embeddings re-injected - and additionally runs a SEP token
and one *thought* token alongside the cells:

    pass i:  tokens  [z_cells, SEP, t_i]                 (P cells + SEP + 1 thought, attending to each other)
             memory  K/V of t_1 .. t_{i-1}, per layer     (read by every token)
             [z_cells, z_sep, out_i] = Core([z_cells, SEP, t_i]; memory)
             write t_i's per-layer K/V into the memory
    t_1 = BOT (learned),  t_i = latent_embed + out_{i-1}   (rms_norm(out_{i-1}) first if `thought_norm`;
                                                        simplicial(rms_norm(out_{i-1})) if `sem_thought`)

The cells and SEP (a learned input embedding) carry their latents between passes; the thought's latent starts at
latent_init every pass, so its only link to the previous pass is its input. A thought's keys/values are those computed
in the pass that wrote it (not recomputed when the cells change later), Transformer-XL style; the memory has
max_think_iters - 1 slots, preallocated and masked until written. Cost per pass: one core call over P + 2 tokens plus
the memory keys. Each pass is rematerialized in the backward pass (`remat`). No parameter shape depends on m, n or the
iteration cap.

`halting=False` (the `fprm_mem` architecture, arm iso_mem): no halting head; every example runs all max_think_iters
passes and acts with the last one (c = T).

`halting=True` (`fprm_mem_halt`): a yes/no halt at every pass n from min_think_iters on, with probability h_n; the
first halt sets the think time c = n, and no halt before pass T stops at the cap (c = T).

What a decision at pass n reads. Let h_n be the latent at the readout head (the readout token's output at pass n, from
which pi_n is decoded). `halt_signal` picks the pair (z_n, z_{n+1}) a convergence test compares and its distance S:
  'kl':     z_n = pi_n,  S_{n+1} = KL(pi_{n+1} || pi_n)
  'tv':     z_n = pi_n,  S_{n+1} = TV(pi_{n+1}, pi_n) = 1/2 sum_a |pi_{n+1}(a) - pi_n(a)|
  'latent': z_n = h_n,   S_{n+1} = ||rms_norm(h_{n+1}) - rms_norm(h_n)||   (L2 of what the readout heads read)
The head (the kl_only-prev_latent head) looks one pass ahead: logit h_n = beta_{n+1} - alpha log S_{n+1}. Pass n+1 is
computed, and if the head halts, pass n acts (c = n means "pass n+1 agrees with pass n"), so c = 1 is decided by a real
convergence test on pass 2 (use min_think_iters = 1). S is stop-gradient'ed.
alpha = halt_alpha_min + softplus(.) (1 at init), beta a learned scalar (log halt_kl at init) or, with `halt_beta_head`,
plus a zero-initialized Dense on rms_norm(h_{n+1}) (stop-gradient'ed unless `halt_head_grad`); h_n = 1/2 at S = halt_kl
at init. `halt_threshold_fixed`: beta = alpha log halt_kl, so the threshold stays at halt_kl and alpha only sets the
sharpness. Decision 1's S_1 is against uniform ('kl', 'tv') or the initial latent ('latent').

Each pass runs [cells, SEP, thought]. SEP is a learned input embedding whose latent is carried between passes like the
cells' (attending to the cells, the current thought and the thought memory); the thought's per-layer K/V is what the
memory keeps. `readout_token` picks where every readout (logits, value, the halting head's input) comes from: the
thought's output ('thought') or SEP's latent ('sep'; the thoughts then only feed the memory and the next thought, so
they need not be decodable as a policy themselves). `value_token` / `halt_token` ('' = readout_token) move the value
or the halting head's h_n to the other token, e.g. readout_token='thought', value_token='sep', halt_token='sep': the
thought acts, SEP evaluates and decides the halt.

The PPO ratio scores log p(c | s) = sum_{n < c} log(1 - h_n) + log h_c (no h_T term: c = T is the cap). The rollout
also stores the decision passes (`halt_path`, a bitmask: bit n-1 = a decision at pass n), a function of c. Evaluation
takes the most likely c, and 'halt_entropy' is H(c).

`halt_kl_reverse`: KL_i = KL(pi_{i-1} || pi_i) instead, the expectation under pi_{i-1}, i.e. under pi_c, the policy that
acts; the two directions agree to second order for small policy changes.

`stable_kl` / `stable_softmax`: the fixes of models/stable_softmax.py (Barron et al., arXiv 2610.31642). stable_kl
computes KL_i from the logit difference, exact down to ~1e-6 relative (the standard sum q (log q - log p) has a ~1e-7
floor in fp32 and is non-zero at equal policies). stable_softmax gives the policy and the halting head collapse-free
log-probabilities (no zeroed gradient on an action whose margin exceeds ~16.6).

The think time c is an action of the joint policy: log p(c, a | s) = log p(c | s) +
log pi_c(a | s). c = T is a forced halt: pi_T acts with no pass after it to check it against. All T passes are always
executed; the readout of pass c is selected afterwards, so passes after c get no gradient from the actor loss.

Choice of c in `__call__`: `halt_steps` given (the PPO loss replays the rollout's c) > `halt_key` given (rollout: a
sampled c) > neither (evaluation: the most likely c).

Action heads (`action_head`): 'per_cell' = one logit per cell latent (Lights Out presses), 'readout' = Dense(A) on the
readout token (moves), 'neighbour' = 4 moves of the blank (sliding puzzle; observation channel 0 marks the blank) whose
logits are the per-cell logits of the blank's up / right / down / left neighbours (the tile that would slide into it,
the order of envs/slidingpuzzle_env.MOVES), off-board moves masked to probability 0. 'neighbour' reads the policy from
the cell latents like 'per_cell' while keeping the 4-move action space. 'blank' = 4 Dense(1) on the blank's own cell
latent, move i's reading only the i-th quarter of its H features, rms-normed on its own (the same 4 moves, unmasked: an off-board move is a
no-op, as with 'readout'). `cell_geometry=False` drops the 6
geometry features (row, col, top, bottom, left, right) from the cell tokens, leaving the observation planes.

The value is read out from the last pass: since c is an action, a baseline that depended on it would bias the
halting gradient.

Sown intermediates: 'logits'/'value' (at the chosen c), 'fp_iterations' (= c), 'fp_residual' (S_{c+1}, the signal that
decided the halt; KL_T without halting), 'fp_converged' / 'fp_stop_reason', 'halt_signal_t' (S_i by pass, [T, B],
unshifted), with halting also 'halt_log_prob'
(log p(c | s)), 'halt_decision_log_probs' (its terms by decision pass, [B, T], 0 off the path), 'halt_entropy', 'halt_expected_c' (E[c | s]), 'halt_path', 'halt_prob' (h_n, [T, B]), 'halt_step_probs' ((h_n, 1 - h_n),
[T, B, 2]), 'halt_step_log_probs' (their logs, [T, B, 2]), 'halt_log_sig_t' (log S_{n+1} at row n, [T, B]) and 'policy_log_probs_t' (log pi_n by pass, [T, B, A]), and always 'policy_kl_t' (KL_i by pass, [T, B], unshifted; pass 1 is
against uniform) and 'latent_residual_t' (the max-token residual of the cell latents between consecutive passes, as
fprm's halt_criterion='latent_residual'; [T, B], pass 1 is against the initial latent) and 'readout_latent_t' (the latent pi_n is decoded from, by pass, [T, B, H]).
"""

import math

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from models.fprm import (
    FPRMConfig,
    FPRMCore,
    STOP_CONVERGED,
    STOP_ITERATION_CAP,
    STOP_NONFINITE,
    compute_residuals,
    make_latent_init,
    rms_norm,
    simplicial,
)
from models import stable_softmax
from models.fprm_thinker import tokenize
from models.tokenizers import ObsTokenizer, check_tokenizer, core_config, token_grid_shape

# ---------------------------------------------------------------------------
# Functional FPRMCore block primitives (same math as models/fprm.py, on FPRMCore's parameters)
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


def _mixed_precision_input(h, a1, b1, cfg: FPRMConfig):
    """Cast the core's mixed input and residual scales to cfg.compute_dtype (as FPRMCore does); the caller casts the
    core's output back to the carried latent's dtype."""
    h = h.astype(cfg.compute_dtype)
    if not isinstance(a1, float):
        a1, b1 = a1.astype(h.dtype), b1.astype(h.dtype)
    return h, a1, b1


def _attn_softmax(logits):
    """Attention softmax computed in float32, returned in the logits' dtype."""
    return jax.nn.softmax(logits.astype(jnp.float32), axis=-1).astype(logits.dtype)


def _qkv(params, x, cfg: FPRMConfig):
    """[..., H] -> q, k, v of shape [..., num_heads, head_dim], in cfg.compute_dtype."""
    H = x.shape[-1]
    x = x.astype(cfg.compute_dtype)
    qkv = x @ params['qkv_proj']['kernel'].astype(x.dtype)
    qkv = qkv.reshape(x.shape[:-1] + (3 * cfg.num_heads, H // cfg.num_heads))
    q, k, v = jnp.split(qkv, 3, axis=-2)
    return q, k, v


def _swiglu(params, x):
    """The block MLP (FPRMSwiGLU) as a pure function; SwiGLU or GELU depending on which parameters exist."""
    if 'up_proj' in params:  # mlp_type='gelu'
        return jax.nn.gelu(x @ params['up_proj']['kernel'].astype(x.dtype)) @ params['down_proj']['kernel'].astype(x.dtype)
    gate, up = jnp.split(x @ params['gate_up_proj']['kernel'].astype(x.dtype), 2, axis=-1)
    return (jax.nn.silu(gate) * up) @ params['down_proj']['kernel'].astype(x.dtype)


def mem_core(params, z, x, mem_k, mem_v, mem_valid, grid_shape, num_cells, cfg: FPRMConfig):
    """One FPRM core call over the current tokens [cells, SEP, thought] that also attend to the thought memory.

    z: [B, N, H] latents (cells, SEP: carried latent; thought: latent_init); x: [B, N, H] inputs (cell embeddings, SEP,
    thought input). mem_k / mem_v: [L, B, S, nh, hd]; mem_valid: [S] bool (written slots). The grid convolution acts
    on the cells' latent only. Returns (h [B, N, H], per-layer [(k, v)] of the last (thought) token, each [B, nh, hd]).
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


_NEG = -1e9  # log of a forced-zero probability (finite: keeps p log p and the gradients NaN-free)
_BLANK_MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))  # action_head='neighbour': up, right, down, left (envs/slidingpuzzle_env.MOVES)


def neighbour_table(m, n):
    """action_head='neighbour': for every blank cell, the cell index of its neighbour in each move [m*n, 4] (the blank
    itself off the board) and whether that move stays on the board [m*n, 4]."""
    r, c = np.divmod(np.arange(m * n), n)
    tr = r[:, None] + np.array([dr for dr, _ in _BLANK_MOVES])
    tc = c[:, None] + np.array([dc for _, dc in _BLANK_MOVES])
    ok = (tr >= 0) & (tr < m) & (tc >= 0) & (tc < n)
    return np.where(ok, tr * n + tc, np.arange(m * n)[:, None]), ok


class SplitDense(nn.Module):
    """action_head='blank': `features` Dense(1), the i-th reading only the i-th of `features` equal chunks of the input,
    each chunk rms-normed on its own (kernel [in / features, features]: column i is head i's weights)."""
    features: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        assert x.shape[-1] % self.features == 0, (x.shape, self.features)
        kernel = self.param('kernel', nn.initializers.lecun_normal(), (x.shape[-1] // self.features, self.features))
        bias = self.param('bias', nn.initializers.zeros, (self.features,))
        return split_dense(x, kernel, bias, self.eps)


def split_dense(x, kernel, bias, eps):
    x = rms_norm(x.reshape(x.shape[:-1] + (kernel.shape[1], -1)), eps)
    return jnp.einsum('...gd,dg->...g', x, kernel) + bias
_KL_EPS = 1e-10  # floor inside log KL (KL may round to 0 or slightly below)


def natural_grad_fisher(f, p, mask, damping):
    """Fisher of the halt logit's (beta, alpha_raw), d logit = d beta + f d alpha_raw: mean over states of sum over
    decision passes of p (1 - p) phi phi^T, phi = [1, f], plus damping * I. f, p, mask: [passes, B]."""
    w = jnp.where(mask, p * (1 - p), 0.0)
    phi = jnp.stack([jnp.ones_like(f), f], axis=-1)
    return jnp.einsum('nb,nbi,nbj->ij', w, phi, phi) / p.shape[1] + damping * jnp.eye(2)


def _decision_passes(T, min_iters):
    """[T, 1] pass numbers n and whether pass n decides (min_iters <= n < T)."""
    n = jnp.arange(1, T + 1)[:, None]
    return n, (n >= min_iters) & (n < T)


def c_log_probs(lp, min_iters):
    """log p(c | s) for c = 1..T, [T, B], from lp [T, B, 2] (row n - 1: log h_n, log(1 - h_n)); _NEG where c < min_iters."""
    T = lp.shape[0]
    n, dec = _decision_passes(T, min_iters)
    lp_cont = jnp.where(dec, lp[..., 1], 0.0)
    before = jnp.cumsum(jnp.concatenate([jnp.zeros_like(lp_cont[:1]), lp_cont[:-1]]), axis=0)  # sum over passes < c
    return jnp.where(n >= min(min_iters, T), before + jnp.where(dec, lp[..., 0], 0.0), _NEG)


def decision_log_probs(lp, c, min_iters):
    """The terms of log p(c | s) by decision pass, [T, B]: log(1 - h_n) for n < c, log h_c at n = c < T, 0 elsewhere."""
    n, dec = _decision_passes(lp.shape[0], min_iters)
    return jnp.where(dec & (n < c), lp[..., 1], jnp.where(dec & (n == c), lp[..., 0], 0.0))


def decision_mask(c, T, min_iters):
    """The decision passes of the path ending at c as a bitmask [B] (bit n - 1 = a decision at pass n)."""
    n, dec = _decision_passes(T, min_iters)
    return jnp.sum(jnp.where(dec & (n <= c), 1 << (n - 1), 0), axis=0).astype(jnp.int32)


def sample_c(lp, min_iters, key):
    """Sample c [B]: the first decision pass that halts, else T."""
    n, dec = _decision_passes(lp.shape[0], min_iters)
    halt = jax.random.categorical(key, lp, axis=-1) == 0  # [T, B]
    return jnp.min(jnp.where(dec & halt, n, lp.shape[0]), axis=0).astype(jnp.int32)


class FPRMMemHaltActorValue(nn.Module):
    """Looped board + SEP + thought memory, with (halting=True) or without learned halting (see the module docstring)."""

    m: int = 3
    n: int = 3
    config: FPRMConfig = FPRMConfig()
    max_think_iters: int = 8
    min_think_iters: int = 2
    halting: bool = True           # False: no halting head, always act with pass max_think_iters (iso_mem)
    act_iters: int = 0             # halting=False: act with this pass instead (0 = max_think_iters); the value keeps value_pass
    halt_head_grad: bool = False   # let the halting loss backpropagate into the trunk through halt_beta_head's h input
    halt_kl_reverse: bool = False  # KL_i = KL(pi_{i-1} || pi_i) (see the module docstring)
    stable_kl: bool = False        # KL_i without the fp32 rounding floor (see the module docstring)
    stable_softmax: bool = False   # collapse-free log-softmax for the policy and the steps head (see the module docstring)
    halt_signal: str = 'kl'        # 'kl' | 'tv' | 'latent': the distance S the head reads (see the module docstring)
    halt_kl: float = 1e-3          # initialization only, h_n = 1/2 at S = halt_kl (beta = log halt_kl, alpha = 1)
    halt_alpha_min: float = 0.0    # floor on alpha (alpha -> 0 is signal-blind halting)
    halt_beta_head: bool = False   # beta = bias + Dense(rms_norm(h_{n+1})) (state-dependent); False: a scalar
    halt_threshold_fixed: bool = False  # beta = alpha log halt_kl (see the module docstring)
    readout_token: str = 'thought'  # readouts from the newest thought ('thought') or the carried SEP token ('sep')
    value_token: str = ''          # '' (= readout_token) | 'thought' | 'sep': where the value is read from
    halt_token: str = ''           # '' (= readout_token) | 'thought' | 'sep': where the halting head's h_n is read from
    value_pass: str = 'last'       # 'last': the value from pass T | 'halt': from the acting pass c (V then depends on c)
    output_dim_1: int = 9   # action logits (= m * n buttons)
    output_dim_2: int = 1   # value
    obs_channels: int = 0          # see FPRMThinkerActorValue
    action_head: str = 'per_cell'  # 'per_cell' | 'readout' (see FPRMThinkerActorValue) | 'neighbour' | 'blank' (see the module docstring)
    cell_geometry: bool = True     # tokenizer='cell': append the 6 geometry features to each cell token (see the module docstring)
    remat: bool = True             # rematerialize each pass in the backward pass
    thought_norm: bool = False     # rms_norm each thought output before adding the modal embedding
    sem_readout: bool = False      # actor/value heads read simplicial(rms_norm(latent)) instead of rms_norm(latent)
    sem_thought: bool = False      # feed back simplicial(rms_norm(out_i)) as the next thought (replaces thought_norm)
    sem_groups: int = 16           # SEM: number of softmax groups (d_model / sem_groups values per group)
    sem_tau: float = 1.0           # SEM: softmax temperature
    tokenizer: str = 'cell'        # see FPRMThinkerActorValue
    tokenizer_conv_features: int = 64

    def setup(self):
        assert self.action_head in ('per_cell', 'readout', 'neighbour', 'blank'), self.action_head
        assert self.action_head not in ('neighbour', 'blank') or (self.output_dim_1 == 4 and self.obs_channels > 0), \
            f"{self.action_head!r} needs 4 actions and a plane observation whose channel 0 marks the blank"
        assert self.cell_geometry or (self.tokenizer == 'cell' and self.obs_channels > 0), \
            'cell_geometry=False needs tokenizer=cell and a plane observation'

        assert self.max_think_iters >= 1, self.max_think_iters
        assert not (self.sem_readout or self.sem_thought) or self.config.d_model % self.sem_groups == 0, self.sem_groups
        assert not (self.sem_readout and self.action_head == 'blank'), "sem_readout: 'blank' rms-norms its own chunks"
        assert not (self.sem_thought and self.thought_norm), 'sem_thought replaces thought_norm'
        assert self.halt_signal in ('kl', 'tv', 'latent'), self.halt_signal
        assert self.readout_token in ('thought', 'sep'), self.readout_token
        assert self.value_token in ('', 'thought', 'sep'), self.value_token
        assert self.halt_token in ('', 'thought', 'sep'), self.halt_token
        assert self.value_pass in ('last', 'halt'), self.value_pass
        assert 0 <= self.act_iters <= self.max_think_iters and (self.act_iters == 0 or not self.halting), self.act_iters
        assert (self.halt_token or self.readout_token) == self.readout_token or self.halting, 'halt_token needs halting'
        if self.halting:
            assert self.halt_kl > 0 and 0.0 <= self.halt_alpha_min < 1.0, (self.halt_kl, self.halt_alpha_min)
            assert not (self.halt_threshold_fixed and self.halt_beta_head), 'halt_threshold_fixed replaces beta'
            a0 = 1.0 - self.halt_alpha_min
            self.halt_alpha_raw = self.param('halt_alpha_raw', nn.initializers.constant(math.log(math.expm1(a0))), ())
            self.halt_beta = nn.Dense(1, kernel_init=nn.initializers.zeros,
                                      bias_init=nn.initializers.constant(math.log(self.halt_kl)))
        check_tokenizer(self.tokenizer, self.action_head, self.obs_channels)
        H = self.config.d_model
        self.cell_projection = nn.Dense(H)
        if self.tokenizer != 'cell':
            self.obs_tokenizer = ObsTokenizer(
                self.tokenizer, self.m, self.n, self.obs_channels, H, self.tokenizer_conv_features,
            )
        self.bot = self.param('bot', nn.initializers.normal(stddev=0.02), (1, H))
        self.latent_embed = self.param('latent_embed', nn.initializers.normal(stddev=0.02), (1, H))
        self.sep = self.param('sep', nn.initializers.normal(stddev=0.02), (1, H))
        self.core = FPRMCore(core_config(self.config, self.tokenizer))
        if self.action_head == 'per_cell':
            assert self.output_dim_1 % (self.m * self.n) == 0, (self.output_dim_1, self.m, self.n)
        if self.action_head == 'blank':
            assert H % self.output_dim_1 == 0, (H, self.output_dim_1)
            self.actor_head = SplitDense(self.output_dim_1)
        else:
            self.actor_head = nn.Dense(
                {'per_cell': self.output_dim_1 // (self.m * self.n), 'neighbour': 1}.get(self.action_head, self.output_dim_1),
                bias_init=nn.initializers.zeros,
            )
        self.value_head = nn.Dense(self.output_dim_2, bias_init=nn.initializers.zeros)
        self.latent_init = make_latent_init(self.config)

    def __call__(self, x, normalizer_params=None, halt_key=None, halt_steps=None):
        if self.action_head in ('neighbour', 'blank'):  # the blank from the raw observation, before normalization
            blank = jnp.argmax(x.reshape(-1, self.m * self.n, self.obs_channels)[..., 0], axis=-1)  # [B]
        if self.action_head == 'neighbour':
            nbr_idx, nbr_ok = (jnp.asarray(t)[blank] for t in neighbour_table(self.m, self.n))  # [B, 4] each
        if normalizer_params is not None:
            x = (x - normalizer_params.mean) / (normalizer_params.std)

        cfg = core_config(self.config, self.tokenizer)
        input_shape = x.shape
        x_flat = x.reshape(-1, input_shape[-1])
        B = x_flat.shape[0]
        H = cfg.d_model
        T = self.max_think_iters
        S = max(T - 1, 1)
        per_cell = self.action_head in ('per_cell', 'neighbour')

        if self.tokenizer == 'cell':
            cells = self.cell_projection(tokenize(x_flat, self.m, self.n, self.obs_channels, self.cell_geometry))
        else:
            cells = self.obs_tokenizer(x_flat)  # [B, P, H]
        P = cells.shape[1]
        grid_shape = token_grid_shape(self.tokenizer, self.m, self.n, x_flat.shape[-1], self.tokenizer_conv_features)
        z0 = self.latent_init.astype(cells.dtype)

        if self.is_initializing():
            seq = jnp.concatenate([cells[:1], jnp.zeros((1, 1, H), cells.dtype)], axis=1)
            self.core(seq, seq, jnp.ones(seq.shape[:2], bool), grid_shape, prefix_len=0, suffix_len=1)
            self.actor_head(jnp.zeros((1, H)))
            self.value_head(jnp.zeros((1, H)))
            if self.halting:
                self.halt_beta(jnp.zeros((1, H)))
        params = self.variables['params']
        core_p, actor_p, value_p = params['core'], params['actor_head'], params['value_head']

        sep = self.readout_token == 'sep'
        value_sep = (self.value_token or self.readout_token) == 'sep'
        halt_sep = (self.halt_token or self.readout_token) == 'sep'
        sep_in = jnp.broadcast_to(self.sep, (B, 1, H))

        def feat(h):  # what the actor/value heads read
            h = rms_norm(h, eps=cfg.rms_norm_eps)
            return simplicial(h, self.sem_groups, self.sem_tau) if self.sem_readout else h

        def readout(z_cells, act_latent, value_latent):
            if per_cell:
                cell_latents = feat(z_cells)
                logits = (cell_latents @ actor_p['kernel'] + actor_p['bias']).reshape(B, -1)
                if self.action_head == 'neighbour':
                    logits = jnp.where(nbr_ok, jnp.take_along_axis(logits, nbr_idx, axis=-1), _NEG)
            elif self.action_head == 'blank':
                act_latent = jnp.take_along_axis(z_cells, blank[:, None, None], axis=1)[:, 0]
                logits = split_dense(act_latent, actor_p['kernel'], actor_p['bias'], cfg.rms_norm_eps)
            else:
                logits = feat(act_latent) @ actor_p['kernel'] + actor_p['bias']
            assert logits.shape[-1] == self.output_dim_1, (logits.shape, self.output_dim_1)
            value_latent = feat(value_latent)
            value = jnp.squeeze(value_latent @ value_p['kernel'] + value_p['bias'], axis=-1)
            return logits, value

        bot = jnp.broadcast_to(self.bot, (B, H))
        z0_thought = jnp.broadcast_to(z0[None, None], (B, 1, H))
        kv_shape = (cfg.num_layers, B, S, cfg.num_heads, H // cfg.num_heads)
        memory = (jnp.zeros(kv_shape, cfg.compute_dtype), jnp.zeros(kv_shape, cfg.compute_dtype))
        z_init = jnp.broadcast_to(z0[None, None], (B, P + 1, H))
        token_init = jnp.zeros((B, H), cells.dtype)
        logits_init = jnp.zeros((B, self.output_dim_1), cells.dtype)
        value_init = jnp.zeros((B,), cells.dtype)

        def halt_head(h, sig, finite):
            """Logits of (halt, continue) at this pass [B, 2] (row i is read by the decision at pass i - 1)."""
            alpha = self.halt_alpha_min + jax.nn.softplus(self.halt_alpha_raw)
            beta_p = self.variables['params']['halt_beta']
            if self.halt_threshold_fixed:
                beta = alpha * math.log(self.halt_kl)
            elif self.halt_beta_head:
                sg = (lambda v: v) if self.halt_head_grad else jax.lax.stop_gradient
                beta = (sg(rms_norm(h, eps=cfg.rms_norm_eps)).astype(jnp.float32) @ beta_p['kernel'] + beta_p['bias'])[:, 0]
            else:
                beta = beta_p['bias'][0]
            logit = beta - alpha * jnp.log(jnp.maximum(sig, 0.0) + _KL_EPS)
            return jnp.where(finite[:, None], jnp.stack([logit, jnp.zeros_like(logit)], axis=-1), 0.0)

        def body(carry, i):
            # One pass (see the module docstring); never frozen. A nonfinite pass keeps the last finite state (and is flagged).
            z, token, logits_prev, value_prev, nonfinite, (mem_k, mem_v) = carry  # z: cells + SEP latents
            fed_back = rms_norm(token, eps=cfg.rms_norm_eps) if self.thought_norm or self.sem_thought else token
            if self.sem_thought:
                fed_back = simplicial(fed_back, self.sem_groups, self.sem_tau)
            thought = jnp.where(i == 0, bot, fed_back + self.latent_embed)
            z_in = jnp.concatenate([z, z0_thought], axis=1)
            x_in = jnp.concatenate([cells, sep_in, thought[:, None]], axis=1)
            h, thought_kv = mem_core(core_p, z_in, x_in, mem_k, mem_v, jnp.arange(S) < i, grid_shape, P, cfg)
            write = jnp.arange(S) == i
            mem_k = jnp.where(write[None, None, :, None, None], jnp.stack([k for k, _ in thought_kv])[:, :, None], mem_k)
            mem_v = jnp.where(write[None, None, :, None, None], jnp.stack([v for _, v in thought_kv])[:, :, None], mem_v)

            z_cells, out = h[:, :-1], h[:, -1]  # z_cells: cells + SEP
            logits, value = readout(z_cells[:, :P], z_cells[:, P] if sep else out, z_cells[:, P] if value_sep else out)
            finite = (jnp.all(jnp.isfinite(logits), axis=-1) & jnp.isfinite(value)
                      & jnp.all(jnp.isfinite(out), axis=-1) & jnp.all(jnp.isfinite(z_cells), axis=(-2, -1)))
            z_cells = jnp.where(finite[:, None, None], z_cells, z)
            latent_res, _, _ = compute_residuals(z[:, :P], z_cells[:, :P], jnp.ones((B, P), bool), cfg.eps)
            out = jnp.where(finite[:, None], out, token)
            logits = jnp.where(finite[:, None], logits, logits_prev)
            value = jnp.where(finite, value, value_prev)

            z_q, z_p = (logits_prev, logits) if self.halt_kl_reverse else (logits, logits_prev)
            if self.stable_kl:
                kl = stable_softmax.kl(z_q, z_p)
            else:
                lp_q, lp_p = jax.nn.log_softmax(z_q, axis=-1), jax.nn.log_softmax(z_p, axis=-1)
                kl = jnp.sum(jnp.exp(lp_q) * (lp_q - lp_p), axis=-1)
            kl = jax.lax.stop_gradient(kl).astype(jnp.float32)  # pass 1: against uniform (logits_init = 0), unused
            h_prev, h_new = (z[:, P], z_cells[:, P]) if halt_sep else (token, out)  # latents at the halting head
            if self.halt_signal == 'latent':  # L2; pass 1: against the initial latent, unused
                dh = rms_norm(h_new, eps=cfg.rms_norm_eps) - rms_norm(h_prev, eps=cfg.rms_norm_eps)
                sig = jnp.linalg.norm(jax.lax.stop_gradient(dh).astype(jnp.float32), axis=-1)
            elif self.halt_signal == 'tv':  # pass 1: against uniform (logits_init = 0)
                softmax = lambda v: jnp.exp((stable_softmax.log_softmax if self.stable_softmax else jax.nn.log_softmax)(v))
                dp = softmax(logits.astype(jnp.float32)) - softmax(logits_prev.astype(jnp.float32))
                sig = 0.5 * jnp.sum(jnp.abs(jax.lax.stop_gradient(dp)), axis=-1)
            else:
                sig = kl
            logits_k = halt_head(h_new, sig, finite) if self.halting else jnp.zeros((B, 1), jnp.float32)
            nonfinite = nonfinite | ~finite
            carry = (z_cells, out, logits, value, nonfinite, (mem_k, mem_v))
            return carry, (logits, value, logits_k, kl, sig, nonfinite, latent_res, z_cells[:, P] if sep else out)

        if self.remat:
            body = jax.checkpoint(body, prevent_cse=False)
        init = (z_init, token_init, logits_init, value_init, jnp.zeros((B,), bool), memory)
        _, (logits_t, value_t, head_t, kl_t, sig_t, nonfinite_t, latent_res_t, readout_latent_t) = jax.lax.scan(
            body, init, jnp.arange(T))
        # logits_t [T, B, A]; head_t [T, B, 2]; value_t, kl_t, sig_t, nonfinite_t [T, B]

        self.sow('intermediates', 'policy_kl_t', kl_t)
        self.sow('intermediates', 'halt_signal_t', sig_t)
        if self.halting:
            c, sig_c = self._halt(head_t, sig_t, halt_key, halt_steps, B, T, input_shape[:-1])
            self.sow('intermediates', 'policy_log_probs_t',
                     (stable_softmax.log_softmax if self.stable_softmax else jax.nn.log_softmax)(logits_t))
        else:
            c = jnp.full((B,), self.act_iters or T, jnp.int32)
            sig_c = sig_t[c[0] - 1]
        idx = c - 1
        logits = jnp.take_along_axis(logits_t, idx[None, :, None], axis=0)[0]
        nonfinite = jnp.take_along_axis(nonfinite_t.T, idx[:, None], axis=-1)[:, 0]

        logits = logits.reshape(input_shape[:-1] + (-1,))
        value = value_t[-1] if self.value_pass == 'last' else jnp.take_along_axis(value_t.T, idx[:, None], axis=-1)[:, 0]
        value = value.reshape(input_shape[:-1])
        self.sow('intermediates', 'logits', logits)
        self.sow('intermediates', 'value', value)
        self.sow('intermediates', 'value_t', value_t)  # [T, B]: every pass's value
        self.sow('intermediates', 'latent_residual_t', latent_res_t)
        self.sow('intermediates', 'readout_latent_t', readout_latent_t)  # [T, B, H]: the latent pi_n is decoded from

        reason = jnp.where(c >= T, STOP_ITERATION_CAP, STOP_CONVERGED)
        reason = jnp.where(nonfinite, STOP_NONFINITE, reason)
        self.sow('intermediates', 'fp_iterations', c)
        self.sow('intermediates', 'fp_residual', sig_c)
        self.sow('intermediates', 'fp_converged', reason == STOP_CONVERGED)
        self.sow('intermediates', 'fp_stop_reason', reason)

        return (stable_softmax.Categorical if self.stable_softmax else distrax.Categorical)(logits=logits), value

    def _halt(self, head_t, sig_t, halt_key, halt_steps, B, T, batch_shape):
        """Halting: the think time c [B] and the signal that decided it; sows the halting intermediates."""
        # lookahead: row n holds pass n+1's head and signal (pass T never decides)
        head_t = jnp.concatenate([head_t[1:], head_t[-1:]], axis=0)
        sig_t = jnp.concatenate([sig_t[1:], sig_t[-1:]], axis=0)
        self.sow('intermediates', 'halt_log_sig_t', jnp.log(jnp.maximum(sig_t, 0.0) + _KL_EPS))  # ppo.py's natural-grad Fisher
        lp_t = (stable_softmax.log_softmax if self.stable_softmax else jax.nn.log_softmax)(head_t)  # [T, B, 2]
        min_iters = 1 if T == 1 else self.min_think_iters
        lp_c = c_log_probs(lp_t, min_iters)  # [T, B]
        if halt_steps is not None:
            c = jnp.asarray(halt_steps, jnp.int32).reshape(B)
        elif halt_key is not None:
            c = sample_c(lp_t, min_iters, halt_key)
        else:
            c = (jnp.argmax(lp_c, axis=0) + 1).astype(jnp.int32)
        halt_decision_log_probs = decision_log_probs(lp_t, c, min_iters)
        p_c = jnp.exp(lp_c)
        sig_c = jnp.take_along_axis(sig_t.T, (c - 1)[:, None], axis=-1)[:, 0]
        self.sow('intermediates', 'halt_path', decision_mask(c, T, min_iters))
        self.sow('intermediates', 'halt_step_probs', jnp.exp(lp_t))
        self.sow('intermediates', 'halt_log_prob', jnp.sum(halt_decision_log_probs, axis=0).reshape(batch_shape))
        self.sow('intermediates', 'halt_decision_log_probs', halt_decision_log_probs.T.reshape(batch_shape + (T,)))
        self.sow('intermediates', 'halt_entropy', -jnp.sum(p_c * lp_c, axis=0).reshape(batch_shape))
        self.sow('intermediates', 'halt_expected_c', jnp.sum(p_c * jnp.arange(1, T + 1)[:, None], axis=0).reshape(batch_shape))
        self.sow('intermediates', 'halt_prob', jnp.exp(lp_t[..., 0]))
        self.sow('intermediates', 'halt_step_log_probs', lp_t)
        return c, sig_c
