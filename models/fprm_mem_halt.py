"""FPRM with a thought memory (the `fprm_mem` and `fprm_mem_halt` architectures), with or without learned halting.

Trunk. Every pass refines the board exactly as `fprm` does - the cells carry a latent z that goes through one FPRM core
call per pass, with the grid convolution on z and the cell embeddings re-injected - and additionally runs a SEP token
and one *thought* token alongside the cells:

    pass i:  tokens  [z_cells, SEP, t_i]                 (P cells + SEP + 1 thought, attending to each other)
             memory  K/V of t_1 .. t_{i-1}, per layer     (read by every token)
             [z_cells, z_sep, out_i] = Core([z_cells, SEP, t_i]; memory)
             write t_i's per-layer K/V into the memory
    t_1 = BOT (learned),  t_i = latent_embed + out_{i-1}   (rms_norm(out_{i-1}) first if `thought_norm`)

The cells and SEP (a learned input embedding) carry their latents between passes; the thought's latent starts at
latent_init every pass, so its only link to the previous pass is its input. A thought's keys/values are those computed
in the pass that wrote it (not recomputed when the cells change later), Transformer-XL style; the memory has
max_think_iters - 1 slots, preallocated and masked until written. Cost per pass: one core call over P + 2 tokens plus
the memory keys. Each pass is rematerialized in the backward pass (`remat`). No parameter shape depends on m, n or the
iteration cap.

`halting=False` (the `fprm_mem` architecture, arm iso_mem): no halting head; every example runs all max_think_iters
passes and acts with the last one (c = T).

`halting=True` (`fprm_mem_halt`): halting is a chain of decisions. At a decision pass n a head picks k in 0..K
(K = `halt_max_skip`) more passes: k = 0 halts and acts with pi_n; k > 0 runs k passes without deciding and decides again
at n + k; any n + k >= T stops at the cap (c = T). The first decision is at min_think_iters. K = 1 is a yes/no halt at
every pass (h_n = p_n(0)); K = 4 can skip up to 4 passes at once.

What a decision at pass n reads. Let h_n be the latent at the readout head (the readout token's output at pass n, from
which pi_n is decoded). `halt_signal` picks the pair (z_n, z_{n+1}) a convergence test compares and its distance S:
  'kl':     z_n = pi_n,  S_{n+1} = KL(pi_{n+1} || pi_n)
  'latent': z_n = h_n,   S_{n+1} = ||rms_norm(h_{n+1}) - rms_norm(h_n)||   (L2 of what the readout heads read)
`halt_head` picks the head, each a one-hidden-layer GELU MLP (width `halt_head_width`) with K + 1 outputs and a
zero-initialized output layer, so k is uniform at init:
  'none':     p_n = softmax(MLP(rms_norm(h_n)))                      no signal, no lookahead (ACT / PonderNet style)
  'distance': p_n = softmax(MLP(log S_{n+1}))
  'raw':      p_n = softmax(MLP([z_n, z_{n+1}]))                     z = log pi (log-softmax) or rms_norm(h)
'distance' and 'raw' look one pass ahead: pass n+1 is computed, and if the head halts, pass n acts (c = n means "pass
n+1 agrees with pass n"), so c = 1 is decided by a real convergence test on pass 2 (use min_think_iters = 1). S is
stop-gradient'ed; the h and pi inputs are too unless `halt_head_grad`.

Each pass runs [cells, SEP, thought]. SEP is a learned input embedding whose latent is carried between passes like the
cells' (attending to the cells, the current thought and the thought memory); the thought's per-layer K/V is what the
memory keeps. `readout_token` picks where every readout (logits, value, the halting head's input) comes from: the
thought's output ('thought') or SEP's latent ('sep'; the thoughts then only feed the memory and the next thought, so
they need not be decodable as a policy themselves).

The PPO ratio scores the sampled path, log p(path) = sum of the log-probabilities of its decisions (all k that reach
the cap count as one outcome), so the rollout stores the decision passes (`halt_path`, a bitmask) along with c. For
K = 1 the path is determined by c and log p(path) = log p(c | s). Evaluation takes the most likely path (a max-product
recursion over passes; for K = 1 the most likely c), and 'halt_entropy' is the exact entropy of the path (= H(c) for
K = 1), from the probability r_n of reaching a decision at pass n: H = sum_n r_n H(p_n).

`halt_kl_reverse`: KL_i = KL(pi_{i-1} || pi_i) instead, the expectation under pi_{i-1}, i.e. under pi_c, the policy that
acts; the two directions agree to second order for small policy changes.

`stable_kl` / `stable_softmax`: the fixes of models/stable_softmax.py (Barron et al., arXiv 2610.31642). stable_kl
computes KL_i from the logit difference, exact down to ~1e-6 relative (the standard sum q (log q - log p) has a ~1e-7
floor in fp32 and is non-zero at equal policies). stable_softmax gives the policy and the halting head collapse-free
log-probabilities (no zeroed gradient on an action whose margin exceeds ~16.6).

The think time c (with its decision path) is an action of the joint policy: log p(c, a | s) = log p(path | s) +
log pi_c(a | s). c = T is a forced halt: pi_T acts with no pass after it to check it against. All T passes are always
executed; the readout of pass c is selected afterwards, so passes after c get no gradient from the actor loss.

Choice of c in `__call__`: `halt_steps` and `halt_path` given (the PPO loss replays the rollout's path) > `halt_key`
given (rollout: a sampled path) > neither (evaluation: the most likely path).

The value is read out from the last pass: since c is an action, a baseline that depended on it would bias the
halting gradient.

Sown intermediates: 'logits'/'value' (at the chosen c), 'fp_iterations' (= c), 'fp_residual' (S_{c+1}, the signal that
decided the halt; KL_T without halting), 'fp_converged' / 'fp_stop_reason', 'halt_signal_t' (S_i by pass, [T, B],
unshifted), with halting also 'halt_log_prob'
(log p(path | s)), 'halt_entropy', 'halt_path', 'halt_prob' (p_n(0), [T, B]) and 'halt_step_probs' (p_n(k),
[T, B, K + 1]), and always 'policy_kl_t' (KL_i by pass, [T, B], unshifted; pass 1 is
against uniform) and 'latent_residual_t' (the max-token residual of the cell latents between consecutive passes, as
fprm's halt_criterion='latent_residual'; [T, B], pass 1 is against the initial latent).
"""

import math

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp

from models.fprm import (
    FPRMConfig,
    FPRMCore,
    STOP_CONVERGED,
    STOP_ITERATION_CAP,
    STOP_NONFINITE,
    compute_residuals,
    make_latent_init,
    rms_norm,
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
_KL_EPS = 1e-10  # floor inside log KL (KL may round to 0 or slightly below)


def step_path_log_prob(lp, mask, c):
    """log p(path) from per-pass log-probabilities lp [T, B, K + 1] of k, the decision
    passes mask [B] (bit n-1 = a decision at pass n) and the think time c [B]."""
    T, B, K = lp.shape
    n = jnp.arange(1, T + 1)[:, None]  # [T, 1]
    dec = ((mask[None] >> (n - 1)) & 1).astype(bool)  # [T, B]
    first_from = jax.lax.cummin(jnp.where(dec, n, T + 1)[::-1], axis=0)[::-1]  # first decision pass >= n
    nxt = jnp.concatenate([first_from[1:], jnp.full((1, B), T + 1)], axis=0)  # first decision pass > n
    lp_next = jnp.take_along_axis(lp, jnp.clip(nxt - n, 0, K - 1)[..., None], axis=-1)[..., 0]
    reaches_cap = jnp.arange(K) >= jnp.maximum(1, T - n)[..., None]  # [T, 1, K]: the k with n + k >= T
    lp_cap = jax.scipy.special.logsumexp(jnp.where(reaches_cap, lp, _NEG), axis=-1)
    out = jnp.where((c[None] == n) & (n < T), lp[..., 0], jnp.where(nxt <= T, lp_next, lp_cap))
    return jnp.sum(jnp.where(dec, out, 0.0), axis=0)


def sample_step_path(lp, min_iters, key=None):
    """Sample a path (k at each decision); returns (mask, c), [B]."""
    T, B, _ = lp.shape
    keys = jax.random.split(key, T)

    def body(carry, inp):
        nxt, done, mask, c = carry
        n, lp_n, key_n = inp
        k = jax.random.categorical(key_n, lp_n, axis=-1).astype(jnp.int32)
        here = ~done & (nxt == n) & (n < T)
        halt, cap = here & (k == 0), here & (k > 0) & (n + k >= T)
        mask = jnp.where(here, mask | (1 << (n - 1)), mask)
        c = jnp.where(halt, n, jnp.where(cap, T, c))
        return (jnp.where(here & (k > 0), n + k, nxt), done | halt | cap, mask, c), None

    init = (jnp.full((B,), min_iters, jnp.int32), jnp.zeros((B,), bool), jnp.zeros((B,), jnp.int32), jnp.full((B,), T, jnp.int32))
    (_, _, mask, c), _ = jax.lax.scan(body, init, (jnp.arange(1, T + 1), lp, keys))
    return mask, c


def _decisions(lp, n):
    """Outcomes of a decision at pass n from lp [T, B, K + 1]: [(next pass or None for the cap or a halt, c, log p)]."""
    T, _, K1 = lp.shape
    out = [(None, n, lp[n - 1, :, 0])] + [(n + k, None, lp[n - 1, :, k]) for k in range(1, K1) if n + k < T]
    cap = [k for k in range(1, K1) if n + k >= T]
    return out + ([(None, T, jax.scipy.special.logsumexp(lp[n - 1, :, cap[0]:], axis=-1))] if cap else [])


def best_step_path(lp, min_iters):
    """The most likely path (max-product over passes); returns (mask, c), [B]."""
    T, B, _ = lp.shape
    best = {min_iters: (jnp.zeros((B,)), jnp.zeros((B,), jnp.int32))}  # pass -> (log p of the best path to it, its mask)
    ends = [(jnp.zeros((B,)), jnp.zeros((B,), jnp.int32), jnp.full((B,), T, jnp.int32))] if min_iters >= T else []
    for n in range(min_iters, T):
        if n not in best:
            continue
        score, mask = best.pop(n)
        mask = mask | (1 << (n - 1))
        for nxt, c, lp_k in _decisions(lp, n):
            if nxt is None:
                ends.append((score + lp_k, mask, jnp.full((B,), c, jnp.int32)))
            else:
                old = best.get(nxt, (jnp.full((B,), -jnp.inf), mask))
                better = score + lp_k > old[0]
                best[nxt] = (jnp.where(better, score + lp_k, old[0]), jnp.where(better, mask, old[1]))
    i = jnp.argmax(jnp.stack([e[0] for e in ends]), axis=0)
    pick = lambda j: jnp.take_along_axis(jnp.stack([e[j] for e in ends]), i[None], axis=0)[0]
    return pick(1), pick(2)


def step_path_entropy(lp, min_iters):
    """Entropy of the path distribution, sum_n r_n H(p_n) with r_n the probability of a decision at pass n."""
    T, B, _ = lp.shape
    reach, ent = {min_iters: jnp.ones((B,))}, jnp.zeros((B,))
    for n in range(min_iters, T):
        if n not in reach:
            continue
        r = reach.pop(n)
        for nxt, _, lp_k in _decisions(lp, n):
            ent = ent - r * jnp.exp(lp_k) * lp_k
            if nxt is not None:
                reach[nxt] = reach.get(nxt, 0.0) + r * jnp.exp(lp_k)
    return ent


class FPRMMemHaltActorValue(nn.Module):
    """Looped board + SEP + thought memory, with (halting=True) or without learned halting (see the module docstring)."""

    m: int = 3
    n: int = 3
    config: FPRMConfig = FPRMConfig()
    max_think_iters: int = 8
    min_think_iters: int = 2
    halting: bool = True           # False: no halting head, always act with pass max_think_iters (iso_mem)
    halt_max_skip: int = 1         # K: each decision picks k in 0..K more passes (1: halt yes/no at every pass)
    halt_head_grad: bool = False   # let the halting loss backpropagate into the trunk through the head's h / pi inputs
    halt_kl_reverse: bool = False  # KL_i = KL(pi_{i-1} || pi_i) (see the module docstring)
    stable_kl: bool = False        # KL_i without the fp32 rounding floor (see the module docstring)
    stable_softmax: bool = False   # collapse-free log-softmax for the policy and the steps head (see the module docstring)
    halt_signal: str = 'kl'        # 'kl' | 'latent': what 'distance' / 'raw' compare (see the module docstring)
    halt_head: str = 'distance'    # 'none' | 'distance' | 'raw' (see the module docstring)
    halt_head_width: int = 32      # hidden width of the halting MLP
    readout_token: str = 'thought'  # readouts from the newest thought ('thought') or the carried SEP token ('sep')
    output_dim_1: int = 9   # action logits (= m * n buttons)
    output_dim_2: int = 1   # value
    obs_channels: int = 0          # see FPRMThinkerActorValue
    action_head: str = 'per_cell'  # see FPRMThinkerActorValue
    remat: bool = True             # rematerialize each pass in the backward pass
    thought_norm: bool = False     # rms_norm each thought output before adding the modal embedding
    tokenizer: str = 'cell'        # see FPRMThinkerActorValue
    tokenizer_conv_features: int = 64

    def setup(self):
        assert self.action_head in ('per_cell', 'readout'), self.action_head
        assert self.max_think_iters >= 1, self.max_think_iters
        assert not self.halting or self.halt_max_skip >= 1
        assert self.halt_signal in ('kl', 'latent'), self.halt_signal
        assert self.readout_token in ('thought', 'sep'), self.readout_token
        assert self.halt_head in ('none', 'distance', 'raw'), self.halt_head
        if self.halting:
            self.halt_hidden = nn.Dense(  # 'distance': a scalar input, so spread the GELU kinks over its range
                self.halt_head_width, bias_init=nn.initializers.normal(stddev=1.0 if self.halt_head == 'distance' else 0.0))
            self.halt_out = nn.Dense(self.halt_max_skip + 1, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros)
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
        self.actor_head = nn.Dense(
            self.output_dim_1 // (self.m * self.n) if self.action_head == 'per_cell' else self.output_dim_1,
            bias_init=nn.initializers.zeros,
        )
        self.value_head = nn.Dense(self.output_dim_2, bias_init=nn.initializers.zeros)
        self.latent_init = make_latent_init(self.config)

    def _halt_mlp(self, x):
        """The halting head's logits of k [..., K + 1] from its input x [..., in]."""
        p_in, p_out = self.variables['params']['halt_hidden'], self.variables['params']['halt_out']
        return jax.nn.gelu(x @ p_in['kernel'] + p_in['bias']) @ p_out['kernel'] + p_out['bias']

    def __call__(self, x, normalizer_params=None, halt_key=None, halt_steps=None, halt_path=None):
        if normalizer_params is not None:
            x = (x - normalizer_params.mean) / (normalizer_params.std)

        cfg = core_config(self.config, self.tokenizer)
        input_shape = x.shape
        x_flat = x.reshape(-1, input_shape[-1])
        B = x_flat.shape[0]
        H = cfg.d_model
        T = self.max_think_iters
        S = max(T - 1, 1)
        per_cell = self.action_head == 'per_cell'

        if self.tokenizer == 'cell':
            cells = self.cell_projection(tokenize(x_flat, self.m, self.n, self.obs_channels))
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
                n_in = {'none': H, 'distance': 1, 'raw': 2 * (self.output_dim_1 if self.halt_signal == 'kl' else H)}[self.halt_head]
                self.halt_out(self.halt_hidden(jnp.zeros((1, n_in))))
        params = self.variables['params']
        core_p, actor_p, value_p = params['core'], params['actor_head'], params['value_head']

        sep = self.readout_token == 'sep'
        sep_in = jnp.broadcast_to(self.sep, (B, 1, H))

        def readout(z_cells, out):
            last_latent = rms_norm(out, eps=cfg.rms_norm_eps)
            if per_cell:
                cell_latents = rms_norm(z_cells, eps=cfg.rms_norm_eps)
                logits = (cell_latents @ actor_p['kernel'] + actor_p['bias']).reshape(B, -1)
            else:
                logits = last_latent @ actor_p['kernel'] + actor_p['bias']
            assert logits.shape[-1] == self.output_dim_1, (logits.shape, self.output_dim_1)
            value = jnp.squeeze(last_latent @ value_p['kernel'] + value_p['bias'], axis=-1)
            return logits, value

        bot = jnp.broadcast_to(self.bot, (B, H))
        z0_thought = jnp.broadcast_to(z0[None, None], (B, 1, H))
        kv_shape = (cfg.num_layers, B, S, cfg.num_heads, H // cfg.num_heads)
        memory = (jnp.zeros(kv_shape, cfg.compute_dtype), jnp.zeros(kv_shape, cfg.compute_dtype))
        z_init = jnp.broadcast_to(z0[None, None], (B, P + 1, H))
        token_init = jnp.zeros((B, H), cells.dtype)
        logits_init = jnp.zeros((B, self.output_dim_1), cells.dtype)
        value_init = jnp.zeros((B,), cells.dtype)

        def halt_head(h_prev, h, logits_prev, logits, sig, finite):
            """Logits of k at this pass [B, K + 1] (row i; 'distance' / 'raw' are read by the decision at pass i - 1)."""
            sg = (lambda v: v) if self.halt_head_grad else jax.lax.stop_gradient
            if self.halt_head == 'none':
                x = sg(rms_norm(h, eps=cfg.rms_norm_eps))
            elif self.halt_head == 'distance':
                x = jnp.log(jnp.maximum(sig, 0.0) + _KL_EPS)[:, None]
            elif self.halt_signal == 'kl':
                log_softmax = stable_softmax.log_softmax if self.stable_softmax else jax.nn.log_softmax
                x = sg(jnp.concatenate([log_softmax(logits_prev), log_softmax(logits)], -1))
            else:
                x = sg(jnp.concatenate([rms_norm(h_prev, eps=cfg.rms_norm_eps), rms_norm(h, eps=cfg.rms_norm_eps)], -1))
            return jnp.where(finite[:, None], self._halt_mlp(x.astype(jnp.float32)), 0.0)

        def body(carry, i):
            # One pass (see the module docstring); never frozen. A nonfinite pass keeps the last finite state (and is flagged).
            z, token, logits_prev, value_prev, nonfinite, (mem_k, mem_v) = carry  # z: cells + SEP latents
            fed_back = rms_norm(token, eps=cfg.rms_norm_eps) if self.thought_norm else token
            thought = jnp.where(i == 0, bot, fed_back + self.latent_embed)
            z_in = jnp.concatenate([z, z0_thought], axis=1)
            x_in = jnp.concatenate([cells, sep_in, thought[:, None]], axis=1)
            h, thought_kv = mem_core(core_p, z_in, x_in, mem_k, mem_v, jnp.arange(S) < i, grid_shape, P, cfg)
            write = jnp.arange(S) == i
            mem_k = jnp.where(write[None, None, :, None, None], jnp.stack([k for k, _ in thought_kv])[:, :, None], mem_k)
            mem_v = jnp.where(write[None, None, :, None, None], jnp.stack([v for _, v in thought_kv])[:, :, None], mem_v)

            z_cells, out = h[:, :-1], h[:, -1]  # z_cells: cells + SEP
            logits, value = readout(z_cells[:, :P], z_cells[:, P] if sep else out)
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
            h_prev, h_new = (z[:, P], z_cells[:, P]) if sep else (token, out)  # latents at the readout head
            if self.halt_signal == 'latent':  # L2; pass 1: against the initial latent, unused
                dh = rms_norm(h_new, eps=cfg.rms_norm_eps) - rms_norm(h_prev, eps=cfg.rms_norm_eps)
                sig = jnp.linalg.norm(jax.lax.stop_gradient(dh).astype(jnp.float32), axis=-1)
            else:
                sig = kl
            logits_k = (halt_head(h_prev, h_new, logits_prev, logits, sig, finite) if self.halting
                        else jnp.zeros((B, 1), jnp.float32))
            nonfinite = nonfinite | ~finite
            carry = (z_cells, out, logits, value, nonfinite, (mem_k, mem_v))
            return carry, (logits, value, logits_k, kl, sig, nonfinite, latent_res)

        if self.remat:
            body = jax.checkpoint(body, prevent_cse=False)
        init = (z_init, token_init, logits_init, value_init, jnp.zeros((B,), bool), memory)
        _, (logits_t, value_t, head_t, kl_t, sig_t, nonfinite_t, latent_res_t) = jax.lax.scan(body, init, jnp.arange(T))
        # logits_t [T, B, A]; head_t [T, B, K + 1]; value_t, kl_t, sig_t, nonfinite_t [T, B]

        self.sow('intermediates', 'policy_kl_t', kl_t)
        self.sow('intermediates', 'halt_signal_t', sig_t)
        if self.halting:
            c, sig_c = self._halt(head_t, sig_t, halt_key, halt_steps, halt_path, B, T, input_shape[:-1])
        else:
            c, sig_c = jnp.full((B,), T, jnp.int32), sig_t[-1]
        idx = c - 1
        logits = jnp.take_along_axis(logits_t, idx[None, :, None], axis=0)[0]
        nonfinite = jnp.take_along_axis(nonfinite_t.T, idx[:, None], axis=-1)[:, 0]

        logits = logits.reshape(input_shape[:-1] + (-1,))
        value = value_t[-1].reshape(input_shape[:-1])
        self.sow('intermediates', 'logits', logits)
        self.sow('intermediates', 'value', value)
        self.sow('intermediates', 'latent_residual_t', latent_res_t)

        reason = jnp.where(c >= T, STOP_ITERATION_CAP, STOP_CONVERGED)
        reason = jnp.where(nonfinite, STOP_NONFINITE, reason)
        self.sow('intermediates', 'fp_iterations', c)
        self.sow('intermediates', 'fp_residual', sig_c)
        self.sow('intermediates', 'fp_converged', reason == STOP_CONVERGED)
        self.sow('intermediates', 'fp_stop_reason', reason)

        return (stable_softmax.Categorical if self.stable_softmax else distrax.Categorical)(logits=logits), value

    def _halt(self, head_t, sig_t, halt_key, halt_steps, halt_path, B, T, batch_shape):
        """Halting: the think time c [B] and the signal that decided it; sows the halting intermediates."""
        if self.halt_head != 'none':  # lookahead: row n holds pass n+1's head and signal (pass T never decides)
            head_t = jnp.concatenate([head_t[1:], head_t[-1:]], axis=0)
            sig_t = jnp.concatenate([sig_t[1:], sig_t[-1:]], axis=0)
        lp_t = (stable_softmax.log_softmax if self.stable_softmax else jax.nn.log_softmax)(head_t)  # [T, B, K + 1]
        min_iters = 1 if T == 1 else self.min_think_iters
        if halt_steps is not None:
            c, mask = jnp.asarray(halt_steps, jnp.int32).reshape(B), jnp.asarray(halt_path, jnp.int32).reshape(B)
        elif halt_key is not None:
            mask, c = sample_step_path(lp_t, min_iters, halt_key)
        else:
            mask, c = best_step_path(lp_t, min_iters)
        halt_log_prob = step_path_log_prob(lp_t, mask, c)
        halt_entropy = step_path_entropy(lp_t, min_iters)
        h_t = jnp.exp(lp_t[..., 0])  # p(k = 0) at every pass
        sig_c = jnp.take_along_axis(sig_t.T, (c - 1)[:, None], axis=-1)[:, 0]
        self.sow('intermediates', 'halt_path', mask)
        self.sow('intermediates', 'halt_step_probs', jnp.exp(lp_t))
        self.sow('intermediates', 'halt_log_prob', halt_log_prob.reshape(batch_shape))
        self.sow('intermediates', 'halt_entropy', halt_entropy.reshape(batch_shape))
        self.sow('intermediates', 'halt_prob', h_t)
        return c, sig_c
