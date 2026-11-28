"""FPRM with a thought memory (the `fprm_mem` architecture): a looped actor with a uniform or variable think budget,
and a separate fixed-budget FPRM critic.

Trunk. Every pass refines the board as the plain FPRM does - the cells carry a latent z that goes through one FPRM core
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

Critic. The value comes from `critic`, a plain FPRM model (models/fprm_thinker.py, own parameters) run for a fixed
`value_iters` core calls, whatever the actor's budget: a baseline that depended on c would bias the halting gradient.

Think budget (ppo.py --act_iters=C:u|v sets max_think_iters = T = C and `halting`):
  `halting=False` (C:u): every example runs T passes and acts with the last one (c = T).
  `halting=True` (C:v): a decision at every pass n from min_think_iters to T - 1; no halt before pass T stops at the
  cap (c = T). `halt_criterion` picks the rule:
    'kl':      halt at n once S_{n+1} = KL(pi_{n+1} || pi_n) < halt_threshold     (deterministic, one pass of lookahead)
    'l2':      halt at n once S_{n+1} = ||rms_norm(h_{n+1}) - rms_norm(h_n)|| < halt_threshold, h_n the readout latent
    'sigmoid': a learned yes/no halt with probability h_n = sigmoid(w . rms_norm(h_n) + b), w and b zero at init
               (h_n = 1/2), no lookahead; h_n is stop-gradient'ed unless `halt_head_grad`.
  In every case pass c acts. S_1 is against uniform ('kl') or the initial latent ('l2'); with 'sigmoid', S is the KL,
  logged as a diagnostic ('halt_signal_t', 'halt_log_sig_t', 'fp_residual').

Policy parameterization (`policy_param`): 'last' decodes pi_n from pass n's logits l_n; 'sum' (PonderNet style)
decodes it from the accumulated logits sum_{k<=n} l_k. `logit_sum_stop_grad` stops the gradient through l_k, k < n,
so only the acting pass's l_c is trained. The KL criterion compares these acting policies. 'mix' ('sigmoid' halting
only) acts with the mixture pi(a | s, c) = sum_{n<=c} w_n pi_n(a | s) of the 'last' policies, w_n = p(halt at n) for
n < c and w_c = P(no halt before c): the halt distribution cut at the sampled c, never looking past it.
`mix_weight_stop_grad` stops the actor loss's gradient into the halting head through w (log p(c) still trains it).

Each pass runs [cells, SEP, thought]. SEP is a learned input embedding whose latent is carried between passes like the
cells' (attending to the cells, the current thought and the thought memory); the thought's per-layer K/V is what the
memory keeps. `readout_token` picks where the actor's readouts (the logits of the 'readout' / 'action_tokens' heads, the
halting head's input, the 'l2' latent) come from: the thought's output ('thought') or SEP's latent ('sep'; the thoughts
then only feed the memory and the next thought, so they need not be decodable as a policy themselves).

'sigmoid': the PPO ratio scores log p(c | s) = sum_{n < c} log(1 - h_n) + log h_c (no h_T term: c = T is the cap);
`halt_sg_survival` keeps its value but passes no gradient through the continue terms log(1 - h_n), n < c. The rollout
also stores the decision passes (`halt_path`, a `[B, T]` bool mask: col n-1 = a decision at pass n), a function of c. Evaluation
takes the most likely c, and 'halt_entropy' is H(c).

`halt_kl_reverse`: KL_i = KL(pi_{i-1} || pi_i) instead, the expectation under pi_{i-1}, i.e. under pi_c, the policy that
acts; the two directions agree to second order for small policy changes.

`stable_kl` / `stable_softmax`: the fixes of models/stable_softmax.py (Barron et al., arXiv 2610.31642). stable_kl
computes KL_i from the logit difference, exact down to ~1e-6 relative (the standard sum q (log q - log p) has a ~1e-7
floor in fp32 and is non-zero at equal policies). stable_softmax gives the policy and the halting head collapse-free
log-probabilities (no zeroed gradient on an action whose margin exceeds ~16.6).

With 'sigmoid' the think time c is an action of the joint policy: log p(c, a | s) = log p(c | s) +
log pi_c(a | s). c = T is a forced halt: pi_T acts with no pass after it to check it against. All T passes are always
executed; the readout of pass c is selected afterwards, so passes after c get no gradient from the actor loss.

Choice of c in `__call__` (halting=True): `halt_steps` given (the PPO loss replays the rollout's c, for every criterion)
> 'kl' / 'l2' (the rule above) > `halt_key` given ('sigmoid' rollout: a sampled c) > neither ('sigmoid' evaluation: the
most likely c).

Action heads (`action_head`): 'per_cell' = one logit per cell latent (Lights Out presses), 'readout' = Dense(A) on the
readout token (moves), 'neighbour' = 4 moves of the blank (sliding puzzle; observation channel 0 marks the blank) whose
logits are the per-cell logits of the blank's up / right / down / left neighbours (the tile that would slide into it,
the order of envs/slidingpuzzle_env.MOVES), off-board moves masked to probability 0. 'neighbour' reads the policy from
the cell latents like 'per_cell' while keeping the 4-move action space. 'blank' = 4 Dense(1) on the blank's own cell
latent, move i's reading only the i-th quarter of its H features, rms-normed on its own (the same 4 moves, unmasked: an off-board move is a
no-op, as with 'readout'). 'action_tokens' = A learned action tokens after SEP (each pass runs [cells, SEP, a_1..a_A,
thought]; their latents are carried between passes like SEP's), logit a = Dense(1) on action token a's latent. `cell_geometry=False` drops the 6
geometry features (row, col, top, bottom, left, right) from the cell tokens, leaving the observation planes.

Sown intermediates: 'logits' (at the chosen c) / 'value' (the critic's), 'fp_iterations' (= c), 'fp_residual' (S_{c+1},
whether pass c+1 agrees with pass c; S_T at the cap), 'fp_converged' / 'fp_stop_reason', 'halt_signal_t' (S_i by pass,
[T, B], unshifted), with halting 'policy_log_probs_t', with 'sigmoid' also 'halt_log_prob'
(log p(c | s)), 'halt_decision_log_probs' (its terms by decision pass, [B, T], 0 off the path), 'halt_entropy', 'halt_expected_c' (E[c | s]), 'halt_path', 'halt_prob' (h_n, [T, B]), 'halt_step_probs' ((h_n, 1 - h_n),
[T, B, 2]), 'halt_step_log_probs' (their logs, [T, B, 2]), 'halt_log_sig_t' (log S_{n+1} at row n, [T, B]), and always 'policy_kl_t' (KL_i by pass, [T, B], unshifted; pass 1 is
against uniform) and 'latent_residual_t' (the max-token residual of the cell latents between consecutive passes, as
the plain FPRM's halt_criterion='latent_residual'; [T, B], pass 1 is against the initial latent) and 'readout_latent_t' (the latent pi_n is decoded from, by pass, [T, B, H]).
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
    exclusive_self_attention,
    fused_attention,
    make_latent_init,
    rms_norm,
    mlp_activation,
)
from models import stable_softmax
from models.fprm_thinker import FPRMThinkerActorValue, tokenize
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


def _qkv(params, x, cfg: FPRMConfig):
    """[..., H] -> q, k, v of shape [..., num_heads, head_dim], in cfg.compute_dtype."""
    H = x.shape[-1]
    x = x.astype(cfg.compute_dtype)
    qkv = x @ params['qkv_proj']['kernel'].astype(x.dtype)
    qkv = qkv.reshape(x.shape[:-1] + (3 * cfg.num_heads, H // cfg.num_heads))
    q, k, v = jnp.split(qkv, 3, axis=-2)
    return q, k, v


def _mlp(params, x, cfg: FPRMConfig):
    """The block GELU MLP (FPRMMLP) as a pure function."""
    y = mlp_activation(x @ params['up_proj']['kernel'].astype(x.dtype), cfg)
    return y @ params['down_proj']['kernel'].astype(x.dtype)


def rope_2d(x, grid_shape, num_cells, base=100.0):
    """2-D rotary position embedding on x [B, N, nh, hd]: token i < num_cells (cell (r, c) = divmod(i, C)) is rotated
    by position (r + 1, c + 1), the first half of the rotated pairs by the row and the second half by the column; the
    other tokens (SEP, action tokens, thought) sit at (0, 0), i.e. are not rotated."""
    N, hd = x.shape[1], x.shape[-1]
    assert hd % 4 == 0, hd
    freqs = base ** (-jnp.arange(hd // 4) / (hd // 4))
    r, c = jnp.divmod(jnp.arange(num_cells), grid_shape[1])
    ang = jnp.concatenate([(r + 1)[:, None] * freqs, (c + 1)[:, None] * freqs], axis=-1)  # [P, hd / 2]
    ang = jnp.pad(ang, ((0, N - num_cells), (0, 0)))[:, None]  # [N, 1, hd / 2]
    cos, sin = jnp.cos(ang), jnp.sin(ang)
    x1, x2 = jnp.split(x.astype(jnp.float32), 2, axis=-1)
    return jnp.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1).astype(x.dtype)


def mem_core(params, z, x, mem_k, mem_v, mem_valid, grid_shape, num_cells, cfg: FPRMConfig, rope=False):
    """One FPRM core call over the current tokens [cells, SEP, thought] that also attend to the thought memory.

    z: [B, N, H] latents (cells, SEP: carried latent; thought: latent_init); x: [B, N, H] inputs (cell embeddings, SEP,
    thought input). mem_k / mem_v: [L, B, S, nh, hd]; mem_valid: [S] bool (written slots). The grid convolution acts
    on the cells' latent only; `rope`: 2-D RoPE (rope_2d) on the current tokens' q and k (the thought, at (0, 0), is not
    rotated, so the memory keys are unchanged). Returns (h [B, N, H], per-layer [(k, v)] of the last (thought) token, each [B, nh, hd]).
    """
    B, N, H = x.shape
    a1, b1, a2, b2 = _residual_scales(params, cfg)
    h, a1, b1 = _mixed_precision_input(a2 * z + b2 * x, a1, b1, cfg)
    scale = 1.0 / (math.sqrt(H // cfg.num_heads) * cfg.softmax_temp)
    thought_kv = []
    for i in range(cfg.num_layers):
        p = params[f'block_{i}']
        q, k, v = _qkv(p['self_attn'], rms_norm(h, cfg.rms_norm_eps), cfg)  # [B, N, nh, hd]
        if rope:
            q, k = rope_2d(q, grid_shape, num_cells), rope_2d(k, grid_shape, num_cells)
        # keys [current tokens, memory]: the written memory slots are a prefix, so the valid keys are the first N + #written
        out = fused_attention(q, jnp.concatenate([k, mem_k[i]], axis=1), jnp.concatenate([v, mem_v[i]], axis=1),
                              scale, N + jnp.sum(mem_valid))
        if cfg.xsa:  # the projection uses each current token's own value, not the memory's
            out = exclusive_self_attention(out, v)
        out = out.reshape(B, N, H) @ p['self_attn']['o_proj']['kernel'].astype(h.dtype)
        h = a1 * h + b1 * out
        h = a1 * h + b1 * _mlp(p['mlp'], rms_norm(h, cfg.rms_norm_eps), cfg)
        thought_kv.append((k[:, -1], v[:, -1]))
    return h.astype(x.dtype), thought_kv


_NEG = -1e9  # log of a forced-zero probability (finite: keeps p log p and the gradients NaN-free)
_KL_EPS = 1e-10  # floor inside log KL (KL may round to 0 or slightly below)


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


def decision_log_probs(lp, c, min_iters, sg_survival=False):
    """The terms of log p(c | s) by decision pass, [T, B]: log(1 - h_n) for n < c, log h_c at n = c < T, 0 elsewhere.
    sg_survival: the continue terms log(1 - h_n), n < c, carry no gradient (same values)."""
    n, dec = _decision_passes(lp.shape[0], min_iters)
    cont = jax.lax.stop_gradient(lp[..., 1]) if sg_survival else lp[..., 1]
    return jnp.where(dec & (n < c), cont, jnp.where(dec & (n == c), lp[..., 0], 0.0))


def mix_log_weights(lp, c, min_iters):
    """log w_n [T, B] of policy_param='mix': log p(halt at n) for n < c, log P(no halt before c) at n = c, _NEG after c."""
    T = lp.shape[0]
    n, dec = _decision_passes(T, min_iters)
    lp_cont = jnp.where(dec, lp[..., 1], 0.0)
    before = jnp.cumsum(jnp.concatenate([jnp.zeros_like(lp_cont[:1]), lp_cont[:-1]]), axis=0)  # log P(no halt before n)
    return jnp.where(n < c, c_log_probs(lp, min_iters), jnp.where(n == c, before, _NEG))


def decision_mask(c, T, min_iters):
    """The decision passes of the path ending at c as a bool mask [B, T] (col n - 1 = a decision at pass n)."""
    n, dec = _decision_passes(T, min_iters)
    return (dec & (n <= c)).T


def sample_c(lp, min_iters, key):
    """Sample c [B]: the first decision pass that halts, else T."""
    n, dec = _decision_passes(lp.shape[0], min_iters)
    halt = jax.random.categorical(key, lp, axis=-1) == 0  # [T, B]
    return jnp.min(jnp.where(dec & halt, n, lp.shape[0]), axis=0).astype(jnp.int32)


class FPRMMemHaltActorValue(nn.Module):
    """Looped board + SEP + thought memory actor, with a uniform or variable think budget, and a separate plain-FPRM
    critic (see the module docstring)."""

    m: int = 3
    n: int = 3
    config: FPRMConfig = FPRMConfig()
    max_think_iters: int = 8       # C of act_iters=C:u|v: the think cap T
    min_think_iters: int = 2
    halting: bool = True           # act_iters=C:v (variable budget); False = C:u (always act with pass T)
    halt_criterion: str = 'sigmoid'  # halting=True: 'kl' | 'l2' (deterministic, lookahead) | 'sigmoid' (learned, stochastic)
    halt_threshold: float = 1e-3   # 'kl' / 'l2': halt at pass n once d(n, n+1) < this
    halt_head_grad: bool = False   # 'sigmoid': let the halting loss backpropagate into the trunk through the head's h_n input
    halt_sg_survival: bool = False  # 'sigmoid': halt_log_prob's gradient skips the continue terms log(1 - h_c'), c' < c
    halt_kl_reverse: bool = False  # KL_i = KL(pi_{i-1} || pi_i) (see the module docstring)
    stable_kl: bool = False        # KL_i without the fp32 rounding floor (see the module docstring)
    stable_softmax: bool = False   # collapse-free log-softmax for the policy and the steps head (see the module docstring)
    policy_param: str = 'last'     # 'last': pi_n from pass n's logits l_n | 'sum': pi_n from sum_(k<=n) l_k (PonderNet style)
    logit_sum_stop_grad: bool = False  # policy_param='sum': no gradient through l_k, k < n (only l_n of the acting pass)
    mix_weight_stop_grad: bool = False  # policy_param='mix': no gradient into the halting head through the weights w_n
    value_iters: int = 8           # the critic's fixed number of FPRM core calls
    readout_token: str = 'thought'  # readouts from the newest thought ('thought') or the carried SEP token ('sep')
    output_dim_1: int = 9   # action logits (= m * n buttons)
    output_dim_2: int = 1   # value
    obs_channels: int = 0          # see FPRMThinkerActorValue
    action_head: str = 'readout'   # 'readout' | 'action_tokens'
    cell_pos_embed: bool = False   # tokenizer='cell': add a learned embedding per cell index to the cell tokens (actor only)
    cell_rope: bool = False        # 2-D RoPE on the cells' attention queries / keys by (row, col) (actor only; see rope_2d)
    remat: bool = True             # rematerialize each pass in the backward pass
    thought_norm: bool = False     # rms_norm each thought output before adding the modal embedding
    tokenizer: str = 'cell'        # see FPRMThinkerActorValue
    tokenizer_conv_features: int = 64

    def setup(self):
        assert self.action_head in ('readout', 'action_tokens'), self.action_head

        assert self.max_think_iters >= 1 and self.value_iters >= 1, (self.max_think_iters, self.value_iters)
        assert self.halt_criterion in ('kl', 'l2', 'sigmoid'), self.halt_criterion
        assert self.policy_param in ('last', 'sum', 'mix'), self.policy_param
        assert self.policy_param != 'mix' or (self.halting and self.halt_criterion == 'sigmoid'), \
            "policy_param='mix' needs the sigmoid halting head (act_iters=C:v, halt_criterion='sigmoid')"
        assert self.readout_token in ('thought', 'sep'), self.readout_token
        if self.halting and self.halt_criterion == 'sigmoid':
            self.halt_linear = nn.Dense(1, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.zeros)
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
        if self.cell_pos_embed:
            assert self.tokenizer == 'cell', 'cell_pos_embed needs tokenizer=cell'
            self.cell_pos = self.param('cell_pos_embed', nn.initializers.normal(stddev=0.02), (self.m * self.n, H))
        if self.action_head == 'action_tokens':
            self.action_embed = self.param('action_embed', nn.initializers.normal(stddev=0.02), (self.output_dim_1, H))
        self.core = FPRMCore(core_config(self.config, self.tokenizer))
        self.actor_head = nn.Dense(
            1 if self.action_head == 'action_tokens' else self.output_dim_1,
            bias_init=nn.initializers.zeros,
        )
        self.latent_init = make_latent_init(self.config)
        # The critic: the plain FPRM model (own parameters) for a fixed value_iters core calls; only its value is used.
        self.critic = FPRMThinkerActorValue(
            m=self.m, n=self.n, config=self.config, max_think_iters=self.value_iters, min_think_iters=self.value_iters,
            halt_criterion='latent_residual', halt_residual_thresh=-1.0, output_dim_1=1,  # its logits are unused
            output_dim_2=self.output_dim_2, obs_channels=self.obs_channels, action_head='readout',
            cell_pos_embed=self.cell_pos_embed,
            tokenizer=self.tokenizer, tokenizer_conv_features=self.tokenizer_conv_features,
        )

    def __call__(self, x, normalizer_params=None, halt_key=None, halt_steps=None):
        _, value = self.critic(x, normalizer_params)
        if normalizer_params is not None:
            x = (x - normalizer_params.mean) / (normalizer_params.std)

        cfg = core_config(self.config, self.tokenizer)
        input_shape = x.shape
        x_flat = x.reshape(-1, input_shape[-1])
        B = x_flat.shape[0]
        H = cfg.d_model
        T = self.max_think_iters
        S = max(T - 1, 1)
        sigmoid_head = self.halting and self.halt_criterion == 'sigmoid'

        if self.tokenizer == 'cell':
            cells = self.cell_projection(tokenize(x_flat, self.m, self.n, self.obs_channels))
            if self.cell_pos_embed:
                cells = cells + self.cell_pos.astype(cells.dtype)
        else:
            cells = self.obs_tokenizer(x_flat)  # [B, P, H]
        P = cells.shape[1]
        grid_shape = token_grid_shape(self.tokenizer, self.m, self.n, x_flat.shape[-1], self.tokenizer_conv_features)
        z0 = self.latent_init.astype(cells.dtype)

        if self.is_initializing():
            seq = jnp.concatenate([cells[:1], jnp.zeros((1, 1, H), cells.dtype)], axis=1)
            self.core(seq, seq, jnp.ones(seq.shape[:2], bool), grid_shape, prefix_len=0, suffix_len=1)
            self.actor_head(jnp.zeros((1, H)))
            if sigmoid_head:
                self.halt_linear(jnp.zeros((1, H)))
        params = self.variables['params']
        core_p, actor_p = params['core'], params['actor_head']

        sep = self.readout_token == 'sep'
        sep_in = jnp.broadcast_to(self.sep, (B, 1, H))
        act_tokens = self.action_head == 'action_tokens'
        act_in = jnp.broadcast_to(self.action_embed[None], (B, self.output_dim_1, H)) if act_tokens else sep_in[:, :0]

        def feat(h):  # what the actor head reads
            return rms_norm(h, eps=cfg.rms_norm_eps)

        def readout(z_carried, act_latent):  # z_carried: cells + SEP (+ action tokens)
            if act_tokens:
                logits = (feat(z_carried[:, P + 1:]) @ actor_p['kernel'] + actor_p['bias']).reshape(B, -1)
            else:
                logits = feat(act_latent) @ actor_p['kernel'] + actor_p['bias']
            assert logits.shape[-1] == self.output_dim_1, (logits.shape, self.output_dim_1)
            return logits

        bot = jnp.broadcast_to(self.bot, (B, H))
        z0_thought = jnp.broadcast_to(z0[None, None], (B, 1, H))
        kv_shape = (cfg.num_layers, B, S, cfg.num_heads, H // cfg.num_heads)
        memory = (jnp.zeros(kv_shape, cfg.compute_dtype), jnp.zeros(kv_shape, cfg.compute_dtype))
        z_init = jnp.broadcast_to(z0[None, None], (B, P + 1 + act_in.shape[1], H))
        token_init = jnp.zeros((B, H), cells.dtype)
        logits_init = jnp.zeros((B, self.output_dim_1), cells.dtype)

        def halt_head(h, finite):
            """Logits of (halt, continue) at this pass [B, 2]; continuing is 0, so h_n = sigmoid(logit)."""
            p = self.variables['params']['halt_linear']
            sg = (lambda v: v) if self.halt_head_grad else jax.lax.stop_gradient
            logit = (sg(rms_norm(h, eps=cfg.rms_norm_eps)).astype(jnp.float32) @ p['kernel'] + p['bias'])[:, 0]
            return jnp.where(finite[:, None], jnp.stack([logit, jnp.zeros_like(logit)], axis=-1), 0.0)

        def body(carry, i):
            # One pass (see the module docstring); never frozen. A nonfinite pass keeps the last finite state (and is flagged).
            z, token, logits_prev, nonfinite, (mem_k, mem_v) = carry  # z: cells + SEP latents
            fed_back = rms_norm(token, eps=cfg.rms_norm_eps) if self.thought_norm else token
            thought = jnp.where(i == 0, bot, fed_back + self.latent_embed)
            z_in = jnp.concatenate([z, z0_thought], axis=1)
            x_in = jnp.concatenate([cells, sep_in, act_in, thought[:, None]], axis=1)
            h, thought_kv = mem_core(core_p, z_in, x_in, mem_k, mem_v, jnp.arange(S) < i, grid_shape, P, cfg,
                                     rope=self.cell_rope)
            write = jnp.arange(S) == i
            mem_k = jnp.where(write[None, None, :, None, None], jnp.stack([k for k, _ in thought_kv])[:, :, None], mem_k)
            mem_v = jnp.where(write[None, None, :, None, None], jnp.stack([v for _, v in thought_kv])[:, :, None], mem_v)

            z_cells, out = h[:, :-1], h[:, -1]  # z_cells: cells + SEP (+ action tokens)
            logits = readout(z_cells, z_cells[:, P] if sep else out)
            finite = (jnp.all(jnp.isfinite(logits), axis=-1)
                      & jnp.all(jnp.isfinite(out), axis=-1) & jnp.all(jnp.isfinite(z_cells), axis=(-2, -1)))
            z_cells = jnp.where(finite[:, None, None], z_cells, z)
            latent_res, _, _ = compute_residuals(z[:, :P], z_cells[:, :P], jnp.ones((B, P), bool), cfg.eps)
            out = jnp.where(finite[:, None], out, token)
            logits = jnp.where(finite[:, None], logits, logits_prev)

            h_prev, h_new = (z[:, P], z_cells[:, P]) if sep else (token, out)  # readout latents (passes n-1, n)
            dh = rms_norm(h_new, eps=cfg.rms_norm_eps) - rms_norm(h_prev, eps=cfg.rms_norm_eps)
            l2 = jnp.linalg.norm(jax.lax.stop_gradient(dh).astype(jnp.float32), axis=-1)  # pass 1: against the initial latent
            logits_k = halt_head(h_new, finite) if sigmoid_head else jnp.zeros((B, 1), jnp.float32)
            nonfinite = nonfinite | ~finite
            carry = (z_cells, out, logits, nonfinite, (mem_k, mem_v))
            return carry, (logits, logits_k, l2, nonfinite, latent_res, h_new)

        if self.remat:
            body = jax.checkpoint(body, prevent_cse=False)
        init = (z_init, token_init, logits_init, jnp.zeros((B,), bool), memory)
        _, (logits_t, head_t, l2_t, nonfinite_t, latent_res_t, readout_latent_t) = jax.lax.scan(body, init, jnp.arange(T))
        # logits_t [T, B, A] (l_n); head_t [T, B, 2]; l2_t, nonfinite_t [T, B]

        if self.policy_param == 'sum':  # pi_n from sum_(k<=n) l_k
            csum = jnp.cumsum(logits_t, axis=0)
            if self.logit_sum_stop_grad:
                logits_t = jax.lax.stop_gradient(csum - logits_t) + logits_t
            else:
                logits_t = csum

        # KL_n between the acting policies of passes n and n-1 (pass 1: against uniform)
        prev_t = jnp.concatenate([jnp.zeros_like(logits_t[:1]), logits_t[:-1]], axis=0)
        z_q, z_p = (prev_t, logits_t) if self.halt_kl_reverse else (logits_t, prev_t)
        if self.stable_kl:
            kl_t = stable_softmax.kl(z_q, z_p)
        else:
            lp_q, lp_p = jax.nn.log_softmax(z_q, axis=-1), jax.nn.log_softmax(z_p, axis=-1)
            kl_t = jnp.sum(jnp.exp(lp_q) * (lp_q - lp_p), axis=-1)
        kl_t = jax.lax.stop_gradient(kl_t).astype(jnp.float32)
        sig_t = l2_t if self.halting and self.halt_criterion == 'l2' else kl_t
        # row n: S_(n+1), whether pass n+1 agrees with pass n (row T repeats S_T; pass T never decides)
        sig_next = jnp.concatenate([sig_t[1:], sig_t[-1:]], axis=0)

        self.sow('intermediates', 'policy_kl_t', kl_t)
        self.sow('intermediates', 'halt_signal_t', sig_t)
        min_iters = 1 if T == 1 else self.min_think_iters
        if not self.halting:
            c = jnp.full((B,), T, jnp.int32)
        elif halt_steps is not None:  # the PPO loss replays the rollout's c
            c = jnp.asarray(halt_steps, jnp.int32).reshape(B)
        elif not sigmoid_head:  # 'kl' / 'l2': the first decision pass n whose next pass agrees, d(n, n+1) < threshold
            n, dec = _decision_passes(T, min_iters)
            c = jnp.min(jnp.where(dec & (sig_next < self.halt_threshold), n, T), axis=0).astype(jnp.int32)
        else:
            c = None
        if sigmoid_head:
            c, lp_halt_t = self._halt(head_t, sig_next, halt_key, c, B, T, input_shape[:-1])
        if self.halting:
            self.sow('intermediates', 'policy_log_probs_t',
                     (stable_softmax.log_softmax if self.stable_softmax else jax.nn.log_softmax)(logits_t))
        idx = c - 1
        sig_c = jnp.take_along_axis(sig_next.T, idx[:, None], axis=-1)[:, 0]
        logits = jnp.take_along_axis(logits_t, idx[None, :, None], axis=0)[0]
        if self.policy_param == 'mix':  # log sum_(n<=c) w_n pi_n: normalized log-probs, used as the logits
            log_w = mix_log_weights(lp_halt_t, c, min_iters)
            log_w = jax.lax.stop_gradient(log_w) if self.mix_weight_stop_grad else log_w
            log_pi = (stable_softmax.log_softmax if self.stable_softmax else jax.nn.log_softmax)(logits_t.astype(jnp.float32))
            logits = jax.nn.logsumexp(log_w[..., None] + log_pi, axis=0)
        nonfinite = jnp.take_along_axis(nonfinite_t.T, idx[:, None], axis=-1)[:, 0]

        logits = logits.reshape(input_shape[:-1] + (-1,))
        self.sow('intermediates', 'logits', logits)
        self.sow('intermediates', 'value', value)
        self.sow('intermediates', 'latent_residual_t', latent_res_t)
        self.sow('intermediates', 'readout_latent_t', readout_latent_t)  # [T, B, H]: the latent pi_n is decoded from

        reason = jnp.where(c >= T, STOP_ITERATION_CAP, STOP_CONVERGED)
        reason = jnp.where(nonfinite, STOP_NONFINITE, reason)
        self.sow('intermediates', 'fp_iterations', c)
        self.sow('intermediates', 'fp_residual', sig_c)
        self.sow('intermediates', 'fp_converged', reason == STOP_CONVERGED)
        self.sow('intermediates', 'fp_stop_reason', reason)

        return (stable_softmax.Categorical if self.stable_softmax else distrax.Categorical)(logits=logits), value

    def _halt(self, head_t, sig_next, halt_key, halt_steps, B, T, batch_shape):
        """Sigmoid halting: the think time c [B]; sows the halting intermediates. Row n of head_t is pass n's own
        decision (no lookahead; pass T never decides); sig_next (row n: S_(n+1)) is for ppo.py's halt_signal_diag."""
        self.sow('intermediates', 'halt_log_sig_t', jnp.log(jnp.maximum(sig_next, 0.0) + _KL_EPS))
        lp_t = (stable_softmax.log_softmax if self.stable_softmax else jax.nn.log_softmax)(head_t)  # [T, B, 2]
        min_iters = 1 if T == 1 else self.min_think_iters
        lp_c = c_log_probs(lp_t, min_iters)  # [T, B]
        if halt_steps is not None:
            c = halt_steps
        elif halt_key is not None:
            c = sample_c(lp_t, min_iters, halt_key)
        else:
            c = (jnp.argmax(lp_c, axis=0) + 1).astype(jnp.int32)
        halt_decision_log_probs = decision_log_probs(lp_t, c, min_iters, self.halt_sg_survival)
        p_c = jnp.exp(lp_c)
        self.sow('intermediates', 'halt_path', decision_mask(c, T, min_iters).reshape(batch_shape + (T,)))
        self.sow('intermediates', 'halt_step_probs', jnp.exp(lp_t))
        self.sow('intermediates', 'halt_log_prob', jnp.sum(halt_decision_log_probs, axis=0).reshape(batch_shape))
        self.sow('intermediates', 'halt_decision_log_probs', halt_decision_log_probs.T.reshape(batch_shape + (T,)))
        self.sow('intermediates', 'halt_entropy', -jnp.sum(p_c * lp_c, axis=0).reshape(batch_shape))
        self.sow('intermediates', 'halt_expected_c', jnp.sum(p_c * jnp.arange(1, T + 1)[:, None], axis=0).reshape(batch_shape))
        self.sow('intermediates', 'halt_prob', jnp.exp(lp_t[..., 0]))
        self.sow('intermediates', 'halt_step_log_probs', lp_t)
        return c, lp_t
