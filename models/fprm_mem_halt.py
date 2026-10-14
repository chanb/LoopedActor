"""`fprm_mem` with stochastic, learned halting driven by the policy KL between consecutive passes.

The trunk is exactly `fprm_mem` (looped board + thought memory, `mem_core`); only halting differs. Instead of the
hard rule KL_i <= halt_kl, pass i halts with probability

    h_i = sigmoid(beta_i - alpha * log KL_i),     KL_i = KL(pi_i || pi_{i-1}),  alpha = softplus(.) > 0

so h_i -> 1 as the policy stops changing (KL_i -> 0) and h_i -> 0 while thinking still changes its mind. beta_i is a
learned bias plus (`halt_beta_head`) a zero-initialized Dense on the rms-normed newest thought of pass i, i.e.
state-dependent. alpha = 1, beta = log(halt_kl) at initialization, so h_i = 1/2 exactly at the old threshold
KL_i = halt_kl. KL_i is stop-gradient'ed.

The think time c is an action of the joint policy:

    p(c = n | s) = h_n * prod_{i<n} (1 - h_i),    h_i = 0 for i < max(min_think_iters, 2), h_T = 1 (T = max_think_iters)
    log p(c, a | s) = log p(c | s) + log pi_c(a | s)

(pass 1 has no previous policy, so it never halts unless T = 1). All T passes are always executed (as in fprm_mem);
the readout of pass c is selected afterwards, so passes after c get no gradient from the actor loss.

Choice of c in `__call__`: `halt_steps` given (the PPO loss: the c of the rollout, 1-indexed) > `halt_key` given
(rollout: c ~ p(c | s)) > neither (evaluation: c = argmax p(c | s)).

The value is read out from the last pass (fprm_mem's `value_at_max_iters`): since c is an action, a baseline that
depended on it would bias the halting gradient.

Sown intermediates: those of fprm_mem ('logits'/'value' at the chosen c, 'fp_iterations' = c, 'fp_residual' = KL_c)
plus 'halt_log_prob' (log p(c | s)), 'halt_entropy' (H(p(. | s))), 'halt_prob' (h_i, [T, B]), 'halt_alpha',
'halt_beta_t' (beta_i, [T, B]).
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
    make_latent_init,
    rms_norm,
)
from models.fprm_mem import mem_core
from models.fprm_thinker import tokenize
from models.tokenizers import ObsTokenizer, check_tokenizer, core_config, token_grid_shape

_NEG = -1e9  # log of a forced-zero probability (finite: keeps p log p and the gradients NaN-free)
_KL_EPS = 1e-10  # floor inside log KL (KL may round to 0 or slightly below)


def halt_log_probs(halt_logit, min_iters):
    """log p(c = n | s), n = 1..T, from per-pass halting logits [T, B], with h = 0 for n < min_iters and h_T = 1."""
    T = halt_logit.shape[0]
    n = jnp.arange(1, T + 1)[:, None]
    forced0 = n < min_iters
    forced1 = n == T
    log_h = jnp.where(forced1, 0.0, jnp.where(forced0, _NEG, jax.nn.log_sigmoid(halt_logit)))
    log_not_h = jnp.where(forced1, _NEG, jnp.where(forced0, 0.0, jax.nn.log_sigmoid(-halt_logit)))
    # sum_{i<n} log(1 - h_i), as a shifted cumsum (subtracting log_not_h would cancel catastrophically against _NEG).
    survive = jnp.concatenate([jnp.zeros_like(log_not_h[:1]), jnp.cumsum(log_not_h[:-1], axis=0)], axis=0)
    return log_h + survive, jnp.exp(log_h)


class FPRMMemHaltActorValue(nn.Module):
    """fprm_mem trunk with the stochastic KL-sigmoid halting policy (see the module docstring)."""

    m: int = 3
    n: int = 3
    config: FPRMConfig = FPRMConfig()
    max_think_iters: int = 8
    min_think_iters: int = 2
    halt_kl: float = 1e-3          # initialization only: h_i = 1/2 at KL_i = halt_kl (beta = log halt_kl, alpha = 1)
    halt_beta_head: bool = False    # beta_i = bias + Dense(rms_norm(thought_i)) (state-dependent); False: a learned scalar
    halt_head_grad: bool = False   # let the halting loss backpropagate into the trunk through the beta head's input
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
        assert self.halt_kl > 0, 'halt_kl initializes beta = log(halt_kl)'
        check_tokenizer(self.tokenizer, self.action_head, self.obs_channels)
        H = self.config.d_model
        self.cell_projection = nn.Dense(H)
        if self.tokenizer != 'cell':
            self.obs_tokenizer = ObsTokenizer(
                self.tokenizer, self.m, self.n, self.obs_channels, H, self.tokenizer_conv_features,
            )
        self.bot = self.param('bot', nn.initializers.normal(stddev=0.02), (1, H))
        self.latent_embed = self.param('latent_embed', nn.initializers.normal(stddev=0.02), (1, H))
        self.core = FPRMCore(core_config(self.config, self.tokenizer))
        if self.action_head == 'per_cell':
            assert self.output_dim_1 % (self.m * self.n) == 0, (self.output_dim_1, self.m, self.n)
        self.actor_head = nn.Dense(
            self.output_dim_1 // (self.m * self.n) if self.action_head == 'per_cell' else self.output_dim_1,
            bias_init=nn.initializers.zeros,
        )
        self.value_head = nn.Dense(self.output_dim_2, bias_init=nn.initializers.zeros)
        # Halting: alpha = softplus(halt_alpha_raw) (= 1 at init); beta from a zero-kernel Dense (= log halt_kl at init).
        self.halt_alpha_raw = self.param('halt_alpha_raw', nn.initializers.constant(math.log(math.e - 1.0)), ())
        self.halt_beta = nn.Dense(
            1, kernel_init=nn.initializers.zeros, bias_init=nn.initializers.constant(math.log(self.halt_kl)),
        )
        self.latent_init = make_latent_init(self.config)

    def __call__(self, x, normalizer_params=None, halt_key=None, halt_steps=None):
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
            self.halt_beta(jnp.zeros((1, H)))
        params = self.variables['params']
        core_p, actor_p, value_p = params['core'], params['actor_head'], params['value_head']
        beta_p = params['halt_beta']

        def readout(z_cells, out):
            last_latent = rms_norm(out, eps=cfg.rms_norm_eps)
            if per_cell:
                cell_latents = rms_norm(z_cells, eps=cfg.rms_norm_eps)
                logits = (cell_latents @ actor_p['kernel'] + actor_p['bias']).reshape(B, -1)
            else:
                logits = last_latent @ actor_p['kernel'] + actor_p['bias']
            assert logits.shape[-1] == self.output_dim_1, (logits.shape, self.output_dim_1)
            value = jnp.squeeze(last_latent @ value_p['kernel'] + value_p['bias'], axis=-1)
            if self.halt_beta_head:
                beta_in = last_latent if self.halt_head_grad else jax.lax.stop_gradient(last_latent)
                beta = jnp.squeeze(beta_in @ beta_p['kernel'] + beta_p['bias'], axis=-1)
            else:
                beta = jnp.broadcast_to(beta_p['bias'][0], (B,))
            return logits, value, beta.astype(jnp.float32)

        bot = jnp.broadcast_to(self.bot, (B, H))
        latent_embed = self.latent_embed
        z0_thought = jnp.broadcast_to(z0[None, None], (B, 1, H))
        kv_shape = (cfg.num_layers, B, S, cfg.num_heads, H // cfg.num_heads)
        memory = (jnp.zeros(kv_shape, cfg.compute_dtype), jnp.zeros(kv_shape, cfg.compute_dtype))
        z_init = jnp.broadcast_to(z0[None, None], cells.shape)
        token_init = jnp.zeros((B, H), cells.dtype)
        logits_init = jnp.zeros((B, self.output_dim_1), cells.dtype)
        value_init = jnp.zeros((B,), cells.dtype)

        def body(carry, i):
            # Same pass as fprm_mem; never frozen. A nonfinite pass keeps the last finite state (and is flagged).
            z, token, logits_prev, value_prev, nonfinite, (mem_k, mem_v) = carry
            fed_back = rms_norm(token, eps=cfg.rms_norm_eps) if self.thought_norm else token
            thought = jnp.where(i == 0, bot, fed_back + latent_embed)
            z_in = jnp.concatenate([z, z0_thought], axis=1)
            x_in = jnp.concatenate([cells, thought[:, None]], axis=1)
            h, thought_kv = mem_core(core_p, z_in, x_in, mem_k, mem_v, jnp.arange(S) < i, grid_shape, P, cfg)
            write = jnp.arange(S) == i
            mem_k = jnp.where(write[None, None, :, None, None], jnp.stack([k for k, _ in thought_kv])[:, :, None], mem_k)
            mem_v = jnp.where(write[None, None, :, None, None], jnp.stack([v for _, v in thought_kv])[:, :, None], mem_v)

            z_cells, out = h[:, :P], h[:, P]
            logits, value, beta = readout(z_cells, out)
            finite = (jnp.all(jnp.isfinite(logits), axis=-1) & jnp.isfinite(value)
                      & jnp.all(jnp.isfinite(out), axis=-1) & jnp.all(jnp.isfinite(z_cells), axis=(-2, -1)))
            z_cells = jnp.where(finite[:, None, None], z_cells, z)
            out = jnp.where(finite[:, None], out, token)
            logits = jnp.where(finite[:, None], logits, logits_prev)
            value = jnp.where(finite, value, value_prev)
            beta = jnp.where(finite, beta, 0.0)

            lp_new = jax.nn.log_softmax(logits, axis=-1)
            lp_prev = jax.nn.log_softmax(logits_prev, axis=-1)
            kl = jnp.sum(jnp.exp(lp_new) * (lp_new - lp_prev), axis=-1)
            kl = jax.lax.stop_gradient(kl).astype(jnp.float32)  # pass 1 (no previous policy) is forced h = 0 below
            nonfinite = nonfinite | ~finite
            carry = (z_cells, out, logits, value, nonfinite, (mem_k, mem_v))
            return carry, (logits, value, beta, kl, nonfinite)

        if self.remat:
            body = jax.checkpoint(body, prevent_cse=False)
        init = (z_init, token_init, logits_init, value_init, jnp.zeros((B,), bool), memory)
        _, (logits_t, value_t, beta_t, kl_t, nonfinite_t) = jax.lax.scan(body, init, jnp.arange(T))
        # logits_t [T, B, A]; value_t, beta_t, kl_t, nonfinite_t [T, B]

        alpha = jax.nn.softplus(self.halt_alpha_raw)
        halt_logit = beta_t - alpha * jnp.log(jnp.maximum(kl_t, 0.0) + _KL_EPS)
        min_iters = 1 if T == 1 else max(self.min_think_iters, 2)
        log_pc, h_t = halt_log_probs(halt_logit, min_iters)  # [T, B]
        log_pc_b = log_pc.T  # [B, T]

        if halt_steps is not None:
            c = jnp.asarray(halt_steps, jnp.int32).reshape(B)
        elif halt_key is not None:
            c = jax.random.categorical(halt_key, log_pc_b, axis=-1).astype(jnp.int32) + 1
        else:
            c = jnp.argmax(log_pc_b, axis=-1).astype(jnp.int32) + 1
        idx = c - 1

        logits = jnp.take_along_axis(logits_t, idx[None, :, None], axis=0)[0]
        halt_log_prob = jnp.take_along_axis(log_pc_b, idx[:, None], axis=-1)[:, 0]
        halt_entropy = -jnp.sum(jnp.exp(log_pc_b) * log_pc_b, axis=-1)
        kl_c = jnp.take_along_axis(kl_t.T, idx[:, None], axis=-1)[:, 0]
        nonfinite = jnp.take_along_axis(nonfinite_t.T, idx[:, None], axis=-1)[:, 0]

        logits = logits.reshape(input_shape[:-1] + (-1,))
        value = value_t[-1].reshape(input_shape[:-1])
        self.sow('intermediates', 'logits', logits)
        self.sow('intermediates', 'value', value)
        self.sow('intermediates', 'halt_log_prob', halt_log_prob.reshape(input_shape[:-1]))
        self.sow('intermediates', 'halt_entropy', halt_entropy.reshape(input_shape[:-1]))
        self.sow('intermediates', 'halt_prob', h_t)
        self.sow('intermediates', 'halt_alpha', alpha)
        self.sow('intermediates', 'halt_beta_t', beta_t)

        reason = jnp.where(c >= T, STOP_ITERATION_CAP, STOP_CONVERGED)
        reason = jnp.where(nonfinite, STOP_NONFINITE, reason)
        self.sow('intermediates', 'fp_iterations', c)
        self.sow('intermediates', 'fp_residual', kl_c)
        self.sow('intermediates', 'fp_converged', reason == STOP_CONVERGED)
        self.sow('intermediates', 'fp_stop_reason', reason)

        return distrax.Categorical(logits=logits), value
