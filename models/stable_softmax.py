"""log-softmax, categorical and KL without the two rounding bugs of Barron et al., "A Fix for Softmax and
Cross-Entropy" (arXiv 2610.31642).

Softmax Collapse: jax.nn.log_softmax forms 1 + e^-m (m = the argmax logit's margin over the LSE of the rest), which
rounds to 1 once m > 16.6 in fp32, so log p_max and its gradient round to 0 while the other gradients stay non-zero.
`log_softmax` never adds 1 to e^-m (softplus of the margin instead).

Softmax Cancellation: KL(q || p) = sum q (log q - log p) from two separate log-softmaxes keeps each one's rounding
error (~1e-7 in fp32, a per-row constant that does not cancel), so it has a ~1e-7 noise floor and is non-zero at
q = p. `kl` works from the logit difference in log space (expm1 / log1p) and is exactly 0 at equal logits.
"""

import distrax
import jax
import jax.numpy as jnp


def log_softmax(z):
    i_max = jnp.argmax(z, axis=-1)
    z_max = jnp.take_along_axis(z, i_max[..., None], axis=-1)
    is_max = jax.nn.one_hot(i_max, z.shape[-1], dtype=bool)
    shifted = jnp.where(is_max, 0.0, z - z_max)  # exact 0 at the argmax
    m = -jax.nn.logsumexp(jnp.where(is_max, -jnp.inf, shifted), axis=-1, keepdims=True)  # margin
    return shifted - jax.nn.softplus(-m)


def kl(z_q, z_p):
    """KL(softmax(z_q) || softmax(z_p)) over the last axis, via s = log(q / p) from the paper's eq. 9."""
    p = jnp.exp(log_softmax(z_p))
    delta = z_q - z_p
    d = delta - jnp.sum(p * delta, axis=-1, keepdims=True)  # center under p
    s = d - jnp.log1p(jnp.sum(p * jnp.expm1(d), axis=-1, keepdims=True))  # log(q / p)
    # sum q s would cancel first-order terms into a second-order KL; with sum p e^s = 1 it equals
    # sum p (s e^s - expm1(s)) = sum p (s expm1(s) - (expm1(s) - s)), every term >= 0 (~ s^2 / 2).
    em1 = jnp.expm1(s)
    small = jnp.abs(s) < 0.1
    sm = jnp.where(small, s, 0.0)
    em1_minus_s = jnp.where(small, sm * sm / 2 * (1 + sm / 3 * (1 + sm / 4 * (1 + sm / 5 * (1 + sm / 6)))), em1 - s)
    return jnp.sum(p * (s * em1 - em1_minus_s), axis=-1)


class Categorical(distrax.Categorical):
    """distrax.Categorical with log-probabilities (log_prob, entropy) from the collapse-free `log_softmax`."""

    def __init__(self, logits, dtype=int):
        super().__init__(logits=logits, dtype=dtype)
        self._logits = log_softmax(jnp.asarray(logits))

    def entropy(self):
        return -jnp.sum(jnp.exp(self._logits) * self._logits, axis=-1)
