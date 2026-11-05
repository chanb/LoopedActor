"""ppo.halt_soft_entropy: sum_n d^(n-1) H(p_n) over each path's decision passes, against a loop (CPU, instant)."""
import os
import sys

import jax
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ppo import halt_soft_entropy  # noqa: E402


def test_sums_discounted_entropy_over_decision_passes():
    T, B, K1, d = 6, 3, 3, 0.9
    lp = np.asarray(jax.nn.log_softmax(jax.random.normal(jax.random.PRNGKey(0), (T, B, K1)), axis=-1))
    path = np.array([0b000001, 0b010110, 0b011111])  # decisions at passes {1}, {2, 3, 5}, {1..5}
    want = [sum(d ** (n - 1) * -np.sum(np.exp(lp[n - 1, b]) * lp[n - 1, b]) for n in range(1, T + 1) if path[b] >> (n - 1) & 1)
            for b in range(B)]
    np.testing.assert_allclose(halt_soft_entropy(lp, path, d), want, rtol=1e-5)
