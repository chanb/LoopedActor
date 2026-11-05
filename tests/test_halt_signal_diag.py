import jax, jax.numpy as jnp
from ppo import halt_signal_diag


def test_fit_recovers_threshold():
    x = jax.random.uniform(jax.random.PRNGKey(0), (16, 512), minval=-14.0, maxval=-4.0)
    gain = 0.01 * (x + 9.0) + 0.002 * jax.random.normal(jax.random.PRNGKey(1), x.shape)  # continue better iff log S > -9
    out = halt_signal_diag(x, gain, jnp.ones_like(x, bool), jnp.zeros_like(x), 'all')
    a, b = out['halt_diag/all_alpha_fit'], out['halt_diag/all_beta_fit']
    assert a > 1.0 and abs(b / a + 9.0) < 0.3, (a, b)
    assert out['halt_diag/all_obj_fit'] > out['halt_diag/all_obj_const'] + 1e-3
    assert out['halt_diag/all_corr'] > 0.8


def test_blind_signal_gives_small_alpha():
    x = jax.random.uniform(jax.random.PRNGKey(0), (16, 512), minval=-14.0, maxval=-4.0)
    gain = 0.001 + 0.01 * jax.random.normal(jax.random.PRNGKey(1), x.shape)
    out = halt_signal_diag(x, gain, jnp.ones_like(x, bool), jnp.zeros_like(x), 'all')
    assert abs(out['halt_diag/all_alpha_fit']) < 0.1, out['halt_diag/all_alpha_fit']
