"""halt_head='convtest' tests (CPU, tiny model): the learned convergence test of fprm_mem_halt."""
import math
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.fprm import FPRMConfig  # noqa: E402
from models.fprm_mem_halt import FPRMMemHaltActorValue  # noqa: E402

CFG = FPRMConfig(d_model=16, num_heads=2)
H, T = 16, 4
CONV = dict(halt_head='convtest', halt_signal='latent', halt_max_skip=1, min_think_iters=1)
GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'fprm_mem_halt_golden.npz')
GOLDEN_CONFIGS = [  # the pre-convtest heads, as configured in the runs
    dict(), dict(halt_signal='latent'), dict(halt_head='raw', halt_signal='latent'), dict(halt_head='raw'),
    dict(halt_head='none', halt_max_skip=4), dict(halt_max_skip=4, halt_signal='latent'),
    dict(readout_token='thought', value_token='sep', halt_token='sep', halt_head='none'),
]


def _net(**kw):
    return FPRMMemHaltActorValue(m=3, n=3, config=CFG, max_think_iters=T, **kw)


def _apply(net, params, x, **kw):
    _, inter = net.apply({'params': params}, x, mutable=['intermediates'], **kw)
    return {k: v[0] for k, v in inter['intermediates'].items()}


def _x(B=3, key=1):
    return jax.random.normal(jax.random.PRNGKey(key), (B, 18))


def _c_probs(lp):
    """P(c = n), n = 1..T, of a K = 1 head from its log-probabilities lp [T, B, 2] (decisions at every pass < T)."""
    h = jnp.exp(lp[:-1, :, 0])  # hazards at passes 1..T-1
    alive = jnp.concatenate([jnp.ones_like(h[:1]), jnp.cumprod(1 - h, axis=0)], axis=0)  # P(c >= n)
    return jnp.concatenate([alive[:-1] * h, alive[-1:]], axis=0)


def test_features_formulas_and_clipping():
    from models.fprm_mem_halt import convtest_features
    s_prev = jnp.array([4.0, 2.0, 1.0, 1.0, 1e-9, 4.0, 40.0])
    s_next = jnp.array([2.0, 1.0, 1.5, 1e-9, 1e-9, 39.0, 41.0])  # ratios 0.5, 0.5, 1.5 (clip .999), tiny, 1, ~10, ~1
    a, L, f = convtest_features(s_next, s_prev, math.sqrt(H))
    s_n, s_p = np.asarray(s_next, np.float64), np.asarray(s_prev, np.float64)
    a_want = np.clip(np.log((s_n + 1e-12) / math.sqrt(H)), np.log(1e-4), np.log(2))
    L_want = np.clip((s_n + 1e-12) / (s_p + 1e-12), 1e-3, 0.999)
    np.testing.assert_allclose(a, a_want, rtol=1e-5)
    np.testing.assert_allclose(L, L_want, rtol=1e-5)
    np.testing.assert_allclose(f, a_want + np.log(L_want / (1 - L_want)), rtol=1e-5, atol=1e-5)
    assert float(a[3]) == pytest.approx(math.log(1e-4)) and float(a[6]) == pytest.approx(math.log(2))
    assert float(L[2]) == pytest.approx(0.999) and float(L[3]) == pytest.approx(1e-3)


def test_features_sown_after_the_shift():
    # The decision at pass n reads S_(n+1) and S_n (pass 1: S_2 against S_1; no decision reads S_0); halt_signal_t is unshifted.
    from models.fprm_mem_halt import convtest_features
    x = _x()
    net = _net(**CONV)
    out = _apply(net, net.init(jax.random.PRNGKey(0), x)['params'], x)
    s = out['halt_signal_t']  # [T, B], row i = S_(i+1)
    _, _, f = convtest_features(s[1:], s[:-1], math.sqrt(H))
    np.testing.assert_allclose(out['halt_feature_t'][:-1], f, rtol=1e-5, atol=1e-5)
    assert out['halt_feature_t'].shape == (T, 3)


@pytest.mark.parametrize('stable_kl', [False, True])
def test_features_kl(stable_kl):
    # halt_signal='kl': the features read sqrt(2 KL), unnormalized; decision 1's S_1 is KL(pi_1 || uniform).
    from models.fprm_mem_halt import convtest_features
    x = _x()
    net = _net(halt_head='convtest', halt_max_skip=1, min_think_iters=1, stable_kl=stable_kl)
    out = _apply(net, net.init(jax.random.PRNGKey(0), x)['params'], x)
    s = out['halt_signal_t']
    np.testing.assert_allclose(s, out['policy_kl_t'], rtol=1e-6)
    d = jnp.sqrt(2 * jnp.maximum(s, 0.0))
    _, _, f = convtest_features(d[1:], d[:-1], 1.0)
    np.testing.assert_allclose(out['halt_feature_t'][:-1], f, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize('offset', ['uniform', 'none'])
def test_init(offset):
    x = _x()
    net = _net(halt_offset=offset, **CONV)
    params = net.init(jax.random.PRNGKey(0), x)['params']
    assert float(params['theta1']) == 0.0
    lp = _apply(net, params, x)['halt_step_log_probs']
    if offset == 'uniform':  # c ~ Uniform{1..T}
        np.testing.assert_allclose(_c_probs(lp), 1.0 / T, atol=1e-6)
    else:  # hazard 1/T at every decision pass
        assert float(params['theta0']) == pytest.approx(-math.log(T - 1))
        np.testing.assert_allclose(jnp.exp(lp[:-1, :, 0]), 1.0 / T, atol=1e-6)


def _trained(params, theta0=-0.3, theta1=-0.7):
    return {**params, 'theta0': jnp.float32(theta0), 'theta1': jnp.float32(theta1)}


def test_no_batch_dependence():
    x = _x(128)
    net = _net(halt_offset='uniform', **CONV)
    params = _trained(net.init(jax.random.PRNGKey(0), x)['params'])
    big, one = _apply(net, params, x), _apply(net, params, x[:1])
    # The head mixes nothing across the batch: from the same features its logits are bit-identical ...
    feats = big['halt_feature_t'].T
    np.testing.assert_array_equal(_apply(net, params, x, halt_features=feats)['halt_step_log_probs'][:, :1],
                                  _apply(net, params, x[:1], halt_features=feats[:1])['halt_step_log_probs'])
    # ... and recomputed through the trunk they differ by float rounding only (the trunk's matmuls depend on B).
    np.testing.assert_allclose(big['halt_feature_t'][:, :1], one['halt_feature_t'], rtol=1e-5)
    np.testing.assert_allclose(big['halt_step_log_probs'][:, :1], one['halt_step_log_probs'], rtol=1e-5, atol=1e-5)


def test_replay_features():
    x = _x(8)
    net = _net(halt_offset='uniform', **CONV)
    params = _trained(net.init(jax.random.PRNGKey(0), x)['params'])
    roll = _apply(net, params, x, halt_key=jax.random.PRNGKey(3))
    noisy = jax.tree_util.tree_map(lambda p: p + 0.3 * jax.random.normal(jax.random.PRNGKey(4), p.shape), params['core'])
    perturbed = {**params, 'core': noisy}
    replay = dict(halt_steps=roll['fp_iterations'], halt_path=roll['halt_path'])
    stored = _apply(net, perturbed, x, halt_features=roll['halt_feature_t'].T, **replay)
    np.testing.assert_allclose(stored['halt_log_prob'], roll['halt_log_prob'], atol=1e-6)
    recomputed = _apply(net, perturbed, x, **replay)  # without the stored features the path probability moves
    assert not np.allclose(recomputed['halt_log_prob'], roll['halt_log_prob'], atol=1e-4)


def test_fisher_preconditioner_logistic():
    # Logistic regression over decision passes: g = grad of the mean (over states) summed NLL, Sigma = the Fisher
    # mean_b sum_n p (1 - p) phi phi^T + damping I; Sigma^-1 g must equal the analytic Fisher step.
    from models.fprm_mem_halt import convtest_fisher
    B, n_pass, damping = 64, 5, 1e-3
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)
    f = jax.random.normal(k1, (n_pass, B)) * 2.0
    mask = jax.random.bernoulli(k2, 0.7, (n_pass, B))
    y = jax.random.bernoulli(k3, jax.nn.sigmoid(1.0 - 1.5 * f)).astype(jnp.float32)
    theta = jnp.array([0.2, -0.4])

    def nll(theta):
        z = theta[0] + theta[1] * f
        return jnp.mean(jnp.sum(jnp.where(mask, -(y * jax.nn.log_sigmoid(z) + (1 - y) * jax.nn.log_sigmoid(-z)), 0.0), 0))

    g = jax.grad(nll)(theta)
    p = jax.nn.sigmoid(theta[0] + theta[1] * f)
    step = jnp.linalg.solve(convtest_fisher(f, p, mask, damping), g)

    f64, p64, m64 = (np.asarray(v, np.float64) for v in (f, p, mask))
    F = np.zeros((2, 2))
    for b in range(B):
        for n in range(n_pass):
            if m64[n, b]:
                phi = np.array([1.0, f64[n, b]])
                F += p64[n, b] * (1 - p64[n, b]) * np.outer(phi, phi)
    F = F / B + damping * np.eye(2)
    np.testing.assert_allclose(step, np.linalg.solve(F, np.asarray(g, np.float64)), rtol=1e-4)


def golden_outputs(kw):
    x = _x(5)
    net = _net(min_think_iters=1, **kw)
    params = net.init(jax.random.PRNGKey(0), x)['params']
    w = params['halt_out']['kernel']
    params = {**params, 'halt_out': {**params['halt_out'], 'kernel': jax.random.normal(jax.random.PRNGKey(2), w.shape)}}
    keys = ('logits', 'value', 'halt_log_prob', 'halt_step_log_probs', 'fp_iterations', 'halt_path', 'halt_signal_t')
    sampled = _apply(net, params, x, halt_key=jax.random.PRNGKey(3))
    greedy = _apply(net, params, x)
    return {**{f's/{k}': sampled[k] for k in keys}, **{f'g/{k}': greedy[k] for k in keys}}


def test_old_heads_bit_identical():
    golden = np.load(GOLDEN)
    made_with = str(golden['jax_version']) if 'jax_version' in golden else '0.5.3'
    if made_with != jax.__version__:  # XLA versions round differently in the last bit
        pytest.skip(f'golden outputs made under jax {made_with}, running {jax.__version__} (tests/make_convtest_golden.py)')
    for i, kw in enumerate(GOLDEN_CONFIGS):
        for k, v in golden_outputs(kw).items():
            np.testing.assert_array_equal(np.asarray(v), golden[f'{i}/{k}'], err_msg=f'{kw} {k}')
