"""fprm_mem_halt halting tests (CPU, a few seconds)."""
import math
import os
import sys

import jax
import jax.numpy as jnp
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.fprm import FPRMConfig  # noqa: E402
from models.fprm_mem_halt import (  # noqa: E402
    FPRMMemHaltActorValue, best_step_path, step_path_entropy, step_path_log_prob,
)

CFG = FPRMConfig(d_model=16, num_heads=2)


def _net(**kw):
    return FPRMMemHaltActorValue(m=3, n=3, config=CFG, max_think_iters=4, **kw)


def _intermediates(net, params, x, **kw):
    _, inter = net.apply({'params': params}, x, mutable=['intermediates'], **kw)
    return {k: v[0] for k, v in inter['intermediates'].items()}


def _with(params, name, **leaves):
    return {**params, name: {**params[name], **leaves}}


def _random_head(params, key=2, scale=1.0):
    """The halting MLP's zero-initialized output layer made random, so the head is no longer uniform."""
    w = params['halt_out']['kernel']
    return _with(params, 'halt_out', kernel=scale * jax.random.normal(jax.random.PRNGKey(key), w.shape))


def _paths(T, n, K):
    """Every (mask, c) reachable from a decision at pass n, by enumeration (bit n-1 = decision at pass n)."""
    out = [(1 << (n - 1), n)]  # k = 0
    for k in range(1, K + 1):
        if n + k >= T:
            out.append((1 << (n - 1), T))
        else:
            out += [((1 << (n - 1)) | m, c) for m, c in _paths(T, n + k, K)]
    return list(set(out))  # overshooting k's are one outcome (stop at T)


def _k1_path(c, min_iters, T):
    """K = 1: the decision-pass mask of the path ending at c (a decision at every pass from min_iters on)."""
    return sum(1 << (n - 1) for n in range(min_iters, min(c, T - 1) + 1))


def test_latent_residual_sown():
    net = _net()
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 18))
    params = net.init(jax.random.PRNGKey(0), x)['params']
    res = _intermediates(net, params, x)['latent_residual_t']
    assert res.shape == (4, 2) and bool(jnp.all(jnp.isfinite(res))) and bool(jnp.all(res >= 0))


@pytest.mark.parametrize('K', [1, 4])
def test_head_starts_uniform(K):
    net = _net(min_think_iters=1, halt_max_skip=K)
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 18))
    p = _intermediates(net, net.init(jax.random.PRNGKey(0), x)['params'], x)['halt_step_probs']
    assert p.shape == (4, 3, K + 1) and jnp.allclose(p, 1 / (K + 1), atol=1e-6)


@pytest.mark.parametrize('K', [1, 4])
def test_can_halt_after_one_pass(K):
    net = _net(min_think_iters=1, halt_max_skip=K)
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 18))
    params = net.init(jax.random.PRNGKey(0), x)['params']
    eager = _with(params, 'halt_out', bias=jnp.zeros(K + 1).at[0].set(20.0))
    assert bool(jnp.all(_intermediates(net, eager, x)['fp_iterations'] == 1))
    assert bool(jnp.all(_intermediates(net, eager, x, halt_key=jax.random.PRNGKey(3))['fp_iterations'] == 1))


def test_k1_is_sigmoid_halting():
    # K = 1 is a yes/no halt at every pass: p(c | s) = h_c prod_{i<c} (1 - h_i), h_n = p_n(k = 0), h_T = 1.
    T, min_iters = 4, 2
    net = _net(min_think_iters=min_iters)
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 18))
    params = _random_head(net.init(jax.random.PRNGKey(0), x)['params'], scale=0.05)  # h strictly inside (0, 1) in fp32
    h = _intermediates(net, params, x)['halt_prob']
    log_pc = []
    for c in range(min_iters, T + 1):
        replay = _intermediates(net, params, x, halt_steps=jnp.full((3,), c), halt_path=jnp.full((3,), _k1_path(c, min_iters, T)))
        want = jnp.sum(jnp.log(1 - h[min_iters - 1:c - 1]), axis=0) + (jnp.log(h[c - 1]) if c < T else 0.0)
        assert jnp.allclose(replay['halt_log_prob'], want, atol=1e-5)
        log_pc.append(want)
    log_pc = jnp.stack(log_pc)
    assert jnp.allclose(jnp.sum(jnp.exp(log_pc), axis=0), 1.0, atol=1e-5)
    inter = _intermediates(net, params, x)  # evaluation: the most likely c, and the entropy is H(c)
    assert jnp.array_equal(inter['fp_iterations'], jnp.argmax(log_pc, axis=0) + min_iters)
    assert jnp.allclose(inter['halt_entropy'], -jnp.sum(jnp.exp(log_pc) * log_pc, axis=0), atol=1e-5)


@pytest.mark.parametrize('K,min_iters', [(1, 1), (4, 1), (4, 2)])
def test_best_path_and_entropy_match_enumeration(K, min_iters):
    T, B = 7, 5
    lp = jax.nn.log_softmax(2.0 * jax.random.normal(jax.random.PRNGKey(0), (T, B, K + 1)), axis=-1)
    paths = _paths(T, min_iters, K)
    log_p = jnp.stack([step_path_log_prob(lp, jnp.full((B,), m), jnp.full((B,), c)) for m, c in paths])  # [paths, B]
    mask, c = best_step_path(lp, min_iters)
    i = jnp.argmax(log_p, axis=0)
    assert jnp.array_equal(mask, jnp.array([paths[j][0] for j in i])) and jnp.array_equal(c, jnp.array([paths[j][1] for j in i]))
    assert jnp.allclose(step_path_entropy(lp, min_iters), -jnp.sum(jnp.exp(log_p) * log_p, axis=0), atol=1e-5)


@pytest.mark.parametrize('head_grad', [False, True])
def test_halting_gradient_reaches_trunk_only_with_head_grad(head_grad):
    net = _net(halt_head='none', halt_head_grad=head_grad)  # reads h_n, so the trunk is reachable
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 18))
    params = _random_head(net.init(jax.random.PRNGKey(0), x)['params'], scale=0.3)
    path = jnp.array([_k1_path(2, 2, 4), _k1_path(3, 2, 4)])

    def halt_loss(p):
        return jnp.sum(_intermediates(net, p, x, halt_steps=jnp.array([2, 3]), halt_path=path)['halt_log_prob'])

    g = jax.grad(halt_loss)(params)
    trunk = sum(float(jnp.sum(jnp.abs(v))) for v in jax.tree_util.tree_leaves(g['core']))
    assert (trunk > 0) == head_grad
    assert float(jnp.sum(jnp.abs(g['halt_hidden']['kernel']))) > 0


@pytest.mark.parametrize('signal', ['kl', 'latent'])
def test_lookahead_halts_with_previous_pass(signal):
    # Lookahead: c = 1 is decided on pass 2 and acts with pi_1 (= the output of a 1-pass net); fp_residual is the
    # signal of pass 2.
    net = _net(min_think_iters=1, halt_signal=signal)
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 18))
    params = net.init(jax.random.PRNGKey(0), x)['params']
    eager = _with(params, 'halt_out', bias=jnp.array([20.0, 0.0]))
    inter = _intermediates(net, eager, x)
    assert bool(jnp.all(inter['fp_iterations'] == 1))
    one_pass = _intermediates(FPRMMemHaltActorValue(m=3, n=3, config=CFG, max_think_iters=1, halt_signal=signal), eager, x)
    assert jnp.allclose(inter['logits'], one_pass['logits'], atol=1e-5)
    assert jnp.allclose(inter['fp_residual'], inter['halt_signal_t'][1], atol=1e-6)


@pytest.mark.parametrize('K', [1, 4])
@pytest.mark.parametrize('signal', ['kl', 'latent'])
def test_distance_head_reads_the_next_signal(K, signal):
    # 'distance': the decision at pass n is softmax(MLP(log S_{n+1})), a GELU MLP of the scalar.
    net = _net(min_think_iters=1, halt_signal=signal, halt_max_skip=K)
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 18))
    params = _random_head(net.init(jax.random.PRNGKey(0), x)['params'])
    inter = _intermediates(net, params, x)
    if signal == 'kl':
        assert jnp.allclose(inter['halt_signal_t'], inter['policy_kl_t'])
    u = jnp.log(jnp.maximum(inter['halt_signal_t'][1:], 0.0) + 1e-10)[..., None]
    hid, out = params['halt_hidden'], params['halt_out']
    g = jax.nn.gelu(u @ hid['kernel'] + hid['bias']) @ out['kernel'] + out['bias']
    assert jnp.allclose(inter['halt_step_probs'][:-1], jax.nn.softmax(g, axis=-1), atol=1e-5)


def test_latent_signal_is_l2_of_readout_latents():
    # S_i = ||h_i - h_{i-1}|| with h the readout token's latent: finite and positive (the passes differ); the signal changes
    # only S, not the passes (the same KLs).
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 18))
    net = _net(halt_signal='latent')
    params = net.init(jax.random.PRNGKey(0), x)['params']
    sig = _intermediates(net, params, x)['halt_signal_t']
    assert bool(jnp.all(jnp.isfinite(sig))) and bool(jnp.all(sig > 0))
    # kl and latent see the same passes; only S differs
    kl_net = _net(halt_signal='kl')
    assert jnp.allclose(_intermediates(kl_net, params, x)['policy_kl_t'], _intermediates(net, params, x)['policy_kl_t'])


@pytest.mark.parametrize('K', [1, 4])
def test_no_lookahead_without_signal(K):
    # halt_head='none' (a plain halting head on h_n): the decision at pass 1 reads pass 1 only, as in a 1-pass net.
    kw = dict(min_think_iters=1, halt_head='none', halt_max_skip=K)
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 18))
    params = _random_head(_net(**kw).init(jax.random.PRNGKey(0), x)['params'])
    four = _intermediates(_net(**kw), params, x)
    one = _intermediates(FPRMMemHaltActorValue(m=3, n=3, config=CFG, max_think_iters=1, **kw), params, x)
    assert jnp.allclose(four['halt_step_probs'][0], one['halt_step_probs'][0], atol=1e-4)


def test_sep_token_in_every_pass():
    # Every pass runs [cells, SEP, thought]; readout_token only picks where the readouts come from. The passes (here the
    # cell latents' residuals) are the same for both, the value differs, and SEP reaches the readout either way.
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 18))
    params = _net().init(jax.random.PRNGKey(0), x)['params']
    assert 'sep' in params
    out = {r: _intermediates(_net(readout_token=r), params, x) for r in ('thought', 'sep')}
    assert jnp.allclose(out['thought']['latent_residual_t'], out['sep']['latent_residual_t'])
    assert not jnp.allclose(out['thought']['value'], out['sep']['value'], atol=1e-4)
    for r in ('thought', 'sep'):
        def value_sum(sep):
            return jnp.sum(_intermediates(_net(readout_token=r), {**params, 'sep': sep}, x)['value'] ** 2)

        assert float(jnp.abs(jax.grad(value_sum)(params['sep'])).sum()) > 0


@pytest.mark.parametrize('min_iters', [1, 2])
def test_step_path_probabilities_sum_to_one(min_iters):
    T = 7
    lp = jax.nn.log_softmax(jax.random.normal(jax.random.PRNGKey(0), (T, 1, 5)), axis=-1)
    paths = _paths(T, min_iters, 4)
    total = sum(float(jnp.exp(step_path_log_prob(lp, jnp.array([m]), jnp.array([c])))[0]) for m, c in paths)
    assert abs(total - 1.0) < 1e-5, total


def test_step_path_log_prob_by_hand():
    # T = 8: decide at 1 (k = 2), at 3 (k = 4), at 7 (any k >= 1 reaches the cap) -> c = 8.
    lp = jax.nn.log_softmax(jax.random.normal(jax.random.PRNGKey(1), (8, 1, 5)), axis=-1)
    mask = (1 << 0) | (1 << 2) | (1 << 6)
    want = lp[0, 0, 2] + lp[2, 0, 4] + jax.scipy.special.logsumexp(lp[6, 0, 1:])
    assert jnp.allclose(step_path_log_prob(lp, jnp.array([mask]), jnp.array([8]))[0], want, atol=1e-6)


@pytest.mark.parametrize('K', [1, 4])
def test_replay_matches_rollout(K):
    # The loss replays (c, path) from the rollout and must recover the rollout's log p(path).
    net = _net(min_think_iters=1, halt_max_skip=K)
    x = jax.random.normal(jax.random.PRNGKey(1), (6, 18))
    params = net.init(jax.random.PRNGKey(0), x)['params']
    roll = _intermediates(net, params, x, halt_key=jax.random.PRNGKey(3))
    replay = _intermediates(net, params, x, halt_steps=roll['fp_iterations'], halt_path=roll['halt_path'])
    assert jnp.allclose(roll['halt_log_prob'], replay['halt_log_prob'], atol=1e-5)
    assert bool(jnp.all(roll['halt_log_prob'] < 0)) and bool(jnp.all((roll['fp_iterations'] >= 1) & (roll['fp_iterations'] <= 4)))


def test_halting_batched_jit():
    # Batched jit of the halting path sampling (an earlier head's concat was miscompiled by XLA GPU at batch > 1).
    net = FPRMMemHaltActorValue(m=5, n=5, config=FPRMConfig(num_layers=2), max_think_iters=4, min_think_iters=1,
                                halt_max_skip=4, output_dim_1=4, action_head='readout')
    x = jax.random.normal(jax.random.PRNGKey(1), (8, 50))
    params = net.init(jax.random.PRNGKey(0), x)
    _, inter = jax.jit(lambda p, x, k: net.apply(p, x, halt_key=k, mutable=['intermediates']))(params, x, jax.random.PRNGKey(2))
    assert bool(jnp.all((inter['intermediates']['fp_iterations'][0] >= 1) & (inter['intermediates']['fp_iterations'][0] <= 4)))


def test_stable_softmax_and_kl():
    # stable_kl: the halting KL from the logit difference (exact 0 at equal logits, no ~1e-7 floor); it matches the
    # standard KL where that one is accurate. stable_softmax: the policy and halting-head log-probs are collapse-free.
    from models.stable_softmax import Categorical
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 18))
    params = _net(min_think_iters=1).init(jax.random.PRNGKey(0), x)['params']
    kl_std = _intermediates(_net(min_think_iters=1), params, x)['policy_kl_t']
    net = _net(stable_kl=True, stable_softmax=True, min_think_iters=1)
    dist, _ = net.apply({'params': params}, x)
    kl = _intermediates(net, params, x)['policy_kl_t']
    assert isinstance(dist, Categorical) and bool(jnp.all(kl >= 0))
    big = kl_std > 1e-4
    assert bool(jnp.any(big)) and jnp.allclose(kl[big], kl_std[big], rtol=1e-3)


def _signal_head_case(K, signal, head):
    kw = dict(min_think_iters=1, halt_max_skip=K, halt_signal=signal, halt_head=head, halt_head_width=8)
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 18))
    params = _net(**kw).init(jax.random.PRNGKey(0), x)['params']
    out = params['halt_out']
    out = {'kernel': jax.random.normal(jax.random.PRNGKey(2), out['kernel'].shape),
           'bias': jax.random.normal(jax.random.PRNGKey(3), out['bias'].shape)}
    return kw, x, params, {**params, 'halt_out': out}


@pytest.mark.parametrize('K', [1, 4])
@pytest.mark.parametrize('signal', ['kl', 'latent'])
@pytest.mark.parametrize('head', ['none', 'distance', 'raw'])
def test_every_head_starts_uniform(K, signal, head):
    kw, x, params, _ = _signal_head_case(K, signal, head)
    assert jnp.allclose(_intermediates(_net(**kw), params, x)['halt_step_probs'], 1 / (K + 1), atol=1e-6)


@pytest.mark.parametrize('K', [1, 4])
def test_raw_kl_head_reads_both_policies(K):
    # The decision at pass 1 is softmax(g([log pi_1, log pi_2])); pi_n = the policy of an n-pass net.
    kw, x, _, params = _signal_head_case(K, 'kl', 'raw')
    log_pi = [jax.nn.log_softmax(_intermediates(FPRMMemHaltActorValue(m=3, n=3, config=CFG, max_think_iters=n, **kw), params, x,
                                        halt_steps=jnp.full((3,), n), halt_path=jnp.zeros((3,), int))['logits'], -1)
          for n in (1, 2)]  # replaying c = n just selects pi_n
    hid, out = params['halt_hidden'], params['halt_out']
    g = jax.nn.gelu(jnp.concatenate(log_pi, -1) @ hid['kernel'] + hid['bias']) @ out['kernel'] + out['bias']
    assert jnp.allclose(_intermediates(_net(**kw), params, x)['halt_step_probs'][0], jax.nn.softmax(g, -1), atol=1e-4)


def test_raw_latent_head_varies_and_trains():
    kw, x, _, params = _signal_head_case(1, 'latent', 'raw')
    path = jnp.array([_k1_path(c, 1, 4) for c in (1, 2, 3)])

    def halt_loss(p):
        return jnp.sum(_intermediates(_net(**kw), p, x, halt_steps=jnp.array([1, 2, 3]), halt_path=path)['halt_log_prob'])

    h = _intermediates(_net(**kw), params, x)['halt_prob'][:-1]
    assert float(jnp.max(h) - jnp.min(h)) > 1e-3
    g = jax.grad(halt_loss)(params)
    assert float(jnp.sum(jnp.abs(g['halt_hidden']['kernel']))) > 0
    assert sum(float(jnp.sum(jnp.abs(v))) for v in jax.tree_util.tree_leaves(g['core'])) == 0  # stop-gradient'ed inputs


def test_no_halting_acts_with_the_last_pass():
    # halting=False (iso_mem): no halting parameters, c = T for every state, and the same readout as the halting model
    # forced to act at the last pass.
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 18))
    params = _net(halting=False).init(jax.random.PRNGKey(0), x)['params']
    assert not any(k.startswith('halt_') for k in params)
    iso = _intermediates(_net(halting=False), params, x)
    assert bool(jnp.all(iso['fp_iterations'] == 4)) and 'halt_log_prob' not in iso
    head = {k: v for k, v in _net().init(jax.random.PRNGKey(0), x)['params'].items() if k.startswith('halt_')}
    forced = _intermediates(_net(), {**params, **head}, x, halt_steps=jnp.full((3,), 4),
                            halt_path=jnp.full((3,), _k1_path(4, 2, 4)))
    assert jnp.allclose(iso['logits'], forced['logits'], atol=1e-6) and jnp.allclose(iso['value'], forced['value'], atol=1e-6)
