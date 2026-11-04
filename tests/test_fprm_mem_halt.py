"""fprm_mem_halt halting tests (CPU, a few seconds)."""
import math
import os
import sys

import jax
import jax.numpy as jnp
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.fprm import FPRMConfig, rms_norm  # noqa: E402
from models.fprm_mem_halt import (  # noqa: E402
    FPRMMemHaltActorValue, c_log_probs, decision_log_probs, decision_mask, sample_c,
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
    """The halting head moved off its init: a random state-dependent beta (read with halt_beta_head) and alpha != 1."""
    w = params['halt_beta']['kernel']
    params = _with(params, 'halt_beta', kernel=scale * jax.random.normal(jax.random.PRNGKey(key), w.shape), bias=jnp.array([-3.0]))
    return {**params, 'halt_alpha_raw': jnp.float32(0.3)}


def _eager(params):
    """A halting head that halts at its first decision."""
    return _with(params, 'halt_beta', bias=jnp.array([50.0]))


def _k1_path(c, min_iters, T):
    """K = 1: the decision-pass mask of the path ending at c (a decision at every pass from min_iters on)."""
    return sum(1 << (n - 1) for n in range(min_iters, min(c, T - 1) + 1))


def test_latent_residual_sown():
    net = _net()
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 18))
    params = net.init(jax.random.PRNGKey(0), x)['params']
    res = _intermediates(net, params, x)['latent_residual_t']
    assert res.shape == (4, 2) and bool(jnp.all(jnp.isfinite(res))) and bool(jnp.all(res >= 0))


def test_can_halt_after_one_pass():
    net = _net(min_think_iters=1)
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 18))
    eager = _eager(net.init(jax.random.PRNGKey(0), x)['params'])
    assert bool(jnp.all(_intermediates(net, eager, x)['fp_iterations'] == 1))
    assert bool(jnp.all(_intermediates(net, eager, x, halt_key=jax.random.PRNGKey(3))['fp_iterations'] == 1))


def test_log_p_c_is_sigmoid_halting():
    # A yes/no halt at every pass: p(c | s) = h_c prod_{i<c} (1 - h_i), h_T = 1.
    T, min_iters = 4, 2
    net = _net(min_think_iters=min_iters)
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 18))
    params = _random_head(net.init(jax.random.PRNGKey(0), x)['params'])
    h = _intermediates(net, params, x)['halt_prob']
    assert bool(jnp.all((h > 1e-3) & (h < 1 - 1e-3)))  # strictly inside (0, 1) in fp32
    log_pc = []
    for c in range(min_iters, T + 1):
        replay = _intermediates(net, params, x, halt_steps=jnp.full((3,), c))
        assert jnp.array_equal(replay['halt_path'], jnp.full((3,), _k1_path(c, min_iters, T)))
        want = jnp.sum(jnp.log(1 - h[min_iters - 1:c - 1]), axis=0) + (jnp.log(h[c - 1]) if c < T else 0.0)
        assert jnp.allclose(replay['halt_log_prob'], want, atol=1e-5)
        log_pc.append(want)
    log_pc = jnp.stack(log_pc)
    assert jnp.allclose(jnp.sum(jnp.exp(log_pc), axis=0), 1.0, atol=1e-5)
    inter = _intermediates(net, params, x)  # evaluation: the most likely c, and the entropy is H(c)
    assert jnp.array_equal(inter['fp_iterations'], jnp.argmax(log_pc, axis=0) + min_iters)
    assert jnp.allclose(inter['halt_entropy'], -jnp.sum(jnp.exp(log_pc) * log_pc, axis=0), atol=1e-5)


@pytest.mark.parametrize('min_iters', [1, 3, 7])
def test_c_distribution_by_hand(min_iters):
    # c_log_probs against a loop over c, its terms by decision pass, the decision mask, and sample_c's frequencies.
    T, B = 7, 3
    lp = jax.nn.log_softmax(jax.random.normal(jax.random.PRNGKey(0), (T, B, 2)), axis=-1)
    m = min(min_iters, T)
    want = jnp.full((T, B), -1e9)
    for c in range(m, T + 1):
        want = want.at[c - 1].set(sum((lp[n - 1, :, 1] for n in range(m, c)), jnp.zeros(B)) + (lp[c - 1, :, 0] if c < T else 0.0))
    lp_c = c_log_probs(lp, min_iters)
    assert jnp.allclose(lp_c, want, atol=1e-6) and jnp.allclose(jnp.sum(jnp.exp(lp_c), axis=0), 1.0, atol=1e-5)
    for c in range(m, T + 1):
        cs = jnp.full((B,), c)
        d = decision_log_probs(lp, cs, min_iters)
        assert jnp.allclose(jnp.sum(d, axis=0), lp_c[c - 1], atol=1e-6)
        assert jnp.array_equal(decision_mask(cs, T, min_iters), jnp.full((B,), _k1_path(c, m, T)))
    n = 20000
    c = jax.vmap(lambda k: sample_c(lp, min_iters, k))(jax.random.split(jax.random.PRNGKey(1), n))  # [n, B]
    freq = jnp.stack([jnp.mean(c == k, axis=0) for k in range(1, T + 1)])
    assert jnp.allclose(freq, jnp.exp(lp_c), atol=0.015)


@pytest.mark.parametrize('head_grad', [False, True])
def test_halting_gradient_reaches_trunk_only_with_head_grad(head_grad):
    net = _net(halt_beta_head=True, halt_head_grad=head_grad)  # beta reads h_(n+1), so the trunk is reachable
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 18))
    params = _random_head(net.init(jax.random.PRNGKey(0), x)['params'], scale=0.3)

    def halt_loss(p):
        return jnp.sum(_intermediates(net, p, x, halt_steps=jnp.array([2, 3]))['halt_log_prob'])

    g = jax.grad(halt_loss)(params)
    trunk = sum(float(jnp.sum(jnp.abs(v))) for v in jax.tree_util.tree_leaves(g['core']))
    assert (trunk > 0) == head_grad
    assert float(jnp.sum(jnp.abs(g['halt_beta']['kernel']))) > 0 and float(jnp.abs(g['halt_alpha_raw'])) > 0


@pytest.mark.parametrize('signal', ['kl', 'latent', 'tv'])
def test_lookahead_halts_with_previous_pass(signal):
    # Lookahead: c = 1 is decided on pass 2 and acts with pi_1 (= the output of a 1-pass net); fp_residual is the
    # signal of pass 2.
    net = _net(min_think_iters=1, halt_signal=signal)
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 18))
    eager = _eager(net.init(jax.random.PRNGKey(0), x)['params'])
    inter = _intermediates(net, eager, x)
    assert bool(jnp.all(inter['fp_iterations'] == 1))
    one_pass = _intermediates(FPRMMemHaltActorValue(m=3, n=3, config=CFG, max_think_iters=1, halt_signal=signal), eager, x)
    assert jnp.allclose(inter['logits'], one_pass['logits'], atol=1e-5)
    assert jnp.allclose(inter['fp_residual'], inter['halt_signal_t'][1], atol=1e-6)


@pytest.mark.parametrize('signal', ['kl', 'latent', 'tv'])
def test_head_reads_the_next_signal(signal):
    # The decision at pass n is sigmoid(beta - alpha log S_(n+1)).
    net = _net(min_think_iters=1, halt_signal=signal)
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 18))
    params = _random_head(net.init(jax.random.PRNGKey(0), x)['params'])
    inter = _intermediates(net, params, x)
    if signal == 'kl':
        assert jnp.allclose(inter['halt_signal_t'], inter['policy_kl_t'])
    logit = -3.0 - jax.nn.softplus(0.3) * jnp.log(jnp.maximum(inter['halt_signal_t'][1:], 0.0) + 1e-10)
    assert jnp.allclose(inter['halt_prob'][:-1], jax.nn.sigmoid(logit), atol=1e-5)


def test_latent_signal_is_l2_of_readout_latents():
    # S_i = ||rms_norm(h_i) - rms_norm(h_{i-1})|| with h the readout token's latent: finite and positive (the passes differ);
    # the signal changes only S, not the passes (the same KLs).
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 18))
    net = _net(halt_signal='latent')
    params = net.init(jax.random.PRNGKey(0), x)['params']
    sig = _intermediates(net, params, x)['halt_signal_t']
    assert bool(jnp.all(jnp.isfinite(sig))) and bool(jnp.all(sig > 0))
    # pass 1 compares against the zero initial thought (rms_norm(0) = 0), so S_1 = ||rms_norm(h_1)|| = sqrt(d_model)
    assert jnp.allclose(sig[0], CFG.d_model ** 0.5, rtol=1e-3)
    # kl and latent see the same passes; only S differs
    kl_net = _net(halt_signal='kl')
    assert jnp.allclose(_intermediates(kl_net, params, x)['policy_kl_t'], _intermediates(net, params, x)['policy_kl_t'])


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


def test_value_and_halt_tokens():
    # (halt_signal='latent': with 'kl' S comes from the acting policy, which halt_token does not move.)
    # value_token / halt_token move one readout to the other token: the action stays with readout_token's, the value
    # and the halting head match readout_token='sep'.
    kw = dict(action_head='readout', output_dim_1=4, min_think_iters=1, halt_signal='latent', halt_beta_head=True)
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 18))
    params = _random_head(_net(**kw).init(jax.random.PRNGKey(0), x)['params'])
    out = {r: _intermediates(_net(readout_token=r, **kw), params, x) for r in ('thought', 'sep')}
    split = _intermediates(_net(readout_token='thought', value_token='sep', halt_token='sep', **kw), params, x)
    assert jnp.allclose(split['policy_log_probs_t'], out['thought']['policy_log_probs_t'], atol=1e-6)
    assert jnp.allclose(split['value'], out['sep']['value'], atol=1e-6)
    assert jnp.allclose(split['halt_step_probs'], out['sep']['halt_step_probs'], atol=1e-6)
    assert not jnp.allclose(split['halt_step_probs'], out['thought']['halt_step_probs'], atol=1e-4)


def test_replay_matches_rollout():
    # The loss replays c from the rollout and must recover the rollout's log p(c | s).
    net = _net(min_think_iters=1)
    x = jax.random.normal(jax.random.PRNGKey(1), (6, 18))
    params = net.init(jax.random.PRNGKey(0), x)['params']
    roll = _intermediates(net, params, x, halt_key=jax.random.PRNGKey(3))
    replay = _intermediates(net, params, x, halt_steps=roll['fp_iterations'])
    assert jnp.allclose(roll['halt_log_prob'], replay['halt_log_prob'], atol=1e-5)
    assert bool(jnp.all(roll['halt_log_prob'] < 0)) and bool(jnp.all((roll['fp_iterations'] >= 1) & (roll['fp_iterations'] <= 4)))


def test_halting_batched_jit():
    # Batched jit of the halting sampling (an earlier head's concat was miscompiled by XLA GPU at batch > 1).
    net = FPRMMemHaltActorValue(m=5, n=5, config=FPRMConfig(num_layers=2), max_think_iters=4, min_think_iters=1,
                                output_dim_1=4, action_head='readout')
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


def test_no_halting_acts_with_the_last_pass():
    # halting=False (iso_mem): no halting parameters, c = T for every state, and the same readout as the halting model
    # forced to act at the last pass.
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 18))
    params = _net(halting=False).init(jax.random.PRNGKey(0), x)['params']
    assert not any(k.startswith('halt_') for k in params)
    iso = _intermediates(_net(halting=False), params, x)
    assert bool(jnp.all(iso['fp_iterations'] == 4)) and 'halt_log_prob' not in iso
    head = {k: v for k, v in _net().init(jax.random.PRNGKey(0), x)['params'].items() if k.startswith('halt_')}
    forced = _intermediates(_net(), {**params, **head}, x, halt_steps=jnp.full((3,), 4))
    assert jnp.allclose(iso['logits'], forced['logits'], atol=1e-6) and jnp.allclose(iso['value'], forced['value'], atol=1e-6)


def test_value_pass_halt_reads_the_value_at_c():
    # value_pass='halt': the value comes from the acting pass c (the default 'last' reads pass T for every c).
    net = _net(min_think_iters=1)
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 18))
    params = net.init(jax.random.PRNGKey(0), x)['params']
    c = jnp.array([1, 2, 4])
    kw = dict(halt_steps=c)
    by_c = [_intermediates(FPRMMemHaltActorValue(m=3, n=3, config=CFG, max_think_iters=int(n), min_think_iters=1), params, x)
            ['value'] for n in c]  # pass n = the last pass of a T = n model
    last = _intermediates(net, params, x, **kw)['value']
    inter = _intermediates(_net(min_think_iters=1, value_pass='halt'), params, x, **kw)
    halt = inter['value']
    assert jnp.allclose(halt, inter['value_t'][c - 1, jnp.arange(3)], atol=1e-6)  # value_t [T, B]: every pass's value
    assert jnp.allclose(halt, jnp.array([v[i] for i, v in enumerate(by_c)]), atol=1e-5)
    assert jnp.allclose(last[2], halt[2], atol=1e-6) and not jnp.allclose(last[:2], halt[:2], atol=1e-3)


def test_act_iters_acts_at_c_and_values_at_T():
    # halting=False, act_iters=c < T: the fixed-c actor of a T = c model with the critic of the T model.
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 18))
    params = _net(halting=False).init(jax.random.PRNGKey(0), x)['params']
    fixed_c = _intermediates(FPRMMemHaltActorValue(m=3, n=3, config=CFG, max_think_iters=2, halting=False), params, x)
    full_T = _intermediates(_net(halting=False), params, x)
    mixed = _intermediates(_net(halting=False, act_iters=2), params, x)
    assert bool(jnp.all(mixed['fp_iterations'] == 2))
    assert jnp.allclose(mixed['logits'], fixed_c['logits'], atol=1e-5)
    assert jnp.allclose(mixed['value'], full_T['value'], atol=1e-6)


@pytest.mark.parametrize('readout', ['thought', 'sep'])
def test_readout_latent_sown(readout):
    # readout_latent_t [T, B, H]: the latent pi_n is decoded from; its rms_norm differences are the latent halt signal
    x = jax.random.normal(jax.random.PRNGKey(1), (3, 18))
    net = _net(halt_signal='latent', readout_token=readout)
    params = net.init(jax.random.PRNGKey(0), x)['params']
    inter = _intermediates(net, params, x)
    h = inter['readout_latent_t']
    assert h.shape == (4, 3, CFG.d_model)
    rn = lambda v: rms_norm(v, CFG.rms_norm_eps)
    assert jnp.allclose(jnp.linalg.norm(rn(h[1:]) - rn(h[:-1]), axis=-1), inter['halt_signal_t'][1:], rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize('N', [3, 4])
def test_neighbour_table_matches_env_moves(N):
    # action_head='neighbour' must index the tile each of the env's 4 blank moves slides, and mask the no-op moves.
    from envs.slidingpuzzle_env import SlidingPuzzleEnv, default_config
    from models.fprm_mem_halt import neighbour_table
    env = SlidingPuzzleEnv(config=default_config().copy_and_resolve_references())
    env.m = env.n = N
    idx, ok = neighbour_table(N, N)
    puzzle = jnp.arange(N * N, dtype=jnp.int32).reshape(N, N)
    for b in range(N * N):
        for a in range(4):
            _, target, moved = env.move(puzzle, jnp.array([b // N, b % N]), a)
            assert bool(moved) == ok[b, a] and int(target[0] * N + target[1]) == idx[b, a], (b, a)


@pytest.mark.parametrize('halting', [False, True])
def test_neighbour_head_gathers_the_cell_logits(halting):
    # 'neighbour' logits = the per-cell logits (same params) of the blank's neighbours, off-board moves at _NEG, also
    # with observation normalization (the blank is read from the raw observation).
    from envs.slidingpuzzle_env import SlidingPuzzleEnv, default_config
    from models.fprm_mem_halt import _NEG, neighbour_table
    from utils import running_statistics
    N = 3
    env = SlidingPuzzleEnv(config=default_config())
    obs = jax.vmap(lambda k: env._make_state(k, env.scramble(k, 20)).obs)(jax.random.split(jax.random.PRNGKey(0), 16))
    kw = dict(halting=halting, min_think_iters=1, obs_channels=N * N, stable_softmax=True)
    nbr, cell = _net(action_head='neighbour', output_dim_1=4, **kw), _net(action_head='per_cell', output_dim_1=N * N, **kw)
    params = nbr.init(jax.random.PRNGKey(1), obs)['params']
    norm = running_statistics.update(running_statistics.init_state(obs.shape[-1:]), obs)
    blank = jnp.argmax(obs.reshape(16, N * N, N * N)[..., 0], axis=-1)
    idx, ok = (jnp.asarray(t)[blank] for t in neighbour_table(N, N))
    assert int(ok.sum(-1).min()) >= 2 and int(ok.sum(-1).max()) <= 4
    key = 'policy_log_probs_t' if halting else 'logits'  # by pass [T, B, A]; without halting the last pass's [B, A]
    for n in (None, norm):
        got = jax.nn.log_softmax(_intermediates(nbr, params, obs, normalizer_params=n)[key])
        logits = _intermediates(cell, params, obs, normalizer_params=n)[key]
        want = jax.nn.log_softmax(jnp.where(ok, jnp.take_along_axis(logits, jnp.broadcast_to(idx, logits.shape[:-1] + (4,)), -1), _NEG))
        assert jnp.allclose(got, want, atol=1e-5)
        assert bool(jnp.all(jnp.where(ok, True, jnp.exp(got) == 0.0)))


def _puzzle_obs(n_boards=16):
    from envs.slidingpuzzle_env import SlidingPuzzleEnv, default_config
    env = SlidingPuzzleEnv(config=default_config())
    return jax.vmap(lambda k: env._make_state(k, env.scramble(k, 20)).obs)(jax.random.split(jax.random.PRNGKey(0), n_boards))


def test_blank_head_reads_the_blank_cell():
    # 'blank' logits = move i's Dense(1) on the i-th quarter of the blank cell's latent, each quarter rms-normed on its
    # own, unmasked, also with observation normalization (the blank is read from the raw observation). The cell latents
    # come from a per_cell net with an identity kernel (rms_norm is scale-invariant, so its full-latent norm drops out).
    from models.fprm import rms_norm
    from utils import running_statistics
    N, H = 3, CFG.d_model
    obs = _puzzle_obs()
    kw = dict(halting=False, obs_channels=N * N)
    blk, cell = _net(action_head='blank', output_dim_1=4, **kw), _net(action_head='per_cell', output_dim_1=H * N * N, **kw)
    params = blk.init(jax.random.PRNGKey(1), obs)['params']
    k, b = params['actor_head']['kernel'], jax.random.normal(jax.random.PRNGKey(2), (4,))  # k: [H / 4, 4]
    assert k.shape == (H // 4, 4)
    params = _with(params, 'actor_head', bias=b)
    cell_params = _with(params, 'actor_head', kernel=jnp.eye(H), bias=jnp.zeros(H))
    norm = running_statistics.update(running_statistics.init_state(obs.shape[-1:]), obs)
    blank = jnp.argmax(obs.reshape(16, N * N, N * N)[..., 0], axis=-1)
    for n in (None, norm):
        got = _intermediates(blk, params, obs, normalizer_params=n)['logits']
        latents = _intermediates(cell, cell_params, obs, normalizer_params=n)['logits'].reshape(16, N * N, H)
        quarters = latents[jnp.arange(16), blank].reshape(16, 4, H // 4)
        want = jnp.stack([rms_norm(quarters[:, i], CFG.rms_norm_eps) @ k[:, i] + b[i] for i in range(4)], axis=-1)
        assert jnp.allclose(got, want, atol=1e-4)


@pytest.mark.parametrize('action_head', ['neighbour', 'blank'])
def test_no_cell_geometry_drops_the_geometry_features(action_head):
    # cell_geometry=False = cell_geometry=True with the cell projection's 6 geometry rows zeroed.
    N = 3
    obs = _puzzle_obs()
    kw = dict(halting=False, obs_channels=N * N, action_head=action_head, output_dim_1=4)
    geo, nogeo = _net(**kw), _net(cell_geometry=False, **kw)
    params = nogeo.init(jax.random.PRNGKey(1), obs)['params']
    assert params['cell_projection']['kernel'].shape[0] == N * N
    k = params['cell_projection']['kernel']
    params_geo = {**params, 'cell_projection': {**params['cell_projection'],
                                                'kernel': jnp.concatenate([k, jnp.zeros((6, k.shape[1]), k.dtype)])}}
    got = _intermediates(nogeo, params, obs)['logits']
    want = _intermediates(geo, params_geo, obs)['logits']
    assert jnp.allclose(got, want, atol=1e-5)
