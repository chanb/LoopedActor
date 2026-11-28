"""Evaluate a Lights Out PPO checkpoint per goal distance.

Lights Out instances are procedural (no level bank), so for every goal distance `dist` (number of distinct presses
that generated the goal, 1 .. M*N-1) the script draws `--levels_per_dist` instances from fixed seeds
(`LightsOutEnv.reset_with_dist`) and rolls out the deterministic policy (argmax) for `--episode_length` steps below the
difficulty threshold and `--eval_episode_length` steps at or above it (the training budgets of train.sh).
Reports per-distance success rate, mean realized think iterations, KL-convergence rate and mean episode length,
plus aggregates over the training distances [1, threshold) and the evaluation distances [threshold, M*N).

Examples (repo root)
  python lightsout/eval_checkpoint.py --ckpt exp/lightsout/iso_mem_s1/checkpoints/<run>/params_101.pkl
  python lightsout/eval_checkpoint.py --ckpt exp/lightsout/iso_mem_s1/checkpoints/<run>/params_101.pkl --act_iters 48:u   # test-time cap raise
  python lightsout/eval_checkpoint.py --ckpt <run>/params_101.pkl --arch multi_block --num_blocks 16 --grid 3x3
"""
import argparse
import json
import os
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.lightsout_env import LightsOutEnv, default_config  # noqa: E402
from models.fprm import FPRMConfig  # noqa: E402
from utils import running_statistics  # noqa: E402
from utils.eval_model import add_model_args, build_mem, cap_tag, think_cap  # noqa: E402
from models.transformer_baseline import TransformerActorValue  # noqa: E402


def build_module(args, m, n):
    cfg = FPRMConfig(d_model=args.d_model, num_heads=4, num_layers=args.num_layers,
                      mlp_type=args.mlp_type, mlp_round_to=args.mlp_round_to, compute_dtype=args.compute_dtype)
    kw = dict(m=m, n=n, config=cfg, output_dim_1=m * n, obs_channels=0, action_head='per_cell')
    if args.arch == 'fprm_mem':
        return build_mem(args, **kw)
    return TransformerActorValue(num_blocks=1 if args.arch == 'single_block' else args.num_blocks, **kw)


def evaluate(env, mod, policy_vars, norm, dist, n_levels, T, B, seed, halt_mode='mode', halt_fixed=0):
    @jax.jit
    def act(obs, goal, key):
        x = jnp.concatenate([obs, goal], axis=-1)
        # fprm_mem C:v, sigmoid: the most likely c (mode), a sampled one (sample) or a fixed c for every state (fixed).
        kw = {}
        if halt_mode == 'sample':
            kw = {'halt_key': key}
        elif halt_mode == 'fixed':
            kw = {'halt_steps': jnp.full((obs.shape[0],), halt_fixed, jnp.int32)}
        (pi, _), inter = mod.apply(policy_vars, x, norm, mutable=['intermediates'], **kw)
        I = inter['intermediates']
        conv = I['fp_converged'][0] if 'fp_converged' in I else jnp.zeros(obs.shape[0], bool)
        return pi.mode(), I['fp_iterations'][0], conv

    reset_v = jax.jit(jax.vmap(env.reset_with_dist, in_axes=(0, None)))
    step_v = jax.jit(jax.vmap(env.step))
    keys = jax.random.split(jax.random.PRNGKey(seed * 1000 + dist), n_levels)
    halt_key = jax.random.PRNGKey(seed * 1000 + dist + 7919)
    succ = np.zeros(n_levels, bool); length = np.full(n_levels, T, np.int32)
    think_sum = conv_sum = 0.0; steps = 0
    for s0 in range(0, n_levels, B):
        st = reset_v(keys[s0:s0 + B], dist); done = np.zeros(st.obs.shape[0], bool)
        for t in range(T):
            halt_key, k = jax.random.split(halt_key)
            a, its, cv = act(st.obs, st.info['target_goal'], k)
            alive = ~done
            think_sum += float(np.asarray(its)[alive].sum()); conv_sum += float(np.asarray(cv)[alive].sum())
            steps += int(alive.sum())
            st = step_v(st, a)
            d_now = np.asarray(st.done) > 0.5; newly = d_now & ~done
            s_now = np.asarray(st.metrics['success']) > 0.5
            succ[s0 + np.flatnonzero(newly & s_now)] = True; length[s0 + np.flatnonzero(newly)] = t + 1; done |= d_now
            if done.all():
                break
    return dict(levels=int(n_levels), success=float(succ.mean()), mean_len=float(length.mean()),
                think=think_sum / max(steps, 1), converged=conv_sum / max(steps, 1),
                solved_len=float(length[succ].mean()) if succ.any() else None)


def aggregate(per_dist, dists):
    rows = [per_dist[d] for d in dists if d in per_dist]
    if not rows:
        return None
    w = np.array([r['levels'] for r in rows], float)
    return {k: float(np.average([r[k] for r in rows], weights=w)) for k in ('success', 'mean_len', 'think', 'converged')}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True, help='params_<k>.pkl written by ppo.py: (policy params, normalizer params)')
    add_model_args(ap)
    ap.add_argument('--halt_mode', default='mode', choices=['mode', 'sample', 'fixed'],
                    help='fprm_mem C:v: think time c of the most likely halting path (the training eval), of a sampled path '
                         '(sigmoid only), or --halt_fixed for every state')
    ap.add_argument('--halt_fixed', type=int, default=0, help='fprm_mem C:v, --halt_mode fixed: the think time c (1 .. C)')
    ap.add_argument('--num_blocks', type=int, default=16, help='multi_block: number of untied cores')
    ap.add_argument('--d_model', type=int, default=128)
    ap.add_argument('--num_layers', type=int, default=2)
    ap.add_argument('--mlp_type', default='swiglu', choices=['swiglu', 'gelu'], help='the checkpoint\'s --mlp_type')
    ap.add_argument('--mlp_round_to', type=int, default=256, help='the checkpoint\'s --mlp_round_to')
    ap.add_argument('--compute_dtype', default='float32', choices=['float32', 'float16', 'bfloat16'], help='block computation dtype (any checkpoint)')
    ap.add_argument('--grid', default='5x4', help='board MxN the checkpoint was trained on')
    ap.add_argument('--difficulty_threshold', type=float, default=0.5, help='splits distances into train / eval ranges')
    ap.add_argument('--episode_length', type=int, default=0,
                    help='steps per episode for distances below the threshold (0 = the threshold, the train.sh default; '
                         'checkpoints trained before that default used M*N)')
    ap.add_argument('--eval_episode_length', type=int, default=0,
                    help='steps per episode for distances at or above the threshold (0 = M*N, the train.sh default)')
    ap.add_argument('--dists', type=int, nargs='+', default=None, help='goal distances to evaluate (default 1 .. M*N-1)')
    ap.add_argument('--levels_per_dist', type=int, default=1000)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None, help='JSON path (default: next to the checkpoint, eval_<name>[_cap<N>].json)')
    args = ap.parse_args()

    m, n = (int(v) for v in args.grid.split('x'))
    cfg = default_config(); cfg.m, cfg.n = m, n; cfg.difficulty_threshold = args.difficulty_threshold
    env = LightsOutEnv(cfg)
    T_train = args.episode_length or env.threshold
    T_eval = args.eval_episode_length or m * n
    dists = args.dists or list(range(1, m * n))
    mod = build_module(args, m, n)
    with open(args.ckpt, 'rb') as f:
        policy_vars, norm = pickle.load(f)
    if running_statistics.is_identity(norm):
        norm = None  # trained without observation normalization (--normalize_observations off)

    per_dist = {}
    for d in dists:
        T = T_train if d < env.threshold else T_eval
        per_dist[d] = evaluate(env, mod, policy_vars, norm, d, args.levels_per_dist, T, args.batch, args.seed,
                               args.halt_mode, args.halt_fixed)
        r = per_dist[d]
        print(f'{args.ckpt} dist {d:3d} n={r["levels"]:5d} success {r["success"]:.3f} think {r["think"]:.2f} '
              f'conv {r["converged"]:.3f} len {r["mean_len"]:.1f}', flush=True)
    summary = {'train_dists': aggregate(per_dist, range(1, env.threshold)),
               'eval_dists': aggregate(per_dist, range(env.threshold, m * n))}
    for name, s in summary.items():
        if s is not None:
            print(f'{name:11s} success {s["success"]:.3f} think {s["think"]:.2f} conv {s["converged"]:.3f} len {s["mean_len"]:.1f}')

    stem = os.path.splitext(os.path.basename(args.ckpt))[0]
    tag = cap_tag(args)
    if args.act_iters.endswith(':v') and args.halt_mode != 'mode':
        tag += f'_{args.halt_mode}' + (str(args.halt_fixed) if args.halt_mode == 'fixed' else '')
    out = args.out or os.path.join(os.path.dirname(args.ckpt), f'eval_{stem}{tag}.json')
    json.dump(dict(ckpt=args.ckpt, arch=args.arch, grid=args.grid, episode_length=T_train, eval_episode_length=T_eval,
                   act_iters=args.act_iters if think_cap(args) else None,
                   threshold=env.threshold, results={str(d): r for d, r in per_dist.items()}, summary=summary),
              open(out, 'w'), indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
