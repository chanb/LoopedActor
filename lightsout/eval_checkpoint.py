"""Evaluate a Lights Out PPO checkpoint per goal distance.

Lights Out instances are procedural (no level bank), so for every goal distance `dist` (number of distinct presses
that generated the goal, 1 .. M*N-1) the script draws `--levels_per_dist` instances from fixed seeds
(`LightsOutEnv.reset_with_dist`) and rolls out the deterministic policy (argmax) for `--episode_length` steps.
Reports per-distance success rate, mean realized think iterations, KL-convergence rate and mean episode length,
plus aggregates over the training distances [1, threshold) and the evaluation distances [threshold, M*N).

Examples (repo root)
  python lightsout/eval_checkpoint.py --ckpt exp/lightsout/looped_s1/checkpoints/<run>/params_101.pkl
  python lightsout/eval_checkpoint.py --ckpt exp/lightsout/looped_s1/checkpoints/<run>/params_101.pkl --max_iters 48   # test-time cap raise
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
from models.fprm_thinker import FPRMThinkerActorValue  # noqa: E402
from models.fprm_cot_thinker import FPRMCoTThinkerActorValue  # noqa: E402
from models.fprm_perceiver import FPRMPerceiverActorValue  # noqa: E402
from models.transformer_baseline import TransformerActorValue  # noqa: E402


def build_module(args, m, n):
    cfg = FPRMConfig(d_model=args.d_model, num_heads=4, num_layers=args.num_layers)
    kw = dict(m=m, n=n, config=cfg, output_dim_1=m * n, obs_channels=0, action_head='per_cell')
    if args.arch == 'fprm':
        return FPRMThinkerActorValue(max_think_iters=args.max_iters, min_think_iters=2, halt_criterion='kl',
                                     halt_kl=args.halt_kl, **kw)
    if args.arch == 'fprm_cot':
        return FPRMCoTThinkerActorValue(max_think_iters=args.max_iters, min_think_iters=2, halt_kl=args.halt_kl,
                                        thought_norm=args.thought_norm, input_grid_conv=args.input_grid_conv, **kw)
    if args.arch == 'fprm_perceiver':
        return FPRMPerceiverActorValue(max_think_iters=args.max_iters, min_think_iters=2, halt_kl=args.halt_kl,
                                       thought_norm=args.thought_norm, input_grid_conv=args.input_grid_conv, **kw)
    return TransformerActorValue(num_blocks=1 if args.arch == 'single_block' else args.num_blocks, **kw)


def evaluate(env, mod, policy_vars, norm, dist, n_levels, T, B, seed):
    @jax.jit
    def act(obs, goal):
        x = jnp.concatenate([obs, goal], axis=-1)
        (pi, _), inter = mod.apply(policy_vars, x, norm, mutable=['intermediates'])
        I = inter['intermediates']
        conv = I['fp_converged'][0] if 'fp_converged' in I else jnp.zeros(obs.shape[0], bool)
        return pi.mode(), I['fp_iterations'][0], conv

    reset_v = jax.jit(jax.vmap(env.reset_with_dist, in_axes=(0, None)))
    step_v = jax.jit(jax.vmap(env.step))
    keys = jax.random.split(jax.random.PRNGKey(seed * 1000 + dist), n_levels)
    succ = np.zeros(n_levels, bool); length = np.full(n_levels, T, np.int32)
    think_sum = conv_sum = 0.0; steps = 0
    for s0 in range(0, n_levels, B):
        st = reset_v(keys[s0:s0 + B], dist); done = np.zeros(st.obs.shape[0], bool)
        for t in range(T):
            a, its, cv = act(st.obs, st.info['target_goal'])
            alive = ~done
            think_sum += float(np.asarray(its)[alive].sum()); conv_sum += float(np.asarray(cv)[alive].sum())
            steps += int(alive.sum())
            st = step_v(st, a)
            d_now = np.asarray(st.done) > 0.5; newly = d_now & ~done
            succ[s0 + np.flatnonzero(newly)] = True; length[s0 + np.flatnonzero(newly)] = t + 1; done |= d_now
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
    ap.add_argument('--arch', default='fprm', choices=['fprm', 'fprm_cot', 'fprm_perceiver', 'multi_block', 'single_block'])
    ap.add_argument('--max_iters', type=int, default=16, help='fprm / fprm_cot / fprm_perceiver: think-iteration cap at evaluation (training cap 16)')
    ap.add_argument('--halt_kl', type=float, default=1e-3)
    ap.add_argument('--thought_norm', action='store_true', help='fprm_cot / fprm_perceiver: the checkpoint was trained with --cot_thought_norm')
    ap.add_argument('--input_grid_conv', action='store_true', help='fprm_cot / fprm_perceiver: the checkpoint was trained with --input_grid_conv')
    ap.add_argument('--num_blocks', type=int, default=16, help='multi_block: number of untied cores')
    ap.add_argument('--d_model', type=int, default=128)
    ap.add_argument('--num_layers', type=int, default=2)
    ap.add_argument('--grid', default='5x4', help='board MxN the checkpoint was trained on')
    ap.add_argument('--difficulty_threshold', type=float, default=0.5, help='splits distances into train / eval ranges')
    ap.add_argument('--episode_length', type=int, default=0, help='steps per episode (0 = M*N, the training default)')
    ap.add_argument('--dists', type=int, nargs='+', default=None, help='goal distances to evaluate (default 1 .. M*N-1)')
    ap.add_argument('--levels_per_dist', type=int, default=1000)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None, help='JSON path (default: next to the checkpoint, eval_<name>[_cap<N>].json)')
    args = ap.parse_args()

    m, n = (int(v) for v in args.grid.split('x'))
    cfg = default_config(); cfg.m, cfg.n = m, n; cfg.difficulty_threshold = args.difficulty_threshold
    env = LightsOutEnv(cfg)
    T = args.episode_length or m * n
    dists = args.dists or list(range(1, m * n))
    mod = build_module(args, m, n)
    with open(args.ckpt, 'rb') as f:
        policy_vars, norm = pickle.load(f)

    per_dist = {}
    for d in dists:
        per_dist[d] = evaluate(env, mod, policy_vars, norm, d, args.levels_per_dist, T, args.batch, args.seed)
        r = per_dist[d]
        print(f'{args.ckpt} dist {d:3d} n={r["levels"]:5d} success {r["success"]:.3f} think {r["think"]:.2f} '
              f'conv {r["converged"]:.3f} len {r["mean_len"]:.1f}', flush=True)
    summary = {'train_dists': aggregate(per_dist, range(1, env.threshold)),
               'eval_dists': aggregate(per_dist, range(env.threshold, m * n))}
    for name, s in summary.items():
        if s is not None:
            print(f'{name:11s} success {s["success"]:.3f} think {s["think"]:.2f} conv {s["converged"]:.3f} len {s["mean_len"]:.1f}')

    stem = os.path.splitext(os.path.basename(args.ckpt))[0]
    tag = f'_cap{args.max_iters}' if args.arch in ('fprm', 'fprm_cot', 'fprm_perceiver') and args.max_iters != 16 else ''
    out = args.out or os.path.join(os.path.dirname(args.ckpt), f'eval_{stem}{tag}.json')
    json.dump(dict(ckpt=args.ckpt, arch=args.arch, grid=args.grid, episode_length=T,
                   max_iters=args.max_iters if args.arch in ('fprm', 'fprm_cot', 'fprm_perceiver') else None,
                   threshold=env.threshold, results={str(d): r for d, r in per_dist.items()}, summary=summary),
              open(out, 'w'), indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
