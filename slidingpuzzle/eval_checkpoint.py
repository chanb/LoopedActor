"""Evaluate a sliding puzzle PPO checkpoint per scramble length.

Sliding puzzle instances are procedural (no level bank), so for every scramble length K in `--scramble_moves`
(random walk of the blank from the solved board, as in training) the script draws `--levels` scrambles from fixed
seeds and rolls out the deterministic policy (argmax) for `--episode_length` steps. Reports success rate, mean
realized think iterations, KL-convergence rate and mean episode length per scramble length. Scrambles that are
already solved at reset (possible for short random walks) are excluded and counted separately.

Examples (repo root)
  python slidingpuzzle/eval_checkpoint.py --ckpt exp/slidingpuzzle/looped_s1/checkpoints/<run>/params_101.pkl
  python slidingpuzzle/eval_checkpoint.py --ckpt <run>/params_101.pkl --max_iters 48   # test-time cap raise
  python slidingpuzzle/eval_checkpoint.py --ckpt <run>/params_101.pkl --arch multi_block --num_blocks 16 --scramble_moves 100 200 400
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
from envs.slidingpuzzle_env import SlidingPuzzleEnv, default_config  # noqa: E402
from models.fprm import FPRMConfig  # noqa: E402
from models.fprm_thinker import FPRMThinkerActorValue  # noqa: E402
from models.fprm_cot_thinker import FPRMCoTThinkerActorValue  # noqa: E402
from models.fprm_perceiver import FPRMPerceiverActorValue  # noqa: E402
from models.transformer_baseline import TransformerActorValue  # noqa: E402


def build_module(args, N):
    cfg = FPRMConfig(d_model=args.d_model, num_heads=4, num_layers=args.num_layers)
    kw = dict(m=N, n=N, config=cfg, output_dim_1=4, obs_channels=N * N, action_head='readout',
              tokenizer=args.tokenizer, tokenizer_conv_features=args.tokenizer_conv_features)
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


def evaluate(env, mod, policy_vars, norm, K, n_levels, T, B, seed):
    @jax.jit
    def act(obs):
        (pi, _), inter = mod.apply(policy_vars, obs, norm, mutable=['intermediates'])
        I = inter['intermediates']
        conv = I['fp_converged'][0] if 'fp_converged' in I else jnp.zeros(obs.shape[0], bool)
        return pi.mode(), I['fp_iterations'][0], conv

    reset_v = jax.jit(jax.vmap(lambda k: env._make_state(k, env.scramble(k, K))))
    step_v = jax.jit(jax.vmap(env.step))
    keys = jax.random.split(jax.random.PRNGKey(seed * 1000 + K), n_levels)
    succ = np.zeros(n_levels, bool); length = np.full(n_levels, T, np.int32); pre_solved = np.zeros(n_levels, bool)
    think_sum = conv_sum = 0.0; steps = 0
    for s0 in range(0, n_levels, B):
        st = reset_v(keys[s0:s0 + B])
        pre = np.asarray(jnp.all(st.data.puzzle == env.solved, axis=(1, 2)))
        pre_solved[s0:s0 + len(pre)] = pre
        done = pre.copy()
        for t in range(T):
            if done.all():
                break
            a, its, cv = act(st.obs)
            alive = ~done
            think_sum += float(np.asarray(its)[alive].sum()); conv_sum += float(np.asarray(cv)[alive].sum())
            steps += int(alive.sum())
            st = step_v(st, a)
            d_now = np.asarray(st.done) > 0.5; newly = d_now & ~done
            succ[s0 + np.flatnonzero(newly)] = True; length[s0 + np.flatnonzero(newly)] = t + 1; done |= d_now
    keep = ~pre_solved
    s, L = succ[keep], length[keep]
    return dict(levels=int(keep.sum()), already_solved=int(pre_solved.sum()), success=float(s.mean()) if keep.any() else None,
                mean_len=float(L.mean()) if keep.any() else None, think=think_sum / max(steps, 1),
                converged=conv_sum / max(steps, 1), solved_len=float(L[s].mean()) if s.any() else None)


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
    ap.add_argument('--tokenizer', default='cell', choices=['cell', 'flat', 'conv'], help='the checkpoint\'s --tokenizer')
    ap.add_argument('--tokenizer_conv_features', type=int, default=64, help='the checkpoint\'s --tokenizer_conv_features')
    ap.add_argument('--grid', type=int, default=3, help='board size N (N x N) the checkpoint was trained on')
    ap.add_argument('--episode_length', type=int, default=40)
    ap.add_argument('--scramble_moves', type=int, nargs='+', default=[10, 50, 100, 200, 400])
    ap.add_argument('--levels', type=int, default=2000, help='scrambles per scramble length')
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default=None, help='JSON path (default: next to the checkpoint, eval_<name>[_cap<N>].json)')
    args = ap.parse_args()

    cfg = default_config(); cfg.grid_size = args.grid
    env = SlidingPuzzleEnv(cfg)
    mod = build_module(args, args.grid)
    with open(args.ckpt, 'rb') as f:
        policy_vars, norm = pickle.load(f)

    results = {}
    for K in args.scramble_moves:
        r = results[K] = evaluate(env, mod, policy_vars, norm, K, args.levels, args.episode_length, args.batch, args.seed)
        fmt = lambda v, f: 'n/a' if v is None else format(v, f)
        print(f'{args.ckpt} scramble {K:4d} n={r["levels"]:5d} (+{r["already_solved"]} pre-solved) success {fmt(r["success"], ".3f")} '
              f'think {r["think"]:.2f} conv {r["converged"]:.3f} len {fmt(r["mean_len"], ".1f")}', flush=True)

    stem = os.path.splitext(os.path.basename(args.ckpt))[0]
    tag = f'_cap{args.max_iters}' if args.arch in ('fprm', 'fprm_cot', 'fprm_perceiver') and args.max_iters != 16 else ''
    out = args.out or os.path.join(os.path.dirname(args.ckpt), f'eval_{stem}{tag}.json')
    json.dump(dict(ckpt=args.ckpt, arch=args.arch, grid=args.grid, episode_length=args.episode_length,
                   max_iters=args.max_iters if args.arch in ('fprm', 'fprm_cot', 'fprm_perceiver') else None,
                   results={str(K): r for K, r in results.items()}), open(out, 'w'), indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
