"""Evaluate a Boxoban PPO checkpoint on whole level banks, each level exactly once.

Deterministic policy (argmax), 120-step episodes, levels rolled out in parallel with the JAX env.
Reports success rate, mean realized think iterations, KL-convergence rate and mean episode length per
split, and optionally stores the full trajectories (actions, per-step think iterations, convergence
flags) for later analysis.

Examples (repo root)
  python sokoban/eval_checkpoint.py --ckpt exp/sokoban/looped_s1/checkpoints/<run>/params_101.pkl
  python sokoban/eval_checkpoint.py --ckpt exp/sokoban/looped_s1/checkpoints/<run>/params_101.pkl --max_iters 48   # test-time cap raise
  python sokoban/eval_checkpoint.py --ckpt <run>/params_101.pkl --arch multi_block --num_blocks 16 --splits unfiltered_valid unfiltered_test
  python sokoban/eval_checkpoint.py --ckpt exp/sokoban/cot_s1/checkpoints/<run>/params_101.pkl --arch fprm_cot
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
from envs.sokoban_env import SokobanData, SokobanEnv, default_config  # noqa: E402
from envs.utils import State  # noqa: E402
from models.fprm import FPRMConfig  # noqa: E402
from utils import running_statistics  # noqa: E402
from models.fprm_thinker import FPRMThinkerActorValue  # noqa: E402
from models.fprm_cot_thinker import FPRMCoTThinkerActorValue  # noqa: E402
from models.fprm_perceiver import FPRMPerceiverActorValue  # noqa: E402
from models.fprm_mem import FPRMMemActorValue  # noqa: E402
from models.fprm_rim import FPRMRiMActorValue  # noqa: E402
from models.transformer_baseline import TransformerActorValue  # noqa: E402


def build_module(args):
    cfg = FPRMConfig(d_model=args.d_model, num_heads=4, num_layers=args.num_layers,
                      mlp_type=args.mlp_type, mlp_round_to=args.mlp_round_to, compute_dtype=args.compute_dtype)
    kw = dict(m=10, n=10, config=cfg, output_dim_1=4, obs_channels=4, action_head='readout',
              tokenizer=args.tokenizer, tokenizer_conv_features=args.tokenizer_conv_features)
    if args.arch == 'fprm':
        return FPRMThinkerActorValue(max_think_iters=args.max_iters, min_think_iters=2, halt_criterion='kl',
                                     halt_kl=args.halt_kl, **kw)
    if args.arch == 'fprm_cot':
        return FPRMCoTThinkerActorValue(max_think_iters=args.max_iters, min_think_iters=2, halt_kl=args.halt_kl,
                                        thought_norm=args.thought_norm, input_grid_conv=args.input_grid_conv, input_grid_conv_depth=args.input_grid_conv_depth, **kw)
    if args.arch == 'fprm_rim':
        return FPRMRiMActorValue(max_think_iters=args.max_iters, min_think_iters=2, halt_criterion=args.rim_halt,
                                 halt_residual_thresh=args.halt_residual_thresh, halt_kl=args.halt_kl, **{k: v for k, v in kw.items() if k not in ('tokenizer', 'tokenizer_conv_features')})
    if args.arch == 'fprm_mem':
        return FPRMMemActorValue(max_think_iters=args.max_iters, min_think_iters=2, halt_kl=args.halt_kl,
                                 thought_norm=args.thought_norm, **kw)
    if args.arch == 'fprm_perceiver':
        return FPRMPerceiverActorValue(max_think_iters=args.max_iters, min_think_iters=2, halt_kl=args.halt_kl,
                                       thought_norm=args.thought_norm, input_grid_conv=args.input_grid_conv, input_grid_conv_depth=args.input_grid_conv_depth,
                                       step_grid_conv=args.perceiver_step_grid_conv,
                                       num_initial_latents=args.perceiver_num_initial_latents,
                                       cell_encoder_iters=args.perceiver_cell_encoder_iters, **kw)
    return TransformerActorValue(num_blocks=1 if args.arch == 'single_block' else args.num_blocks, **kw)


def evaluate(mod, policy_vars, norm, split, max_levels, T=120, B=512, trajectories=False):
    ecfg = default_config(); ecfg.train_split = split; ecfg.eval_split = split
    env = SokobanEnv(ecfg)
    n_lv = env.num_eval_levels if max_levels <= 0 else min(max_levels, env.num_eval_levels)

    @jax.jit
    def act(obs):
        (dist, _), inter = mod.apply(policy_vars, obs, norm, mutable=['intermediates'])
        I = inter['intermediates']
        conv = I['fp_converged'][0] if 'fp_converged' in I else jnp.zeros_like(I['fp_iterations'][0], dtype=bool)
        return dist.mode(), I['fp_iterations'][0], conv

    def reset_one(i):
        bank = env.eval_bank
        d = SokobanData(walls=bank['walls'][i], boxes=bank['boxes'][i], targets=bank['targets'][i], player=bank['player'][i],
                        level=i, on_target=jnp.sum(bank['boxes'][i] & bank['targets'][i]).astype(jnp.int32))
        return State(data=d, obs=env._obs(d), reward=jnp.float32(0), done=jnp.float32(0),
                     metrics={'success': 0.0, 'reward': 0.0, 'boxes_on_target': 0.0},
                     info={'rng': jax.random.PRNGKey(0), 'target_goal': jnp.zeros((0,), jnp.float32)})
    reset_v = jax.jit(jax.vmap(reset_one)); step_v = jax.jit(jax.vmap(env.step))
    succ = np.zeros(n_lv, bool); length = np.full(n_lv, T, np.int32)
    A = np.full((n_lv, T), -1, np.int8); TH = np.zeros((n_lv, T), np.int16); CV = np.zeros((n_lv, T), bool)
    for s0 in range(0, n_lv, B):
        idx = jnp.arange(s0, min(s0 + B, n_lv)); st = reset_v(idx); done = np.zeros(len(idx), bool)
        for t in range(T):
            a, its, cv = act(st.obs)
            alive = ~done; rows = s0 + np.flatnonzero(alive)
            A[rows, t] = np.asarray(a)[alive]; TH[rows, t] = np.asarray(its)[alive]; CV[rows, t] = np.asarray(cv)[alive]
            st = step_v(st, a)
            d_now = np.asarray(st.done) > 0.5; newly = d_now & ~done
            succ[s0 + np.flatnonzero(newly)] = True; length[s0 + np.flatnonzero(newly)] = t + 1; done |= d_now
            if done.all(): break
    m = np.arange(T)[None] < length[:, None]
    res = dict(levels=int(n_lv), success=float(succ.mean()), mean_len=float(length.mean()), think=float(TH[m].mean()),
               converged=float(CV[m].mean()), solved_len=float(length[succ].mean()) if succ.any() else None)
    traj = dict(success=succ, length=length, actions=A, think=TH, conv=CV) if trajectories else None
    return res, traj


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True, help='params_<k>.pkl written by ppo.py: (policy params, normalizer params)')
    ap.add_argument('--arch', default='fprm', choices=['fprm', 'fprm_cot', 'fprm_perceiver', 'fprm_mem', 'fprm_rim', 'multi_block', 'single_block'])
    ap.add_argument('--max_iters', type=int, default=16, help='fprm / fprm_cot / fprm_perceiver: think-iteration cap at evaluation (training cap 16)')
    ap.add_argument('--halt_kl', type=float, default=1e-3,
                    help='halt once KL between consecutive policies <= this (0 = only at a fixed point, < 0 = never, e.g. -1 for iso_* checkpoints)')
    ap.add_argument('--rim_halt', default='latent', choices=['latent', 'kl'], help='fprm_rim: halting rule (latent = SEP convergence)')
    ap.add_argument('--halt_residual_thresh', type=float, default=0.1, help='fprm_rim, --rim_halt latent: the checkpoint\'s threshold')
    ap.add_argument('--thought_norm', action='store_true', help='fprm_cot / fprm_perceiver: the checkpoint was trained with --cot_thought_norm')
    ap.add_argument('--input_grid_conv', action='store_true', help='fprm_cot / fprm_perceiver: the checkpoint was trained with --input_grid_conv')
    ap.add_argument('--input_grid_conv_depth', type=int, default=1, help='fprm_cot / fprm_perceiver: the checkpoint\'s --input_grid_conv_depth')
    ap.add_argument('--perceiver_step_grid_conv', action='store_true', help='fprm_perceiver: the checkpoint was trained with --perceiver_step_grid_conv')
    ap.add_argument('--perceiver_num_initial_latents', type=int, default=1, help='fprm_perceiver: the checkpoint\'s --perceiver_num_initial_latents')
    ap.add_argument('--perceiver_cell_encoder_iters', type=int, default=0, help='fprm_perceiver: the checkpoint\'s --perceiver_cell_encoder_iters')
    ap.add_argument('--num_blocks', type=int, default=16, help='multi_block: number of untied cores')
    ap.add_argument('--d_model', type=int, default=128)
    ap.add_argument('--tokenizer', default='cell', choices=['cell', 'flat', 'conv'], help='the checkpoint\'s --tokenizer')
    ap.add_argument('--tokenizer_conv_features', type=int, default=64, help='the checkpoint\'s --tokenizer_conv_features')
    ap.add_argument('--num_layers', type=int, default=2)
    ap.add_argument('--mlp_type', default='swiglu', choices=['swiglu', 'gelu'], help='the checkpoint\'s --mlp_type')
    ap.add_argument('--mlp_round_to', type=int, default=256, help='the checkpoint\'s --mlp_round_to')
    ap.add_argument('--compute_dtype', default='float32', choices=['float32', 'float16', 'bfloat16'], help='block computation dtype (any checkpoint)')
    ap.add_argument('--splits', nargs='+', default=['unfiltered_valid', 'unfiltered_test'])
    ap.add_argument('--max_levels', type=int, default=2000, help='levels per split (0 = whole bank)')
    ap.add_argument('--out', default=None, help='JSON path (default: next to the checkpoint, eval_<name>[_cap<N>].json)')
    ap.add_argument('--save_trajectories', action='store_true', help='also write <out stem>_<split>.npz with per-step actions / think / conv')
    args = ap.parse_args()
    mod = build_module(args)
    with open(args.ckpt, 'rb') as f:
        policy_vars, norm = pickle.load(f)
    if running_statistics.is_identity(norm):
        norm = None  # trained without observation normalization (--normalize_observations off)
    stem = os.path.splitext(os.path.basename(args.ckpt))[0]
    tag = f'_cap{args.max_iters}' if args.arch in ('fprm', 'fprm_cot', 'fprm_perceiver', 'fprm_mem', 'fprm_rim') and args.max_iters != 16 else ''
    out = args.out or os.path.join(os.path.dirname(args.ckpt), f'eval_{stem}{tag}.json')
    results = {}
    for split in args.splits:
        res, traj = evaluate(mod, policy_vars, norm, split, args.max_levels, trajectories=args.save_trajectories)
        results[split] = res
        print(f'{args.ckpt} {split:16s} n={res["levels"]:5d} success {res["success"]:.3f} think {res["think"]:.2f} conv {res["converged"]:.3f} len {res["mean_len"]:.1f}', flush=True)
        if traj is not None:
            np.savez_compressed(os.path.splitext(out)[0] + f'_{split}.npz', **traj)
    json.dump(dict(ckpt=args.ckpt, arch=args.arch, max_iters=args.max_iters, results=results), open(out, 'w'), indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
