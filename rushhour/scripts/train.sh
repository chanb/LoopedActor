#!/bin/bash
# Train one arm of the Rush Hour study.
#   rushhour/scripts/train.sh <arm> <seed> [out_dir] [device]
#   arm: looped | iso_flops | iso_param | iso_loop
# Optional environment variables:
#   MAX_THINK_ITERS=N   think-iteration cap for looped / iso_loop (default 16)
#   NUM_BLOCKS=N        number of untied cores for iso_flops (default 16)
#   DISCOUNT_ACT=g      per-decision discount (default 0.99)
#   DISCOUNT_COMPUTE=g  per-extra-think-iteration discount (default 1.0 = standard return;
#                       = DISCOUNT_ACT gives the RAMDP return)
# Non-default values are appended to the run directory name (e.g. looped_it8_gc0.99_s0).
# All arms: PPO on the easy_train bank (Fogleman puzzles with <= 15 moves), in-training evaluation on easy_valid
# (eval/test_* metrics), 1024 envs x 64 steps per rollout, 64 minibatches x 4 epochs, 100M env steps, 2-layer d=128
# core, plane tokenizer (7 planes per cell), per-cell action head (2 logits per cell = 72 actions), global grad-norm
# clip 1.0, potential-based reward shaping (weight 1.0), loss on the final readout for every arm.
#   looped    : weight-tied core, cap 16 think iterations, policy-KL halting (1e-3)
#   iso_flops : 16 untied cores stacked (16x the parameters)
#   iso_param : one core applied once
#   iso_loop  : weight-tied core, always runs the full 16 think iterations (no halting)
# Requires data/rushhour/{easy_train,easy_valid,easy_test}.npz (python data_scripts/build_rushhour_banks.py --src rush.txt).
# Usage:
# MAX_THINK_ITERS=4 rushhour/scripts/train.sh looped 1 exp/rushhour 2 &
# MAX_THINK_ITERS=4 DISCOUNT_COMPUTE=0.99 rushhour/scripts/train.sh looped 1 exp/rushhour 3 &
#
set -euo pipefail
ARM="${1:?arm}"; SEED="${2:?seed}"; OUT="${3:-exp/rushhour}"; DEVICE="${4:-0}";
DISCOUNT_ACT="${DISCOUNT_ACT:-0.99}"; DISCOUNT_COMPUTE="${DISCOUNT_COMPUTE:-1.0}"
MAX_THINK_ITERS="${MAX_THINK_ITERS:-16}"; NUM_BLOCKS="${NUM_BLOCKS:-16}"
RUN="$ARM"
case "$ARM" in
  looped)    ARM_FLAGS=(--architecture=fprm --max_think_iters="$MAX_THINK_ITERS" --halt_criterion=kl --halt_kl=1e-3)
             [[ "$MAX_THINK_ITERS" != 16 ]] && RUN+="_it${MAX_THINK_ITERS}" ;;
  iso_flops) ARM_FLAGS=(--architecture=multi_block --num_blocks="$NUM_BLOCKS")
             [[ "$NUM_BLOCKS" != 16 ]] && RUN+="_b${NUM_BLOCKS}" ;;
  iso_param) ARM_FLAGS=(--architecture=single_block) ;;
  iso_loop)  ARM_FLAGS=(--architecture=fprm --max_think_iters="$MAX_THINK_ITERS" --halt_criterion=kl --halt_kl=0.0)
             [[ "$MAX_THINK_ITERS" != 16 ]] && RUN+="_it${MAX_THINK_ITERS}" ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
ARM_FLAGS+=(--discount_act="$DISCOUNT_ACT" --discount_compute="$DISCOUNT_COMPUTE")
[[ "$DISCOUNT_ACT" != 0.99 ]] && RUN+="_ga${DISCOUNT_ACT}"
[[ "$DISCOUNT_COMPUTE" != 1.0 ]] && RUN+="_gc${DISCOUNT_COMPUTE}"
RUN+="_s${SEED}"
export WANDB_MODE=${WANDB_MODE:-disabled}
cd "$(dirname "$0")/../.."
mkdir -p "$OUT/${RUN}"

source .venv/bin/activate
CUDA_VISIBLE_DEVICES="${DEVICE}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 python -u ppo.py --env_id=rushhour-easy_train-easy_valid --seed="$SEED" --wandb_dir="$OUT/${RUN}" --track \
  --fprm_num_layers=2 --max_grad_norm=1.0 \
  --num_envs=1024 --rollout_length=64 --num_minibatches_per_rollout=64 --num_epochs_per_rollout=4 \
  --num_timesteps=100000000 --num_eval_steps=100 --num_reset_steps=100 --num_eval_envs=256 \
  "${ARM_FLAGS[@]}" 2>&1 | tee -a "$OUT/${RUN}/train.log"
