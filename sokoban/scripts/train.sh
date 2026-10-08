#!/bin/bash
# Train one arm of the Boxoban study.
#   sokoban/scripts/train.sh <arm> <seed> [out_dir]
#   arm: looped | iso_flops | iso_param | iso_loop
# Optional environment variables:
#   RAMDP=1             discount rewards by compute time (think iterations / blocks) instead of env steps (default 0)
#   MAX_THINK_ITERS=N   think-iteration cap for looped / iso_loop (default 16)
#   NUM_BLOCKS=N        number of untied cores for iso_flops (default 16)
# Non-default values are appended to the run directory name (e.g. looped_it8_ramdp_s0).
# All arms: PPO on the 900k unfiltered training levels, held-out evaluation on unfiltered_valid during
# training (eval/test_* metrics), 1024 envs x 64 steps per rollout, 64 minibatches x 4 epochs, 100M env
# steps, 2-layer d=128 core, plane tokenizer (wall / box / target / player per cell), readout action head,
# global grad-norm clip 1.0, loss on the final readout for every arm.
#   looped    : weight-tied core, cap 16 think iterations, policy-KL halting (1e-3)        ~1.3k env steps/s on one B200 (~21 h)
#   iso_flops : 16 untied cores stacked (16x the parameters)                               ~1.8k env steps/s (~16 h)
#   iso_param : one core applied once                                                      ~28k env steps/s (~1 h)
#   iso_loop  : weight-tied core, always runs the full 16 think iterations (no halting)
# Requires data/boxoban/*.npz (python data_scripts/build_boxoban_banks.py).
set -euo pipefail
ARM="${1:?arm}"; SEED="${2:?seed}"; OUT="${3:-exp/sokoban}"; DEVICE="${4:-0}";
RAMDP="${RAMDP:-0}"; MAX_THINK_ITERS="${MAX_THINK_ITERS:-16}"; NUM_BLOCKS="${NUM_BLOCKS:-16}"
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
case "$RAMDP" in
  1) ARM_FLAGS+=(--ramdp); RUN+="_ramdp" ;;
  0) ARM_FLAGS+=(--no-ramdp) ;;
  *) echo "RAMDP must be 0 or 1, got $RAMDP"; exit 1 ;;
esac
RUN+="_s${SEED}"
export WANDB_MODE=${WANDB_MODE:-disabled}
cd "$(dirname "$0")/../.."
mkdir -p "$OUT/${RUN}"

source .venv/bin/activate
CUDA_VISIBLE_DEVICES="${DEVICE}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 python -u ppo.py --env_id=sokoban-unfiltered_train-unfiltered_valid --seed="$SEED" --wandb_dir="$OUT/${RUN}" --track \
  --fprm_num_layers=2 --max_grad_norm=1.0 \
  --discount=0.999 \
  --num_envs=512 --rollout_length=64 --num_minibatches_per_rollout=64 --num_epochs_per_rollout=4 \
  --num_timesteps=100000000 --num_eval_steps=100 --num_reset_steps=100 --num_eval_envs=256 \
  "${ARM_FLAGS[@]}" 2>&1 | tee -a "$OUT/${RUN}/train.log"
