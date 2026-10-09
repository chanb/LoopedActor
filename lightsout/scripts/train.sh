#!/bin/bash
# Train one arm of the Lights Out study.
#   lightsout/scripts/train.sh <arm> <seed> [out_dir] [device]
#   arm: looped | iso_flops | iso_param | iso_loop | cot | iso_cot | perceiver | iso_perceiver
# Optional environment variables:
#   GRID=MxN            board size (default 5x4)
#   EPISODE_LENGTH=N    steps per episode (default M*N, as in the Stoix Lights Out sweep)
#   DIFFICULTY_THRESHOLD=f  training goals need < int(M*N*f) presses, evaluation goals >= it (default 0.5)
#   DISCOUNT_ACT=g      per-decision discount (default 0.99)
#   DISCOUNT_COMPUTE=g  per-extra-think-iteration discount (default 1.0 = standard return;
#                       = DISCOUNT_ACT gives the RAMDP return)
#   MAX_THINK_ITERS=N   think-iteration cap for looped / iso_loop / cot / iso_cot / perceiver / iso_perceiver (default 16)
#   NUM_BLOCKS=N        number of untied cores for iso_flops (default 16)
#   THOUGHT_NORM=1      cot / iso_cot / perceiver / iso_perceiver: RMS-normalize each thought / latent before feeding it back (default 0)
#   INPUT_GRID_CONV=1   cot / iso_cot / perceiver / iso_perceiver: residual depthwise 3x3 conv over the cell embeddings
#                       (gives the cell tokens local spatial mixing; needs the cell tokenizer) (default 0)
# (No TOKENIZER option: Lights Out uses the per-cell press head, which needs the cell tokenizer.)
# The wandb group is <arm>_<uuid>, where the uuid is derived from every hyperparameter flag except the seed,
# so seeds of the same setting are grouped together.
# Non-default values are appended to the run directory name (e.g. looped_it8_g3x3_s0).
# All arms: PPO on procedurally generated instances (lightsout-<M>x<N>: random initial grid, goal = initial
# toggled by dist random presses; train dist in [1, threshold), in-training evaluation (eval/test_* metrics) on
# dist in [threshold, M*N)), sparse reward (1 on solve), 1024 envs x 64 steps per rollout, 64 minibatches x 4
# epochs, 100M env steps, 2-layer d=128 core, flat grid + goal tokenizer (current / target / mismatch per cell),
# per-cell press head, global grad-norm clip 1.0, loss on the final readout for every arm.
#   looped    : weight-tied core, cap 16 think iterations, policy-KL halting (1e-3)
#   iso_flops : 16 untied cores stacked (16x the parameters)
#   iso_param : one core applied once
#   iso_loop  : weight-tied core, always runs the full 16 think iterations (no halting)
#   cot       : implicit CoT (appends continuous thought tokens, causal, KV-cached), cap 16 passes, policy-KL halting (1e-3)
#   iso_cot   : implicit CoT, always runs the full 16 passes (no halting)
#   perceiver : Perceiver AR-style latent CoT (latents from <BOT> attend to the cell tokens + earlier latents at every
#               layer; cells never processed), cap 16 passes, policy-KL halting (1e-3)
#   iso_perceiver : Perceiver AR-style latent CoT, always runs the full 16 passes (no halting)
# Usage:
# MAX_THINK_ITERS=4 lightsout/scripts/train.sh looped 1 exp/lightsout 0 &
# GRID=3x3 MAX_THINK_ITERS=4 DISCOUNT_COMPUTE=0.99 lightsout/scripts/train.sh looped 1 exp/lightsout 1 &
#
set -euo pipefail
ARM="${1:?arm}"; SEED="${2:?seed}"; OUT="${3:-exp/lightsout}"; DEVICE="${4:-0}";
GRID="${GRID:-5x4}"; DIFFICULTY_THRESHOLD="${DIFFICULTY_THRESHOLD:-0.5}"
[[ "$GRID" =~ ^([0-9]+)x([0-9]+)$ ]] || { echo "GRID must be MxN, got $GRID"; exit 1; }
DEFAULT_EPISODE_LENGTH=$(( BASH_REMATCH[1] * BASH_REMATCH[2] ))
EPISODE_LENGTH="${EPISODE_LENGTH:-$DEFAULT_EPISODE_LENGTH}"
DISCOUNT_ACT="${DISCOUNT_ACT:-0.99}"; DISCOUNT_COMPUTE="${DISCOUNT_COMPUTE:-1.0}"
MAX_THINK_ITERS="${MAX_THINK_ITERS:-16}"; NUM_BLOCKS="${NUM_BLOCKS:-16}"; THOUGHT_NORM="${THOUGHT_NORM:-0}"; INPUT_GRID_CONV="${INPUT_GRID_CONV:-0}"
RUN="$ARM"
case "$ARM" in
  looped)    ARM_FLAGS=(--architecture=fprm --max_think_iters="$MAX_THINK_ITERS" --halt_criterion=kl --halt_kl=1e-3)
             [[ "$MAX_THINK_ITERS" != 16 ]] && RUN+="_it${MAX_THINK_ITERS}" ;;
  iso_flops) ARM_FLAGS=(--architecture=multi_block --num_blocks="$NUM_BLOCKS")
             [[ "$NUM_BLOCKS" != 16 ]] && RUN+="_b${NUM_BLOCKS}" ;;
  iso_param) ARM_FLAGS=(--architecture=single_block) ;;
  iso_loop)  ARM_FLAGS=(--architecture=fprm --max_think_iters="$MAX_THINK_ITERS" --halt_criterion=kl --halt_kl=0.0)
             [[ "$MAX_THINK_ITERS" != 16 ]] && RUN+="_it${MAX_THINK_ITERS}" ;;
  cot)       ARM_FLAGS=(--architecture=fprm_cot --max_think_iters="$MAX_THINK_ITERS" --halt_criterion=kl --halt_kl=1e-3)
             [[ "$MAX_THINK_ITERS" != 16 ]] && RUN+="_it${MAX_THINK_ITERS}" ;;
  iso_cot)   ARM_FLAGS=(--architecture=fprm_cot --max_think_iters="$MAX_THINK_ITERS" --halt_criterion=kl --halt_kl=0.0)
             [[ "$MAX_THINK_ITERS" != 16 ]] && RUN+="_it${MAX_THINK_ITERS}" ;;
  perceiver) ARM_FLAGS=(--architecture=fprm_perceiver --max_think_iters="$MAX_THINK_ITERS" --halt_criterion=kl --halt_kl=1e-3)
             [[ "$MAX_THINK_ITERS" != 16 ]] && RUN+="_it${MAX_THINK_ITERS}" ;;
  iso_perceiver) ARM_FLAGS=(--architecture=fprm_perceiver --max_think_iters="$MAX_THINK_ITERS" --halt_criterion=kl --halt_kl=0.0)
             [[ "$MAX_THINK_ITERS" != 16 ]] && RUN+="_it${MAX_THINK_ITERS}" ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
case "$THOUGHT_NORM" in
  0) ;;
  1) [[ "$ARM" == cot || "$ARM" == iso_cot || "$ARM" == perceiver || "$ARM" == iso_perceiver ]] \
       || { echo "THOUGHT_NORM=1 needs arm cot, iso_cot, perceiver or iso_perceiver"; exit 1; }
     ARM_FLAGS+=(--cot_thought_norm); RUN+="_tn" ;;
  *) echo "THOUGHT_NORM must be 0 or 1, got $THOUGHT_NORM"; exit 1 ;;
esac
case "$INPUT_GRID_CONV" in
  0) ;;
  1) [[ "$ARM" == cot || "$ARM" == iso_cot || "$ARM" == perceiver || "$ARM" == iso_perceiver ]] \
       || { echo "INPUT_GRID_CONV=1 needs arm cot, iso_cot, perceiver or iso_perceiver"; exit 1; }
     ARM_FLAGS+=(--input_grid_conv); RUN+="_igc" ;;
  *) echo "INPUT_GRID_CONV must be 0 or 1, got $INPUT_GRID_CONV"; exit 1 ;;
esac
[[ "$GRID" != 5x4 ]] && RUN+="_g${GRID}"
[[ "$EPISODE_LENGTH" != "$DEFAULT_EPISODE_LENGTH" ]] && RUN+="_el${EPISODE_LENGTH}"
[[ "$DIFFICULTY_THRESHOLD" != 0.5 ]] && RUN+="_dt${DIFFICULTY_THRESHOLD}"
ARM_FLAGS+=(--discount_act="$DISCOUNT_ACT" --discount_compute="$DISCOUNT_COMPUTE")
[[ "$DISCOUNT_ACT" != 0.99 ]] && RUN+="_ga${DISCOUNT_ACT}"
[[ "$DISCOUNT_COMPUTE" != 1.0 ]] && RUN+="_gc${DISCOUNT_COMPUTE}"
RUN+="_s${SEED}"
# Every hyperparameter flag except the seed; hashed into the wandb group.
HPARAM_FLAGS=(--env_id="lightsout-${GRID}" --lightsout_episode_length="$EPISODE_LENGTH"
  --lightsout_difficulty_threshold="$DIFFICULTY_THRESHOLD"
  --fprm_num_layers=2 --max_grad_norm=1.0
  --num_envs=1024 --rollout_length=64 --num_minibatches_per_rollout=64 --num_epochs_per_rollout=4
  --num_timesteps=100000000 --num_eval_steps=100 --num_reset_steps=100 --num_eval_envs=256
  "${ARM_FLAGS[@]}")
GROUP="${ARM}_$(uuidgen --sha1 --namespace @oid --name "${HPARAM_FLAGS[*]}")"
export WANDB_MODE=${WANDB_MODE:-disabled}
cd "$(dirname "$0")/../.."
mkdir -p "$OUT/${RUN}"

source .venv/bin/activate
CUDA_VISIBLE_DEVICES="${DEVICE}" XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 python -u ppo.py --seed="$SEED" --wandb_dir="$OUT/${RUN}" --track --wandb_group="$GROUP" \
  "${HPARAM_FLAGS[@]}" 2>&1 | tee -a "$OUT/${RUN}/train.log"
