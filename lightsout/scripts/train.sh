#!/bin/bash
# Train one arm of the Lights Out study.
#   lightsout/scripts/train.sh <arm> <seed> [out_dir] [device]
#   arm: looped | iso_flops | iso_param | iso_loop | cot | iso_cot | perceiver | iso_perceiver | mem | iso_mem | exchange | iso_exchange
# Optional environment variables:
#   GRID=MxN            board size (default 5x4)
#   EPISODE_LENGTH=N    steps per episode (default M*N, as in the Stoix Lights Out sweep)
#   DIFFICULTY_THRESHOLD=f  training goals need < int(M*N*f) presses, evaluation goals >= it (default 0.5)
#   DISCOUNT_ACT=g      per-decision discount (default 0.99)
#   DISCOUNT_COMPUTE=g  per-extra-think-iteration discount (default 1.0 = standard return;
#                       = DISCOUNT_ACT gives the RAMDP return)
#   MAX_THINK_ITERS=N   think-iteration cap for looped / iso_loop / cot / iso_cot / perceiver / iso_perceiver (default 16)
#   NUM_BLOCKS=N        number of untied cores for iso_flops (default 16)
#   THOUGHT_NORM=1      cot / perceiver / mem / exchange arms (and iso_): RMS-normalize each thought / latent before feeding it back (default 0)
#   INPUT_GRID_CONV=1   cot / iso_cot / perceiver / iso_perceiver: residual depthwise 3x3 conv over the cell embeddings
#                       (gives the cell tokens local spatial mixing; needs the cell tokenizer) (default 0)
#   INPUT_GRID_CONV_DEPTH=D  with INPUT_GRID_CONV=1: 1 = one linear depthwise conv; D > 1 = D ConvNeXt-style blocks
#                       (depthwise 3x3 + SwiGLU MLP, nonlinear board processing) (default 1)
#   STEP_GRID_CONV=1    perceiver / iso_perceiver: apply a residual depthwise 3x3 conv to the cells at every thinking
#                       step (step i reads the cells convolved i times; needs the cell tokenizer) (default 0)
#   NUM_INITIAL_LATENTS=K  perceiver / iso_perceiver: K learned initial latents (bidirectional among themselves, all
#                       in the first thinking step) instead of a single <BOT> (default 1)
#   CELL_ENCODER_ITERS=N  perceiver / iso_perceiver: looped board encoder - the cells go N times through their own
#                       FPRM core before thinking; latent layer l reads its last layer-l output (default 0 = off)
#   EXCHANGE_LATENTS=learned|patch  exchange / iso_exchange: persistent latents (default learned)
#   EXCHANGE_NUM_LATENTS=K  exchange, learned: number of persistent latents, 0 = thoughts only (default 16)
#   EXCHANGE_PATCH=p    exchange, patch: one latent per p x p patch (default 2)
#   EXCHANGE_PATCH_LOCAL=0|1  exchange, patch: patch latents read / write only their own patch (default 1)
# (No TOKENIZER option: Lights Out uses the per-cell press head, which needs the cell tokenizer.)
#   MLP_TYPE=swiglu|gelu  block MLP of every architecture (default swiglu)
#   MLP_ROUND_TO=r      MLP hidden width rounded up to a multiple of r (default 256; at d=128 swiglu: 256 -> 512,
#                       64 -> 384, 1 -> 341)
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
#   mem       : fprm's looped board plus a memory of thought tokens (cells and thoughts attend to each other, readout
#               from the newest thought), cap 16 passes, policy-KL halting (1e-3)
#   iso_mem   : the same, always runs the full 16 passes (no halting)
#   exchange  : autoregressive thoughts + persistent latents that read the cells and write back every step, local
#               conv/MLP cell update, no cell-to-cell attention; cap 16 steps, policy-KL halting (1e-3)
#   iso_exchange : the same, always runs the full 16 steps (no halting)
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
MAX_THINK_ITERS="${MAX_THINK_ITERS:-16}"; NUM_BLOCKS="${NUM_BLOCKS:-16}"; THOUGHT_NORM="${THOUGHT_NORM:-0}"; INPUT_GRID_CONV="${INPUT_GRID_CONV:-0}"; INPUT_GRID_CONV_DEPTH="${INPUT_GRID_CONV_DEPTH:-1}"; STEP_GRID_CONV="${STEP_GRID_CONV:-0}"; NUM_INITIAL_LATENTS="${NUM_INITIAL_LATENTS:-1}"; CELL_ENCODER_ITERS="${CELL_ENCODER_ITERS:-0}"
MLP_TYPE="${MLP_TYPE:-swiglu}"; MLP_ROUND_TO="${MLP_ROUND_TO:-256}"
EXCHANGE_LATENTS="${EXCHANGE_LATENTS:-learned}"; EXCHANGE_NUM_LATENTS="${EXCHANGE_NUM_LATENTS:-16}"; EXCHANGE_PATCH="${EXCHANGE_PATCH:-2}"; EXCHANGE_PATCH_LOCAL="${EXCHANGE_PATCH_LOCAL:-1}"
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
  mem)       ARM_FLAGS=(--architecture=fprm_mem --max_think_iters="$MAX_THINK_ITERS" --halt_criterion=kl --halt_kl=1e-3)
             [[ "$MAX_THINK_ITERS" != 16 ]] && RUN+="_it${MAX_THINK_ITERS}" ;;
  iso_mem)   ARM_FLAGS=(--architecture=fprm_mem --max_think_iters="$MAX_THINK_ITERS" --halt_criterion=kl --halt_kl=0.0)
             [[ "$MAX_THINK_ITERS" != 16 ]] && RUN+="_it${MAX_THINK_ITERS}" ;;
  exchange)  ARM_FLAGS=(--architecture=fprm_exchange --max_think_iters="$MAX_THINK_ITERS" --halt_criterion=kl --halt_kl=1e-3)
             [[ "$MAX_THINK_ITERS" != 16 ]] && RUN+="_it${MAX_THINK_ITERS}" ;;
  iso_exchange) ARM_FLAGS=(--architecture=fprm_exchange --max_think_iters="$MAX_THINK_ITERS" --halt_criterion=kl --halt_kl=0.0)
             [[ "$MAX_THINK_ITERS" != 16 ]] && RUN+="_it${MAX_THINK_ITERS}" ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
case "$THOUGHT_NORM" in
  0) ;;
  1) [[ "$ARM" =~ ^(iso_)?(cot|perceiver|mem|exchange)$ ]] \
       || { echo "THOUGHT_NORM=1 needs arm cot, perceiver, mem or exchange (or their iso_ variants)"; exit 1; }
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
[[ "$INPUT_GRID_CONV_DEPTH" =~ ^[1-9][0-9]*$ ]] || { echo "INPUT_GRID_CONV_DEPTH must be a positive integer, got $INPUT_GRID_CONV_DEPTH"; exit 1; }
if [[ "$INPUT_GRID_CONV_DEPTH" != 1 ]]; then
  [[ "$INPUT_GRID_CONV" == 1 ]] || { echo "INPUT_GRID_CONV_DEPTH needs INPUT_GRID_CONV=1"; exit 1; }
  ARM_FLAGS+=(--input_grid_conv_depth="$INPUT_GRID_CONV_DEPTH"); RUN+="_igcd${INPUT_GRID_CONV_DEPTH}"
fi
case "$STEP_GRID_CONV" in
  0) ;;
  1) [[ "$ARM" == perceiver || "$ARM" == iso_perceiver ]] || { echo "STEP_GRID_CONV=1 needs arm perceiver or iso_perceiver"; exit 1; }
     ARM_FLAGS+=(--perceiver_step_grid_conv); RUN+="_sgc" ;;
  *) echo "STEP_GRID_CONV must be 0 or 1, got $STEP_GRID_CONV"; exit 1 ;;
esac
[[ "$NUM_INITIAL_LATENTS" =~ ^[1-9][0-9]*$ ]] || { echo "NUM_INITIAL_LATENTS must be a positive integer, got $NUM_INITIAL_LATENTS"; exit 1; }
if [[ "$NUM_INITIAL_LATENTS" != 1 ]]; then
  [[ "$ARM" == perceiver || "$ARM" == iso_perceiver ]] || { echo "NUM_INITIAL_LATENTS needs arm perceiver or iso_perceiver"; exit 1; }
  ARM_FLAGS+=(--perceiver_num_initial_latents="$NUM_INITIAL_LATENTS"); RUN+="_il${NUM_INITIAL_LATENTS}"
fi
[[ "$CELL_ENCODER_ITERS" =~ ^[0-9]+$ ]] || { echo "CELL_ENCODER_ITERS must be a non-negative integer, got $CELL_ENCODER_ITERS"; exit 1; }
if [[ "$CELL_ENCODER_ITERS" != 0 ]]; then
  [[ "$ARM" == perceiver || "$ARM" == iso_perceiver ]] || { echo "CELL_ENCODER_ITERS needs arm perceiver or iso_perceiver"; exit 1; }
  [[ "$STEP_GRID_CONV" == 0 ]] || { echo "CELL_ENCODER_ITERS and STEP_GRID_CONV cannot be combined"; exit 1; }
  ARM_FLAGS+=(--perceiver_cell_encoder_iters="$CELL_ENCODER_ITERS"); RUN+="_ce${CELL_ENCODER_ITERS}"
fi
if [[ "$ARM" == exchange || "$ARM" == iso_exchange ]]; then
  case "$EXCHANGE_LATENTS" in
    learned) ARM_FLAGS+=(--exchange_latents=learned --exchange_num_latents="$EXCHANGE_NUM_LATENTS"); RUN+="_lat${EXCHANGE_NUM_LATENTS}" ;;
    patch)   ARM_FLAGS+=(--exchange_latents=patch --exchange_patch="$EXCHANGE_PATCH"); RUN+="_patch${EXCHANGE_PATCH}"
             case "$EXCHANGE_PATCH_LOCAL" in
               1) ;;
               0) ARM_FLAGS+=(--no-exchange_patch_local); RUN+="g" ;;
               *) echo "EXCHANGE_PATCH_LOCAL must be 0 or 1, got $EXCHANGE_PATCH_LOCAL"; exit 1 ;;
             esac ;;
    *) echo "EXCHANGE_LATENTS must be learned or patch, got $EXCHANGE_LATENTS"; exit 1 ;;
  esac
fi
[[ "$GRID" != 5x4 ]] && RUN+="_g${GRID}"
[[ "$EPISODE_LENGTH" != "$DEFAULT_EPISODE_LENGTH" ]] && RUN+="_el${EPISODE_LENGTH}"
[[ "$DIFFICULTY_THRESHOLD" != 0.5 ]] && RUN+="_dt${DIFFICULTY_THRESHOLD}"
case "$MLP_TYPE" in
  swiglu) ;;
  gelu) ARM_FLAGS+=(--mlp_type=gelu); RUN+="_gelu" ;;
  *) echo "MLP_TYPE must be swiglu or gelu, got $MLP_TYPE"; exit 1 ;;
esac
[[ "$MLP_ROUND_TO" =~ ^[1-9][0-9]*$ ]] || { echo "MLP_ROUND_TO must be a positive integer, got $MLP_ROUND_TO"; exit 1; }
[[ "$MLP_ROUND_TO" != 256 ]] && { ARM_FLAGS+=(--mlp_round_to="$MLP_ROUND_TO"); RUN+="_mr${MLP_ROUND_TO}"; }
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
