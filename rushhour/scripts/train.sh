#!/bin/bash
# Train one arm of the Rush Hour study.
#   rushhour/scripts/train.sh <arm> <seed> [out_dir] [device]
#   arm: looped | iso_flops | iso_param | iso_loop | cot | iso_cot | perceiver | iso_perceiver | mem | iso_mem | mem_halt | rim | iso_rim
# Optional environment variables:
#   MAX_THINK_ITERS=N   think-iteration cap for looped / iso_loop / cot / iso_cot / perceiver / iso_perceiver (default 16)
#   NUM_BLOCKS=N        number of untied cores for iso_flops (default 16)
#   THOUGHT_NORM=1      cot / perceiver / mem arms (and iso_): RMS-normalize each thought / latent before feeding it back (default 0)
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
#   RIM_HALT_THRESH=r   rim: halt once ||sep_s - sep_{s-1}|| / ||sep_s|| < r (default 0.1)
#   HALT_KL=k           looped / cot / perceiver / mem: halt once the policy KL between passes < k (default 1e-3);
#                       mem_halt: initialization only (h_i = 1/2 at KL_i = k)
#   HALT_ENTROPY_COST=c  mem_halt: entropy bonus on the think-time distribution p(c|s) (default 0)
#   HALT_ALPHA_MIN=a    mem_halt: floor on the KL slope alpha, in [0, 1) (default 0 = no floor)
#   HALT_BETA_HEAD=0|1  mem_halt: state-dependent beta (Dense on the newest thought) instead of a scalar (default 0)
#   HALT_HEAD_GRAD=0|1  mem_halt: let the halting loss backpropagate into the trunk through the beta head (default 0)
#   HALT_THRESHOLD_PARAM=0|1  mem_halt: learn the halting threshold log KL* itself, h = sigmoid(alpha (log KL* - log KL))
#                       (alpha then sets only the sharpness; default 0: beta - alpha log KL, threshold beta / alpha)
#   MIN_THINK_ITERS=n   halting arms: earliest pass that may halt, >= 1 (default 2; mem_halt: 1 lets pass 1 halt on
#                       KL(pi_1 || uniform))
#   DISCOUNT_ACT=g      per-decision discount (default 0.99)
#   DISCOUNT_COMPUTE=g  per-extra-think-iteration discount (default 1.0 = standard return;
#                       = DISCOUNT_ACT gives the RAMDP return)
#   MLP_TYPE=swiglu|gelu  block MLP of every architecture (default swiglu)
#   MLP_ROUND_TO=r      MLP hidden width rounded up to a multiple of r (default 256; at d=128 swiglu: 256 -> 512,
#                       64 -> 384, 1 -> 341)
#   COMPUTE_DTYPE=float32|float16|bfloat16  block computation dtype (default float32). bfloat16: ~1.6-1.75x faster, fp32 range;
#                       float16: slightly faster and more precise, but activations can overflow (multi_block NaNs)
#   DYNAMIC_LOSS_SCALE=0|1  COMPUTE_DTYPE=float16: dynamic loss scaling against fp16 gradient underflow (default 1)
#   NORMALIZE_OBS=0|1  running mean/std observation normalization (default 0 = raw observations; runs before this
#                       option always normalized)
#   TARGET_KL=k         stop a rollout's remaining minibatch updates once approx_kl_action > k (the action policy's KL only, not the
#                       think time's; default 0 = off; skipped updates
#                       are not computed, so this also saves time; training/kl_stop_frac = fraction skipped)
#   NUM_EPOCHS=n        PPO epochs per rollout (default 4; fewer = smaller policy change per rollout and proportionally faster)
#   NUM_TIMESTEPS=n     total environment steps (default 100000000; _ts<n/1e6>M in the run name when changed)
#   VALUE_AT_MAX_ITERS=0|1  mem / iso_mem: read the value out from the MAX_THINK_ITERS-th pass's latent instead of the
#                       halt pass's (halting then fixes only the policy readout) (default 0)
#   VALUE_EXPECTILE=tau  value loss expectile in (0, 1) (default 0.5 = MSE; e.g. 0.9 = optimistic upper-expectile value,
#                       as Stoix --expectile)
# The wandb group is <arm>_<uuid>, where the uuid is derived from every hyperparameter flag except the seed,
# so seeds of the same setting are grouped together.
# WANDB_PROJECT=name sets the wandb project (default looped_actor; not part of the run directory name).
# Non-default values are appended to the run directory name (e.g. looped_it8_gc0.99_s0).
# All arms: PPO on the easy_train bank (Fogleman puzzles with <= 15 moves), in-training evaluation on easy_valid
# (eval/test_* metrics), 1024 envs x 64 steps per rollout, 64 minibatches x 4 epochs, 100M env steps, 2-layer d=128
# core, plane tokenizer (7 planes per cell), per-cell action head (2 logits per cell = 72 actions), global grad-norm
# clip 1.0, potential-based reward shaping (weight 1.0), loss on the final readout for every arm.
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
#   mem_halt  : mem with stochastic learned halting, c ~ p(c|s) from h_i = sigmoid(beta - alpha log KL_i); the PPO
#               ratio uses log p(c|s) + log pi_c(a|s); value from the last pass (models/fprm_mem_halt.py)
#   rim       : RiM-style - each step appends [grid-conv'd cell output latents, SEP] (block-causal, KV-cached); halts
#               once the SEP latent converges (relative change < RIM_HALT_THRESH), cap 16 steps
#   iso_rim   : the same, always runs the full 16 steps (no halting)
# Requires data/rushhour/{easy_train,easy_valid,easy_test}.npz (python data_scripts/build_rushhour_banks.py --src rush.txt).
# Usage:
# MAX_THINK_ITERS=4 rushhour/scripts/train.sh looped 1 exp/rushhour 2 &
# MAX_THINK_ITERS=4 DISCOUNT_COMPUTE=0.99 rushhour/scripts/train.sh looped 1 exp/rushhour 3 &
#
set -euo pipefail
ARM="${1:?arm}"; SEED="${2:?seed}"; OUT="${3:-exp/rushhour}"; DEVICE="${4:-0}";
DISCOUNT_ACT="${DISCOUNT_ACT:-0.99}"; DISCOUNT_COMPUTE="${DISCOUNT_COMPUTE:-1.0}"
MAX_THINK_ITERS="${MAX_THINK_ITERS:-16}"; NUM_BLOCKS="${NUM_BLOCKS:-16}"; THOUGHT_NORM="${THOUGHT_NORM:-0}"; INPUT_GRID_CONV="${INPUT_GRID_CONV:-0}"; INPUT_GRID_CONV_DEPTH="${INPUT_GRID_CONV_DEPTH:-1}"; STEP_GRID_CONV="${STEP_GRID_CONV:-0}"; NUM_INITIAL_LATENTS="${NUM_INITIAL_LATENTS:-1}"; CELL_ENCODER_ITERS="${CELL_ENCODER_ITERS:-0}"
MLP_TYPE="${MLP_TYPE:-swiglu}"; MLP_ROUND_TO="${MLP_ROUND_TO:-256}"; COMPUTE_DTYPE="${COMPUTE_DTYPE:-float32}"; DYNAMIC_LOSS_SCALE="${DYNAMIC_LOSS_SCALE:-1}"; NORMALIZE_OBS="${NORMALIZE_OBS:-0}"; VALUE_EXPECTILE="${VALUE_EXPECTILE:-0.5}"; TARGET_KL="${TARGET_KL:-0}"; VALUE_AT_MAX_ITERS="${VALUE_AT_MAX_ITERS:-0}"; NUM_EPOCHS="${NUM_EPOCHS:-4}"
RIM_HALT_THRESH="${RIM_HALT_THRESH:-0.1}"
HALT_KL="${HALT_KL:-1e-3}"
HALT_ENTROPY_COST="${HALT_ENTROPY_COST:-0}"; HALT_BETA_HEAD="${HALT_BETA_HEAD:-0}"; HALT_ALPHA_MIN="${HALT_ALPHA_MIN:-0}"; HALT_HEAD_GRAD="${HALT_HEAD_GRAD:-0}"; HALT_THRESHOLD_PARAM="${HALT_THRESHOLD_PARAM:-0}"; MIN_THINK_ITERS="${MIN_THINK_ITERS:-2}"
RUN="$ARM"
case "$ARM" in
  looped)    ARM_FLAGS=(--architecture=fprm --max_think_iters="$MAX_THINK_ITERS" --halt_criterion=kl --halt_kl="$HALT_KL")
             [[ "$MAX_THINK_ITERS" != 16 ]] && RUN+="_it${MAX_THINK_ITERS}" ;;
  iso_flops) ARM_FLAGS=(--architecture=multi_block --num_blocks="$NUM_BLOCKS")
             [[ "$NUM_BLOCKS" != 16 ]] && RUN+="_b${NUM_BLOCKS}" ;;
  iso_param) ARM_FLAGS=(--architecture=single_block) ;;
  iso_loop)  ARM_FLAGS=(--architecture=fprm --max_think_iters="$MAX_THINK_ITERS" --halt_criterion=kl --halt_kl=-1.0)
             [[ "$MAX_THINK_ITERS" != 16 ]] && RUN+="_it${MAX_THINK_ITERS}" ;;
  cot)       ARM_FLAGS=(--architecture=fprm_cot --max_think_iters="$MAX_THINK_ITERS" --halt_criterion=kl --halt_kl="$HALT_KL")
             [[ "$MAX_THINK_ITERS" != 16 ]] && RUN+="_it${MAX_THINK_ITERS}" ;;
  iso_cot)   ARM_FLAGS=(--architecture=fprm_cot --max_think_iters="$MAX_THINK_ITERS" --halt_criterion=kl --halt_kl=-1.0)
             [[ "$MAX_THINK_ITERS" != 16 ]] && RUN+="_it${MAX_THINK_ITERS}" ;;
  perceiver) ARM_FLAGS=(--architecture=fprm_perceiver --max_think_iters="$MAX_THINK_ITERS" --halt_criterion=kl --halt_kl="$HALT_KL")
             [[ "$MAX_THINK_ITERS" != 16 ]] && RUN+="_it${MAX_THINK_ITERS}" ;;
  iso_perceiver) ARM_FLAGS=(--architecture=fprm_perceiver --max_think_iters="$MAX_THINK_ITERS" --halt_criterion=kl --halt_kl=-1.0)
             [[ "$MAX_THINK_ITERS" != 16 ]] && RUN+="_it${MAX_THINK_ITERS}" ;;
  mem)       ARM_FLAGS=(--architecture=fprm_mem --max_think_iters="$MAX_THINK_ITERS" --halt_criterion=kl --halt_kl="$HALT_KL")
             [[ "$MAX_THINK_ITERS" != 16 ]] && RUN+="_it${MAX_THINK_ITERS}" ;;
  iso_mem)   ARM_FLAGS=(--architecture=fprm_mem --max_think_iters="$MAX_THINK_ITERS" --halt_criterion=kl --halt_kl=-1.0)
             [[ "$MAX_THINK_ITERS" != 16 ]] && RUN+="_it${MAX_THINK_ITERS}" ;;
  mem_halt)  ARM_FLAGS=(--architecture=fprm_mem_halt --max_think_iters="$MAX_THINK_ITERS" --halt_criterion=kl --halt_kl="$HALT_KL")
             [[ "$MAX_THINK_ITERS" != 16 ]] && RUN+="_it${MAX_THINK_ITERS}" ;;
  rim)       ARM_FLAGS=(--architecture=fprm_rim --max_think_iters="$MAX_THINK_ITERS" --halt_criterion=latent_residual --halt_residual_thresh="$RIM_HALT_THRESH")
             [[ "$MAX_THINK_ITERS" != 16 ]] && RUN+="_it${MAX_THINK_ITERS}"
             [[ "$RIM_HALT_THRESH" != 0.1 ]] && RUN+="_ht${RIM_HALT_THRESH}" ;;
  iso_rim)   ARM_FLAGS=(--architecture=fprm_rim --max_think_iters="$MAX_THINK_ITERS" --halt_criterion=latent_residual --halt_residual_thresh=0.0)
             [[ "$MAX_THINK_ITERS" != 16 ]] && RUN+="_it${MAX_THINK_ITERS}" ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
[[ "$ARM" =~ ^(looped|cot|perceiver|mem|mem_halt)$ && "$HALT_KL" != 1e-3 ]] && RUN+="_hkl${HALT_KL}"
[[ "$MIN_THINK_ITERS" =~ ^[1-9][0-9]*$ ]] || { echo "MIN_THINK_ITERS must be an integer >= 1, got $MIN_THINK_ITERS"; exit 1; }
[[ "$MIN_THINK_ITERS" != 2 ]] && { ARM_FLAGS+=(--min_think_iters="$MIN_THINK_ITERS"); RUN+="_minit${MIN_THINK_ITERS}"; }
if [[ "$ARM" == mem_halt ]]; then
  awk -v c="$HALT_ENTROPY_COST" 'BEGIN { exit !(c + 0 == c && c >= 0) }' || { echo "HALT_ENTROPY_COST must be >= 0, got $HALT_ENTROPY_COST"; exit 1; }
  awk -v c="$HALT_ENTROPY_COST" 'BEGIN { exit !(c > 0) }' && { ARM_FLAGS+=(--halt_entropy_cost="$HALT_ENTROPY_COST"); RUN+="_hent${HALT_ENTROPY_COST}"; }
  awk -v a="$HALT_ALPHA_MIN" 'BEGIN { exit !(a + 0 == a && a >= 0 && a < 1) }' || { echo "HALT_ALPHA_MIN must be in [0, 1), got $HALT_ALPHA_MIN"; exit 1; }
  awk -v a="$HALT_ALPHA_MIN" 'BEGIN { exit !(a > 0) }' && { ARM_FLAGS+=(--halt_alpha_min="$HALT_ALPHA_MIN"); RUN+="_amin${HALT_ALPHA_MIN}"; }
  case "$HALT_BETA_HEAD" in
    0) ;;
    1) ARM_FLAGS+=(--halt_beta_head); RUN+="_bh" ;;
    *) echo "HALT_BETA_HEAD must be 0 or 1, got $HALT_BETA_HEAD"; exit 1 ;;
  esac
  case "$HALT_HEAD_GRAD" in
    0) ;;
    1) ARM_FLAGS+=(--halt_head_grad); RUN+="_hgrad" ;;
    *) echo "HALT_HEAD_GRAD must be 0 or 1, got $HALT_HEAD_GRAD"; exit 1 ;;
  esac
  case "$HALT_THRESHOLD_PARAM" in
    0) ;;
    1) ARM_FLAGS+=(--halt_threshold_param); RUN+="_hthr" ;;
    *) echo "HALT_THRESHOLD_PARAM must be 0 or 1, got $HALT_THRESHOLD_PARAM"; exit 1 ;;
  esac
fi  # other arms ignore HALT_ENTROPY_COST / HALT_ALPHA_MIN / HALT_BETA_HEAD / HALT_HEAD_GRAD / HALT_THRESHOLD_PARAM (shared study settings)
case "$THOUGHT_NORM" in
  0) ;;
  1) [[ "$ARM" =~ ^(iso_)?(cot|perceiver|mem)$ || "$ARM" == mem_halt ]] \
       || { echo "THOUGHT_NORM=1 needs arm cot, perceiver or mem (or their iso_ variants)"; exit 1; }
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
case "$MLP_TYPE" in
  swiglu) ;;
  gelu) ARM_FLAGS+=(--mlp_type=gelu); RUN+="_gelu" ;;
  *) echo "MLP_TYPE must be swiglu or gelu, got $MLP_TYPE"; exit 1 ;;
esac
[[ "$MLP_ROUND_TO" =~ ^[1-9][0-9]*$ ]] || { echo "MLP_ROUND_TO must be a positive integer, got $MLP_ROUND_TO"; exit 1; }
[[ "$MLP_ROUND_TO" != 256 ]] && { ARM_FLAGS+=(--mlp_round_to="$MLP_ROUND_TO"); RUN+="_mr${MLP_ROUND_TO}"; }
case "$COMPUTE_DTYPE" in
  float32) ;;
  float16) ARM_FLAGS+=(--compute_dtype=float16); RUN+="_fp16" ;;
  bfloat16) ARM_FLAGS+=(--compute_dtype=bfloat16); RUN+="_bf16" ;;
  *) echo "COMPUTE_DTYPE must be float32, float16 or bfloat16, got $COMPUTE_DTYPE"; exit 1 ;;
esac
[[ "$DYNAMIC_LOSS_SCALE" == 0 || "$DYNAMIC_LOSS_SCALE" == 1 ]] || { echo "DYNAMIC_LOSS_SCALE must be 0 or 1, got $DYNAMIC_LOSS_SCALE"; exit 1; }
[[ "$COMPUTE_DTYPE" == float16 && "$DYNAMIC_LOSS_SCALE" == 0 ]] && { ARM_FLAGS+=(--no-dynamic_loss_scale); RUN+="_nols"; }
case "$NORMALIZE_OBS" in
  0) ARM_FLAGS+=(--no-normalize_observations) ;;
  1) ARM_FLAGS+=(--normalize_observations); RUN+="_obsnorm" ;;
  *) echo "NORMALIZE_OBS must be 0 or 1, got $NORMALIZE_OBS"; exit 1 ;;
esac
awk -v t="$VALUE_EXPECTILE" 'BEGIN { exit !(t + 0 == t && t > 0 && t < 1) }' \
  || { echo "VALUE_EXPECTILE must be in (0, 1), got $VALUE_EXPECTILE"; exit 1; }
[[ "$VALUE_EXPECTILE" != 0.5 ]] && { ARM_FLAGS+=(--value_expectile="$VALUE_EXPECTILE"); RUN+="_ve${VALUE_EXPECTILE}"; }
awk -v k="$TARGET_KL" 'BEGIN { exit !(k + 0 == k && k >= 0) }' || { echo "TARGET_KL must be >= 0, got $TARGET_KL"; exit 1; }
awk -v k="$TARGET_KL" 'BEGIN { exit !(k > 0) }' && { ARM_FLAGS+=(--target_kl="$TARGET_KL"); RUN+="_tkl${TARGET_KL}"; }
[[ "$NUM_EPOCHS" =~ ^[1-9][0-9]*$ ]] || { echo "NUM_EPOCHS must be a positive integer, got $NUM_EPOCHS"; exit 1; }
[[ "$NUM_EPOCHS" != 4 ]] && RUN+="_ep${NUM_EPOCHS}"
NUM_TIMESTEPS="${NUM_TIMESTEPS:-100000000}"
[[ "$NUM_TIMESTEPS" =~ ^[1-9][0-9]*$ ]] || { echo "NUM_TIMESTEPS must be a positive integer, got $NUM_TIMESTEPS"; exit 1; }
if [[ "$NUM_TIMESTEPS" != 100000000 ]]; then
  (( NUM_TIMESTEPS % 1000000 == 0 )) && RUN+="_ts$(( NUM_TIMESTEPS / 1000000 ))M" || RUN+="_ts${NUM_TIMESTEPS}"
fi
case "$VALUE_AT_MAX_ITERS" in
  0) ;;
  1) [[ "$ARM" =~ ^(iso_)?mem$ ]] || { echo "VALUE_AT_MAX_ITERS=1 needs arm mem or iso_mem"; exit 1; }
     ARM_FLAGS+=(--value_at_max_iters); RUN+="_vT" ;;
  *) echo "VALUE_AT_MAX_ITERS must be 0 or 1, got $VALUE_AT_MAX_ITERS"; exit 1 ;;
esac
ARM_FLAGS+=(--discount_act="$DISCOUNT_ACT" --discount_compute="$DISCOUNT_COMPUTE")
[[ "$DISCOUNT_ACT" != 0.99 ]] && RUN+="_ga${DISCOUNT_ACT}"
[[ "$DISCOUNT_COMPUTE" != 1.0 ]] && RUN+="_gc${DISCOUNT_COMPUTE}"
RUN+="_s${SEED}"
[[ "${PRINT_RUN_DIR:-0}" == 1 ]] && { echo "$OUT/${RUN}"; exit 0; }   # just print the run directory (slurm/submit_mem.sh)
# Every hyperparameter flag except the seed; hashed into the wandb group.
HPARAM_FLAGS=(--env_id=rushhour-easy_train-easy_valid
  --fprm_num_layers=2 --max_grad_norm=1.0
  --num_envs=1024 --rollout_length=64 --num_minibatches_per_rollout=64 --num_epochs_per_rollout="$NUM_EPOCHS"
  --num_timesteps="$NUM_TIMESTEPS" --num_eval_steps=100 --num_reset_steps=100 --num_eval_envs=256
  "${ARM_FLAGS[@]}")
GROUP="${ARM}_$(uuidgen --sha1 --namespace @oid --name "${HPARAM_FLAGS[*]}")"
export WANDB_MODE=${WANDB_MODE:-disabled}
cd "$(dirname "$0")/../.."
mkdir -p "$OUT/${RUN}"

source .venv/bin/activate
CUDA_VISIBLE_DEVICES="${DEVICE}" XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.8}" python -u ppo.py --seed="$SEED" --wandb_dir="$OUT/${RUN}" --track --wandb_group="$GROUP" --wandb_project_name="${WANDB_PROJECT:-looped_actor}" \
  "${HPARAM_FLAGS[@]}" 2>&1 | tee -a "$OUT/${RUN}/train.log"
