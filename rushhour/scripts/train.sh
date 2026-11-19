#!/bin/bash
# Train one arm of the Rush Hour study.
#   rushhour/scripts/train.sh <arm> <seed> [out_dir] [device]
#   arm: iso_flops | iso_param | iso_mem | mem_halt
# Optional environment variables:
#   MAX_THINK_ITERS=C   iso_mem / mem_halt: the think cap C of --act_iters=C:u (iso_mem) / C:v (mem_halt) (default 16)
#   VALUE_ITERS=n       iso_mem / mem_halt: core calls of the separate FPRM critic (default MAX_THINK_ITERS; _vi<n> otherwise)
#   POLICY_PARAM=last|sum  iso_mem / mem_halt: pi_n from pass n's logits (last, default) or from their running sum (_psum)
#   LOGIT_SUM_STOP_GRAD=1  POLICY_PARAM=sum: no gradient through the logits of passes before c (_lsg)
#   NUM_BLOCKS=N        number of untied cores for iso_flops (default 16)
#   THOUGHT_NORM=1      iso_mem / mem_halt: RMS-normalize each thought before feeding it back (default 0)
#   HALT_CRITERION=sigmoid|kl|l2  mem_halt: learned stochastic halting head (sigmoid, default), or halt at pass n once
#                       KL(pi_(n+1) || pi_n) (kl) / ||rms_norm(h_(n+1)) - rms_norm(h_n)|| (l2) < HALT_THRESHOLD (_hkl / _hl2)
#   HALT_THRESHOLD=t    mem_halt, HALT_CRITERION=kl|l2: the threshold (default 1e-3; _ht<t>)
#   HALT_SG_SURVIVAL=1  mem_halt, sigmoid: log p(c|s) without gradient through the continue terms before c (_hsgs)
#   HALT_ENTROPY_COST=c  mem_halt, sigmoid: entropy bonus on the think-time distribution p(c|s) (default 0)
#   mem_halt's halting head is linear: h_n = sigmoid(w . rms_norm(h_n) + b), zero init (run names get _hlin)
#   HALT_HEAD_GRAD=0|1  mem_halt, sigmoid: let the halting loss backpropagate into the trunk through the halting head (default 0)
#   HALT_KL_REVERSE=0|1  mem_halt: halting KL_i = KL(pi_{i-1} || pi_i), under the acting policy (default 0)
#   MIN_THINK_ITERS=n   mem_halt: earliest pass that may halt, >= 1 (default 2)
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
#   VALUE_EXPECTILE=tau  value loss expectile in (0, 1) (default 0.5 = MSE; e.g. 0.9 = optimistic upper-expectile value,
#                       as Stoix --expectile)
# The wandb group is <arm>_<uuid>, where the uuid is derived from every hyperparameter flag except the seed,
# so seeds of the same setting are grouped together.
# WANDB_PROJECT=name sets the wandb project (default looped_actor; not part of the run directory name).
# Non-default values are appended to the run directory name (e.g. iso_mem_it8_gc0.99_s0).
# All arms: PPO on the easy_train bank (Fogleman puzzles with <= 15 moves), in-training evaluation on easy_valid
# (eval/test_* metrics), 1024 envs x 64 steps per rollout, 64 minibatches x 4 epochs, 100M env steps, 2-layer d=128
# core, plane tokenizer (7 planes per cell), per-cell action head (2 logits per cell = 72 actions), global grad-norm
# clip 1.0, potential-based reward shaping (weight 1.0), loss on the final readout for every arm.
#   iso_flops : 16 untied cores stacked (16x the parameters)
#   iso_param : one core applied once
#   iso_mem   : looped FPRM board plus a SEP token and a memory of thought tokens (cells, SEP and the newest thought
#               attend to each other and to earlier thoughts' K/V), always acts with pass 16 (--act_iters=16:u); the
#               critic is a separate FPRM run for VALUE_ITERS core calls (models/fprm_mem_halt.py)
#   mem_halt  : iso_mem with a variable budget (--act_iters=16:v), halting by HALT_CRITERION
# Requires data/rushhour/{easy_train,easy_valid,easy_test}.npz (python data_scripts/build_rushhour_banks.py --src rush.txt).
# Usage:
# MAX_THINK_ITERS=4 rushhour/scripts/train.sh iso_mem 1 exp/rushhour 2 &
# MAX_THINK_ITERS=4 DISCOUNT_COMPUTE=0.99 rushhour/scripts/train.sh iso_mem 1 exp/rushhour 3 &
#
set -euo pipefail
ARM="${1:?arm}"; SEED="${2:?seed}"; OUT="${3:-exp/rushhour}"; DEVICE="${4:-0}";
DISCOUNT_ACT="${DISCOUNT_ACT:-0.99}"; DISCOUNT_COMPUTE="${DISCOUNT_COMPUTE:-1.0}"
MAX_THINK_ITERS="${MAX_THINK_ITERS:-16}"; VALUE_ITERS="${VALUE_ITERS:-$MAX_THINK_ITERS}"; POLICY_PARAM="${POLICY_PARAM:-last}"; LOGIT_SUM_STOP_GRAD="${LOGIT_SUM_STOP_GRAD:-0}"; NUM_BLOCKS="${NUM_BLOCKS:-16}"; THOUGHT_NORM="${THOUGHT_NORM:-0}"
MLP_TYPE="${MLP_TYPE:-swiglu}"; MLP_ROUND_TO="${MLP_ROUND_TO:-256}"; COMPUTE_DTYPE="${COMPUTE_DTYPE:-float32}"; DYNAMIC_LOSS_SCALE="${DYNAMIC_LOSS_SCALE:-1}"; NORMALIZE_OBS="${NORMALIZE_OBS:-0}"; VALUE_EXPECTILE="${VALUE_EXPECTILE:-0.5}"; TARGET_KL="${TARGET_KL:-0}"; NUM_EPOCHS="${NUM_EPOCHS:-4}"
HALT_CRITERION="${HALT_CRITERION:-sigmoid}"; HALT_THRESHOLD="${HALT_THRESHOLD:-1e-3}"; HALT_SG_SURVIVAL="${HALT_SG_SURVIVAL:-0}"
HALT_ENTROPY_COST="${HALT_ENTROPY_COST:-0}"; HALT_HEAD_GRAD="${HALT_HEAD_GRAD:-0}"; HALT_KL_REVERSE="${HALT_KL_REVERSE:-0}"; MIN_THINK_ITERS="${MIN_THINK_ITERS:-2}"
RUN="$ARM"
case "$ARM" in
  iso_flops) ARM_FLAGS=(--architecture=multi_block --num_blocks="$NUM_BLOCKS")
             [[ "$NUM_BLOCKS" != 16 ]] && RUN+="_b${NUM_BLOCKS}" ;;
  iso_param) ARM_FLAGS=(--architecture=single_block) ;;
  iso_mem)   ARM_FLAGS=(--architecture=fprm_mem --act_iters="${MAX_THINK_ITERS}:u")
             [[ "$MAX_THINK_ITERS" != 16 ]] && RUN+="_it${MAX_THINK_ITERS}" ;;
  mem_halt)  ARM_FLAGS=(--architecture=fprm_mem --act_iters="${MAX_THINK_ITERS}:v" --halt_criterion="$HALT_CRITERION")
             [[ "$MAX_THINK_ITERS" != 16 ]] && RUN+="_it${MAX_THINK_ITERS}"
             case "$HALT_CRITERION" in
               sigmoid) ;;
               kl|l2) ARM_FLAGS+=(--halt_threshold="$HALT_THRESHOLD"); RUN+="_h${HALT_CRITERION}_ht${HALT_THRESHOLD}" ;;
               *) echo "HALT_CRITERION must be sigmoid, kl or l2, got $HALT_CRITERION"; exit 1 ;;
             esac ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
[[ "$MIN_THINK_ITERS" =~ ^[1-9][0-9]*$ ]] || { echo "MIN_THINK_ITERS must be an integer >= 1, got $MIN_THINK_ITERS"; exit 1; }
[[ "$MIN_THINK_ITERS" != 2 ]] && { ARM_FLAGS+=(--min_think_iters="$MIN_THINK_ITERS"); RUN+="_minit${MIN_THINK_ITERS}"; }
if [[ "$ARM" == mem_halt && "$HALT_CRITERION" == sigmoid ]]; then
  case "$HALT_SG_SURVIVAL" in
    0) ;;
    1) ARM_FLAGS+=(--halt_sg_survival); RUN+="_hsgs" ;;
    *) echo "HALT_SG_SURVIVAL must be 0 or 1, got $HALT_SG_SURVIVAL"; exit 1 ;;
  esac
  awk -v c="$HALT_ENTROPY_COST" 'BEGIN { exit !(c + 0 == c && c >= 0) }' || { echo "HALT_ENTROPY_COST must be >= 0, got $HALT_ENTROPY_COST"; exit 1; }
  awk -v c="$HALT_ENTROPY_COST" 'BEGIN { exit !(c > 0) }' && { ARM_FLAGS+=(--halt_entropy_cost="$HALT_ENTROPY_COST"); RUN+="_hent${HALT_ENTROPY_COST}"; }
  case "$HALT_HEAD_GRAD" in
    0) ;;
    1) ARM_FLAGS+=(--halt_head_grad); RUN+="_hgrad" ;;
    *) echo "HALT_HEAD_GRAD must be 0 or 1, got $HALT_HEAD_GRAD"; exit 1 ;;
  esac
  RUN+="_hlin"  # linear halting head
fi
if [[ "$ARM" == mem_halt ]]; then
  case "$HALT_KL_REVERSE" in
    0) ;;
    1) ARM_FLAGS+=(--halt_kl_reverse); RUN+="_rkl" ;;
    *) echo "HALT_KL_REVERSE must be 0 or 1, got $HALT_KL_REVERSE"; exit 1 ;;
  esac
fi  # other arms ignore HALT_ENTROPY_COST / HALT_HEAD_GRAD / HALT_KL_REVERSE (shared study settings)
if [[ "$ARM" =~ ^(iso_mem|mem_halt)$ ]]; then
  ARM_FLAGS+=(--readout_token=thought); RUN+="_rthought"  # SEP token in every pass since 25 Oct
  [[ "$VALUE_ITERS" =~ ^[1-9][0-9]*$ ]] || { echo "VALUE_ITERS must be an integer >= 1, got $VALUE_ITERS"; exit 1; }
  ARM_FLAGS+=(--value_iters="$VALUE_ITERS"); [[ "$VALUE_ITERS" != "$MAX_THINK_ITERS" ]] && RUN+="_vi${VALUE_ITERS}"
  case "$POLICY_PARAM" in
    last) ;;
    sum) ARM_FLAGS+=(--policy_param=sum); RUN+="_psum"
         case "$LOGIT_SUM_STOP_GRAD" in
           0) ;;
           1) ARM_FLAGS+=(--logit_sum_stop_grad); RUN+="_lsg" ;;
           *) echo "LOGIT_SUM_STOP_GRAD must be 0 or 1, got $LOGIT_SUM_STOP_GRAD"; exit 1 ;;
         esac ;;
    *) echo "POLICY_PARAM must be last or sum, got $POLICY_PARAM"; exit 1 ;;
  esac
fi
case "$THOUGHT_NORM" in
  0) ;;
  1) [[ "$ARM" =~ ^(iso_mem|mem_halt)$ ]] \
       || { echo "THOUGHT_NORM=1 needs arm iso_mem or mem_halt"; exit 1; }
     ARM_FLAGS+=(--cot_thought_norm); RUN+="_tn" ;;
  *) echo "THOUGHT_NORM must be 0 or 1, got $THOUGHT_NORM"; exit 1 ;;
esac
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
