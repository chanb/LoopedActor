#!/bin/bash
# Train one arm of the sliding puzzle study.
#   slidingpuzzle/scripts/train.sh <arm> <seed> [out_dir] [device]
#   arm: looped | iso_flops | iso_param | iso_loop | iso_mem | mem_halt
# Optional environment variables:
#   GRID=N              board size N x N (default 3 = 8-puzzle)
#   EPISODE_LENGTH=N    steps per training episode (default 40; also used by the eval/train_* episodes)
#   EVAL_EPISODE_LENGTH=N  steps per evaluation-scramble (eval/test_*) episode (default EPISODE_LENGTH)
#   NUM_RANDOM_MOVES=K  training scramble length (random walk of the blank; default 100)
#   MIN_RANDOM_MOVES=k  > 0: each training scramble's length is drawn uniformly from [k, NUM_RANDOM_MOVES]
#                       (eval/test_* scrambles stay fixed at EVAL_NUM_RANDOM_MOVES; eval/train_* episodes use the mix;
#                       default 0 = fixed length)
#   RANDOM_MOVES_ZIPF=s  MIN_RANDOM_MOVES > 0: draw the length with p(k) ∝ k^-s instead of uniformly (default 0 = uniform;
#                       s = 1 puts ~half of [1, 200] on k <= 10)
#   EVAL_NUM_RANDOM_MOVES=K  evaluation scramble length (default 200)
#   DISCOUNT_ACT=g      per-decision discount (default 0.99)
#   DISCOUNT_COMPUTE=g  per-extra-think-iteration discount (default 0.999; 1.0 = standard return;
#                       = DISCOUNT_ACT gives the RAMDP return)
#   MAX_THINK_ITERS=N   think-iteration cap for looped / iso_loop / iso_mem / mem_halt (default 16)
#   NUM_BLOCKS=N        number of untied cores for iso_flops (default 16)
#   THOUGHT_NORM=1      iso_mem / mem_halt: RMS-normalize each thought before feeding it back (default 0)
#   HALT_KL=k           looped: halt once the policy KL between passes < k (default 1e-3);
#                       mem_halt with HALT_HEAD=distance: the KL at the center of the MLP's input
#   HALT_ENTROPY_COST=c  mem_halt: entropy bonus on the think-time distribution p(c|s) (default 0)
#   HALT_HEAD_GRAD=0|1  mem_halt: let the halting loss backpropagate into the trunk through the halting head (default 0)
#   HALT_KL_REVERSE=0|1  mem_halt: halting KL_i = KL(pi_{i-1} || pi_i), under the acting policy (default 0)
#   STABLE_KL=0|1       mem_halt: halting KL from the logit difference, no fp32 rounding floor (default 0, run names get _skl)
#   STABLE_SOFTMAX=0|1  iso_mem / mem_halt: collapse-free log-softmax for the action policy (mem_halt also: the
#                       halting head) (default 0, run names get _ssm)
#   HALT_MAX_SKIP=K     mem_halt: each halting decision picks k in 0..K more passes from softmax(MLP(...)) (HALT_HEAD)
#                       (0 = halt; decide again after k passes; scored by the sampled decision path). K = 1 is a yes/no
#                       halt at every pass (default 1; run names get _hk<K>)
#   HALT_HEAD=none|distance|raw  mem_halt: the halting head, a GELU MLP on the readout latent h_n (_hhnone), on log S_(n+1)
#                       (_hhdist) or on [z_n, z_(n+1)] (_hhraw) (default distance). distance / raw look ahead: decide
#                       on pass c+1, act with pi_c (_look)
#   HALT_SIGNAL=kl|latent  mem_halt, HALT_HEAD=distance|raw: z = the policies, S = KL; or z = the readout latents h,
#                       S = ||h_(n+1) - h_n|| (_hlat) (default kl)
#   HALT_RES=r          mem_halt, HALT_SIGNAL=latent, HALT_HEAD=distance: the L2 distance at the center of the MLP's
#                       input (default 0.1)
#   HALT_HEAD_WIDTH=W   mem_halt: hidden width of the halting MLP (default 32; run names get W if not 32)
#   READOUT=thought|sep  iso_mem / mem_halt: every pass runs [cells, SEP, thought] (SEP's latent carried, the thought's K/V kept in
#                       memory); read actions / value / halting from the newest thought or from SEP (default thought;
#                       run names get _rthought / _rsep)
#   MIN_THINK_ITERS=n   halting arms: earliest pass that may halt, >= 1 (default 2; mem_halt: 1 lets pass 1 halt on
#                       KL(pi_1 || uniform))
#   TOKENIZER=t         observation tokenization, all arms (default cell; see models/tokenizers.py):
#                         cell = one token per board cell, flat = one token per observation value,
#                         conv = 3x3 conv, one token per output channel
#   TOKENIZER_CONV_FEATURES=N  TOKENIZER=conv: conv output channels = number of tokens (default 64)
#   MLP_TYPE=swiglu|gelu  block MLP of every architecture (default swiglu)
#   EXPANSION=x         MLP hidden width = x * d (* 2/3 for swiglu) before rounding to MLP_ROUND_TO (default 4)
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
# Non-default values are appended to the run directory name (e.g. looped_it8_g4_s0).
# All arms: PPO on procedurally generated scrambles (slidingpuzzle-<N>x<N>: random walk of the blank from the
# solved board; training scrambles of NUM_RANDOM_MOVES, in-training evaluation (eval/test_* metrics) on
# EVAL_NUM_RANDOM_MOVES), sparse reward (1 on solve), 1024 envs x 64 steps per rollout, 64 minibatches x 4
# epochs, 100M env steps, 2-layer d=128 core, per-cell one-hot tile tokens, readout action head (move the blank
# up / right / down / left), global grad-norm clip 1.0, loss on the final readout for every arm.
# Defaults follow the Stoix sliding tile sweeps (3x3, time limit 40, 100 / 200 scramble moves).
#   looped    : weight-tied core, cap 16 think iterations, policy-KL halting (1e-3)
#   iso_flops : 16 untied cores stacked (16x the parameters)
#   iso_param : one core applied once
#   iso_loop  : weight-tied core, always runs the full 16 think iterations (no halting)
#   iso_mem   : fprm's looped board plus a SEP token and a memory of thought tokens (cells, SEP and the newest thought
#               attend to each other and to earlier thoughts' K/V), always runs the full 16 passes (no halting)
#   mem_halt  : iso_mem with stochastic learned halting: decisions k ~ softmax(MLP(...)) (HALT_HEAD) pick how many more
#               passes to run; the PPO ratio uses log p(path|s) + log pi_c(a|s); value from the last pass
#               (models/fprm_mem_halt.py)
# Usage:
# MAX_THINK_ITERS=4 slidingpuzzle/scripts/train.sh looped 1 exp/slidingpuzzle 0 &
# GRID=4 NUM_RANDOM_MOVES=200 EVAL_NUM_RANDOM_MOVES=400 EPISODE_LENGTH=80 slidingpuzzle/scripts/train.sh looped 1 exp/slidingpuzzle 1 &
#
set -euo pipefail
ARM="${1:?arm}"; SEED="${2:?seed}"; OUT="${3:-exp/slidingpuzzle}"; DEVICE="${4:-0}";
GRID="${GRID:-3}"; EPISODE_LENGTH="${EPISODE_LENGTH:-40}"; EVAL_EPISODE_LENGTH="${EVAL_EPISODE_LENGTH:-$EPISODE_LENGTH}"
NUM_RANDOM_MOVES="${NUM_RANDOM_MOVES:-100}"; MIN_RANDOM_MOVES="${MIN_RANDOM_MOVES:-0}"; RANDOM_MOVES_ZIPF="${RANDOM_MOVES_ZIPF:-0}"; EVAL_NUM_RANDOM_MOVES="${EVAL_NUM_RANDOM_MOVES:-200}"
DISCOUNT_ACT="${DISCOUNT_ACT:-0.99}"; DISCOUNT_COMPUTE="${DISCOUNT_COMPUTE:-0.999}"
MAX_THINK_ITERS="${MAX_THINK_ITERS:-16}"; NUM_BLOCKS="${NUM_BLOCKS:-16}"; THOUGHT_NORM="${THOUGHT_NORM:-0}"
MLP_TYPE="${MLP_TYPE:-swiglu}"; MLP_ROUND_TO="${MLP_ROUND_TO:-256}"; EXPANSION="${EXPANSION:-4}"; COMPUTE_DTYPE="${COMPUTE_DTYPE:-float32}"; DYNAMIC_LOSS_SCALE="${DYNAMIC_LOSS_SCALE:-1}"; NORMALIZE_OBS="${NORMALIZE_OBS:-0}"; VALUE_EXPECTILE="${VALUE_EXPECTILE:-0.5}"; TARGET_KL="${TARGET_KL:-0}"; NUM_EPOCHS="${NUM_EPOCHS:-4}"
HALT_KL="${HALT_KL:-1e-3}"
HALT_ENTROPY_COST="${HALT_ENTROPY_COST:-0}"; HALT_HEAD_GRAD="${HALT_HEAD_GRAD:-0}"; HALT_KL_REVERSE="${HALT_KL_REVERSE:-0}"; STABLE_KL="${STABLE_KL:-0}"; STABLE_SOFTMAX="${STABLE_SOFTMAX:-0}"; HALT_SIGNAL="${HALT_SIGNAL:-kl}"; HALT_RES="${HALT_RES:-0.1}"; HALT_MAX_SKIP="${HALT_MAX_SKIP:-1}"; HALT_HEAD="${HALT_HEAD:-distance}"; HALT_HEAD_WIDTH="${HALT_HEAD_WIDTH:-32}"; READOUT="${READOUT:-thought}"; MIN_THINK_ITERS="${MIN_THINK_ITERS:-2}"
TOKENIZER="${TOKENIZER:-cell}"; TOKENIZER_CONV_FEATURES="${TOKENIZER_CONV_FEATURES:-64}"
RUN="$ARM"
case "$ARM" in
  looped)    ARM_FLAGS=(--architecture=fprm --max_think_iters="$MAX_THINK_ITERS" --halt_criterion=kl --halt_kl="$HALT_KL")
             [[ "$MAX_THINK_ITERS" != 16 ]] && RUN+="_it${MAX_THINK_ITERS}" ;;
  iso_flops) ARM_FLAGS=(--architecture=multi_block --num_blocks="$NUM_BLOCKS")
             [[ "$NUM_BLOCKS" != 16 ]] && RUN+="_b${NUM_BLOCKS}" ;;
  iso_param) ARM_FLAGS=(--architecture=single_block) ;;
  iso_loop)  ARM_FLAGS=(--architecture=fprm --max_think_iters="$MAX_THINK_ITERS" --halt_criterion=kl --halt_kl=-1.0)
             [[ "$MAX_THINK_ITERS" != 16 ]] && RUN+="_it${MAX_THINK_ITERS}" ;;
  iso_mem)   ARM_FLAGS=(--architecture=fprm_mem --max_think_iters="$MAX_THINK_ITERS" --halt_criterion=kl --halt_kl=-1.0)
             [[ "$MAX_THINK_ITERS" != 16 ]] && RUN+="_it${MAX_THINK_ITERS}" ;;
  mem_halt)  ARM_FLAGS=(--architecture=fprm_mem_halt --max_think_iters="$MAX_THINK_ITERS" --halt_criterion=kl --halt_kl="$HALT_KL")
             [[ "$MAX_THINK_ITERS" != 16 ]] && RUN+="_it${MAX_THINK_ITERS}" ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
[[ "$ARM" =~ ^(looped|mem_halt)$ && "$HALT_KL" != 1e-3 ]] && RUN+="_hkl${HALT_KL}"
[[ "$MIN_THINK_ITERS" =~ ^[1-9][0-9]*$ ]] || { echo "MIN_THINK_ITERS must be an integer >= 1, got $MIN_THINK_ITERS"; exit 1; }
[[ "$MIN_THINK_ITERS" != 2 ]] && { ARM_FLAGS+=(--min_think_iters="$MIN_THINK_ITERS"); RUN+="_minit${MIN_THINK_ITERS}"; }
if [[ "$ARM" == mem_halt ]]; then
  awk -v c="$HALT_ENTROPY_COST" 'BEGIN { exit !(c + 0 == c && c >= 0) }' || { echo "HALT_ENTROPY_COST must be >= 0, got $HALT_ENTROPY_COST"; exit 1; }
  awk -v c="$HALT_ENTROPY_COST" 'BEGIN { exit !(c > 0) }' && { ARM_FLAGS+=(--halt_entropy_cost="$HALT_ENTROPY_COST"); RUN+="_hent${HALT_ENTROPY_COST}"; }
  case "$HALT_HEAD_GRAD" in
    0) ;;
    1) ARM_FLAGS+=(--halt_head_grad); RUN+="_hgrad" ;;
    *) echo "HALT_HEAD_GRAD must be 0 or 1, got $HALT_HEAD_GRAD"; exit 1 ;;
  esac
  [[ "$HALT_MAX_SKIP" =~ ^[1-9][0-9]*$ ]] || { echo "HALT_MAX_SKIP must be an integer >= 1, got $HALT_MAX_SKIP"; exit 1; }
  ARM_FLAGS+=(--halt_max_skip="$HALT_MAX_SKIP"); RUN+="_hk${HALT_MAX_SKIP}"
  case "$HALT_HEAD" in
    none|distance|raw) ARM_FLAGS+=(--halt_head="$HALT_HEAD" --halt_head_width="$HALT_HEAD_WIDTH"); RUN+="_hh${HALT_HEAD/distance/dist}"
                       [[ "$HALT_HEAD_WIDTH" != 32 ]] && RUN+="${HALT_HEAD_WIDTH}"
                       [[ "$HALT_HEAD" != none ]] && RUN+="_look" ;;
    *) echo "HALT_HEAD must be none, distance or raw, got $HALT_HEAD"; exit 1 ;;
  esac
  case "$HALT_KL_REVERSE" in
    0) ;;
    1) ARM_FLAGS+=(--halt_kl_reverse); RUN+="_rkl" ;;
    *) echo "HALT_KL_REVERSE must be 0 or 1, got $HALT_KL_REVERSE"; exit 1 ;;
  esac
  case "$STABLE_KL" in
    0) ;;
    1) ARM_FLAGS+=(--stable_kl); RUN+="_skl" ;;
    *) echo "STABLE_KL must be 0 or 1, got $STABLE_KL"; exit 1 ;;
  esac
  case "$HALT_SIGNAL" in
    kl) ;;
    latent) ARM_FLAGS+=(--halt_signal=latent --halt_res="$HALT_RES"); RUN+="_hlat"; [[ "$HALT_RES" != 0.1 ]] && RUN+="${HALT_RES}" ;;
    *) echo "HALT_SIGNAL must be kl or latent (no signal: HALT_HEAD=none), got $HALT_SIGNAL"; exit 1 ;;
  esac
fi  # other arms ignore HALT_ENTROPY_COST / HALT_HEAD_GRAD / HALT_KL_REVERSE / STABLE_KL / HALT_MAX_SKIP (shared study settings)
if [[ "$ARM" =~ ^(iso_mem|mem_halt)$ ]]; then
  case "$STABLE_SOFTMAX" in
    0) ;;
    1) ARM_FLAGS+=(--stable_softmax); RUN+="_ssm" ;;
    *) echo "STABLE_SOFTMAX must be 0 or 1, got $STABLE_SOFTMAX"; exit 1 ;;
  esac
  case "$READOUT" in  # SEP is in every pass since 25 Oct; READOUT only picks the readout token
    thought|sep) ARM_FLAGS+=(--readout_token="$READOUT"); RUN+="_r${READOUT}" ;;
    *) echo "READOUT must be thought or sep, got $READOUT"; exit 1 ;;
  esac
fi  # other arms ignore STABLE_SOFTMAX / READOUT
case "$THOUGHT_NORM" in
  0) ;;
  1) [[ "$ARM" =~ ^(iso_mem|mem_halt)$ ]] \
       || { echo "THOUGHT_NORM=1 needs arm iso_mem or mem_halt"; exit 1; }
     ARM_FLAGS+=(--cot_thought_norm); RUN+="_tn" ;;
  *) echo "THOUGHT_NORM must be 0 or 1, got $THOUGHT_NORM"; exit 1 ;;
esac
case "$TOKENIZER" in
  cell) ;;
  flat) ARM_FLAGS+=(--tokenizer=flat); RUN+="_tkflat" ;;
  conv) ARM_FLAGS+=(--tokenizer=conv --tokenizer_conv_features="$TOKENIZER_CONV_FEATURES")
        RUN+="_tkconv${TOKENIZER_CONV_FEATURES}" ;;
  *) echo "TOKENIZER must be cell, flat or conv, got $TOKENIZER"; exit 1 ;;
esac
[[ "$GRID" =~ ^[0-9]+$ ]] || { echo "GRID must be an integer N (board N x N), got $GRID"; exit 1; }
[[ "$GRID" != 3 ]] && RUN+="_g${GRID}"
[[ "$EPISODE_LENGTH" != 40 ]] && RUN+="_el${EPISODE_LENGTH}"
[[ "$EVAL_EPISODE_LENGTH" != "$EPISODE_LENGTH" ]] && RUN+="_eel${EVAL_EPISODE_LENGTH}"
[[ "$NUM_RANDOM_MOVES" != 100 ]] && RUN+="_rm${NUM_RANDOM_MOVES}"
[[ "$MIN_RANDOM_MOVES" =~ ^[0-9]+$ ]] && (( MIN_RANDOM_MOVES <= NUM_RANDOM_MOVES )) \
  || { echo "MIN_RANDOM_MOVES must be an integer in [0, NUM_RANDOM_MOVES=$NUM_RANDOM_MOVES], got $MIN_RANDOM_MOVES"; exit 1; }
(( MIN_RANDOM_MOVES > 0 )) && { ARM_FLAGS+=(--slidingpuzzle_min_random_moves="$MIN_RANDOM_MOVES"); RUN+="_rmin${MIN_RANDOM_MOVES}"; }
awk -v s="$RANDOM_MOVES_ZIPF" 'BEGIN { exit !(s + 0 == s && s >= 0) }' || { echo "RANDOM_MOVES_ZIPF must be >= 0, got $RANDOM_MOVES_ZIPF"; exit 1; }
if awk -v s="$RANDOM_MOVES_ZIPF" 'BEGIN { exit !(s > 0) }'; then
  (( MIN_RANDOM_MOVES > 0 )) || { echo "RANDOM_MOVES_ZIPF > 0 needs MIN_RANDOM_MOVES > 0"; exit 1; }
  ARM_FLAGS+=(--slidingpuzzle_random_moves_zipf="$RANDOM_MOVES_ZIPF"); RUN+="_rzipf${RANDOM_MOVES_ZIPF}"
fi
[[ "$EVAL_NUM_RANDOM_MOVES" != 200 ]] && RUN+="_erm${EVAL_NUM_RANDOM_MOVES}"
case "$MLP_TYPE" in
  swiglu) ;;
  gelu) ARM_FLAGS+=(--mlp_type=gelu); RUN+="_gelu" ;;
  *) echo "MLP_TYPE must be swiglu or gelu, got $MLP_TYPE"; exit 1 ;;
esac
[[ "$MLP_ROUND_TO" =~ ^[1-9][0-9]*$ ]] || { echo "MLP_ROUND_TO must be a positive integer, got $MLP_ROUND_TO"; exit 1; }
awk -v x="$EXPANSION" 'BEGIN { exit !(x + 0 == x && x > 0) }' || { echo "EXPANSION must be > 0, got $EXPANSION"; exit 1; }
[[ "$EXPANSION" != 4 ]] && { ARM_FLAGS+=(--fprm_expansion="$EXPANSION"); RUN+="_x${EXPANSION}"; }
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
[[ "$DISCOUNT_COMPUTE" != 1.0 ]] && RUN+="_gc${DISCOUNT_COMPUTE}"  # suffix kept vs 1.0 so new runs never reuse the old 1.0 run dirs
RUN+="_s${SEED}"
[[ "${PRINT_RUN_DIR:-0}" == 1 ]] && { echo "$OUT/${RUN}"; exit 0; }   # just print the run directory (slurm/submit_mem.sh)
# Every hyperparameter flag except the seed; hashed into the wandb group.
HPARAM_FLAGS=(--env_id="slidingpuzzle-${GRID}x${GRID}" --slidingpuzzle_episode_length="$EPISODE_LENGTH" --slidingpuzzle_eval_episode_length="$EVAL_EPISODE_LENGTH"
  --slidingpuzzle_num_random_moves="$NUM_RANDOM_MOVES" --slidingpuzzle_eval_num_random_moves="$EVAL_NUM_RANDOM_MOVES"
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
