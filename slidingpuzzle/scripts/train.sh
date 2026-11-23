#!/bin/bash
# Train one arm of the sliding puzzle study.
#   slidingpuzzle/scripts/train.sh <arm> <seed> [out_dir] [device]
#   arm: iso_flops | iso_param | iso_mem | mem_halt
# Optional environment variables:
#   GRID=N              board size N x N (default 3 = 8-puzzle)
#   EPISODE_LENGTH=N    steps per training episode (default 40; also used by the eval/train_* episodes)
#   EVAL_EPISODE_LENGTH=N  steps per evaluation-scramble (eval/test_*) episode (default EPISODE_LENGTH)
#   OPTIMAL_HORIZON=1   BANK only: an unsolved episode terminates (reward 0) once it has taken the board's optimal
#                       number of moves, so only optimal solutions are rewarded (_opth; default 0)
#   BANK=<train>-<eval>  draw training / evaluation boards from the level banks data/slidingpuzzle/<N>x<N>/<split>.npz
#                       instead of scrambling (e.g. train-test; _bank<train>-<eval>; *_RANDOM_MOVES are then unused)
#   NUM_RANDOM_MOVES=K  training scramble length (random walk of the blank; default 100)
#   MIN_RANDOM_MOVES=k  > 0: each training scramble's length is drawn uniformly from [k, NUM_RANDOM_MOVES]
#                       (eval/test_* scrambles stay fixed at EVAL_NUM_RANDOM_MOVES; eval/train_* episodes use the mix;
#                       default 0 = fixed length)
#   RANDOM_MOVES_ZIPF=s  MIN_RANDOM_MOVES > 0: draw the length with p(k) ∝ k^-s instead of uniformly (default 0 = uniform;
#                       s = 1 puts ~half of [1, 200] on k <= 10; s < 0 favours long scrambles, s = -1: p(k) ∝ k)
#   EVAL_NUM_RANDOM_MOVES=K  evaluation scramble length (default 200)
#   DISCOUNT_ACT=g      per-decision discount (default 0.99)
#   DISCOUNT_COMPUTE=g  per-extra-think-iteration discount (default 0.999; 1.0 = standard return;
#                       = DISCOUNT_ACT gives the RAMDP return)
#   MAX_THINK_ITERS=C   iso_mem / mem_halt: the think cap C of --act_iters=C:u (iso_mem) / C:v (mem_halt) (default 16)
#   VALUE_ITERS=n       iso_mem / mem_halt: core calls of the separate FPRM critic (default MAX_THINK_ITERS; _vi<n> otherwise)
#   POLICY_PARAM=last|sum|mix  iso_mem / mem_halt: pi_n from pass n's logits (last, default) or from their running sum (_psum);
#                          mix (mem_halt sigmoid only, _pmix): act with sum_(n<=c) w_n pi_n, w = the halt distribution cut at c
#   MIX_WEIGHT_STOP_GRAD=1 POLICY_PARAM=mix: no gradient into the halting head through the mixture weights (_mwsg)
#   LOGIT_SUM_STOP_GRAD=1  POLICY_PARAM=sum: no gradient through the logits of passes before c (_lsg)
#   NUM_BLOCKS=N        number of untied cores for iso_flops (default 16)
#   THOUGHT_NORM=1      iso_mem / mem_halt: RMS-normalize each thought before feeding it back (default 0)
#   HALT_CRITERION=sigmoid|kl|l2  mem_halt: learned stochastic halting head (sigmoid, default), or halt at pass n once
#                       KL(pi_(n+1) || pi_n) (kl) / ||rms_norm(h_(n+1)) - rms_norm(h_n)|| (l2) < HALT_THRESHOLD (_hkl / _hl2)
#   HALT_THRESHOLD=t    mem_halt, HALT_CRITERION=kl|l2: the threshold (default 1e-3; _ht<t>)
#   HALT_SG_SURVIVAL=1  mem_halt, sigmoid: log p(c|s) without gradient through the continue terms before c (_hsgs)
#   HALT_ENTROPY_COST=c  mem_halt, sigmoid (as every halting-head knob below): entropy bonus on the think-time distribution p(c|s) (default 0)
#   HALT_OBJECTIVE=pg|next|score  mem_halt: train the halting head through PPO's ratio (pg, default) or by scoring its softmax
#                       over k at the sampled c with J_(c+k) - J_c, J_n the clipped importance-weighted return of acting
#                       with pass n (_hobjnext), or by REINFORCE on log p(c|s) with weight pi_c(a|s) G -
#                       DISCOUNT_COMPUTE pi_(c+1)(a|s) G (_hobjscore)
#   HALT_CLIP=joint|path|each  mem_halt, HALT_OBJECTIVE=pg: one PPO clip over the joint ratio p(path) pi_c(a|s) (joint,
#                       default), or pi_c(a|s) clipped apart from p(path) as a whole (_hclippath) or from each decision (_hclipeach)
#   HALT_RHO_MAX=r      mem_halt, HALT_OBJECTIVE=next: clip of the ratio in J_n (default 2; _rho<r> if not 2)
#   HALT_PRIOR_COEF=b   mem_halt: weight of KL(p_n || q) with q a prior favouring continuing, decayed linearly to 0
#                       over training (default 0 = off; _hprior<b>)
#   HALT_PRIOR_HALT=q0  mem_halt, HALT_PRIOR_COEF > 0: the prior's q(k = 0); k >= 1 share the rest (default 0.1;
#                       _q<q0> if not 0.1)
#   HALT_ADAM_B1=b1 HALT_ADAM_B2=b2  mem_halt: Adam betas of the halting head only (default 0.9 / 0.999 as the rest;
#                       arXiv 2601.21739 suggests 0.99 / 0.99; run names get _hadam<b1>_<b2> if not the default)
#   HALT_UPDATE_PERIOD=p  mem_halt: trunk at every grad step, halting head only at every p-th (_hup<p>; default 1)
#   HALT_HEAD_GRAD=0|1  mem_halt: let the halting loss backpropagate into the trunk through the halting head (default 0)
#   HALT_KL_REVERSE=0|1  mem_halt: KL_i = KL(pi_{i-1} || pi_i), under the acting policy (default 0)
#   STABLE_KL=0|1       mem_halt: KL from the logit difference, no fp32 rounding floor (default 0, run names get _skl)
#   STABLE_SOFTMAX=0|1  iso_mem / mem_halt: collapse-free log-softmax for the action policy (mem_halt also: the
#                       halting head) (default 0, run names get _ssm)
#   mem_halt's sigmoid halting head is linear: h_n = sigmoid(w . rms_norm(h_n) + b), zero init (run names get _hlin)
#   READOUT=thought|sep  iso_mem / mem_halt: every pass runs [cells, SEP, thought] (SEP's latent carried, the thought's K/V kept in
#                       memory); read the actor's readouts from the newest thought or from SEP (default thought;
#                       run names get _rthought / _rsep)
#   MIN_THINK_ITERS=n   mem_halt: earliest pass that may halt, >= 1 (default 2)
#   TOKENIZER=t         observation tokenization, all arms (default cell; see models/tokenizers.py):
#                         cell = one token per board cell, flat = one token per observation value,
#                         conv = 3x3 conv, one token per output channel
#   TOKENIZER_CONV_FEATURES=N  TOKENIZER=conv: conv output channels = number of tokens (default 64)
#   ACTION_HEAD=neighbour  iso_mem / mem_halt: read the logit of each blank move from the cell latent of the tile that
#                       would slide (off-board moves masked) instead of the readout token's Dense(4) (_anbr; default: readout)
#   ACTION_HEAD=blank   iso_mem / mem_halt: 4 Dense(1) on the blank's own cell latent, move i on its i-th quarter of
#                       the features (rms-normed per quarter), unmasked (_ablank)
#   CELL_GEOMETRY=0     iso_mem / mem_halt, TOKENIZER=cell: cell tokens without the 6 row/col/edge features (_nogeo; default 1)
#   MLP_TYPE=swiglu|gelu  block MLP of every architecture (default swiglu)
#   EXPANSION=x         MLP hidden width = x * d (* 2/3 for swiglu) before rounding to MLP_ROUND_TO (default 4)
#   MLP_ROUND_TO=r      MLP hidden width rounded up to a multiple of r (default 256; at d=128 swiglu: 256 -> 512,
#                       64 -> 384, 1 -> 341)
#   COMPUTE_DTYPE=float32|float16|bfloat16  block computation dtype (default float32). bfloat16: ~1.6-1.75x faster, fp32 range;
#                       float16: slightly faster and more precise, but activations can overflow (multi_block NaNs)
#   DYNAMIC_LOSS_SCALE=0|1  COMPUTE_DTYPE=float16: dynamic loss scaling against fp16 gradient underflow (default 1)
#   NORMALIZE_OBS=0|1  running mean/std observation normalization (default 0 = raw observations; runs before this
#                       option always normalized)
#   LEARNING_RATE=lr   constant learning rate (default 1e-4; _lr<lr> otherwise)
#   OPTIMIZER=adam|muon  muon: Muon on the 2-D kernels of the looped core (actor and critic), Adam(W) on the rest (_muon; default adam)
#   WEIGHT_DECAY=w      decoupled weight decay (AdamW) on every param except the halting head's (default 0 = Adam; _wd<w>)
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
# Non-default values are appended to the run directory name (e.g. iso_mem_it8_g4_s0).
# All arms: PPO on procedurally generated scrambles (slidingpuzzle-<N>x<N>: random walk of the blank from the
# solved board; training scrambles of NUM_RANDOM_MOVES, in-training evaluation (eval/test_* metrics) on
# EVAL_NUM_RANDOM_MOVES), sparse reward (1 on solve), 1024 envs x 64 steps per rollout, 64 minibatches x 4
# epochs, 100M env steps, 2-layer d=128 core, per-cell one-hot tile tokens, readout action head (move the blank
# up / right / down / left), global grad-norm clip 1.0, loss on the final readout for every arm.
# Defaults follow the Stoix sliding tile sweeps (3x3, time limit 40, 100 / 200 scramble moves).
#   iso_flops : 16 untied cores stacked (16x the parameters)
#   iso_param : one core applied once
#   iso_mem   : looped FPRM board plus a SEP token and a memory of thought tokens (cells, SEP and the newest thought
#               attend to each other and to earlier thoughts' K/V), always acts with pass 16 (--act_iters=16:u); the
#               critic is a separate FPRM run for VALUE_ITERS core calls (models/fprm_mem_halt.py)
#   mem_halt  : iso_mem with a variable budget (--act_iters=16:v), halting by HALT_CRITERION
# Usage:
# MAX_THINK_ITERS=4 slidingpuzzle/scripts/train.sh iso_mem 1 exp/slidingpuzzle 0 &
# HALT_CRITERION=kl GRID=4 slidingpuzzle/scripts/train.sh mem_halt 1 exp/slidingpuzzle 1 &
#
set -euo pipefail
ARM="${1:?arm}"; SEED="${2:?seed}"; OUT="${3:-exp/slidingpuzzle}"; DEVICE="${4:-0}";
GRID="${GRID:-3}"; BANK="${BANK:-}"; EPISODE_LENGTH="${EPISODE_LENGTH:-40}"; EVAL_EPISODE_LENGTH="${EVAL_EPISODE_LENGTH:-$EPISODE_LENGTH}"
NUM_RANDOM_MOVES="${NUM_RANDOM_MOVES:-100}"; MIN_RANDOM_MOVES="${MIN_RANDOM_MOVES:-0}"; RANDOM_MOVES_ZIPF="${RANDOM_MOVES_ZIPF:-0}"; EVAL_NUM_RANDOM_MOVES="${EVAL_NUM_RANDOM_MOVES:-200}"
DISCOUNT_ACT="${DISCOUNT_ACT:-0.99}"; DISCOUNT_COMPUTE="${DISCOUNT_COMPUTE:-0.999}"
MAX_THINK_ITERS="${MAX_THINK_ITERS:-16}"; VALUE_ITERS="${VALUE_ITERS:-$MAX_THINK_ITERS}"; POLICY_PARAM="${POLICY_PARAM:-last}"; LOGIT_SUM_STOP_GRAD="${LOGIT_SUM_STOP_GRAD:-0}"; MIX_WEIGHT_STOP_GRAD="${MIX_WEIGHT_STOP_GRAD:-0}"; NUM_BLOCKS="${NUM_BLOCKS:-16}"; THOUGHT_NORM="${THOUGHT_NORM:-0}"
MLP_TYPE="${MLP_TYPE:-swiglu}"; MLP_ROUND_TO="${MLP_ROUND_TO:-256}"; EXPANSION="${EXPANSION:-4}"; COMPUTE_DTYPE="${COMPUTE_DTYPE:-float32}"; DYNAMIC_LOSS_SCALE="${DYNAMIC_LOSS_SCALE:-1}"; NORMALIZE_OBS="${NORMALIZE_OBS:-0}"; VALUE_EXPECTILE="${VALUE_EXPECTILE:-0.5}"; TARGET_KL="${TARGET_KL:-0}"; WEIGHT_DECAY="${WEIGHT_DECAY:-0}"; NUM_EPOCHS="${NUM_EPOCHS:-4}"
HALT_CRITERION="${HALT_CRITERION:-sigmoid}"; HALT_THRESHOLD="${HALT_THRESHOLD:-1e-3}"; HALT_SG_SURVIVAL="${HALT_SG_SURVIVAL:-0}"
HALT_ENTROPY_COST="${HALT_ENTROPY_COST:-0}"; HALT_HEAD_GRAD="${HALT_HEAD_GRAD:-0}"; HALT_ADAM_B1="${HALT_ADAM_B1:-0.9}"; HALT_ADAM_B2="${HALT_ADAM_B2:-0.999}"; HALT_KL_REVERSE="${HALT_KL_REVERSE:-0}"; STABLE_KL="${STABLE_KL:-0}"; STABLE_SOFTMAX="${STABLE_SOFTMAX:-0}"; READOUT="${READOUT:-thought}"; MIN_THINK_ITERS="${MIN_THINK_ITERS:-2}"
TOKENIZER="${TOKENIZER:-cell}"; TOKENIZER_CONV_FEATURES="${TOKENIZER_CONV_FEATURES:-64}"; ACTION_HEAD="${ACTION_HEAD:-readout}"; CELL_GEOMETRY="${CELL_GEOMETRY:-1}"
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
  case "${HALT_OBJECTIVE:-pg}" in
    pg) case "${HALT_CLIP:-joint}" in
          joint) ;;
          path|each) ARM_FLAGS+=(--halt_clip="$HALT_CLIP"); RUN+="_hclip${HALT_CLIP}" ;;
          *) echo "HALT_CLIP must be joint, path or each, got $HALT_CLIP"; exit 1 ;;
        esac ;;
    next) ARM_FLAGS+=(--halt_objective=next); RUN+="_hobjnext"
      [[ "${HALT_RHO_MAX:-2}" != 2 ]] && { ARM_FLAGS+=(--halt_rho_max="$HALT_RHO_MAX"); RUN+="_rho${HALT_RHO_MAX}"; } ;;
    score) ARM_FLAGS+=(--halt_objective=score); RUN+="_hobjscore" ;;
    *) echo "HALT_OBJECTIVE must be pg, next or score, got $HALT_OBJECTIVE"; exit 1 ;;
  esac
  if awk -v b="${HALT_PRIOR_COEF:-0}" 'BEGIN { exit !(b > 0) }'; then
    ARM_FLAGS+=(--halt_prior_coef="$HALT_PRIOR_COEF"); RUN+="_hprior${HALT_PRIOR_COEF}"
    [[ "${HALT_PRIOR_HALT:-0.1}" != 0.1 ]] && { ARM_FLAGS+=(--halt_prior_halt="$HALT_PRIOR_HALT"); RUN+="_q${HALT_PRIOR_HALT}"; }
  fi
  if [[ "${HALT_UPDATE_PERIOD:-1}" != 1 ]]; then
    ARM_FLAGS+=(--halt_update_period="$HALT_UPDATE_PERIOD"); RUN+="_hup${HALT_UPDATE_PERIOD}"
  fi
  if [[ "$HALT_ADAM_B1" != 0.9 || "$HALT_ADAM_B2" != 0.999 ]]; then
    ARM_FLAGS+=(--halt_adam_b1="$HALT_ADAM_B1" --halt_adam_b2="$HALT_ADAM_B2"); RUN+="_hadam${HALT_ADAM_B1}_${HALT_ADAM_B2}"
  fi
  case "$HALT_HEAD_GRAD" in
    0) ;;
    1) ARM_FLAGS+=(--halt_head_grad); RUN+="_hgrad" ;;
    *) echo "HALT_HEAD_GRAD must be 0 or 1, got $HALT_HEAD_GRAD"; exit 1 ;;
  esac
  RUN+="_hlin"  # linear halting head
fi  # other arms / criteria ignore the halting-head knobs (shared study settings)
if [[ "$ARM" == mem_halt ]]; then
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
fi  # other arms ignore HALT_KL_REVERSE / STABLE_KL
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
    mix) ARM_FLAGS+=(--policy_param=mix); RUN+="_pmix"
         case "$MIX_WEIGHT_STOP_GRAD" in
           0) ;;
           1) ARM_FLAGS+=(--mix_weight_stop_grad); RUN+="_mwsg" ;;
           *) echo "MIX_WEIGHT_STOP_GRAD must be 0 or 1, got $MIX_WEIGHT_STOP_GRAD"; exit 1 ;;
         esac ;;
    *) echo "POLICY_PARAM must be last, sum or mix, got $POLICY_PARAM"; exit 1 ;;
  esac
fi  # other arms ignore STABLE_SOFTMAX / READOUT / VALUE_ITERS / POLICY_PARAM
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
case "$ACTION_HEAD" in
  readout) ;;
  neighbour) [[ "$ARM" =~ ^(iso_mem|mem_halt)$ && "$TOKENIZER" == cell ]] \
               || { echo "ACTION_HEAD=neighbour needs arm iso_mem or mem_halt and TOKENIZER=cell"; exit 1; }
             ARM_FLAGS+=(--action_head=neighbour); RUN+="_anbr" ;;
  blank) [[ "$ARM" =~ ^(iso_mem|mem_halt)$ && "$TOKENIZER" == cell ]] \
           || { echo "ACTION_HEAD=blank needs arm iso_mem or mem_halt and TOKENIZER=cell"; exit 1; }
         ARM_FLAGS+=(--action_head=blank); RUN+="_ablank" ;;
  *) echo "ACTION_HEAD must be readout, neighbour or blank, got $ACTION_HEAD"; exit 1 ;;
esac
case "$CELL_GEOMETRY" in
  1) ;;
  0) [[ "$ARM" =~ ^(iso_mem|mem_halt)$ && "$TOKENIZER" == cell ]] \
       || { echo "CELL_GEOMETRY=0 needs arm iso_mem or mem_halt and TOKENIZER=cell"; exit 1; }
     ARM_FLAGS+=(--no-cell_geometry); RUN+="_nogeo" ;;
  *) echo "CELL_GEOMETRY must be 0 or 1, got $CELL_GEOMETRY"; exit 1 ;;
esac
[[ "$GRID" =~ ^[0-9]+$ ]] || { echo "GRID must be an integer N (board N x N), got $GRID"; exit 1; }
[[ "$GRID" != 3 ]] && RUN+="_g${GRID}"
[[ -z "$BANK" || "$BANK" =~ ^[a-z0-9_]+-[a-z0-9_]+$ ]] || { echo "BANK must be <train>-<eval>, got $BANK"; exit 1; }
[[ -n "$BANK" ]] && RUN+="_bank${BANK}"
[[ "$EPISODE_LENGTH" != 40 ]] && RUN+="_el${EPISODE_LENGTH}"
[[ "$EVAL_EPISODE_LENGTH" != "$EPISODE_LENGTH" ]] && RUN+="_eel${EVAL_EPISODE_LENGTH}"
case "${OPTIMAL_HORIZON:-0}" in
  0) ;;
  1) ARM_FLAGS+=(--slidingpuzzle_optimal_horizon); RUN+="_opth" ;;
  *) echo "OPTIMAL_HORIZON must be 0 or 1, got $OPTIMAL_HORIZON"; exit 1 ;;
esac
[[ "$NUM_RANDOM_MOVES" != 100 ]] && RUN+="_rm${NUM_RANDOM_MOVES}"
[[ "$MIN_RANDOM_MOVES" =~ ^[0-9]+$ ]] && (( MIN_RANDOM_MOVES <= NUM_RANDOM_MOVES )) \
  || { echo "MIN_RANDOM_MOVES must be an integer in [0, NUM_RANDOM_MOVES=$NUM_RANDOM_MOVES], got $MIN_RANDOM_MOVES"; exit 1; }
(( MIN_RANDOM_MOVES > 0 )) && { ARM_FLAGS+=(--slidingpuzzle_min_random_moves="$MIN_RANDOM_MOVES"); RUN+="_rmin${MIN_RANDOM_MOVES}"; }
awk -v s="$RANDOM_MOVES_ZIPF" 'BEGIN { exit !(s + 0 == s) }' || { echo "RANDOM_MOVES_ZIPF must be a number, got $RANDOM_MOVES_ZIPF"; exit 1; }
if awk -v s="$RANDOM_MOVES_ZIPF" 'BEGIN { exit !(s != 0) }'; then
  (( MIN_RANDOM_MOVES > 0 )) || { echo "RANDOM_MOVES_ZIPF != 0 needs MIN_RANDOM_MOVES > 0"; exit 1; }
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
awk -v w="$WEIGHT_DECAY" 'BEGIN { exit !(w + 0 == w && w >= 0) }' || { echo "WEIGHT_DECAY must be >= 0, got $WEIGHT_DECAY"; exit 1; }
awk -v w="$WEIGHT_DECAY" 'BEGIN { exit !(w > 0) }' && { ARM_FLAGS+=(--weight_decay="$WEIGHT_DECAY"); RUN+="_wd${WEIGHT_DECAY}"; }
case "${OPTIMIZER:-adam}" in
  adam) ;;
  muon) ARM_FLAGS+=(--optimizer=muon); RUN+="_muon" ;;
  *) echo "OPTIMIZER must be adam or muon, got $OPTIMIZER"; exit 1 ;;
esac
awk -v r="${LEARNING_RATE:-1e-4}" 'BEGIN { exit !(r + 0 == r && r > 0) }' || { echo "LEARNING_RATE must be > 0, got $LEARNING_RATE"; exit 1; }
[[ "${LEARNING_RATE:-1e-4}" != 1e-4 ]] && { ARM_FLAGS+=(--learning_rate="$LEARNING_RATE"); RUN+="_lr${LEARNING_RATE}"; }
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
HPARAM_FLAGS=(--env_id="slidingpuzzle-${GRID}x${GRID}${BANK:+-$BANK}" --slidingpuzzle_episode_length="$EPISODE_LENGTH" --slidingpuzzle_eval_episode_length="$EVAL_EPISODE_LENGTH"
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
