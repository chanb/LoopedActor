# The fprm_mem study, sourced by slurm/submit_mem.sh and run_mem_local.sh: iso_mem at every cap in ISO_ITERS (fixed
# compute) and mem_halt at every cap in MEM_HALT_ITERS (learned halting) on all four environments. Every run: bfloat16 + GELU
# MLP, DISCOUNT_ACT=0.99, DISCOUNT_COMPUTE=0.999 (unless set, see DISCOUNT_COMPUTES), each env's default board / episode settings.
#   ISO_ITERS   iso_mem caps (default "1 2 3 4")
#   MEM_HALT_ITERS  mem_halt caps (iso_mem with stochastic learned halting; default empty = none; e.g.
#               MEM_HALT_ITERS="4" ISO_ITERS= runs only mem_halt; ISO_ITERS= runs no iso_mem). Same cost as iso_mem at the same cap
#               (every pass runs); HALT_ENTROPY_COST / HALT_MAX_SKIP / HALT_HEAD_GRAD / HALT_KL_REVERSE are read by train.sh
#   VALUE_EXPECTILES  value-loss expectiles, crossed with every config above (default "0.5" = MSE; e.g. "0.5 0.7 0.9"
#               multiplies the study by 3; train.sh adds _ve<tau> to the run name for tau != 0.5)
#   TARGET_KL, NUM_EPOCHS  passed to every run when set (train.sh adds _tkl<k> / _ep<n> to the run name; slurm job names
#               get the same tag; NUM_EPOCHS also rescales the time estimates)
#   NUM_TIMESTEPS  total env steps per run, passed to every run when set (default 100000000; train.sh adds _ts<n/1e6>M
#               to the run name, slurm job names get the same tag; the time estimates scale linearly with it)
#   DISCOUNT_COMPUTES  per-extra-think-iteration discounts, crossed with every config above like VALUE_EXPECTILES
#               (default "${DISCOUNT_COMPUTE:-0.999}", so DISCOUNT_COMPUTE=g still sets a single value; e.g.
#               DISCOUNT_COMPUTES="0.9 0.99 0.999" multiplies the study by 3; train.sh adds _gc<g> to the run name when
#               != 1.0; slurm job names get _gc<g> when g != 0.999)
#   MIN_RANDOM_MOVES  slidingpuzzle: training scrambles of a uniform length in [k, NUM_RANDOM_MOVES] (train.sh adds
#               _rmin<k> to the run name; slurm job names get _rmin<k>; other envs ignore it)
#   RANDOM_MOVES_ZIPF  with MIN_RANDOM_MOVES: length k drawn with p(k) ∝ k^-s (default 0 = uniform; _rzipf<s>)
MEM_STUDY_ENVS="sokoban rushhour lightsout slidingpuzzle"
MEM_STUDY_SEEDS="1 2 3 4 5 6 7 8 9 10"
MEM_STUDY_COMMON="COMPUTE_DTYPE=bfloat16 MLP_TYPE=gelu DISCOUNT_ACT=0.99"   # DISCOUNT_COMPUTE: per config
MEM_STUDY_TAG="${MEM_STUDY_TAG:-}"   # appended to slurm job names (preserves a caller-set prefix/salt, if any)
[[ -n "${MIN_RANDOM_MOVES:-}" ]] && { MEM_STUDY_COMMON+=" MIN_RANDOM_MOVES=${MIN_RANDOM_MOVES}"; MEM_STUDY_TAG+="_rmin${MIN_RANDOM_MOVES}"; }
[[ -n "${RANDOM_MOVES_ZIPF:-}" ]] && { MEM_STUDY_COMMON+=" RANDOM_MOVES_ZIPF=${RANDOM_MOVES_ZIPF}"; MEM_STUDY_TAG+="_rzipf${RANDOM_MOVES_ZIPF}"; }
[[ -n "${HALT_ENTROPY_COST:-}" ]] && { MEM_STUDY_COMMON+=" HALT_ENTROPY_COST=${HALT_ENTROPY_COST}"; MEM_STUDY_TAG+="_hent${HALT_ENTROPY_COST}"; }
[[ -n "${HALT_MAX_SKIP:-}" ]] && { MEM_STUDY_COMMON+=" HALT_MAX_SKIP=${HALT_MAX_SKIP}"; MEM_STUDY_TAG+="_hk${HALT_MAX_SKIP}"; }
[[ "${HALT_HEAD_GRAD:-0}" == 1 ]] && { MEM_STUDY_COMMON+=" HALT_HEAD_GRAD=1"; MEM_STUDY_TAG+="_hgrad"; }
[[ "${HALT_KL_REVERSE:-0}" == 1 ]] && { MEM_STUDY_COMMON+=" HALT_KL_REVERSE=1"; MEM_STUDY_TAG+="_rkl"; }
[[ -n "${TARGET_KL:-}" ]] && { MEM_STUDY_COMMON+=" TARGET_KL=${TARGET_KL}"; MEM_STUDY_TAG+="_tkl${TARGET_KL}"; }
[[ -n "${NUM_EPOCHS:-}" ]] && { MEM_STUDY_COMMON+=" NUM_EPOCHS=${NUM_EPOCHS}"; MEM_STUDY_TAG+="_ep${NUM_EPOCHS}"; }
if [[ -n "${NUM_TIMESTEPS:-}" ]]; then
  MEM_STUDY_COMMON+=" NUM_TIMESTEPS=${NUM_TIMESTEPS}"
  (( NUM_TIMESTEPS % 1000000 == 0 )) && MEM_STUDY_TAG+="_ts$(( NUM_TIMESTEPS / 1000000 ))M" || MEM_STUDY_TAG+="_ts${NUM_TIMESTEPS}"
fi
# "<config name>|<env vars> <arm>"
MEM_STUDY_CONFIGS=()
for _gc in ${DISCOUNT_COMPUTES:-${DISCOUNT_COMPUTE:-0.999}}; do
for _ve in ${VALUE_EXPECTILES:-0.5}; do
  _sfx=""; [[ "$_ve" != 0.5 ]] && _sfx+="_ve${_ve}"; [[ "$_gc" != 0.999 ]] && _sfx+="_gc${_gc}"
  for _t in ${ISO_ITERS-1 2 3 4}; do MEM_STUDY_CONFIGS+=("iso_mem_it${_t}${_sfx}|MAX_THINK_ITERS=${_t} VALUE_EXPECTILE=${_ve} DISCOUNT_COMPUTE=${_gc} iso_mem"); done
  for _t in ${MEM_HALT_ITERS-}; do MEM_STUDY_CONFIGS+=("mem_halt_it${_t}${_sfx}|MAX_THINK_ITERS=${_t} VALUE_EXPECTILE=${_ve} DISCOUNT_COMPUTE=${_gc} mem_halt"); done
done
done
unset _t _ve _gc _sfx

# Measured cost of one run alone (bf16, GELU, each env's train.sh settings: 1024 envs x 64 steps, 64 minibatches x 4
# epochs, 100M steps; RTX A4500 - the L40S should be at least as fast):
#   env           T=1: steps/s  hours   T=4: steps/s  hours   peak GPU mem (T=1 / T=4)   host RAM
#   lightsout          42.0k    0.7          9.6k     2.9     0.40 / 0.54 GiB             2.7 GiB
#   slidingpuzzle      46.3k    0.6         12.6k     2.2     0.32 / 0.32 GiB             2.6 GiB
#   rushhour          >14.6k   <1.9         >3.5k    <7.9     1.06 / 1.13 GiB             3.7 GiB   (shared GPU: speeds are lower bounds)
#   sokoban             7.6k    3.7          1.8k    15.6     2.71 / 2.98 GiB             6.1 GiB
# mem_halt and iso_mem at the same cap cost the same (halting does not skip compute); the cost of a cap
# T is interpolated linearly through the T=1 and T=4 measurements (each think iteration is one more core call).
# lightsout is refit to the L40S runs of looped_actor_gamma_ablation (5x4, T = 1..16, GPU-hours per run with 2-3 runs
# per GPU): ~1.25-1.5x the old "0.25 1.0" at every T, which timed out the T=16 jobs; "0.3 1.35" stays within SAFETY.
declare -A MEM_STUDY_HOURS=(  # "<t0> <hours at t0> <t1> <hours at t1>"; mem_study_hours interpolates/extrapolates
  # linearly through these two (cap, hours) points.
  [lightsout]="1 0.3 4 1.35" [slidingpuzzle]="1 0.6 4 2.2" [rushhour]="1 1.9 4 7.9" [sokoban]="1 3.7 4 15.6"
  # slidingpuzzle with GRID=4 / GRID=5 (L40S runs of looped_actor_slidingpuzzle): the cost is far from linear in T
  # (T=4 ~ T=8), so these use far-apart anchor points instead of T=1/T=4.
  [slidingpuzzle4]="4 1.25 16 1.85"
  # slidingpuzzle5 refit 2026-10-25 from real wandb_mode=online training/sps on the running fprm_mem_halt
  # (STABLE_SOFTMAX=1, bf16, gelu) jobs of the iso_mem readout sweep, both already at NUM_EPOCHS=2: T=8 steady sps
  # ~4990 (seeds 2-3, pack2of3, 3-way GPU sharing) -> 7.34h incl. ~100 evals' overhead; T=16 steady sps ~2282
  # (seed 1, pack1of3) -> 13.83h. Stored here as their NUM_EPOCHS=4-equivalent (/0.525), since mem_study_hours()
  # re-applies the epoch scale on every call - storing the already-scaled NUM_EPOCHS=2 values directly would
  # double-discount them for any NUM_EPOCHS=2 caller (as a first pass at this fix briefly did).
  # The old "2.2 3.1" (T=4/T=16) pair was from an earlier/cheaper variant of this architecture and underestimated
  # T=16 by >2x - it put pack1of3 (5xT16 + T8 + T4, 3-concurrent) at a 23:59:00 limit against a real ~27.6h need.
  [slidingpuzzle5]="8 13.981 16 26.343")
declare -A MEM_STUDY_GPU_FRACTION=(  # XLA memory fraction per run on a ~45GiB L40S: >= 2x the measured peak, room for T > 4
  [lightsout]=0.08 [slidingpuzzle]=0.08 [rushhour]=0.10 [sokoban]=0.15)
declare -A MEM_STUDY_HOST_GB=(  # host RAM per run (measured + headroom)
  [lightsout]=4 [slidingpuzzle]=4 [rushhour]=5 [sokoban]=8)

mem_study_hours() {  # mem_study_hours <env> <cap>: estimated hours of one run alone
  # (measured with 4 epochs and 100M steps; ~95% of the time is the update, which scales with NUM_EPOCHS; all of it
  # scales with NUM_TIMESTEPS)
  local key="$1"
  [[ -n "${GRID:-}" && -n "${MEM_STUDY_HOURS[$1${GRID}]:-}" ]] && key="$1${GRID}"
  local t0 h0 t1 h1; read -r t0 h0 t1 h1 <<< "${MEM_STUDY_HOURS[$key]}"
  awk -v a="$h0" -v b="$h1" -v t="$2" -v t0="$t0" -v t1="$t1" -v e="${NUM_EPOCHS:-4}" -v n="${NUM_TIMESTEPS:-100000000}" \
    'BEGIN { printf "%.2f", (a + (b - a) * (t - t0) / (t1 - t0)) * (0.05 + 0.95 * e / 4) * n / 1e8 }'
}

mem_study_cap() {  # mem_study_cap "<env vars>": the MAX_THINK_ITERS value in a config's env vars
  local v; for v in $1; do [[ "$v" == MAX_THINK_ITERS=* ]] && echo "${v#MAX_THINK_ITERS=}"; done; return 0
}
