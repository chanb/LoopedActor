# The fprm_mem study, sourced by slurm/submit_mem.sh and run_mem_local.sh: iso_mem at every cap in ISO_ITERS (fixed
# compute) and mem at every cap in MEM_ITERS (policy-KL halting) on all four environments. Every run: bfloat16 + GELU
# MLP, DISCOUNT_ACT=0.99, DISCOUNT_COMPUTE=0.999 (unless set, see DISCOUNT_COMPUTES), each env's default board / episode settings.
#   ISO_ITERS   iso_mem caps (default "1 2 3 4")
#   MEM_ITERS   mem caps (default "2 3 4"; a cap of 1 is iso_mem T=1, since halting acts from pass 2; an empty list
#               (MEM_ITERS=) runs no mem configs, likewise ISO_ITERS=)
#   ISO_RIM_ITERS, RIM_ITERS  iso_rim / rim caps (default empty = none; e.g. ISO_RIM_ITERS="1 2" ISO_ITERS= MEM_ITERS= runs
#               only iso_rim). The time estimates below were measured for fprm_mem; rim's cost was not measured
#   MEM_HALT_ITERS  mem_halt caps (fprm_mem with stochastic learned halting; default empty = none; e.g.
#               MEM_HALT_ITERS="4" ISO_ITERS= MEM_ITERS= runs only mem_halt). Same cost as mem at the same cap
#               (every pass runs); halting from a head on the newest thought (config / job names get _hlat);
#               HALT_ENTROPY_COST / HALT_HEAD_GRAD are read by train.sh
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
#   VALUE_AT_MAX_ITERS=1  value from the MAX_THINK_ITERS-th pass's latent (train.sh adds _vT to the run name; slurm job
#               names get _vT; mem / iso_mem only)
#   MIN_RANDOM_MOVES  slidingpuzzle: training scrambles of a uniform length in [k, NUM_RANDOM_MOVES] (train.sh adds
#               _rmin<k> to the run name; slurm job names get _rmin<k>; other envs ignore it)
#   RANDOM_MOVES_ZIPF  with MIN_RANDOM_MOVES: length k drawn with p(k) ∝ k^-s (default 0 = uniform; _rzipf<s>)
#   HALT_KL     mem configs' policy-KL halting threshold (default 1e-3; train.sh adds _hkl<k> to mem run names when
#               != 1e-3; slurm job names get _hkl<k> when set; iso_mem never halts, so it is unaffected)
MEM_STUDY_ENVS="sokoban rushhour lightsout slidingpuzzle"
MEM_STUDY_SEEDS="1 2 3 4 5 6 7 8 9 10"
MEM_STUDY_COMMON="COMPUTE_DTYPE=bfloat16 MLP_TYPE=gelu DISCOUNT_ACT=0.99"   # DISCOUNT_COMPUTE: per config
MEM_STUDY_TAG=""   # appended to slurm job names
[[ -z "${HALT_ALPHA_MINS:-}" ]] || { echo "HALT_ALPHA_MINS was removed (mem_halt halts from its latent only)" >&2; return 1; }
[[ -n "${MIN_RANDOM_MOVES:-}" ]] && { MEM_STUDY_COMMON+=" MIN_RANDOM_MOVES=${MIN_RANDOM_MOVES}"; MEM_STUDY_TAG+="_rmin${MIN_RANDOM_MOVES}"; }
[[ -n "${RANDOM_MOVES_ZIPF:-}" ]] && { MEM_STUDY_COMMON+=" RANDOM_MOVES_ZIPF=${RANDOM_MOVES_ZIPF}"; MEM_STUDY_TAG+="_rzipf${RANDOM_MOVES_ZIPF}"; }
[[ -n "${HALT_KL:-}" ]] && { MEM_STUDY_COMMON+=" HALT_KL=${HALT_KL}"; MEM_STUDY_TAG+="_hkl${HALT_KL}"; }
[[ -n "${HALT_ENTROPY_COST:-}" ]] && { MEM_STUDY_COMMON+=" HALT_ENTROPY_COST=${HALT_ENTROPY_COST}"; MEM_STUDY_TAG+="_hent${HALT_ENTROPY_COST}"; }
[[ "${HALT_HEAD_GRAD:-0}" == 1 ]] && { MEM_STUDY_COMMON+=" HALT_HEAD_GRAD=1"; MEM_STUDY_TAG+="_hgrad"; }
[[ -n "${TARGET_KL:-}" ]] && { MEM_STUDY_COMMON+=" TARGET_KL=${TARGET_KL}"; MEM_STUDY_TAG+="_tkl${TARGET_KL}"; }
[[ -n "${NUM_EPOCHS:-}" ]] && { MEM_STUDY_COMMON+=" NUM_EPOCHS=${NUM_EPOCHS}"; MEM_STUDY_TAG+="_ep${NUM_EPOCHS}"; }
if [[ -n "${NUM_TIMESTEPS:-}" ]]; then
  MEM_STUDY_COMMON+=" NUM_TIMESTEPS=${NUM_TIMESTEPS}"
  (( NUM_TIMESTEPS % 1000000 == 0 )) && MEM_STUDY_TAG+="_ts$(( NUM_TIMESTEPS / 1000000 ))M" || MEM_STUDY_TAG+="_ts${NUM_TIMESTEPS}"
fi
[[ "${VALUE_AT_MAX_ITERS:-0}" == 1 ]] && { MEM_STUDY_COMMON+=" VALUE_AT_MAX_ITERS=1"; MEM_STUDY_TAG+="_vT"; }
# "<config name>|<env vars> <arm>"
MEM_STUDY_CONFIGS=()
for _gc in ${DISCOUNT_COMPUTES:-${DISCOUNT_COMPUTE:-0.999}}; do
for _ve in ${VALUE_EXPECTILES:-0.5}; do
  _sfx=""; [[ "$_ve" != 0.5 ]] && _sfx+="_ve${_ve}"; [[ "$_gc" != 0.999 ]] && _sfx+="_gc${_gc}"
  for _t in ${ISO_ITERS-1 2 3 4}; do MEM_STUDY_CONFIGS+=("iso_mem_it${_t}${_sfx}|MAX_THINK_ITERS=${_t} VALUE_EXPECTILE=${_ve} DISCOUNT_COMPUTE=${_gc} iso_mem"); done
  for _t in ${MEM_ITERS-2 3 4}; do MEM_STUDY_CONFIGS+=("mem_it${_t}${_sfx}|MAX_THINK_ITERS=${_t} VALUE_EXPECTILE=${_ve} DISCOUNT_COMPUTE=${_gc} mem"); done
  for _t in ${ISO_RIM_ITERS-}; do MEM_STUDY_CONFIGS+=("iso_rim_it${_t}${_sfx}|MAX_THINK_ITERS=${_t} VALUE_EXPECTILE=${_ve} DISCOUNT_COMPUTE=${_gc} iso_rim"); done
  for _t in ${MEM_HALT_ITERS-}; do MEM_STUDY_CONFIGS+=("mem_halt_it${_t}${_sfx}_hlat|MAX_THINK_ITERS=${_t} VALUE_EXPECTILE=${_ve} DISCOUNT_COMPUTE=${_gc} mem_halt"); done
  for _t in ${RIM_ITERS-}; do MEM_STUDY_CONFIGS+=("rim_it${_t}${_sfx}|MAX_THINK_ITERS=${_t} VALUE_EXPECTILE=${_ve} DISCOUNT_COMPUTE=${_gc} rim"); done
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
# mem and iso_mem at the same cap cost the same (halting freezes examples but does not skip compute); the cost of a cap
# T is interpolated linearly through the T=1 and T=4 measurements (each think iteration is one more core call).
# lightsout is refit to the L40S runs of looped_actor_gamma_ablation (5x4, T = 1..16, GPU-hours per run with 2-3 runs
# per GPU): ~1.25-1.5x the old "0.25 1.0" at every T, which timed out the T=16 jobs; "0.3 1.35" stays within SAFETY.
declare -A MEM_STUDY_HOURS=(  # "<hours at T=1> <hours at T=4>"
  [lightsout]="0.3 1.35" [slidingpuzzle]="0.6 2.2" [rushhour]="1.9 7.9" [sokoban]="3.7 15.6"
  # slidingpuzzle with GRID=4 / GRID=5 (L40S runs of looped_actor_slidingpuzzle). The cost is far from linear in T
  # (T=4 ~ T=8), so the line goes through T=4 and T=16 and overestimates T=1 and T=8 rather than underestimate T=2-4.
  [slidingpuzzle4]="1.25 1.85" [slidingpuzzle5]="2.2 3.1")
declare -A MEM_STUDY_GPU_FRACTION=(  # XLA memory fraction per run on a ~45GiB L40S: >= 2x the measured peak, room for T > 4
  [lightsout]=0.08 [slidingpuzzle]=0.08 [rushhour]=0.10 [sokoban]=0.15)
declare -A MEM_STUDY_HOST_GB=(  # host RAM per run (measured + headroom)
  [lightsout]=4 [slidingpuzzle]=4 [rushhour]=5 [sokoban]=8)

mem_study_hours() {  # mem_study_hours <env> <cap>: estimated hours of one run alone
  # (measured with 4 epochs and 100M steps; ~95% of the time is the update, which scales with NUM_EPOCHS; all of it
  # scales with NUM_TIMESTEPS)
  local key="$1"; [[ -n "${MEM_STUDY_HOURS[$1${GRID:-}]:-}" ]] && key="$1${GRID:-}"
  local h1 h4; read -r h1 h4 <<< "${MEM_STUDY_HOURS[$key]}"
  awk -v a="$h1" -v b="$h4" -v t="$2" -v e="${NUM_EPOCHS:-4}" -v n="${NUM_TIMESTEPS:-100000000}" \
    'BEGIN { printf "%.2f", (a + (b - a) * (t - 1) / 3) * (0.05 + 0.95 * e / 4) * n / 1e8 }'
}

mem_study_cap() {  # mem_study_cap "<env vars>": the MAX_THINK_ITERS value in a config's env vars
  local v; for v in $1; do [[ "$v" == MAX_THINK_ITERS=* ]] && echo "${v#MAX_THINK_ITERS=}"; done; return 0
}
