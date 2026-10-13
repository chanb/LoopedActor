# The fprm_mem study, sourced by slurm/submit_mem.sh and run_mem_local.sh: iso_mem at every cap in ISO_ITERS (fixed
# compute) and mem at every cap in MEM_ITERS (policy-KL halting) on all four environments. Every run: bfloat16 + GELU
# MLP, DISCOUNT_ACT=0.99, DISCOUNT_COMPUTE=0.999, each env's default board / episode settings.
#   ISO_ITERS   iso_mem caps (default "1 2 3 4")
#   MEM_ITERS   mem caps (default "2 3 4"; a cap of 1 is iso_mem T=1, since halting acts from pass 2; an empty list
#               (MEM_ITERS=) runs no mem configs, likewise ISO_ITERS=)
#   VALUE_EXPECTILES  value-loss expectiles, crossed with every config above (default "0.5" = MSE; e.g. "0.5 0.7 0.9"
#               multiplies the study by 3; train.sh adds _ve<tau> to the run name for tau != 0.5)
#   TARGET_KL, NUM_EPOCHS  passed to every run when set (train.sh adds _tkl<k> / _ep<n> to the run name; slurm job names
#               get the same tag; NUM_EPOCHS also rescales the time estimates)
MEM_STUDY_ENVS="sokoban rushhour lightsout slidingpuzzle"
MEM_STUDY_SEEDS="1 2 3 4 5 6 7 8 9 10"
MEM_STUDY_COMMON="COMPUTE_DTYPE=bfloat16 MLP_TYPE=gelu DISCOUNT_ACT=0.99 DISCOUNT_COMPUTE=0.999"
MEM_STUDY_TAG=""   # appended to slurm job names
[[ -n "${TARGET_KL:-}" ]] && { MEM_STUDY_COMMON+=" TARGET_KL=${TARGET_KL}"; MEM_STUDY_TAG+="_tkl${TARGET_KL}"; }
[[ -n "${NUM_EPOCHS:-}" ]] && { MEM_STUDY_COMMON+=" NUM_EPOCHS=${NUM_EPOCHS}"; MEM_STUDY_TAG+="_ep${NUM_EPOCHS}"; }
# "<config name>|<env vars> <arm>"
MEM_STUDY_CONFIGS=()
for _ve in ${VALUE_EXPECTILES:-0.5}; do
  _sfx=""; [[ "$_ve" != 0.5 ]] && _sfx="_ve${_ve}"
  for _t in ${ISO_ITERS-1 2 3 4}; do MEM_STUDY_CONFIGS+=("iso_mem_it${_t}${_sfx}|MAX_THINK_ITERS=${_t} VALUE_EXPECTILE=${_ve} iso_mem"); done
  for _t in ${MEM_ITERS-2 3 4}; do MEM_STUDY_CONFIGS+=("mem_it${_t}${_sfx}|MAX_THINK_ITERS=${_t} VALUE_EXPECTILE=${_ve} mem"); done
done
unset _t _ve _sfx

# Measured cost of one run alone (bf16, GELU, each env's train.sh settings: 1024 envs x 64 steps, 64 minibatches x 4
# epochs, 100M steps; RTX A4500 - the L40S should be at least as fast):
#   env           T=1: steps/s  hours   T=4: steps/s  hours   peak GPU mem (T=1 / T=4)   host RAM
#   lightsout          42.0k    0.7          9.6k     2.9     0.40 / 0.54 GiB             2.7 GiB
#   slidingpuzzle      46.3k    0.6         12.6k     2.2     0.32 / 0.32 GiB             2.6 GiB
#   rushhour          >14.6k   <1.9         >3.5k    <7.9     1.06 / 1.13 GiB             3.7 GiB   (shared GPU: speeds are lower bounds)
#   sokoban             7.6k    3.7          1.8k    15.6     2.71 / 2.98 GiB             6.1 GiB
# mem and iso_mem at the same cap cost the same (halting freezes examples but does not skip compute); the cost of a cap
# T is interpolated linearly through the T=1 and T=4 measurements (each think iteration is one more core call).
declare -A MEM_STUDY_HOURS=(  # "<hours at T=1> <hours at T=4>"
  [lightsout]="0.3 1.0" [slidingpuzzle]="0.6 2.2" [rushhour]="1.9 7.9" [sokoban]="3.7 15.6")
declare -A MEM_STUDY_GPU_FRACTION=(  # XLA memory fraction per run on a ~45GiB L40S: >= 2x the measured peak, room for T > 4
  [lightsout]=0.08 [slidingpuzzle]=0.08 [rushhour]=0.10 [sokoban]=0.15)
declare -A MEM_STUDY_HOST_GB=(  # host RAM per run (measured + headroom)
  [lightsout]=4 [slidingpuzzle]=4 [rushhour]=5 [sokoban]=8)

mem_study_hours() {  # mem_study_hours <env> <cap>: estimated hours of one run alone
  # (measured with 4 epochs; ~95% of the time is the update, which scales with NUM_EPOCHS)
  local h1 h4; read -r h1 h4 <<< "${MEM_STUDY_HOURS[$1]}"
  awk -v a="$h1" -v b="$h4" -v t="$2" -v e="${NUM_EPOCHS:-4}" 'BEGIN { printf "%.2f", (a + (b - a) * (t - 1) / 3) * (0.05 + 0.95 * e / 4) }'
}

mem_study_cap() {  # mem_study_cap "<env vars>": the MAX_THINK_ITERS value in a config's env vars
  local v; for v in $1; do [[ "$v" == MAX_THINK_ITERS=* ]] && echo "${v#MAX_THINK_ITERS=}"; done
}
