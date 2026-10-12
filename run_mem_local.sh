#!/bin/bash
set -uo pipefail

# Runs the fprm_mem study (slurm/mem_study.sh: iso_mem T=1, iso_mem T=4, mem T=4 on all four environments, 10 seeds)
# on local GPUs without slurm. Runs are queued longest first (T=4 before T=1, sokoban > rushhour > lightsout >
# slidingpuzzle) and started on the next free slot; each GPU has RUNS_PER_GPU slots.
#
#   ./run_mem_local.sh                                    # all 120 runs on GPUs 0-7, one run per GPU
#   GPUS="3 5 7" RUNS_PER_GPU=2 ./run_mem_local.sh
#   ENVS="lightsout slidingpuzzle" SEEDS="1 2 3" ./run_mem_local.sh
#   DRY_RUN=1 ./run_mem_local.sh                          # print the queue only
#   nohup ./run_mem_local.sh > mem_local.log 2>&1 &       # keep running after logout
# Optional: OUT_ROOT (default exp/mem_study; runs go to <OUT_ROOT>/<env>/<run>/train.log), MEM_FRACTION (GPU memory
# fraction per run, default 0.85 / RUNS_PER_GPU), SKIP_EXISTING (default 1: skip runs whose directory already exists,
# so a restart continues where it stopped - delete the directory of a crashed run to rerun it), WANDB_MODE.
# Every config peaks at <= 3 GiB of GPU memory (see slurm/submit_mem.sh), so memory never limits RUNS_PER_GPU; runs
# sharing a GPU just split its throughput. ppo.py cannot resume: an interrupted run restarts from scratch.

cd "$(dirname "$0")"
source slurm/mem_study.sh
read -r -a GPUS <<< "${GPUS:-0 1 2 3 4 5 6 7}"
RUNS_PER_GPU="${RUNS_PER_GPU:-1}"
MEM_FRACTION="${MEM_FRACTION:-$(awk -v n="$RUNS_PER_GPU" 'BEGIN { printf "%.2f", 0.85 / n }')}"
OUT_ROOT="${OUT_ROOT:-exp/mem_study}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
read -r -a ENVS <<< "${ENVS:-$MEM_STUDY_ENVS}"
read -r -a SEEDS <<< "${SEEDS:-$MEM_STUDY_SEEDS}"
[[ "$RUNS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || { echo "RUNS_PER_GPU must be a positive integer, got $RUNS_PER_GPU"; exit 1; }

run_name() {  # run_name <env> <arm> <env vars...>: the run directory name train.sh will use
  local env=$1 arm=$2; shift 2
  env "$@" bash -c "$(sed -n '/^set -euo/,/^GROUP=/p' "$env/scripts/train.sh"); echo \"\$RUN\"" _ "$arm" 1 2>/dev/null
}

# Queue: T=4 configs first, then T=1; within each, the envs in order (slowest first); then seeds.
QUEUE=()   # "<env>|<arm>|<seed>|<env vars>"
for pass in 4 1; do
  for env in "${ENVS[@]}"; do
    for entry in "${MEM_STUDY_CONFIGS[@]}"; do
      spec="${entry#*|}"; arm="${spec##* }"; vars="${spec% *}"
      [[ "$vars" == *"MAX_THINK_ITERS=${pass}"* ]] || continue
      for seed in "${SEEDS[@]}"; do QUEUE+=("${env}|${arm}|${seed}|${MEM_STUDY_COMMON} ${vars}"); done
    done
  done
done

SLOTS=()     # one entry per (GPU, slot): the GPU index
for g in "${GPUS[@]}"; do for ((k = 0; k < RUNS_PER_GPU; k++)); do SLOTS+=("$g"); done; done
declare -A SLOT_PID=() SLOT_DESC=()
mkdir -p "$OUT_ROOT/launcher_logs"
n_started=0; n_skipped=0; n_failed=0
log() { echo "[$(date '+%F %T')] $*"; }
log "${#QUEUE[@]} runs, ${#SLOTS[@]} slots (GPUs ${GPUS[*]} x ${RUNS_PER_GPU}), memory fraction ${MEM_FRACTION} per run, output ${OUT_ROOT}"

reap() {  # collect finished runs; sets FREE to a free slot index (or empty)
  FREE=""
  for i in "${!SLOTS[@]}"; do
    pid="${SLOT_PID[$i]:-}"
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"; code=$?
      if (( code == 0 )); then log "finished  ${SLOT_DESC[$i]}"; else log "FAILED (exit ${code})  ${SLOT_DESC[$i]}"; n_failed=$(( n_failed + 1 )); fi
      SLOT_PID[$i]=""
    fi
    [[ -z "$FREE" && -z "${SLOT_PID[$i]:-}" ]] && FREE=$i
  done
}

for item in "${QUEUE[@]}"; do
  IFS='|' read -r env arm seed vars <<< "$item"
  read -r -a var_list <<< "$vars"
  run="$(run_name "$env" "$arm" "${var_list[@]}")"
  run="${run%_s1}_s${seed}"
  dir="${OUT_ROOT}/${env}/${run}"
  if [[ "$SKIP_EXISTING" == 1 && -d "$dir" ]]; then
    log "skip (exists)  ${dir}"; n_skipped=$(( n_skipped + 1 )); continue
  fi
  cmd="XLA_PYTHON_CLIENT_MEM_FRACTION=${MEM_FRACTION} ${vars} ${env}/scripts/train.sh ${arm} ${seed} ${OUT_ROOT}/${env}"
  if [[ "$DRY_RUN" == 1 ]]; then
    echo "  ${cmd} <gpu>    -> ${dir}"; continue
  fi
  reap
  while [[ -z "$FREE" ]]; do sleep 20; reap; done
  gpu="${SLOTS[$FREE]}"
  env "${var_list[@]}" XLA_PYTHON_CLIENT_MEM_FRACTION="${MEM_FRACTION}" \
    "${env}/scripts/train.sh" "$arm" "$seed" "${OUT_ROOT}/${env}" "$gpu" \
    > "${OUT_ROOT}/launcher_logs/${env}_${run}.log" 2>&1 &
  SLOT_PID[$FREE]=$!; SLOT_DESC[$FREE]="GPU ${gpu}  ${dir}"
  log "started   GPU ${gpu}  ${dir}"; n_started=$(( n_started + 1 ))
  sleep 5   # stagger startups (compilation / data loading)
done

if [[ "$DRY_RUN" == 1 ]]; then
  echo "${#QUEUE[@]} runs (dry run, nothing started)"; exit 0
fi
while true; do
  reap
  busy=0; for i in "${!SLOTS[@]}"; do [[ -n "${SLOT_PID[$i]:-}" ]] && busy=1; done
  (( busy )) || break
  sleep 20
done
log "done: ${n_started} started, ${n_skipped} skipped, ${n_failed} failed"
(( n_failed == 0 ))
