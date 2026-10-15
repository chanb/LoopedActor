#!/bin/bash
set -euo pipefail

# Submits the fprm_mem study (slurm/mem_study.sh: iso_mem at caps ISO_ITERS, default 1 2 3 4, and mem at caps MEM_ITERS,
# default 2 3 4) on all four environments, 10 seeds each.
# The runs of an env (every config and seed) are packed onto shared L40S GPUs (48GB, run with CUDA MPS); see bin_pack() below.
#
#   slurm/submit_mem.sh                    # submit everything
#   DRY_RUN=1 slurm/submit_mem.sh          # print the sbatch commands only
#   ENVS="lightsout slidingpuzzle" slurm/submit_mem.sh
#   SEEDS="1 2 3" slurm/submit_mem.sh
#   ISO_ITERS="1 2 4 8" MEM_ITERS="4 8" slurm/submit_mem.sh
#   VALUE_EXPECTILES="0.5 0.7 0.9" ENVS=lightsout slurm/submit_mem.sh
#   TARGET_KL=0.03 ENVS=lightsout ISO_ITERS=4 MEM_ITERS= slurm/submit_mem.sh
#   MAX_RUNS_PER_GPU=3 MAX_JOB_HOURS=48 slurm/submit_mem.sh   # longer jobs, 3 runs at a time, the next starting as one ends
# Optional: REPO_DIR (repo on the cluster), OUT_ROOT (experiment output root, <OUT_ROOT>/<env>/<run>/train.log),
# LOG_DIR (slurm logs), WANDB_MODE (default offline; `wandb sync` afterwards), WANDB_PROJECT (default looped_actor).
# SKIP_COMPLETED (default 1) leaves out runs whose train.log reached the final evaluation; failed, crashed or
# timed-out runs (and ones still queued or running) are submitted again. SKIP_COMPLETED=0 submits everything.
# The cluster checkout needs LoopedActor/.venv, data/boxoban/*.npz and data/rushhour/*.npz
# (data_scripts/build_rushhour_banks.py --src rush.txt).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run_pack.sh"
source "${SCRIPT_DIR}/mem_study.sh"
REPO_DIR="${REPO_DIR:-/home/chanb/scratch/code/LoopedActor}"
LOG_DIR="${LOG_DIR:-/home/chanb/scratch/logs/looped_actor}"
OUT_ROOT="${OUT_ROOT:-/home/chanb/scratch/looped_actor/exp}"
WANDB_MODE="${WANDB_MODE:-offline}"
WANDB_PROJECT="${WANDB_PROJECT:-looped_actor}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_COMPLETED="${SKIP_COMPLETED:-1}"
read -r -a ENVS <<< "${ENVS:-$MEM_STUDY_ENVS}"
read -r -a SEEDS <<< "${SEEDS:-$MEM_STUDY_SEEDS}"
COMMON="$MEM_STUDY_COMMON"
CONFIGS=("${MEM_STUDY_CONFIGS[@]}")   # experiment definitions: slurm/mem_study.sh

# Packing, from the measured costs in slurm/mem_study.sh:
#   MAX_RUNS_PER_GPU  runs executing on a job's GPU at once (default 10; lowered per env so their memory fractions add
#                     up to <= 0.9). A job's further runs wait and start as running ones finish. The 0.9 of the
#                     GPU is split evenly among the runs executing at once (XLA_PYTHON_CLIENT_MEM_FRACTION).
#   MAX_JOB_HOURS     cluster time limit per job (default 24); SAFETY (default 1.3) = headroom over the estimate.
#   TIERS             the cluster's time-limit tiers in hours (default "3 12 24 72 168"); a dry run prints how many
#                     jobs each tier would need, to choose MAX_JOB_HOURS.
# Runs on one GPU share it, so a job takes about the sum of its runs' single-run hours however many run at once. All
# runs of an env (every config and seed) are packed first-fit decreasing into the fewest jobs whose estimate x SAFETY
# (+10 min startup) fits in MAX_JOB_HOURS; each job runs its longest runs first. ppo.py cannot resume, so a run still
# going at the time limit is lost (runs that already finished are kept).
MAX_JOB_HOURS="${MAX_JOB_HOURS:-24}"
SAFETY="${SAFETY:-1.1}"
TIERS="${TIERS:-3 12 24 72 168}"
MAX_RUNS_PER_GPU="${MAX_RUNS_PER_GPU:-${CONCURRENCY:-10}}"   # CONCURRENCY: older name, same meaning
[[ "$MAX_RUNS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_RUNS_PER_GPU must be a positive integer, got $MAX_RUNS_PER_GPU"; exit 1; }
concurrency() {  # concurrency <env>: runs at once on one GPU (MAX_RUNS_PER_GPU, capped by memory)
  awk -v c="$MAX_RUNS_PER_GPU" -v f="${MEM_STUDY_GPU_FRACTION[$1]}" 'BEGIN { k = int(0.9 / f + 1e-9); print (c < k ? c : k) }'
}
job_capacity() {  # job_capacity <hours>: minutes of (SAFETY-scaled) work that fit in a job of that time limit
  awk -v m="$1" 'BEGIN { print int(m * 60) - 11 }'
}
bin_pack() {  # bin_pack <hours>: reads "<minutes> <id>" lines, prints "<job> <id>" lines, longest runs first
  # (first-fit decreasing; a run longer than a job gets a job of its own)
  sort -k1,1nr | awk -v cap="$(job_capacity "$1")" \
    '{ for (b = 0; b < n; b++) if (used[b] + $1 <= cap) break; if (b == n) n++; used[b] += $1; print b, $2 }'
}

completed() {  # completed <env> <env vars> <arm> <seed>: the run's train.log reached the last evaluation
  # (every train.sh passes --num_eval_steps=100; a resubmitted run appends to the same train.log)
  local dir
  dir="$(env ${COMMON} $2 PRINT_RUN_DIR=1 bash "${SCRIPT_DIR}/../$1/scripts/train.sh" "$3" "$4" "${OUT_ROOT}/$1")" \
    || { echo "could not get the run directory of $1 $3 seed $4 ($2)" >&2; exit 1; }
  grep -q '^Evaluation step 100:' "${dir}/train.log" 2>/dev/null
}

[[ "$DRY_RUN" == 1 ]] || mkdir -p "${LOG_DIR}"
n_jobs=0; n_skipped=0; req_minutes=0; est_minutes=0
declare -A tier_jobs=()
for env in "${ENVS[@]}"; do
  cmds=(); minutes=(); names=(); seeds=()   # every run of this env still to do
  for entry in "${CONFIGS[@]}"; do
    name="${entry%%|*}"; spec="${entry#*|}"
    arm="${spec##* }"; vars="${spec% *}"
    cap="$(mem_study_cap "$vars")"; h="$(mem_study_hours "$env" "$cap")"
    m=$(awk -v h="$h" -v s="$SAFETY" 'BEGIN { print int(h * s * 60 + 0.999) }')
    if (( m > $(job_capacity "$MAX_JOB_HOURS") )); then
      echo "warning: ${env} ${name} needs ~${h}h alone (x${SAFETY} + 10 min startup > ${MAX_JOB_HOURS}h limit); its runs may hit the time limit" >&2
    fi
    for seed in "${SEEDS[@]}"; do
      if [[ "$SKIP_COMPLETED" == 1 ]] && completed "$env" "$vars" "$arm" "$seed"; then
        n_skipped=$(( n_skipped + 1 )); continue
      fi
      cmds+=("WANDB_MODE=${WANDB_MODE} WANDB_PROJECT=${WANDB_PROJECT} ${COMMON} ${vars} ${env}/scripts/train.sh ${arm} ${seed} ${OUT_ROOT}/${env} 0")
      minutes+=("$m"); names+=("$name"); seeds+=("$seed")
    done
  done
  (( ${#cmds[@]} > 0 )) || continue
  items="$(for i in "${!cmds[@]}"; do echo "${minutes[$i]} $i"; done)"
  for tier in $TIERS; do
    tier_jobs[$tier]=$(( ${tier_jobs[$tier]:-0} + $(bin_pack "$tier" <<< "$items" | awk '{ print $1 }' | sort -u | wc -l) ))
  done
  declare -A job_ids=()
  while read -r b i; do job_ids[$b]+=" $i"; done < <(bin_pack "$MAX_JOB_HOURS" <<< "$items")
  n_env_jobs=${#job_ids[@]}
  for (( b = 0; b < n_env_jobs; b++ )); do
    read -r -a ids <<< "${job_ids[$b]}"
    job_cmds=(); total=0
    for i in "${ids[@]}"; do job_cmds+=("${cmds[$i]}"); total=$(( total + minutes[$i] )); done
    job_names=$(for i in "${ids[@]}"; do echo "${names[$i]}"; done | sort -u)
    if [[ "$(wc -l <<< "$job_names")" == 1 ]]; then
      job="mem-${env}-${job_names}${MEM_STUDY_TAG}-s$(for i in "${ids[@]}"; do echo "${seeds[$i]}"; done | sort -n | paste -sd_)"
    else
      job="mem-${env}-pack$(( b + 1 ))of${n_env_jobs}${MEM_STUDY_TAG}"
    fi
    limit=$(( total + 10 )); max_limit=$(awk -v m="$MAX_JOB_HOURS" 'BEGIN { print int(m * 60) - 1 }')
    (( limit > max_limit )) && limit=$max_limit
    time=$(printf '%02d:%02d:00' $(( limit / 60 )) $(( limit % 60 )))
    req_minutes=$(( req_minutes + limit ))
    est=$(awk -v t="$total" -v s="$SAFETY" 'BEGIN { print int(t / s) }'); est_minutes=$(( est_minutes + est ))
    running=$(concurrency "$env"); (( running > ${#job_cmds[@]} )) && running=${#job_cmds[@]}
    frac=$(awk -v r="$running" 'BEGIN { printf "%.3f", int(900 / r) / 1000 }')   # 0.9 of the GPU split among them
    job_cmds=("${job_cmds[@]/#/XLA_PYTHON_CLIENT_MEM_FRACTION=${frac} }")
    CMDS_B64="$(printf '%s\n' "${job_cmds[@]}" | base64 -w0)"
    host_mem="$(( running * MEM_STUDY_HOST_GB[$env] + 2 ))G"
    sbatch_cmd=(sbatch --export=ALL,REPO_DIR="${REPO_DIR}",CMDS_B64="${CMDS_B64}",CONCURRENCY="${running}" --job-name="${job}"
                --time="${time}" --mem="${host_mem}" --cpus-per-task=$(( running + 1 ))
                --output="${LOG_DIR}/%x_%j.out" "${RUN_SCRIPT}")
    if [[ "$DRY_RUN" == 1 ]]; then
      echo "sbatch --job-name=${job} --time=${time} --mem=${host_mem} --cpus-per-task=$(( running + 1 ))  (${#job_cmds[@]} runs, ${running} at a time, ~$(awk -v e="$est" 'BEGIN { printf "%.1f", e / 60 }')h estimated)"
      printf '    %s\n' "${job_cmds[@]}"
    else
      "${sbatch_cmd[@]}"
    fi
    n_jobs=$(( n_jobs + 1 ))
  done
  unset job_ids
done
echo "${n_jobs} jobs $([[ "$DRY_RUN" == 1 ]] && echo "(dry run, nothing submitted)" || echo submitted), ${n_skipped} completed runs skipped"
if (( n_jobs > 0 )); then
  awk -v r="$req_minutes" -v e="$est_minutes" \
    'BEGIN { printf "requested %.1f GPU-hours for ~%.1f estimated (%.0f%% of the time limits)\n", r / 60, e / 60, 100 * e / r }'
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'jobs needed per tier (MAX_JOB_HOURS):'; for tier in $TIERS; do printf '  %sh: %s' "$tier" "${tier_jobs[$tier]}"; done; echo
  fi
fi
