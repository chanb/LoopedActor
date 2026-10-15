#!/bin/bash
set -euo pipefail

# Submits the fprm_mem study (slurm/mem_study.sh: iso_mem at caps ISO_ITERS, default 1 2 3 4, and mem at caps MEM_ITERS,
# default 2 3 4) on all four environments, 10 seeds each.
# Seeds of one (env, config) are packed onto shared L40S GPUs (48GB, run with CUDA MPS); see pack() below.
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
# Runs on one GPU share it, so a job takes about <runs in the job> x <single-run hours> however many run at once; the
# number of runs per job is the most whose estimate x SAFETY fits in MAX_JOB_HOURS. ppo.py cannot resume, so a run still
# going at the time limit is lost (runs that already finished are kept).
MAX_JOB_HOURS="${MAX_JOB_HOURS:-24}"
SAFETY="${SAFETY:-1.3}"
MAX_RUNS_PER_GPU="${MAX_RUNS_PER_GPU:-${CONCURRENCY:-10}}"   # CONCURRENCY: older name, same meaning
[[ "$MAX_RUNS_PER_GPU" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_RUNS_PER_GPU must be a positive integer, got $MAX_RUNS_PER_GPU"; exit 1; }
concurrency() {  # concurrency <env>: runs at once on one GPU (MAX_RUNS_PER_GPU, capped by memory)
  awk -v c="$MAX_RUNS_PER_GPU" -v f="${MEM_STUDY_GPU_FRACTION[$1]}" 'BEGIN { k = int(0.9 / f + 1e-9); print (c < k ? c : k) }'
}
pack() {  # pack <env> <cap> <number of runs>: "<runs per job> <time limit>"
  local env=$1 runs=$3 h n minutes
  h="$(mem_study_hours "$env" "$2")"
  n=$(awk -v m="$MAX_JOB_HOURS" -v s="$SAFETY" -v h="$h" 'BEGIN { n = int(m * 60 / (s * h * 60 + 1)); print (n < 1 ? 1 : n) }')
  # balance: split the seeds into the fewest jobs of at most n runs, as evenly as possible
  local jobs=$(( (runs + n - 1) / n )); n=$(( (runs + jobs - 1) / jobs ))
  minutes=$(awk -v n="$n" -v s="$SAFETY" -v h="$h" -v m="$MAX_JOB_HOURS" \
        'BEGIN { t = int(n * h * s * 60) + 30; if (t > m * 60 - 1) t = m * 60 - 1; print t }')
  if awk -v s="$SAFETY" -v h="$h" -v m="$MAX_JOB_HOURS" 'BEGIN { exit !(s * h > m) }'; then
    echo "warning: ${env} T=$2 needs ~${h}h alone (x${SAFETY} > ${MAX_JOB_HOURS}h limit); its runs may hit the time limit" >&2
  fi
  printf '%s %02d:%02d:00\n' "$n" $(( minutes / 60 )) $(( minutes % 60 ))
}

completed() {  # completed <env> <env vars> <arm> <seed>: the run's train.log reached the last evaluation
  # (every train.sh passes --num_eval_steps=100; a resubmitted run appends to the same train.log)
  local dir
  dir="$(env ${COMMON} $2 PRINT_RUN_DIR=1 bash "${SCRIPT_DIR}/../$1/scripts/train.sh" "$3" "$4" "${OUT_ROOT}/$1")" \
    || { echo "could not get the run directory of $1 $3 seed $4 ($2)" >&2; exit 1; }
  grep -q '^Evaluation step 100:' "${dir}/train.log" 2>/dev/null
}

[[ "$DRY_RUN" == 1 ]] || mkdir -p "${LOG_DIR}"
n_jobs=0; n_skipped=0
for env in "${ENVS[@]}"; do
  for entry in "${CONFIGS[@]}"; do
    name="${entry%%|*}"; spec="${entry#*|}"
    arm="${spec##* }"; vars="${spec% *}"
    todo=()
    for seed in "${SEEDS[@]}"; do
      if [[ "$SKIP_COMPLETED" == 1 ]] && completed "$env" "$vars" "$arm" "$seed"; then
        n_skipped=$(( n_skipped + 1 ))
      else
        todo+=("$seed")
      fi
    done
    (( ${#todo[@]} > 0 )) || continue
    read -r per_gpu time <<< "$(pack "$env" "$(mem_study_cap "$vars")" "${#todo[@]}")"
    cmds=()
    for seed in "${todo[@]}"; do
      cmds+=("WANDB_MODE=${WANDB_MODE} WANDB_PROJECT=${WANDB_PROJECT} ${COMMON} ${vars} ${env}/scripts/train.sh ${arm} ${seed} ${OUT_ROOT}/${env} 0")
      if (( ${#cmds[@]} == per_gpu )) || [[ "$seed" == "${todo[-1]}" ]]; then
        job="mem-${env}-${name}${MEM_STUDY_TAG}-s$(echo "${cmds[@]}" | grep -o "${arm} [0-9]* " | awk '{print $2}' | paste -sd_)"
        running=$(concurrency "$env"); (( running > ${#cmds[@]} )) && running=${#cmds[@]}
        frac=$(awk -v r="$running" 'BEGIN { printf "%.3f", int(900 / r) / 1000 }')   # 0.9 of the GPU split among them
        cmds=("${cmds[@]/#/XLA_PYTHON_CLIENT_MEM_FRACTION=${frac} }")
        CMDS_B64="$(printf '%s\n' "${cmds[@]}" | base64 -w0)"
        host_mem="$(( running * MEM_STUDY_HOST_GB[$env] + 2 ))G"
        sbatch_cmd=(sbatch --export=ALL,REPO_DIR="${REPO_DIR}",CMDS_B64="${CMDS_B64}",CONCURRENCY="${running}" --job-name="${job}"
                    --time="${time}" --mem="${host_mem}" --cpus-per-task=$(( running + 1 ))
                    --output="${LOG_DIR}/%x_%j.out" "${RUN_SCRIPT}")
        if [[ "$DRY_RUN" == 1 ]]; then
          echo "sbatch --job-name=${job} --time=${time} --mem=${host_mem} --cpus-per-task=$(( running + 1 ))  (${#cmds[@]} runs, ${running} at a time)"
          printf '    %s\n' "${cmds[@]}"
        else
          "${sbatch_cmd[@]}"
        fi
        n_jobs=$(( n_jobs + 1 )); cmds=()
      fi
    done
  done
done
echo "${n_jobs} jobs $([[ "$DRY_RUN" == 1 ]] && echo "(dry run, nothing submitted)" || echo submitted), ${n_skipped} completed runs skipped"
