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
#   CONCURRENCY=3 MAX_JOB_HOURS=48 slurm/submit_mem.sh   # longer jobs, 3 runs at a time, the next starting as one ends
# Optional: REPO_DIR (repo on the cluster), OUT_ROOT (experiment output root, <OUT_ROOT>/<env>/<run>/train.log),
# LOG_DIR (slurm logs), WANDB_MODE (default offline; `wandb sync` afterwards).
# The cluster checkout needs LoopedActor/.venv, data/boxoban/*.npz and data/rushhour/*.npz
# (data_scripts/build_rushhour_banks.py --src rush.txt).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run_pack.sh"
source "${SCRIPT_DIR}/mem_study.sh"
REPO_DIR="${REPO_DIR:-/home/chanb/research/iclr_2027/LoopedActor}"
LOG_DIR="${LOG_DIR:-/home/chanb/scratch/logs/looped_actor}"
OUT_ROOT="${OUT_ROOT:-/home/chanb/scratch/looped_actor/exp}"
WANDB_MODE="${WANDB_MODE:-offline}"
DRY_RUN="${DRY_RUN:-0}"
read -r -a ENVS <<< "${ENVS:-$MEM_STUDY_ENVS}"
read -r -a SEEDS <<< "${SEEDS:-$MEM_STUDY_SEEDS}"
COMMON="$MEM_STUDY_COMMON"
CONFIGS=("${MEM_STUDY_CONFIGS[@]}")   # experiment definitions: slurm/mem_study.sh

# Packing, from the measured costs in slurm/mem_study.sh. GPU memory is never the constraint (every config peaks at
# <= 3 GiB), so the runs per GPU are set by the time limit: packed runs share the GPU, so a job takes about
# <runs per GPU> x <single-run hours>; the limit leaves SAFETY headroom over that. ppo.py cannot resume, so a run that
# hits the limit is lost.
MAX_JOB_HOURS="${MAX_JOB_HOURS:-24}"        # cluster time limit per job
SAFETY="${SAFETY:-1.3}"                     # job time limit = SAFETY x estimated time
MAX_RUNS_PER_GPU="${MAX_RUNS_PER_GPU:-10}"     # runs per job
CONCURRENCY="${CONCURRENCY:-0}"             # runs executing at once within a job (0 = all); the others wait for a free
                                            # slot. Runs share the GPU either way, so a job's estimated time is the same;
                                            # but at the time limit only the runs still in flight are lost, and the job
                                            # requests CPUs / RAM for CONCURRENCY runs only.
[[ "$CONCURRENCY" =~ ^[0-9]+$ ]] || { echo "CONCURRENCY must be a non-negative integer, got $CONCURRENCY"; exit 1; }
pack() {  # pack <env> <cap>: "<runs per GPU> <XLA memory fraction per run> <time limit> <host RAM per job>"
  local env=$1 h n minutes
  h="$(mem_study_hours "$env" "$2")"
  n=$(awk -v m="$MAX_JOB_HOURS" -v s="$SAFETY" -v h="$h" -v c="$MAX_RUNS_PER_GPU" \
        'BEGIN { n = int(m * 60 / (s * h * 60 + 1)); if (n < 1) n = 1; if (n > c) n = c; print n }')
  # memory: the fractions of the runs on one GPU at once add up to at most 0.9
  if (( CONCURRENCY == 0 )); then
    n=$(awk -v n="$n" -v f="${MEM_STUDY_GPU_FRACTION[$env]}" 'BEGIN { k = int(0.9 / f + 1e-9); print (n < k ? n : k) }')
  fi
  # balance: split the seeds into the fewest jobs of at most n runs, as evenly as possible
  local jobs=$(( (${#SEEDS[@]} + n - 1) / n )); n=$(( (${#SEEDS[@]} + jobs - 1) / jobs ))
  minutes=$(awk -v n="$n" -v s="$SAFETY" -v h="$h" -v m="$MAX_JOB_HOURS" \
        'BEGIN { t = int(n * h * s * 60) + 30; if (t > m * 60 - 1) t = m * 60 - 1; print t }')
  if awk -v s="$SAFETY" -v h="$h" -v m="$MAX_JOB_HOURS" 'BEGIN { exit !(s * h > m) }'; then
    echo "warning: ${env} T=$2 needs ~${h}h alone (x${SAFETY} > ${MAX_JOB_HOURS}h limit); its runs may hit the time limit" >&2
  fi
  printf '%s %s %02d:%02d:00\n' "$n" "${MEM_STUDY_GPU_FRACTION[$env]}" $(( minutes / 60 )) $(( minutes % 60 ))
}
(( CONCURRENCY == 0 )) || for env in ${ENVS:-$MEM_STUDY_ENVS}; do   # CONCURRENCY runs must fit in GPU memory
  awk -v c="$CONCURRENCY" -v f="${MEM_STUDY_GPU_FRACTION[$env]}" 'BEGIN { exit !(c * f <= 0.9 + 1e-9) }' \
    || { echo "CONCURRENCY=${CONCURRENCY} x memory fraction ${MEM_STUDY_GPU_FRACTION[$env]} (${env}) exceeds 0.9 of the GPU" >&2; exit 1; }
done

[[ "$DRY_RUN" == 1 ]] || mkdir -p "${LOG_DIR}"
n_jobs=0
for env in "${ENVS[@]}"; do
  for entry in "${CONFIGS[@]}"; do
    name="${entry%%|*}"; spec="${entry#*|}"
    arm="${spec##* }"; vars="${spec% *}"
    read -r per_gpu frac time <<< "$(pack "$env" "$(mem_study_cap "$vars")")"
    cmds=()
    for seed in "${SEEDS[@]}"; do
      cmds+=("XLA_PYTHON_CLIENT_MEM_FRACTION=${frac} WANDB_MODE=${WANDB_MODE} ${COMMON} ${vars} ${env}/scripts/train.sh ${arm} ${seed} ${OUT_ROOT}/${env} 0")
      if (( ${#cmds[@]} == per_gpu )) || [[ "$seed" == "${SEEDS[-1]}" ]]; then
        job="mem-${env}-${name}${MEM_STUDY_TAG}-s$(echo "${cmds[@]}" | grep -o "${arm} [0-9]* " | awk '{print $2}' | paste -sd_)"
        CMDS_B64="$(printf '%s\n' "${cmds[@]}" | base64 -w0)"
        running=${#cmds[@]}; (( CONCURRENCY > 0 && CONCURRENCY < running )) && running=$CONCURRENCY
        host_mem="$(( running * MEM_STUDY_HOST_GB[$env] + 2 ))G"
        sbatch_cmd=(sbatch --export=ALL,REPO_DIR="${REPO_DIR}",CMDS_B64="${CMDS_B64}",CONCURRENCY="${CONCURRENCY}" --job-name="${job}"
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
echo "${n_jobs} jobs $([[ "$DRY_RUN" == 1 ]] && echo "(dry run, nothing submitted)" || echo submitted)"
