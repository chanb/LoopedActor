#!/bin/bash
set -euo pipefail

# Submits the fprm_mem study (run_lightsout.sh's three configs) on all four environments, 10 seeds each:
#   iso_mem MAX_THINK_ITERS=1 | iso_mem MAX_THINK_ITERS=4 | mem MAX_THINK_ITERS=4
# all bfloat16 + GELU MLP, DISCOUNT_ACT=0.99, DISCOUNT_COMPUTE=0.999, each env's default board / episode settings.
# Seeds of one (env, config) are packed onto shared L40S GPUs (48GB, run with CUDA MPS); see PACK below.
#
#   slurm/submit_mem.sh                    # submit everything
#   DRY_RUN=1 slurm/submit_mem.sh          # print the sbatch commands only
#   ENVS="lightsout slidingpuzzle" slurm/submit_mem.sh
#   SEEDS="1 2 3" slurm/submit_mem.sh
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

# PACK[<env>:<config>] = "<runs per GPU> <XLA memory fraction per run> <time limit> <host RAM per job>"
# From the measurements below. GPU memory is never the constraint (every config peaks at <= 3 GiB), so runs per GPU are
# set by the time limit: packed runs share the GPU, so a job takes about <runs per GPU> x <single-run time>; limits
# leave >= 30% headroom over that. ppo.py cannot resume, so a run that hits the limit is lost - lower the packing
# rather than tighten the limit. Memory fractions (of the ~45GiB L40S) give each run >= 2x its measured peak.
declare -A PACK=(
  [lightsout:iso_mem_it1]="10 0.08 12:00:00 40G"
  [lightsout:iso_mem_it4]="5 0.08 23:59:00 20G"
  [lightsout:mem_it4]="5 0.08 23:59:00 20G"
  [slidingpuzzle:iso_mem_it1]="10 0.08 12:00:00 40G"
  [slidingpuzzle:iso_mem_it4]="5 0.08 20:00:00 20G"
  [slidingpuzzle:mem_it4]="5 0.08 20:00:00 20G"
  [rushhour:iso_mem_it1]="5 0.10 16:00:00 24G"
  [rushhour:iso_mem_it4]="2 0.10 23:59:00 12G"
  [rushhour:mem_it4]="2 0.10 23:59:00 12G"
  [sokoban:iso_mem_it1]="2 0.15 12:00:00 16G"
  [sokoban:iso_mem_it4]="1 0.15 23:59:00 10G"
  [sokoban:mem_it4]="1 0.15 23:59:00 10G"
)
# Measured (bf16, GELU, each env's train.sh settings: 1024 envs x 64 steps, 64 minibatches x 4 epochs; RTX A4500,
# one run alone; the L40S should be at least as fast):
#   env           config   peak GPU mem  steps/s   100M steps   host RAM
#   lightsout     T=1      0.40 GiB      42.0k     0.7 h        2.7 GiB
#   lightsout     T=4      0.54 GiB       9.6k     2.9 h        2.7 GiB
#   slidingpuzzle T=1      0.32 GiB      46.3k     0.6 h        2.6 GiB
#   slidingpuzzle T=4      0.32 GiB      12.6k     2.2 h        2.5 GiB
#   rushhour      T=1      1.06 GiB     >14.6k    <1.9 h        3.7 GiB   (measured on a shared GPU: speeds are lower bounds)
#   rushhour      T=4      1.13 GiB      >3.5k    <7.9 h        3.6 GiB
#   sokoban       T=1      2.71 GiB       7.6k     3.7 h        6.1 GiB
#   sokoban       T=4      2.98 GiB       1.8k    15.6 h        6.1 GiB
# (mem and iso_mem at the same cap cost the same: halting freezes examples but does not skip compute.)

[[ "$DRY_RUN" == 1 ]] || mkdir -p "${LOG_DIR}"
n_jobs=0
for env in "${ENVS[@]}"; do
  for entry in "${CONFIGS[@]}"; do
    name="${entry%%|*}"; spec="${entry#*|}"
    arm="${spec##* }"; vars="${spec% *}"
    key="${env}:${name}"
    [[ -n "${PACK[$key]:-}" ]] || { echo "no PACK entry for ${key}" >&2; exit 1; }
    read -r per_gpu frac time host_mem <<< "${PACK[$key]}"
    cmds=()
    for seed in "${SEEDS[@]}"; do
      cmds+=("XLA_PYTHON_CLIENT_MEM_FRACTION=${frac} WANDB_MODE=${WANDB_MODE} ${COMMON} ${vars} ${env}/scripts/train.sh ${arm} ${seed} ${OUT_ROOT}/${env} 0")
      if (( ${#cmds[@]} == per_gpu )) || [[ "$seed" == "${SEEDS[-1]}" ]]; then
        job="mem-${env}-${name}-s$(echo "${cmds[@]}" | grep -o "${arm} [0-9]* " | awk '{print $2}' | paste -sd_)"
        CMDS_B64="$(printf '%s\n' "${cmds[@]}" | base64 -w0)"
        sbatch_cmd=(sbatch --export=ALL,REPO_DIR="${REPO_DIR}",CMDS_B64="${CMDS_B64}" --job-name="${job}"
                    --time="${time}" --mem="${host_mem}" --cpus-per-task=$(( ${#cmds[@]} + 1 ))
                    --output="${LOG_DIR}/%x_%j.out" "${RUN_SCRIPT}")
        if [[ "$DRY_RUN" == 1 ]]; then
          echo "sbatch --job-name=${job} --time=${time} --mem=${host_mem} --cpus-per-task=$(( ${#cmds[@]} + 1 ))"
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
