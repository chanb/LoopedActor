#!/bin/bash
#SBATCH --account=aip-schuurma
#SBATCH --gres=gpu:1
# --time, --mem, --cpus-per-task and --output are set per job by submit_mem.sh.

# Runs a pack of train.sh commands concurrently on one GPU (CUDA MPS), one per line.
# Expects the newline-separated commands, base64-encoded, in $CMDS_B64 (sbatch --export splits on commas), e.g.
#   CMDS_B64="$(printf '%s\n' "cmd 1" "cmd 2" | base64 -w0)"
#   sbatch --export=ALL,CMDS_B64="${CMDS_B64}" --job-name=<name> run_pack.sh
# Each command must set its own XLA_PYTHON_CLIENT_MEM_FRACTION so the pack fits in GPU memory.
# Submitted by submit_mem.sh.

module load StdEnv/2023
module load cuda/12.2

mkdir -p $SLURM_TMPDIR/tmp
export CUDA_MPS_PIPE_DIRECTORY=$SLURM_TMPDIR/tmp
export CUDA_MPS_LOG_DIRECTORY=$SLURM_TMPDIR/tmp
nvidia-cuda-mps-control -d

cd "${REPO_DIR:-/home/chanb/research/iclr_2027/LoopedActor}"

echo "hostname: $(hostname)"
echo "starting at: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ -z "${CMDS_B64:-}" ]; then
    echo "CMDS_B64 environment variable is not set, nothing to run" >&2
    exit 1
fi

mapfile -t CMDS < <(base64 -d <<< "${CMDS_B64}")
pids=()
for cmd in "${CMDS[@]}"; do
    [ -z "$cmd" ] && continue
    echo "running: $cmd"
    bash -c "$cmd" > /dev/null 2>&1 &   # train.sh tees its output to <out>/<run>/train.log
    pids+=($!)
done

status=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "finished: ${CMDS[$i]}"
    else
        code=$?
        echo "FAILED (exit code $code): ${CMDS[$i]}"
        status=1
    fi
done

echo quit | nvidia-cuda-mps-control
echo "finished with status $status at: $(date)"
exit $status
