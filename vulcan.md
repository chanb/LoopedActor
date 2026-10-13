# Initial setup
```
module load StdEnv/2023
module load cuda/12.2

uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -r ogbench/requirements.txt
uv pip install -U "jax[cuda12]==0.10.2"

python data_scripts/build_boxoban_banks.py
```

# Test run
```
mkdir -p $SLURM_TMPDIR/tmp
export CUDA_MPS_LOG_DIRECTORY=$SLURM_TMPDIR/tmp
nvidia-cuda-mps-control -d

MAX_THINK_ITERS=5 DISCOUNT_ACT=0.99 DISCOUNT_COMPUTE=1.0 sokoban/scripts/train-vulcan.sh looped 1 exp/sokoban 0 &
MAX_THINK_ITERS=4 DISCOUNT_ACT=0.99 DISCOUNT_COMPUTE=0.999 sokoban/scripts/train-vulcan.sh looped 1 exp/sokoban 0 &
wait
```

```
TARGET_KL=0.03 VALUE_EXPECTILES="0.5" ISO_ITERS="4" MEM_ITERS="4" GPUS="0" MAX_RUNS_PER_GPU=6 MAX_JOB_HOURS=3 ENVS="lightsout" SEEDS="1 2 3 4 5" slurm/submit_mem.sh

TARGET_KL=0.03 VALUE_EXPECTILES="0.1" ISO_ITERS="4" MEM_ITERS="4" GPUS="0" MAX_RUNS_PER_GPU=6 MAX_JOB_HOURS=3 ENVS="lightsout" SEEDS="1 2 3 4 5" slurm/submit_mem.sh

NUM_EPOCHS=2 VALUE_EXPECTILES="0.5" ISO_ITERS="4" MEM_ITERS="4" GPUS="0" MAX_RUNS_PER_GPU=6 MAX_JOB_HOURS=3 ENVS="lightsout" SEEDS="1 2 3 4 5" slurm/submit_mem.sh

NUM_EPOCHS=2 VALUE_EXPECTILES="0.1" ISO_ITERS="4" MEM_ITERS="4" GPUS="0" MAX_RUNS_PER_GPU=6 MAX_JOB_HOURS=3 ENVS="lightsout" SEEDS="1 2 3 4 5" slurm/submit_mem.sh
```