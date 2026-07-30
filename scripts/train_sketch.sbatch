#!/bin/bash
#SBATCH --job-name=starx_ft
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#
# Fine-tune TripoSR on the synthetic sketch dataset, on both A100s, detached
# from any terminal - the job survives a closed laptop, a dropped VPN and a
# dead SSH session, which a Jupyter tunnel does not.
#
#   sbatch scripts/train_sketch.sbatch
#   squeue -u $USER            # watch it queue and start
#   tail -f starx_ft-<jobid>.out
#   scancel <jobid>            # stop it; the next submit resumes the run
#
# Resumable by design: it writes checkpoints into the run directory and
# picks up from the newest one, so re-submitting after the 24 h wall clock
# (or after a crash) continues rather than restarting. Chain it with
#   sbatch --dependency=afterany:<jobid> scripts/train_sketch.sbatch
# to cover a run longer than one allocation.

set -euo pipefail

REPO="${STARX_REPO:-$HOME/StarX}"
DATA_ROOT="${STARX_DATA:-$REPO/data/StarX}"
RUN_NAME="${STARX_RUN:-sketch_ft_ddp}"
PYTHON="${STARX_PYTHON:-/ibex/user/$USER/miniforge/envs/starx/bin/python}"

# never let a --user install shadow the environment's torch
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

echo "=============================================================="
echo "job        : ${SLURM_JOB_ID:-none} on $(hostname)"
echo "repo       : $REPO"
echo "data root  : $DATA_ROOT"
echo "run name   : $RUN_NAME"
echo "python     : $PYTHON"
echo "gpus       : ${SLURM_GPUS_ON_NODE:-?}"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
echo "=============================================================="

cd "$REPO"

# 1. Preflight on the CPU FIRST. Imports, versions, paths, the shards, the
#    Dataset, the DataLoader, the unfreeze schedule, the optimizer grouping
#    and a checkpoint round-trip - everything that can fail for a boring
#    reason. If any of it is wrong the job stops here, in seconds, instead
#    of after the model has loaded.
echo ">>> preflight"
"$PYTHON" scripts/preflight.py --data-root "$DATA_ROOT" --run-name "$RUN_NAME" || {
    echo "PREFLIGHT FAILED - not starting training"
    exit 1
}

# 2. Train on every allocated GPU. torchrun --standalone rendezvouses within
#    this node, which is all a 2x A100 node needs.
NPROC="${SLURM_GPUS_ON_NODE:-2}"
echo ">>> training on $NPROC GPU(s)"
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
  "$PYTHON" -m torch.distributed.run \
    --standalone --nproc_per_node="$NPROC" \
    scripts/train_sketch.py \
      --data-root "$DATA_ROOT" \
      --run-name "$RUN_NAME" \
      --supervision-views 13 \
      --batch-designs 4 \
      --total-steps 12000 \
      --lr 1e-4 \
      --warmup-steps 500 \
      --workers 8

echo ">>> finished at $(date)"
