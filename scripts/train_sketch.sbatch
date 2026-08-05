#!/bin/bash
#SBATCH --job-name=starx_ft
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out
#
# Fine-tune TripoSR on the synthetic sketch dataset, on both A100s, detached
# from any terminal - the job survives a closed laptop, a dropped VPN and a
# dead SSH session, which a Jupyter tunnel does not.
#
#   mkdir -p slurm_logs && sbatch scripts/train_sketch.sbatch
#   squeue -u $USER            # watch it queue and start
#   tail -f slurm_logs/starx_ft-<jobid>.out
#   scancel <jobid>            # stop it; the next submit resumes the run
#
# Measured on 2x A100 (jobs 49612917 / 49613075 / 49613076), 13 supervision
# views: throughput is FLAT in batch size - 0.72 / 0.80 / 0.64 designs per
# second at 2 / 8 / 16 designs per rank - and peak VRAM barely moves (12.2 /
# 12.3 / 12.4 GiB of 80). The two-stage backward frees each view's graph as
# it goes, so memory is per-view and the cost is the 13 sequential crop
# renders per design, not occupancy. Bigger batches therefore buy nothing
# and cost proportionally more per step, so the batch is deliberately SMALL:
# it maximizes optimizer updates per hour. The lever for real speed is
# fewer supervision views, or batching a design's views into one render.
#
# At this setting: 12,000 steps x 4 designs = 48,000 design-visits, ~17 h.
#
# Resumable by design: it writes checkpoints into the run directory and
# picks up from the newest one, so re-submitting after the 24 h wall clock
# (or after a crash) continues rather than restarting. Chain it with
#   sbatch --dependency=afterany:<jobid> scripts/train_sketch.sbatch
# to cover a run longer than one allocation.

set -euo pipefail

REPO="${STARX_REPO:-$HOME/StarX}"
# see finetune_ssim3d.sbatch's comment: these match this project's actual
# Ibex layout (notebook 08's /ibex/user/$USER/StarX, $HOME/miniconda3), not
# generic placeholders - a wrong fallback here fails preflight in ~1s rather
# than silently training against the wrong data or a missing interpreter.
DATA_ROOT="${STARX_DATA:-/ibex/user/$USER/StarX}"
RUN_NAME="${STARX_RUN:-sketch_ft_ddp}"
PYTHON="${STARX_PYTHON:-$HOME/miniconda3/envs/starx/bin/python}"

# never let a --user install shadow the environment's torch
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
# a compute node may have no route out: use the cached TripoSR checkpoint and
# fail fast instead of hanging on a network timeout mid-allocation
export HF_HUB_OFFLINE=1

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
      --batch-designs 2 \
      --total-steps 12000 \
      --lr 1e-4 \
      --warmup-steps 500 \
      --workers 8

echo ">>> finished at $(date)"
