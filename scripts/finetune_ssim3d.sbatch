#!/bin/bash
#SBATCH --job-name=starx_ssim3d_ft
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out
#
# Re-fine-tune the already-fine-tuned sketch_ft_ddp checkpoint with the 3D
# SSIM3D term added and the assembly dataset merged in, on both A100s,
# detached from any terminal - the job survives a closed laptop, a dropped
# VPN and a dead SSH session.
#
#   mkdir -p slurm_logs && sbatch scripts/finetune_ssim3d.sbatch
#   squeue -u $USER            # watch it queue and start
#   tail -f slurm_logs/starx_ssim3d_ft-<jobid>.out
#   scancel <jobid>            # stop it; the next submit resumes the run
#
# Resumable by design, same as train_sketch.sbatch: it writes checkpoints
# into the run directory and picks up from the newest one, so re-submitting
# after the 24 h wall clock (or after a crash) continues rather than
# restarting or re-seeding from --init-checkpoint. Chain it with
#   sbatch --dependency=afterany:<jobid> scripts/finetune_ssim3d.sbatch
# to cover a run longer than one allocation.
#
# INIT_RUN must already exist (e.g. the output of train_sketch.sbatch) -
# its newest checkpoint seeds this run's weights. The assembly shards
# (STARX_DATA/assembly_shards, assembly_sketch_shards) must exist too -
# build them with notebook 16 first, or pass STARX_NO_ASSEMBLY=1 to skip.

set -euo pipefail

REPO="${STARX_REPO:-$HOME/StarX}"
# defaults match this project's actual Ibex layout - notebook 08 lands data at
# /ibex/user/$USER/StarX (never $REPO/data/StarX, which is a bare local dir),
# and the conda env lives under $HOME/miniconda3, not /ibex/user's miniforge.
# A job that silently fell back to the old wrong defaults failed its own
# preflight after 1s, which is the gate working - but better to just have the
# right default than rely on STARX_DATA/STARX_PYTHON being re-exported every
# session.
DATA_ROOT="${STARX_DATA:-/ibex/user/$USER/StarX}"
RUN_NAME="${STARX_RUN:-ssim3d_assembly_ft}"
INIT_RUN="${STARX_INIT_RUN:-$DATA_ROOT/runs/sketch_ft_ddp}"
PYTHON="${STARX_PYTHON:-$HOME/miniconda3/envs/starx/bin/python}"
LAMBDA_SSIM3D="${STARX_LAMBDA_SSIM3D:-0.1}"
LAMBDA_OCC="${STARX_LAMBDA_OCC:-0.0}"
LAMBDA_LAPLACIAN="${STARX_LAMBDA_LAPLACIAN:-0.0}"
HULL_RES="${STARX_HULL_RES:-48}"
TOTAL_STEPS="${STARX_TOTAL_STEPS:-12000}"
if [ "${STARX_NO_ASSEMBLY:-0}" = "1" ]; then
    ASSEMBLY_FLAG="--no-assembly"
    PREFLIGHT_ASSEMBLY_FLAG=""
    ASSEMBLY_LABEL="excluded"
else
    ASSEMBLY_FLAG=""
    PREFLIGHT_ASSEMBLY_FLAG="--assembly"
    ASSEMBLY_LABEL="included"
fi

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
echo "init run   : $INIT_RUN"
echo "lambda_ssim3d: $LAMBDA_SSIM3D, lambda_occ: $LAMBDA_OCC, lambda_laplacian: $LAMBDA_LAPLACIAN"
echo "hull_res   : $HULL_RES, total_steps: $TOTAL_STEPS"
echo "assembly   : $ASSEMBLY_LABEL"
echo "python     : $PYTHON"
echo "gpus       : ${SLURM_GPUS_ON_NODE:-?}"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
echo "=============================================================="

cd "$REPO"

# 1. Preflight on the CPU FIRST - same reasoning as train_sketch.sbatch: a
#    typo, a missing shard set or an import error stops here in seconds
#    instead of after the model has loaded on an allocated GPU.
echo ">>> preflight"
"$PYTHON" scripts/preflight.py --data-root "$DATA_ROOT" --run-name "$RUN_NAME" \
    --script scripts/finetune_ssim3d.py \
    $PREFLIGHT_ASSEMBLY_FLAG \
    --init-checkpoint "$INIT_RUN" || {
    echo "PREFLIGHT FAILED - not starting training"
    exit 1
}

# 2. Train on every allocated GPU. torchrun --standalone rendezvouses within
#    this node, which is all a 2x A100 node needs. Deliberately sized at 2
#    rather than 4: a 2-GPU request backfills onto a node that already has
#    2 of its 4 A100s taken by someone else, where a 4-GPU request cannot -
#    much shorter queue time in exchange for roughly double the wall clock.
NPROC="${SLURM_GPUS_ON_NODE:-2}"
echo ">>> training on $NPROC GPU(s)"
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-16}" \
  "$PYTHON" -m torch.distributed.run \
    --standalone --nproc_per_node="$NPROC" \
    scripts/finetune_ssim3d.py \
      --data-root "$DATA_ROOT" \
      --run-name "$RUN_NAME" \
      --init-checkpoint "$INIT_RUN" \
      --lambda-ssim3d "$LAMBDA_SSIM3D" \
      --lambda-occ "$LAMBDA_OCC" \
      --lambda-laplacian "$LAMBDA_LAPLACIAN" \
      --hull-res "$HULL_RES" \
      $ASSEMBLY_FLAG \
      --supervision-views 13 \
      --batch-designs 2 \
      --total-steps "$TOTAL_STEPS" \
      --lr 1e-4 \
      --warmup-steps 500 \
      --workers 8

echo ">>> finished at $(date)"
