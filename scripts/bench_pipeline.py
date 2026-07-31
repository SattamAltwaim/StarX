"""Is the step slow because of compute, or because it is waiting for data?

A single-rank profile over a handful of designs said the compute in a
fully-unfrozen step is ~2s, while the real two-rank job over the whole
dataset does 18s. This reproduces the REAL conditions - both ranks, the
full training split, the same worker count - and separates the time spent
blocked on the DataLoader from the time spent computing.

  torchrun --standalone --nproc_per_node=2 scripts/bench_pipeline.py \
      --data-root $HOME/StarX/data/StarX --workers 8

If "waiting for data" dominates, the fix is the input pipeline (workers,
CPU oversubscription, filesystem) and not the model.
"""

import argparse
import os
import sys
import time
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

import numpy as np
import torch
import torch.distributed as dist

from starx import sketchdata
from starx import model as smodel
from starx import train as strain
from starx.config import StarXConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--supervision-views", type=int, default=13)
    parser.add_argument("--batch-designs", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--stage", type=str, default="last",
                        choices=["first", "last"])
    parser.add_argument("--triposr-dir",
                        default=str(REPO_DIR / "third_party" / "TripoSR"))
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    main_rank = rank == 0

    cfg = StarXConfig(
        drive_root=args.data_root,
        local_root=os.path.join(args.data_root, "local_cache"),
        lambda_occ=0.0, composite_bg=0.5,
        supervision_views=args.supervision_views,
        batch_designs=args.batch_designs,
    )
    cache = Path(cfg.local_root) / "train" / "cache"
    train_ids, _ = sketchdata.split_designs(cache, seed=cfg.seed)
    dataset = sketchdata.SketchDataset(
        cache, design_ids=train_ids,
        supervision_views=args.supervision_views, seed=cfg.seed, cfg=cfg,
    )
    sampler = (
        torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=world, rank=rank, shuffle=True, drop_last=True
        )
        if world > 1 else None
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_designs, shuffle=sampler is None,
        sampler=sampler, num_workers=args.workers,
        collate_fn=sketchdata.collate, drop_last=True, pin_memory=True,
        prefetch_factor=4 if args.workers else None,
    )

    model = smodel.load_pretrained_tsr(args.triposr_dir, device=device)
    model.renderer.set_chunk_size(0)
    total_steps = 10**9
    strain.apply_unfreeze_stage(
        model, 0 if args.stage == "first" else total_steps, total_steps
    )
    optimizer, scheduler = strain.build_finetune_optimizer(model, cfg, 1000)
    trainable = [p for p in model.parameters() if p.requires_grad]

    from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True)
    lpips = lpips.to(device).requires_grad_(False).eval()
    amp_dtype, _ = strain.pick_amp(device)
    rng = np.random.default_rng(0)

    if main_rank:
        print(f"world {world}, {torch.cuda.get_device_name(0)}, "
              f"{args.workers} workers/rank, stage={args.stage}")
        print(f"dataset {len(dataset):,} samples over {len(dataset.designs):,} designs, "
              f"sketches={dataset.sketches}")
        print(f"CPUs visible to this process: {len(os.sched_getaffinity(0))}, "
              f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}\n")

    wait_total = compute_total = 0.0
    counted = 0
    iterator = iter(loader)
    for index in range(args.steps):
        torch.cuda.synchronize()
        t0 = time.time()
        batch = next(iterator)
        torch.cuda.synchronize()
        t1 = time.time()

        totals = strain.paper_train_step(
            batch, model, optimizer, scheduler, trainable, lpips, cfg,
            device, amp_dtype, rng, 1.0, world_size=world,
        )
        torch.cuda.synchronize()
        t2 = time.time()

        if index >= 2:  # discard warmup
            wait_total += t1 - t0
            compute_total += t2 - t1
            counted += 1
        if main_rank:
            print(f"  step {index:2d}  wait {t1 - t0:6.2f}s  compute {t2 - t1:6.2f}s  "
                  f"total {t2 - t0:6.2f}s", flush=True)

    if main_rank and counted:
        step = (wait_total + compute_total) / counted
        print(f"\naveraged over {counted} steps:")
        print(f"  waiting for data : {wait_total / counted:6.2f}s  "
              f"({100 * wait_total / (wait_total + compute_total):.0f}%)")
        print(f"  compute          : {compute_total / counted:6.2f}s  "
              f"({100 * compute_total / (wait_total + compute_total):.0f}%)")
        print(f"  step             : {step:6.2f}s  -> "
              f"{12000 * step / 3600:.1f} h for 12,000 steps")
        print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")

    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
