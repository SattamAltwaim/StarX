"""Multi-GPU training runner - the same loop as notebook 05, torchrun-able.

Single GPU:
    python scripts/train.py --data-root $HOME/StarX/data/StarX --run-name baseline_ibex

Multiple GPUs on one node (Ibex example):
    torchrun --standalone --nproc_per_node=4 scripts/train.py \
        --data-root $HOME/StarX/data/StarX --run-name baseline_ibex_4gpu

Checkpoints, logs, and validation grids land in <data-root>/runs/<run-name>/
in exactly the notebook's format, so notebook 05's analysis cells (curves,
gallery, before/after, probe) read a script-trained run unchanged - and a
run started in the notebook can be continued here or vice versa.

Every rank builds an identical model replica (same seed, same checkpoint),
handles an interleaved slice of each step's designs, and gradients are
summed across ranks inside starx.train.train_step before the identical
optimizer update on every rank.
"""

import argparse
import os
import sys
import time
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from starx import cameras, checkpoint, shards
from starx import data as sdata
from starx import model as smodel
from starx import train as strain
from starx.config import FOVY_DEG, StarXConfig, run_dir, shard_dir


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True,
                        help="the StarX data root (holds shards/, runs/)")
    parser.add_argument("--run-name", default="baseline_ibex")
    parser.add_argument("--smoke", action="store_true",
                        help="overfit the 20-design smoke shards")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--accum-designs", type=int, default=None,
                        help="designs per optimizer step, across all ranks")
    parser.add_argument("--views-per-design", type=int, default=2)
    parser.add_argument("--render-crop", type=int, default=128)
    parser.add_argument("--lambda-occ", type=float, default=0.5)
    parser.add_argument("--hull-res", type=int, default=48)
    parser.add_argument("--ckpt-every", type=int, default=None)
    parser.add_argument("--val-every", type=int, default=None)
    parser.add_argument("--triposr-dir", default=str(REPO_DIR / "third_party" / "TripoSR"))
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main():
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        import torch.distributed as dist

        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    main_rank = rank == 0

    smoke = args.smoke
    prefix = "smoke_" if smoke else ""
    cfg = StarXConfig(
        drive_root=args.data_root,
        local_root=os.path.join(args.data_root, "local_cache"),
        total_steps=args.total_steps or (1000 if smoke else 20000),
        lr=args.lr,
        warmup_steps=100 if smoke else 500,
        accum_designs=args.accum_designs or (4 if smoke else 8),
        views_per_design=args.views_per_design,
        render_crop=args.render_crop,
        lambda_occ=args.lambda_occ,
        hull_res=args.hull_res,
        ckpt_every=args.ckpt_every or (250 if smoke else 500),
        val_every=args.val_every or (200 if smoke else 1000),
        seed=args.seed,
    )
    amp_dtype, grad_scale = strain.pick_amp("cuda" if device.startswith("cuda") else device)
    if main_rank:
        print(f"run {args.run_name}: {cfg.total_steps} steps, "
              f"{cfg.accum_designs} designs x {cfg.views_per_design} views per step "
              f"on {world_size} rank(s), amp={amp_dtype}, lambda_occ={cfg.lambda_occ}")

    # data: rank 0 extracts the shard cache, everyone else waits, all read it
    train_local = Path(cfg.local_root) / f"{prefix}train"
    val_local = Path(cfg.local_root) / f"{prefix}test"
    if main_rank:
        shards.prepare_local(shard_dir(cfg, f"{prefix}train"), train_local)
        shards.prepare_local(shard_dir(cfg, f"{prefix}test"), val_local)
    if distributed:
        import torch.distributed as dist

        dist.barrier()
    train_dataset = sdata.DesignDataset(train_local / "cache")
    val_dataset = sdata.DesignDataset(val_local / "cache")
    if len(val_dataset) == 0:
        val_dataset = train_dataset
    assert len(train_dataset) > 0, "no train shards under the data root"
    if main_rank:
        print(f"train designs: {len(train_dataset)}   val designs: {len(val_dataset)}")

    # identical replicas need identical LoRA init before any checkpoint load
    torch.manual_seed(cfg.seed)
    model, build_info = smodel.build_starx_model(cfg, args.triposr_dir, device=device)
    model.image_tokenizer.model.gradient_checkpointing_enable()
    model.backbone.gradient_checkpointing = True
    model.renderer.set_chunk_size(0)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer, scheduler = strain.build_optimizer(model, trainable_params, cfg)

    from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True)
    lpips_metric = lpips_metric.to(device).requires_grad_(False)
    lpips_metric.eval()

    rdir = run_dir(cfg, args.run_name)
    start_step = 0
    latest = None if args.no_resume else checkpoint.find_latest(rdir)
    if latest is not None:
        ckpt_path, start_step = latest
        state = checkpoint.load_checkpoint(ckpt_path)
        smodel.load_trainable_state_dict(model, state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        checkpoint.restore_rng(state["rng"])
        if main_rank:
            print(f"resumed {args.run_name} at step {start_step}")
    elif main_rank:
        print(f"starting {args.run_name} fresh")

    log_path = rdir / "logs" / "train_log.jsonl"
    grid_dir = rdir / "val_grids"
    if main_rank:
        grid_dir.mkdir(parents=True, exist_ok=True)
    val_items = [val_dataset[i] for i in range(min(4, len(val_dataset)))]

    def save_val_grid(step_number):
        model.eval()
        model.renderer.set_chunk_size(cfg.eval_chunk)
        fig, axes = plt.subplots(len(val_items), 3, figsize=(9.6, 3.1 * len(val_items)))
        axes = np.atleast_2d(axes)
        with torch.no_grad():
            for row, item in enumerate(val_items):
                with torch.autocast("cuda", dtype=amp_dtype, enabled=device.startswith("cuda")):
                    code = smodel.encode_sketches(model, item["sketch"][None].to(device))[0]
                code = code.float()
                rays_o, rays_d = cameras.rays_full(item["c2ws"][0], FOVY_DEG, cfg.gt_size)
                rgb_fg, opacity = strain.render_rays(
                    model, code, rays_o.to(device), rays_d.to(device)
                )
                pred = strain.composite_over_gray(rgb_fg, opacity).clamp(0, 1).cpu().numpy()
                gt = item["views"][0].astype(np.float32) / 255.0
                gt = np.where(item["masks"][0][..., None], gt, 0.5)
                axes[row, 0].imshow(item["stack_uint8"][0], cmap="gray", vmin=0, vmax=255)
                axes[row, 1].imshow(gt)
                axes[row, 2].imshow(pred)
        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        axes[0, 2].set_title(f"prediction @ step {step_number}")
        fig.savefig(grid_dir / f"step_{step_number:07d}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        model.renderer.set_chunk_size(0)
        model.train()

    for step in range(start_step, cfg.total_steps):
        step_started = time.time()
        totals = strain.train_step(
            step, model, train_dataset, optimizer, scheduler, trainable_params,
            lpips_metric, cfg, device, amp_dtype, grad_scale,
            rank=rank, world_size=world_size,
        )
        completed = step + 1
        if main_rank:
            if step == start_step:
                print(f"first step: {time.time() - step_started:.1f}s   "
                      f"peak VRAM: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
            if completed % 10 == 0:
                print(f"step {completed}/{cfg.total_steps}  "
                      f"loss {totals['loss']:.3f}  occ {totals['occ']:.3f}  "
                      f"({time.time() - step_started:.1f}s)", flush=True)
            if completed % 50 == 0 or completed == cfg.total_steps:
                checkpoint.append_log(log_path, {"step": completed, **totals})
            if completed % cfg.ckpt_every == 0 or completed == cfg.total_steps:
                checkpoint.save_checkpoint(
                    rdir, completed,
                    {
                        "model": smodel.trainable_state_dict(model),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "rng": checkpoint.rng_states(),
                    },
                    keep_k=cfg.keep_k,
                )
            if (completed % cfg.val_every == 0 or completed == cfg.total_steps
                    or completed == start_step + 1):
                save_val_grid(completed)

    if distributed:
        import torch.distributed as dist

        dist.destroy_process_group()
    if main_rank:
        print("training finished")


if __name__ == "__main__":
    main()
