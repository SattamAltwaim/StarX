"""Fine-tune stock TripoSR on the materialized sketch dataset, Ibex-ready.

This is the paper-recipe arm: no input surgery, no auxiliary 3D term, an
ordinary supervised loop over a torch DataLoader. Notebook 10 builds the
dataset it reads; notebook 11 drives it interactively.

Single GPU:
    python scripts/train_sketch.py --data-root $HOME/StarX/data/StarX \
        --run-name sketch_paper

Both A100s on an Ibex node:
    torchrun --standalone --nproc_per_node=2 scripts/train_sketch.py \
        --data-root $HOME/StarX/data/StarX --run-name sketch_paper_2gpu

Under a Slurm allocation, run it inside the job (srun --pty ... then
torchrun), or submit it with an sbatch wrapper - torchrun's --standalone
rendezvous is per-node, which is all a 2x A100 node needs.

What "the paper recipe" means and does not mean
-----------------------------------------------
TripoSR's report states exactly six training numbers: AdamW, lr 4e-4,
CosineAnnealingLR, 2,000 warm-up steps, lambda_LPIPS 2.0, lambda_mask 0.05
- plus 128px foreground-biased crops out of 512px ground truth and a BCE
mask loss on rendered opacity. Those are all reproduced here.

The report is SILENT on batch size, total steps, supervision views per
object, weight decay, gradient clipping, and precision. Those are taken
from LRM, the architecture TripoSR builds on, or fitted to this hardware,
and every one is a flag below. Three things genuinely cannot match:

- ground truth is 256px here, not the paper's 512px, because that is what
  notebook 03 rendered; a 128px crop therefore covers four times as much
  of the object as the paper's does. Re-render at 512 to close this.
- lr 4e-4 is a FROM-SCRATCH rate for ~950k objects at batch 1024. This run
  fine-tunes a converged checkpoint on a few thousand designs, where that
  rate will walk it a long way from pretraining. --lr is the knob; 1e-4 or
  lower is the safer fine-tuning choice, and --lora makes it safer still.
- the paper's batch was LRM's 1024 shapes per iteration on 128 A100s.
"""

import argparse
import json
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

from starx import cameras, checkpoint, shards, sketchdata
from starx import model as smodel
from starx import train as strain
from starx.config import (
    CAMERA_DISTANCE,
    FOVY_DEG,
    StarXConfig,
    run_dir,
    shard_dir,
    sketch_shard_dir,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", required=True,
                        help="the StarX data root (holds shards/, sketch_shards/, runs/)")
    parser.add_argument("--run-name", default="sketch_paper")
    parser.add_argument("--smoke", action="store_true",
                        help="rehearse on the 20-design smoke shards")
    parser.add_argument("--no-resume", action="store_true")

    # the paper's six
    parser.add_argument("--lr", type=float, default=4e-4,
                        help="paper value; lower it for fine-tuning (see module docstring)")
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--lambda-lpips", type=float, default=2.0)
    parser.add_argument("--lambda-mask", type=float, default=0.05)
    parser.add_argument("--render-crop", type=int, default=128)

    # the paper is silent on these
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=None,
                        help="stop here regardless of epochs (default: run all epochs)")
    parser.add_argument("--batch-designs", type=int, default=8,
                        help="shapes per optimizer step PER RANK")
    parser.add_argument("--supervision-views", type=int, default=4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--composite-bg", type=float, default=0.5,
                        help="0.5 = TripoSR's inference gray, 1.0 = LRM's white")
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--lora", action="store_true",
                        help="LoRA instead of full fine-tuning (not the paper, but "
                             "far cheaper and much gentler on the checkpoint)")
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--keep-k", type=int, default=2)
    parser.add_argument("--triposr-dir",
                        default=str(REPO_DIR / "third_party" / "TripoSR"))
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

    prefix = "smoke_" if args.smoke else ""
    cfg = StarXConfig(
        drive_root=args.data_root,
        local_root=os.path.join(args.data_root, "local_cache"),
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        lambda_lpips=args.lambda_lpips,
        lambda_mask=args.lambda_mask,
        lambda_occ=0.0,  # the paper relies on rendering losses alone
        render_crop=args.render_crop,
        batch_designs=args.batch_designs,
        supervision_views=args.supervision_views,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        composite_bg=args.composite_bg,
        ckpt_every=args.ckpt_every,
        val_every=args.val_every,
        keep_k=args.keep_k,
        seed=args.seed,
    )
    amp_dtype, grad_scale = strain.pick_amp(
        "cuda" if device.startswith("cuda") else device
    )

    # data: rank 0 extracts both shard sets into one cache, the others wait
    train_local = Path(cfg.local_root) / f"{prefix}train"
    val_local = Path(cfg.local_root) / f"{prefix}test"
    if main_rank:
        for split, local in ((f"{prefix}train", train_local), (f"{prefix}test", val_local)):
            shards.prepare_local(shard_dir(cfg, split), local)
            sketch_dir = sketch_shard_dir(cfg, split)
            if not shards.list_done_shards(sketch_dir, sketchdata.SKETCH_PREFIX):
                raise SystemExit(
                    f"no sketch shards under {sketch_dir} - run notebook 10 "
                    f"(or starx.sketchdata.build_sketch_shards) first"
                )
            sketchdata.check_params(sketch_dir, cfg)
            shards.prepare_local(sketch_dir, local, prefix=sketchdata.SKETCH_PREFIX)
    if distributed:
        import torch.distributed as dist

        dist.barrier()

    train_dataset = sketchdata.SketchDataset(
        train_local / "cache", supervision_views=cfg.supervision_views, seed=cfg.seed
    )
    val_dataset = sketchdata.SketchDataset(
        val_local / "cache", supervision_views=cfg.supervision_views, seed=cfg.seed
    )
    if len(val_dataset) == 0:
        val_dataset = train_dataset
    if len(train_dataset) == 0:
        raise SystemExit("no samples - was the sketch dataset built for this split?")

    sampler = None
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank,
            shuffle=True, drop_last=True, seed=cfg.seed,
        )
    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_designs,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.workers,
        collate_fn=sketchdata.collate,
        drop_last=True,
        pin_memory=device.startswith("cuda"),
        persistent_workers=args.workers > 0,
    )
    steps_per_epoch = len(loader)
    epochs = args.epochs or (2 if args.smoke else 20)
    total_steps = args.total_steps or steps_per_epoch * epochs

    if main_rank:
        print(f"run {args.run_name}")
        print(f"  {train_dataset.describe()}")
        print(f"  {cfg.batch_designs} designs x {cfg.supervision_views} views per step "
              f"per rank, {world_size} rank(s)")
        print(f"  {steps_per_epoch} steps/epoch x {epochs} epochs = {total_steps} steps")
        print(f"  lr {cfg.lr}, warmup {cfg.warmup_steps}, wd {cfg.weight_decay}, "
              f"betas {tuple(cfg.adam_betas)}, amp {amp_dtype}")
        print(f"  fine-tuning: {'LoRA' if args.lora else 'every parameter'}")
        if not args.lora and cfg.lr >= 4e-4:
            print("  NOTE: 4e-4 is the paper's FROM-SCRATCH rate; for fine-tuning a "
                  "converged checkpoint consider --lr 1e-4 or lower")

    torch.manual_seed(cfg.seed)  # identical replicas before any checkpoint load
    model, build_info = smodel.build_stock_lora_model(
        cfg, args.triposr_dir, device=device, full_finetune=not args.lora
    )
    model.image_tokenizer.model.gradient_checkpointing_enable()
    model.backbone.gradient_checkpointing = True
    model.renderer.set_chunk_size(0)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer, scheduler = strain.build_paper_optimizer(model, cfg, total_steps)
    if main_rank:
        totals = build_info["param_table"]["ALL"]
        print(f"  trainable {totals['trainable']:,} / {totals['total']:,}")

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
            print(f"  resumed at step {start_step}")
    elif main_rank:
        print("  starting fresh")

    log_path = rdir / "logs" / "train_log.jsonl"
    grid_dir = rdir / "val_grids"
    if main_rank:
        grid_dir.mkdir(parents=True, exist_ok=True)
        (rdir / "logs").mkdir(parents=True, exist_ok=True)
        (rdir / "recipe.json").write_text(json.dumps(vars(args), indent=1))

    novel_c2w = cameras.build_spherical_c2w(90.0, 20.0, CAMERA_DISTANCE)
    val_indices = [
        i * val_dataset.n_views for i in range(min(4, len(val_dataset.designs)))
    ]

    def save_val_grid(step_number):
        """Input | ground truth | prediction | prediction turned 90 degrees."""
        model.eval()
        model.renderer.set_chunk_size(cfg.eval_chunk)
        rows = [val_dataset[i] for i in val_indices]
        fig, axes = plt.subplots(len(rows), 4, figsize=(12.8, 3.1 * len(rows)))
        axes = np.atleast_2d(axes)
        with torch.no_grad():
            for row, item in enumerate(rows):
                with torch.autocast("cuda", dtype=amp_dtype,
                                    enabled=device.startswith("cuda")):
                    code = smodel.encode_sketches(model, item["input"][None].to(device))[0]
                code = code.float()
                gt_size = item["views"].shape[1]
                renders = []
                for c2w in (item["c2ws"][0].numpy(), novel_c2w):
                    rays_o, rays_d = cameras.rays_full(c2w, FOVY_DEG, gt_size)
                    rgb_fg, opacity = strain.render_rays(
                        model, code, rays_o.to(device), rays_d.to(device)
                    )
                    renders.append(
                        strain.composite_over_gray(rgb_fg, opacity, cfg.composite_bg)
                        .clamp(0, 1).cpu().numpy()
                    )
                gt = item["views"][0].numpy().astype(np.float32) / 255.0
                gt = np.where(item["masks"][0].numpy()[..., None], gt, cfg.composite_bg)
                axes[row, 0].imshow(item["input"][0], cmap="gray", vmin=0, vmax=1)
                axes[row, 1].imshow(gt)
                axes[row, 2].imshow(renders[0])
                axes[row, 3].imshow(renders[1])
        for ax in axes.ravel():
            ax.set_xticks([])
            ax.set_yticks([])
        for col, title in enumerate(
            ["sketch (input)", "ground truth", f"prediction @ {step_number}", "turned 90 deg"]
        ):
            axes[0, col].set_title(title, fontsize=10)
        fig.savefig(grid_dir / f"step_{step_number:07d}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        model.renderer.set_chunk_size(0)
        model.train()

    step = start_step
    crop_rng = np.random.default_rng(cfg.seed + 9973 * start_step)
    started = time.time()
    while step < total_steps:
        epoch = step // max(1, steps_per_epoch)
        train_dataset.set_epoch(epoch)
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            if step >= total_steps:
                break
            step_started = time.time()
            totals = strain.paper_train_step(
                batch, model, optimizer, scheduler, trainable_params,
                lpips_metric, cfg, device, amp_dtype, crop_rng, grad_scale,
                world_size=world_size,
            )
            step += 1
            if main_rank:
                if step == start_step + 1:
                    peak = (torch.cuda.max_memory_allocated() / 2**30
                            if device.startswith("cuda") else 0.0)
                    print(f"  first step {time.time() - step_started:.1f}s, "
                          f"peak VRAM {peak:.1f} GiB")
                if step % 10 == 0:
                    rate = (step - start_step) / (time.time() - started)
                    print(f"step {step}/{total_steps} (epoch {epoch})  "
                          f"loss {totals['loss']:.4f}  lr {totals['lr']:.2e}  "
                          f"{rate:.2f} step/s", flush=True)
                if step % 50 == 0 or step == total_steps:
                    checkpoint.append_log(log_path, {"step": step, "epoch": epoch, **totals})
                if step % cfg.ckpt_every == 0 or step == total_steps:
                    checkpoint.save_checkpoint(
                        rdir, step,
                        {
                            "model": smodel.trainable_state_dict(model),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "rng": checkpoint.rng_states(),
                        },
                        keep_k=cfg.keep_k,
                    )
                if step % cfg.val_every == 0 or step == total_steps:
                    save_val_grid(step)

    if main_rank:
        print(f"finished {total_steps} steps in {(time.time() - started) / 3600:.2f} h")
    if distributed:
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    main()
