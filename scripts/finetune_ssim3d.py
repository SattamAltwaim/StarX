"""Fine-tune an ALREADY fine-tuned TripoSR checkpoint with an added 3D SSIM
term, on the reconstruction dataset plus the assembly dataset. Ibex-ready,
same shape as train_sketch.py - this is that script with three things
layered on top:

1. --init-checkpoint seeds model weights from a prior run's final checkpoint
   (e.g. the sketch_ft_ddp run scripts/train_sketch.sbatch produces) instead
   of starting from stock TripoSR. Optimizer/scheduler/step are NOT carried
   over: this is a new run with a new loss, not a resumed one. Resuming an
   in-progress run of THIS script (same --run-name, re-submitted after a
   crash or the wall clock) still takes priority, same as train_sketch.py.

2. cfg.lambda_ssim3d turns on a 3D structural-similarity term between the
   triplane's predicted occupancy grid and the design's visual hull, scored
   by starx.ssim3d (SSIM3D, differentiable, separable-Gaussian). See
   starx.train.occupancy_3d_terms / ssim3d_occupancy_loss.

3. --no-assembly aside, the assembly dataset (notebook 16's
   assembly_shards/assembly_sketch_shards - Fusion 360 Gallery joint data,
   j1.0.0) is extracted into the SAME local cache as the reconstruction
   shards. starx.shards' tar/meta.json schema is dataset-agnostic and
   SketchDataset just globs *.meta.json, so the two design pools merge into
   one training set with no Dataset-side changes - see starx.config.
   assembly_shard_dir's docstring.

Single GPU:
    python scripts/finetune_ssim3d.py --data-root $HOME/StarX/data/StarX \
        --run-name ssim3d_assembly_ft \
        --init-checkpoint $HOME/StarX/data/StarX/runs/sketch_ft_ddp

Both A100s on an Ibex node:
    torchrun --standalone --nproc_per_node=2 scripts/finetune_ssim3d.py \
        --data-root $HOME/StarX/data/StarX --run-name ssim3d_assembly_ft_2gpu \
        --init-checkpoint $HOME/StarX/data/StarX/runs/sketch_ft_ddp

--init-checkpoint accepts either a run directory (its newest checkpoint is
used) or a specific state_*.pt file.

Everything scripts/train_sketch.py's module docstring says about what "the
paper recipe" does and does not reproduce still applies here - this script
starts from that recipe's own output, not from the paper's numbers.
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
    assembly_shard_dir,
    assembly_sketch_shard_dir,
    run_dir,
    shard_dir,
    sketch_shard_dir,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", required=True,
                        help="the StarX data root (holds shards/, assembly_shards/, runs/)")
    parser.add_argument("--run-name", default="ssim3d_assembly_ft")
    parser.add_argument("--smoke", action="store_true",
                        help="rehearse on the 20-design smoke shards")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--init-checkpoint", default=None,
                        help="seed model weights from a prior run's checkpoint (run "
                             "dir or state_*.pt file) when this run has none of its "
                             "own yet - see the module docstring")

    # data
    parser.add_argument("--no-assembly", action="store_true",
                         help="train on the reconstruction dataset only; by default "
                              "the assembly (joint) dataset is merged in too")

    # the paper's six (see train_sketch.py)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="fine-tuning a converged checkpoint again - default is "
                             "already the conservative choice, not the paper's 4e-4")
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--lambda-lpips", type=float, default=2.0)
    parser.add_argument("--lambda-mask", type=float, default=0.05)
    parser.add_argument("--render-crop", type=int, default=128)

    # the 3D terms
    parser.add_argument("--lambda-ssim3d", type=float, default=0.1,
                        help="3D SSIM term weight against the visual hull; 0 disables")
    parser.add_argument("--ssim3d-win", type=int, default=7,
                        help="odd, must be <= --hull-res")
    parser.add_argument("--ssim3d-sigma", type=float, default=1.5)
    parser.add_argument("--lambda-occ", type=float, default=0.0,
                        help="soft-Dice term against the same hull; 0 disables, can "
                             "be combined with --lambda-ssim3d")
    parser.add_argument("--hull-res", type=int, default=48)

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


def resolve_init_checkpoint(path) -> Path:
    """A run dir (its newest checkpoint) or a specific state_*.pt file."""
    path = Path(path)
    if path.is_dir():
        latest = checkpoint.find_latest(path)
        if latest is None:
            raise FileNotFoundError(f"no checkpoints under {path}")
        return latest[0]
    if not path.exists():
        raise FileNotFoundError(f"--init-checkpoint {path} does not exist")
    return path


def _shard_design_ids(shard_dir_path, prefix: str = "shard") -> set:
    """Design ids covered by a shard set, read from the small .done.json
    markers only - no tar extraction needed."""
    ids = set()
    for tar in shards.list_done_shards(shard_dir_path, prefix):
        marker = json.loads(shards.marker_path(tar).read_text())
        ids.update(marker["ids"])
    return ids


def extract_split(cfg, split: str, local_dir: Path, include_assembly: bool) -> dict:
    """Extract one split's shards into local_dir/cache, optionally merging
    the assembly shard set into the SAME cache (see the module docstring).

    Reconstruction and assembly ids come from disjoint Fusion 360 Gallery
    datasets, but nothing enforces that at write time, and a collision would
    silently make one design's files overwrite the other's in the shared
    cache - so this checks the id sets before touching the assembly tars.
    """
    counts = {}
    shards.prepare_local(shard_dir(cfg, split), local_dir)
    recon_ids = _shard_design_ids(shard_dir(cfg, split))
    counts["reconstruction"] = len(recon_ids)
    sk_dir = sketch_shard_dir(cfg, split)
    if shards.list_done_shards(sk_dir, sketchdata.SKETCH_PREFIX):
        sketchdata.check_params(sk_dir, cfg)
        shards.prepare_local(sk_dir, local_dir, prefix=sketchdata.SKETCH_PREFIX)

    if include_assembly:
        a_dir = assembly_shard_dir(cfg, split)
        assembly_ids = _shard_design_ids(a_dir)
        if not assembly_ids:
            raise FileNotFoundError(
                f"--no-assembly was not passed but no assembly shards exist at "
                f"{a_dir} - run notebook 16 for this split, or pass --no-assembly"
            )
        overlap = recon_ids & assembly_ids
        if overlap:
            raise ValueError(
                f"{len(overlap)} design id(s) collide between the reconstruction "
                f"and assembly shards for split {split!r} (e.g. {sorted(overlap)[:3]}) "
                f"- merging them into one cache would silently drop samples"
            )
        shards.prepare_local(a_dir, local_dir)
        counts["assembly"] = len(assembly_ids)
        a_sk_dir = assembly_sketch_shard_dir(cfg, split)
        if shards.list_done_shards(a_sk_dir, sketchdata.SKETCH_PREFIX):
            sketchdata.check_params(a_sk_dir, cfg)
            shards.prepare_local(a_sk_dir, local_dir, prefix=sketchdata.SKETCH_PREFIX)
    return counts


def main():
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        import datetime

        import torch.distributed as dist

        # rank 0 alone does the (possibly slow, first-run) shard extraction
        # below before every rank's first collective op (the barrier right
        # after it) - the default 10-minute NCCL store timeout has nothing
        # to do with how long that takes and was measured failing here.
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=60))
        torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    main_rank = rank == 0
    include_assembly = not args.no_assembly

    prefix = "smoke_" if args.smoke else ""
    cfg = StarXConfig(
        drive_root=args.data_root,
        local_root=os.path.join(args.data_root, "local_cache"),
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        lambda_lpips=args.lambda_lpips,
        lambda_mask=args.lambda_mask,
        lambda_occ=args.lambda_occ,
        lambda_ssim3d=args.lambda_ssim3d,
        ssim3d_win_size=args.ssim3d_win,
        ssim3d_sigma=args.ssim3d_sigma,
        hull_res=args.hull_res,
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
    use_3d = cfg.lambda_occ > 0 or cfg.lambda_ssim3d > 0
    amp_dtype, grad_scale = strain.pick_amp(
        "cuda" if device.startswith("cuda") else device
    )

    # data: rank 0 extracts every shard set into one cache, the others wait
    train_local = Path(cfg.local_root) / f"{prefix}train"
    val_local = Path(cfg.local_root) / f"{prefix}test"
    if main_rank:
        for split, local in ((f"{prefix}train", train_local), (f"{prefix}test", val_local)):
            counts = extract_split(cfg, split, local, include_assembly)
            print(f"  {split}: {counts}", flush=True)
    if distributed:
        import torch.distributed as dist

        dist.barrier()

    train_ids, val_ids = sketchdata.split_designs(
        train_local / "cache", val_fraction=0.05, seed=cfg.seed
    )
    common = dict(supervision_views=cfg.supervision_views, seed=cfg.seed, cfg=cfg,
                  return_all_masks=use_3d)
    train_dataset = sketchdata.SketchDataset(
        train_local / "cache", design_ids=train_ids, **common
    )
    val_dataset = sketchdata.SketchDataset(
        train_local / "cache", design_ids=val_ids, **common
    )
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
        print(f"  assembly dataset: {'included' if include_assembly else 'excluded'}")
        print(f"  {cfg.batch_designs} designs x {cfg.supervision_views} views per step "
              f"per rank, {world_size} rank(s)")
        print(f"  {steps_per_epoch} steps/epoch x {epochs} epochs = {total_steps} steps")
        print(f"  lr {cfg.lr}, warmup {cfg.warmup_steps}, wd {cfg.weight_decay}, "
              f"betas {tuple(cfg.adam_betas)}, amp {amp_dtype}")
        print(f"  fine-tuning: {'LoRA' if args.lora else 'every parameter'}")
        print(f"  3D terms: lambda_occ={cfg.lambda_occ}, "
              f"lambda_ssim3d={cfg.lambda_ssim3d} "
              f"(win={cfg.ssim3d_win_size}, sigma={cfg.ssim3d_sigma}, "
              f"hull_res={cfg.hull_res})")
        if args.init_checkpoint:
            print(f"  init checkpoint: {args.init_checkpoint}")

    torch.manual_seed(cfg.seed)  # identical replicas before any checkpoint load
    model, build_info = smodel.build_stock_lora_model(
        cfg, args.triposr_dir, device=device, full_finetune=not args.lora
    )
    model.image_tokenizer.model.gradient_checkpointing_enable()
    model.backbone.gradient_checkpointing = True
    model.renderer.set_chunk_size(0)
    stage_name = strain.apply_unfreeze_stage(model, 0, total_steps)
    optimizer, scheduler = strain.build_finetune_optimizer(model, cfg, total_steps)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
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
        stage_name = strain.apply_unfreeze_stage(model, start_step, total_steps)
        smodel.load_trainable_state_dict(model, state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        checkpoint.restore_rng(state["rng"])
        if main_rank:
            print(f"  resumed at step {start_step}")
    elif args.init_checkpoint:
        init_path = resolve_init_checkpoint(args.init_checkpoint)
        state = checkpoint.load_checkpoint(init_path)
        info = smodel.load_full_state_dict(model, state["model"])
        if main_rank:
            print(f"  seeded {info['loaded']:,} parameters from {init_path}")
    elif main_rank:
        print("  starting fresh (no --init-checkpoint - stock TripoSR)")

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
            opened = strain.apply_unfreeze_stage(model, step, total_steps)
            if opened != stage_name:
                stage_name = opened
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                if main_rank:
                    live = sum(p.numel() for p in trainable_params)
                    print(f"  step {step}: unfroze -> '{stage_name}' "
                          f"({live:,} trainable)", flush=True)
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
                    extra = ""
                    if use_3d:
                        extra = f"  occ {totals['occ']:.4f}  ssim3d {totals['ssim3d']:.4f}"
                    print(f"step {step}/{total_steps} (epoch {epoch})  "
                          f"loss {totals['loss']:.4f}  lr {totals['lr']:.2e}  "
                          f"{rate:.2f} step/s{extra}", flush=True)
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
