"""Where does a training step's time actually go?

Attributes one fully-unfrozen step to its parts - encoder forward, the per-
view crop renders, LPIPS, the per-view backward, and the single encoder
backward - so that a speed decision is made on measurement rather than on
which part looks expensive.

    python scripts/profile_step.py --data-root $HOME/StarX/data/StarX

Everything is timed with torch.cuda.synchronize() around it, because CUDA
is asynchronous and a naive timer attributes a kernel's cost to whichever
line happens to block next.
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

from starx import cameras, data, sketchdata
from starx import model as smodel
from starx import train as strain
from starx.config import FOVY_DEG, StarXConfig, shard_dir


class Timer:
    def __init__(self):
        self.totals = {}

    def __call__(self, name):
        return _Section(self, name)

    def add(self, name, seconds):
        self.totals[name] = self.totals.get(name, 0.0) + seconds

    def report(self, label, step_time):
        print(f"\n{label}   total {step_time:.2f}s")
        print(f"  {'part':28s} {'seconds':>9s} {'share':>7s}")
        for name, seconds in sorted(self.totals.items(), key=lambda kv: -kv[1]):
            print(f"  {name:28s} {seconds:>9.2f} {100 * seconds / step_time:>6.1f}%")
        accounted = sum(self.totals.values())
        print(f"  {'(unattributed)':28s} {step_time - accounted:>9.2f} "
              f"{100 * (step_time - accounted) / step_time:>6.1f}%")


class _Section:
    def __init__(self, timer, name):
        self.timer, self.name = timer, name

    def __enter__(self):
        torch.cuda.synchronize()
        self.t0 = time.time()

    def __exit__(self, *exc):
        torch.cuda.synchronize()
        self.timer.add(self.name, time.time() - self.t0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--supervision-views", type=int, default=13)
    parser.add_argument("--batch-designs", type=int, default=2)
    parser.add_argument("--crop", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--triposr-dir",
                        default=str(REPO_DIR / "third_party" / "TripoSR"))
    args = parser.parse_args()

    device = "cuda:0"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = StarXConfig(
        drive_root=args.data_root,
        local_root=os.path.join(args.data_root, "local_cache"),
        render_crop=args.crop, lambda_occ=0.0, composite_bg=0.5,
        supervision_views=args.supervision_views,
    )
    cache = Path(cfg.local_root) / "train" / "cache"
    train_ids, _ = sketchdata.split_designs(cache, seed=cfg.seed)
    dataset = sketchdata.SketchDataset(
        cache, design_ids=train_ids[: 8 * args.batch_designs],
        supervision_views=args.supervision_views, cfg=cfg,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_designs, shuffle=True,
        collate_fn=sketchdata.collate, num_workers=4, drop_last=True,
    )

    model = smodel.load_pretrained_tsr(args.triposr_dir, device=device)
    model.renderer.set_chunk_size(0)
    strain.apply_unfreeze_stage(model, 10**9, 10**9)   # the worst case
    optimizer, scheduler = strain.build_finetune_optimizer(model, cfg, 1000)

    from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True)
    lpips = lpips.to(device).requires_grad_(False).eval()

    rng = np.random.default_rng(0)
    amp_dtype, _ = strain.pick_amp(device)
    batches = [b for _, b in zip(range(args.repeats + 1), loader)]

    print(f"device      {torch.cuda.get_device_name(0)}")
    print(f"config      {args.batch_designs} designs x {args.supervision_views} "
          f"views, {args.crop}px crops, all {sum(p.numel() for p in model.parameters()):,} "
          f"params trainable")
    print(f"renderer    {model.renderer.cfg.num_samples_per_ray} samples/ray -> "
          f"{args.crop ** 2 * model.renderer.cfg.num_samples_per_ray / 1e6:.1f}M "
          f"queries per crop")

    for index, batch in enumerate(batches):
        warmup = index == 0
        timer = Timer()
        torch.cuda.synchronize()
        step_start = time.time()

        inputs = batch["input"].to(device)
        B, V = inputs.shape[0], batch["views"].shape[1]
        n_terms = B * V

        with timer("encoder forward"):
            with torch.autocast("cuda", dtype=amp_dtype):
                codes = smodel.encode_sketches(model, inputs)
            codes = codes.float()

        code_grads = torch.zeros_like(codes)
        for b in range(B):
            leaf = codes[b].detach().requires_grad_(True)
            for v in range(V):
                views, masks = batch["views"][b].numpy(), batch["masks"][b].numpy()
                box = data.sample_crop_box(masks[v], args.crop, rng)
                top, left, h, w = box
                with timer("ray setup (cpu)"):
                    rays_o, rays_d = cameras.rays_for_crop(
                        batch["c2ws"][b][v].numpy(), FOVY_DEG, views.shape[1], box
                    )
                    rays_o, rays_d = rays_o.to(device), rays_d.to(device)
                with timer("render forward"):
                    rgb, alpha = strain.render_rays(model, leaf, rays_o, rays_d)
                with timer("gt to gpu"):
                    gt_rgb = torch.from_numpy(
                        np.ascontiguousarray(views[v][top:top + h, left:left + w])
                    ).float().to(device) / 255.0
                    gt_mask = torch.from_numpy(
                        np.ascontiguousarray(masks[v][top:top + h, left:left + w])
                    ).to(device)
                with timer("loss: mse+mask"):
                    bg = cfg.composite_bg
                    gt = torch.where(gt_mask[..., None], gt_rgb,
                                     torch.full_like(gt_rgb, bg))
                    pred = strain.composite_over_gray(rgb, alpha, bg)
                    mse = torch.nn.functional.mse_loss(pred, gt)
                    mask_term = torch.nn.functional.binary_cross_entropy(
                        alpha.clamp(1e-4, 1 - 1e-4), gt_mask.float()
                    )
                with timer("loss: lpips (vgg)"):
                    lpips_term = lpips(
                        pred.clamp(0, 1).permute(2, 0, 1)[None],
                        gt.permute(2, 0, 1)[None],
                    )
                total = mse + cfg.lambda_lpips * lpips_term + cfg.lambda_mask * mask_term
                with timer("per-view backward"):
                    (total / n_terms).backward()
            code_grads[b] = leaf.grad

        with timer("encoder+backbone backward"):
            codes.backward(gradient=code_grads)
        with timer("clip + optimizer step"):
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        step_time = time.time() - step_start
        if warmup:
            print(f"\n(warmup step {step_time:.2f}s, discarded)")
            continue
        timer.report(f"step {index}", step_time)

    print(f"\npeak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
