"""Score a fine-tuned run on the held-out test split, against stock TripoSR.

The training loss says the run went down; it does not say the model got
better at the task, and it says nothing about designs it never saw. This
renders full frames from held-out designs and compares the fine-tuned
checkpoint with the pretrained one it started from, on identical inputs.

    python scripts/eval_sketch.py --data-root $HOME/StarX/data/StarX \
        --run-name sketch_ft_ddp --designs 200

Two cameras per design: the INPUT view (which the model was conditioned on)
and a quarter turn away (which it was not). The turned view is the honest
one - a flat billboard facing the input camera scores well on the first and
falls apart on the second.
"""

import argparse
import json
import os
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from starx import cameras, checkpoint, sketchdata
from starx import model as smodel
from starx import train as strain
from starx.config import CAMERA_DISTANCE, FOVY_DEG, StarXConfig, run_dir


def full_frame_metrics(model, code, item, view, c2w, lpips, cfg, device):
    size = item["views"].shape[1]
    rays_o, rays_d = cameras.rays_full(c2w, FOVY_DEG, size)
    rgb, alpha = strain.render_rays(model, code, rays_o.to(device), rays_d.to(device))
    pred = strain.composite_over_gray(rgb, alpha, cfg.composite_bg).clamp(0, 1)
    gt_rgb = item["views"][view].to(device).float() / 255.0
    gt_mask = item["masks"][view].to(device)
    gt = torch.where(gt_mask[..., None], gt_rgb,
                     torch.full_like(gt_rgb, cfg.composite_bg))
    mse = torch.nn.functional.mse_loss(pred, gt)
    value = lpips(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])
    lpips.reset()
    # silhouette IoU: the geometry question, independent of colour
    pred_mask = alpha > 0.5
    inter = (pred_mask & gt_mask).sum().float()
    union = (pred_mask | gt_mask).sum().float()
    return {
        "mse": float(mse),
        "psnr": float(10 * torch.log10(1.0 / mse.clamp_min(1e-10))),
        "lpips": float(value),
        "iou": float(inter / union.clamp_min(1)),
    }, pred.cpu().numpy()


def evaluate(model, dataset, indices, lpips, cfg, device, novel_c2w, label):
    model.eval()
    model.renderer.set_chunk_size(cfg.eval_chunk)
    totals = {}
    frames = []
    with torch.no_grad():
        for n, index in enumerate(indices):
            item = dataset[index]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                code = smodel.encode_sketches(model, item["input"][None].to(device))[0]
            code = code.float()
            at_input, pred_in = full_frame_metrics(
                model, code, item, 0, item["c2ws"][0].numpy(), lpips, cfg, device
            )
            # the turned camera has no ground truth to compare against, so
            # only its render is kept, for the eye
            rays_o, rays_d = cameras.rays_full(novel_c2w, FOVY_DEG,
                                               item["views"].shape[1])
            rgb, alpha = strain.render_rays(
                model, code, rays_o.to(device), rays_d.to(device)
            )
            turned = strain.composite_over_gray(
                rgb, alpha, cfg.composite_bg
            ).clamp(0, 1).cpu().numpy()
            for k, v in at_input.items():
                totals[k] = totals.get(k, 0.0) + v
            if n < 4:
                frames.append((item, pred_in, turned))
            if (n + 1) % 25 == 0:
                print(f"  {label}: {n + 1}/{len(indices)}", flush=True)
    model.renderer.set_chunk_size(0)
    return {k: v / len(indices) for k, v in totals.items()}, frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--run-name", default="sketch_ft_ddp")
    parser.add_argument("--designs", type=int, default=200)
    parser.add_argument("--supervision-views", type=int, default=2)
    parser.add_argument("--triposr-dir",
                        default=str(REPO_DIR / "third_party" / "TripoSR"))
    args = parser.parse_args()

    device = "cuda:0"
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg = StarXConfig(
        drive_root=args.data_root,
        local_root=os.path.join(args.data_root, "local_cache"),
        composite_bg=0.5, lambda_occ=0.0,
    )
    cache = Path(cfg.local_root) / "test" / "cache"
    dataset = sketchdata.SketchDataset(
        cache, supervision_views=args.supervision_views, seed=cfg.seed, cfg=cfg
    )
    n = min(args.designs, len(dataset.designs))
    indices = [i * dataset.n_views for i in range(n)]
    print(f"held-out test: {n} of {len(dataset.designs)} designs, "
          f"sketches={dataset.sketches}")

    from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True)
    lpips = lpips.to(device).requires_grad_(False).eval()
    novel = cameras.build_spherical_c2w(90.0, 20.0, CAMERA_DISTANCE)

    rdir = run_dir(cfg, args.run_name)
    found = checkpoint.find_latest(rdir)
    if found is None:
        raise SystemExit(f"no checkpoint under {rdir}")
    ckpt_path, step = found
    print(f"checkpoint: {ckpt_path.name} (step {step})\n")

    results = {}
    galleries = {}

    model = smodel.load_pretrained_tsr(args.triposr_dir, device=device)
    model.renderer.set_chunk_size(0)
    results["pretrained"], galleries["pretrained"] = evaluate(
        model, dataset, indices, lpips, cfg, device, novel, "stock"
    )

    strain.apply_unfreeze_stage(model, 10**9, 10**9)  # every key the ckpt holds
    state = checkpoint.load_checkpoint(ckpt_path)
    smodel.load_trainable_state_dict(model, state["model"])
    results[f"fine-tuned@{step}"], galleries["fine-tuned"] = evaluate(
        model, dataset, indices, lpips, cfg, device, novel, "fine-tuned"
    )

    print(f"\n{'':16s} {'PSNR':>8s} {'MSE':>9s} {'LPIPS':>8s} {'mask IoU':>9s}")
    for name, r in results.items():
        print(f"{name:16s} {r['psnr']:>8.2f} {r['mse']:>9.4f} "
              f"{r['lpips']:>8.4f} {r['iou']:>9.3f}")
    a, b = list(results.values())
    print(f"\n{'change':16s} {b['psnr'] - a['psnr']:>+8.2f} "
          f"{b['mse'] - a['mse']:>+9.4f} {b['lpips'] - a['lpips']:>+8.4f} "
          f"{b['iou'] - a['iou']:>+9.3f}")
    print("(PSNR and IoU up is better; MSE and LPIPS down is better)")

    out = rdir / "test_metrics.json"
    out.write_text(json.dumps({"step": step, "designs": n, **results}, indent=1))
    print(f"\nwrote {out}")

    rows = galleries["fine-tuned"]
    fig, axes = plt.subplots(len(rows), 5, figsize=(15.5, 3.1 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, ((item, pred_in, turned), (_, pre_in, pre_turn)) in enumerate(
        zip(rows, galleries["pretrained"])
    ):
        gt = item["views"][0].numpy().astype(np.float32) / 255.0
        gt = np.where(item["masks"][0].numpy()[..., None], gt, cfg.composite_bg)
        for c, img in enumerate([item["input"][0].numpy(), gt, pre_in,
                                 pred_in, turned]):
            axes[r, c].imshow(img, cmap="gray" if img.ndim == 2 else None,
                              vmin=0, vmax=1)
        axes[r, 0].set_ylabel(item["design_id"][:11], fontsize=7)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    for c, t in enumerate(["sketch in", "ground truth", "stock TripoSR",
                           f"fine-tuned @{step}", "fine-tuned, turned 90"]):
        axes[0, c].set_title(t, fontsize=10)
    fig.tight_layout()
    path = rdir / "test_gallery.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
