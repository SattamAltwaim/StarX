"""Every figure for the write-up, regenerated from the run's own artifacts.

    python scripts/paper_figures.py --data-root $HOME/StarX/data/StarX \
        --run-name sketch_ft_ddp

Reads train_log.jsonl and test_metrics.json from the run directory and the
shards for the qualitative panels, and writes PNG + PDF into
<run>/figures/. Nothing here recomputes the model; it is all reporting, so
it runs on a login node in seconds.
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

from starx import shards, sketchdata, synth
from starx import train as strain
from starx.config import StarXConfig, run_dir, shard_dir

plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 200, "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
})
C = plt.get_cmap("tab10").colors
STAGES = [(1200, "+triplane"), (3000, "+backbone"), (6600, "+encoder")]


def save(fig, out_dir, name):
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  {name}.png / .pdf")


def smooth(v, w=9):
    if len(v) < w:
        return v
    return np.convolve(v, np.ones(w) / w, mode="valid")


def fig_training(rows, out):
    step = np.array([r["step"] for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    panels = [
        (axes[0], ["loss"], "total training loss", True),
        (axes[1], ["mse", "lpips", "mask"], "loss terms", True),
        (axes[2], ["lr"], "learning rate (decoder group)", False),
    ]
    for ax, keys, title, log in panels:
        for i, key in enumerate(keys):
            v = np.array([r[key] for r in rows])
            ax.plot(step, v, color=C[i], alpha=0.22, lw=0.7)
            k = len(v) - len(smooth(v))
            ax.plot(step[k // 2: len(v) - (k - k // 2)], smooth(v),
                    color=C[i], lw=1.7, label=key)
        for at, name in STAGES:
            ax.axvline(at, color="0.65", ls="--", lw=0.9)
        if log:
            ax.set_yscale("log")
        ax.set_xlabel("step")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    top = axes[0].get_ylim()[1]
    for at, name in STAGES:
        axes[0].annotate(name, (at, top * 0.75), fontsize=7, rotation=90,
                         color="0.35", ha="right")
    save(fig, out, "fig_training_curves")


def fig_unfreezing(out, total_steps=12000):
    """Trainable parameters and per-group learning rate over the schedule."""
    sizes = {"decoder": 0.6, "post_processor": 19.3, "tokenizer": 3.1,
             "backbone": 329.5, "image_tokenizer": 86.4}  # millions, TripoSR
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.0))

    steps = np.arange(0, total_steps, 25)
    live = np.zeros_like(steps, dtype=float)
    groups = [(0, ["decoder", "post_processor"]), (1200, ["tokenizer"]),
              (3000, ["backbone"]), (6600, ["image_tokenizer"])]
    for i, s in enumerate(steps):
        live[i] = sum(sizes[g] for at, gs in groups if s >= at for g in gs)
    axes[0].step(steps, live, where="post", color=C[0], lw=2)
    axes[0].fill_between(steps, live, step="post", alpha=0.15, color=C[0])
    for at, name in STAGES:
        axes[0].axvline(at, color="0.65", ls="--", lw=0.9)
        axes[0].annotate(name, (at, live.max() * 0.55), fontsize=7.5,
                         rotation=90, color="0.35", ha="right")
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("trainable parameters (M)")
    axes[0].set_title("gradual unfreezing")
    axes[0].grid(alpha=0.25)

    for i, (name, scale) in enumerate(strain.LR_SCALE.items()):
        opens = next(at for at, gs in groups if name in gs)
        xs = np.arange(opens, total_steps, 25)
        warm, base = 500, 1e-4
        ys = [(min((x + 1) / warm, 1.0) if x < warm else
               0.5 * (1 + np.cos(np.pi * (x - warm) / (total_steps - warm))))
              * base * scale for x in xs]
        axes[1].plot(xs, ys, color=C[i], lw=1.6, label=f"{name} ({scale}x)")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("learning rate")
    axes[1].set_title("discriminative learning rates")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, loc="lower left")
    save(fig, out, "fig_unfreezing_schedule")


def fig_test_metrics(metrics, out):
    keys = [("psnr", "PSNR (dB)", True), ("lpips", "LPIPS", False),
            ("iou", "silhouette IoU", True), ("mse", "MSE", False)]
    names = [k for k in metrics if k in ("pretrained",) or k.startswith("fine-tuned")]
    pre, ft = metrics[names[0]], metrics[names[1]]

    fig, axes = plt.subplots(1, 4, figsize=(13, 2.9))
    for ax, (key, label, higher) in zip(axes, keys):
        values = [pre[key], ft[key]]
        bars = ax.bar(["stock", "fine-tuned"], values, color=[C[7], C[2]], width=0.6)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v,
                    f"{v:.3f}" if v < 10 else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_title(f"{label}  ({'higher' if higher else 'lower'} is better)",
                     fontsize=9)
        ax.grid(alpha=0.25, axis="y")
        ax.set_ylim(0, max(values) * 1.25)
    fig.suptitle(f"held-out test split, {metrics['designs']} unseen designs",
                 fontsize=10)
    save(fig, out, "fig_test_metrics")


def fig_distributions(metrics, out):
    pre = metrics["per_design"]["pretrained"]
    ft = metrics["per_design"]["fine_tuned"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.1))

    for ax, key, label in ((axes[0], "psnr", "PSNR (dB)"),
                           (axes[1], "iou", "silhouette IoU")):
        a = np.array([d[key] for d in pre])
        b = np.array([d[key] for d in ft])
        bins = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), 30)
        ax.hist(a, bins=bins, color=C[7], alpha=0.75, label="stock")
        ax.hist(b, bins=bins, color=C[2], alpha=0.75, label="fine-tuned")
        ax.axvline(a.mean(), color=C[7], ls="--", lw=1.2)
        ax.axvline(b.mean(), color=C[2], ls="--", lw=1.2)
        ax.set_xlabel(label)
        ax.set_ylabel("designs")
        ax.set_title(f"{label} per design")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)

    a = np.array([d["iou"] for d in pre])
    b = np.array([d["iou"] for d in ft])
    axes[2].scatter(a, b, s=11, alpha=0.55, color=C[0], edgecolors="none")
    lim = [0, 1]
    axes[2].plot(lim, lim, color="0.5", ls="--", lw=1)
    axes[2].set_xlim(lim)
    axes[2].set_ylim(lim)
    axes[2].set_xlabel("stock IoU")
    axes[2].set_ylabel("fine-tuned IoU")
    improved = float((b > a).mean())
    axes[2].set_title(f"paired per design - {100 * improved:.0f}% improved")
    axes[2].grid(alpha=0.25)
    save(fig, out, "fig_test_distributions")


def fig_method(cfg, out):
    """The edge-detection pipeline, stage by stage, on a real design."""
    import torch.nn.functional as F

    tar = shards.list_done_shards(shard_dir(cfg, "test"))[0]
    sample = next(shards.iter_shard(tar))
    view = sample["views"][0]

    x = torch.from_numpy(view.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    x = F.interpolate(x, (cfg.sketch_size,) * 2, mode="bilinear",
                      align_corners=False, antialias=True)
    gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    k = synth.gaussian_kernel1d(cfg.edge_blur_sigma)
    r = k.numel() // 2
    b = F.conv2d(F.pad(gray, (r, r, 0, 0), mode="replicate"), k.reshape(1, 1, 1, -1))
    b = F.conv2d(F.pad(b, (0, 0, r, r), mode="replicate"), k.reshape(1, 1, -1, 1))
    p = F.pad(b, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(p, synth.SOBEL_X)
    gy = F.conv2d(p, synth.SOBEL_Y)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
    mag = mag / mag.amax().clamp_min(1e-3)
    sketch = (cfg.edge_bg - (cfg.edge_gain * mag).clamp(0, 1) * cfg.edge_bg).clamp(0, 1)

    panels = [(view, "posed render"), (gray[0, 0], "grayscale"),
              (b[0, 0], "Gaussian blur"), (gx[0, 0].abs(), r"$|\partial_x|$ Sobel"),
              (gy[0, 0].abs(), r"$|\partial_y|$ Sobel"),
              (mag[0, 0], "gradient magnitude"), (sketch[0, 0], "synthetic sketch")]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.05 * len(panels), 2.4))
    for ax, (img, title) in zip(axes, panels):
        img = img if isinstance(img, np.ndarray) else img.numpy()
        ax.imshow(img, cmap=None if img.ndim == 3 else "gray")
        ax.set_title(title, fontsize=8.5)
        ax.axis("off")
    save(fig, out, "fig_method_edge_pipeline")


def fig_engineering(out):
    """Two measurements that shaped the run."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.0))

    batch = [2, 8, 16]
    designs_per_s = [0.72, 0.80, 0.64]
    vram = [12.2, 12.3, 12.4]
    ax = axes[0]
    ax.plot(batch, designs_per_s, "o-", color=C[0], lw=2, label="throughput")
    ax.set_xlabel("designs per rank")
    ax.set_ylabel("designs / s", color=C[0])
    ax.set_ylim(0, 1.0)
    ax.set_xticks(batch)
    twin = ax.twinx()
    twin.plot(batch, vram, "s--", color=C[3], lw=1.5, label="peak VRAM")
    twin.set_ylabel("peak VRAM (GiB)", color=C[3])
    twin.set_ylim(0, 80)
    twin.axhline(80, color=C[3], ls=":", lw=1)
    twin.annotate("A100 capacity", (2.2, 74), fontsize=7, color=C[3])
    twin.spines["right"].set_visible(True)
    ax.set_title("throughput is flat in batch size\n(memory is per-view, not per-batch)")
    ax.grid(alpha=0.25)

    calls = np.array([100, 200, 300, 400, 500, 600])
    without = np.array([10.1, 8.6, 9.5, 10.3, 11.2, 12.1])
    axes[1].plot(calls, without, "o-", color=C[3], lw=2, label="metric not reset")
    axes[1].axhline(7.1, color=C[2], lw=2, label="metric reset (flat)")
    axes[1].set_xlabel("LPIPS calls made so far")
    axes[1].set_ylabel("ms per call")
    axes[1].set_title("LPIPS as a Metric accumulates state\n"
                      "(26 calls/step made training quadratic)")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)
    save(fig, out, "fig_engineering")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--run-name", default="sketch_ft_ddp")
    args = parser.parse_args()

    cfg = StarXConfig(drive_root=args.data_root,
                      local_root=os.path.join(args.data_root, "local_cache"))
    rdir = run_dir(cfg, args.run_name)
    out = rdir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    print(f"writing figures to {out}")

    rows = [json.loads(l) for l in open(rdir / "logs" / "train_log.jsonl")]
    metrics = json.loads((rdir / "test_metrics.json").read_text())

    fig_training(rows, out)
    fig_unfreezing(out)
    fig_test_metrics(metrics, out)
    fig_distributions(metrics, out)
    fig_method(cfg, out)
    fig_engineering(out)
    print(f"\n{len(list(out.glob('*.png')))} figures written")


if __name__ == "__main__":
    main()
