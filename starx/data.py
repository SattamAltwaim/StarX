"""Training dataset over the unpacked local shard cache + step-seeded sampling.

Sampling is i.i.d. and seeded by the global step, so resuming a run at step
N reproduces exactly the draws a never-interrupted run would have made - no
dataloader or epoch state needs checkpointing.
"""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

from starx.rasterize import stack_to_tensor, strip_to_stack


class DesignDataset:
    """Map-style dataset over a cache directory of extracted shard members."""

    def __init__(self, cache_dir, split: str = None):
        self.cache_dir = Path(cache_dir)
        self.ids = []
        for meta_path in sorted(self.cache_dir.glob("*.meta.json")):
            if split is not None:
                meta = json.loads(meta_path.read_text())
                if meta.get("split") != split:
                    continue
            self.ids.append(meta_path.name.replace(".meta.json", ""))

    def __len__(self) -> int:
        return len(self.ids)

    def meta(self, index: int) -> dict:
        design_id = self.ids[index]
        return json.loads((self.cache_dir / f"{design_id}.meta.json").read_text())

    def __getitem__(self, index: int) -> dict:
        design_id = self.ids[index]
        meta = self.meta(index)
        strip = np.asarray(Image.open(self.cache_dir / f"{design_id}.sketch.png"))
        stack = strip_to_stack(strip, meta["n_channels"])
        views, masks = [], []
        for v in range(meta["n_views"]):
            rgba = np.asarray(
                Image.open(self.cache_dir / f"{design_id}.view{v:02d}.png")
            )
            views.append(rgba[..., :3])
            masks.append(rgba[..., 3] > 127)
        c2ws = np.load(self.cache_dir / f"{design_id}.cameras.npy")
        return {
            "design_id": design_id,
            "sketch": stack_to_tensor(stack),
            "stack_uint8": stack,
            "views": np.stack(views),
            "masks": np.stack(masks),
            "c2ws": c2ws,
            "meta": meta,
        }


def draw_design_indices(step: int, n: int, dataset_len: int, seed: int) -> list:
    """i.i.d. design draw for one optimizer step, seeded by the step number."""
    rng = np.random.default_rng(seed * 1_000_003 + step)
    return rng.integers(0, dataset_len, size=n).tolist()


def choose_views(step: int, design_id: str, n_available: int, k: int, seed: int):
    """Deterministic without-replacement view choice for (step, design)."""
    key = zlib.crc32(f"{design_id}:{step}:{seed}".encode())
    rng = np.random.default_rng(key)
    k = min(k, n_available)
    return rng.choice(n_available, size=k, replace=False).tolist()


def sample_crop_box(
    mask: np.ndarray, crop: int, rng: np.random.Generator, foreground_p: float = 0.8
):
    """Foreground-biased (top, left, h, w) crop window inside a GT view.

    With probability foreground_p the window centers on a random mask pixel
    (clamped inside the image); otherwise it is uniform. Falls back to
    uniform when the mask is empty.
    """
    size = mask.shape[0]
    if crop >= size:
        return 0, 0, size, size
    ys, xs = np.nonzero(mask)
    if len(ys) > 0 and rng.random() < foreground_p:
        pick = rng.integers(len(ys))
        top = int(np.clip(ys[pick] - crop // 2, 0, size - crop))
        left = int(np.clip(xs[pick] - crop // 2, 0, size - crop))
    else:
        top = int(rng.integers(0, size - crop + 1))
        left = int(rng.integers(0, size - crop + 1))
    return top, left, crop, crop
