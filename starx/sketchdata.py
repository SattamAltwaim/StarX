"""The materialized synthetic-sketch dataset: build pass + Dataset class.

Notebook 09 edge-detects a render inside the training step. That is fine for
one Colab run, but it recomputes the same drawings every epoch and hides the
dataset from inspection. This module writes them down instead, once, so that
training on Ibex is an ordinary supervised job: a torch Dataset, a DataLoader
with workers, and a batch dimension.

Storage rides alongside the design shards rather than duplicating them. The
build reads notebook 03's shards and writes a parallel set of "sketch" shards
holding one grayscale PNG per (design, view); both sets extract into the SAME
local cache, so a design ends up with its renders, masks, cameras and now its
sketches side by side, keyed by design id. Measured on the real dataset: 11.7
KiB per 512px sketch, about 1.5 GiB for all 8,622 designs x 16 views.

A sample is a (design, input view) PAIR, not a design - which is the point of
the whole variant. Every view of every design serves as an input in turn, so
the 8,622 designs become ~138k training samples, and unlike the notebook 09
hook, which view you get is now the sampler's business rather than a hash of
the step number.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from starx import shards, synth

SKETCH_PREFIX = "sketch"
PARAMS_NAME = "sketch_params.json"


def sketch_member_name(design_id: str, view: int) -> str:
    return f"{design_id}.edge{view:02d}.png"


def edge_params(cfg) -> dict:
    """The parameters a stored sketch depends on. Written next to the shards
    at build time and checked when the dataset is opened, so a config edit
    cannot silently train on drawings made with different settings."""
    return {
        "sketch_size": int(cfg.sketch_size),
        "edge_blur_sigma": float(cfg.edge_blur_sigma),
        "edge_gain": float(cfg.edge_gain),
        "edge_bg": float(cfg.edge_bg),
    }


def write_params(out_dir, cfg) -> Path:
    path = Path(out_dir) / PARAMS_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(edge_params(cfg), indent=1))
    return path


def read_params(shard_dir):
    path = Path(shard_dir) / PARAMS_NAME
    return json.loads(path.read_text()) if path.exists() else None


def check_params(shard_dir, cfg, strict: bool = True):
    """Compare the stored build parameters against the current config."""
    stored = read_params(shard_dir)
    if stored is None:
        return None
    wanted = edge_params(cfg)
    drift = {k: (stored.get(k), v) for k, v in wanted.items() if stored.get(k) != v}
    if drift and strict:
        raise ValueError(
            f"sketch dataset at {shard_dir} was built with different edge "
            f"parameters (stored, wanted): {drift}. Rebuild it, or set the "
            f"config back to the stored values."
        )
    return drift


def _png_bytes(gray_uint8: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(gray_uint8, mode="L").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def build_sketch_shards(
    design_shard_dir,
    out_dir,
    cfg,
    device: str = "cpu",
    progress=None,
    limit_shards: int = None,
) -> dict:
    """Edge-detect every stored view of every design into parallel shards.

    One sketch shard per design shard, same index, so resuming is per-shard:
    a shard whose .done.json marker exists is skipped, and an interrupted one
    is rebuilt from scratch. Nothing is read into memory beyond one design.
    """
    design_shard_dir, out_dir = Path(design_shard_dir), Path(out_dir)
    design_tars = shards.list_done_shards(design_shard_dir)
    if limit_shards is not None:
        design_tars = design_tars[:limit_shards]
    write_params(out_dir, cfg)

    stats = {"shards_built": 0, "shards_skipped": 0, "designs": 0, "sketches": 0}
    iterator = progress(design_tars) if progress is not None else design_tars
    for design_tar in iterator:
        index = int(design_tar.stem.split("_")[-1])
        target = out_dir / shards.shard_name(index, SKETCH_PREFIX)
        if shards.marker_path(target).exists():
            stats["shards_skipped"] += 1
            continue

        writer = shards.ShardWriter(out_dir, index, prefix=SKETCH_PREFIX)
        for sample in shards.iter_shard(design_tar):
            sketches = synth.sketches_for_views(sample["views"], cfg, device)
            quantized = (
                sketches.mul(255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            )
            writer.add([
                (sketch_member_name(sample["design_id"], v), _png_bytes(quantized[v]))
                for v in range(quantized.shape[0])
            ])
            stats["designs"] += 1
            stats["sketches"] += int(quantized.shape[0])
        writer.close()
        stats["shards_built"] += 1
    return stats


def load_sketch(cache_dir, design_id: str, view: int) -> torch.Tensor:
    """One stored sketch -> (3, S, S) float in [0, 1], the model's input.

    Stored as one grayscale channel because that is all a line drawing is;
    repeated to three here because the stock encoder wants RGB.
    """
    path = Path(cache_dir) / sketch_member_name(design_id, view)
    gray = torch.from_numpy(np.array(Image.open(path), dtype=np.uint8))
    return (gray.float() / 255.0)[None].repeat(3, 1, 1)


class SketchDataset(torch.utils.data.Dataset):
    """(design, input view) pairs over the extracted cache.

    __getitem__ returns everything one supervised step needs and nothing more:
    the input drawing, a few ground-truth views to be scored against, their
    masks, and the cameras - CANONICALIZED so the input view sits at azimuth
    zero, which is the frame TripoSR reconstructs in. Only the views actually
    supervised are read from disk, so a sample costs a handful of small PNGs
    rather than the design's whole rig.

    Supervision views are drawn per __getitem__ from a generator seeded by
    (seed, index, epoch): set_epoch changes them between passes, and a run
    resumed at the same epoch draws the same ones.
    """

    def __init__(
        self,
        cache_dir,
        split: str = None,
        supervision_views: int = 4,
        seed: int = 1337,
        return_all_masks: bool = False,
        require_sketches: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.supervision_views = supervision_views
        self.seed = seed
        self.return_all_masks = return_all_masks
        self.epoch = 0

        self.designs, self.n_views = [], None
        for meta_path in sorted(self.cache_dir.glob("*.meta.json")):
            design_id = meta_path.name.replace(".meta.json", "")
            meta = json.loads(meta_path.read_text())
            if split is not None and meta.get("split") != split:
                continue
            if require_sketches and not (
                self.cache_dir / sketch_member_name(design_id, 0)
            ).exists():
                continue
            self.designs.append((design_id, int(meta["n_views"])))
            self.n_views = min(self.n_views or 10**9, int(meta["n_views"]))

        self.n_views = self.n_views or 0
        if self.designs and len({n for _, n in self.designs}) > 1:
            # ragged rigs would make index arithmetic lie about which view a
            # sample refers to; notebook 03 renders a fixed count, so this is
            # a corrupted-cache signal rather than something to accommodate
            raise ValueError(
                "designs have differing view counts: "
                f"{sorted({n for _, n in self.designs})}"
            )

    def set_epoch(self, epoch: int) -> None:
        """Re-draw supervision views for the next pass (call before each epoch)."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.designs) * self.n_views

    def describe(self) -> dict:
        return {
            "designs": len(self.designs),
            "views_per_design": self.n_views,
            "samples": len(self),
            "supervision_views": self.supervision_views,
        }

    def _meta(self, design_id: str) -> dict:
        return json.loads((self.cache_dir / f"{design_id}.meta.json").read_text())

    def __getitem__(self, index: int) -> dict:
        design_index, input_view = divmod(index, self.n_views)
        design_id, n_views = self.designs[design_index]
        meta = self._meta(design_id)

        rng = np.random.default_rng(
            [self.seed, index, self.epoch]
        )
        k = min(self.supervision_views, n_views)
        view_ids = rng.choice(n_views, size=k, replace=False)

        rgbs, masks = [], []
        for v in view_ids:
            rgba = np.asarray(Image.open(self.cache_dir / f"{design_id}.view{v:02d}.png"))
            rgbs.append(rgba[..., :3])
            masks.append(rgba[..., 3] > 127)

        c2ws = np.load(self.cache_dir / f"{design_id}.cameras.npy")
        c2ws = synth.canonicalize_to_view(
            c2ws, float(meta["view_angles"][input_view][0])
        )

        item = {
            "input": load_sketch(self.cache_dir, design_id, input_view),
            "views": torch.from_numpy(np.stack(rgbs)),          # (k, H, W, 3) uint8
            "masks": torch.from_numpy(np.stack(masks)),         # (k, H, W) bool
            "c2ws": torch.from_numpy(c2ws[view_ids].copy()),    # (k, 4, 4) float32
            "design_id": design_id,
            "input_view": int(input_view),
        }
        if self.return_all_masks:
            # the visual-hull term needs every silhouette, not just the
            # supervised ones - off by default because it is not in the paper
            all_masks = [
                np.asarray(Image.open(self.cache_dir / f"{design_id}.view{v:02d}.png"))[..., 3] > 127
                for v in range(n_views)
            ]
            item["all_masks"] = torch.from_numpy(np.stack(all_masks))
            item["all_c2ws"] = torch.from_numpy(c2ws.copy())
        return item


def collate(batch: list) -> dict:
    """Stack the tensors, keep the identifiers as plain lists."""
    out = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        out[key] = (
            torch.stack(values) if torch.is_tensor(values[0]) else values
        )
    return out
