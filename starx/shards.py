"""Tar-shard storage for preprocessed designs.

Google Drive is pathologically slow with many small files, so every design
is packed into tar shards of a few hundred designs. A shard only counts as
existing once its .done.json marker is on disk - an interrupted build leaves
no marker and the shard is rebuilt from scratch on the next run.

One design = one sample = four kinds of tar members:
    {id}.sketch.png    uint8 grayscale strip (H, C*W) - the channel stack
    {id}.view{v}.png   RGBA ground-truth render (RGB shaded, A = mask)
    {id}.cameras.npy   float32 (V, 4, 4) camera-to-world matrices
    {id}.meta.json     rasterization + rendering metadata
"""

from __future__ import annotations

import io
import json
import os
import shutil
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image

from starx.rasterize import stack_to_strip, strip_to_stack


def _png_bytes(array: np.ndarray, mode: str) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(array, mode=mode).save(buf, format="PNG")
    return buf.getvalue()


def _npy_bytes(array: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, array)
    return buf.getvalue()


def encode_sample(design_id, stack, views_rgb, masks, c2ws, meta) -> list:
    """One design to its tar members as (name, bytes) pairs."""
    meta = dict(meta)
    meta["n_channels"] = int(stack.shape[0])
    meta["n_views"] = int(views_rgb.shape[0])
    members = [
        (f"{design_id}.sketch.png", _png_bytes(stack_to_strip(stack), "L")),
        (f"{design_id}.cameras.npy", _npy_bytes(np.asarray(c2ws, dtype=np.float32))),
        (
            f"{design_id}.meta.json",
            json.dumps(meta, indent=1).encode("utf-8"),
        ),
    ]
    for v in range(views_rgb.shape[0]):
        alpha = (np.asarray(masks[v], dtype=np.uint8) * 255)[..., None]
        rgba = np.concatenate([views_rgb[v], alpha], axis=-1)
        members.append((f"{design_id}.view{v:02d}.png", _png_bytes(rgba, "RGBA")))
    return members


def decode_sample(members: dict) -> dict:
    """Inverse of encode_sample: {member name: bytes} -> arrays + meta."""
    design_id = next(iter(members)).split(".", 1)[0]
    meta = json.loads(members[f"{design_id}.meta.json"])
    strip = np.asarray(Image.open(io.BytesIO(members[f"{design_id}.sketch.png"])))
    stack = strip_to_stack(strip, meta["n_channels"])
    c2ws = np.load(io.BytesIO(members[f"{design_id}.cameras.npy"]))
    views, masks = [], []
    for v in range(meta["n_views"]):
        rgba = np.asarray(
            Image.open(io.BytesIO(members[f"{design_id}.view{v:02d}.png"]))
        )
        views.append(rgba[..., :3])
        masks.append(rgba[..., 3] > 127)
    return {
        "design_id": design_id,
        "stack": stack,
        "views": np.stack(views),
        "masks": np.stack(masks),
        "c2ws": c2ws,
        "meta": meta,
    }


def shard_name(index: int) -> str:
    return f"shard_{index:05d}.tar"


def marker_path(tar_path: Path) -> Path:
    return Path(tar_path).with_suffix("").with_suffix(".done.json")


class ShardWriter:
    """Writes one shard: assemble the tar in a temp file, move it into place
    atomically, then write the .done.json marker last."""

    def __init__(self, out_dir, shard_index: int):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.final_path = self.out_dir / shard_name(shard_index)
        self.tmp_path = self.out_dir / (self.final_path.name + ".tmp")
        self._tar = tarfile.open(self.tmp_path, "w")
        self.design_ids: list = []

    def add(self, members: list) -> None:
        design_id = members[0][0].split(".", 1)[0]
        for name, data in members:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            self._tar.addfile(info, io.BytesIO(data))
        self.design_ids.append(design_id)

    def close(self) -> Path:
        self._tar.close()
        os.replace(self.tmp_path, self.final_path)
        marker = marker_path(self.final_path)
        tmp_marker = marker.with_suffix(".tmp")
        tmp_marker.write_text(
            json.dumps({"n_samples": len(self.design_ids), "ids": self.design_ids})
        )
        os.replace(tmp_marker, marker)
        return self.final_path


def iter_shard(tar_path):
    """Yield decoded samples from one shard, grouped by design id."""
    with tarfile.open(tar_path, "r") as tar:
        groups: dict = {}
        order: list = []
        for member in tar.getmembers():
            design_id = member.name.split(".", 1)[0]
            if design_id not in groups:
                groups[design_id] = {}
                order.append(design_id)
            groups[design_id][member.name] = tar.extractfile(member).read()
        for design_id in order:
            yield decode_sample(groups[design_id])


def list_done_shards(shard_dir) -> list:
    """Shards whose done-marker exists, sorted by name."""
    shard_dir = Path(shard_dir)
    if not shard_dir.exists():
        return []
    return sorted(
        p for p in shard_dir.glob("shard_*.tar") if marker_path(p).exists()
    )


def prepare_local(remote_dir, local_dir, progress=None) -> list:
    """Copy done shards to fast local disk and extract them once.

    Returns the shard names newly extracted this call; already-extracted
    shards (sentinel file present) are skipped, so re-running is cheap.
    """
    local_dir = Path(local_dir)
    cache = local_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    fresh = []
    shard_paths = list_done_shards(remote_dir)
    iterator = progress(shard_paths) if progress is not None else shard_paths
    for remote_tar in iterator:
        sentinel = cache / f".extracted_{remote_tar.stem}"
        if sentinel.exists():
            continue
        local_tar = local_dir / remote_tar.name
        shutil.copyfile(remote_tar, local_tar)
        with tarfile.open(local_tar, "r") as tar:
            tar.extractall(cache, filter="data")
        local_tar.unlink()
        sentinel.touch()
        fresh.append(remote_tar.name)
    return fresh
