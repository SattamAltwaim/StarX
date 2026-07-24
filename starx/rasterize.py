"""Rasterization of parametric sketches into channel-stacked line drawings.

Each sketch of a design becomes one uint8 grayscale channel: dark strokes on
a mid-gray background (matching TripoSR's gray-composite pretraining
convention). Channels are stacked in timeline order, blank-padded up to a
fixed channel count, and stored as a horizontal strip PNG inside shards.

Anti-aliasing comes from drawing at supersampled resolution with PIL and
downsampling with LANCZOS; stroke_width is specified in supersampled pixels.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from starx.config import StarXConfig
from starx.fusion import Design, sketch_bbox, sketch_polylines


def rasterize_sketch(
    polylines: list,
    center_xy,
    px_per_cm: float,
    size: int,
    stroke_width: int,
    stroke: int,
    bg: int,
    supersample: int,
) -> np.ndarray:
    """One sketch's polylines to a uint8 (size, size) image.

    center_xy (sketch cm coordinates) lands on the image center; y is flipped
    so that +y in the sketch points up in the image.
    """
    hi_size = size * supersample
    img = Image.new("L", (hi_size, hi_size), color=bg)
    draw = ImageDraw.Draw(img)
    cx, cy = float(center_xy[0]), float(center_xy[1])
    scale = px_per_cm * supersample
    half = hi_size / 2.0
    for poly in polylines:
        xs = half + (poly[:, 0] - cx) * scale
        ys = half - (poly[:, 1] - cy) * scale
        pts = list(zip(xs.tolist(), ys.tolist()))
        if len(pts) >= 2:
            draw.line(pts, fill=stroke, width=stroke_width, joint="curve")
    if supersample != 1:
        img = img.resize((size, size), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def rasterize_design(design: Design, cfg: StarXConfig):
    """Full (C, H, W) uint8 stack for a design, plus a meta dict.

    Shared normalization (default): one px/cm factor for the whole design,
    chosen so the largest sketch fills the frame minus margin - relative
    sketch sizes are preserved. Each channel is centered on its own sketch.
    """
    C = cfg.max_sketch_channels
    if len(design.sketches) > C and not cfg.truncate_extra_sketches:
        raise ValueError(
            f"{design.design_id}: {len(design.sketches)} sketches exceed "
            f"max_sketch_channels={C} and truncation is disabled"
        )

    used = design.sketches[:C]
    warnings: list = []
    per_sketch = []
    for sketch in used:
        polys = sketch_polylines(sketch, cfg.include_construction, warnings)
        bbox = sketch_bbox(polys)
        per_sketch.append((polys, bbox))

    target = cfg.sketch_size * (1.0 - 2.0 * cfg.margin)
    extents = []
    for _, bbox in per_sketch:
        if bbox is not None:
            lo, hi = bbox
            extents.append(max(float(hi[0] - lo[0]), float(hi[1] - lo[1])))

    shared_scale = None
    if cfg.normalization_mode == "shared":
        max_extent = max(extents) if extents else 0.0
        shared_scale = target / max_extent if max_extent > 0 else 1.0

    stack = np.full(
        (C, cfg.sketch_size, cfg.sketch_size), cfg.bg_value, dtype=np.uint8
    )
    scales = []
    for i, (polys, bbox) in enumerate(per_sketch):
        if bbox is None:
            scales.append(None)
            continue
        lo, hi = bbox
        extent = max(float(hi[0] - lo[0]), float(hi[1] - lo[1]))
        if cfg.normalization_mode == "shared":
            scale = shared_scale
        else:
            scale = target / extent if extent > 0 else 1.0
        scales.append(scale)
        stack[i] = rasterize_sketch(
            polys,
            center_xy=(lo + hi) / 2.0,
            px_per_cm=scale,
            size=cfg.sketch_size,
            stroke_width=cfg.stroke_width,
            stroke=cfg.stroke_value,
            bg=cfg.bg_value,
            supersample=cfg.supersample,
        )

    blank = [bool(np.all(stack[i] == cfg.bg_value)) for i in range(C)]
    meta = {
        "design_id": design.design_id,
        "n_sketches_total": len(design.sketches),
        "n_channels_used": len(used),
        "truncated": len(design.sketches) > C,
        "blank_channels": blank,
        "all_blank": all(blank),
        "px_per_cm": shared_scale if cfg.normalization_mode == "shared" else scales,
        "normalization_mode": cfg.normalization_mode,
        "n_curve_warnings": len(warnings),
    }
    return stack, meta


def stack_to_strip(stack: np.ndarray) -> np.ndarray:
    """(C, H, W) -> (H, C*W) uint8 strip, human-viewable as one PNG."""
    return np.concatenate(list(stack), axis=1)


def strip_to_stack(strip: np.ndarray, n_channels: int) -> np.ndarray:
    """Inverse of stack_to_strip."""
    return np.stack(np.split(strip, n_channels, axis=1))


def stack_to_tensor(stack_uint8: np.ndarray):
    """uint8 (C, H, W) stack -> float32 torch tensor in [0, 1].

    Deliberately NOT mean/std normalized: TripoSR's image tokenizer applies
    its own normalization buffers internally (extended to CHANNEL_MEAN /
    CHANNEL_STD per channel during model surgery), so the model consumes raw
    [0, 1] images exactly like the pretrained pipeline did.
    """
    import torch

    return torch.from_numpy(np.ascontiguousarray(stack_uint8)).float() / 255.0
