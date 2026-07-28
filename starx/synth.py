"""Synthetic sketches from posed renders - the no-surgery variant.

Instead of feeding the design's native CAD sketches to a modified model,
this path renders-to-sketch: take one of the sample's stored posed views,
run convolutional edge detection (Gaussian blur, then Sobel gradients),
and style the magnitude as a line drawing - dark strokes on a light page.
The result is an ordinary 3-channel image, so the STOCK TripoSR consumes
it with no input surgery, fine-tuned with the same render-loss recipe.

The key property: the input is a picture of the object from a specific
viewpoint, exactly like TripoSR's pretraining photos. The model
reconstructs in its input view's frame, so before supervising we rotate
the world about z so the input camera sits at azimuth zero - the images
themselves never change, only how their cameras are expressed.
"""

from __future__ import annotations

import math
import zlib

import numpy as np
import torch
import torch.nn.functional as F

SOBEL_X = torch.tensor(
    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
).reshape(1, 1, 3, 3) / 4.0
SOBEL_Y = SOBEL_X.transpose(-1, -2).contiguous()


def gaussian_kernel1d(sigma: float) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    xs = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (xs / sigma) ** 2)
    return kernel / kernel.sum()


def sobel_sketch(
    view_rgb_uint8: np.ndarray,
    out_size: int = 512,
    blur_sigma: float = 1.2,
    gain: float = 3.0,
    bg: float = 1.0,
) -> torch.Tensor:
    """One posed render (H, W, 3) uint8 -> a line-drawing (3, S, S) float.

    Pipeline: upscale, grayscale, Gaussian blur (separable conv), Sobel
    gradient magnitude (fixed conv kernels), per-image normalization with a
    contrast gain, then invert onto a light background. Deterministic.

    Every convolution pads by edge replication, not zeros: zero padding
    would fabricate a brightness step at the image border, drawing a frame
    around every sketch and - worse - inflating the per-image maximum that
    the magnitude is normalized by, which washes out the real edges.
    """
    x = torch.from_numpy(view_rgb_uint8.astype(np.float32) / 255.0)
    x = x.permute(2, 0, 1)[None]  # (1, 3, H, W)
    x = F.interpolate(
        x, (out_size, out_size), mode="bilinear", align_corners=False, antialias=True
    )
    gray = (
        0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
    )  # (1, 1, S, S)

    kernel = gaussian_kernel1d(blur_sigma)
    radius = kernel.numel() // 2
    gray = F.conv2d(
        F.pad(gray, (radius, radius, 0, 0), mode="replicate"),
        kernel.reshape(1, 1, 1, -1),
    )
    gray = F.conv2d(
        F.pad(gray, (0, 0, radius, radius), mode="replicate"),
        kernel.reshape(1, 1, -1, 1),
    )

    padded = F.pad(gray, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(padded, SOBEL_X)
    gy = F.conv2d(padded, SOBEL_Y)
    magnitude = torch.sqrt(gx * gx + gy * gy + 1e-12)

    # normalize by the strongest edge, with a floor far below any real one:
    # a featureless view (object out of frame, mesh that failed to render)
    # would otherwise divide its own numerical noise by itself and come out
    # solid black instead of blank.
    magnitude = magnitude / magnitude.amax().clamp_min(1e-3)
    edges = (gain * magnitude).clamp(0.0, 1.0)
    sketch = (bg - edges * bg).clamp(0.0, 1.0)[0, 0]  # dark lines on bg
    return sketch[None].repeat(3, 1, 1)  # (3, S, S)


def rotation_z(azimuth_deg: float) -> np.ndarray:
    """(4, 4) rotation about the world z axis."""
    a = math.radians(azimuth_deg)
    ca, sa = math.cos(a), math.sin(a)
    rot = np.eye(4, dtype=np.float32)
    rot[0, 0], rot[0, 1] = ca, -sa
    rot[1, 0], rot[1, 1] = sa, ca
    return rot


def canonicalize_to_view(c2ws: np.ndarray, input_azimuth_deg: float) -> np.ndarray:
    """Re-express all cameras in a world rotated so the input camera sits at
    azimuth zero (its elevation is untouched). Rotating the world changes no
    image - only the matrices - and matches TripoSR's pretraining setup,
    where the object faces the input camera."""
    rot = rotation_z(-input_azimuth_deg)
    return np.einsum("ij,vjk->vik", rot, np.asarray(c2ws, dtype=np.float32))


def pick_input_view(design_id: str, step: int, n_views: int, seed: int) -> int:
    """Deterministic input-view choice per (design, step) - every view of a
    design eventually serves as the input, a free 16x data augmentation."""
    key = zlib.crc32(f"{design_id}:{step}:input:{seed}".encode())
    return int(key % n_views)


def make_input_fn(cfg, device: str):
    """The train_step hook for the synthetic-edge variant: returns the edge
    sketch of one stored view as a (1, 3, S, S) tensor plus all supervision
    cameras canonicalized to that view's azimuth."""

    def input_fn(item: dict, step: int):
        v_in = pick_input_view(
            item["design_id"], step, item["meta"]["n_views"], cfg.seed
        )
        sketch = sobel_sketch(
            item["views"][v_in],
            out_size=cfg.sketch_size,
            blur_sigma=cfg.edge_blur_sigma,
            gain=cfg.edge_gain,
            bg=cfg.edge_bg,
        )
        azimuth = float(item["meta"]["view_angles"][v_in][0])
        c2ws = canonicalize_to_view(item["c2ws"], azimuth)
        return sketch[None].to(device), c2ws

    return input_fn
