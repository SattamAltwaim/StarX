"""
Differentiable SSIM / MS-SSIM for 3D volumetric (voxel) data.

Input layout: (N, C, D, H, W)

Design notes
------------
* The Gaussian window is applied as three separable 1-D convolutions
  (D, then H, then W) instead of one dense k^3 kernel. For k=11 that is
  33 multiply-adds per voxel instead of 1331 -- the difference between
  a usable loss and an OOM on a 256^3 volume.
* Everything is built from conv3d / pointwise ops, so autograd handles
  the backward pass; there is no custom Function to get wrong.
* win_size / sigma / pooling accept per-axis values for anisotropic
  voxel spacing (common in MRI / CT, e.g. 0.5x0.5x3.0 mm).

Used in StarX by starx.train.ssim3d_occupancy_loss, which scores the
predicted triplane occupancy grid against the visual-hull target built
in starx.train.visual_hull_grid (see StarXConfig.lambda_ssim3d).
"""

from __future__ import annotations

from typing import Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

Triple = Union[int, float, Sequence]

__all__ = ["ssim3d", "ms_ssim3d", "SSIM3DLoss", "MSSSIM3DLoss"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _as_triple(v: Triple):
    if isinstance(v, (int, float)):
        return (v, v, v)
    v = tuple(v)
    if len(v) != 3:
        raise ValueError(f"expected a scalar or a length-3 sequence, got {v}")
    return v


def _gaussian_1d(win_size: int, sigma: float, dtype, device) -> torch.Tensor:
    """Normalised 1-D Gaussian of length `win_size`."""
    if win_size % 2 == 0:
        raise ValueError("win_size must be odd")
    coords = torch.arange(win_size, dtype=dtype, device=device)
    coords = coords - (win_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def _blur3d(x: torch.Tensor, kernels, padding: str, pad_mode: str) -> torch.Tensor:
    """Separable 3-D convolution with per-axis 1-D kernels (D, H, W)."""
    c = x.shape[1]
    for axis, k1d in enumerate(kernels):
        k = k1d.numel()
        if k == 1:
            continue
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = k
        w = k1d.reshape(shape).expand(c, 1, shape[2], shape[3], shape[4]).contiguous()
        if padding == "same":
            p = k // 2
            pad = [0] * 6
            base = (2 - axis) * 2          # F.pad order is (W, W, H, H, D, D)
            pad[base] = pad[base + 1] = p
            x = F.pad(x, pad, mode=pad_mode)
        x = F.conv3d(x, w, groups=c)
    return x


def _check_size(x: torch.Tensor, win: Sequence[int]) -> None:
    spatial = x.shape[-3:]
    if any(s < w for s, w in zip(spatial, win)):
        raise ValueError(
            f"volume {tuple(spatial)} is smaller than the window {tuple(win)}; "
            "reduce win_size (per axis if the volume is anisotropic)"
        )


# --------------------------------------------------------------------------- #
# single-scale SSIM
# --------------------------------------------------------------------------- #
def _ssim_maps(
    x, y, data_range, win_size, sigma, K, padding, pad_mode,
):
    win_size = _as_triple(win_size)
    sigma = _as_triple(sigma)
    _check_size(x, win_size)

    kernels = [
        _gaussian_1d(int(w), float(s), x.dtype, x.device)
        for w, s in zip(win_size, sigma)
    ]

    k1, k2 = K
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    # One blur call per statistic. Concatenating along the channel axis lets
    # the five blurs share a single pass over memory.
    c = x.shape[1]
    stacked = torch.cat([x, y, x * x, y * y, x * y], dim=1)
    blurred = _blur3d(stacked, kernels, padding, pad_mode)
    mu_x, mu_y, xx, yy, xy = torch.split(blurred, c, dim=1)

    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    # clamp only the variances: rounding can push them a hair below zero,
    # which would make the cs denominator non-positive.
    sig_x2 = (xx - mu_x2).clamp_min(0.0)
    sig_y2 = (yy - mu_y2).clamp_min(0.0)
    sig_xy = xy - mu_xy                      # covariance may legitimately be < 0

    cs = (2.0 * sig_xy + c2) / (sig_x2 + sig_y2 + c2)
    luminance = (2.0 * mu_xy + c1) / (mu_x2 + mu_y2 + c1)
    return luminance * cs, cs


def _reduce(t: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return t.mean()
    if reduction == "sum":
        return t.sum()
    if reduction == "none":
        return t.flatten(1).mean(dim=1)      # per-sample scalar, shape (N,)
    raise ValueError(f"unknown reduction {reduction!r}")


def ssim3d(
    x: torch.Tensor,
    y: torch.Tensor,
    data_range: float = 1.0,
    win_size: Triple = 11,
    sigma: Triple = 1.5,
    K: Sequence[float] = (0.01, 0.03),
    padding: str = "valid",
    pad_mode: str = "replicate",
    reduction: str = "mean",
) -> torch.Tensor:
    """Structural similarity between two volumes, shape (N, C, D, H, W).

    Parameters
    ----------
    data_range : dynamic range of the data (max - min). Use a *fixed* value
        matching your normalisation (1.0 for [0,1], 2.0 for [-1,1]). Do not
        derive it from the batch itself -- it makes the loss scale-dependent
        on the noise in the prediction.
    win_size, sigma : scalar, or a (D, H, W) triple for anisotropic voxels.
        Pass e.g. win_size=(3, 11, 11), sigma=(0.5, 1.5, 1.5) for thick slices.
    padding : "valid" (crop the border, matches skimage) or "same".
    reduction : "mean" | "sum" | "none" (per-sample, shape (N,)).
    """
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")
    if x.dim() != 5:
        raise ValueError(f"expected (N, C, D, H, W), got {tuple(x.shape)}")
    ssim_map, _ = _ssim_maps(x, y, data_range, win_size, sigma, K, padding, pad_mode)
    return _reduce(ssim_map, reduction)


# --------------------------------------------------------------------------- #
# multi-scale SSIM
# --------------------------------------------------------------------------- #
# Wang et al.'s 5 scales are usually unreachable for volumes (they need
# >= 161 voxels per axis with win=11), so the default is the first 3,
# renormalised. Override `weights` if your volumes are large enough.
_MSSSIM_W5 = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
_MSSSIM_W3 = (0.0710, 0.4524, 0.4766)


def _downsample(x: torch.Tensor, pool: Sequence[int], pad_mode: str) -> torch.Tensor:
    """avg-pool by `pool`, replicate-padding odd axes first."""
    pad = [0] * 6
    for axis, p in enumerate(pool):
        if p > 1 and x.shape[2 + axis] % p != 0:
            pad[(2 - axis) * 2 + 1] = p - (x.shape[2 + axis] % p)
    if any(pad):
        x = F.pad(x, pad, mode=pad_mode)
    return F.avg_pool3d(x, kernel_size=tuple(pool), stride=tuple(pool))


def ms_ssim3d(
    x: torch.Tensor,
    y: torch.Tensor,
    data_range: float = 1.0,
    weights: Sequence[float] = _MSSSIM_W3,
    win_size: Triple = 11,
    sigma: Triple = 1.5,
    K: Sequence[float] = (0.01, 0.03),
    padding: str = "valid",
    pad_mode: str = "replicate",
    pool: Triple = 2,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Multi-scale SSIM for volumes.

    `pool` accepts a triple: use (1, 2, 2) to avoid downsampling a short
    axis (e.g. 24 slices at 3 mm) that would vanish after two levels.
    """
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")
    pool = tuple(int(p) for p in _as_triple(pool))
    levels = len(weights)

    win = _as_triple(win_size)
    need = [w * (p ** (levels - 1)) for w, p in zip(win, pool)]
    if any(s < n for s, n in zip(x.shape[-3:], need)):
        raise ValueError(
            f"volume {tuple(x.shape[-3:])} is too small for {levels} scales: "
            f"needs at least {tuple(need)} voxels. Use fewer weights, a smaller "
            f"win_size, or pool=(1, 2, 2) to spare a short axis."
        )

    w = torch.tensor(weights, dtype=x.dtype, device=x.device)
    w = w / w.sum()

    cs_vals = []
    for i in range(levels):
        ssim_map, cs_map = _ssim_maps(
            x, y, data_range, win_size, sigma, K, padding, pad_mode
        )
        if i < levels - 1:
            cs_vals.append(_reduce(cs_map, "none"))
            x = _downsample(x, pool, pad_mode)
            y = _downsample(y, pool, pad_mode)
        else:
            last_ssim = _reduce(ssim_map, "none")

    # cs^w with negative cs would be NaN, and the gradient of x^w at x=0
    # is infinite -- clamp before the power.
    stack = torch.stack(cs_vals + [last_ssim], dim=0).clamp_min(eps)  # (L, N)
    out = torch.prod(stack ** w.view(-1, 1), dim=0)                   # (N,)

    if reduction == "none":
        return out
    return _reduce(out, reduction)


# --------------------------------------------------------------------------- #
# nn.Module wrappers
# --------------------------------------------------------------------------- #
class SSIM3DLoss(nn.Module):
    """1 - SSIM. Usually worth combining with an L1/L2 term, e.g.
    `loss = 0.84 * ssim_loss(p, t) + 0.16 * F.l1_loss(p, t)`, since SSIM
    alone gives almost no gradient in flat, textureless regions."""

    def __init__(self, data_range=1.0, win_size=11, sigma=1.5,
                 K=(0.01, 0.03), padding="valid", pad_mode="replicate",
                 reduction="mean"):
        super().__init__()
        self.kw = dict(data_range=data_range, win_size=win_size, sigma=sigma,
                       K=K, padding=padding, pad_mode=pad_mode,
                       reduction=reduction)

    def forward(self, x, y):
        return 1.0 - ssim3d(x, y, **self.kw)


class MSSSIM3DLoss(nn.Module):
    def __init__(self, data_range=1.0, weights=_MSSSIM_W3, win_size=11,
                 sigma=1.5, K=(0.01, 0.03), padding="valid",
                 pad_mode="replicate", pool=2, reduction="mean"):
        super().__init__()
        self.kw = dict(data_range=data_range, weights=weights,
                       win_size=win_size, sigma=sigma, K=K, padding=padding,
                       pad_mode=pad_mode, pool=pool, reduction=reduction)

    def forward(self, x, y):
        return 1.0 - ms_ssim3d(x, y, **self.kw)
