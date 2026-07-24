"""Camera builders shared by GT rendering, training rays, and evaluation.

TripoSR's convention (mirrored from tsr/utils.py and unit-tested for exact
agreement): right-handed world with z up; a camera at spherical position
(azimuth, elevation, distance) looks at the origin; the camera-to-world
rotation columns are [right, up, -lookat] - the OpenGL camera convention,
which pyrender shares, so c2w_to_pyrender_pose is a plain copy. Notebook 03
verifies this empirically before any bulk rendering (the alignment gate).

Ray directions use a pinhole model with focal = 0.5 * H / tan(fovy / 2) and
camera-frame directions [(x - cx) / f, -(y - cy) / f, -1], normalized.
"""

from __future__ import annotations

import math
import zlib

import numpy as np


def build_spherical_c2w(
    azimuth_deg: float, elevation_deg: float, distance: float
) -> np.ndarray:
    """(4, 4) float32 camera-to-world matrix in the TripoSR convention."""
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    position = np.array(
        [
            distance * math.cos(elevation) * math.cos(azimuth),
            distance * math.cos(elevation) * math.sin(azimuth),
            distance * math.sin(elevation),
        ]
    )
    up = np.array([0.0, 0.0, 1.0])
    lookat = -position / np.linalg.norm(position)
    right = np.cross(lookat, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, lookat)
    up = up / np.linalg.norm(up)

    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -lookat
    c2w[:3, 3] = position
    return c2w.astype(np.float32)


def _design_rng(design_id: str, seed: int) -> np.random.Generator:
    # crc32 rather than hash(): python's hash is salted per process, and view
    # sampling must be reproducible across sessions
    key = zlib.crc32(f"{design_id}:{seed}".encode())
    return np.random.default_rng(key)


def sample_design_views(design_id: str, n_views: int, elev_range, seed: int = 0):
    """Deterministic per-design cameras: (V, 4, 4) c2ws + (V, 2) azim/elev deg."""
    rng = _design_rng(design_id, seed)
    azimuths = rng.uniform(0.0, 360.0, size=n_views)
    elevations = rng.uniform(elev_range[0], elev_range[1], size=n_views)
    c2ws = np.stack(
        [
            build_spherical_c2w(az, el, distance=_camera_distance())
            for az, el in zip(azimuths, elevations)
        ]
    )
    angles = np.stack([azimuths, elevations], axis=1).astype(np.float32)
    return c2ws, angles


def _camera_distance() -> float:
    from starx.config import CAMERA_DISTANCE

    return CAMERA_DISTANCE


def c2w_to_pyrender_pose(c2w: np.ndarray) -> np.ndarray:
    """TSR c2w -> pyrender camera pose.

    Both use the OpenGL camera convention (camera looks along -Z, +Y up), so
    this is a copy; it exists as the single named conversion point so any
    future convention fix happens in exactly one place.
    """
    return np.asarray(c2w, dtype=np.float64).copy()


def _directions_grid(height: int, width: int, fovy_deg: float) -> np.ndarray:
    """(H, W, 3) unnormalized camera-frame ray directions (pixel centers)."""
    focal = 0.5 * height / math.tan(0.5 * math.radians(fovy_deg))
    xs = np.arange(width, dtype=np.float64) + 0.5
    ys = np.arange(height, dtype=np.float64) + 0.5
    i, j = np.meshgrid(xs, ys)  # each (H, W)
    return np.stack(
        [
            (i - width / 2.0) / focal,
            -(j - height / 2.0) / focal,
            -np.ones_like(i),
        ],
        axis=-1,
    )


def _rays_from_directions(directions: np.ndarray, c2w: np.ndarray):
    rotation = np.asarray(c2w, dtype=np.float64)[:3, :3]
    rays_d = directions @ rotation.T
    rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)
    rays_o = np.broadcast_to(
        np.asarray(c2w, dtype=np.float64)[:3, 3], rays_d.shape
    ).copy()
    return rays_o, rays_d


def rays_full(c2w: np.ndarray, fovy_deg: float, size: int):
    """Full-image rays as float32 torch tensors of shape (size, size, 3)."""
    import torch

    directions = _directions_grid(size, size, fovy_deg)
    rays_o, rays_d = _rays_from_directions(directions, c2w)
    return (
        torch.from_numpy(rays_o.astype(np.float32)),
        torch.from_numpy(rays_d.astype(np.float32)),
    )


def rays_for_crop(c2w: np.ndarray, fovy_deg: float, image_size: int, crop_box):
    """Rays for a (top, left, height, width) window of the full-image grid.

    The window is cut from the full image's pixel grid, so a rendered crop
    aligns exactly with the same crop of a ground-truth image.
    """
    import torch

    top, left, crop_h, crop_w = crop_box
    directions = _directions_grid(image_size, image_size, fovy_deg)
    window = directions[top : top + crop_h, left : left + crop_w]
    rays_o, rays_d = _rays_from_directions(window, c2w)
    return (
        torch.from_numpy(rays_o.astype(np.float32)),
        torch.from_numpy(rays_d.astype(np.float32)),
    )
