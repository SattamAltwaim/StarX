import math

import numpy as np
import torch

from starx import cameras
from starx import train as strain
from starx.config import CAMERA_DISTANCE, SCENE_RADIUS


def test_soft_dice_extremes():
    target = torch.zeros(100)
    target[:50] = 1.0
    perfect = strain.soft_dice_loss(target.clone(), target)
    assert float(perfect) < 1e-4
    disjoint = strain.soft_dice_loss(1.0 - target, target)
    assert float(disjoint) > 0.999
    half = strain.soft_dice_loss(torch.full_like(target, 0.5), target)
    assert 0.2 < float(half) < 0.8


def test_soft_dice_gradient_direction():
    target = torch.zeros(64)
    target[:32] = 1.0
    pred = torch.full((64,), 0.5, requires_grad=True)
    strain.soft_dice_loss(pred, target).backward()
    # increasing predicted occupancy inside the target must lower the loss,
    # increasing it outside must raise it
    assert pred.grad[:32].mean() < 0
    assert pred.grad[32:].mean() > 0


def _masks_for_box(half_extent: float, n_views: int, size: int = 64):
    """Analytic silhouettes of an axis-aligned cube via corner projection."""
    focal = 0.5 * size / math.tan(0.5 * math.radians(40.0))
    corners = np.array(
        [[sx * half_extent, sy * half_extent, sz * half_extent]
         for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    )
    masks, c2ws = [], []
    for k in range(n_views):
        c2w = cameras.build_spherical_c2w(360.0 * k / n_views, 15.0, CAMERA_DISTANCE)
        cam = (corners - c2w[:3, 3]) @ c2w[:3, :3]
        z = -cam[:, 2]
        u = size / 2 + focal * cam[:, 0] / z
        w = size / 2 - focal * cam[:, 1] / z
        mask = np.zeros((size, size), dtype=bool)
        u0, u1 = int(np.floor(u.min())), int(np.ceil(u.max()))
        w0, w1 = int(np.floor(w.min())), int(np.ceil(w.max()))
        mask[max(w0, 0) : w1 + 1, max(u0, 0) : u1 + 1] = True
        masks.append(mask)
        c2ws.append(c2w)
    return np.stack(masks), np.stack(c2ws)


def test_visual_hull_full_and_empty():
    size, res = 64, 24
    c2ws = np.stack(
        [cameras.build_spherical_c2w(a, 15.0, CAMERA_DISTANCE) for a in (0.0, 90.0)]
    )
    full = np.ones((2, size, size), dtype=bool)
    hull, observed = strain.visual_hull_grid(full, c2ws, res, size, "cpu")
    assert observed.any()
    assert hull[observed].all()  # all-covering silhouettes carve nothing seen
    empty = np.zeros((2, size, size), dtype=bool)
    hull_empty, observed_empty = strain.visual_hull_grid(empty, c2ws, res, size, "cpu")
    assert not hull_empty[observed_empty].any()  # everything seen is carved


def test_visual_hull_contains_cube_and_carves_far_space():
    half = 0.3
    masks, c2ws = _masks_for_box(half, n_views=8)
    res = 32
    hull, observed = strain.visual_hull_grid(masks, c2ws, res, masks.shape[1], "cpu")
    hull_3d = hull.reshape(res, res, res).numpy()
    observed_3d = observed.reshape(res, res, res).numpy()
    supervised = hull_3d & observed_3d
    axis = np.linspace(-SCENE_RADIUS, SCENE_RADIUS, res)
    inside = (np.abs(axis) <= half * 0.8)
    # every voxel well inside the cube is observed and survives the carving
    assert supervised[np.ix_(inside, inside, inside)].all()
    # scene-box corners are either carved or never observed - not "occupied"
    assert not supervised[0, 0, 0] and not supervised[-1, -1, -1]
    # over the observed region, the hull is a superset of the cube but far
    # smaller than the whole box
    cube_frac = (2 * half) ** 3 / (2 * SCENE_RADIUS) ** 3
    hull_frac_observed = hull_3d[observed_3d].mean()
    assert cube_frac * 0.8 <= supervised.mean()
    assert hull_frac_observed < 0.6
