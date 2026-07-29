import math

import numpy as np
import pytest
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


def test_paper_optimizer_decays_only_matrix_parameters():
    """The paper's AdamW leaves biases and normalization scales undecayed."""
    import torch.nn as nn

    from starx.config import StarXConfig

    model = nn.Sequential(
        nn.Linear(8, 8), nn.LayerNorm(8), nn.Conv2d(3, 4, 3),
    )
    cfg = StarXConfig(lr=4e-4, warmup_steps=10, weight_decay=0.05)
    optimizer, _ = strain.build_paper_optimizer(model, cfg, total_steps=100)

    decay, no_decay = optimizer.param_groups
    assert decay["weight_decay"] == 0.05 and no_decay["weight_decay"] == 0.0
    assert optimizer.defaults["betas"] == (0.9, 0.95)
    # every decayed tensor is a matrix/kernel; every undecayed one is 1-D
    assert all(p.ndim > 1 for p in decay["params"])
    assert all(p.ndim <= 1 for p in no_decay["params"])
    assert len(decay["params"]) == 2  # Linear.weight, Conv2d.weight
    assert len(no_decay["params"]) == 4  # two biases + LayerNorm weight/bias


def test_paper_optimizer_skips_frozen_parameters():
    import torch.nn as nn

    from starx.config import StarXConfig

    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    model[1].requires_grad_(False)
    optimizer, _ = strain.build_paper_optimizer(
        model, StarXConfig(warmup_steps=1), total_steps=10
    )
    counted = sum(len(g["params"]) for g in optimizer.param_groups)
    assert counted == 2  # only the trainable Linear's weight and bias


def test_paper_schedule_warms_up_then_cosines_to_zero():
    import torch.nn as nn

    from starx.config import StarXConfig

    model = nn.Linear(4, 4)
    cfg = StarXConfig(lr=4e-4, warmup_steps=100)
    optimizer, scheduler = strain.build_paper_optimizer(model, cfg, total_steps=1000)

    seen = []
    for _ in range(1000):
        seen.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()

    assert seen[0] < cfg.lr / 50  # starts near zero
    assert abs(seen[99] - cfg.lr) < 1e-9  # full rate at the end of warmup
    assert max(seen) == pytest.approx(cfg.lr)
    assert seen[-1] < cfg.lr / 100  # anneals to zero, not to a floor
    warm = seen[:100]
    assert warm == sorted(warm)  # monotone ramp


def test_pick_amp_falls_back_to_scaled_fp16_without_native_bf16(monkeypatch):
    """A Turing T4 reports bf16 "supported" only through emulation, which is
    slow - the fp16 path exists for exactly that card and must engage."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=0: (7, 5))
    dtype, scale = strain.pick_amp("cuda")
    assert dtype is torch.float16 and scale > 1.0

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=0: (8, 0))
    dtype, scale = strain.pick_amp("cuda")
    assert dtype is torch.bfloat16 and scale == 1.0


def test_pick_amp_handles_indexed_devices(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=0: (8, 6))
    assert strain.pick_amp("cuda:1")[0] is torch.bfloat16


def test_pick_amp_on_cpu():
    assert strain.pick_amp("cpu")[0] is torch.float16
