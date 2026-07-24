import numpy as np
import pytest
import trimesh

from starx import eval as steval

SPHERE = trimesh.creation.icosphere(subdivisions=3, radius=0.5)
CUBE = trimesh.creation.box(extents=(1.0, 1.0, 1.0))


def test_chamfer_known_value():
    p1 = np.array([[0.0, 0.0, 0.0]])
    p2 = np.array([[1.0, 0.0, 0.0]])
    assert steval.chamfer_distance(p1, p2) == pytest.approx(1.0)
    assert steval.chamfer_distance(p1, p1) == pytest.approx(0.0)


def test_chamfer_symmetry_and_ordering():
    ps = steval.sample_surface_points(SPHERE, 2000, seed=0)
    ps2 = steval.sample_surface_points(SPHERE, 2000, seed=1)
    pc = steval.sample_surface_points(CUBE, 2000, seed=0)
    self_cd = steval.chamfer_distance(ps, ps2)
    cross_cd = steval.chamfer_distance(ps, pc)
    assert steval.chamfer_distance(pc, ps) == pytest.approx(cross_cd)
    assert self_cd < 0.05
    assert cross_cd > 2 * self_cd


def test_fscore_thresholds():
    p1 = np.array([[0.0, 0.0, 0.0]])
    p2 = np.array([[1.0, 0.0, 0.0]])
    f_tight, precision, recall = steval.fscore(p1, p2, tau=0.5)
    assert (f_tight, precision, recall) == (0.0, 0.0, 0.0)
    f_loose, _, _ = steval.fscore(p1, p2, tau=2.0)
    assert f_loose == pytest.approx(1.0)

    ps = steval.sample_surface_points(SPHERE, 2000, seed=0)
    ps2 = steval.sample_surface_points(SPHERE, 2000, seed=1)
    f_self, _, _ = steval.fscore(ps, ps2, tau=0.05)
    assert f_self > 0.95


def test_occupancy_fills_interior():
    points = steval.sample_surface_points(SPHERE, 20000, seed=0)
    grid = steval.occupancy_grid(points, res=64)
    center = grid.shape[0] // 2
    assert grid[center, center, center]  # hole filling reached the middle
    # occupied volume ~ sphere volume as a fraction of the scene cube
    expected = (4 / 3) * np.pi * 0.5**3 / (2 * steval.SCENE_RADIUS) ** 3
    actual = grid.mean()
    assert 0.6 * expected < actual < 1.6 * expected


def test_voxel_iou_identity_disjoint_and_ordering():
    ps = steval.sample_surface_points(SPHERE, 20000, seed=0)
    pc = steval.sample_surface_points(CUBE, 20000, seed=0)
    gs = steval.occupancy_grid(ps, res=64)
    gc = steval.occupancy_grid(pc, res=64)
    assert steval.voxel_iou(gs, gs) == 1.0
    assert steval.voxel_iou(gs, ~gs) == 0.0
    iou_cross = steval.voxel_iou(gs, gc)
    assert 0.2 < iou_cross < 0.9  # sphere inscribed in the unit cube
    empty = np.zeros_like(gs)
    assert steval.voxel_iou(empty, empty) == 0.0
