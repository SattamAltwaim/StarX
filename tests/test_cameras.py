import sys
import types

import numpy as np
import pytest

from starx import cameras
from starx.config import CAMERA_DISTANCE, FOVY_DEG

from conftest import triposr_dir


def test_c2w_is_orthonormal_and_at_distance():
    c2w = cameras.build_spherical_c2w(azimuth_deg=33.0, elevation_deg=21.0, distance=1.9)
    rotation = c2w[:3, :3]
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-6)
    assert np.linalg.norm(c2w[:3, 3]) == pytest.approx(1.9, abs=1e-6)
    # camera looks at the origin: -Z column points from camera to origin
    view_dir = -c2w[:3, 2]
    expected = -c2w[:3, 3] / np.linalg.norm(c2w[:3, 3])
    np.testing.assert_allclose(view_dir, expected, atol=1e-6)


def test_sample_design_views_deterministic_and_distinct():
    c2ws_a, angles_a = cameras.sample_design_views("design_x", 4, (-10, 45), seed=7)
    c2ws_b, angles_b = cameras.sample_design_views("design_x", 4, (-10, 45), seed=7)
    np.testing.assert_array_equal(c2ws_a, c2ws_b)
    np.testing.assert_array_equal(angles_a, angles_b)
    c2ws_c, _ = cameras.sample_design_views("design_y", 4, (-10, 45), seed=7)
    assert not np.allclose(c2ws_a, c2ws_c)
    assert angles_a[:, 1].min() >= -10 and angles_a[:, 1].max() <= 45


def test_crop_rays_match_full_grid_window():
    c2w = cameras.build_spherical_c2w(120.0, 15.0, CAMERA_DISTANCE)
    full_o, full_d = cameras.rays_full(c2w, FOVY_DEG, size=64)
    crop_o, crop_d = cameras.rays_for_crop(c2w, FOVY_DEG, 64, (10, 20, 16, 16))
    np.testing.assert_allclose(
        crop_d.numpy(), full_d[10:26, 20:36].numpy(), atol=1e-7
    )
    np.testing.assert_allclose(
        crop_o.numpy(), full_o[10:26, 20:36].numpy(), atol=1e-7
    )


@pytest.mark.skipif(triposr_dir() is None, reason="local TripoSR clone not found")
def test_rays_equal_tsr_reference():
    sys.path.insert(0, str(triposr_dir()))
    sys.modules.setdefault("rembg", types.ModuleType("rembg"))
    from tsr.utils import get_spherical_cameras

    n_views, elevation, size = 4, 20.0, 32
    ref_o, ref_d = get_spherical_cameras(
        n_views, elevation, CAMERA_DISTANCE, FOVY_DEG, size, size
    )
    for k in range(n_views):
        azimuth = 360.0 * k / n_views
        c2w = cameras.build_spherical_c2w(azimuth, elevation, CAMERA_DISTANCE)
        mine_o, mine_d = cameras.rays_full(c2w, FOVY_DEG, size)
        np.testing.assert_allclose(mine_o.numpy(), ref_o[k].numpy(), atol=1e-5)
        np.testing.assert_allclose(mine_d.numpy(), ref_d[k].numpy(), atol=1e-5)
