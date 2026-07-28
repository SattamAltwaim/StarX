import numpy as np
import torch

from starx import cameras, synth
from starx.config import CAMERA_DISTANCE


def _step_image(size: int = 64) -> np.ndarray:
    """A single vertical brightness step - one edge, at a known column."""
    image = np.full((size, size, 3), 40, dtype=np.uint8)
    image[:, size // 2 :] = 220
    return image


def test_sobel_sketch_shape_and_range():
    sketch = synth.sobel_sketch(_step_image(), out_size=32)
    assert sketch.shape == (3, 32, 32)
    assert float(sketch.min()) >= 0.0 and float(sketch.max()) <= 1.0
    # the three channels are the same drawing - the stock encoder wants RGB
    assert torch.equal(sketch[0], sketch[1]) and torch.equal(sketch[0], sketch[2])


def test_sobel_sketch_draws_the_edge_and_leaves_flat_areas_blank():
    sketch = synth.sobel_sketch(_step_image(), out_size=64, blur_sigma=1.0, gain=3.0)[0]
    mid_row = sketch[32]
    assert float(mid_row[30:34].min()) < 0.3  # dark stroke on the step
    assert float(mid_row[:10].min()) > 0.9  # flat left region stays blank
    assert float(mid_row[54:].min()) > 0.9  # flat right region stays blank


def test_blank_image_produces_a_blank_page():
    flat = np.full((64, 64, 3), 128, dtype=np.uint8)
    sketch = synth.sobel_sketch(flat, out_size=32)
    assert float(sketch.min()) > 0.99  # no gradient anywhere -> nothing drawn


def test_gaussian_kernel_is_normalized_and_symmetric():
    kernel = synth.gaussian_kernel1d(1.5)
    assert abs(float(kernel.sum()) - 1.0) < 1e-6
    assert torch.allclose(kernel, kernel.flip(0), atol=1e-7)


def test_sobel_kernels_sum_to_zero():
    assert abs(float(synth.SOBEL_X.sum())) < 1e-7
    assert abs(float(synth.SOBEL_Y.sum())) < 1e-7
    assert torch.equal(synth.SOBEL_X[0, 0].T, synth.SOBEL_Y[0, 0])


def test_canonicalize_puts_the_input_camera_at_azimuth_zero():
    angles = [(37.0, 10.0), (150.0, -5.0), (300.0, 30.0)]
    c2ws = np.stack(
        [cameras.build_spherical_c2w(a, e, CAMERA_DISTANCE) for a, e in angles]
    )
    rotated = synth.canonicalize_to_view(c2ws, angles[0][0])
    azimuth = np.degrees(np.arctan2(rotated[0, 1, 3], rotated[0, 0, 3]))
    assert abs((azimuth + 180.0) % 360.0 - 180.0) < 1e-3
    # elevation (height above the ground plane) is untouched
    assert np.allclose(rotated[:, 2, 3], c2ws[:, 2, 3], atol=1e-5)


def test_canonicalize_is_a_rigid_rotation():
    angles = [(0.0, 0.0), (90.0, 20.0), (215.0, -8.0), (330.0, 44.0)]
    c2ws = np.stack(
        [cameras.build_spherical_c2w(a, e, CAMERA_DISTANCE) for a, e in angles]
    )
    rotated = synth.canonicalize_to_view(c2ws, angles[2][0])

    def pairwise(c):
        return np.linalg.norm(c[:, None, :3, 3] - c[None, :, :3, 3], axis=-1)

    assert np.abs(pairwise(c2ws) - pairwise(rotated)).max() < 1e-4
    # still valid camera frames: rotation blocks stay orthonormal
    for c2w in rotated:
        assert np.allclose(c2w[:3, :3] @ c2w[:3, :3].T, np.eye(3), atol=1e-5)


def test_pick_input_view_is_deterministic_and_covers_every_view():
    n_views = 16
    first = [synth.pick_input_view("design_a", s, n_views, 1337) for s in range(500)]
    again = [synth.pick_input_view("design_a", s, n_views, 1337) for s in range(500)]
    assert first == again
    assert set(first) == set(range(n_views))
    assert first != [synth.pick_input_view("design_b", s, n_views, 1337) for s in range(500)]
