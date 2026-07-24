import dataclasses

import numpy as np
import pytest

from starx import rasterize
from starx.config import StarXConfig

CFG = StarXConfig()


def test_stack_shape_padding_and_values(fixture_design):
    stack, meta = rasterize.rasterize_design(fixture_design, CFG)
    C, H, W = CFG.max_sketch_channels, CFG.sketch_size, CFG.sketch_size
    assert stack.shape == (C, H, W) and stack.dtype == np.uint8
    # fixture has 4 sketches: channels 0-3 drawn, 4-5 blank padding
    assert meta["n_sketches_total"] == 4
    assert meta["blank_channels"] == [False, False, False, False, True, True]
    assert meta["truncated"] is False and meta["all_blank"] is False
    for i in range(4):
        assert stack[i].min() < CFG.bg_value  # dark strokes present
        assert stack[i, 0, 0] == CFG.bg_value  # corners stay background
    for i in (4, 5):
        assert np.all(stack[i] == CFG.bg_value)


def test_truncation(fixture_design):
    cfg = dataclasses.replace(CFG, max_sketch_channels=2)
    stack, meta = rasterize.rasterize_design(fixture_design, cfg)
    assert stack.shape[0] == 2
    assert meta["truncated"] is True and meta["n_channels_used"] == 2

    strict = dataclasses.replace(cfg, truncate_extra_sketches=False)
    with pytest.raises(ValueError):
        rasterize.rasterize_design(fixture_design, strict)


def test_normalization_modes(fixture_design):
    _, meta_shared = rasterize.rasterize_design(fixture_design, CFG)
    assert np.isscalar(meta_shared["px_per_cm"])

    cfg = dataclasses.replace(CFG, normalization_mode="per_sketch")
    stack, meta_ps = rasterize.rasterize_design(fixture_design, cfg)
    scales = [s for s in meta_ps["px_per_cm"] if s is not None]
    assert len(scales) == 4
    # fixture sketches differ in extent, so per-sketch scales must differ
    assert max(scales) > min(scales) * 1.01
    # per-sketch mode: every drawn channel's ink spans ~ the margin box
    target = cfg.sketch_size * (1.0 - 2.0 * cfg.margin)
    for i in range(4):
        ys, xs = np.where(stack[i] < cfg.bg_value)
        span = max(xs.max() - xs.min(), ys.max() - ys.min())
        assert abs(span - target) < 0.03 * cfg.sketch_size


def test_strip_roundtrip(fixture_design):
    stack, _ = rasterize.rasterize_design(fixture_design, CFG)
    strip = rasterize.stack_to_strip(stack)
    assert strip.shape == (CFG.sketch_size, CFG.max_sketch_channels * CFG.sketch_size)
    back = rasterize.strip_to_stack(strip, CFG.max_sketch_channels)
    np.testing.assert_array_equal(back, stack)


def test_determinism(fixture_design):
    a, _ = rasterize.rasterize_design(fixture_design, CFG)
    b, _ = rasterize.rasterize_design(fixture_design, CFG)
    np.testing.assert_array_equal(a, b)


def test_stack_to_tensor_is_raw_01():
    # raw [0,1] floats - the model's internal buffers do the normalization
    stack = np.full((2, 4, 4), 128, dtype=np.uint8)
    x = rasterize.stack_to_tensor(stack)
    assert x.shape == (2, 4, 4)
    np.testing.assert_allclose(x.numpy(), 128 / 255.0, rtol=1e-6)
