import pytest
import torch

from starx.ssim3d import MSSSIM3DLoss, SSIM3DLoss, ms_ssim3d, ssim3d


def test_identical_volumes_score_near_one():
    x = torch.rand(1, 1, 16, 16, 16)
    assert float(ssim3d(x, x, win_size=5, sigma=1.0)) == pytest.approx(1.0, abs=1e-4)


def test_ssim_drops_for_dissimilar_volumes():
    torch.manual_seed(0)
    x = torch.rand(1, 1, 16, 16, 16)
    y = torch.rand(1, 1, 16, 16, 16)
    identical = float(ssim3d(x, x, win_size=5, sigma=1.0))
    different = float(ssim3d(x, y, win_size=5, sigma=1.0))
    assert different < identical


def test_shape_mismatch_raises():
    x = torch.rand(1, 1, 16, 16, 16)
    y = torch.rand(1, 1, 16, 16, 8)
    with pytest.raises(ValueError, match="shape mismatch"):
        ssim3d(x, y)


def test_wrong_ndim_raises():
    x = torch.rand(1, 16, 16, 16)  # missing the channel dim
    with pytest.raises(ValueError, match="N, C, D, H, W"):
        ssim3d(x, x)


def test_volume_smaller_than_window_raises():
    x = torch.rand(1, 1, 4, 4, 4)
    with pytest.raises(ValueError, match="smaller than the window"):
        ssim3d(x, x, win_size=7)


def test_even_win_size_raises():
    x = torch.rand(1, 1, 8, 8, 8)
    with pytest.raises(ValueError, match="odd"):
        ssim3d(x, x, win_size=4)


def test_gradient_flows_to_first_argument():
    x = torch.rand(1, 1, 12, 12, 12, requires_grad=True)
    y = torch.rand(1, 1, 12, 12, 12)
    loss = 1.0 - ssim3d(x, y, win_size=5, sigma=1.0)
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_reduction_none_returns_per_sample():
    x = torch.rand(3, 1, 10, 10, 10)
    y = torch.rand(3, 1, 10, 10, 10)
    out = ssim3d(x, y, win_size=5, sigma=1.0, reduction="none")
    assert out.shape == (3,)


def test_anisotropic_window_matches_per_axis_triples():
    x = torch.rand(1, 1, 4, 12, 12)
    # a (3, 11, 11) window on a 4-slice volume must not raise, unlike a
    # scalar win_size=11 which would need >= 11 on every axis
    out = ssim3d(x, x, win_size=(3, 11, 11), sigma=(0.5, 1.5, 1.5))
    assert float(out) == pytest.approx(1.0, abs=1e-4)


def test_same_padding_preserves_shape_valid_crops():
    x = torch.rand(1, 1, 16, 16, 16)
    y = torch.rand(1, 1, 16, 16, 16)
    same = ssim3d(x, y, win_size=5, sigma=1.0, padding="same", reduction="none")
    valid = ssim3d(x, y, win_size=5, sigma=1.0, padding="valid", reduction="none")
    assert same.shape == (1,) and valid.shape == (1,)  # reduced the same way


def test_ms_ssim_identical_volumes_score_near_one():
    x = torch.rand(1, 1, 24, 24, 24)
    out = float(ms_ssim3d(x, x, win_size=5, sigma=1.0, pool=2))
    assert out == pytest.approx(1.0, abs=1e-3)


def test_ms_ssim_too_small_raises():
    x = torch.rand(1, 1, 8, 8, 8)
    with pytest.raises(ValueError, match="too small"):
        ms_ssim3d(x, x, win_size=5, sigma=1.0, pool=2)


def test_loss_modules_are_one_minus_metric():
    x = torch.rand(1, 1, 12, 12, 12)
    y = torch.rand(1, 1, 12, 12, 12)
    ssim_loss = SSIM3DLoss(win_size=5, sigma=1.0)
    torch.testing.assert_close(
        ssim_loss(x, y), 1.0 - ssim3d(x, y, win_size=5, sigma=1.0)
    )
    ms_loss = MSSSIM3DLoss(win_size=5, sigma=1.0, pool=1)
    torch.testing.assert_close(
        ms_loss(x, y), 1.0 - ms_ssim3d(x, y, win_size=5, sigma=1.0, pool=1)
    )
