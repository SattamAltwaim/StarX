"""Local (no-CUDA, no-weights) tests for starx.model: the tsr import with its
stubs, and surgery math on a synthetic stand-in conv. The full surgery is
verified against the real checkpoint in notebook 04 (needs the 1.68 GB
download)."""

import sys

import pytest
import torch

from starx import model as smodel
from conftest import triposr_dir


@pytest.mark.skipif(triposr_dir() is None, reason="local TripoSR clone not found")
def test_import_tsr_with_stubs():
    tsr = smodel.import_tsr(triposr_dir())
    assert "rembg" in sys.modules and "torchmcubes" in sys.modules
    from tsr.models.nerf_renderer import TriplaneNeRFRenderer  # imports cleanly

    with pytest.raises(RuntimeError, match="stubbed"):
        sys.modules["torchmcubes"].marching_cubes(None)
    assert tsr is not None


def _fake_model_with_conv():
    """Minimal object tree mimicking model.image_tokenizer.model paths."""

    class Namespace:
        pass

    conv = torch.nn.Conv2d(3, 8, kernel_size=4, stride=4)
    patch = Namespace()
    patch.projection = conv
    patch.num_channels = 3
    embeddings = Namespace()
    embeddings.patch_embeddings = patch
    vit = Namespace()
    vit.embeddings = embeddings
    vit.config = Namespace()
    vit.config.num_channels = 3
    tokenizer = Namespace()
    tokenizer.model = vit
    fake = Namespace()
    fake.image_tokenizer = tokenizer
    return fake, conv


def test_inflate_i3d_mean_preserves_uniform_response():
    fake, old = _fake_model_with_conv()
    new = smodel.inflate_patch_conv(fake, 6, mode="i3d_mean")
    assert new.weight.shape == (8, 6, 4, 4)
    assert fake.image_tokenizer.model.config.num_channels == 6
    gray_rgb = torch.full((1, 3, 8, 8), 0.5)
    gray_six = torch.full((1, 6, 8, 8), 0.5)
    with torch.no_grad():
        torch.testing.assert_close(new(gray_six), old(gray_rgb), atol=1e-5, rtol=1e-5)


def test_inflate_rgb_zero_keeps_first_three_channels():
    fake, old = _fake_model_with_conv()
    new = smodel.inflate_patch_conv(fake, 6, mode="rgb_zero")
    rgb = torch.rand(1, 3, 8, 8)
    padded = torch.cat([rgb, torch.rand(1, 3, 8, 8)], dim=1)
    with torch.no_grad():
        torch.testing.assert_close(new.weight[:, 3:].abs().sum(), torch.tensor(0.0))
        # extra channels are zero-weighted, so any content there is invisible
        torch.testing.assert_close(new(padded), old(rgb), atol=1e-5, rtol=1e-5)

    with pytest.raises(ValueError):
        smodel.inflate_patch_conv(_fake_model_with_conv()[0], 6, mode="bogus")


def test_trainable_state_roundtrip_on_toy_module():
    toy = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    for p in toy[0].parameters():
        p.requires_grad_(False)
    state = smodel.trainable_state_dict(toy)
    assert set(state) == {"1.weight", "1.bias"}

    clone = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    for p in clone[0].parameters():
        p.requires_grad_(False)
    smodel.load_trainable_state_dict(clone, state)
    torch.testing.assert_close(clone[1].weight, toy[1].weight)

    with pytest.raises(ValueError, match="mismatch"):
        smodel.load_trainable_state_dict(clone, {"bogus.weight": torch.zeros(1)})


def test_load_full_state_dict_ignores_current_requires_grad():
    """Seeding a new run from a prior run's checkpoint must copy every
    parameter's VALUE regardless of which ones this run's own unfreeze
    schedule currently has trainable (apply_unfreeze_stage(step=0, ...)
    freezes most of the model before this ever gets called)."""
    src = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    state = {name: p.detach().clone() for name, p in src.named_parameters()}

    dst = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    for p in dst[0].parameters():
        p.requires_grad_(False)  # simulates the fresh schedule's frozen group
    info = smodel.load_full_state_dict(dst, state)
    assert info["loaded"] == 4  # two Linear layers, weight + bias each

    for (name_s, p_s), (name_d, p_d) in zip(
        src.named_parameters(), dst.named_parameters()
    ):
        assert name_s == name_d
        torch.testing.assert_close(p_s, p_d)
    assert not dst[0].weight.requires_grad  # load must not touch requires_grad
    assert dst[1].weight.requires_grad


def test_load_full_state_dict_rejects_mismatched_keys():
    dst = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    with pytest.raises(ValueError, match="match"):
        smodel.load_full_state_dict(dst, {"bogus.weight": torch.zeros(1)})

    partial = {"0.weight": dst[0].weight.detach().clone()}
    with pytest.raises(ValueError, match="match"):
        smodel.load_full_state_dict(dst, partial)
