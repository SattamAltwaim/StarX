import dataclasses

import numpy as np
import pytest
import torch

from starx import cameras, shards, sketchdata, synth
from starx.config import CAMERA_DISTANCE, StarXConfig

N_VIEWS = 4
GT_SIZE = 16


def _cfg(**overrides) -> StarXConfig:
    base = StarXConfig(sketch_size=32, gt_size=GT_SIZE, n_views=N_VIEWS)
    return dataclasses.replace(base, **overrides) if overrides else base


def _fake_design(design_id: str, split: str = "train"):
    """A design shaped like notebook 03's output, small enough to be fast."""
    rng = np.random.default_rng(abs(hash(design_id)) % 2**32)
    stack = rng.integers(0, 256, (2, GT_SIZE, GT_SIZE), dtype=np.uint8)
    views = np.full((N_VIEWS, GT_SIZE, GT_SIZE, 3), 40, dtype=np.uint8)
    views[:, 4:12, 4:12] = 220  # a bright square: something to find an edge on
    masks = np.zeros((N_VIEWS, GT_SIZE, GT_SIZE), dtype=bool)
    masks[:, 4:12, 4:12] = True
    angles = [(90.0 * v, 15.0) for v in range(N_VIEWS)]
    c2ws = np.stack(
        [cameras.build_spherical_c2w(a, e, CAMERA_DISTANCE) for a, e in angles]
    )
    meta = {
        "design_id": design_id,
        "split": split,
        "view_angles": [[float(a), float(e)] for a, e in angles],
    }
    return shards.encode_sample(design_id, stack, views, masks, c2ws, meta)


def _design_shards(out_dir, design_ids, per_shard=2):
    for index in range(0, len(design_ids), per_shard):
        writer = shards.ShardWriter(out_dir, index // per_shard)
        for design_id in design_ids[index : index + per_shard]:
            writer.add(_fake_design(design_id))
        writer.close()
    return out_dir


@pytest.fixture
def built(tmp_path):
    """Design shards built, sketches built, both extracted into one cache."""
    cfg = _cfg()
    src = _design_shards(tmp_path / "design", [f"d{i:02d}" for i in range(4)])
    out = tmp_path / "sketch"
    stats = sketchdata.build_sketch_shards(src, out, cfg)
    cache = tmp_path / "local"
    shards.prepare_local(src, cache)
    shards.prepare_local(out, cache, prefix=sketchdata.SKETCH_PREFIX)
    return {"cfg": cfg, "src": src, "out": out, "cache": cache / "cache", "stats": stats}


def test_build_covers_every_design_and_view(built):
    assert built["stats"] == {
        "shards_built": 2, "shards_skipped": 0, "designs": 4, "sketches": 4 * N_VIEWS,
    }
    for index in range(2):
        tar = built["out"] / shards.shard_name(index, sketchdata.SKETCH_PREFIX)
        assert tar.exists() and shards.marker_path(tar).exists()


def test_build_is_resumable(built):
    again = sketchdata.build_sketch_shards(built["src"], built["out"], built["cfg"])
    assert again["shards_built"] == 0 and again["shards_skipped"] == 2


def test_unmarked_shard_is_rebuilt(built):
    """An interrupted shard leaves no marker and must be redone."""
    tar = built["out"] / shards.shard_name(0, sketchdata.SKETCH_PREFIX)
    shards.marker_path(tar).unlink()
    again = sketchdata.build_sketch_shards(built["src"], built["out"], built["cfg"])
    assert again["shards_built"] == 1 and again["shards_skipped"] == 1


def test_both_shard_sets_extract_into_one_cache(built):
    """The sketch set must not be mistaken for already-extracted design shards."""
    cache = built["cache"]
    for design_id in [f"d{i:02d}" for i in range(4)]:
        assert (cache / f"{design_id}.meta.json").exists()
        assert (cache / f"{design_id}.view00.png").exists()
        for v in range(N_VIEWS):
            assert (cache / sketchdata.sketch_member_name(design_id, v)).exists()


def test_stored_sketch_matches_the_on_the_fly_path(built):
    cfg = built["cfg"]
    sample = next(shards.iter_shard(built["src"] / "shard_00000.tar"))
    live = synth.sobel_sketch(
        sample["views"][2], cfg.sketch_size, cfg.edge_blur_sigma,
        cfg.edge_gain, cfg.edge_bg,
    )
    stored = sketchdata.load_sketch(built["cache"], sample["design_id"], 2)
    assert stored.shape == live.shape
    assert float((stored - live).abs().max()) <= 1.0 / 255 + 1e-6


def test_input_view_is_supervised_first(built):
    """The input's own view must lead the supervision list - the gallery's
    first prediction column and the azimuth-zero camera depend on it."""
    ds = sketchdata.SketchDataset(built["cache"], supervision_views=2)
    for index in range(N_VIEWS):
        item = ds[index]
        c2w = item["c2ws"][0].numpy()
        azimuth = np.degrees(np.arctan2(c2w[1, 3], c2w[0, 3]))
        assert abs((azimuth + 180.0) % 360.0 - 180.0) < 1e-3


def test_live_sketches_match_stored(built):
    """With no sketch shards present the dataset edge-detects on load; the
    two paths must agree to within the uint8 quantization."""
    stored = sketchdata.SketchDataset(built["cache"], supervision_views=1,
                                      sketches="stored")
    live = sketchdata.SketchDataset(built["cache"], supervision_views=1,
                                    sketches="live", cfg=built["cfg"])
    assert float((stored[0]["input"] - live[0]["input"]).abs().max()) <= 1 / 255 + 1e-6


def test_split_designs_is_disjoint_and_covers(built):
    train_ids, val_ids = sketchdata.split_designs(built["cache"], val_fraction=0.5)
    assert not set(train_ids) & set(val_ids)
    assert len(train_ids) + len(val_ids) == 4


def test_dataset_is_one_sample_per_design_view_pair(built):
    ds = sketchdata.SketchDataset(built["cache"], supervision_views=2)
    assert len(ds) == 4 * N_VIEWS
    assert ds.describe()["views_per_design"] == N_VIEWS
    seen = [ds[i]["input_view"] for i in range(N_VIEWS)]
    assert sorted(seen) == list(range(N_VIEWS))  # design 0 uses each view once


def test_item_shapes_and_supervision_count(built):
    ds = sketchdata.SketchDataset(built["cache"], supervision_views=3)
    item = ds[5]
    assert item["input"].shape == (3, built["cfg"].sketch_size, built["cfg"].sketch_size)
    assert item["views"].shape == (3, GT_SIZE, GT_SIZE, 3)
    assert item["masks"].shape == (3, GT_SIZE, GT_SIZE)
    assert item["c2ws"].shape == (3, 4, 4)
    assert 0.0 <= float(item["input"].min()) and float(item["input"].max()) <= 1.0


def test_cameras_are_canonicalized_to_the_input_view(built):
    """The whole point of the frame fix: whichever view is the input, its
    camera must land at azimuth zero in the returned matrices."""
    ds = sketchdata.SketchDataset(built["cache"], supervision_views=N_VIEWS)
    for index in range(N_VIEWS):
        item = ds[index]
        c2ws = item["c2ws"].numpy()
        azimuths = np.degrees(np.arctan2(c2ws[:, 1, 3], c2ws[:, 0, 3]))
        # the input view is among the supervision views here (k == n_views),
        # so exactly one camera must sit at azimuth zero
        centered = np.abs((azimuths + 180.0) % 360.0 - 180.0)
        assert centered.min() < 1e-3


def test_epoch_redraws_supervision_views_reproducibly(built):
    """A new epoch redraws the non-input views; the same epoch reproduces
    them exactly. Checked across samples, since with few views to choose
    from any single sample may legitimately redraw the same one."""
    ds = sketchdata.SketchDataset(built["cache"], supervision_views=2)
    ds.set_epoch(0)
    first = [ds[i]["c2ws"].clone() for i in range(len(ds))]
    ds.set_epoch(1)
    second = [ds[i]["c2ws"].clone() for i in range(len(ds))]
    ds.set_epoch(0)
    again = [ds[i]["c2ws"] for i in range(len(ds))]

    assert all(torch.equal(a, b) for a, b in zip(first, again))
    assert any(not torch.equal(a, b) for a, b in zip(first, second))
    # the input view leads every list, so it never moves between epochs
    assert all(torch.equal(a[0], b[0]) for a, b in zip(first, second))


def test_all_masks_option_returns_the_whole_rig(built):
    ds = sketchdata.SketchDataset(
        built["cache"], supervision_views=1, return_all_masks=True
    )
    item = ds[0]
    assert item["all_masks"].shape == (N_VIEWS, GT_SIZE, GT_SIZE)
    assert item["all_c2ws"].shape == (N_VIEWS, 4, 4)


def test_split_filter(built):
    assert len(sketchdata.SketchDataset(built["cache"], split="train")) == 4 * N_VIEWS
    assert len(sketchdata.SketchDataset(built["cache"], split="test")) == 0


def test_collate_stacks_tensors_and_keeps_ids(built):
    ds = sketchdata.SketchDataset(built["cache"], supervision_views=2)
    batch = sketchdata.collate([ds[0], ds[1], ds[2]])
    assert batch["input"].shape == (3, 3, built["cfg"].sketch_size, built["cfg"].sketch_size)
    assert batch["views"].shape == (3, 2, GT_SIZE, GT_SIZE, 3)
    assert isinstance(batch["design_id"], list) and len(batch["design_id"]) == 3


def test_params_guard_catches_a_config_edit(built):
    sketchdata.check_params(built["out"], built["cfg"])  # matching config: fine
    drifted = _cfg(edge_gain=9.9)
    with pytest.raises(ValueError, match="different edge"):
        sketchdata.check_params(built["out"], drifted)
    drift = sketchdata.check_params(built["out"], drifted, strict=False)
    assert drift["edge_gain"] == (built["cfg"].edge_gain, 9.9)


def test_batched_and_per_view_sketches_agree_after_quantization():
    rng = np.random.default_rng(0)
    views = rng.integers(0, 256, (3, 24, 24, 3), dtype=np.uint8)
    cfg = _cfg()
    batched = synth.sobel_sketch_batch(
        views, cfg.sketch_size, cfg.edge_blur_sigma, cfg.edge_gain, cfg.edge_bg
    )
    per_view = synth.sketches_for_views(views, cfg, "cpu")
    assert torch.allclose(batched, per_view, atol=1e-4)
    as_bytes = lambda t: (t * 255).round().to(torch.uint8).int()
    assert int((as_bytes(batched) - as_bytes(per_view)).abs().max()) <= 1
