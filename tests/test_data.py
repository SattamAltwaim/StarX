import numpy as np

from starx import data, shards
from test_shards import _write_shard


def _make_cache(tmp_path):
    remote = tmp_path / "remote"
    local = tmp_path / "local"
    _write_shard(remote, 0, ["design_a", "design_b", "design_c"])
    shards.prepare_local(remote, local)
    return local / "cache"


def test_dataset_len_and_item(tmp_path):
    dataset = data.DesignDataset(_make_cache(tmp_path))
    assert len(dataset) == 3
    item = dataset[0]
    assert item["design_id"] == "design_a"
    assert item["sketch"].shape == (3, 8, 8)
    assert item["sketch"].dtype.is_floating_point
    assert 0.0 <= item["sketch"].min() and item["sketch"].max() <= 1.0
    assert item["views"].shape == (2, 8, 8, 3)
    assert item["masks"].shape == (2, 8, 8) and item["masks"].dtype == bool
    assert item["c2ws"].shape == (2, 4, 4)


def test_dataset_split_filter(tmp_path):
    cache = _make_cache(tmp_path)
    assert len(data.DesignDataset(cache, split="train")) == 3
    assert len(data.DesignDataset(cache, split="test")) == 0


def test_draw_design_indices_deterministic():
    a = data.draw_design_indices(step=12, n=8, dataset_len=100, seed=1337)
    b = data.draw_design_indices(step=12, n=8, dataset_len=100, seed=1337)
    c = data.draw_design_indices(step=13, n=8, dataset_len=100, seed=1337)
    assert a == b and a != c and len(a) == 8
    assert all(0 <= i < 100 for i in a)


def test_choose_views_deterministic_no_replacement():
    a = data.choose_views(step=5, design_id="x", n_available=16, k=4, seed=0)
    b = data.choose_views(step=5, design_id="x", n_available=16, k=4, seed=0)
    assert a == b and len(set(a)) == 4
    assert data.choose_views(5, "x", n_available=2, k=4, seed=0).__len__() == 2


def test_sample_crop_box_foreground_bias():
    rng = np.random.default_rng(0)
    mask = np.zeros((64, 64), dtype=bool)
    mask[40:50, 10:20] = True
    hits = 0
    for _ in range(200):
        top, left, h, w = data.sample_crop_box(mask, 16, rng, foreground_p=1.0)
        assert 0 <= top <= 48 and 0 <= left <= 48 and h == w == 16
        if mask[top : top + h, left : left + w].any():
            hits += 1
    assert hits == 200  # always contains foreground when centered on mask pixels

    empty = np.zeros((64, 64), dtype=bool)
    top, left, h, w = data.sample_crop_box(empty, 16, rng)
    assert 0 <= top <= 48 and 0 <= left <= 48

    assert data.sample_crop_box(mask, 64, rng) == (0, 0, 64, 64)
