import numpy as np

from starx import shards


def _fake_sample(design_id, n_channels=3, n_views=2, size=8):
    rng = np.random.default_rng(hash(design_id) % 2**32)
    stack = rng.integers(0, 256, (n_channels, size, size), dtype=np.uint8)
    views = rng.integers(0, 256, (n_views, size, size, 3), dtype=np.uint8)
    masks = rng.random((n_views, size, size)) > 0.5
    c2ws = rng.random((n_views, 4, 4)).astype(np.float32)
    meta = {"design_id": design_id, "split": "train"}
    return stack, views, masks, c2ws, meta


def _write_shard(out_dir, index, design_ids):
    writer = shards.ShardWriter(out_dir, index)
    payloads = {}
    for design_id in design_ids:
        stack, views, masks, c2ws, meta = _fake_sample(design_id)
        writer.add(shards.encode_sample(design_id, stack, views, masks, c2ws, meta))
        payloads[design_id] = (stack, views, masks, c2ws)
    return writer.close(), payloads


def test_write_read_roundtrip(tmp_path):
    tar_path, payloads = _write_shard(tmp_path, 0, ["design_a", "design_b"])
    assert tar_path.name == "shard_00000.tar"
    assert shards.marker_path(tar_path).exists()

    samples = list(shards.iter_shard(tar_path))
    assert [s["design_id"] for s in samples] == ["design_a", "design_b"]
    for s in samples:
        stack, views, masks, c2ws = payloads[s["design_id"]]
        np.testing.assert_array_equal(s["stack"], stack)
        np.testing.assert_array_equal(s["views"], views)
        np.testing.assert_array_equal(s["masks"], masks)
        np.testing.assert_array_equal(s["c2ws"], c2ws)
        assert s["meta"]["n_views"] == 2


def test_done_marker_gates_visibility(tmp_path):
    _write_shard(tmp_path, 0, ["design_a"])
    # a tar without a marker (interrupted build) must be invisible
    (tmp_path / "shard_00001.tar").write_bytes(b"partial garbage")
    done = shards.list_done_shards(tmp_path)
    assert [p.name for p in done] == ["shard_00000.tar"]
    assert shards.list_done_shards(tmp_path / "missing") == []


def test_no_tmp_left_behind(tmp_path):
    _write_shard(tmp_path, 3, ["design_x"])
    assert list(tmp_path.glob("*.tmp")) == []


def test_prepare_local_extracts_once(tmp_path):
    remote = tmp_path / "remote"
    local = tmp_path / "local"
    _write_shard(remote, 0, ["design_a", "design_b"])
    _write_shard(remote, 1, ["design_c"])

    fresh = shards.prepare_local(remote, local)
    assert sorted(fresh) == ["shard_00000.tar", "shard_00001.tar"]
    cache = local / "cache"
    assert (cache / "design_a.sketch.png").exists()
    assert (cache / "design_c.meta.json").exists()
    # no stray tar copies remain locally
    assert list(local.glob("*.tar")) == []

    # second call is a no-op thanks to sentinels
    assert shards.prepare_local(remote, local) == []
