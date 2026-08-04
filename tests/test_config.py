from pathlib import Path

from starx.config import (
    StarXConfig,
    assembly_shard_dir,
    assembly_sketch_shard_dir,
    run_dir,
    shard_dir,
    sketch_shard_dir,
)


def test_assembly_dirs_are_siblings_of_the_reconstruction_dirs():
    cfg = StarXConfig(drive_root="/data/StarX")
    assert assembly_shard_dir(cfg, "train") == Path("/data/StarX/assembly_shards/train")
    assert assembly_sketch_shard_dir(cfg, "train") == Path(
        "/data/StarX/assembly_sketch_shards/train"
    )
    # never the same path as the reconstruction dataset's own dirs
    assert assembly_shard_dir(cfg, "train") != shard_dir(cfg, "train")
    assert assembly_sketch_shard_dir(cfg, "train") != sketch_shard_dir(cfg, "train")


def test_assembly_dirs_vary_by_split():
    cfg = StarXConfig(drive_root="/data/StarX")
    assert assembly_shard_dir(cfg, "train") != assembly_shard_dir(cfg, "test")


def test_run_dir_unaffected_by_assembly_additions():
    cfg = StarXConfig(drive_root="/data/StarX")
    assert run_dir(cfg, "ssim3d_assembly_ft") == Path("/data/StarX/runs/ssim3d_assembly_ft")
