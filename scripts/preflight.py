"""Everything the training job needs, checked on a CPU login node first.

A Slurm allocation that dies two minutes in on a typo, a missing shard or a
shadowed torch has still cost the queue wait. This runs every non-GPU part
of the pipeline end to end - imports, versions, paths, the data, the Dataset,
the DataLoader with workers, the split, the unfreeze schedule, the optimizer
grouping, checkpoint write/rotate/resume - and exits non-zero on the first
thing that would have killed the job.

    python scripts/preflight.py --data-root $HOME/StarX/data/StarX

It deliberately does NOT touch CUDA or load TripoSR's weights: the point is
that it runs anywhere, in seconds, before anything is queued. The sbatch
script runs it as its first step, so a bad config fails the job immediately
rather than after the model has loaded.
"""

import argparse
import importlib
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

CHECKS = []
FAILURES = []


def check(name):
    def wrap(fn):
        CHECKS.append((name, fn))
        return fn
    return wrap


def run(args):
    print(f"preflight for {REPO_DIR}")
    print(f"data root: {args.data_root}\n")
    width = max(len(n) for n, _ in CHECKS)
    for name, fn in CHECKS:
        try:
            detail = fn(args) or ""
            print(f"  PASS  {name:<{width}}  {detail}")
        except Exception as error:  # noqa: BLE001 - a preflight reports everything
            FAILURES.append((name, error))
            print(f"  FAIL  {name:<{width}}  {type(error).__name__}: {error}")
            if args.traceback:
                traceback.print_exc()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed - fix these before submitting:")
        for name, error in FAILURES:
            print(f"  - {name}: {error}")
        return 1
    print(f"all {len(CHECKS)} checks passed - safe to sbatch")
    return 0


# ---------------------------------------------------------------- environment

@check("python + torch")
def _torch(args):
    import torch

    user_site = ""
    try:
        import site

        user_site = site.getusersitepackages()
    except Exception:  # noqa: BLE001
        pass
    if user_site and torch.__file__.startswith(user_site):
        raise RuntimeError(
            f"torch is loading from USER site-packages ({user_site}), which "
            f"shadows the environment - move it aside"
        )
    return (f"{torch.__version__} (cuda {torch.version.cuda}) "
            f"py{sys.version_info.major}.{sys.version_info.minor}")


@check("imports")
def _imports(args):
    missing = []
    for module in ("numpy", "PIL", "matplotlib", "tqdm", "torchmetrics",
                   "torchvision", "transformers", "peft", "einops", "omegaconf",
                   "trimesh", "pandas"):
        try:
            importlib.import_module(module)
        except Exception as error:  # noqa: BLE001
            missing.append(f"{module} ({type(error).__name__})")
    if missing:
        raise ImportError("cannot import: " + ", ".join(missing))
    import transformers

    return f"transformers {transformers.__version__}"


@check("starx package")
def _starx(args):
    from starx import cameras, checkpoint, data, shards, sketchdata, synth  # noqa: F401
    from starx import model as smodel  # noqa: F401
    from starx import train as strain  # noqa: F401

    return "all modules import"


@check("triposr weights cached")
def _weights(args):
    """The checkpoint comes from HuggingFace, and an Ibex compute node may
    have no route to the internet - so it has to already be in the cache,
    and the job runs with HF_HUB_OFFLINE=1 to make a miss fail loudly here
    rather than hang on a network timeout inside the allocation."""
    from starx import model as smodel

    roots = [Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))]
    if os.environ.get("HF_HUB_CACHE"):
        roots.insert(0, Path(os.environ["HF_HUB_CACHE"]).parent)
    repo = "models--" + smodel.HF_REPO.replace("/", "--")
    for root in roots:
        for hub in (root / "hub", root):
            snapshots = hub / repo / "snapshots"
            if not snapshots.is_dir():
                continue
            for snap in snapshots.iterdir():
                have = {p.name for p in snap.iterdir()}
                missing = {"model.ckpt", "config.yaml"} - have
                if not missing:
                    size = (snap / "model.ckpt").resolve().stat().st_size
                    return f"{smodel.HF_REPO} cached ({size / 2**30:.1f} GiB)"
    raise FileNotFoundError(
        f"{smodel.HF_REPO} is not in the HuggingFace cache. Pre-download it on "
        f"a node with internet:  python -c \"from huggingface_hub import "
        f"hf_hub_download as d; [d('{smodel.HF_REPO}', f) for f in "
        f"('config.yaml','model.ckpt')]\""
    )


@check("triposr clone")
def _triposr(args):
    from starx import pins

    tsr = Path(args.triposr_dir)
    if not (tsr / "tsr" / "system.py").exists():
        raise FileNotFoundError(f"no TripoSR at {tsr}")
    head = (tsr / ".git" / "HEAD")
    return f"{tsr}  (pin {pins.TRIPOSR_COMMIT[:8]}, HEAD file {'ok' if head.exists() else 'missing'})"


# ------------------------------------------------------------------ the data

@check("shards on disk")
def _shards(args):
    from starx import shards, sketchdata
    from starx.config import StarXConfig, shard_dir, sketch_shard_dir

    cfg = StarXConfig(drive_root=args.data_root)
    notes = []
    for split in ("train", "test"):
        designs = shards.list_done_shards(shard_dir(cfg, split))
        if not designs:
            raise FileNotFoundError(
                f"no design shards at {shard_dir(cfg, split)} - run notebook 08"
            )
        sk = shards.list_done_shards(
            sketch_shard_dir(cfg, split), sketchdata.SKETCH_PREFIX
        )
        notes.append(f"{split}: {len(designs)} design + {len(sk)} sketch shards")
    return "; ".join(notes)


@check("local cache")
def _cache(args):
    from starx import shards, sketchdata
    from starx.config import StarXConfig, shard_dir, sketch_shard_dir

    cfg = StarXConfig(drive_root=args.data_root,
                      local_root=os.path.join(args.data_root, "local_cache"))
    notes = []
    for split in ("train", "test"):
        local = Path(cfg.local_root) / split
        shards.prepare_local(shard_dir(cfg, split), local)
        sk_dir = sketch_shard_dir(cfg, split)
        if shards.list_done_shards(sk_dir, sketchdata.SKETCH_PREFIX):
            sketchdata.check_params(sk_dir, cfg)
            shards.prepare_local(sk_dir, local, prefix=sketchdata.SKETCH_PREFIX)
        n = len(list((local / "cache").glob("*.meta.json")))
        if n == 0:
            raise FileNotFoundError(f"{split} cache is empty at {local / 'cache'}")
        notes.append(f"{split}: {n} designs")
    return "; ".join(notes)


@check("dataset + split")
def _dataset(args):
    from starx import sketchdata
    from starx.config import StarXConfig

    cfg = StarXConfig(drive_root=args.data_root,
                      local_root=os.path.join(args.data_root, "local_cache"))
    cache = Path(cfg.local_root) / "train" / "cache"
    train_ids, val_ids = sketchdata.split_designs(cache, seed=cfg.seed)
    if set(train_ids) & set(val_ids):
        raise AssertionError("train and val designs overlap")
    ds = sketchdata.SketchDataset(
        cache, design_ids=train_ids, supervision_views=args.supervision_views, cfg=cfg
    )
    item = ds[0]
    v = args.supervision_views
    assert item["input"].shape == (3, cfg.sketch_size, cfg.sketch_size), item["input"].shape
    assert item["views"].shape[0] == v, item["views"].shape
    assert item["c2ws"].shape == (v, 4, 4), item["c2ws"].shape
    # the input's own view must be supervised first, at azimuth zero
    import numpy as np

    c = item["c2ws"][0].numpy()
    az = (np.degrees(np.arctan2(c[1, 3], c[0, 3])) + 180) % 360 - 180
    if abs(az) > 1e-3:
        raise AssertionError(f"input camera not canonicalized: azimuth {az}")
    return (f"{len(train_ids)} train / {len(val_ids)} val designs, "
            f"{len(ds):,} samples, sketches={ds.sketches}")


@check("dataloader + workers")
def _loader(args):
    import torch
    from starx import sketchdata
    from starx.config import StarXConfig

    cfg = StarXConfig(drive_root=args.data_root,
                      local_root=os.path.join(args.data_root, "local_cache"))
    cache = Path(cfg.local_root) / "train" / "cache"
    train_ids, _ = sketchdata.split_designs(cache, seed=cfg.seed)
    ds = sketchdata.SketchDataset(
        cache, design_ids=train_ids[: 4 * args.batch],
        supervision_views=args.supervision_views, cfg=cfg,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True, num_workers=min(2, args.workers),
        collate_fn=sketchdata.collate, drop_last=True,
    )
    batch = next(iter(loader))
    shapes = {k: tuple(v.shape) for k, v in batch.items() if torch.is_tensor(v)}
    if shapes["input"][0] != args.batch:
        raise AssertionError(f"bad batch: {shapes}")
    return f"batch {shapes['input']}, views {shapes['views']}"


# ------------------------------------------------------------ training plumbing

@check("unfreeze schedule")
def _stages(args):
    import torch.nn as nn
    from starx import train as strain

    stub = nn.Module()
    for name in ("decoder", "post_processor", "tokenizer", "backbone",
                 "image_tokenizer"):
        stub.add_module(name, nn.Linear(8, 8))
    total = args.total_steps
    seen = []
    for step in (0, *(b + 1 for b in strain.stage_boundaries(total)[1:])):
        name = strain.apply_unfreeze_stage(stub, step, total)
        live = sum(p.numel() for p in stub.parameters() if p.requires_grad)
        seen.append(f"{step}:{name}({live})")
    final = strain.apply_unfreeze_stage(stub, total, total)
    if any(not p.requires_grad for p in stub.parameters()):
        raise AssertionError("final stage does not unfreeze everything")
    return f"{' -> '.join(seen)} -> {final}"


@check("optimizer grouping")
def _optimizer(args):
    import torch.nn as nn
    from starx import train as strain
    from starx.config import StarXConfig

    stub = nn.Module()
    for name in ("decoder", "post_processor", "tokenizer", "backbone",
                 "image_tokenizer"):
        stub.add_module(name, nn.Linear(8, 8))
    cfg = StarXConfig(lr=1e-4, warmup_steps=10)
    optimizer, scheduler = strain.build_finetune_optimizer(
        stub, cfg, args.total_steps
    )
    lrs = {g["name"]: g["lr"] for g in optimizer.param_groups}
    if lrs["image_tokenizer"] >= lrs["decoder"]:
        raise AssertionError(f"encoder lr not discounted: {lrs}")
    for _ in range(cfg.warmup_steps):
        optimizer.step()
        scheduler.step()
    if abs(scheduler.get_last_lr()[0] - cfg.lr) > 1e-9:
        raise AssertionError("warmup does not reach the full rate")
    return f"{len(optimizer.param_groups)} groups, encoder at {lrs['image_tokenizer']:.1e}"


@check("checkpoint roundtrip")
def _checkpoint(args):
    import torch
    import torch.nn as nn
    from starx import checkpoint

    tmp = Path(tempfile.mkdtemp(prefix="starx_preflight_"))
    try:
        model = nn.Linear(8, 8)
        for step in (100, 200, 300):
            checkpoint.save_checkpoint(
                tmp, step,
                {"model": model.state_dict(), "rng": checkpoint.rng_states()},
                keep_k=2,
            )
        kept = sorted(p.name for p in (tmp / "checkpoints").glob("state_*.pt"))
        if len(kept) != 2:
            raise AssertionError(f"rotation kept {kept}")
        found = checkpoint.find_latest(tmp)
        if found is None or found[1] != 300:
            raise AssertionError(f"find_latest returned {found}")
        state = checkpoint.load_checkpoint(found[0])
        checkpoint.restore_rng(state["rng"])
        checkpoint.append_log(tmp / "logs" / "t.jsonl", {"step": 1, "loss": 0.5})
        rows = checkpoint.read_log(tmp / "logs" / "t.jsonl")
        if len(rows) != 1:
            raise AssertionError("jsonl log did not round-trip")
        return f"kept {kept}, resumed at {found[1]}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@check("run dir writable")
def _rundir(args):
    from starx.config import StarXConfig, run_dir

    cfg = StarXConfig(drive_root=args.data_root)
    rdir = run_dir(cfg, args.run_name)
    (rdir / "logs").mkdir(parents=True, exist_ok=True)
    probe = rdir / "logs" / ".preflight"
    probe.write_text("ok")
    probe.unlink()
    existing = checkpoint_count(rdir)
    return f"{rdir}  ({existing} existing checkpoint(s))"


def checkpoint_count(rdir):
    from starx import checkpoint

    return len(list(checkpoint.checkpoint_dir(rdir).glob("state_*.pt")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--run-name", default="sketch_ft_ddp")
    parser.add_argument("--supervision-views", type=int, default=13)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--total-steps", type=int, default=12000)
    parser.add_argument("--triposr-dir",
                        default=str(REPO_DIR / "third_party" / "TripoSR"))
    parser.add_argument("--traceback", action="store_true")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
