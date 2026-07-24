"""Atomic, resumable checkpointing plus append-only jsonl run logs.

Checkpoints are written to a temp file and moved into place with os.replace,
so a session that dies mid-write never corrupts the newest checkpoint. Only
the newest keep_k checkpoints are retained. What goes inside the checkpoint
is the caller's business (the training notebook saves the trainable-only
state dict from starx.model plus optimizer/scheduler/RNG states).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch


def rng_states() -> dict:
    """Capture python/numpy/torch RNG states (CPU + CUDA when present)."""
    states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def restore_rng(states: dict) -> None:
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])
    if "cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["cuda"])


def checkpoint_dir(run_dir) -> Path:
    return Path(run_dir) / "checkpoints"


def _step_of(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def save_checkpoint(run_dir, step: int, state: dict, keep_k: int = 3) -> Path:
    """Atomically write state as state_{step}.pt and rotate old checkpoints."""
    ckpt_dir = checkpoint_dir(run_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final = ckpt_dir / f"state_{step:07d}.pt"
    tmp = ckpt_dir / (final.name + ".tmp")
    state = dict(state)
    state["step"] = step
    torch.save(state, tmp)
    tmp.replace(final)
    rotate(run_dir, keep_k)
    return final


def rotate(run_dir, keep_k: int) -> None:
    """Delete all but the newest keep_k step checkpoints (best_*.pt is kept)."""
    ckpts = sorted(checkpoint_dir(run_dir).glob("state_*.pt"), key=_step_of)
    for old in ckpts[:-keep_k] if keep_k > 0 else []:
        old.unlink()


def find_latest(run_dir):
    """(path, step) of the newest checkpoint, or None when starting fresh."""
    ckpts = sorted(checkpoint_dir(run_dir).glob("state_*.pt"), key=_step_of)
    if not ckpts:
        return None
    return ckpts[-1], _step_of(ckpts[-1])


def load_checkpoint(path, map_location="cpu") -> dict:
    """Load a checkpoint dict; the caller applies its pieces explicitly."""
    return torch.load(path, map_location=map_location, weights_only=False)


def append_log(jsonl_path, record: dict) -> None:
    """Append one json record; append-only files survive session death."""
    jsonl_path = Path(jsonl_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def read_log(jsonl_path):
    """Read a jsonl log into a pandas DataFrame (empty log -> empty frame)."""
    import pandas as pd

    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        return pd.DataFrame()
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)
