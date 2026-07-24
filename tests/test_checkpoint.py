import random

import numpy as np
import torch

from starx import checkpoint


def _make_state(value: float) -> dict:
    model = torch.nn.Linear(4, 2)
    with torch.no_grad():
        model.weight.fill_(value)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.ones(1, 4)).sum().backward()
    optimizer.step()
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": checkpoint.rng_states(),
    }


def test_save_rotate_find_latest(tmp_path):
    for step in [100, 200, 300, 400, 500]:
        checkpoint.save_checkpoint(tmp_path, step, _make_state(step), keep_k=3)
    ckpts = sorted(checkpoint.checkpoint_dir(tmp_path).glob("state_*.pt"))
    assert [p.name for p in ckpts] == [
        "state_0000300.pt",
        "state_0000400.pt",
        "state_0000500.pt",
    ]
    path, step = checkpoint.find_latest(tmp_path)
    assert step == 500 and path.name == "state_0000500.pt"
    assert list(checkpoint.checkpoint_dir(tmp_path).glob("*.tmp")) == []


def test_find_latest_empty(tmp_path):
    assert checkpoint.find_latest(tmp_path) is None


def test_state_roundtrip(tmp_path):
    saved = _make_state(1.5)
    checkpoint.save_checkpoint(tmp_path, 42, saved, keep_k=3)
    path, step = checkpoint.find_latest(tmp_path)
    loaded = checkpoint.load_checkpoint(path)
    assert loaded["step"] == 42
    torch.testing.assert_close(loaded["model"]["weight"], saved["model"]["weight"])
    assert loaded["optimizer"]["state"].keys() == saved["optimizer"]["state"].keys()


def test_rng_restore_reproduces_draws():
    states = checkpoint.rng_states()
    draws_a = (random.random(), np.random.rand(), torch.rand(3))
    checkpoint.restore_rng(states)
    draws_b = (random.random(), np.random.rand(), torch.rand(3))
    assert draws_a[0] == draws_b[0]
    assert draws_a[1] == draws_b[1]
    torch.testing.assert_close(draws_a[2], draws_b[2])


def test_log_append_and_read(tmp_path):
    log = tmp_path / "logs" / "train_log.jsonl"
    for step in range(3):
        checkpoint.append_log(log, {"step": step, "loss": 1.0 / (step + 1)})
    df = checkpoint.read_log(log)
    assert list(df["step"]) == [0, 1, 2]
    assert checkpoint.read_log(tmp_path / "missing.jsonl").empty
