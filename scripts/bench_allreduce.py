"""Is the multi-GPU gradient sync the reason a step costs 18s?

A single-GPU profile says the compute in a fully-unfrozen step is ~2s, so
the other ~16s of the real two-rank step is unaccounted for. The obvious
suspect is the gradient all-reduce, which currently issues ONE COLLECTIVE
PER PARAMETER TENSOR. Every collective carries a fixed latency and a
synchronization; hundreds of them per step is exactly the pathology that
DDP's gradient bucketing exists to avoid.

This times the two arrangements on the real parameter shapes:

  torchrun --standalone --nproc_per_node=2 scripts/bench_allreduce.py \
      --triposr-dir third_party/TripoSR
"""

import argparse
import os
import sys
import time
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

import torch
import torch.distributed as dist

from starx import model as smodel
from starx import train as strain


def timed(fn, repeats=5):
    fn()  # warm the collective up
    torch.cuda.synchronize()
    dist.barrier()
    start = time.time()
    for _ in range(repeats):
        fn()
    torch.cuda.synchronize()
    return (time.time() - start) / repeats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triposr-dir",
                        default=str(REPO_DIR / "third_party" / "TripoSR"))
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    model = smodel.load_pretrained_tsr(args.triposr_dir, device=device)
    strain.apply_unfreeze_stage(model, 10**9, 10**9)
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.grad = torch.randn_like(p)
    elements = sum(p.numel() for p in params)

    if rank == 0:
        print(f"world size {world}, {torch.cuda.get_device_name(0)}")
        print(f"{len(params)} trainable TENSORS, {elements:,} elements "
              f"({elements * 4 / 2**30:.2f} GiB of gradient)\n")

    def per_tensor():
        for p in params:
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world)

    def bucketed():
        grads = [p.grad for p in params]
        flat = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        flat.div_(world)
        for g, merged in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
            g.copy_(merged)

    a = timed(per_tensor, args.repeats)
    b = timed(bucketed, args.repeats)

    if rank == 0:
        print(f"  per-tensor all_reduce ({len(params)} collectives): {a:8.3f} s/step")
        print(f"  one bucketed all_reduce            : {b:8.3f} s/step")
        print(f"  speedup: {a / max(b, 1e-9):.1f}x   saving {a - b:.2f}s per step")
        print(f"\n  per-collective overhead: {a / len(params) * 1000:.2f} ms")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
