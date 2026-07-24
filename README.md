# StarX

Reconstructing 3D mechanical parts from their 2D CAD sketches.

The baseline in this repository takes a design's native parametric sketches from the
Fusion 360 Gallery reconstruction dataset, rasterizes them into a channel-stacked
image (one channel per sketch in the construction timeline, blank-padded to a fixed
count), and feeds them to a pretrained [TripoSR](https://github.com/VAST-AI-Research/TripoSR)
whose input embedding is inflated from 3 RGB channels to 6 sketch channels. The early
image-encoder layers and the triplane backbone are fine-tuned with LoRA (~1.3% of the
420M parameters train); supervision is a rendering loss against ground-truth renders
of each design's final mesh.

## The notebooks

Everything runs as Colab notebooks in `experiments/`, in order. Each notebook's first
cell clones this repo and a pinned TripoSR commit; data, shards, and checkpoints live
on your Google Drive. Long steps are resumable: re-running a notebook after a
disconnect picks up where it stopped.

| # | Notebook | What it does | Runtime |
|---|----------|--------------|---------|
| 01 | `01_dataset_explore` | Download the dataset to Drive, understand a design, gather the stats behind the preprocessing choices | any, ~30 min |
| 02 | `02_sketch_rasterization` | Develop and validate the sketch rasterizer on real designs | any, ~15 min |
| 03 | `03_build_dataset` | Camera-alignment gate, then the full preprocessing pass into tar shards | GPU, hours (resumable) |
| 04 | `04_model_surgery` | Inflate the patch embedding, inject LoRA, verify every step numerically | GPU, ~15 min |
| 05 | `05_training` | The hand-written, resumable training loop with live validation grids | GPU, ~6 h on L4 |
| 06 | `06_evaluation` | Chamfer / F-score / IoU + image metrics on the test split, galleries, turntables, and a pretrained-on-thumbnail baseline | GPU, ~1 h |
| 07 | `07_playground` | Upload your own drawings, get a mesh | GPU, minutes |

Every notebook has a `SMOKE` switch in its configuration cell. Setting it in 03, 05,
and 06 runs a 20-design end-to-end rehearsal (separate `smoke_*` folders on Drive) -
do that once before any full run.

## Repository layout

```
starx/         the package: parsing, rasterization, cameras, rendering,
               shard/checkpoint IO, model surgery, metrics, plot helpers
experiments/   the notebooks (the deliverable)
tests/         pytest suite; runs on a CPU laptop, no dataset needed
tools/         sketch-viewer.html, a standalone viewer for design jsons
```

The split of responsibilities: notebooks contain the ideas (the surgery walkthrough,
the ray march, the training step); the package contains deterministic plumbing that
several notebooks share and that is unit-tested locally. `starx/pins.py` is the single
source of truth for the TripoSR commit and per-notebook pip installs.

## Local development

```
python -m venv --system-site-packages .venv
.venv/bin/pip install trimesh scikit-image omegaconf einops
.venv/bin/python -m pytest
```

The camera tests compare our ray math against `tsr.utils` and need a local TripoSR
clone at `third_party/TripoSR` (any location via `STARX_TRIPOSR_DIR`); they skip
cleanly when absent.

## Licenses and data

- StarX code: MIT.
- TripoSR code and weights: MIT, fetched from upstream at a pinned commit.
- The Fusion 360 Gallery dataset is licensed by Autodesk for non-commercial research
  and may not be redistributed. This repository never contains dataset files (one
  design json ships as a parsing test fixture); everything downloads directly from
  Autodesk's public bucket into your own Drive.
