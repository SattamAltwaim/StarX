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
| 08 | `08_transfer_to_ibex` | Runs on Ibex: pull the processed data from Google Drive with rclone (Ibex is not reachable from Colab) | Ibex, minutes |
| 09 | `09_synthetic_sketches` | The control variant: convolutional edge detection turns notebook 03's posed renders into synthetic line drawings, and stock 3-channel TripoSR is fine-tuned on them with no surgery | GPU, ~6 h on L4 |
| 10 | `10_build_sketch_dataset` | Materializes those drawings for every design and view into shards beside the design shards (~1.5 GiB), then opens them as the `Dataset` the paper-recipe run uses | any, ~1 h (resumable) |
| 11 | `11_finetune_triposr` | Runs on Ibex: fine-tunes stock TripoSR on that dataset - one sketch in, 13 supervision views, full fine-tuning with gradual unfreezing and discriminative learning rates | Ibex A100 |

Every notebook has a `SMOKE` switch in its configuration cell. Setting it in 03, 05,
and 06 runs a 20-design end-to-end rehearsal (separate `smoke_*` folders on Drive) -
do that once before any full run.

## Repository layout

```
starx/         the package: parsing, rasterization, cameras, rendering,
               shard/checkpoint IO, model surgery, metrics, plot helpers
experiments/   the notebooks (the deliverable)
scripts/       torchrun entry points: train.py (surgered baseline),
               train_sketch.py (paper-recipe run on the sketch dataset)
tests/         pytest suite; runs on a CPU laptop, no dataset needed
tools/         sketch-viewer.html, a standalone viewer for design jsons
```

## The TripoSR recipe, and what is actually in the paper

Notebook 11 and `scripts/train_sketch.py` follow TripoSR's published training
setup. Worth knowing that the report is five pages with no appendix and states
exactly six training numbers: AdamW, lr 4e-4, `CosineAnnealingLR`, 2,000 warmup
steps, `lambda_LPIPS` 2.0, `lambda_mask` 0.05 - plus a BCE mask loss on rendered
opacity and 128px foreground-biased crops taken from 512px ground truth.

It is silent on batch size, total steps, supervision views per object, weight
decay, gradient clipping, and precision. Those come from LRM (the architecture
TripoSR builds on) or are fitted to the hardware, and the config marks which is
which. Three things cannot match here: ground truth is 256px rather than 512px,
the batch is a fraction of LRM's 1024 shapes, and 4e-4 is a from-scratch rate
being applied to a converged checkpoint. The visual-hull term from the baseline
is off in this arm, since the paper's model relies on rendering losses alone.

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
