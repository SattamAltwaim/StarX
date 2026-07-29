"""Configuration for the StarX baseline.

Three tiers, by design:
- FIXED constants below: values baked into the pretrained TripoSR checkpoint
  (camera geometry, normalization statistics). Notebooks print them but must
  never tune them - changing them silently breaks pretrained compatibility.
- StarXConfig: every tunable, with defaults. Notebook config cells construct
  an instance with all relevant fields written out explicitly.
- Notebook-local knobs (RUN_NAME, SMOKE, ...) stay plain variables in the
  notebook config cell.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Fixed by the pretrained TripoSR checkpoint - never tune.
CAMERA_DISTANCE = 1.9
FOVY_DEG = 40.0
SCENE_RADIUS = 0.87  # slightly larger than half the unit-cube diagonal (0.866)
CHANNEL_MEAN = 0.449  # ImageNet grand mean, applied to every sketch channel
CHANNEL_STD = 0.226  # ImageNet grand std, applied to every sketch channel

DATA_URL = (
    "https://fusion-360-gallery-dataset.s3.us-west-2.amazonaws.com/"
    "reconstruction/r1.0.1/r1.0.1.zip"
)


@dataclass(frozen=True)
class StarXConfig:
    # paths and data
    drive_root: str = "/content/drive/MyDrive/StarX"
    local_root: str = "/content/starx_local"
    data_url: str = DATA_URL
    shard_size: int = 200

    # sketch rasterization
    sketch_size: int = 512
    supersample: int = 2
    stroke_width: int = 5  # in supersampled pixels
    stroke_value: int = 0
    bg_value: int = 128
    margin: float = 0.05
    normalization_mode: str = "shared"  # "shared" or "per_sketch"
    max_sketch_channels: int = 6
    include_construction: bool = False
    truncate_extra_sketches: bool = True

    # ground-truth supervision views
    gt_size: int = 256
    n_views: int = 16
    elev_range: tuple = (-10.0, 45.0)

    # synthetic-edge variant (notebook 09: stock model, no surgery)
    edge_blur_sigma: float = 1.2
    edge_gain: float = 3.0
    edge_bg: float = 1.0  # 1.0 = white page, 0.5 = TripoSR's composite gray

    # TripoSR's published training recipe. Its report is 5 pages with no
    # appendix, and states exactly six training numbers (Table 1 + Sec 2.3):
    # AdamW, lr 4e-4, CosineAnnealingLR, warmup 2000, lambda_lpips 2.0,
    # lambda_mask 0.05 - plus 128px foreground-biased crops out of 512px
    # ground truth, and a BCE mask loss. Everything below is something the
    # paper is SILENT on, filled from LRM (the architecture TripoSR builds
    # on) or chosen for this hardware; none of it can be called "the paper's".
    weight_decay: float = 0.05  # LRM's; on non-bias, non-LayerNorm only
    adam_betas: tuple = (0.9, 0.95)  # LRM's; TripoSR does not say
    supervision_views: int = 4  # the paper's V is never given a value
    batch_designs: int = 8  # shapes per optimizer step; LRM's was 1024
    composite_bg: float = 0.5  # TripoSR's inference gray; LRM trained on white

    # model surgery
    conv_init: str = "i3d_mean"  # "i3d_mean" or "rgb_zero"
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    dino_layers: tuple = (0, 1, 2, 3)
    train_triplane_embed: bool = False

    # training
    total_steps: int = 20000
    lr: float = 1e-4
    warmup_steps: int = 500
    accum_designs: int = 8
    views_per_design: int = 2
    render_crop: int = 128
    lambda_mse: float = 1.0
    lambda_lpips: float = 2.0
    lambda_mask: float = 0.05
    lambda_occ: float = 0.5  # 3D visual-hull soft-Dice term (0 disables)
    hull_res: int = 48  # occupancy grid resolution for the 3D term
    grad_clip: float = 1.0
    ckpt_every: int = 500
    keep_k: int = 3
    val_every: int = 1000

    # evaluation
    mc_resolution: int = 256
    mc_threshold: float = 25.0
    iou_res: int = 64
    n_points: int = 10000
    taus: tuple = (0.01, 0.02, 0.05)
    eval_chunk: int = 131072

    seed: int = 1337


def raw_zip_path(cfg: StarXConfig) -> Path:
    """Where the dataset zip lives on Drive."""
    return Path(cfg.drive_root) / "raw" / "r1.0.1.zip"


def stats_dir(cfg: StarXConfig) -> Path:
    """Cached dataset statistics (sketch-count scan, parse stats)."""
    return Path(cfg.drive_root) / "stats"


def shard_dir(cfg: StarXConfig, split: str) -> Path:
    """Drive directory holding one split's tar shards."""
    return Path(cfg.drive_root) / "shards" / split


def sketch_shard_dir(cfg: StarXConfig, split: str) -> Path:
    """Drive directory holding one split's synthetic-sketch shards. A sibling
    of the design shards, extracted into the same local cache."""
    return Path(cfg.drive_root) / "sketch_shards" / split


def run_dir(cfg: StarXConfig, run_name: str) -> Path:
    """Drive directory for one training run (checkpoints, logs, val grids)."""
    return Path(cfg.drive_root) / "runs" / run_name
