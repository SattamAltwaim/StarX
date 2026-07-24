"""StarX: 3D reconstruction of mechanical parts from CAD sketches.

Baseline pipeline: native Fusion 360 sketches, rasterized and channel-stacked,
drive a pretrained TripoSR whose input embedding is inflated to accept them;
early encoder layers and the triplane backbone are fine-tuned with LoRA.
"""

__version__ = "0.1.0"

from starx.config import (
    CAMERA_DISTANCE,
    CHANNEL_MEAN,
    CHANNEL_STD,
    FOVY_DEG,
    SCENE_RADIUS,
    StarXConfig,
)

__all__ = [
    "StarXConfig",
    "CAMERA_DISTANCE",
    "FOVY_DEG",
    "SCENE_RADIUS",
    "CHANNEL_MEAN",
    "CHANNEL_STD",
]
