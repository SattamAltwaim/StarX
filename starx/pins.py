"""Version pins shared by every notebook.

TRIPOSR_COMMIT is the single source of truth for the TripoSR clone; setup
cells define the same value locally (they need it before this package is
importable) and assert equality against this module right after import.

PIP_PINS lists, per notebook, only the packages Colab does not preinstall.
After the first successful full run these lists get frozen to the exact
versions that session reports (Phase 2 of the build plan).

Note: TripoSR's own requirements.txt pins transformers==4.35.0; we
deliberately rely on Colab's preinstalled transformers instead - the
ViTModel API surface TripoSR touches is stable, and modern peft requires a
modern transformers. rembg and torchmcubes are never installed: both are
stubbed out by starx.model.import_tsr (background removal is unused with
synthetic sketches, and mesh extraction uses skimage marching cubes).
"""

TRIPOSR_REPO = "https://github.com/VAST-AI-Research/TripoSR.git"
TRIPOSR_COMMIT = "107cefdc244c39106fa830359024f6a2f1c78871"

PIP_PINS = {
    "01": ["trimesh"],
    "02": [],
    "03": ["pyrender", "trimesh", "omegaconf", "einops"],
    "04": ["omegaconf", "einops", "peft"],
    "05": ["omegaconf", "einops", "peft", "torchmetrics"],
    "06": [
        "omegaconf",
        "einops",
        "peft",
        "torchmetrics",
        "trimesh",
        "pyrender",
        "scikit-image",
        "imageio",
    ],
    "07": ["omegaconf", "einops", "peft", "trimesh", "scikit-image", "imageio"],
}
