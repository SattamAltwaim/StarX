"""Version pins shared by every notebook.

TRIPOSR_COMMIT is the single source of truth for the TripoSR clone; setup
cells define the same value locally (they need it before this package is
importable) and assert equality against this module right after import.

PIP_PINS lists, per notebook, what must be installed on top of (or instead
of) Colab's preinstalled packages.

Two pins are load-bearing, verified end-to-end locally on 2026-07-24:
- transformers==5.5.4: transformers 5.6+ renamed the ViT module tree
  (encoder.layer.N.attention.attention.query -> layers.N.attention.q_proj),
  which makes the TripoSR checkpoint's state dict unloadable and would break
  every surgery path and LoRA target name in this project. 5.5.4 keeps the
  classic naming and works with current huggingface_hub and peft.
- peft==0.19.1: the version the surgery (inject_adapter_in_model with
  layers_to_transform) was verified against.

rembg and torchmcubes are never installed: both are stubbed out by
starx.model.import_tsr (background removal is unused with synthetic
sketches, and mesh extraction uses skimage marching cubes).

PIP_UNINSTALL removes Colab-preinstalled packages that break the pinned
stack: peft's LoRA dispatcher probes torchao and RAISES when it finds a
version older than its minimum (Colab ships 0.10.0), even though nothing
here uses torchao. With torchao absent, the probe correctly reports
unavailable and injection proceeds.
"""

TRIPOSR_REPO = "https://github.com/VAST-AI-Research/TripoSR.git"
TRIPOSR_COMMIT = "107cefdc244c39106fa830359024f6a2f1c78871"

TRANSFORMERS_PIN = "transformers==5.5.4"
PEFT_PIN = "peft==0.19.1"

PIP_UNINSTALL = {
    "04": ["torchao"],
    "05": ["torchao"],
    "06": ["torchao"],
    "07": ["torchao"],
}

PIP_PINS = {
    "01": ["trimesh"],
    "02": [],
    # trimesh appears everywhere tsr is imported: tsr/system.py and
    # tsr/utils.py import it at module top, surgery or not.
    "03": ["pyrender", "trimesh", "omegaconf", "einops", TRANSFORMERS_PIN],
    "04": ["trimesh", "omegaconf", "einops", TRANSFORMERS_PIN, PEFT_PIN],
    "05": ["trimesh", "omegaconf", "einops", TRANSFORMERS_PIN, PEFT_PIN, "torchmetrics"],
    "06": [
        "omegaconf",
        "einops",
        TRANSFORMERS_PIN,
        PEFT_PIN,
        "torchmetrics",
        "trimesh",
        "pyrender",
        "scikit-image",
        "imageio",
    ],
    "07": [
        "omegaconf",
        "einops",
        TRANSFORMERS_PIN,
        PEFT_PIN,
        "trimesh",
        "scikit-image",
        "imageio",
    ],
    "08": [],
}
