"""Reconstruct a mesh from one or more sketch/render images, using a
fine-tuned checkpoint. Each image is put through the SAME Sobel
edge-detection pipeline (starx.synth.sobel_sketch) the model was trained
on - starx/pins.py and the sketch dataset build all key on this staying
consistent, so a differently-preprocessed input would shift the model off
the distribution it learned.

Each input image gets its own independent reconstruction (this is a
"one sketch in, one mesh out" model - it does not fuse multiple views of
the same object into one shape).

    python scripts/infer_sketch.py \
        --checkpoint $STARX_DATA/runs/sketch_ft_ddp \
        --images view_04.png view_06.png view_12.png \
        --out-dir outputs/

--checkpoint accepts either a run directory (newest checkpoint used) or a
specific state_*.pt file, same convention as finetune_ssim3d.py.
"""

import argparse
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

import numpy as np
import torch
from PIL import Image

from starx import checkpoint, synth
from starx import model as smodel
from starx.config import StarXConfig
from starx.eval import extract_mesh


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True,
                        help="run dir (newest checkpoint used) or a state_*.pt file")
    parser.add_argument("--images", required=True, nargs="+",
                        help="one or more sketch/render image paths")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--triposr-dir",
                        default=str(REPO_DIR / "third_party" / "TripoSR"))
    parser.add_argument("--mc-resolution", type=int, default=256)
    parser.add_argument("--mc-threshold", type=float, default=25.0)
    parser.add_argument("--lora", action="store_true",
                        help="match this if the checkpoint was fine-tuned with --lora")
    return parser.parse_args()


def resolve_checkpoint(path) -> Path:
    path = Path(path)
    if path.is_dir():
        latest = checkpoint.find_latest(path)
        if latest is None:
            raise FileNotFoundError(f"no checkpoints under {path}")
        return latest[0]
    if not path.exists():
        raise FileNotFoundError(f"--checkpoint {path} does not exist")
    return path


def load_as_rgb(path, bg=255) -> np.ndarray:
    """Any image -> (H, W, 3) uint8, alpha/transparency composited onto a
    flat background first - sobel_sketch has no notion of transparency, and
    an uncomposited alpha edge would draw a spurious hard-edged checkerboard
    border into the gradient it differentiates."""
    img = Image.open(path).convert("RGBA")
    canvas = Image.new("RGBA", img.size, (bg, bg, bg, 255))
    canvas.alpha_composite(img)
    return np.asarray(canvas.convert("RGB"), dtype=np.uint8)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = StarXConfig()  # sketch_size / edge_blur_sigma / edge_gain / edge_bg
    # the paper-recipe checkpoints (sketch_ft_ddp, ssim3d_assembly_ft) are the
    # stock-3-channel model - no input surgery - see scripts/train_sketch.py
    model, _ = smodel.build_stock_lora_model(
        cfg, args.triposr_dir, device=device, full_finetune=not args.lora
    )
    ckpt_path = resolve_checkpoint(args.checkpoint)
    state = checkpoint.load_checkpoint(ckpt_path)
    info = smodel.load_full_state_dict(model, state["model"])
    print(f"loaded {info['loaded']} tensors from {ckpt_path}")
    model.eval()
    model.renderer.set_chunk_size(cfg.eval_chunk)

    for image_path in args.images:
        image_path = Path(image_path)
        stem = image_path.stem
        print(f"\n=== {image_path.name} ===")

        rgb = load_as_rgb(image_path)
        sketch = synth.sobel_sketch(
            rgb, cfg.sketch_size, cfg.edge_blur_sigma, cfg.edge_gain, cfg.edge_bg
        )  # (3, S, S) float in [0, 1] - the actual model input

        # save the edge-detected sketch so you can SEE what the model saw
        sketch_png = (sketch[0].numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(sketch_png, mode="L").save(out_dir / f"{stem}_sketch.png")

        with torch.no_grad():
            code = smodel.encode_sketches(model, sketch[None].to(device))[0].float()
            mesh = extract_mesh(model, code, res=args.mc_resolution, threshold=args.mc_threshold)

        if mesh is None:
            print(f"  no surface crossed the density threshold - try a lower "
                  f"--mc-threshold (currently {args.mc_threshold})")
            continue

        obj_path = out_dir / f"{stem}.obj"
        mesh.export(obj_path)
        print(f"  sketch preview: {out_dir / f'{stem}_sketch.png'}")
        print(f"  mesh:           {obj_path}  ({len(mesh.vertices)} verts, {len(mesh.faces)} faces)")


if __name__ == "__main__":
    main()
