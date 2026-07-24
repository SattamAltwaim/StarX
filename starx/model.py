"""TripoSR loading, 6-channel input surgery, and LoRA injection.

The surgery is deterministic, so checkpoints only need to store trainable
tensors: rebuilding the model with build_starx_model and loading a trainable
state dict reproduces the full network exactly.

tsr is imported with two heavyweight dependencies stubbed out: rembg
(background removal - StarX inputs are synthetic sketches) and torchmcubes
(its CUDA build is fragile - starx.eval.extract_mesh uses skimage instead).
"""

from __future__ import annotations

import sys
import types

import torch
import torch.nn as nn

from starx.config import CHANNEL_MEAN, CHANNEL_STD, StarXConfig

HF_REPO = "stabilityai/TripoSR"


def import_tsr(triposr_dir):
    """Make the pinned TripoSR clone importable; returns the tsr module."""
    triposr_dir = str(triposr_dir)
    if triposr_dir not in sys.path:
        sys.path.insert(0, triposr_dir)
    if "rembg" not in sys.modules:
        sys.modules["rembg"] = types.ModuleType("rembg")
    if "torchmcubes" not in sys.modules:
        stub = types.ModuleType("torchmcubes")

        def _stubbed(*args, **kwargs):
            raise RuntimeError(
                "torchmcubes is stubbed out - use starx.eval.extract_mesh"
            )

        stub.marching_cubes = _stubbed
        sys.modules["torchmcubes"] = stub
    import tsr

    return tsr


def load_pretrained_tsr(triposr_dir, device: str = "cpu"):
    """Load the pretrained TSR system (weights include the DINO encoder)."""
    import_tsr(triposr_dir)
    from tsr.system import TSR

    model = TSR.from_pretrained(
        HF_REPO, config_name="config.yaml", weight_name="model.ckpt"
    )
    return model.to(device)


def inflate_patch_conv(model, in_channels: int, mode: str = "i3d_mean") -> nn.Conv2d:
    """Replace the DINO patch-embedding conv with an in_channels-input copy.

    i3d_mean: every input slot gets the RGB-mean kernel scaled by 3/N, so a
    uniform image produces exactly the pretrained activations and all sketch
    slots are symmetric. rgb_zero: pretrained kernels in the first three
    slots, zeros elsewhere (extra slots invisible at initialization).
    """
    vit = model.image_tokenizer.model
    old = vit.embeddings.patch_embeddings.projection
    new = nn.Conv2d(
        in_channels,
        old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        bias=old.bias is not None,
    )
    with torch.no_grad():
        weight = old.weight
        n_old = weight.shape[1]
        if mode == "i3d_mean":
            mean_kernel = weight.mean(dim=1, keepdim=True)
            new.weight.copy_(
                mean_kernel.repeat(1, in_channels, 1, 1) * (n_old / in_channels)
            )
        elif mode == "rgb_zero":
            new.weight.zero_()
            new.weight[:, :n_old].copy_(weight)
        else:
            raise ValueError(f"unknown conv_init mode: {mode}")
        if old.bias is not None:
            new.bias.copy_(old.bias)
    new = new.to(weight.device, weight.dtype)
    vit.embeddings.patch_embeddings.projection = new
    vit.embeddings.patch_embeddings.num_channels = in_channels
    vit.config.num_channels = in_channels
    return new


def extend_normalization(model, in_channels: int) -> None:
    """Extend the tokenizer's mean/std buffers to in_channels sketch slots."""
    tokenizer = model.image_tokenizer
    reference = tokenizer.image_mean
    tokenizer.image_mean = torch.full(
        (1, 1, in_channels, 1, 1),
        CHANNEL_MEAN,
        device=reference.device,
        dtype=reference.dtype,
    )
    tokenizer.image_std = torch.full(
        (1, 1, in_channels, 1, 1),
        CHANNEL_STD,
        device=reference.device,
        dtype=reference.dtype,
    )


def _lora_config(cfg: StarXConfig, target_modules, layers_to_transform=None):
    from peft import LoraConfig

    kwargs = {}
    if layers_to_transform is not None:
        kwargs["layers_to_transform"] = list(layers_to_transform)
        kwargs["layers_pattern"] = "layer"
    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(target_modules),
        bias="none",
        **kwargs,
    )


def inject_lora(model, cfg: StarXConfig) -> dict:
    """Two in-place injections: early DINO attention + all backbone attention.

    In-place injection (rather than PeftModel wrapping) preserves module
    identity at TSR's internal call sites.
    """
    from peft import inject_adapter_in_model

    inject_adapter_in_model(
        _lora_config(cfg, ["query", "key", "value"], cfg.dino_layers),
        model.image_tokenizer.model,
        adapter_name="default",
    )
    inject_adapter_in_model(
        _lora_config(cfg, ["to_q", "to_k", "to_v", "to_out.0"]),
        model.backbone,
        adapter_name="default",
    )

    def count(module):
        return sum(1 for name, _ in module.named_modules() if name.endswith("lora_A"))

    return {
        "dino_lora_modules": count(model.image_tokenizer.model),
        "backbone_lora_modules": count(model.backbone),
    }


def set_trainable(model, cfg: StarXConfig) -> dict:
    """Freeze everything, then unfreeze LoRA params, the inflated patch conv,
    and (optionally) the triplane token embedding. Returns a component table."""
    for param in model.parameters():
        param.requires_grad_(False)
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)
    projection = model.image_tokenizer.model.embeddings.patch_embeddings.projection
    for param in projection.parameters():
        param.requires_grad_(True)
    if cfg.train_triplane_embed:
        model.tokenizer.embeddings.requires_grad_(True)
    return param_count_table(model)


def param_count_table(model) -> dict:
    """Per-component {total, trainable} parameter counts."""
    table = {}
    for name, param in model.named_parameters():
        component = name.split(".")[0]
        entry = table.setdefault(component, {"total": 0, "trainable": 0})
        entry["total"] += param.numel()
        if param.requires_grad:
            entry["trainable"] += param.numel()
    table["ALL"] = {
        "total": sum(e["total"] for e in table.values()),
        "trainable": sum(e["trainable"] for e in table.values()),
    }
    return table


def build_starx_model(cfg: StarXConfig, triposr_dir, device: str = "cpu"):
    """Compose load + inflate + extend + inject + freeze; the single
    architecture source that notebooks 05/06/07 load checkpoints against."""
    model = load_pretrained_tsr(triposr_dir, device="cpu")
    inflate_patch_conv(model, cfg.max_sketch_channels, cfg.conv_init)
    extend_normalization(model, cfg.max_sketch_channels)
    lora_counts = inject_lora(model, cfg)
    table = set_trainable(model, cfg)
    return model.to(device), {"lora_counts": lora_counts, "param_table": table}


def trainable_state_dict(model) -> dict:
    """Only the trainable tensors - the ~50 MB thing checkpoints store."""
    return {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def load_trainable_state_dict(model, state: dict) -> None:
    """Strict load of the trainable subset into a freshly built model."""
    trainable = {
        name: param for name, param in model.named_parameters() if param.requires_grad
    }
    if set(trainable) != set(state):
        missing = set(trainable) - set(state)
        unexpected = set(state) - set(trainable)
        raise ValueError(
            f"trainable state mismatch: missing={sorted(missing)[:5]} "
            f"unexpected={sorted(unexpected)[:5]}"
        )
    with torch.no_grad():
        for name, param in trainable.items():
            param.copy_(state[name].to(param.device, param.dtype))


def encode_sketches(model, sketch_batch: torch.Tensor) -> torch.Tensor:
    """Sketch stacks (B, C, H, W) in [0, 1] -> scene codes (B, 3, 40, 64, 64).

    The explicit four-call compose (TSR.forward assumes RGB preprocessing).
    """
    image_tokens = model.image_tokenizer(sketch_batch)  # (B, 768, Nt)
    image_tokens = image_tokens.permute(0, 2, 1)  # (B, Nt, 768)
    tokens = model.tokenizer(sketch_batch.shape[0])
    tokens = model.backbone(tokens, encoder_hidden_states=image_tokens)
    return model.post_processor(model.tokenizer.detokenize(tokens))
