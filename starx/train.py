"""The training core, shared by notebook 05 and the multi-GPU script.

Supervision has two spaces:

2D (render space, the TripoSR paper recipe): render small crops of the
prediction at ground-truth cameras and score them with pixel MSE,
perceptual LPIPS, and a binary-cross-entropy mask term on opacity.

3D (volume space, ours): carve a visual hull from the sample's stored
posed masks - any voxel that falls outside some silhouette is provably
empty; the intersection of all silhouette cones is a tight superset of
the true solid. Predicted per-voxel occupancy (from the triplane density)
is pulled toward that hull with a soft Dice loss (Milletari et al.,
V-Net). This gives a dense volumetric gradient that sparse 2D crops
cannot, and costs only ~hull_res^3 extra queries per design.

Memory discipline is the two-stage backward: every view (and the 3D term)
backpropagates only to a detached scene-code leaf, freeing its graph
immediately; the accumulated leaf gradient then flows through the encoder
once per design. Multi-GPU runs replicate the model per rank, split each
step's designs across ranks, and all-reduce the ~5.5M trainable gradients
before the (identical) optimizer step - simpler and safer than a DDP
wrapper around a custom double backward.
"""

from __future__ import annotations

import math
import zlib

import numpy as np
import torch
import torch.nn.functional as F

from starx import cameras, data
from starx.config import FOVY_DEG, SCENE_RADIUS, StarXConfig


def composite_over_gray(rgb_fg, opacity, bg: float = 0.5):
    return rgb_fg + bg * (1.0 - opacity[..., None])


def render_rays(model, scene_code, rays_o, rays_d):
    """Differentiable ray march returning (rgb_fg, opacity).

    Adapted from TriplaneNeRFRenderer._forward, which hides opacity and
    composites over white. For every ray: sample points between where it
    enters and leaves the scene box, query the triplane NeRF, and blend
    front to back with the classic alpha-compositing recursion.
    """
    from tsr.utils import rays_intersect_bbox

    renderer, decoder = model.renderer, model.decoder
    shape = rays_o.shape[:-1]
    rays_o, rays_d = rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)
    n_rays = rays_o.shape[0]

    t_near, t_far, valid = rays_intersect_bbox(rays_o, rays_d, renderer.cfg.radius)
    t_near, t_far = t_near[valid], t_far[valid]

    t_vals = torch.linspace(
        0, 1, renderer.cfg.num_samples_per_ray + 1, device=scene_code.device
    )
    t_mid = (t_vals[:-1] + t_vals[1:]) / 2.0
    z_vals = t_near * (1 - t_mid[None]) + t_far * t_mid[None]
    xyz = rays_o[valid][:, None, :] + z_vals[..., None] * rays_d[valid][:, None, :]

    out = renderer.query_triplane(decoder, xyz, scene_code)

    deltas = t_vals[1:] - t_vals[:-1]
    alpha = 1 - torch.exp(-deltas * out["density_act"][..., 0])
    transmittance = torch.cat(
        [
            torch.ones_like(alpha[:, :1]),
            torch.cumprod(1 - alpha[:, :-1] + 1e-10, dim=-1),
        ],
        dim=-1,
    )
    weights = alpha * transmittance
    rgb_valid = (weights[..., None] * out["color"]).sum(dim=-2)
    opacity_valid = weights.sum(dim=-1)

    rgb_fg = torch.zeros(n_rays, 3, dtype=rgb_valid.dtype, device=rgb_valid.device)
    opacity = torch.zeros(n_rays, dtype=opacity_valid.dtype, device=opacity_valid.device)
    rgb_fg[valid] = rgb_valid
    opacity[valid] = opacity_valid
    return rgb_fg.reshape(*shape, 3), opacity.reshape(*shape)


def compute_render_loss(rgb_fg, opacity, gt_rgb, gt_mask, lpips_metric, cfg: StarXConfig):
    """2D terms on one crop: MSE + LPIPS on gray-composited images + mask BCE."""
    gt = torch.where(gt_mask[..., None], gt_rgb, torch.full_like(gt_rgb, 0.5))
    pred = composite_over_gray(rgb_fg, opacity)
    mse = F.mse_loss(pred, gt)
    lpips_term = lpips_metric(
        pred.clamp(0, 1).permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]
    )
    mask_term = F.binary_cross_entropy(opacity.clamp(1e-4, 1 - 1e-4), gt_mask.float())
    total = (
        cfg.lambda_mse * mse
        + cfg.lambda_lpips * lpips_term
        + cfg.lambda_mask * mask_term
    )
    return total, {
        "mse": float(mse),
        "lpips": float(lpips_term),
        "mask": float(mask_term),
    }


def _grid_positions(res: int, device) -> torch.Tensor:
    axis = torch.linspace(-SCENE_RADIUS, SCENE_RADIUS, res, device=device)
    grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)
    return grid.reshape(-1, 3)


def visual_hull_grid(masks, c2ws, res: int, gt_size: int, device):
    """Space carving: (hull, observed), both (res^3,) bool tensors.

    A voxel that projects OUTSIDE some silhouette is provably empty and
    gets carved from the hull. A voxel outside a view's frame is simply
    unobserved by that view - it must not be carved (the frustum at this
    field of view is narrower than the scene box). `observed` marks voxels
    seen by at least one view; only those carry supervision signal. No grad.
    """
    positions = _grid_positions(res, device)
    hull = torch.ones(positions.shape[0], dtype=torch.bool, device=device)
    observed = torch.zeros(positions.shape[0], dtype=torch.bool, device=device)
    focal = 0.5 * gt_size / math.tan(0.5 * math.radians(FOVY_DEG))
    masks_t = torch.as_tensor(np.asarray(masks), device=device)
    for v in range(masks_t.shape[0]):
        c2w = torch.as_tensor(np.asarray(c2ws[v]), dtype=torch.float32, device=device)
        cam = (positions - c2w[:3, 3]) @ c2w[:3, :3]
        z = (-cam[:, 2]).clamp_min(1e-6)
        u = (gt_size / 2 + focal * cam[:, 0] / z).long()
        w = (gt_size / 2 - focal * cam[:, 1] / z).long()
        in_frame = (u >= 0) & (u < gt_size) & (w >= 0) & (w < gt_size)
        observed |= in_frame
        inside = torch.ones_like(hull)  # unobserved by this view: keep
        uc, wc = u.clamp(0, gt_size - 1), w.clamp(0, gt_size - 1)
        inside[in_frame] = masks_t[v][wc[in_frame], uc[in_frame]]
        hull &= inside
    return hull, observed


def query_alpha_grid(model, scene_code, res: int) -> torch.Tensor:
    """Differentiable per-voxel occupancy in [0, 1] from the triplane density."""
    positions = _grid_positions(res, scene_code.device)
    out = model.renderer.query_triplane(model.decoder, positions, scene_code)
    delta = 2 * SCENE_RADIUS / res
    return 1 - torch.exp(-delta * out["density_act"][..., 0])


def soft_dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    """1 - soft Dice overlap between predicted occupancy and a binary target."""
    target = target.float()
    intersection = (pred * target).sum()
    return 1 - (2 * intersection + eps) / (pred.sum() + target.sum() + eps)


def train_step(
    step: int,
    model,
    train_dataset,
    optimizer,
    scheduler,
    trainable_params,
    lpips_metric,
    cfg: StarXConfig,
    device: str,
    amp_dtype,
    grad_scale: float = 1.0,
    rank: int = 0,
    world_size: int = 1,
    input_fn=None,
):
    """One optimizer step. With world_size > 1, each rank handles an
    interleaved slice of the step's designs and gradients are summed
    across ranks before the (identical) update on every rank.

    input_fn is the variant hook: given (item, step) it returns the model
    input tensor and the camera matrices to supervise against. The default
    is the baseline - the design's own sketch stack and its stored cameras.
    Notebook 09 passes starx.synth.make_input_fn instead, which feeds an
    edge sketch of one posed view and cameras rotated into that view's
    frame; nothing else about the step changes."""
    model.train()
    indices = data.draw_design_indices(
        step, cfg.accum_designs, len(train_dataset), cfg.seed
    )
    n_terms = cfg.accum_designs * cfg.views_per_design
    totals = {"loss": 0.0, "mse": 0.0, "lpips": 0.0, "mask": 0.0, "occ": 0.0}

    for index in indices[rank::world_size]:
        item = train_dataset[index]
        if input_fn is None:
            model_input, c2ws = item["sketch"][None].to(device), item["c2ws"]
        else:
            model_input, c2ws = input_fn(item, step)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=device == "cuda"):
            from starx import model as smodel

            scene_code = smodel.encode_sketches(model, model_input)[0]
        scene_code = scene_code.float()
        code_leaf = scene_code.detach().requires_grad_(True)

        view_ids = data.choose_views(
            step, item["design_id"], item["meta"]["n_views"],
            cfg.views_per_design, cfg.seed,
        )
        for v in view_ids:
            crop_rng = np.random.default_rng(
                zlib.crc32(f"{item['design_id']}:{step}:{v}".encode())
            )
            box = data.sample_crop_box(item["masks"][v], cfg.render_crop, crop_rng)
            top, left, h, w = box
            rays_o, rays_d = cameras.rays_for_crop(
                c2ws[v], FOVY_DEG, cfg.gt_size, box
            )
            rgb_fg, opacity = render_rays(
                model, code_leaf, rays_o.to(device), rays_d.to(device)
            )
            gt_rgb = torch.from_numpy(
                np.ascontiguousarray(item["views"][v][top : top + h, left : left + w])
            ).float().to(device) / 255.0
            gt_mask = torch.from_numpy(
                np.ascontiguousarray(item["masks"][v][top : top + h, left : left + w])
            ).to(device)
            loss, parts = compute_render_loss(
                rgb_fg, opacity, gt_rgb, gt_mask, lpips_metric, cfg
            )
            (loss / n_terms).backward()  # frees this view's render graph
            totals["loss"] += float(loss) / n_terms
            for key in ("mse", "lpips", "mask"):
                totals[key] += parts[key] / n_terms

        if cfg.lambda_occ > 0:
            with torch.no_grad():
                hull, observed = visual_hull_grid(
                    item["masks"], c2ws, cfg.hull_res, cfg.gt_size, device
                )
            alpha = query_alpha_grid(model, code_leaf, cfg.hull_res)
            occ_loss = soft_dice_loss(alpha[observed], hull[observed])
            (cfg.lambda_occ * occ_loss / cfg.accum_designs).backward()
            totals["occ"] += float(occ_loss) / cfg.accum_designs
            totals["loss"] += cfg.lambda_occ * float(occ_loss) / cfg.accum_designs

        # stage two: the accumulated scene-code gradient through the encoder
        scene_code.backward(gradient=code_leaf.grad * grad_scale)

    if world_size > 1:
        import torch.distributed as dist

        for p in trainable_params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        stats = torch.tensor(
            [totals[k] for k in ("loss", "mse", "lpips", "mask", "occ")], device=device
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        for i, k in enumerate(("loss", "mse", "lpips", "mask", "occ")):
            totals[k] = float(stats[i])

    if grad_scale != 1.0:
        for p in trainable_params:
            if p.grad is not None:
                p.grad.div_(grad_scale)
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
    totals["skipped"] = 0.0
    if torch.isfinite(grad_norm):
        optimizer.step()
    else:
        totals["skipped"] = 1.0  # overflowed fp16 step - dropped, not applied
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    totals["grad_norm"] = float(grad_norm)
    totals["lr"] = scheduler.get_last_lr()[0]
    return totals


def build_optimizer(model, trainable_params, cfg: StarXConfig):
    """AdamW with no weight decay on LoRA and a little on the new conv,
    plus the warmup-into-cosine schedule."""
    conv_params = list(
        model.image_tokenizer.model.embeddings.patch_embeddings.projection.parameters()
    )
    conv_ids = {id(p) for p in conv_params}
    lora_params = [p for p in trainable_params if id(p) not in conv_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "weight_decay": 0.0},
            {"params": conv_params, "weight_decay": 0.01},
        ],
        lr=cfg.lr,
    )

    def lr_lambda(step):
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / max(1, cfg.total_steps - cfg.warmup_steps)
        return 0.1 + 0.45 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def pick_amp(device: str):
    """(amp_dtype, grad_scale) matched to the GPU: bf16 on Ampere+, scaled
    fp16 elsewhere."""
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16, 1.0
    return torch.float16, 4096.0
