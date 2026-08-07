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
from starx.ssim3d import ssim3d as ssim3d_score


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
    """2D terms on one crop: MSE + LPIPS on composited images + mask BCE.

    Both sides composite over the same flat background, cfg.composite_bg.
    StarX's default is TripoSR's inference gray (0.5); LRM trained against
    pure white, so set it to 1.0 for the paper-faithful arrangement.
    """
    bg = getattr(cfg, "composite_bg", 0.5)
    gt = torch.where(gt_mask[..., None], gt_rgb, torch.full_like(gt_rgb, bg))
    pred = composite_over_gray(rgb_fg, opacity, bg)
    mse = F.mse_loss(pred, gt)
    lpips_term = lpips_metric(
        pred.clamp(0, 1).permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]
    )
    # torchmetrics' LPIPS is a Metric, not a plain loss: every call appends
    # to its `all_scores` list state, and its forward() copies that state on
    # each call - so cost grows LINEARLY with the number of calls ever made,
    # making training quadratic in steps. Measured on an A100: 7.1 ms flat
    # with this reset, 8.6 -> 12.1 ms over the first 600 calls without it.
    # At 26 calls per step that reached ~26 s/step by step 4,000 of a real
    # run. We want the value, never the running average, so drop the state.
    lpips_metric.reset()
    mask_term = F.binary_cross_entropy(opacity.clamp(1e-4, 1 - 1e-4), gt_mask.float())
    total = (
        cfg.lambda_mse * mse
        + cfg.lambda_lpips * lpips_term
        + cfg.lambda_mask * mask_term
    )
    return total, {
        "mse": float(mse.detach()),
        "lpips": float(lpips_term.detach()),
        "mask": float(mask_term.detach()),
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


def ssim3d_occupancy_loss(
    alpha: torch.Tensor, hull: torch.Tensor, observed: torch.Tensor,
    res: int, win_size: int, sigma: float,
) -> torch.Tensor:
    """1 - SSIM3D between predicted occupancy and the visual hull, both
    (res**3,) flattened from _grid_positions' (x, y, z) meshgrid order.

    ssim3d has no masking primitive of its own - it is a dense conv over the
    whole grid - so unobserved voxels get their target replaced by the
    (detached) prediction: locally pred == target there, contributing ~zero
    loss and no gradient, instead of teaching the model to match hull=1
    (which at an unobserved voxel means only "no view ruled it out", not
    "occupied" - see visual_hull_grid's docstring).
    """
    target = torch.where(observed, hull.float(), alpha.detach())
    shape = (1, 1, res, res, res)
    return 1.0 - ssim3d_score(
        alpha.reshape(shape), target.reshape(shape),
        data_range=1.0, win_size=win_size, sigma=sigma,
        padding="same", pad_mode="replicate",
    )


_LAPLACIAN_KERNEL_3D = None  # built lazily, cached per (device, dtype)
_laplacian_kernel_cache: dict = {}


def _laplacian_kernel(device, dtype) -> torch.Tensor:
    key = (device, dtype)
    if key not in _laplacian_kernel_cache:
        # standard 7-point discrete Laplacian stencil: center -6, each of the
        # 6 face-adjacent neighbors +1. Zero everywhere else.
        k = torch.zeros(1, 1, 3, 3, 3, device=device, dtype=dtype)
        k[0, 0, 1, 1, 1] = -6.0
        k[0, 0, 0, 1, 1] = k[0, 0, 2, 1, 1] = 1.0
        k[0, 0, 1, 0, 1] = k[0, 0, 1, 2, 1] = 1.0
        k[0, 0, 1, 1, 0] = k[0, 0, 1, 1, 2] = 1.0
        _laplacian_kernel_cache[key] = k
    return _laplacian_kernel_cache[key]


def laplacian_smoothness_loss(alpha: torch.Tensor, res: int) -> torch.Tensor:
    """Mean squared discrete Laplacian of the occupancy grid - the voxel-grid
    analogue of mesh Laplacian smoothing (Desbrun et al. 1999), penalizing
    sharp local jumps in occupancy without needing a differentiable mesh
    extraction (training never extracts one - marching cubes only runs at
    eval/inference time, via skimage, which is not differentiable).

    Unlike ssim3d_occupancy_loss/soft_dice_loss this has no target - it is a
    pure regularizer on the prediction itself, so it needs no hull/observed
    masking; an unobserved voxel is smoothed the same as an observed one.
    """
    kernel = _laplacian_kernel(alpha.device, alpha.dtype)
    grid = alpha.reshape(1, 1, res, res, res)
    # replicate padding, not conv3d's zero-padding default: a constant field
    # has zero Laplacian everywhere, but zero-padding would fabricate a drop
    # to 0 just outside the grid, scoring a fake edge at every boundary voxel
    # (same reasoning as sobel_sketch/ssim3d._blur3d elsewhere in this repo).
    grid = F.pad(grid, (1, 1, 1, 1, 1, 1), mode="replicate")
    lap = F.conv3d(grid, kernel)
    return (lap ** 2).mean()


def occupancy_3d_terms(
    model, code_leaf, masks, c2ws, gt_size: int, cfg: StarXConfig, device: str,
):
    """3D-volume terms against the visual hull: soft-Dice, SSIM3D, and/or a
    Laplacian smoothness regularizer, whichever cfg weights (any subset,
    including none). They share one query_alpha_grid call, so scoring all
    three costs one extra loss evaluation rather than a second pass through
    the triplane.

    `masks`/`c2ws` must cover the design's FULL camera rig, not just the
    views used for the 2D photometric loss - a partial rig would carve a
    hull from evidence the model was never asked to match.
    """
    with torch.no_grad():
        hull, observed = visual_hull_grid(masks, c2ws, cfg.hull_res, gt_size, device)
    alpha = query_alpha_grid(model, code_leaf, cfg.hull_res)
    loss = torch.zeros((), device=device)
    parts = {"occ": 0.0, "ssim3d": 0.0, "laplacian": 0.0}
    if cfg.lambda_occ > 0:
        dice = soft_dice_loss(alpha[observed], hull[observed])
        loss = loss + cfg.lambda_occ * dice
        parts["occ"] = float(dice.detach())
    if cfg.lambda_ssim3d > 0:
        ssim_term = ssim3d_occupancy_loss(
            alpha, hull, observed, cfg.hull_res, cfg.ssim3d_win_size, cfg.ssim3d_sigma
        )
        loss = loss + cfg.lambda_ssim3d * ssim_term
        parts["ssim3d"] = float(ssim_term.detach())
    if cfg.lambda_laplacian > 0:
        lap_term = laplacian_smoothness_loss(alpha, cfg.hull_res)
        loss = loss + cfg.lambda_laplacian * lap_term
        parts["laplacian"] = float(lap_term.detach())
    return loss, parts


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
    totals = {"loss": 0.0, "mse": 0.0, "lpips": 0.0, "mask": 0.0,
              "occ": 0.0, "ssim3d": 0.0, "laplacian": 0.0}

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

        if cfg.lambda_occ > 0 or cfg.lambda_ssim3d > 0 or cfg.lambda_laplacian > 0:
            loss3d, parts3d = occupancy_3d_terms(
                model, code_leaf, item["masks"], c2ws, cfg.gt_size, cfg, device
            )
            (loss3d / cfg.accum_designs).backward()
            for key, val in parts3d.items():
                totals[key] += val / cfg.accum_designs
            totals["loss"] += float(loss3d) / cfg.accum_designs

        # stage two: the accumulated scene-code gradient through the encoder
        scene_code.backward(gradient=code_leaf.grad * grad_scale)

    if world_size > 1:
        import torch.distributed as dist

        for p in trainable_params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        stats = torch.tensor(
            [totals[k] for k in ("loss", "mse", "lpips", "mask", "occ", "ssim3d", "laplacian")],
            device=device,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        for i, k in enumerate(("loss", "mse", "lpips", "mask", "occ", "ssim3d", "laplacian")):
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


# Gradual unfreezing, outermost layers first. At step zero the decoder is
# being asked to explain a kind of image it has never seen, so its gradients
# are large and noisy; letting those reach DINO immediately is how a
# pretrained encoder gets wrecked in the first few hundred steps. Fractions
# are of the total step budget.
UNFREEZE_STAGES = (
    {"at": 0.00, "name": "head", "groups": ("decoder", "post_processor")},
    {"at": 0.10, "name": "+triplane", "groups": ("tokenizer",)},
    {"at": 0.25, "name": "+backbone", "groups": ("backbone",)},
    {"at": 0.55, "name": "+encoder", "groups": ("image_tokenizer",)},
)

# How far each group is allowed to move: the deeper into pretraining it
# sits, the smaller its share of the learning rate.
LR_SCALE = {
    "decoder": 1.0,
    "post_processor": 1.0,
    "tokenizer": 0.5,
    "backbone": 0.5,
    "image_tokenizer": 0.1,
}


def stage_boundaries(total_steps: int, stages=UNFREEZE_STAGES) -> list:
    return [int(s["at"] * total_steps) for s in stages]


def apply_unfreeze_stage(model, step: int, total_steps: int, stages=UNFREEZE_STAGES):
    """Freeze everything, then unlock every group whose stage has arrived.

    Returns the active stage name. Idempotent, so calling it every step is
    free, and calling it before loading a checkpoint rebuilds exactly the
    trainable set that checkpoint was written with.
    """
    bounds = stage_boundaries(total_steps, stages)
    active = [s for s, b in zip(stages, bounds) if step >= b] or [stages[0]]
    for param in model.parameters():
        param.requires_grad_(False)
    for stage in active:
        for group in stage["groups"]:
            getattr(model, group).requires_grad_(True)
    return active[-1]["name"]


def build_finetune_optimizer(model, cfg: StarXConfig, total_steps: int,
                             lr_scale=None):
    """AdamW with per-group learning rates, warmup into cosine.

    Every parameter goes into the optimizer up front, including ones that
    are frozen right now: AdamW skips a param whose grad is None, so a group
    simply starts contributing on the step its stage opens, with its moments
    initialized fresh at that point. Weight decay skips biases and norms.
    """
    lr_scale = lr_scale or LR_SCALE
    groups = []
    for name, scale in lr_scale.items():
        module = getattr(model, name)
        decay = [p for p in module.parameters() if p.ndim > 1]
        plain = [p for p in module.parameters() if p.ndim <= 1]
        if decay:
            groups.append({"params": decay, "lr": cfg.lr * scale,
                           "weight_decay": cfg.weight_decay, "name": name})
        if plain:
            groups.append({"params": plain, "lr": cfg.lr * scale,
                           "weight_decay": 0.0, "name": f"{name}.nodecay"})
    optimizer = torch.optim.AdamW(groups, lr=cfg.lr, betas=tuple(cfg.adam_betas))

    def lr_lambda(step):
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_paper_optimizer(model, cfg: StarXConfig, total_steps: int):
    """AdamW + warmup-into-cosine, as close to TripoSR's report as it gets.

    What the report actually states: AdamW, lr 4e-4, CosineAnnealingLR,
    2,000 warm-up steps. It is silent on weight decay and betas, so those
    default to LRM's, which TripoSR inherits everything else from: decay
    0.05 applied only to non-bias, non-LayerNorm parameters, betas
    (0.9, 0.95). Cosine anneals to zero here rather than to a floor - the
    baseline's schedule keeps 0.1x, which is a StarX choice, not the paper's.

    The learning rate is the one number to think twice about: 4e-4 is a
    from-scratch pretraining rate for ~950k objects at batch 1024. Fine-
    tuning a converged checkpoint on a few thousand designs at that rate
    will move it a long way from where pretraining left it.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # biases and every normalization scale/shift stay undecayed
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=tuple(cfg.adam_betas),
    )

    def lr_lambda(step):
        if step < cfg.warmup_steps:
            return (step + 1) / cfg.warmup_steps
        progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def paper_train_step(
    batch: dict,
    model,
    optimizer,
    scheduler,
    trainable_params,
    lpips_metric,
    cfg: StarXConfig,
    device: str,
    amp_dtype,
    rng: np.random.Generator,
    grad_scale: float = 1.0,
    world_size: int = 1,
):
    """One optimizer step over a DataLoader batch of (design, input view) pairs.

    The recipe is the paper's: encode the batch's drawings, render a
    128x128 foreground-biased crop per supervision view, and score it with
    MSE + 2.0 LPIPS + 0.05 mask BCE. Memory discipline is unchanged from
    the baseline - each view backpropagates only to a detached scene-code
    leaf, then one encoder backward per design carries the accumulated
    gradient, which is the deferred backpropagation LRM describes.

    When cfg.lambda_occ, cfg.lambda_ssim3d, or cfg.lambda_laplacian is set
    (off by default - not in the paper), an additional 3D term scores the
    triplane's occupancy grid against the design's visual hull (or, for the
    Laplacian term, just the occupancy grid's own smoothness), same as
    starx.train.train_step. The hull-based terms need the design's FULL
    camera rig, not just the supervised views, so the batch must come from a
    SketchDataset built with return_all_masks=True.
    """
    from starx import model as smodel

    model.train()
    inputs = batch["input"].to(device)  # (B, 3, S, S)
    batch_size = inputs.shape[0]
    n_terms = batch_size * batch["views"].shape[1]
    totals = {"loss": 0.0, "mse": 0.0, "lpips": 0.0, "mask": 0.0,
              "occ": 0.0, "ssim3d": 0.0, "laplacian": 0.0}
    use_3d = cfg.lambda_occ > 0 or cfg.lambda_ssim3d > 0 or cfg.lambda_laplacian > 0
    if use_3d and "all_masks" not in batch:
        raise ValueError(
            "cfg.lambda_occ / cfg.lambda_ssim3d / cfg.lambda_laplacian is set "
            "but the batch has no 'all_masks' - build the SketchDataset with "
            "return_all_masks=True"
        )

    with torch.autocast("cuda", dtype=amp_dtype, enabled=device == "cuda"):
        scene_codes = smodel.encode_sketches(model, inputs)
    scene_codes = scene_codes.float()

    # one encoder backward for the whole batch: each design's render graph is
    # freed as it goes, its scene-code gradient parked here until the end
    code_grads = torch.zeros_like(scene_codes)
    for b in range(batch_size):
        code_leaf = scene_codes[b].detach().requires_grad_(True)
        views = batch["views"][b].numpy()
        masks = batch["masks"][b].numpy()
        c2ws = batch["c2ws"][b].numpy()
        for v in range(views.shape[0]):
            box = data.sample_crop_box(masks[v], cfg.render_crop, rng)
            top, left, h, w = box
            rays_o, rays_d = cameras.rays_for_crop(
                c2ws[v], FOVY_DEG, views.shape[1], box
            )
            rgb_fg, opacity = render_rays(
                model, code_leaf, rays_o.to(device), rays_d.to(device)
            )
            gt_rgb = torch.from_numpy(
                np.ascontiguousarray(views[v][top : top + h, left : left + w])
            ).float().to(device) / 255.0
            gt_mask = torch.from_numpy(
                np.ascontiguousarray(masks[v][top : top + h, left : left + w])
            ).to(device)
            loss, parts = compute_render_loss(
                rgb_fg, opacity, gt_rgb, gt_mask, lpips_metric, cfg
            )
            (loss / n_terms).backward()
            totals["loss"] += float(loss.detach()) / n_terms
            for key in ("mse", "lpips", "mask"):
                totals[key] += parts[key] / n_terms

        if use_3d:
            all_masks = batch["all_masks"][b].numpy()
            all_c2ws = batch["all_c2ws"][b].numpy()
            loss3d, parts3d = occupancy_3d_terms(
                model, code_leaf, all_masks, all_c2ws, views.shape[1], cfg, device
            )
            (loss3d / batch_size).backward()
            for key, val in parts3d.items():
                totals[key] += val / batch_size
            totals["loss"] += float(loss3d) / batch_size

        code_grads[b] = code_leaf.grad

    scene_codes.backward(gradient=code_grads * grad_scale)

    if world_size > 1:
        import torch.distributed as dist

        # each rank's DataLoader gave it a DIFFERENT batch, and each rank's
        # loss is already a mean over its own - so the reduction must AVERAGE,
        # not sum, or the effective learning rate scales with the GPU count.
        # (starx.train.train_step sums instead, correctly: there the ranks
        # split one step's designs and each divides by the full count.)
        for p in trainable_params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(world_size)
        stats = torch.tensor(
            [totals[k] for k in ("loss", "mse", "lpips", "mask", "occ", "ssim3d", "laplacian")],
            device=device,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        for i, k in enumerate(("loss", "mse", "lpips", "mask", "occ", "ssim3d", "laplacian")):
            totals[k] = float(stats[i]) / world_size

    if grad_scale != 1.0:
        for p in trainable_params:
            if p.grad is not None:
                p.grad.div_(grad_scale)
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
    totals["skipped"] = 0.0
    if torch.isfinite(grad_norm):
        optimizer.step()
    else:
        totals["skipped"] = 1.0
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
    """(amp_dtype, grad_scale) matched to the GPU: bf16 where the hardware
    has it, scaled fp16 everywhere else.

    Ask the compute capability directly rather than torch.cuda.is_bf16_-
    supported(), whose default is including_emulation=True: on a Turing T4
    that answers True, because bf16 runs there - emulated, and slow. Native
    bf16 starts at capability 8.0 (Ampere: A100, L4, and newer), which is
    exactly the line the fp16 path was written for.
    """
    if device.startswith("cuda") and torch.cuda.is_available():
        index = torch.device(device).index or 0
        if torch.cuda.get_device_capability(index)[0] >= 8:
            return torch.bfloat16, 1.0
    return torch.float16, 4096.0
