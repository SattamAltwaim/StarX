"""3D and 2D evaluation metrics, dependency-light by design.

Chamfer distance and F-score run on surface point samples with chunked
torch.cdist (no kaolin/pytorch3d). Volumetric IoU voxelizes surface samples,
dilates one voxel and fills holes - it never calls mesh.contains, because
the dataset's OBJ triangulations are not guaranteed manifold. Mesh
extraction queries the triplane density on a grid and runs skimage's
marching cubes, so the torchmcubes CUDA extension is never needed.
"""

from __future__ import annotations

import numpy as np
import torch

from starx.config import SCENE_RADIUS


def sample_surface_points(mesh, n: int, seed: int = 0) -> np.ndarray:
    """Area-weighted surface samples, (n, 3) float32."""
    import trimesh

    points, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(points, dtype=np.float32)


def _nn_dists(a: torch.Tensor, b: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """For each point in a: euclidean distance to its nearest neighbor in b."""
    dists = []
    for i in range(0, a.shape[0], chunk):
        d = torch.cdist(a[i : i + chunk], b)
        dists.append(d.min(dim=1).values)
    return torch.cat(dists)


def chamfer_distance(p1, p2, chunk: int = 4096) -> float:
    """Symmetric Chamfer: mean of the two directional mean NN distances."""
    a = torch.as_tensor(np.asarray(p1), dtype=torch.float32)
    b = torch.as_tensor(np.asarray(p2), dtype=torch.float32)
    return 0.5 * (
        _nn_dists(a, b, chunk).mean().item() + _nn_dists(b, a, chunk).mean().item()
    )


def fscore(p1, p2, tau: float, chunk: int = 4096):
    """(F, precision, recall) at distance threshold tau."""
    a = torch.as_tensor(np.asarray(p1), dtype=torch.float32)
    b = torch.as_tensor(np.asarray(p2), dtype=torch.float32)
    precision = (_nn_dists(a, b, chunk) < tau).float().mean().item()
    recall = (_nn_dists(b, a, chunk) < tau).float().mean().item()
    if precision + recall == 0:
        return 0.0, precision, recall
    return 2 * precision * recall / (precision + recall), precision, recall


def occupancy_grid(
    points: np.ndarray, res: int, radius: float = SCENE_RADIUS
) -> np.ndarray:
    """Manifold-free occupancy: voxelize surface samples over [-radius,
    radius]^3, dilate one voxel to close small gaps, then fill holes."""
    from scipy import ndimage

    grid = np.zeros((res, res, res), dtype=bool)
    idx = np.floor((points + radius) / (2 * radius) * res).astype(int)
    idx = np.clip(idx, 0, res - 1)
    grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    grid = ndimage.binary_dilation(grid, iterations=1)
    return ndimage.binary_fill_holes(grid)


def voxel_iou(g1: np.ndarray, g2: np.ndarray) -> float:
    union = np.logical_or(g1, g2).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(g1, g2).sum() / union)


def render_opacity(model, scene_code: torch.Tensor, rays_o, rays_d, ray_chunk: int = 16384):
    """No-grad accumulated opacity per ray - the honest object mask for a
    scene code. Color thresholds do not work: volume rendering leaves faint
    fog across the whole scene box, so nearly every pixel is slightly
    off-background even where there is no object."""
    from tsr.utils import rays_intersect_bbox

    device = scene_code.device
    shape = rays_o.shape[:-1]
    rays_o = rays_o.reshape(-1, 3).to(device)
    rays_d = rays_d.reshape(-1, 3).to(device)
    renderer = model.renderer
    t_vals = torch.linspace(
        0, 1, renderer.cfg.num_samples_per_ray + 1, device=device
    )
    t_mid = (t_vals[:-1] + t_vals[1:]) / 2.0
    deltas = t_vals[1:] - t_vals[:-1]
    chunks = []
    with torch.no_grad():
        for i in range(0, rays_o.shape[0], ray_chunk):
            ro, rd = rays_o[i : i + ray_chunk], rays_d[i : i + ray_chunk]
            t_near, t_far, valid = rays_intersect_bbox(ro, rd, renderer.cfg.radius)
            t_near, t_far = t_near[valid], t_far[valid]
            z_vals = t_near * (1 - t_mid[None]) + t_far * t_mid[None]
            xyz = ro[valid][:, None, :] + z_vals[..., None] * rd[valid][:, None, :]
            out = renderer.query_triplane(model.decoder, xyz, scene_code)
            alpha = 1 - torch.exp(-deltas * out["density_act"][..., 0])
            transmittance = torch.cat(
                [
                    torch.ones_like(alpha[:, :1]),
                    torch.cumprod(1 - alpha[:, :-1] + 1e-10, dim=-1),
                ],
                dim=-1,
            )
            opacity = torch.zeros(ro.shape[0], device=device)
            opacity[valid] = (alpha * transmittance).sum(-1)
            chunks.append(opacity)
    return torch.cat(chunks).reshape(shape)


def query_density_grid(
    model, scene_code: torch.Tensor, res: int, chunk: int = 131072
) -> np.ndarray:
    """No-grad activated-density values on a res^3 grid over the scene cube."""
    device = scene_code.device
    axis = torch.linspace(-SCENE_RADIUS, SCENE_RADIUS, res, device=device)
    grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)
    positions = grid.reshape(-1, 3)
    previous_chunk = model.renderer.chunk_size
    model.renderer.set_chunk_size(chunk)
    with torch.no_grad():
        density = model.renderer.query_triplane(
            model.decoder, positions, scene_code
        )["density_act"]
    model.renderer.set_chunk_size(previous_chunk)
    return density.reshape(res, res, res).float().cpu().numpy()


def extract_mesh(model, scene_code: torch.Tensor, res: int = 256, threshold: float = 25.0):
    """Marching cubes over the density grid; None when nothing crosses the
    threshold (a genuine failure case worth surfacing, not an exception)."""
    from skimage import measure
    import trimesh

    density = query_density_grid(model, scene_code, res)
    if density.max() <= threshold or density.min() >= threshold:
        return None
    step = 2 * SCENE_RADIUS / (res - 1)
    verts, faces, _, _ = measure.marching_cubes(
        density, level=threshold, spacing=(step, step, step)
    )
    verts = verts - SCENE_RADIUS
    return trimesh.Trimesh(vertices=verts, faces=faces)


def metrics_2d(pred_rgb: torch.Tensor, gt_rgb: torch.Tensor) -> dict:
    """PSNR / SSIM / LPIPS on (B, 3, H, W) float tensors in [0, 1]."""
    from torchmetrics.functional.image import (
        learned_perceptual_image_patch_similarity,
        peak_signal_noise_ratio,
        structural_similarity_index_measure,
    )

    return {
        "psnr": peak_signal_noise_ratio(pred_rgb, gt_rgb, data_range=1.0).item(),
        "ssim": structural_similarity_index_measure(
            pred_rgb, gt_rgb, data_range=1.0
        ).item(),
        "lpips": learned_perceptual_image_patch_similarity(
            pred_rgb, gt_rgb, net_type="vgg", normalize=True
        ).item(),
    }
