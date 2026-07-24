"""Plot helpers for the notebooks.

Matplotlib for grids and curves (tab10 palette per house convention),
plotly Mesh3d for interactive GT-vs-predicted mesh comparison, imageio for
turntable GIFs. Every function returns the figure so notebooks stay
one-operation-per-cell.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def show_sketch_stack(stack: np.ndarray, meta: dict = None, title: str = None):
    """Contact sheet of a design's sketch channels, blank slots labeled."""
    n_channels = stack.shape[0]
    fig, axes = plt.subplots(1, n_channels, figsize=(2.2 * n_channels, 2.6))
    axes = np.atleast_1d(axes)
    blank = (meta or {}).get("blank_channels", [None] * n_channels)
    n_sketches = (meta or {}).get("n_sketches_total")
    for i, ax in enumerate(axes):
        ax.imshow(stack[i], cmap="gray", vmin=0, vmax=255)
        label = f"channel {i}"
        if blank[i]:
            label += " (blank)" if n_sketches is None or i < n_sketches else " (pad)"
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def show_view_grid(rgbs, masks=None, titles=None, cols: int = 8):
    """Grid of rendered views; optional second row block of masks."""
    n = len(rgbs)
    rows = int(np.ceil(n / cols)) * (2 if masks is not None else 1)
    fig, axes = plt.subplots(rows, cols, figsize=(1.8 * cols, 1.9 * rows))
    axes = np.atleast_2d(axes)
    mask_row_offset = int(np.ceil(n / cols))
    for i in range(n):
        r, c = divmod(i, cols)
        axes[r, c].imshow(rgbs[i])
        if titles is not None:
            axes[r, c].set_title(str(titles[i]), fontsize=8)
        if masks is not None:
            axes[mask_row_offset + r, c].imshow(masks[i], cmap="gray")
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    return fig


def _mesh3d_trace(mesh, color: str, name: str):
    import plotly.graph_objects as go

    v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    return go.Mesh3d(
        x=v[:, 0], y=v[:, 1], z=v[:, 2],
        i=f[:, 0], j=f[:, 1], k=f[:, 2],
        color=color, opacity=1.0, name=name, flatshading=True,
        lighting=dict(ambient=0.45, diffuse=0.8, specular=0.15),
    )


def mesh_side_by_side(gt_mesh, pred_mesh, title: str = ""):
    """Interactive plotly figure: ground truth | prediction, shared camera."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "mesh3d"}, {"type": "mesh3d"}]],
        subplot_titles=("ground truth", "prediction"),
    )
    if gt_mesh is not None:
        fig.add_trace(_mesh3d_trace(gt_mesh, "#4c78a8", "ground truth"), 1, 1)
    if pred_mesh is not None:
        fig.add_trace(_mesh3d_trace(pred_mesh, "#f58518", "prediction"), 1, 2)
    camera = dict(eye=dict(x=1.4, y=1.4, z=0.9))
    fig.update_scenes(aspectmode="data", camera=camera)
    fig.update_layout(title=title, height=420, margin=dict(l=0, r=0, t=60, b=0))
    return fig


def turntable_gif(frames, path, fps: int = 12):
    """Write frames (list of HxWx3 uint8) as a looping GIF; returns path."""
    import imageio.v2 as imageio

    imageio.mimsave(path, list(frames), fps=fps, loop=0)
    return path


def plot_loss_curves(df, keys=("loss", "mse", "lpips", "mask"), logy: bool = True):
    """Per-term training curves from a read_log dataframe."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = plt.get_cmap("tab10").colors
    for i, key in enumerate(keys):
        if key in df.columns:
            ax.plot(df["step"], df[key], label=key, color=colors[i], linewidth=1.2)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def crop_debug_figure(view_rgb, mask, crop_box):
    """Full view, mask overlay, and the sampled crop rectangle."""
    top, left, h, w = crop_box
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    axes[0].imshow(view_rgb)
    axes[0].set_title("ground-truth view", fontsize=9)
    axes[1].imshow(view_rgb)
    axes[1].imshow(mask, cmap="Reds", alpha=0.35)
    axes[1].set_title("foreground mask", fontsize=9)
    axes[2].imshow(view_rgb)
    rect = plt.Rectangle((left, top), w, h, fill=False, edgecolor="#e45756", lw=2)
    axes[2].add_patch(rect)
    axes[2].set_title("sampled crop", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return fig
