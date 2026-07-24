import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd

from starx import viz


def test_sketch_stack_figure():
    stack = np.full((4, 32, 32), 128, dtype=np.uint8)
    meta = {"blank_channels": [False, False, True, True], "n_sketches_total": 2}
    fig = viz.show_sketch_stack(stack, meta, title="fixture")
    assert len(fig.axes) == 4


def test_view_grid_with_masks():
    rgbs = [np.zeros((16, 16, 3), dtype=np.uint8)] * 5
    masks = [np.ones((16, 16), dtype=bool)] * 5
    fig = viz.show_view_grid(rgbs, masks, cols=4)
    assert fig is not None


def test_loss_curves():
    df = pd.DataFrame(
        {"step": [1, 2, 3], "loss": [3.0, 2.0, 1.0], "mse": [1.0, 0.7, 0.4]}
    )
    fig = viz.plot_loss_curves(df)
    assert fig.axes[0].get_yscale() == "log"


def test_crop_debug_figure():
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=bool)
    fig = viz.crop_debug_figure(rgb, mask, (8, 8, 16, 16))
    assert len(fig.axes) == 3


def test_mesh_side_by_side_builds():
    import trimesh

    sphere = trimesh.creation.icosphere(subdivisions=1, radius=0.5)
    fig = viz.mesh_side_by_side(sphere, sphere, title="toy")
    assert len(fig.data) == 2
