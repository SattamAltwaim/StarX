"""Headless ground-truth rendering with pyrender.

EGL (GPU) is tried first; OSMesa (CPU) is the fallback. The platform must be
chosen via the PYOPENGL_PLATFORM environment variable before pyrender's
first import, so switching backends purges and re-imports the GL modules.

The light rig and material are fixed constants: ground truth must be a pure
function of (mesh, camera) or training targets drift between sessions.
"""

from __future__ import annotations

import importlib
import math
import os
import sys

import numpy as np

from starx.cameras import c2w_to_pyrender_pose
from starx.config import FOVY_DEG

# key / fill / back directional lights: (azimuth deg, elevation deg, intensity)
LIGHT_RIG = [(45.0, 45.0, 3.0), (-70.0, 10.0, 1.5), (180.0, -30.0, 1.0)]
AMBIENT = [0.25, 0.25, 0.25]
MESH_COLOR = [0.75, 0.75, 0.78, 1.0]


def _import_pyrender(platform: str):
    for module_name in list(sys.modules):
        if module_name.startswith(("pyrender", "OpenGL")):
            del sys.modules[module_name]
    os.environ["PYOPENGL_PLATFORM"] = platform
    return importlib.import_module("pyrender")


def normalize_mesh(mesh):
    """Center the bbox at the origin and scale the max extent to 1.0.

    Returns (normalized copy, center, scale) so the transform is invertible;
    the result fits the TripoSR scene sphere (radius 0.87 > half-diagonal).
    """
    mesh = mesh.copy()
    lo, hi = mesh.bounds
    center = (lo + hi) / 2.0
    extent = float((hi - lo).max())
    scale = 1.0 / extent if extent > 0 else 1.0
    mesh.apply_translation(-center)
    mesh.apply_scale(scale)
    return mesh, center, scale


class GTRenderer:
    """Offscreen renderer producing shaded RGB + boolean masks."""

    def __init__(self, size: int, backends=("egl", "osmesa")):
        self.size = size
        self.backend = None
        last_error = None
        for backend in backends:
            try:
                self._pyrender = _import_pyrender(backend)
                self._renderer = self._pyrender.OffscreenRenderer(size, size)
                self._smoke_test()
                self.backend = backend
                break
            except Exception as error:  # EGL init failures vary by driver
                last_error = error
        if self.backend is None:
            raise RuntimeError(f"no working pyrender backend: {last_error}")

    def _smoke_test(self):
        import trimesh

        box = trimesh.creation.box(extents=(0.5, 0.5, 0.5))
        from starx.cameras import build_spherical_c2w

        rgb, mask = self.render_mesh(box, [build_spherical_c2w(30.0, 20.0, 1.9)])
        if not mask.any():
            raise RuntimeError("smoke render produced an empty mask")

    def render_mesh(self, mesh, c2w_list, fovy_deg: float = FOVY_DEG):
        """Render at each c2w: (V, H, W, 3) uint8 RGB + (V, H, W) bool mask."""
        pyrender = self._pyrender
        scene = pyrender.Scene(
            ambient_light=AMBIENT, bg_color=[255, 255, 255, 255]
        )
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=MESH_COLOR, metallicFactor=0.2, roughnessFactor=0.7
        )
        scene.add(
            pyrender.Mesh.from_trimesh(mesh, material=material, smooth=False)
        )
        for azimuth, elevation, intensity in LIGHT_RIG:
            light = pyrender.DirectionalLight(intensity=intensity)
            scene.add(light, pose=_light_pose(azimuth, elevation))
        camera = pyrender.PerspectiveCamera(yfov=math.radians(fovy_deg))
        camera_node = scene.add(camera, pose=np.eye(4))

        rgbs, masks = [], []
        for c2w in c2w_list:
            scene.set_pose(camera_node, c2w_to_pyrender_pose(c2w))
            color, depth = self._renderer.render(scene)
            rgbs.append(color[..., :3].astype(np.uint8))
            masks.append(depth > 0)
        return np.stack(rgbs), np.stack(masks)

    def close(self):
        self._renderer.delete()


def _light_pose(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    from starx.cameras import build_spherical_c2w

    return c2w_to_pyrender_pose(build_spherical_c2w(azimuth_deg, elevation_deg, 4.0))
