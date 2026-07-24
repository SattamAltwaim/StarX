"""Parsing of Fusion 360 Gallery reconstruction JSON files.

A design JSON stores a modeling timeline: Sketch entities (2D points and
curves on a plane) alternating with ExtrudeFeature entities. This module
extracts the sketches in timeline order and turns every drawable curve into
a polyline in the sketch's local XY coordinate system (units: centimeters).

Field names are grounded on the dataset docs and the test fixture:
curves reference `points` by UUID, carry a `construction_geom` flag, and
arcs define their sweep by `reference_vector` + `start_angle`/`end_angle`
around `center_point`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CIRCLE_SEGMENTS = 128
SPLINE_SAMPLES_PER_SEGMENT = 16
_ENDPOINT_TOL = 1e-3  # cm; used to pick the arc sweep orientation


@dataclass
class Sketch:
    name: str
    timeline_index: int
    points: dict
    curves: dict
    transform: dict
    reference_plane: dict
    profiles: dict = field(default_factory=dict)


@dataclass
class Design:
    design_id: str
    sketches: list
    n_extrudes: int
    warnings: list = field(default_factory=list)
    raw: dict = None


def load_design(source) -> Design:
    """Parse a reconstruction JSON (path or already-loaded dict) into a Design."""
    if isinstance(source, (str, Path)):
        design_id = Path(source).stem
        with open(source) as f:
            raw = json.load(f)
    else:
        raw = source
        design_id = raw.get("metadata", {}).get("parent_project", "design")

    entities = raw.get("entities", {})
    timeline = raw.get("timeline") or [
        {"index": i, "entity": uid} for i, uid in enumerate(entities)
    ]

    sketches, n_extrudes, warnings = [], 0, []
    for item in sorted(timeline, key=lambda t: t.get("index", 0)):
        entity = entities.get(item.get("entity"))
        if entity is None:
            warnings.append(f"timeline entry {item.get('index')}: missing entity")
            continue
        etype = entity.get("type")
        if etype == "Sketch":
            sketches.append(
                Sketch(
                    name=entity.get("name", f"Sketch{len(sketches) + 1}"),
                    timeline_index=item.get("index", len(sketches)),
                    points=entity.get("points", {}),
                    curves=entity.get("curves", {}),
                    transform=entity.get("transform", {}),
                    reference_plane=entity.get("reference_plane", {}),
                    profiles=entity.get("profiles", {}),
                )
            )
        elif etype == "ExtrudeFeature":
            n_extrudes += 1

    return Design(design_id, sketches, n_extrudes, warnings, raw)


def _resolve_point(ref, points) -> np.ndarray:
    """A point reference is either a UUID into `points` or an inline Point3D."""
    if isinstance(ref, str):
        ref = points[ref]
    return np.array([ref["x"], ref["y"]], dtype=np.float64)


def _vector2(v) -> np.ndarray:
    return np.array([v["x"], v["y"]], dtype=np.float64)


def _arc_points(curve, points) -> np.ndarray:
    """Sample an arc swept from start_angle to end_angle around center_point.

    Angle zero points along reference_vector; the sweep direction is not
    stated in the JSON, so both orientations are generated and the one whose
    endpoints match the referenced start/end points is kept.
    """
    center = _resolve_point(curve["center_point"], points)
    radius = float(curve["radius"])
    ref = _vector2(curve["reference_vector"])
    norm = np.linalg.norm(ref)
    if norm > 0:
        ref = ref / norm
    a0, a1 = float(curve["start_angle"]), float(curve["end_angle"])
    sweep = a1 - a0
    n = max(8, int(math.ceil(CIRCLE_SEGMENTS * abs(sweep) / (2 * math.pi))))
    angles = a0 + np.linspace(0.0, 1.0, n + 1) * sweep

    candidates = []
    for perp in (np.array([-ref[1], ref[0]]), np.array([ref[1], -ref[0]])):
        pts = (
            center
            + radius * np.outer(np.cos(angles), ref)
            + radius * np.outer(np.sin(angles), perp)
        )
        candidates.append(pts)

    start = curve.get("start_point")
    end = curve.get("end_point")
    if start is not None and end is not None:
        s, e = _resolve_point(start, points), _resolve_point(end, points)
        errs = [
            np.linalg.norm(c[0] - s) + np.linalg.norm(c[-1] - e) for c in candidates
        ]
        return candidates[int(np.argmin(errs))]
    return candidates[0]


def _circle_points(curve, points) -> np.ndarray:
    center = _resolve_point(curve["center_point"], points)
    radius = float(curve["radius"])
    angles = np.linspace(0.0, 2 * math.pi, CIRCLE_SEGMENTS + 1)
    return center + radius * np.stack([np.cos(angles), np.sin(angles)], axis=1)


def _ellipse_points(curve, points) -> np.ndarray:
    center = _resolve_point(curve["center_point"], points)
    major = _vector2(curve["major_axis"])
    norm = np.linalg.norm(major)
    if norm > 0:
        major = major / norm
    minor = np.array([-major[1], major[0]])
    ra = float(curve["major_axis_radius"])
    rb = float(curve["minor_axis_radius"])
    a0 = float(curve.get("start_angle", 0.0))
    a1 = float(curve.get("end_angle", 2 * math.pi))
    sweep = a1 - a0 if a1 != a0 else 2 * math.pi
    n = max(8, int(math.ceil(CIRCLE_SEGMENTS * abs(sweep) / (2 * math.pi))))
    angles = a0 + np.linspace(0.0, 1.0, n + 1) * sweep
    return (
        center
        + ra * np.outer(np.cos(angles), major)
        + rb * np.outer(np.sin(angles), minor)
    )


def _catmull_rom(pts: np.ndarray, samples_per_seg: int) -> np.ndarray:
    """Centripetal-free Catmull-Rom through the given points (numpy only)."""
    P = np.asarray(pts, dtype=np.float64)
    if len(P) < 3:
        return P
    ext = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    t = np.linspace(0.0, 1.0, samples_per_seg, endpoint=False)[:, None]
    out = []
    for i in range(len(P) - 1):
        p0, p1, p2, p3 = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        out.append(
            0.5
            * (
                2 * p1
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3
            )
        )
    out.append(P[-1:])
    return np.vstack(out)


def _deboor(u: float, knots: np.ndarray, ctrl: np.ndarray, degree: int) -> np.ndarray:
    """De Boor's algorithm for one parameter value (homogeneous coords ok)."""
    k = int(np.searchsorted(knots, u, side="right") - 1)
    k = min(max(k, degree), len(ctrl) - 1)
    d = [ctrl[j].copy() for j in range(k - degree, k + 1)]
    for r in range(1, degree + 1):
        for j in range(degree, r - 1, -1):
            i = j + k - degree
            denom = knots[i + degree - r + 1] - knots[i]
            alpha = 0.0 if denom == 0 else (u - knots[i]) / denom
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]
    return d[degree]


def _bspline_points(curve, points) -> np.ndarray | None:
    """Sample a spline: prefer fit points (Catmull-Rom), else NURBS via de Boor."""
    fit = curve.get("fit_points")
    if fit:
        pts = np.array([_resolve_point(p, points) for p in fit])
        return _catmull_rom(pts, SPLINE_SAMPLES_PER_SEGMENT)

    ctrl_raw = curve.get("control_points")
    knots = curve.get("knots")
    degree = curve.get("degree")
    if ctrl_raw and knots is not None and degree is not None:
        ctrl = np.array([_resolve_point(p, points) for p in ctrl_raw])
        knots = np.asarray(knots, dtype=np.float64)
        weights = curve.get("weights")
        if weights is not None and curve.get("rational", True):
            w = np.asarray(weights, dtype=np.float64)[:, None]
            ctrl_h = np.hstack([ctrl * w, w])
        else:
            ctrl_h = np.hstack([ctrl, np.ones((len(ctrl), 1))])
        degree = int(degree)
        lo, hi = knots[degree], knots[len(ctrl)]
        n = max(32, len(ctrl) * SPLINE_SAMPLES_PER_SEGMENT)
        us = np.linspace(lo, hi - 1e-9 * max(1.0, abs(hi)), n)
        samples = np.array([_deboor(u, knots, ctrl_h, degree) for u in us])
        return samples[:, :2] / samples[:, 2:3]
    return None


def sample_curve(curve: dict, points: dict, include_construction: bool = False):
    """One curve to an (N, 2) polyline in sketch-local cm; None if skipped."""
    if curve.get("construction_geom") and not include_construction:
        return None
    ctype = curve.get("type")
    try:
        if ctype == "SketchLine":
            return np.stack(
                [
                    _resolve_point(curve["start_point"], points),
                    _resolve_point(curve["end_point"], points),
                ]
            )
        if ctype == "SketchArc":
            return _arc_points(curve, points)
        if ctype == "SketchCircle":
            return _circle_points(curve, points)
        if ctype in ("SketchEllipse", "SketchEllipticalArc"):
            return _ellipse_points(curve, points)
        if ctype == "SketchFittedSpline":
            return _bspline_points(curve, points)
    except (KeyError, TypeError, ValueError):
        return None
    return None


def sketch_polylines(
    sketch: Sketch, include_construction: bool = False, warnings: list = None
) -> list:
    """All drawable polylines of one sketch; unknown/broken curves are recorded."""
    polylines = []
    for cid, curve in sketch.curves.items():
        poly = sample_curve(curve, sketch.points, include_construction)
        if poly is not None and len(poly) >= 2:
            polylines.append(poly)
        elif warnings is not None and not (
            curve.get("construction_geom") and not include_construction
        ):
            warnings.append(f"{sketch.name}/{cid}: unsupported {curve.get('type')}")
    return polylines


def sketch_bbox(polylines: list):
    """(min_xy, max_xy) over a list of polylines; None when nothing drawable."""
    if not polylines:
        return None
    stacked = np.vstack(polylines)
    return stacked.min(axis=0), stacked.max(axis=0)
