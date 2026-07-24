import numpy as np
import pytest

from starx import fusion


def test_load_design_from_dict_uses_explicit_id(fixture_design):
    # the json's own parent_project field lacks the component suffix, so
    # dict loads must accept the id explicitly
    raw = fixture_design.raw
    inferred = fusion.load_design(raw)
    assert inferred.design_id == "20203_7e31e92a"  # wrong for archive lookups
    explicit = fusion.load_design(raw, design_id="20203_7e31e92a_0000")
    assert explicit.design_id == "20203_7e31e92a_0000"


def test_timeline_order_and_counts(fixture_design):
    d = fixture_design
    assert [s.name for s in d.sketches] == ["Sketch1", "Sketch2", "Sketch3", "Sketch4"]
    assert d.n_extrudes == 4
    assert [s.timeline_index for s in d.sketches] == sorted(
        s.timeline_index for s in d.sketches
    )


def test_fixture_curve_type_totals(fixture_design):
    types = {}
    for s in fixture_design.sketches:
        for c in s.curves.values():
            types[c["type"]] = types.get(c["type"], 0) + 1
    assert types == {"SketchLine": 24, "SketchArc": 3}


def test_lines_sample_to_their_endpoints(fixture_design):
    s = fixture_design.sketches[0]
    for curve in s.curves.values():
        if curve["type"] != "SketchLine" or curve.get("construction_geom"):
            continue
        poly = fusion.sample_curve(curve, s.points)
        assert poly.shape == (2, 2)
        start = fusion._resolve_point(curve["start_point"], s.points)
        end = fusion._resolve_point(curve["end_point"], s.points)
        np.testing.assert_allclose(poly[0], start)
        np.testing.assert_allclose(poly[-1], end)


def test_arcs_hit_their_endpoints(fixture_design):
    checked = 0
    for s in fixture_design.sketches:
        for curve in s.curves.values():
            if curve["type"] != "SketchArc":
                continue
            poly = fusion.sample_curve(curve, s.points)
            start = fusion._resolve_point(curve["start_point"], s.points)
            end = fusion._resolve_point(curve["end_point"], s.points)
            radius = curve["radius"]
            assert np.linalg.norm(poly[0] - start) < 1e-6 * max(1.0, radius)
            assert np.linalg.norm(poly[-1] - end) < 1e-6 * max(1.0, radius)
            center = fusion._resolve_point(curve["center_point"], s.points)
            dists = np.linalg.norm(poly - center, axis=1)
            np.testing.assert_allclose(dists, radius, rtol=1e-9)
            checked += 1
    assert checked == 3


def test_construction_geometry_is_skipped(fixture_design):
    # Sketch1 contains a real construction line in the fixture
    s = fixture_design.sketches[0]
    construction = [c for c in s.curves.values() if c.get("construction_geom")]
    assert len(construction) == 1
    curve = construction[0]
    assert fusion.sample_curve(curve, s.points) is None
    assert fusion.sample_curve(curve, s.points, include_construction=True) is not None


def test_unknown_curve_type_is_recorded():
    sketch = fusion.Sketch(
        name="S",
        timeline_index=0,
        points={},
        curves={"c1": {"type": "SketchConicCurve"}},
        transform={},
        reference_plane={},
    )
    warnings = []
    polys = fusion.sketch_polylines(sketch, warnings=warnings)
    assert polys == []
    assert len(warnings) == 1 and "SketchConicCurve" in warnings[0]


def test_circle_sampling_synthetic():
    points = {"c": {"type": "Point3D", "x": 1.0, "y": -2.0, "z": 0.0}}
    curve = {"type": "SketchCircle", "center_point": "c", "radius": 3.0}
    poly = fusion.sample_curve(curve, points)
    np.testing.assert_allclose(poly[0], poly[-1])  # closed
    dists = np.linalg.norm(poly - np.array([1.0, -2.0]), axis=1)
    np.testing.assert_allclose(dists, 3.0)


def test_ellipse_sampling_synthetic():
    curve = {
        "type": "SketchEllipse",
        "center_point": {"x": 0.0, "y": 0.0},
        "major_axis": {"x": 1.0, "y": 0.0},
        "major_axis_radius": 4.0,
        "minor_axis_radius": 2.0,
    }
    poly = fusion.sample_curve(curve, {})
    # on-ellipse check: (x/a)^2 + (y/b)^2 == 1
    vals = (poly[:, 0] / 4.0) ** 2 + (poly[:, 1] / 2.0) ** 2
    np.testing.assert_allclose(vals, 1.0, rtol=1e-9)


def test_spline_fit_points_synthetic():
    pts = {
        f"p{i}": {"x": float(i), "y": float(i % 2), "z": 0.0} for i in range(4)
    }
    curve = {"type": "SketchFittedSpline", "fit_points": [f"p{i}" for i in range(4)]}
    poly = fusion.sample_curve(curve, pts)
    assert len(poly) > 16
    np.testing.assert_allclose(poly[0], [0.0, 0.0])
    np.testing.assert_allclose(poly[-1], [3.0, 1.0])


def test_spline_nurbs_synthetic():
    # quadratic B-spline arc-ish shape; endpoints of a clamped spline are the
    # first/last control points
    curve = {
        "type": "SketchFittedSpline",
        "control_points": [
            {"x": 0.0, "y": 0.0},
            {"x": 1.0, "y": 2.0},
            {"x": 2.0, "y": 0.0},
        ],
        "knots": [0, 0, 0, 1, 1, 1],
        "degree": 2,
    }
    poly = fusion.sample_curve(curve, {})
    assert poly is not None and len(poly) >= 32
    np.testing.assert_allclose(poly[0], [0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(poly[-1], [2.0, 0.0], atol=1e-6)


def test_sketch_bbox(fixture_design):
    s = fixture_design.sketches[0]
    polys = fusion.sketch_polylines(s)
    lo, hi = fusion.sketch_bbox(polys)
    assert np.all(hi > lo)
    assert fusion.sketch_bbox([]) is None
