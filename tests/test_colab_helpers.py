from starx import colab


NAMES = [
    "r1.0.1/reconstruction/aaa_111_0000.json",
    "r1.0.1/reconstruction/aaa_111_0000_0001.obj",
    "r1.0.1/reconstruction/aaa_111_0000_0009.obj",
    "r1.0.1/reconstruction/aaa_111_0000.png",
    "r1.0.1/reconstruction/bbb_222_0000.json",
    "r1.0.1/reconstruction/bbb_222_0000.obj",
    "r1.0.1/train_test.json",
]


def test_design_members():
    members = colab.design_members(NAMES, "aaa_111_0000")
    assert len(members) == 4
    assert all("aaa_111_0000" in m for m in members)


def test_json_member():
    assert colab.json_member(NAMES, "aaa_111_0000").endswith("aaa_111_0000.json")
    assert colab.json_member(NAMES, "zzz") is None


def test_final_obj_prefers_plain_then_last_step():
    # bbb has a plain <id>.obj
    assert colab.final_obj_member(NAMES, "bbb_222_0000").endswith("bbb_222_0000.obj")
    # aaa only has step objs: the highest step wins
    assert colab.final_obj_member(NAMES, "aaa_111_0000").endswith("_0009.obj")
    assert colab.final_obj_member(NAMES, "zzz") is None


def test_build_zip_index_matches_per_call_helpers():
    ids = ["aaa_111_0000", "bbb_222_0000"]
    index = colab.build_zip_index(NAMES, ids)
    for design_id in ids:
        assert index[design_id]["json"] == colab.json_member(NAMES, design_id)
        assert index[design_id]["obj"] == colab.final_obj_member(NAMES, design_id)
    assert index["aaa_111_0000"]["png"].endswith("aaa_111_0000.png")
    assert "png" not in index["bbb_222_0000"]
