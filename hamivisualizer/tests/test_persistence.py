"""模型 JSON 持久化测试."""

import json
from pathlib import Path

import numpy as np

from hamivisualizer.model.boundary import Boundary, BoundaryKind
from hamivisualizer.model.persistence import (
    hop_dict_to_row,
    load_model,
    model_to_dict,
    save_model,
)


def _sample_state():
    site_rows = [(0.0, 0.0, "A"), (0.0, 1.0, "B"), (1.0, 1.0, "A"), (1.0, 0.0, "B")]
    hop_rows = [
        {"name": "t", "from_site": 0, "to_site": 1, "off_x": 0, "off_y": 0,
         "amplitude": "-t", "phase_mode": "phase", "phase": "phi", "phase_sign": 1},
        {"name": "omg", "from_site": 0, "to_site": 0, "off_x": 0, "off_y": 0,
         "amplitude": "omg", "phase_mode": "none", "phase": "0", "phase_sign": 1},
    ]
    boundary = Boundary(BoundaryKind.SEMI, NY=3)
    params = {"t": 1.5, "phi": np.pi / 3, "omg": 0.5}
    return site_rows, hop_rows, boundary, params


def test_roundtrip():
    site_rows, hop_rows, boundary, params = _sample_state()
    obj = model_to_dict(
        site_rows, hop_rows, boundary, "cell", params, 0.25 * np.pi,
        True, True, labels_bottom_up=False,
        lattice_display={"nn": True, "nnn": False, "ghosts": False, "cells": True},
    )
    p = Path(__file__).parent / "._roundtrip_test.json"
    try:
        save_model(p, obj)
        back = load_model(p)
        assert back["version"] == 1
        assert len(back["sites"]) == 4
        assert back["hops"][0]["amplitude"] == "-t"      # 符号串保留
        assert back["hops"][0]["phase"] == "phi"
        assert back["boundary"] == {"kind": "semi", "NX": 2, "NY": 3}
        assert back["params"]["t"] == 1.5
        assert back["symbolic"] is True
        assert back["labels_bottom_up"] is False
        assert back["lattice_display"] == {
            "nn": True, "nnn": False, "ghosts": False, "cells": True,
        }
    finally:
        if p.exists():
            p.unlink()


def test_legacy_document_defaults_to_bottom_up_labels():
    site_rows, hop_rows, boundary, params = _sample_state()
    obj = model_to_dict(
        site_rows, hop_rows, boundary, "cell", params, 0.0, False, True,
    )
    obj.pop("labels_bottom_up")
    obj.pop("lattice_display")
    from hamivisualizer.model.persistence import validate_model_dict
    validated = validate_model_dict(obj)
    assert validated["labels_bottom_up"] is True
    assert all(validated["lattice_display"].values())


def test_hop_dict_to_row():
    row = hop_dict_to_row({"name": "t", "from_site": 2, "to_site": 1,
                           "cell_offset": [1, -1], "amplitude": "-t",
                           "phase_mode": "phase", "phase": "phi", "phase_sign": -1})
    assert row == ["t", 2, 1, 1, -1, "-t", "phase", "phi", -1]


def test_large_band_dimensions_are_not_rejected_by_legacy_10_12_cap():
    """Ribbon studies may use dimensions well beyond the old UI-only caps."""
    site_rows, hop_rows, _boundary, params = _sample_state()
    obj = model_to_dict(
        site_rows,
        hop_rows,
        Boundary(BoundaryKind.SEMI, NY=70),
        "cell",
        params,
        0.0,
        False,
        True,
    )
    assert obj["boundary"]["NY"] == 70


def test_non_rectangular_obc_shape_roundtrips_without_changing_default_schema():
    site_rows, hop_rows, _boundary, params = _sample_state()
    obj = model_to_dict(
        site_rows,
        hop_rows,
        Boundary(BoundaryKind.OBC, NX=4, NY=4, shape="triangle"),
        "cell",
        params,
        0.0,
        False,
        True,
        cell=(2.0, 2.0),
    )
    assert obj["boundary"]["shape"] == "triangle"


def test_load_rejects_unknown_version():
    p = Path(__file__).parent / "._bad_version_test.json"
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"version": 99}, f)
        try:
            load_model(p)
            assert False, "应拒绝未知版本"
        except ValueError:
            pass
    finally:
        if p.exists():
            p.unlink()
