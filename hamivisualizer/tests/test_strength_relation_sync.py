"""Regression tests for canvas strength write-back across folded rows."""

from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from hamivisualizer.controller import ViewController
from hamivisualizer.model.boundary import Boundary, BoundaryKind
from hamivisualizer.model.hamiltonian import HamiltonianBuilder
from hamivisualizer.model.hopping import HoppingTerm
from hamivisualizer.model.lattice import Lattice, Site
from hamivisualizer.model.persistence import load_model, model_to_dict, save_model
from hamivisualizer.view.main_window import MainWindow


def _window():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    return app, window, ViewController(window)


def _reverse_document(kind: str = "semi"):
    lattice = Lattice([Site(0, 0.0, 0.0, "A")], Lx=1.0, Ly=1.0)
    boundary = Boundary(BoundaryKind.SEMI if kind == "semi" else BoundaryKind.OBC, NX=3, NY=1)
    hops = [
        HoppingTerm("t", 0, 0, (1, 0), -1.0),
        HoppingTerm("t", 0, 0, (-1, 0), -1.0),
    ]
    return model_to_dict([(0.0, 0.0, "A")], [
        {"name": h.name, "from_site": h.from_site, "to_site": h.to_site,
         "off_x": h.cell_offset[0], "off_y": h.cell_offset[1],
         "amplitude": "-t", "phase_mode": "none", "phase": "0", "phase_sign": 1}
        for h in hops
    ], boundary, "cell", {"t": 1.0}, 0.0, False, True, (1.0, 1.0))


@pytest.mark.parametrize("kind", ("semi", "obc"))
def test_reverse_rows_sync_and_do_not_create_threefold_bond(kind):
    _app, window, controller = _window()
    controller.apply_document(_reverse_document(kind))
    panel = window.panel
    panel.set_hopping_strength(0, 2.0)
    rows = panel.get_hop_rows()
    assert [row["amplitude"] for row in rows] == ["-t", "-t"]
    controller.rebuild()
    if kind == "semi":
        np.testing.assert_allclose(controller._state[0].to_semi(0.0)[0, 0], -4.0)
    else:
        matrix = np.asarray(controller._state[0].H)
        assert matrix[0, 1] == pytest.approx(-2.0)


def test_same_geometry_different_parameter_families_remain_independent():
    _app, window, controller = _window()
    doc = _reverse_document("semi")
    doc["hops"].append({**doc["hops"][0], "name": "t2", "amplitude": "-t2"})
    doc["params"]["t2"] = 1.0
    controller.apply_document(doc)
    window.panel.set_hopping_strength(0, 2.0)
    rows = window.panel.get_hop_rows()
    assert rows[0]["amplitude"] == rows[1]["amplitude"] == "-t"
    assert rows[2]["amplitude"] == "-t2"


def test_negative_offset_selected_row_syncs_with_reverse_row_and_survives_save_load(tmp_path):
    _app, window, controller = _window()
    controller.apply_document(_reverse_document("semi"))
    window.panel.set_hopping_strength(1, 2.0)
    assert [row["amplitude"] for row in window.panel.get_hop_rows()] == ["-t", "-t"]
    controller.rebuild()
    before = np.asarray(controller._state[0].to_semi(0.37))
    path = tmp_path / "synced.hvisual"
    save_model(path, controller.current_document())
    controller.apply_document(load_model(path))
    after = np.asarray(controller._state[0].to_semi(0.37))
    np.testing.assert_allclose(after, before, atol=1e-12)


def test_same_direction_duplicate_rows_are_explicitly_additive_after_sync():
    _app, window, controller = _window()
    doc = _reverse_document("obc")
    doc["hops"][1]["cell_offset"] = [1, 0]
    controller.apply_document(doc)
    window.panel.set_hopping_strength(0, 2.0)
    assert [row["amplitude"] for row in window.panel.get_hop_rows()] == ["-t", "-t"]
    controller.rebuild()
    # Both same-direction rows are physical contributions, so their sum is 4.
    assert np.asarray(controller._state[0].H)[0, 1] == pytest.approx(-4.0)


def test_synced_strength_edit_is_undoable_and_redoable():
    _app, window, controller = _window()
    from pathlib import Path
    window._workspace_root = Path(".codex-artifacts/test-runs/strength-history")
    controller.apply_document(_reverse_document("semi"))
    window.enable_workspace_mode(controller)
    controller._on_hopping_strength_edited(0, 2.0)
    assert [row["amplitude"] for row in window.panel.get_hop_rows()] == ["-t", "-t"]
    assert window.panel.get_params()["t"] == pytest.approx(2.0)
    window.undo()
    assert window.panel.get_params()["t"] == pytest.approx(1.0)
    window.redo()
    assert window.panel.get_params()["t"] == pytest.approx(2.0)
