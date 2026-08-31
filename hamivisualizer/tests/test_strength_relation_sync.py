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
from hamivisualizer.model.expression import classify_strength_expression, evaluate_expression
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


def test_unrelated_complex_relation_does_not_block_simple_family_edit():
    """A complex term on another bond must remain untouched and editable in the table."""
    _app, window, controller = _window()
    doc = _reverse_document("semi")
    doc["hops"].append({**doc["hops"][0], "off_x": 2, "amplitude": "t*u"})
    doc["params"]["u"] = 7.0
    controller.apply_document(doc)

    summary = window.panel.set_hopping_strength(0, 2.0)

    assert summary is not None
    rows = window.panel.get_hop_rows()
    assert rows[2]["amplitude"] == "t*u"
    assert window.panel.get_params()["u"] == pytest.approx(7.0)
    assert window.panel.error_label.text() == ""


def test_nonrational_strength_edit_preserves_other_bonds_and_parameter():
    """sqrt(2) terms use local scaling instead of fake giant integer ratios."""
    _app, window, controller = _window()
    doc = _reverse_document("semi")
    doc["hops"][0]["amplitude"] = "-sqrt(2)*t"
    doc["hops"].append({**doc["hops"][1], "cell_offset": [2, 0], "amplitude": "-t"})
    controller.apply_document(doc)

    summary = window.panel.set_hopping_strength(0, 2.0)

    assert summary is not None
    assert "局部缩放" in summary
    rows = window.panel.get_hop_rows()
    assert "sqrt(2)" in rows[0]["amplitude"]
    assert rows[2]["amplitude"] == "-t"
    assert window.panel.get_params()["t"] == pytest.approx(1.0)
    params = window.panel.get_params()
    assert abs(evaluate_expression(rows[0]["amplitude"], params)) == pytest.approx(2.0)
    assert abs(evaluate_expression(rows[1]["amplitude"], params)) == pytest.approx(2.0)
    assert abs(evaluate_expression(rows[2]["amplitude"], params)) == pytest.approx(1.0)


def test_rejected_canvas_strength_restores_editor_and_document():
    """Unsupported canvas edits are transactional: field and model stay unchanged."""
    _app, window, controller = _window()
    doc = _reverse_document("semi")
    doc["hops"][0]["amplitude"] = "t*u"
    doc["params"]["u"] = 1.0
    controller.apply_document(doc)
    window.resize(1200, 800)
    window.show()
    QApplication.processEvents()
    window.lattice_mode_btn.setChecked(True)
    window.lattice_coeff_btn.setChecked(True)
    controller.fit_all(force=True)
    QApplication.processEvents()

    scene = window.lattice_scene
    assert scene._edit_proxies
    proxy = scene._edit_proxies[0]
    editor = proxy.widget()
    row = int(proxy.data(1))
    old_text = editor.text()
    old_original = editor.property("hvisualizer-original-strength")
    before = controller.current_document()

    editor.setText("2")
    scene._commit_hop_strength(row, editor)
    QApplication.processEvents()

    assert editor.text() == old_text
    assert editor.property("hvisualizer-original-strength") == old_original
    assert controller.current_document() == before


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


@pytest.mark.parametrize("text", ("-t", "4*t", "t/3", "sqrt(3)*t"))
def test_strength_classifier_accepts_real_linear_forms(text):
    assert classify_strength_expression(text).kind == "linear"


def test_literal_strength_affects_only_selected_relation():
    _app, window, controller = _window()
    doc = _reverse_document("semi")
    doc["hops"][1]["cell_offset"] = [1, 0]
    doc["hops"].append({**doc["hops"][0], "cell_offset": [2, 0], "amplitude": "2"})
    for hop in doc["hops"][:2]:
        hop["amplitude"] = "1"
    controller.apply_document(doc)
    window.panel.set_hopping_strength(0, 3.0)
    rows = window.panel.get_hop_rows()
    assert rows[0]["amplitude"] == rows[1]["amplitude"] == "3.0"
    assert rows[2]["amplitude"] == "2"


@pytest.mark.parametrize("text", ("t*u", "t+t2", "t+1", "t**2", "sqrt(t)", "1/t", "I*t"))
def test_strength_classifier_rejects_nonhomogeneous_or_complex_forms(text):
    assert classify_strength_expression(text).kind == "unsupported"


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
