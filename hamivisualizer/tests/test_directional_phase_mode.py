"""Compatibility and UI guardrails for the reserved directional phase mode."""

from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import sympy as sp
from PySide6.QtWidgets import QApplication

from hamivisualizer.controller import ViewController
from hamivisualizer.model.hopping import HoppingTerm
from hamivisualizer.model.persistence import load_model, save_model, validate_model_dict
from hamivisualizer.model.templates import template_document
from hamivisualizer.view.main_window import MainWindow


def test_legacy_directional_document_validates_and_round_trips(tmp_path):
    document = template_document("方格", boundary_kind="semi")
    document["hops"][0]["phase_mode"] = "directional"
    assert validate_model_dict(document)["hops"][0]["phase_mode"] == "directional"
    path = tmp_path / "legacy.hvisual"
    save_model(path, document)
    assert load_model(path)["hops"][0]["phase_mode"] == "directional"


def test_unknown_phase_mode_is_rejected():
    document = template_document("方格", boundary_kind="semi")
    document["hops"][0]["phase_mode"] = "magnetic"
    with pytest.raises(ValueError, match="phase_mode"):
        validate_model_dict(document)


@pytest.mark.parametrize("amplitude", (1.0, sp.Symbol("t", real=True)))
def test_directional_hopping_is_explicitly_uncomputable(amplitude):
    hop = HoppingTerm("t", 0, 0, (1, 0), amplitude, "directional", sp.Symbol("phi"), 1)
    with pytest.raises(ValueError, match="directional.*none.*phase"):
        hop.evaluate()


def test_panel_rejects_reserved_mode_then_rebuilds_after_phase_repair():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    controller = ViewController(window)
    controller.load_preset("NP")
    window.panel.hop_table.item(0, 6).setText("directional")
    controller.rebuild()
    assert all(token in window.panel.error_label.text() for token in ("directional", "none", "phase"))
    window.panel.hop_table.item(0, 6).setText("phase")
    window.panel.hop_table.item(0, 7).setText("0")
    controller.rebuild()
    assert window.panel.error_label.text() == ""
    assert window.matrix_scene._data is not None
    window.panel.append_hop(["t", 0, 0, 0, 0, "-t", "none", "0", 1])
    assert window.panel.hop_table.item(window.panel.hop_table.rowCount() - 1, 6).text() == "none"
