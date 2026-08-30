"""Qt-side regression tests for strict editing and scalable rendering."""

from __future__ import annotations

import os
import math
import argparse
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import sympy as sp
from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import QCloseEvent, QMouseEvent, QPointingDevice
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsProxyWidget,
    QGraphicsTextItem,
    QLabel,
    QSplitter,
    QMessageBox,
    QDialog,
    QDialogButtonBox,
)

from hamivisualizer.controller import ViewController, _build_lattice_scene
from hamivisualizer.model.boundary import Boundary, BoundaryKind
from hamivisualizer.model.hamiltonian import HamiltonianBuilder
from hamivisualizer.model.hopping import HoppingTerm
from hamivisualizer.model.lattice import Lattice, Site
from hamivisualizer.model.expression import evaluate_expression
from hamivisualizer.model.templates import TEMPLATE_NAMES, template_document
from hamivisualizer.model.workspace import load_preferences
from hamivisualizer.view.main_window import MainWindow
from hamivisualizer.view.dialogs import HoppingDialog, TemplateDialog
from hamivisualizer.view.matrix_view import MARGIN, MatrixView, RASTER_THRESHOLD
from hamivisualizer.view.band_view import BandView
from hamivisualizer.view.lattice_view import LatticeView, _parse_positive_strength
from hamivisualizer.view.rendermodel import LatticeSceneData, MatrixSceneData, WfSceneData
from hamivisualizer.view.zoom_view import ZoomGraphicsView
from hamivisualizer.view.wavefunction_view import WavefunctionView
from hamivisualizer.model.symbolic import ElementFormatter


def test_evidence_renderer_rejects_non_100_percent_output_scale():
    """离屏证据入口必须阻止混入150%/180%截图。"""
    from tools.render_ui_regression import (
        SCREENSHOT_UI_SCALE, _parse_screenshot_ui_scale,
    )

    assert SCREENSHOT_UI_SCALE == 1.0
    assert _parse_screenshot_ui_scale("1.0") == 1.0
    with pytest.raises(argparse.ArgumentTypeError, match="100%"):
        _parse_screenshot_ui_scale("1.5")
    with pytest.raises(argparse.ArgumentTypeError, match="100%"):
        _parse_screenshot_ui_scale("1.8")
    from tools.audit_editor_scales import (
        EVIDENCE_SCREENSHOT_SCALE, _evidence_path,
    )

    assert EVIDENCE_SCREENSHOT_SCALE == 1.0
    assert _evidence_path("np", "dark", 1.0).name == "np-dark-100.png"
    assert _evidence_path("np", "dark", 1.5) is None


def test_evidence_renderer_help_is_renderable():
    """证据脚本的帮助页不能因字面百分号触发 argparse 格式化错误。"""
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(repo / "tools" / "render_ui_regression.py"), "--help"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "1.0（100%）" in result.stdout


def test_editor_scale_audit_help_is_side_effect_free():
    """审计脚本的 --help 只打印说明，不启动整套 Qt 截图任务。"""
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(repo / "tools" / "audit_editor_scales.py"), "--help"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert "100%" in result.stdout
    assert "usage:" in result.stdout


def _window():
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    ctrl = ViewController(win)
    ctrl.load_preset("NP")
    return app, win, ctrl


def test_edit_toolbar_exposes_and_persists_snap_step(tmp_path):
    """吸附步长应在编辑工具栏可达，并与偏好设置保持一致。"""
    _app, win, ctrl = _window()
    win._workspace_root = tmp_path
    win.enable_workspace_mode(ctrl)
    win.show()
    _app.processEvents()

    assert win.lattice_snap_step_widget.isHidden()
    win._set_lattice_edit_mode(True)
    assert win.lattice_snap_step_widget.isVisible()
    assert win.lattice_snap_step_spin.value() == pytest.approx(0.25)

    win.lattice_snap_step_spin.setValue(0.125)
    assert win.lattice_scene.snap_step == pytest.approx(0.125)
    assert load_preferences(tmp_path).snap_step == pytest.approx(0.125)
    assert "吸附间隔" in win.lattice_snap_step_spin.toolTip()

    # Turning snapping off makes the inactive control honest while preserving
    # the chosen interval for the next time snapping is enabled.
    win.lattice_snap_btn.setChecked(False)
    assert not win.lattice_snap_step_spin.isEnabled()
    win.lattice_snap_btn.setChecked(True)
    assert win.lattice_snap_step_spin.isEnabled()
    assert win.lattice_snap_step_spin.value() == pytest.approx(0.125)


def test_workspace_connects_vector_export_actions(tmp_path, monkeypatch):
    """The production workspace wiring must expose both vector exporters."""
    _app = QApplication.instance() or QApplication([])
    win = MainWindow()
    ctrl = ViewController(win, connect_actions=False)
    ctrl.load_preset("NP")
    win._workspace_root = tmp_path
    called = []
    monkeypatch.setattr(ctrl, "export_svg", lambda: called.append("svg"))
    monkeypatch.setattr(ctrl, "export_pdf", lambda: called.append("pdf"))
    win.enable_workspace_mode(ctrl)

    win.action_export_svg.trigger()
    win.action_export_pdf.trigger()
    assert called == ["svg", "pdf"]


def test_dimension_resource_hint_is_live_and_does_not_restrict_editing():
    """尺寸提示预警应即时更新，但大模型仍可编辑/保存。"""
    _app, win, ctrl = _window()
    ctrl.set_runtime_preferences(calculation_mode="manual")
    assert "稠密预估" in win.panel.resource_hint.text()
    win.panel.set_boundary_index(1)  # OBC
    win.panel.set_dim(30, 30)
    assert "超过安全预算" in win.panel.resource_hint.text()
    assert win.panel.get_dim() == (30, 30)
    win.panel.set_dim(2, 2)
    assert "安全预算" in win.panel.resource_hint.text()


def test_control_accordion_uses_a_full_clickable_in_card_header():
    """侧栏标题不应依赖会被滚动裁切的 QGroupBox 外缘 title。"""
    _app, win, _ctrl = _window()
    win.resize(1100, 760)
    win.show()
    QApplication.processEvents()
    group = win.panel.params_group
    assert group.title() == ""
    header = group._header_rect()
    assert header.width() > 50 and header.height() >= group.fontMetrics().height()
    point = header.center().toPoint()
    QTest.mouseClick(group, Qt.LeftButton, pos=point)
    QApplication.processEvents()
    assert not group._expanded
    QTest.mouseClick(group, Qt.LeftButton, pos=point)
    QApplication.processEvents()
    assert group._expanded


def test_fractional_index_is_an_error_and_marks_results_stale():
    _app, win, ctrl = _window()
    old = win.matrix_scene._data
    win.panel.hop_table.item(0, 1).setText("1.9")
    ctrl.rebuild()
    assert "必须是整数" in win.panel.error_label.text()
    assert win.matrix_scene._data is old
    assert not win.result_banner.isHidden()
    assert not win.action_export.isEnabled()
    assert not win.action_export_svg.isEnabled()
    assert not win.action_export_pdf.isEnabled()
    assert not win.action_copy_matrix_latex.isEnabled()


def test_input_waiting_message_stays_in_status_bar():
    _app, win, _ctrl = _window()
    win.panel.kx_slider.setValue(10)
    assert win.result_banner.isHidden()
    assert "输入已更改" in win.statusBar().currentMessage()
    assert "b00020" in win.statusBar().styleSheet()


def test_canvas_hopping_strength_accepts_safe_fraction_input():
    """画布跃迁系数框与参数面板共享正数/分数输入体验。"""
    assert _parse_positive_strength("1/3") == pytest.approx(1 / 3)
    assert _parse_positive_strength(" 3 / 10 ") == pytest.approx(0.3)
    assert _parse_positive_strength("0.125") == pytest.approx(0.125)
    with pytest.raises(ValueError):
        _parse_positive_strength("1/0")
    with pytest.raises(ValueError):
        _parse_positive_strength("-1/3")


def test_canvas_hop_edit_flushes_history_for_immediate_undo(tmp_path):
    """画布提交后立即可撤回，不受普通输入防抖窗口影响。"""
    _app, win, ctrl = _window()
    win._workspace_root = tmp_path
    win.preferences.autosave = False
    win.enable_workspace_mode(ctrl)
    win.resize(1400, 900)
    win.show()
    QApplication.processEvents()
    win.lattice_mode_btn.setChecked(True)
    win.lattice_coeff_btn.setChecked(True)
    QApplication.processEvents()

    proxy = win.lattice_scene._edit_proxies[0]
    editor = proxy.widget()
    point = win.lattice_gv.mapFromScene(proxy.pos())
    point += QPoint(editor.width() // 2, editor.height() // 2)
    QTest.mouseClick(win.lattice_gv.viewport(), Qt.LeftButton, pos=point)
    assert editor.hasFocus()
    editor.selectAll()
    QTest.keyClicks(editor, "1/3")
    QTest.keyClick(editor, Qt.Key_Return)
    QApplication.processEvents()

    assert win._sessions[0].history.can_undo
    assert ctrl.current_document()["params"]["t"] == pytest.approx(1 / 3)
    win.undo()
    QApplication.processEvents()
    assert ctrl.current_document()["params"]["t"] == pytest.approx(1.0)
    assert win._sessions[0].history.can_redo
    win.redo()
    QApplication.processEvents()
    assert ctrl.current_document()["params"]["t"] == pytest.approx(1 / 3)


def test_history_actions_name_the_next_edit_to_undo(tmp_path):
    _app, win, ctrl = _window()
    win._workspace_root = tmp_path
    win.enable_workspace_mode(ctrl)
    before = ctrl.current_document()
    changed = dict(before)
    changed["sites"] = [dict(site) for site in before["sites"]]
    changed["sites"][0]["x"] += 0.25
    ctrl.apply_document(changed)
    assert win.action_undo.text() == "撤销：移动或编辑格点"
    assert "移动或编辑格点" in win.lattice_undo_btn.toolTip()


def test_busy_progress_stays_in_status_bar_without_top_banner():
    _app, win, _ctrl = _window()
    win.set_result_state("busy", "矩阵与晶格已更新，谱数据正在后台计算…", show_banner=False)
    assert win.result_banner.isHidden()
    win.flash_status("矩阵与晶格已更新，谱数据正在后台计算…")
    assert "后台计算" in win.statusBar().currentMessage()


def test_dimension_sliders_and_inputs_stay_in_sync_without_old_small_cap():
    """NX/NY are convenient to scrub, but precise larger values remain valid."""
    _app, win, _ctrl = _window()
    panel = win.panel

    panel.nx_slider.setValue(7)
    assert panel.nx_spin.value() == 7
    assert panel.get_dim()[0] == 7

    panel.ny_spin.setValue(70)
    assert panel.ny_slider.value() == 70
    assert panel.ny_slider.maximum() >= 70
    assert panel.get_dim()[1] == 70

    panel.set_boundary_index(0)
    assert not panel.nx_spin.isEnabled()


def test_large_ui_scale_reflows_lattice_editor_without_widening_window():
    """The visual editor remains usable at the 180% accessibility scale."""
    _app, win, _ctrl = _window()
    win.resize(1440, 920)
    win.show()
    QApplication.processEvents()
    win.lattice_mode_btn.setChecked(True)
    win.lattice_details_btn.setChecked(True)
    win.lattice_coeff_btn.setChecked(True)
    win._set_ui_scale(1.8, persist=False)
    QApplication.processEvents()
    QApplication.processEvents()

    # Before the responsive toolbar, the nine actions formed a single row
    # and Qt raised the window's minimum width above 2,200 px.  The wrapped
    # palette keeps the requested 1440 px viewport and every action visible.
    assert win.minimumSizeHint().width() < 1450
    assert win._lattice_edit_bar.minimumSize().width() < 720
    assert win._lattice_edit_bar.minimumSize().height() > win.lattice_mode_btn.sizeHint().height()


def test_hop_relation_styling_does_not_reenter_cell_changed():
    """Refreshing relation hints must be presentation-only and non-recursive."""
    _app, win, _ctrl = _window()
    panel = win.panel
    panel._update_hop_relation_tooltips()
    assert panel._updating_hop_relation is False
    panel.hop_table.item(0, 3).setText("1")
    assert panel._updating_hop_relation is False
    assert "胞间跃迁" in panel.hop_table.item(0, 3).toolTip()
    assert not panel.nx_slider.isEnabled()
    assert panel.ny_spin.isEnabled()
    assert panel.ny_slider.isEnabled()

    panel.set_boundary_index(1)
    assert panel.nx_spin.isEnabled()
    assert panel.nx_slider.isEnabled()


def test_new_model_dialog_uses_the_same_dimension_controls():
    app = QApplication.instance() or QApplication([])
    dialog = TemplateDialog()
    dialog.nx_slider.setValue(9)
    dialog.ny.setValue(70)
    assert dialog.nx.value() == 9
    assert dialog.ny_slider.value() == 70
    assert dialog.ny_slider.maximum() >= 70
    # Keep the creation preview aligned with the explicit, non-destructive
    # lattice-edit workflow. A normal site click is selection only.
    assert "先点击工具栏“添加跃迁”" in dialog.preview.text()


def test_new_model_dialog_makes_kagome_triangle_nanodisk_explicit():
    """Kagome+OBC opens on the real triangular nanodisk, not a rectangle."""
    app = QApplication.instance() or QApplication([])
    dialog = TemplateDialog()
    dialog.template.setCurrentText("Kagome")
    # Semi-infinite is intentionally neutral; switching to OBC selects the
    # dedicated, physically faithful triangle only when the user has not
    # chosen another shape yet.
    dialog.boundary.setCurrentText("双开边界（OBC）")
    assert dialog.shape.currentData() == "triangle"
    assert "3 格点斜原胞" in dialog.preview.text()
    assert "三条平直等边" in dialog.preview.text()
    # A deliberate alternate choice remains untouched.
    dialog.shape.setCurrentText("圆盘")
    dialog.boundary.setCurrentText("半无限（x-Bloch）")
    dialog.boundary.setCurrentText("双开边界（OBC）")
    assert dialog.shape.currentData() == "disk"


def test_new_model_dialog_explains_ssh_intercell_parameter_pair():
    """The SSH preset advertises which parameter controls the edge state."""
    app = QApplication.instance() or QApplication([])
    dialog = TemplateDialog()
    dialog.template.setCurrentText("SSH")
    assert "胞内 t1" in dialog.preview.text()
    assert "胞间 t2" in dialog.preview.text()
    assert "端点态" in dialog.preview.text()
    assert dialog.findChild(QDialogButtonBox).button(
        QDialogButtonBox.Cancel
    ).text() == "取消"


def test_new_model_dialog_explains_haldane_phase_and_mass_controls():
    """Haldane's complex NNN phase and mass term are discoverable in the wizard."""
    QApplication.instance() or QApplication([])
    dialog = TemplateDialog()
    dialog.template.setCurrentText("Haldane")
    assert dialog.connectivity.currentText() == "最近邻+次近邻"
    assert "复数次近邻相位" in dialog.preview.text()
    assert "子格质量" in dialog.preview.text()
    assert "能带开隙" in dialog.preview.text()


def test_parameter_table_expands_small_symbol_sets_instead_of_nested_scroll():
    """Four normal symbols remain visible; only unusually large sets scroll internally."""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document(
        "Haldane", boundary_kind="semi", connectivity="最近邻+次近邻",
    ))
    table = win.panel.param_table
    QApplication.processEvents()
    assert table.rowCount() == 4
    assert table.verticalScrollBarPolicy() == Qt.ScrollBarAlwaysOff
    required = (
        table.horizontalHeader().height()
        + table.verticalHeader().defaultSectionSize() * table.rowCount()
        + 2 * table.frameWidth() + 2
    )
    assert table.minimumHeight() >= required


def test_hopping_dialog_exposes_semantic_intercell_choices_for_semi_mode():
    """Half-infinite users must not infer Bloch offsets from raw spin boxes."""
    QApplication.instance() or QApplication([])
    dialog = HoppingDialog(0, 1, semi=True)
    assert dialog.cell_relation.count() >= 6
    right = dialog.cell_relation.findData("1,0")
    left = dialog.cell_relation.findData("-1,0")
    assert right >= 0 and left >= 0

    dialog.cell_relation.setCurrentIndex(right)
    assert (dialog.off_x.value(), dialog.off_y.value()) == (1, 0)
    assert not dialog.off_x.isEnabled() and not dialog.off_y.isEnabled()
    assert dialog.row(0, 1)[3:5] == [1, 0]
    assert "进入 x-Bloch H₁" in dialog.relation_effect.text()

    intra = dialog.cell_relation.findData("0,0")
    dialog.cell_relation.setCurrentIndex(intra)
    assert "写入 H₀" in dialog.relation_effect.text()

    custom = dialog.cell_relation.findData("custom")
    assert custom >= 0
    dialog.cell_relation.setCurrentIndex(custom)
    assert dialog.off_x.minimum() == -1000
    assert dialog.off_x.maximum() == 1000
    dialog.off_x.setValue(-250)
    dialog.off_y.setValue(125)
    assert dialog.row(0, 1)[3:5] == [-250, 125]
    assert "H250" in dialog.relation_effect.text()

    dialog.off_x.setValue(0)
    dialog.off_y.setValue(1)
    assert "有限方向胞间项" in dialog.relation_effect.text()


def test_hopping_dialog_prefills_offset_inferred_from_ghost_endpoint():
    """画布跨到周期像后，对话框应直接显示推导出的胞间关系。"""
    QApplication.instance() or QApplication([])
    dialog = HoppingDialog(0, 1, semi=True, cell_offset=(1, 0))
    assert dialog.cell_relation.currentData() == "1,0"
    assert dialog.off_x.value() == 1 and dialog.off_y.value() == 0
    assert "H₁" in dialog.relation_effect.text()


def test_side_panel_hopping_dialog_allows_endpoint_selection():
    """自定义入口应一次完成起点/终点选择，而不是隐式固定 0→1。"""
    QApplication.instance() or QApplication([])
    dialog = HoppingDialog(0, 1, semi=True, site_count=4)
    assert dialog.from_combo is not None and dialog.to_combo is not None
    dialog.from_combo.setCurrentIndex(2)
    dialog.to_combo.setCurrentIndex(3)
    right = dialog.cell_relation.findData("1,0")
    dialog.cell_relation.setCurrentIndex(right)
    row = dialog.row()
    assert row[1:5] == [2, 3, 1, 0]


def test_hopping_dialog_rejects_invalid_expression_before_accepting():
    """A malformed amplitude must not masquerade as a failed inter-cell edit."""
    QApplication.instance() or QApplication([])
    dialog = HoppingDialog(0, 1, semi=True)
    dialog.amplitude.setText("-t*(")
    dialog._accept_if_valid()
    assert dialog.result() != QDialog.Accepted
    assert not dialog.error_label.isHidden()
    assert "表达式无效" in dialog.error_label.text()

    dialog.amplitude.setText("-t")
    dialog.phase.setText("phi")
    dialog.cell_relation.setCurrentIndex(dialog.cell_relation.findData("1,0"))
    dialog._accept_if_valid()
    assert dialog.result() == QDialog.Accepted


def test_add_hop_button_uses_valid_defaults_and_reveals_semi_offsets():
    """The table fallback must support one-site and half-infinite models."""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document("一维链", boundary_kind="semi", ny=2))
    panel = win.panel
    assert "胞内" in panel.add_hop_btn.text()
    assert "dx/dy" in panel.add_hop_mode_btn.toolTip()
    assert panel.site_table.rowCount() == 1
    panel.hop_advanced_check.setChecked(False)
    panel.add_hop_btn.click()
    row = panel.get_hop_rows()[-1]
    assert (row["from_site"], row["to_site"]) == (0, 0)
    assert (row["off_x"], row["off_y"]) == (1, 0)
    assert row["phase_mode"] == "none" and row["phase"] == "0"
    assert panel.hop_advanced_check.isChecked()

    # A multi-site model starts with a valid intra-cell bond instead of
    # inventing a periodic offset; the same advanced columns remain available
    # for changing it to a left/right inter-cell bond.
    ctrl.apply_document(template_document("蜂窝", boundary_kind="semi", ny=2))
    panel.hop_advanced_check.setChecked(False)
    panel.add_hop_btn.click()
    row = panel.get_hop_rows()[-1]
    assert (row["from_site"], row["to_site"], row["off_x"], row["off_y"]) == (0, 1, 0, 0)
    assert panel.hop_advanced_check.isChecked()

    # Rebuild the genuinely empty one-site model as well; this proves the
    # safe default is not merely table-valid but enters the Bloch Hamiltonian.
    ctrl.apply_document(template_document(
        "空白自定义", boundary_kind="semi", ny=2, connectivity="仅格点",
    ))
    panel.add_hop_btn.click()
    ctrl.rebuild()
    assert panel.error_label.isHidden()
    assert win.matrix_scene._data is not None


def test_hopping_table_uses_semantic_display_headers_without_schema_change():
    _app, win, _ctrl = _window()
    panel = win.panel
    assert [panel.hop_table.horizontalHeaderItem(i).text()
            for i in range(panel.hop_table.columnCount())] == [
                "名称", "从", "到", "Δx", "Δy", "幅度", "相位模式", "相位", "符号",
            ]
    # Parsing still returns the stable model keys consumed by persistence and
    # the Hamiltonian builder.
    row = panel.get_hop_rows()[0]
    assert {"name", "from_site", "to_site", "off_x", "off_y", "amplitude"} <= row.keys()


def test_compact_hopping_table_keeps_intra_inter_cell_offsets_visible():
    """dx/dy must remain inspectable even before advanced columns are opened."""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document(
        "空白自定义", boundary_kind="semi", ny=2, connectivity="仅格点",
    ))
    panel = win.panel
    panel.hop_advanced_check.setChecked(False)
    panel.append_hop(["t", 0, 0, 1, 0, "-t", "none", "0", 1])
    assert not panel.hop_table.isColumnHidden(3)
    assert not panel.hop_table.isColumnHidden(4)
    assert panel.hop_table.isColumnHidden(6)
    assert panel.hop_table.isColumnHidden(7)
    assert panel.hop_table.isColumnHidden(8)
    assert "胞间跃迁" in panel.hop_table.item(0, 3).toolTip()
    assert "dx=+1" in panel.hop_table.item(0, 3).toolTip()
    panel.hop_table.item(0, 3).setText("0")
    assert "胞内跃迁" in panel.hop_table.item(0, 3).toolTip()


def test_dense_coordinate_cells_keep_full_values_in_hover_help():
    """Elided high-scale cells must still expose the exact parsed value."""
    _app, win, _ctrl = _window()
    panel = win.panel
    panel.set_lattice_rows([(0.123456789012, 0.987654321098, "A")])
    x_item = panel.site_table.item(0, 0)
    y_item = panel.site_table.item(0, 1)
    assert x_item.text() == "0.12345679"
    assert y_item.text() == "0.98765432"
    assert x_item.toolTip() == "完整值：0.123456789012"
    assert y_item.toolTip() == "完整值：0.987654321098"

    # The tooltip follows an inline edit rather than retaining the old
    # coordinate, which matters when the visible text is ellipsized.
    x_item.setText("12.345678901234")
    assert x_item.toolTip() == "完整值：12.345678901234"

    panel.set_hop_rows([["t", 0, 0, 1, 0, "-t", "none", "0", 1]])
    amp_item = panel.hop_table.item(0, 5)
    assert "胞间跃迁" in amp_item.toolTip()
    assert "完整值：-t" in amp_item.toolTip()


def test_hopping_relation_rows_have_visual_cue_and_context_edit_path():
    """关系提示不只依赖用户阅读 dx/dy，常用切换也可直接完成。"""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document(
        "空白自定义", boundary_kind="semi", ny=2, connectivity="仅格点",
    ))
    panel = win.panel
    panel.append_hop(["t", 0, 0, 0, 0, "-t", "none", "0", 1])
    panel.append_hop(["t", 0, 0, 1, 0, "-t", "none", "0", 1])

    intra = panel.hop_table.item(0, 0)
    inter = panel.hop_table.item(1, 0)
    assert intra.data(Qt.UserRole) == "intra"
    assert inter.data(Qt.UserRole) == "inter"
    assert "胞内跃迁" in intra.data(Qt.ToolTipRole)
    assert "dx=+1" in inter.data(Qt.ToolTipRole)
    # The semantic tint is palette-derived, so it remains useful in both
    # light and dark themes without asserting a brittle hard-coded RGB value.
    assert intra.data(Qt.BackgroundRole).color() != inter.data(Qt.BackgroundRole).color()

    menu = panel._create_hop_context_menu(1)
    menu_texts = [action.text() for action in menu.actions() if not action.isSeparator()]
    assert menu_texts == [
        "复制完整值",
        "设为胞内（dx=0, dy=0）",
        "设为右侧胞间（dx=+1, dy=0）",
        "设为左侧胞间（dx=-1, dy=0）",
        "在表格中编辑 dx / dy…",
    ]
    menu.deleteLater()

    panel.hop_table.selectRow(0)
    panel._set_selected_hop_offset(0, 1)
    changed = panel.get_hop_rows()[0]
    assert (changed["off_x"], changed["off_y"]) == (0, 1)
    assert panel.hop_table.item(0, 0).data(Qt.UserRole) == "inter"


def test_table_context_copy_uses_unabridged_coordinate_and_hop_text():
    """Right-click copy must use source text, not the elided cell display."""
    _app, win, _ctrl = _window()
    panel = win.panel
    panel.set_lattice_rows([(0.123456789012, 0.987654321098, "A")])
    assert panel.site_table.contextMenuPolicy() == Qt.CustomContextMenu
    copied = panel._copy_table_value(panel.site_table, 0, 0)
    assert copied == "0.123456789012"
    assert QApplication.clipboard().text() == copied

    panel.set_hop_rows([["t", 0, 0, 1, 0, "-t", "none", "0", 1]])
    hop_menu = panel._create_hop_context_menu(0, 5)
    copy_action = next(
        action for action in hop_menu.actions() if action.text() == "复制完整值"
    )
    assert copy_action.isEnabled()
    copy_action.trigger()
    assert QApplication.clipboard().text() == "-t"
    hop_menu.deleteLater()


def test_add_hop_mode_menu_creates_explicit_intra_and_inter_cell_rows():
    """The compact add menu must make Bloch relations discoverable."""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document("蜂窝", boundary_kind="semi", ny=2))
    panel = win.panel
    assert panel.add_hop_mode_btn.text() == "其他关系…"
    actions = {action.text(): action for action in panel.add_hop_mode_btn.menu().actions()}
    assert "胞内（dx=0, dy=0）" in actions
    assert "右侧胞间（dx=+1, dy=0）" in actions
    assert "左侧胞间（dx=-1, dy=0）" in actions

    actions["胞内（dx=0, dy=0）"].trigger()
    assert (panel.get_hop_rows()[-1]["from_site"], panel.get_hop_rows()[-1]["to_site"],
            panel.get_hop_rows()[-1]["off_x"], panel.get_hop_rows()[-1]["off_y"]) == (0, 1, 0, 0)
    actions["右侧胞间（dx=+1, dy=0）"].trigger()
    assert (panel.get_hop_rows()[-1]["off_x"], panel.get_hop_rows()[-1]["off_y"]) == (1, 0)
    actions["左侧胞间（dx=-1, dy=0）"].trigger()
    assert (panel.get_hop_rows()[-1]["off_x"], panel.get_hop_rows()[-1]["off_y"]) == (-1, 0)
    assert panel.hop_advanced_check.isChecked()

    # The same discoverability rule applies to finite OBC: an explicit
    # inter-cell choice must not disappear into the compact table.
    ctrl.apply_document(template_document("方格", boundary_kind="obc", nx=2, ny=2))
    panel.hop_advanced_check.setChecked(False)
    obc_actions = {action.text(): action for action in panel.add_hop_mode_btn.menu().actions()}
    obc_actions["右侧胞间（dx=+1, dy=0）"].trigger()
    assert panel.hop_advanced_check.isChecked()
    assert panel.get_hop_rows()[-1]["off_x"] == 1


def test_direct_intercell_button_is_visible_and_reaches_bloch_block():
    """The common '+' path must offer a one-click, unambiguous Bloch bond."""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document(
        "空白自定义", boundary_kind="semi", ny=1, connectivity="仅格点",
    ))
    panel = win.panel
    assert "胞间" in panel.add_inter_hop_btn.text()
    panel.add_inter_hop_btn.click()
    row = panel.get_hop_rows()[-1]
    assert (row["off_x"], row["off_y"]) == (1, 0)
    assert panel.hop_advanced_check.isChecked()
    ctrl.rebuild()
    result = ctrl._state[0]
    assert np.count_nonzero(result.blocks["H1"]) == 1


def test_relation_menu_reuses_selected_hop_endpoints():
    """Changing only the cell relation must not silently reset from/to."""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document("NP", boundary_kind="semi", ny=2))
    panel = win.panel
    row = panel.hop_table.rowCount() - 1
    panel.hop_table.selectRow(row)
    original = panel.get_hop_rows()[row]
    action = next(
        action for action in panel.add_hop_mode_btn.menu().actions()
        if "右侧胞间" in action.text()
    )
    action.trigger()
    added = panel.get_hop_rows()[-1]
    assert (added["from_site"], added["to_site"]) == (
        original["from_site"], original["to_site"]
    )
    assert (added["off_x"], added["off_y"]) == (1, 0)


def test_custom_intercell_menu_adds_long_range_bloch_row(monkeypatch):
    """任意 dx/dy 不应要求用户先理解隐藏的高级表格列。"""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document(
        "空白自定义", boundary_kind="semi", ny=2, connectivity="仅格点",
    ))
    panel = win.panel

    class _AcceptedDialog:
        def __init__(self, *_args, **_kwargs):
            self.cell_relation = self

        def findData(self, value):
            return 0 if value == "custom" else -1

        def setCurrentIndex(self, _index):
            return None

        def exec(self):
            return QDialog.Accepted

        def row(self, from_site=None, to_site=None):
            return ["t2", 0 if from_site is None else from_site,
                    0 if to_site is None else to_site,
                    2, 0, "-t2", "none", "0", 1]

    monkeypatch.setattr("hamivisualizer.view.control_panel.HoppingDialog", _AcceptedDialog)
    actions = {action.text(): action for action in panel.add_hop_mode_btn.menu().actions()}
    assert "自定义胞间偏移…" in actions
    actions["自定义胞间偏移…"].trigger()

    row = panel.get_hop_rows()[-1]
    assert (row["from_site"], row["to_site"], row["off_x"], row["off_y"]) == (0, 0, 2, 0)
    assert panel.hop_advanced_check.isChecked()
    assert "e^(±i2kₓ)" in panel.hop_relation_hint.text()


def test_hopping_relation_summary_explains_intra_and_inter_cell_rows():
    """紧凑跃迁表也要持续告诉用户胞内/胞间的真实数量。"""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document("蜂窝", boundary_kind="semi", ny=2))
    panel = win.panel
    assert "胞内" in panel.hop_relation_hint.text()
    assert "胞间" in panel.hop_relation_hint.text()
    before = panel.hop_relation_hint.text()
    action = next(
        action for action in panel.add_hop_mode_btn.menu().actions()
        if "右侧胞间" in action.text()
    )
    action.trigger()
    after = panel.hop_relation_hint.text()
    assert before != after
    assert "Bloch" in after
    assert "x 方向胞间" in after
    panel.del_hop_btn.click()
    assert panel.hop_relation_hint.text() == before


def test_hopping_relation_summary_reports_long_range_bloch_harmonics():
    """dx=2/3 rows must not be mislabeled as nearest-cell e^(±ikx)."""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document("空白自定义", boundary_kind="semi", ny=1,
                                          connectivity="仅格点"))
    panel = win.panel
    action = next(
        action for action in panel.add_hop_mode_btn.menu().actions()
        if "右侧胞间" in action.text()
    )
    action.trigger()
    # Turn the explicit menu-created nearest-cell row into a second-neighbor
    # row through the same advanced table path a user would use.
    row = panel.hop_table.rowCount() - 1
    panel.hop_table.item(row, 3).setText("2")
    hint = panel.hop_relation_hint.text()
    assert "Bloch 谐波" in hint
    assert "e^(±i2kₓ)" in hint
    assert "e^(±ikₓ)" not in hint


def test_hopping_relation_summary_counts_diagonal_intercell_offsets():
    """混合 dx/dy 偏移必须同时出现在 x、y 和斜向统计中。"""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document(
        "空白自定义", boundary_kind="semi", ny=2, connectivity="仅格点",
    ))
    panel = win.panel
    action = next(
        action for action in panel.add_hop_mode_btn.menu().actions()
        if "右侧胞间" in action.text()
    )
    action.trigger()
    row = panel.hop_table.rowCount() - 1
    panel.hop_table.item(row, 4).setText("1")
    hint = panel.hop_relation_hint.text()
    assert "x 方向胞间 1 条" in hint
    assert "y 方向胞间 1 条" in hint
    assert "斜向胞间 1 条" in hint


def test_menu_created_right_intercell_hop_reaches_bloch_harmonic():
    """语义化“右侧胞间”入口必须真正改变 H(kx)，而不只是改表格。"""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document(
        "空白自定义", boundary_kind="semi", ny=1, connectivity="仅格点",
    ))
    panel = win.panel
    action = next(
        action for action in panel.add_hop_mode_btn.menu().actions()
        if "右侧胞间" in action.text()
    )
    action.trigger()
    ctrl.rebuild()
    result = ctrl._state[0]
    assert np.count_nonzero(result.blocks["H1"]) == 1
    assert result.to_semi(0.0)[0, 0] == pytest.approx(-2.0)
    assert result.to_semi(np.pi)[0, 0] == pytest.approx(2.0)


def test_custom_large_offset_reaches_long_range_bloch_harmonic():
    """The dialog's wider range must survive the controller and builder."""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document(
        "空白自定义", boundary_kind="semi", ny=1, connectivity="仅格点",
    ))
    win.panel.append_hop(["u", 0, 0, 250, 0, "-u", "none", "0", 1])
    ctrl.rebuild()
    result = ctrl._state[0]
    assert any(key[0] == 250 for key in result.extra)
    assert result.to_semi(0.0)[0, 0] == pytest.approx(-2.0)
    assert result.to_semi(np.pi / 500.0)[0, 0] == pytest.approx(0.0, abs=1e-10)


def test_long_range_bloch_term_does_not_expand_hundreds_of_ghost_columns():
    """Matrix harmonics stay exact while the interactive canvas stays bounded."""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document(
        "空白自定义", boundary_kind="semi", ny=1, connectivity="仅格点",
    ))
    win.panel.append_hop(["u", 0, 0, 250, 0, "-u", "none", "0", 1])
    ctrl.rebuild()
    result = ctrl._state[0]
    assert any(key[0] == 250 for key in result.extra)
    # Four complete context columns on each side are enough to inspect local
    # topology; the requested 250-cell harmonic must not create a 501-column
    # scene and collapse the fit-to-window view.
    assert len(win.lattice_scene._data.ghost) == 8


def test_edit_mode_can_restore_geometry_without_losing_hoppings_or_params():
    _app, win, ctrl = _window()
    win.controller = ctrl
    baseline = ctrl.current_document()
    win._set_lattice_edit_mode(True)
    assert not win.lattice_restore_btn.isHidden()
    assert win.lattice_restore_btn.text() == "恢复编辑前构型"
    assert "Ctrl+Z" in win.lattice_restore_btn.toolTip()

    rows = win.panel.get_site_rows()
    changed = list(rows)
    changed[0] = (0.25, 0.25, changed[0][2])
    win.panel.set_lattice_rows(changed)
    ctrl.rebuild()
    assert ctrl.current_document()["sites"] != baseline["sites"]

    win._restore_edit_baseline()
    restored = ctrl.current_document()
    assert restored["sites"] == baseline["sites"]
    assert restored["cell"] == baseline["cell"]
    assert restored["hops"] == baseline["hops"]
    assert restored["params"] == baseline["params"]


def test_restore_edit_baseline_removes_new_spacing_override_when_original_was_auto():
    """自动元胞尺寸也必须能从一次手动间距修改中完整恢复。"""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document(
        "空白自定义", boundary_kind="semi", ny=2, connectivity="仅格点",
    ))
    # Explicitly exercise the automatic-spacing representation.  The blank
    # template carries a safe 1×1 default until the user chooses “自动”.
    win.panel.set_cell_size(None)
    ctrl.rebuild()
    baseline = ctrl.current_document()
    assert baseline["cell"] is None
    win._set_lattice_edit_mode(True)

    win.panel.set_cell_size((3.0, 4.0))
    ctrl.rebuild()
    assert ctrl.current_document()["cell"] == {"Lx": 3.0, "Ly": 4.0}

    win._restore_edit_baseline()
    restored = ctrl.current_document()
    assert restored["cell"] is None
    assert win.panel.get_cell_size() is None


def test_restore_edit_baseline_drops_only_hops_to_new_sites():
    """恢复几何后，新增格点的悬空跃迁不能让整个恢复动作失败。"""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document(
        "空白自定义", boundary_kind="semi", ny=1, connectivity="仅格点",
    ))
    win._set_lattice_edit_mode(True)
    baseline = ctrl.current_document()
    assert len(baseline["sites"]) == 1

    # Simulate a valid topology edit: one new site and one valid original
    # bond plus one bond that points only to the newly-added site.
    current = deepcopy(baseline)
    current["sites"].append({"x": 0.5, "y": 0.0, "sublattice": "B"})
    current["hops"] = [
        {"name": "t", "from_site": 0, "to_site": 0,
         "cell_offset": [1, 0], "amplitude": "1", "phase_mode": "none",
         "phase": "0", "phase_sign": 1},
        {"name": "t", "from_site": 0, "to_site": 1,
         "cell_offset": [0, 0], "amplitude": "1", "phase_mode": "none",
         "phase": "0", "phase_sign": 1},
    ]
    ctrl.apply_document(current)
    win._restore_edit_baseline()

    restored = ctrl.current_document()
    assert restored["sites"] == baseline["sites"]
    assert len(restored["hops"]) == 1
    assert restored["hops"][0]["to_site"] == 0
    assert "新增格点" in win.statusBar().currentMessage()


def test_restore_edit_baseline_is_undoable_and_redoable(tmp_path):
    """恢复构型应作为一次历史操作，且撤销/重做后仍可继续记录编辑。"""
    _app, win, ctrl = _window()
    win._workspace_root = tmp_path
    win.enable_workspace_mode(ctrl)
    win._set_lattice_edit_mode(True)
    baseline = ctrl.current_document()

    changed = deepcopy(baseline)
    changed["sites"] = [dict(site) for site in baseline["sites"]]
    changed["sites"][0]["x"] += 0.25
    ctrl.apply_document(changed)
    assert ctrl.current_document()["sites"] != baseline["sites"]

    win._restore_edit_baseline()
    assert ctrl.current_document()["sites"] == baseline["sites"]
    assert win._sessions[win._active_index].history.undo_label == "恢复编辑前构型"

    win.undo()
    assert ctrl.current_document()["sites"] == changed["sites"]
    assert "恢复编辑前构型" in win.statusBar().currentMessage()

    win.redo()
    assert ctrl.current_document()["sites"] == baseline["sites"]
    assert "恢复编辑前构型" in win.statusBar().currentMessage()

    # History replay must not leak into subsequent edits.
    changed_again = deepcopy(baseline)
    changed_again["sites"] = [dict(site) for site in baseline["sites"]]
    changed_again["sites"][0]["x"] -= 0.25
    ctrl.apply_document(changed_again)
    assert win._sessions[win._active_index].history.can_undo


def test_parameter_outside_slider_range_is_preserved():
    _app, win, _ctrl = _window()
    win.panel.set_params({"custom": -123.456}, force=True)
    assert win.panel.get_params()["custom"] == pytest.approx(-123.456)


def test_large_matrix_uses_one_raster_item():
    QApplication.instance() or QApplication([])
    n = RASTER_THRESHOLD + 20
    matrix = np.eye(n, dtype=complex)
    scene = MatrixView()
    scene.set_data(MatrixSceneData(n=n, values=matrix, matrix=matrix))
    assert any(isinstance(item, QGraphicsPixmapItem) for item in scene.items())
    assert len(scene.items()) < 50


def test_raster_matrix_text_becomes_readable_without_creating_cell_items():
    """Large matrices use a viewport text layer once individual cells are large."""
    QApplication.instance() or QApplication([])
    n = RASTER_THRESHOLD + 20
    matrix = np.eye(n, dtype=complex)
    scene = MatrixView()
    view = ZoomGraphicsView(scene)
    view.resize(640, 480)
    view.show()
    scene.set_data(MatrixSceneData(n=n, values=matrix, matrix=matrix, mode="numeric"))
    QApplication.processEvents()
    view.fitInView(scene.sceneRect())
    assert not scene.text_visible_in_view(view)
    view._scale_by(8.0)
    assert scene.text_visible_in_view(view)
    assert scene._cell_text_at(0, 0) == "1.00"
    assert not any(isinstance(item, QGraphicsTextItem) for item in scene.items())


def test_matrix_cell_click_selects_readable_cell_and_reports_detail():
    QApplication.instance() or QApplication([])
    scene = MatrixView()
    view = ZoomGraphicsView(scene)
    view.resize(480, 360)
    view.show()
    scene.set_data(MatrixSceneData(
        n=2, values=np.eye(2), matrix=np.eye(2), mode="numeric",
    ))
    QApplication.processEvents()
    view.fitInView(scene.sceneRect())
    assert scene.text_visible_in_view(view)
    clicks = []
    scene.cellClicked.connect(lambda i, j: clicks.append((i, j)))
    point = view.mapFromScene(QPointF(MARGIN + scene._cell_size / 2,
                                     MARGIN + scene._cell_size / 2))
    QTest.mouseClick(view.viewport(), Qt.LeftButton, pos=QPoint(point.x(), point.y()))
    assert clicks == [(0, 0)]
    assert scene._selected_cell == (0, 0)
    assert scene._selection_item is not None
    assert scene._selection_item.data(0) == "matrix-selection"
    assert scene.cell_details(0, 0) == ("H[1,1]", "H[1,1]", "1.00")


def test_matrix_cell_click_does_not_arm_canvas_pan():
    """A matrix selection must not make a tiny pointer wobble drag the view."""
    QApplication.instance() or QApplication([])
    scene = MatrixView()
    view = ZoomGraphicsView(scene)
    view.resize(480, 360)
    view.show()
    scene.set_data(MatrixSceneData(
        n=2, values=np.eye(2), matrix=np.eye(2), mode="numeric",
    ))
    QApplication.processEvents()
    view.fitInView(scene.sceneRect())
    point = view.mapFromScene(QPointF(MARGIN + scene._cell_size / 2,
                                      MARGIN + scene._cell_size / 2))
    QTest.mousePress(view.viewport(), Qt.LeftButton,
                     pos=QPoint(point.x(), point.y()))
    QApplication.processEvents()
    assert view._pan_press_pos is None
    assert view._pan_active is False
    assert scene.selected_cell == (0, 0)
    QTest.mouseRelease(view.viewport(), Qt.LeftButton,
                       pos=QPoint(point.x(), point.y()))


def test_middle_drag_remains_available_for_matrix_panning():
    """Middle-button drag keeps a reliable pan path after cell clicks are reserved."""
    QApplication.instance() or QApplication([])
    scene = MatrixView()
    view = ZoomGraphicsView(scene)
    view.resize(480, 360)
    view.show()
    scene.set_data(MatrixSceneData(
        n=2, values=np.eye(2), matrix=np.eye(2), mode="numeric",
    ))
    QApplication.processEvents()
    view.fitInView(scene.sceneRect())
    point = view.mapFromScene(QPointF(MARGIN + scene._cell_size / 2,
                                      MARGIN + scene._cell_size / 2))
    QTest.mousePress(view.viewport(), Qt.MiddleButton,
                     pos=QPoint(point.x(), point.y()))
    assert view._pan_press_pos is not None
    assert view._pan_button == Qt.MiddleButton
    QTest.mouseRelease(view.viewport(), Qt.MiddleButton,
                       pos=QPoint(point.x(), point.y()))
    assert view._pan_press_pos is None


def test_main_window_matrix_click_is_non_modal_and_uses_logical_labels():
    _app, win, _ctrl = _window()
    data = win.matrix_scene._data
    assert data is not None
    win.matrix_scene.cellClicked.emit(0, 1)
    status = win.statusBar().currentMessage()
    assert "已选中" in status
    assert "H[1:1,1:2]" in status
    assert "H[1,2]" in status
    assert win.matrix_scene._selected_cell == (0, 1)


def test_matrix_copy_action_uses_current_selection_and_clipboard():
    _app, win, _ctrl = _window()
    assert not win.action_copy_matrix_cell.isEnabled()
    win.matrix_scene.cellClicked.emit(0, 1)
    assert win.action_copy_matrix_cell.isEnabled()
    win.action_copy_matrix_cell.trigger()
    copied = QApplication.clipboard().text()
    assert copied.startswith("H[1:1,1:2] = ")
    assert "e^{i\\phi}" in copied or "e^{iφ}" in copied
    assert "LaTeX: H[1,2] = " in copied
    assert "\\phi" in copied
    assert "·" not in copied.split("LaTeX:", 1)[1]
    assert "已复制" in win.statusBar().currentMessage()
    # A new result revision clears the selection and invalidates the action;
    # stale matrix details must never remain copyable.
    win.matrix_scene.set_data(win.matrix_scene._data)
    assert not win.action_copy_matrix_cell.isEnabled()


def test_default_smart_semi_matrix_preserves_kx_structure():
    _app, win, _ctrl = _window()
    data = win.matrix_scene._data
    assert data.mode == "smart"
    assert isinstance(data.matrix, sp.MatrixBase)
    assert any(str(symbol) == "kx" for symbol in data.matrix.free_symbols)
    shown = [win.matrix_scene._cell_text_at(i, j)
             for i in range(data.n) for j in range(data.n)]
    assert any("k_{x}" in value for value in shown)
    assert all("0.707106" not in value and r"\frac" not in value for value in shown)


def test_edit_snap_prefers_graphene_coordinates_and_drag_origin():
    QApplication.instance() or QApplication([])
    scene = LatticeView()
    root3 = np.sqrt(3.0)
    scene.set_edit_context(
        [(0.0, 0.0, "A"), (root3 / 2, 0.5, "B")], snap_step=0.25,
    )
    origin = QPointF(root3 / 2, -0.5)
    snapped = scene.snap_position(1, QPointF(0.87, -0.49), origin)
    assert snapped.x() == pytest.approx(root3 / 2)
    assert snapped.y() == pytest.approx(-0.5)


def test_edit_session_keeps_original_graphene_sites_as_snap_targets():
    """A completed drag must not erase the user's route back to the basis."""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document("蜂窝", boundary_kind="semi", ny=4))
    win._set_lattice_edit_mode(True)
    scene = win.lattice_scene
    original = win.panel.get_site_rows()
    assert scene._snap_reference_sites == [
        (float(x), float(y)) for x, y, _sub in original
    ]

    # Simulate the post-drag rebuild: current editable geometry has moved,
    # while the baseline is still a magnetic candidate in the same anchor.
    moved = [(x + 0.45, y + 0.35, sub) for x, y, sub in original]
    scene.set_edit_context(
        moved, cell_vectors=win.panel.get_cell_vectors(),
        anchor_offset=scene.edit_anchor_offset,
    )
    x0, y0, _ = original[0]
    anchor_x, anchor_y = scene.edit_anchor_offset
    snapped = scene.snap_position(0, QPointF(x0 + anchor_x + 0.05, -y0 - anchor_y + 0.04))
    assert snapped.x() == pytest.approx(x0 + anchor_x)
    assert snapped.y() == pytest.approx(-y0 - anchor_y)

    win._set_lattice_edit_mode(False)
    assert scene._snap_reference_sites == []


def test_edit_snap_is_a_true_two_dimensional_toggle():
    QApplication.instance() or QApplication([])
    scene = LatticeView()
    scene.set_edit_context(
        [(0.0, 0.0, "A"), (1.0, 1.0, "B")], snap_step=0.25,
    )
    # Independent x/y snapping used to produce (1, 0), a point that did not
    # exist in this geometry.  The complete 2-D candidate must win instead.
    snapped = scene.snap_position(0, QPointF(0.92, -0.88))
    assert snapped.x() == pytest.approx(1.0)
    assert snapped.y() == pytest.approx(-1.0)
    scene.set_snap_enabled(False)
    free = scene.snap_position(0, QPointF(0.92, -0.88))
    assert free.x() == pytest.approx(0.92)
    assert free.y() == pytest.approx(-0.88)


def test_edit_snap_does_not_stack_sites_during_a_real_drag():
    """Magnetic snap must not hide one editable site under another."""
    QApplication.instance() or QApplication([])
    scene = LatticeView()
    scene.set_edit_context(
        [(0.0, 0.0, "A"), (1.0, 0.0, "B")], snap_step=0.25,
    )
    origin = QPointF(1.0, 0.0)
    # The pointer is close to A.  A non-interactive helper call may still
    # choose the complete geometry candidate, but a drag must not stack B on
    # A or silently create a duplicate coordinate.
    snapped = scene.snap_position(1, QPointF(0.03, -0.02), origin)
    assert math.hypot(snapped.x(), snapped.y()) > 0.03
    assert math.hypot(snapped.x() - 1.0, snapped.y()) > 0.03
    exact = scene.snap_position(1, QPointF(0.0, 0.0), origin)
    assert math.hypot(exact.x(), exact.y()) > 0.12


def test_free_drag_collision_guard_applies_when_snap_is_disabled():
    """Alt/free dragging must retain a valid, non-overlapping geometry."""
    QApplication.instance() or QApplication([])
    scene = LatticeView()
    scene.set_edit_context(
        [(0.0, 0.0, "A"), (1.0, 0.0, "B")], snap_step=0.25,
    )
    scene.set_snap_enabled(False)
    origin = QPointF(1.0, 0.0)
    safe = scene.snap_position(1, QPointF(0.0, 0.0), origin)
    assert math.hypot(safe.x(), safe.y()) > 0.03
    assert math.hypot(safe.x() - 1.0, safe.y()) > 0.03
    # The direct final gate is also safe for a simulated item-change path.
    safe_direct = scene.safe_edit_position(1, QPointF(0.0, 0.0), fallback=origin)
    assert math.hypot(safe_direct.x(), safe_direct.y()) > 0.03


def test_oblique_cell_length_edit_preserves_direction():
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document("蜂窝", connectivity="最近邻"))
    before = win.panel.get_cell_vectors()
    assert before is not None
    a1, a2 = before
    win.panel.lx_spin.setValue(np.hypot(*a1) * 1.25)
    after = win.panel.get_cell_vectors()
    assert after is not None
    assert after[0][1] / after[0][0] == pytest.approx(a1[1] / a1[0])
    assert np.hypot(*after[0]) == pytest.approx(np.hypot(*a1) * 1.25)
    assert after[1][0] / after[1][1] == pytest.approx(a2[0] / a2[1])


def test_cell_length_inputs_are_fully_readable_at_100_percent():
    """The two compact length fields must not elide irrational preset values."""
    _app, win, ctrl = _window()
    win.resize(1920, 1200)
    win.show()
    QApplication.processEvents()
    for template_name in TEMPLATE_NAMES:
        ctrl.apply_document(template_document(
            template_name, boundary_kind="obc", nx=4, ny=4,
            connectivity="最近邻",
        ))
        QApplication.processEvents()
        for field_name in ("lx_spin", "ly_spin"):
            field = getattr(win.panel, field_name)
            text = field.text()
            line_edit = field.lineEdit()
            assert "..." not in text, template_name
            # Leave a small cushion for the style's frame/padding.  This is a
            # geometry assertion rather than a pixel snapshot so it remains
            # deterministic across the bundled Windows/Linux Qt backends.
            available = max(0, line_edit.contentsRect().width() - 4)
            assert field.fontMetrics().horizontalAdvance(text) <= available, (
                template_name, field_name, text,
                field.fontMetrics().horizontalAdvance(text), available,
            )


def test_oblique_cell_length_zero_does_not_leave_an_inconsistent_auto_state():
    """A zero length in oblique mode is rejected and restored visibly."""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document("蜂窝", connectivity="最近邻"))
    before = win.panel.get_cell_vectors()
    assert before is not None
    old_lengths = (np.hypot(*before[0]), np.hypot(*before[1]))

    win.panel.lx_spin.setValue(0.0)
    assert win.panel.get_cell_vectors() == before
    assert win.panel.lx_spin.value() == pytest.approx(old_lengths[0])
    assert win.panel.ly_spin.value() == pytest.approx(old_lengths[1])
    assert "斜原胞" in win.panel.error_label.text()

    # A subsequent valid edit clears the transient validation error.
    win.panel.lx_spin.setValue(old_lengths[0] * 1.1)
    assert win.panel.error_label.isHidden()


def test_cell_spacing_editor_rejects_signed_lengths_and_keeps_auto_pair_atomic():
    """长度输入只能是正值；自动间距必须由 Lx/Ly 成对启用。"""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document(
        "空白自定义", boundary_kind="semi", ny=2, connectivity="仅格点",
    ))
    win.panel.set_cell_size((2.0, 3.0))
    before = win.panel.get_cell_size()
    with pytest.raises(ValueError, match="正数|自动"):
        win.panel.set_cell_size((-1.0, 3.0))
    assert win.panel.get_cell_size() == before

    # A user can temporarily type zero while clearing both fields.  The
    # intermediate half-auto state is rejected by the public getter instead
    # of being silently interpreted as a negative/automatic geometry.
    win.panel.lx_spin.setValue(0.0)
    with pytest.raises(ValueError, match="同时设置|自动"):
        win.panel.get_cell_size()
    win.panel.ly_spin.setValue(0.0)
    assert win.panel.get_cell_size() is None


def test_spacing_edit_reflows_anchor_and_snap_targets_with_new_cell_vectors():
    """间距修改后，编辑锚点、元胞框和吸附参考必须使用同一组新矢量。"""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document(
        "蜂窝", ny=4, boundary_kind="semi", connectivity="最近邻",
    ))
    win._set_lattice_edit_mode(True)
    scene = win.lattice_scene
    old_anchor = scene.edit_anchor_offset
    old_polygons = tuple(scene._data.cell_polygons)
    old_vectors = win.panel.get_cell_vectors()
    assert old_vectors is not None
    (a1x, a1y), (a2x, a2y) = old_vectors
    new_vectors = ((a1x * 1.25, a1y * 1.25), (a2x * 1.1, a2y * 1.1))
    win.panel.set_cell_vectors(new_vectors)
    ctrl.rebuild()

    for actual, expected in zip(scene._cell_vectors, new_vectors):
        assert actual == pytest.approx(expected)
    assert scene.edit_anchor_offset != old_anchor
    assert tuple(scene._data.cell_polygons) != old_polygons
    item = scene._edit_items[0]
    x0, y0, _sub = win.panel.get_site_rows()[0]
    assert item.pos().x() == pytest.approx(x0 + scene.edit_anchor_offset[0])
    assert item.pos().y() == pytest.approx(-y0 - scene.edit_anchor_offset[1])

    # The edit-session baseline remains in local primitive-cell coordinates,
    # but its scene position must be transformed by the newly active anchor.
    snapped = scene.snap_position(
        0,
        item.pos() + QPointF(0.04, -0.03),
    )
    assert snapped.x() == pytest.approx(x0 + scene.edit_anchor_offset[0])
    assert snapped.y() == pytest.approx(-y0 - scene.edit_anchor_offset[1])


def test_set_cell_vectors_copies_mutable_inputs_and_validates_lattice_periods():
    """矢量输入不会被外部列表别名修改，且无效周期会即时拒绝。"""
    _app, win, _ctrl = _window()
    vectors = [[2.0, 0.0], [0.5, 1.5]]
    win.panel.set_cell_vectors(vectors)
    vectors[0][0] = 99.0
    assert win.panel.get_cell_vectors() == ((2.0, 0.0), (0.5, 1.5))
    with pytest.raises(ValueError, match="非零|共线"):
        win.panel.set_cell_vectors(((0.0, 0.0), (0.0, 1.0)))
    with pytest.raises(ValueError, match="非零|共线"):
        win.panel.set_cell_vectors(((1.0, 0.0), (1.0, 0.0)))


def test_oblique_a1_vertical_component_moves_hop_editor_endpoint():
    """Canvas guides must use the same oblique displacement as the solver."""
    QApplication.instance() or QApplication([])
    scene = LatticeView()
    scene.set_edit_context(
        [(0.0, 0.0, "A"), (0.0, 0.0, "B")],
        hops=[{
            "row": 0, "from_site": 0, "to_site": 1,
            "off_x": 1, "off_y": 0, "strength": 1.0,
        }],
        cell_vectors=((1.0, 0.5), (0.0, 1.0)),
    )
    scene.set_edit_mode(True)
    scene.set_data(LatticeSceneData(
        sites=((0.0, 0.0, "0", "A"), (0.0, 0.0, "1", "B")),
        edges=((0, 1, "NN"),),
    ))
    assert scene._edit_guides
    line = scene._edit_guides[0].line()
    assert line.x2() == pytest.approx(1.0)
    assert line.y2() == pytest.approx(-0.5)


def test_honeycomb_strength_editor_normalizes_to_four_to_one():
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document("蜂窝", connectivity="最近邻"))
    summary = None
    for row, value in ((0, 1.2), (1, 0.3), (2, 0.3)):
        summary = win.panel.set_hopping_strength(row, value)
    assert summary is not None
    assert "胞内=4×t" in summary
    assert "胞间(-1,0)=t" in summary
    assert "基准 t=0.3" in summary
    rows = win.panel.get_hop_rows()
    assert [row["amplitude"] for row in rows] == ["-4*t", "-t", "-t"]
    params = win.panel.get_params()
    assert params["t"] == pytest.approx(0.3)
    assert [abs(evaluate_expression(row["amplitude"], params)) for row in rows] == pytest.approx(
        [1.2, 0.3, 0.3]
    )
    win.panel.set_symbolic(True)
    ctrl.rebuild()
    # The assembled matrix may combine several physical bonds into one entry;
    # verify the 4:1 symbolic ratio at the source rows instead of relying on
    # the post-assembly sum.
    assert any("4*t" in str(hop.amplitude) for hop in ctrl._display_hops)


def test_hopping_strength_uses_table_row_after_blank_row():
    """A blank row must not shift the physical bond edited on the canvas."""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document("蜂窝", connectivity="最近邻"))
    panel = win.panel
    panel.hop_table.insertRow(0)
    # The original first three rows now live at table rows 1, 2 and 3.
    panel.set_hopping_strength(1, 1.2)
    assert panel.hop_table.item(1, 5).text() == "-6*t"
    assert panel.hop_table.item(2, 5).text() == "-5*t"
    assert panel.hop_table.item(3, 5).text() == "-5*t"
    assert panel.hop_table.item(0, 5) is None


def test_site_position_edit_uses_table_row_after_blank_row():
    """A blank site row must not redirect a canvas drag to another site."""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document("蜂窝", connectivity="最近邻"))
    panel = win.panel
    original = panel.get_site_rows()
    panel.site_table.insertRow(0)

    panel.update_site_position(0, 0.25, 0.75)

    assert panel.site_table.item(0, 0) is None
    assert panel.site_table.item(0, 1) is None
    assert panel.site_table.item(1, 0).text() == "0.25"
    assert panel.site_table.item(1, 1).text() == "0.75"
    rows = panel.get_site_rows()
    assert rows[0][:2] == pytest.approx((0.25, 0.75))
    assert rows[1][0] == pytest.approx(original[1][0])


def test_site_position_edit_rejects_duplicate_coordinates_at_panel_boundary():
    """Synthetic/plugin position signals must not create duplicate sites."""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document("蜂窝", connectivity="最近邻"))
    panel = win.panel
    before = panel.get_site_rows()
    panel.update_site_position(1, before[0][0], before[0][1])

    assert "不能重复" in panel.error_label.text()
    after = panel.get_site_rows()
    assert after[1][:2] == pytest.approx(before[1][:2])


def test_site_delete_uses_table_row_after_blank_row():
    """Deleting a compact site index removes the corresponding populated row."""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document("蜂窝", connectivity="最近邻"))
    panel = win.panel
    panel.site_table.insertRow(0)

    panel.remove_site(0)

    assert panel.site_table.item(0, 0) is None
    assert panel.site_table.item(1, 0) is not None
    assert len(panel.get_site_rows()) == 1


def test_new_hopping_starter_ignores_blank_site_rows():
    """One-site models keep a valid inter-cell starter after table gaps."""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document("空白自定义", connectivity="仅格点"))
    panel = win.panel
    panel.site_table.insertRow(0)

    panel._add_default_hop()

    row = panel.hop_table.rowCount() - 1
    # The editor shows conventional one-based labels, while the parsed model
    # payload remains zero-based for the Hamiltonian builder.
    assert panel.hop_table.item(row, 1).text() == "1"
    assert panel.hop_table.item(row, 2).text() == "1"
    assert panel.hop_table.item(row, 3).text() == "1"
    assert panel.hop_table.item(row, 4).text() == "0"
    parsed = panel.get_hop_rows()[-1]
    assert (parsed["from_site"], parsed["to_site"]) == (0, 0)


def test_hopping_table_displays_one_based_endpoints_but_parses_zero_based():
    """Visible hop labels match lattice labels without changing file semantics."""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document("蜂窝", connectivity="最近邻"))
    panel = win.panel
    assert panel.hop_table.item(0, 1).text() == "1"
    assert panel.hop_table.item(0, 2).text() == "2"
    assert (panel.get_hop_rows()[0]["from_site"],
            panel.get_hop_rows()[0]["to_site"]) == (0, 1)


def test_edit_mode_renders_one_strength_editor_per_physical_bond():
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document("蜂窝", connectivity="最近邻"))
    win.lattice_scene.set_edit_mode(True)
    proxies = [item for item in win.lattice_scene.items()
               if isinstance(item, QGraphicsProxyWidget)]
    assert len(proxies) == 3


def test_edit_strength_editors_are_offset_from_bond_centers():
    """Controls remain readable instead of stacking directly on nodes/lines."""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document("蜂窝", connectivity="最近邻"))
    win.lattice_scene.set_edit_mode(True)
    scene = win.lattice_scene
    (a1x, a1y), (a2x, a2y) = scene._cell_vectors
    physical = []
    for hop in scene._edit_hops:
        fr, to = int(hop["from_site"]), int(hop["to_site"])
        ox, oy = int(hop.get("off_x", 0)), int(hop.get("off_y", 0))
        if fr == to and ox == 0 and oy == 0:
            continue
        x1, y1, _ = scene._edit_sites[fr]
        x2, y2, _ = scene._edit_sites[to]
        x2 += ox * a1x + oy * a2x
        y2 += ox * a1y + oy * a2y
        physical.append(((x1 + x2) / 2, -(y1 + y2) / 2))
    assert len(physical) == len(scene._edit_proxy_anchors)
    assert all(
        math.hypot(anchor_x - mx, anchor_y - my) >= 0.15
        for (_proxy, anchor_x, anchor_y), (mx, my)
        in zip(scene._edit_proxy_anchors, physical)
    )


def test_edit_strength_rail_keeps_multiple_bonds_on_the_right_without_drift():
    """One rendered line gets one field; the table still keeps duplicate rows."""
    _app, win, ctrl = _window()
    document = template_document(
        "空白自定义", boundary_kind="semi", connectivity="仅格点",
    )
    document["sites"] = [
        {"x": 0.2, "y": 0.0, "sublattice": "A"},
        {"x": 0.6, "y": 0.8, "sublattice": "A"},
    ]
    document["hops"] = [
        {"name": "t", "from_site": 0, "to_site": 1, "off_x": 0,
         "off_y": 0, "amplitude": "t", "phase": "none",
         "phase_sign": 1},
        {"name": "t", "from_site": 0, "to_site": 1, "off_x": 0,
         "off_y": 0, "amplitude": "t", "phase": "none",
         "phase_sign": 1},
    ]
    document["params"] = {"t": 2.0}
    ctrl.apply_document(document)
    win.resize(1440, 920)
    win.show()
    QApplication.processEvents()
    win.lattice_mode_btn.setChecked(True)
    ctrl.fit_all(force=True)
    QApplication.processEvents()

    scene = win.lattice_scene
    assert len(scene._edit_proxies) == 1
    assert len(scene._edit_leader_links) == 1
    assert scene._edit_proxies[0].widget().toolTip().endswith("等价表格行：1, 2")
    # Both fields are in one stable right-hand rail; they must not drift by
    # inheriting the preceding hop's cell anchor.
    assert len({round(proxy.pos().x(), 6) for proxy in scene._edit_proxies}) == 1
    for diagonal, horizontal, mx, my, proxy in scene._edit_leader_links:
        assert diagonal.line().p1() == QPointF(mx, my)
        assert diagonal.line().p2() == horizontal.line().p1()
        factor = scene._last_editor_layout[0]
        assert abs(horizontal.line().x2() - horizontal.line().x1()) >= 6.0 / factor
        assert horizontal.line().y1() == pytest.approx(horizontal.line().y2())
        assert proxy.pos().x() > mx


def test_compact_editors_explain_adjacent_intra_and_intercell_bonds():
    """共线的 SSH 两条键也要能一眼分辨胞内/胞间关系。"""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document(
        "SSH", boundary_kind="semi", connectivity="最近邻", nx=4, ny=4,
    ))
    win.resize(1368, 900)
    win.show()
    QApplication.processEvents()
    win.lattice_mode_btn.setChecked(True)
    ctrl.fit_all(force=True)
    QApplication.processEvents()

    scene = win.lattice_scene
    ghost_nn = [item for item in scene.items()
                if item.data(0) == "ghost-edge-nn"]
    assert ghost_nn
    assert all(item.pen().style() == Qt.DashLine for item in ghost_nn)
    # Badges sit above their corresponding fixed-pixel fields at both the
    # normal and accessibility UI scales, without escaping the viewport.
    for scale in (1.0, 1.8):
        win._set_ui_scale(scale, persist=False)
        ctrl.fit_all(force=True)
        QApplication.processEvents()
        badges = sorted(
            (item for item in scene.items()
             if item.data(0) == "hopping-editor-relation"),
            key=lambda item: next(
                int(proxy.data(1))
                for badge, proxy in scene._edit_relation_badges
                if badge is item
            ),
        )
        assert [badge.toPlainText() for badge in badges] == [
            "胞内", "胞间 +1,+0",
        ]
        assert len(scene._edit_proxies) == 2
        view = win.lattice_gv
        for badge, proxy in scene._edit_relation_badges:
            badge_top = view.mapFromScene(badge.pos()).y()
            editor_top = view.mapFromScene(proxy.pos()).y()
            assert badge_top <= editor_top
            assert -2 <= badge_top <= view.viewport().height() + 2


@pytest.mark.parametrize("template_name", TEMPLATE_NAMES)
@pytest.mark.parametrize("boundary_kind", ("semi", "obc"))
@pytest.mark.parametrize("ui_scale", (1.0, 1.8))
def test_edit_strength_editors_are_not_clipped_after_fit(
    template_name, boundary_kind, ui_scale,
):
    """Fixed-pixel editors remain visible for every template/boundary pair."""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document(
        template_name, connectivity="最近邻", boundary_kind=boundary_kind,
    ))
    win.resize(1440, 920)
    win.show()
    QApplication.processEvents()
    # The default 100% path catches ordinary regressions; the upper supported
    # UI scale is equally important because stylesheet metrics and the fixed
    # pixel proxy editors change independently there.
    win._set_ui_scale(ui_scale, persist=False)
    QApplication.processEvents()
    win.lattice_mode_btn.setChecked(True)
    ctrl.fit_all(force=True)
    QApplication.processEvents()
    scene, view = win.lattice_scene, win.lattice_gv
    # Exhaustive mode remains available for visual regression; the normal UI
    # uses progressive disclosure for dense lattices.
    scene.set_show_all_hop_editors(True)
    QApplication.processEvents()
    # The blank template intentionally has no physical bonds, hence no
    # coefficient editors; all other templates expose one per editable bond.
    if not scene._edit_proxies:
        assert template_name == "空白自定义"
        return
    viewport = view.viewport().rect()
    editor_rects = []
    for proxy in scene._edit_proxies:
        # ``sceneBoundingRect`` is not a useful measurement for a fixed-pixel
        # proxy. Measure the actual pixel footprint from its top-left scene
        # anchor instead.
        top_left = view.mapFromScene(proxy.pos())
        widget = proxy.widget()
        mapped = viewport.adjusted(0, 0, 0, 0)
        mapped.setTopLeft(top_left)
        mapped.setSize(widget.size())
        assert mapped.top() >= -2
        assert mapped.bottom() <= viewport.height() + 2
        assert mapped.left() >= -2
        assert mapped.right() <= viewport.width() + 2
        # Keep a visible screen-space cushion for the complete rounded border
        # at high-DPI/UI-scale transforms, not merely a mathematically
        # in-bounds top-left corner.
        assert mapped.bottom() <= viewport.height() - 8
        assert mapped.right() <= viewport.width() - 8
        editor_rects.append(mapped)
    # The right rail is a real interaction contract, not just a screenshot
    # convention: fields must remain separate at every supported UI scale.
    for index, first in enumerate(editor_rects):
        for second in editor_rects[index + 1:]:
            assert not first.adjusted(-1, -1, 1, 1).intersects(second)
    visible_points = list(scene._data.sites)
    if scene._show_ghosts:
        visible_points += list(scene._data.ghost)
    if visible_points:
        radius_px = abs(float(view.transform().m11())) * float(scene._site_radius)
        rightmost_node = max(
            view.mapFromScene(QPointF(float(point[0]), -float(point[1]))).x()
            + radius_px
            for point in visible_points
        )
        assert all(rect.left() >= rightmost_node + 1 for rect in editor_rects)


def test_edit_strength_editors_keep_full_borders_and_avoid_node_overlap():
    """The visible control itself must be a complete, non-occluded target."""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document(
        "蜂窝", connectivity="最近邻+次近邻", boundary_kind="semi", ny=4,
    ))
    win.resize(1440, 920)
    win.show()
    QApplication.processEvents()
    win.lattice_mode_btn.setChecked(True)
    ctrl.fit_all(force=True)
    QApplication.processEvents()
    scene, view = win.lattice_scene, win.lattice_gv
    scene.set_show_all_hop_editors(True)
    QApplication.processEvents()
    assert scene._edit_proxies
    editor_rects = []
    for proxy in scene._edit_proxies:
        widget = proxy.widget()
        assert widget.objectName() == "hopStrengthEditor"
        assert widget.height() >= 26
        # The proxy must expose the same polished widget footprint.  A
        # stylesheet min-height larger than setFixedSize used to leave the
        # proxy bounding rect at 26 px while the child widget was 41 px,
        # clipping the focused editor's lower border and hit area.
        assert proxy.boundingRect().height() >= widget.height()
        assert proxy.boundingRect().width() >= widget.width()
        top_left = view.mapFromScene(proxy.pos())
        rect = view.viewport().rect().adjusted(0, 0, 0, 0)
        rect.setTopLeft(top_left)
        rect.setSize(widget.size())
        editor_rects.append(rect)
    # A one-pixel gap is intentional; it makes each field's rounded bottom
    # border visibly distinct instead of merging into its neighbour.
    for i, first in enumerate(editor_rects):
        for second in editor_rects[i + 1:]:
            assert not first.adjusted(-1, -1, 1, 1).intersects(second)


def test_rebuilding_edit_scene_disposes_old_embedded_strength_editors():
    """A rebuild must not leave stale proxy widgets painted at old positions."""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document(
        "蜂窝", connectivity="最近邻+次近邻", boundary_kind="semi", ny=4,
    ))
    win.resize(1440, 920)
    win.show()
    QApplication.processEvents()
    win.lattice_mode_btn.setChecked(True)
    win.lattice_coeff_btn.setChecked(True)
    QApplication.processEvents()
    scene = win.lattice_scene
    assert scene._edit_proxies
    editors = [widget for widget in QApplication.allWidgets()
               if widget.objectName() == "hopStrengthEditor"]
    assert len(editors) == len(scene._edit_proxies)

    # Theme/parameter changes rebuild the same shared scene in normal use.
    # Re-run that path directly and verify that every old embedded QWidget was
    # detached and deleted before the new editor layer was painted.
    scene.set_data(scene._data)
    QApplication.processEvents()
    editors = [widget for widget in QApplication.allWidgets()
               if widget.objectName() == "hopStrengthEditor"]
    assert len(editors) == len(scene._edit_proxies)
    assert all(widget.isVisible() for widget in editors)


def test_focused_strength_editor_is_revealed_after_panning():
    """Focusing a field near an edge scrolls it fully into the viewport."""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document("蜂窝", connectivity="最近邻"))
    win.resize(1440, 920)
    win.show()
    QApplication.processEvents()
    win.lattice_mode_btn.setChecked(True)
    ctrl.fit_all(force=True)
    QApplication.processEvents()
    scene, view = win.lattice_scene, win.lattice_gv
    proxy = scene._edit_proxies[-1]
    # Move the viewport to an edge before focusing the field.  The focus hook
    # must reveal the whole fixed-pixel widget rather than just its anchor.
    view.verticalScrollBar().setValue(view.verticalScrollBar().maximum())
    view.horizontalScrollBar().setValue(view.horizontalScrollBar().maximum())
    QApplication.processEvents()
    point = view.mapFromScene(proxy.pos())
    QTest.mouseClick(view.viewport(), Qt.LeftButton, pos=point)
    QApplication.processEvents()
    top_left = view.mapFromScene(proxy.pos())
    widget = proxy.widget()
    mapped = view.viewport().rect().adjusted(0, 0, 0, 0)
    mapped.setTopLeft(top_left)
    mapped.setSize(widget.size())
    viewport = view.viewport().rect()
    assert proxy.zValue() == pytest.approx(30.0)
    assert mapped.top() >= 0
    assert mapped.left() >= 0
    assert mapped.bottom() <= viewport.height()
    assert mapped.right() <= viewport.width()


def test_pressing_strength_editor_does_not_arm_canvas_pan():
    """Typing/dragging inside a coefficient field must not pan the lattice."""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document("蜂窝", connectivity="最近邻"))
    win.resize(1440, 920)
    win.show()
    QApplication.processEvents()
    win.lattice_mode_btn.setChecked(True)
    ctrl.fit_all(force=True)
    QApplication.processEvents()
    scene, view = win.lattice_scene, win.lattice_gv
    proxy = scene._edit_proxies[0]
    point = view.mapFromScene(proxy.pos())
    QTest.mousePress(view.viewport(), Qt.LeftButton, pos=point)
    QApplication.processEvents()
    assert view._pan_press_pos is None
    assert view._pan_active is False
    QTest.mouseRelease(view.viewport(), Qt.LeftButton, pos=point)


def test_strength_editor_center_click_accepts_fraction_without_canvas_pan():
    """The visible editor's full pixel footprint is a real text target.

    A previous regression only exercised the proxy's top-left anchor.  That
    could pass while a user click in the middle of the fixed-pixel field was
    delivered to the canvas, arming a pan instead of entering text.
    """
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document("蜂窝", connectivity="最近邻"))
    win.resize(1440, 920)
    win.show()
    QApplication.processEvents()
    win.lattice_mode_btn.setChecked(True)
    win.lattice_coeff_btn.setChecked(True)
    ctrl.fit_all(force=True)
    QApplication.processEvents()
    scene, view = win.lattice_scene, win.lattice_gv
    assert scene._edit_proxies
    proxy = scene._edit_proxies[0]
    editor = proxy.widget()
    top_left = view.mapFromScene(proxy.pos())
    center = top_left + QPoint(editor.width() // 2, editor.height() // 2)
    before = ctrl.current_document()
    QTest.mouseClick(view.viewport(), Qt.LeftButton, pos=center)
    QApplication.processEvents()
    assert editor.hasFocus()
    assert view._pan_press_pos is None
    editor.selectAll()
    QTest.keyClicks(editor, "1/3")
    QTest.keyClick(editor, Qt.Key_Return)
    QApplication.processEvents()
    assert view._pan_press_pos is None
    after = ctrl.current_document()
    assert after != before
    # The canvas field and side table share the same positive-fraction parser;
    # the first hopping row must now carry the normalized absolute strength.
    assert float(win.panel.get_params()["t"]) == pytest.approx(1 / 3)
    assert editor.text().startswith("0.")
    assert editor.fontMetrics().horizontalAdvance(editor.text()) <= editor.width() - 20
    assert editor.cursorPosition() == 0
    assert "已归一化" in win.statusBar().currentMessage()
    assert "基准 t=" in win.statusBar().currentMessage()


def test_strength_editor_remeasures_after_ui_scale_changes():
    """Changing global UI scale after editing must not reintroduce clipping."""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document("蜂窝", connectivity="最近邻"))
    win.resize(1440, 920)
    win.show()
    QApplication.processEvents()
    win.lattice_mode_btn.setChecked(True)
    win.lattice_coeff_btn.setChecked(True)
    ctrl.fit_all(force=True)
    QApplication.processEvents()
    proxy = win.lattice_scene._edit_proxies[0]
    editor = proxy.widget()
    editor.setText("1/3")
    win.lattice_scene._commit_hop_strength(0, editor)
    QApplication.processEvents()
    assert float(win.panel.get_params()["t"]) == pytest.approx(1 / 3)
    win._set_ui_scale(1.8, persist=False)
    QApplication.processEvents()
    editor = win.lattice_scene._edit_proxies[0].widget()
    assert editor.text().startswith("0.")
    assert editor.fontMetrics().horizontalAdvance(editor.text()) <= editor.width() - 20
    assert editor.height() >= 26


def test_strength_proxy_tracks_editor_after_production_font_initialization():
    """Windows font setup plus a high UI scale must keep the proxy border whole."""
    from PySide6.QtGui import QFont
    from hamivisualizer.main import _configure_font

    app, win, ctrl = _window()
    original_font = QFont(app.font())
    try:
        # The normal launcher registers the CJK/math fonts before constructing
        # MainWindow.  Most lightweight tests intentionally skip that step;
        # exercise it here so proxy/widget geometry is tested in production
        # order rather than only with Qt's default Sans Serif metrics.
        _configure_font(app)
        ctrl.apply_document(template_document("NP", connectivity="最近邻"))
        win.resize(1440, 920)
        win.show()
        QApplication.processEvents()
        win._set_ui_scale(1.5, persist=False)
        QApplication.processEvents()
        win.lattice_mode_btn.setChecked(True)
        win.lattice_scene.set_show_all_hop_editors(True)
        ctrl.fit_all(force=True)
        QApplication.processEvents()
        assert win.lattice_scene._edit_proxies
        for proxy in win.lattice_scene._edit_proxies:
            editor = proxy.widget()
            assert proxy.boundingRect().width() >= editor.width()
            assert proxy.boundingRect().height() >= editor.height()
    finally:
        app.setFont(original_font)


def test_rebuilt_fraction_editor_keeps_leading_zero_visible_at_large_scale():
    """A committed fraction must not reopen with its first glyph scrolled away."""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document("蜂窝", connectivity="最近邻"))
    win.resize(1440, 920)
    win.show()
    QApplication.processEvents()
    win.lattice_mode_btn.setChecked(True)
    win.lattice_scene.set_show_all_hop_editors(True)
    ctrl.fit_all(force=True)
    QApplication.processEvents()
    editor = win.lattice_scene._edit_proxies[0].widget()
    editor.setText("1/3")
    win.lattice_scene._commit_hop_strength(0, editor)
    QApplication.processEvents()
    win._set_ui_scale(1.8, persist=False)
    win._set_theme_mode("dark")
    ctrl.fit_all(force=True)
    QApplication.processEvents()
    rebuilt = win.lattice_scene._edit_proxies[0].widget()
    assert rebuilt.text().startswith("0.")
    assert rebuilt.cursorPosition() == 0
    assert rebuilt.fontMetrics().horizontalAdvance(rebuilt.text()) <= rebuilt.width() - 20


def test_double_click_on_existing_lattice_content_never_adds_or_deletes():
    QApplication.instance() or QApplication([])
    scene = LatticeView()
    scene.set_edit_context([(0.0, 0.0, "A")])
    scene.set_edit_mode(True)
    scene.set_data(LatticeSceneData(sites=((0.0, 0.0, "0", "A"),)))
    view = ZoomGraphicsView(scene)
    view.resize(420, 320)
    view.show()
    QApplication.processEvents()
    view.fitInView(scene.sceneRect())
    added = []
    scene.siteAddRequested.connect(lambda x, y: added.append((x, y)))
    point = view.mapFromScene(QPointF(0.0, 0.0))
    QTest.mouseDClick(view.viewport(), Qt.LeftButton, pos=point)
    assert added == []
    assert len(scene._edit_items) == 1


def test_blank_canvas_site_creation_requires_the_explicit_tool():
    """Double-clicking must not create surprise sites in a dense diagram."""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document("SC", connectivity="仅格点"))
    win.resize(1000, 700)
    win.show()
    QApplication.processEvents()
    win.lattice_mode_btn.setChecked(True)
    ctrl.fit_all(force=True)
    QApplication.processEvents()
    scene, view = win.lattice_scene, win.lattice_gv
    before = win.panel.site_table.rowCount()
    # Add inside the highlighted editable primitive cell.  The old test used
    # an arbitrary canvas coordinate that happened to be outside SC's
    # translated cell; accepting it would reintroduce the invalid-coordinate
    # bug this interaction contract is meant to prevent.
    anchor_x, anchor_y = scene.edit_anchor_offset
    target = QPointF(anchor_x + 0.25, -(anchor_y + 0.25))
    point = view.mapFromScene(target)

    QTest.mouseDClick(view.viewport(), Qt.LeftButton, pos=point)
    QApplication.processEvents()
    assert win.panel.site_table.rowCount() == before

    win.lattice_add_site_btn.setChecked(True)
    QApplication.processEvents()
    assert scene.site_creation_mode
    # Window-level Esc also works while the toolbar still owns focus.
    QTest.keyClick(win.lattice_add_site_btn, Qt.Key_Escape)
    QApplication.processEvents()
    assert not scene.site_creation_mode
    assert not win.lattice_add_site_btn.isChecked()

    win.lattice_add_site_btn.setChecked(True)
    QApplication.processEvents()
    # Topology tools are mutually exclusive: switching to bonds cannot leave
    # a latent one-click site insertion armed behind the visible button.
    win.lattice_add_hop_btn.setChecked(True)
    QApplication.processEvents()
    assert scene.hop_creation_mode
    assert not scene.site_creation_mode
    assert not win.lattice_add_site_btn.isChecked()
    win.lattice_add_hop_btn.setChecked(False)

    win.lattice_add_site_btn.setChecked(True)
    QApplication.processEvents()
    # Arming the tool can reflow the top action strip, so resolve the scene
    # coordinate after the layout settles rather than reusing a stale viewport
    # pixel from the pre-tool geometry.
    point = view.mapFromScene(target)
    QTest.mouseClick(view.viewport(), Qt.LeftButton, pos=point)
    QApplication.processEvents()
    assert win.panel.site_table.rowCount() == before + 1
    assert not scene.site_creation_mode
    assert not win.lattice_add_site_btn.isChecked()


def test_clicking_site_guide_or_hop_editor_is_non_destructive():
    """Plain clicks must never rebuild away an edit-mode lattice element."""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document("蜂窝", connectivity="最近邻"))
    win.resize(1200, 800)
    win.show()
    QApplication.processEvents()
    win.lattice_mode_btn.setChecked(True)
    ctrl.fit_all(force=True)
    QApplication.processEvents()
    scene, view = win.lattice_scene, win.lattice_gv
    before = (len(scene.items()), len(scene._edit_items), win.panel.hop_table.rowCount())

    handle = next(iter(scene._edit_items.values()))
    QTest.mouseClick(view.viewport(), Qt.LeftButton,
                     pos=view.mapFromScene(handle.scenePos()))
    QApplication.processEvents()
    assert (len(scene.items()), len(scene._edit_items), win.panel.hop_table.rowCount()) == before
    assert not scene.hop_creation_mode
    assert scene._hop_start is None

    # Bond creation is an explicit tool.  A first click is only interpreted
    # as its start after the toolbar button was deliberately armed.
    win.lattice_add_hop_btn.setChecked(True)
    QApplication.processEvents()
    assert scene.hop_creation_mode
    QTest.mouseClick(view.viewport(), Qt.LeftButton,
                     pos=view.mapFromScene(handle.scenePos()))
    QApplication.processEvents()
    assert scene._hop_start in scene._edit_items
    assert (len(scene.items()), len(scene._edit_items), win.panel.hop_table.rowCount()) == before
    win.lattice_add_hop_btn.setChecked(False)
    QApplication.processEvents()
    assert not scene.hop_creation_mode
    assert scene._hop_start is None

    guide = next(item for item in scene.items() if isinstance(item, QGraphicsLineItem)
                 and item.data(0) == "hopping-guide")
    QTest.mouseClick(view.viewport(), Qt.LeftButton,
                     pos=view.mapFromScene(guide.line().center()))
    QApplication.processEvents()
    assert (len(scene.items()), len(scene._edit_items), win.panel.hop_table.rowCount()) == before

    proxy = next(item for item in scene.items()
                 if isinstance(item, QGraphicsProxyWidget))
    QTest.mouseClick(view.viewport(), Qt.LeftButton,
                     pos=view.mapFromScene(proxy.sceneBoundingRect().center()))
    QApplication.processEvents()
    assert (len(scene.items()), len(scene._edit_items), win.panel.hop_table.rowCount()) == before


def test_plain_site_click_is_non_mutating_for_all_default_templates():
    """普通点击在每个内置模型上都只选择，不让格点或连线凭空消失。"""
    _app, win, ctrl = _window()
    win.resize(1200, 800)
    win.show()
    QApplication.processEvents()
    for template_name in TEMPLATE_NAMES:
        ctrl.apply_document(template_document(
            template_name, boundary_kind="semi", nx=4, ny=3,
            connectivity="最近邻",
        ))
        if win.lattice_mode_btn.isChecked():
            win.lattice_mode_btn.setChecked(False)
        win.lattice_mode_btn.setChecked(True)
        ctrl.fit_all(force=True)
        QApplication.processEvents()
        scene, view = win.lattice_scene, win.lattice_gv
        before = deepcopy(ctrl.current_document())
        assert scene._edit_items, template_name
        site_index, handle = next(iter(scene._edit_items.items()))
        QTest.mouseClick(
            view.viewport(), Qt.LeftButton,
            pos=view.mapFromScene(handle.scenePos()),
        )
        QApplication.processEvents()
        assert ctrl.current_document() == before, template_name
        assert site_index in scene._edit_items, template_name
        assert scene._hop_start is None, template_name
        assert not scene.hop_creation_mode, template_name
        win.lattice_mode_btn.setChecked(False)


def test_real_press_move_release_drag_updates_site_and_keeps_hops():
    """A real viewport drag must edit one site without dropping topology."""
    _app, win, ctrl = _window()
    win.resize(1200, 800)
    win.show()
    QApplication.processEvents()
    for template_name in TEMPLATE_NAMES:
        ctrl.apply_document(template_document(
            template_name, boundary_kind="semi", nx=4, ny=3,
            connectivity="最近邻",
        ))
        win.lattice_mode_btn.setChecked(True)
        ctrl.fit_all(force=True)
        QApplication.processEvents()
        scene, view = win.lattice_scene, win.lattice_gv
        before = deepcopy(ctrl.current_document())
        assert scene._edit_items, template_name
        site_index, handle = next(iter(scene._edit_items.items()))
        point = view.mapFromScene(handle.scenePos())
        moved_point = point + QPoint(80, 35)
        device = QPointingDevice.primaryPointingDevice()

        def send(kind, pos, button, buttons):
            event = QMouseEvent(
                kind, QPointF(pos), QPointF(pos), QPointF(pos),
                button, buttons, Qt.NoModifier,
                Qt.MouseEventSynthesizedByApplication, device,
            )
            QApplication.sendEvent(view.viewport(), event)
            QApplication.processEvents()

        QTest.mousePress(view.viewport(), Qt.LeftButton, Qt.NoModifier, point)
        QApplication.processEvents()
        send(QEvent.MouseMove, moved_point, Qt.NoButton, Qt.LeftButton)
        QTest.mouseRelease(view.viewport(), Qt.LeftButton, Qt.NoModifier, moved_point)
        QApplication.processEvents()

        after = ctrl.current_document()
        assert after["sites"] != before["sites"], template_name
        assert len(after["hops"]) == len(before["hops"]), template_name
        assert site_index < len(after["sites"]), template_name
        # A drag is a committed edit, not a latent pointer state.
        assert not scene._hop_creation_mode, template_name
        win._restore_edit_baseline()
        QApplication.processEvents()
        assert ctrl.current_document()["sites"] == before["sites"], template_name
        assert len(ctrl.current_document()["hops"]) == len(before["hops"]), template_name
        win.lattice_mode_btn.setChecked(False)
        QApplication.processEvents()


def test_edit_handle_hit_shape_does_not_cover_neighbouring_sites():
    """A handle's clickable area must match its disc, not its outline box."""
    _app, win, ctrl = _window()
    win.resize(1200, 800)
    win.show()
    QApplication.processEvents()
    ctrl.apply_document(template_document(
        "蜂窝", boundary_kind="semi", nx=4, ny=3,
        connectivity="最近邻",
    ))
    win.lattice_mode_btn.setChecked(True)
    ctrl.fit_all(force=True)
    QApplication.processEvents()
    scene, view = win.lattice_scene, win.lattice_gv
    handles = list(scene._edit_items.items())
    assert len(handles) >= 2
    first_index, first = handles[0]
    second_index, second = handles[1]
    first_point = first.scenePos()
    second_point = second.scenePos()
    # The two primitive-cell sites are one bond length apart.  Only the
    # handle centred at the queried scene point may be returned as editable.
    at_first = [item for item in scene.items(first_point)
                if isinstance(item, type(first))]
    at_second = [item for item in scene.items(second_point)
                 if isinstance(item, type(second))]
    assert [item.site_index for item in at_first] == [first_index]
    assert [item.site_index for item in at_second] == [second_index]
    # Also guard the user-visible drag path: a press on the first handle must
    # not commit a change to the neighbouring row.
    before = deepcopy(ctrl.current_document()["sites"])
    point = view.mapFromScene(first_point)
    QTest.mousePress(view.viewport(), Qt.LeftButton, Qt.NoModifier, point)
    QApplication.processEvents()
    moved = point + QPoint(36, 18)
    device = QPointingDevice.primaryPointingDevice()
    event = QMouseEvent(
        QEvent.MouseMove, QPointF(moved), QPointF(moved), QPointF(moved),
        Qt.NoButton, Qt.LeftButton, Qt.NoModifier,
        Qt.MouseEventSynthesizedByApplication, device,
    )
    QApplication.sendEvent(view.viewport(), event)
    QApplication.processEvents()
    QTest.mouseRelease(view.viewport(), Qt.LeftButton, Qt.NoModifier, moved)
    QApplication.processEvents()
    after = ctrl.current_document()["sites"]
    assert after[first_index] != before[first_index]
    assert after[second_index] == before[second_index]


def test_edit_position_constraint_handles_oblique_cells_without_affecting_standalone_scene():
    """Oblique cells clamp in fractional coordinates; lightweight scenes stay free."""
    QApplication.instance() or QApplication([])
    scene = LatticeView()
    scene.set_edit_context(
        [(0.1, 0.1, "A"), (0.4, 0.2, "B")],
        cell_vectors=((2.0, 0.0), (0.75, 1.5)),
        anchor_offset=(0.3, -0.2),
    )
    constrained = scene.constrain_edit_position(QPointF(4.0, -4.0))
    x = constrained.x() - 0.3
    y = -constrained.y() + 0.2
    det = 2.0 * 1.5
    u = (x * 1.5 - y * 0.75) / det
    v = (2.0 * y) / det
    assert 0.0 <= u < 1.0
    assert 0.0 <= v < 1.0

    free_scene = LatticeView()
    free_scene.set_edit_context([(0.0, 0.0, "A")])
    point = QPointF(4.0, -4.0)
    assert free_scene.constrain_edit_position(point) == point


def test_explicit_bond_tool_rebuilds_without_losing_sites(monkeypatch):
    """A completed visual bond action is one safe, self-terminating edit."""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document("SC", connectivity="仅格点"))
    win.resize(1200, 800)
    win.show()
    QApplication.processEvents()
    win.lattice_mode_btn.setChecked(True)
    QApplication.processEvents()

    dialog_calls = []

    class _AcceptedDialog:
        def __init__(self, *_args, **_kwargs):
            dialog_calls.append((_args, _kwargs))

        def exec(self):
            return QDialog.Accepted

        @staticmethod
        def row(from_site, to_site):
            return [
                "t", from_site, to_site,
                # Exercise the actual half-infinite inter-cell path rather
                # than only proving that an intra-cell row can be appended.
                1, 0, "-1", "none", "0", 1,
            ]

    monkeypatch.setattr("hamivisualizer.controller.HoppingDialog", _AcceptedDialog)
    scene = win.lattice_scene
    original_sites = len(scene._data.sites)
    original_rows = win.panel.hop_table.rowCount()
    win.lattice_add_hop_btn.setChecked(True)
    scene.activate_site(0)
    scene.activate_site(1)
    # The panel emits a debounced input change just like a real dialog accept.
    QTest.qWait(420)
    QApplication.processEvents()
    assert not scene.hop_creation_mode
    assert scene._hop_start is None
    assert win.panel.hop_table.rowCount() == original_rows + 1
    assert len(scene._data.sites) == original_sites
    assert len(scene._edit_items) == len(win.panel.get_site_rows())
    assert dialog_calls and dialog_calls[-1][1].get("semi") is True
    assert win.panel.get_hop_rows()[-1]["off_x"] == 1
    assert win.panel.get_hop_rows()[-1]["off_y"] == 0
    assert win.panel.hop_advanced_check.isChecked()
    assert "x 方向胞间" in win.panel.hop_relation_hint.text()


def test_semi_ghost_endpoint_derives_intercell_offset_without_stealing_plain_clicks():
    """半无限虚影只在显式建键工具中可点，并自动得到 dx。"""
    app = QApplication.instance() or QApplication([])
    scene = LatticeView()
    scene.set_edit_context(
        [(0.0, 0.0, "A")],
        hops=(),
        cell_vectors=((1.0, 0.0), (0.0, 1.0)),
    )
    scene.set_data(LatticeSceneData(
        sites=((0.0, 0.0, "1:1", "A"),),
        semi=True,
        ghost=((1.0, 0.0, "1:1"),),
        ghost_sites=((1.0, 0.0, "1:1", 0, 1, 0),),
    ))
    assert scene._ghost_items
    assert scene._ghost_items[0].acceptedMouseButtons() == Qt.NoButton
    events = []
    scene.hoppingRequestedWithOffset.connect(lambda *args: events.append(args))
    scene.set_edit_mode(True)
    scene.set_hop_creation_mode(True)
    assert scene._ghost_items[0].acceptedMouseButtons() == Qt.LeftButton
    assert scene._ghost_items[0].data(0) == "ghost-endpoint"
    scene.activate_site(0)
    scene.activate_ghost(0, 1, 0)
    assert events == [(0, 0, 1, 0)]
    assert not scene.hop_creation_mode
    assert scene._hop_start is None


def test_semi_ghost_endpoint_can_create_reverse_offset_when_clicked_first():
    """从左侧虚影开始、回到中心格点应得到等价的正向 dx。"""
    QApplication.instance() or QApplication([])
    scene = LatticeView()
    scene.set_edit_context(
        [(0.0, 0.0, "A")],
        cell_vectors=((1.0, 0.0), (0.0, 1.0)),
    )
    scene.set_data(LatticeSceneData(
        sites=((0.0, 0.0, "1:1", "A"),), semi=True,
        ghost_sites=((-1.0, 0.0, "1:1", 0, -1, 0),),
    ))
    events = []
    scene.hoppingRequestedWithOffset.connect(lambda *args: events.append(args))
    scene.set_edit_mode(True)
    scene.set_hop_creation_mode(True)
    scene.activate_ghost(0, -1, 0)
    scene.activate_site(0)
    assert events == [(0, 0, 1, 0)]


def test_generated_semi_ghost_metadata_uses_primitive_site_indices_and_labels():
    """模型 DTO 的虚影元数据不应泄漏 ribbon 展开的绝对矩阵索引。"""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document("蜂窝", boundary_kind="semi", ny=4))
    data = win.lattice_scene._data
    assert data.ghost_sites
    site_count = len(win.panel.get_site_rows())
    assert all(0 <= int(item[3]) < site_count for item in data.ghost_sites)
    assert all(int(item[4]) != 0 for item in data.ghost_sites)
    assert all(len(item) == 6 for item in data.ghost_sites)


def test_generated_ghost_endpoint_keeps_vertical_cell_offset_for_canvas_hops(monkeypatch):
    """A ghost in another ribbon row must carry its dy, not only its dx."""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document("蜂窝", boundary_kind="semi", ny=4))
    scene = win.lattice_scene
    # The real controller opens the hop editor after two endpoints are
    # selected.  This geometry-only assertion must not leave a modal dialog
    # waiting for human input during the automated regression suite.
    class _AcceptedDialog:
        def __init__(self, *_args, **_kwargs):
            pass

        def exec(self):
            return QDialog.Accepted

        @staticmethod
        def row(from_site, to_site):
            return ["t", from_site, to_site, 1, 0, "-1", "none", "0", 1]

    monkeypatch.setattr("hamivisualizer.controller.HoppingDialog", _AcceptedDialog)
    scene.set_edit_mode(True)
    events = []
    scene.hoppingRequestedWithOffset.connect(lambda *args: events.append(args))
    # The editor anchor is the middle y-cell.  Pick the same primitive site
    # in the row immediately below it on the right periodic image.
    candidate = next(
        item for item in scene._data.ghost_sites
        if int(item[3]) == 0 and int(item[4]) == 1 and int(item[5]) == -1
    )
    scene.set_hop_creation_mode(True)
    scene.activate_site(0)
    scene.activate_ghost(0, int(candidate[4]), int(candidate[5]))
    assert events == [(0, 0, 1, -1)]


def test_dense_hop_editor_uses_progressive_disclosure_until_requested():
    """Kagome starts as a nearest-neighbour editing layer, not a line mesh."""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document(
        "Kagome", connectivity="最近邻+次近邻", boundary_kind="obc", nx=4, ny=4,
    ))
    win.resize(1200, 800)
    win.show()
    QApplication.processEvents()
    win.lattice_mode_btn.setChecked(True)
    ctrl.fit_all(force=True)
    QApplication.processEvents()
    scene, view = win.lattice_scene, win.lattice_gv
    assert len(scene._editable_hops()) > 3
    primary_rows = scene._primary_editable_rows(scene._editable_hops())
    assert primary_rows
    assert {guide.row for guide in scene._edit_guides} == primary_rows
    assert not any(item.data(0) == "physical-edge-nnn" for item in scene.items())
    assert scene._edit_proxies == []

    guide = scene._edit_guides[0]
    QTest.mouseClick(view.viewport(), Qt.LeftButton,
                     pos=view.mapFromScene(guide.line().center()))
    QApplication.processEvents()
    assert scene.active_hop_row in primary_rows
    assert len(scene._edit_proxies) == 1

    scene.set_show_edit_details(True)
    QApplication.processEvents()
    assert len(scene._edit_guides) == len(scene._editable_hops())
    assert any(item.data(0) == "physical-edge-nnn" for item in scene.items())
    scene.set_show_all_hop_editors(True)
    QApplication.processEvents()
    assert len(scene._edit_proxies) == len(scene._editable_hops())


def test_dense_all_coefficient_editors_hide_crossing_leaders_until_inspected():
    """All fields stay editable while only the inspected bond gets a leader."""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document(
        "Kagome", connectivity="最近邻+次近邻", boundary_kind="obc", nx=4, ny=4,
    ))
    win.resize(1200, 800)
    win.show()
    QApplication.processEvents()
    win.lattice_mode_btn.setChecked(True)
    ctrl.fit_all(force=True)
    QApplication.processEvents()

    scene = win.lattice_scene
    scene.set_show_all_hop_editors(True)
    QApplication.processEvents()
    links = scene._edit_leader_links
    assert len(links) > 3
    # The rail is fully usable immediately, but the canvas starts quiet: no
    # diagonal fan should cover the lattice before the user inspects a row.
    assert not any(diagonal.isVisible() or horizontal.isVisible()
                   for diagonal, horizontal, *_rest in links)

    selected_row = int(links[0][4].data(1))
    scene._set_hovered_hop_row(selected_row)
    assert all(diagonal.isVisible() == (int(proxy.data(1)) == selected_row)
               and horizontal.isVisible() == (int(proxy.data(1)) == selected_row)
               for diagonal, horizontal, _mx, _my, proxy in links)

    scene._set_hovered_hop_row(None)
    assert not any(diagonal.isVisible() or horizontal.isVisible()
                   for diagonal, horizontal, *_rest in links)


def test_all_coefficient_visibility_does_not_enable_physical_long_range_layer():
    """The coefficient rail and NNN visual layer remain independently opt-in."""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document(
        "Kagome", connectivity="最近邻+次近邻", boundary_kind="obc", nx=4, ny=4,
    ))
    win.resize(1200, 800)
    win.show()
    QApplication.processEvents()
    win.lattice_mode_btn.setChecked(True)
    QApplication.processEvents()
    scene = win.lattice_scene
    assert not scene.show_edit_details
    assert not any(item.data(0) == "physical-edge-nnn" for item in scene.items())

    # Showing all values must keep the clean nearest-neighbour canvas.
    win.lattice_coeff_btn.setChecked(True)
    QApplication.processEvents()
    assert scene.show_all_hop_editors
    assert not scene.show_edit_details
    assert not any(item.data(0) == "physical-edge-nnn" for item in scene.items())

    # The separate details toggle can still reveal/hide the real long-range
    # physics without dismissing the coefficient rail.
    win.lattice_details_btn.setChecked(True)
    QApplication.processEvents()
    assert scene.show_edit_details
    assert scene.show_all_hop_editors
    assert any(item.data(0) == "physical-edge-nnn" for item in scene.items())
    win.lattice_details_btn.setChecked(False)
    QApplication.processEvents()
    assert not scene.show_edit_details
    assert scene.show_all_hop_editors


def test_dense_coefficient_editor_hover_reveals_its_own_leader():
    """A real pointer move over a rail field reveals the matching bond only."""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document(
        "Kagome", connectivity="最近邻+次近邻", boundary_kind="obc", nx=4, ny=4,
    ))
    win.resize(1200, 800)
    win.show()
    QApplication.processEvents()
    win.lattice_mode_btn.setChecked(True)
    ctrl.fit_all(force=True)
    QApplication.processEvents()
    scene, view = win.lattice_scene, win.lattice_gv
    scene.set_show_all_hop_editors(True)
    QApplication.processEvents()
    links = scene._edit_leader_links
    assert len(links) > 3
    # Pick a field with a full margin around it.  The final field can be
    # partially adjacent to the viewport edge on smaller CI geometries;
    # testing the first row keeps the pointer unambiguously inside one
    # screen-space rectangle while still exercising the dense-rail path.
    target_proxy = links[0][4]
    target_row = int(target_proxy.data(1))
    # Exercise the same path as a user moving the pointer across the canvas.
    # Sending directly to the embedded child bypasses QGraphicsView's device
    # hit-test on the offscreen Qt platform and does not represent real use.
    top_left = view.mapFromScene(target_proxy.pos())
    widget = target_proxy.widget()
    point = top_left + QPoint(widget.width() // 2, widget.height() // 2)
    # QApplication.sendEvent keeps this a real viewport mouse event while
    # avoiding QTest's platform-dependent global-cursor bookkeeping when the
    # complete suite has several offscreen windows alive at once.
    event = QMouseEvent(
        QEvent.MouseMove, QPointF(point), QPointF(point), QPointF(point),
        Qt.NoButton, Qt.NoButton, Qt.NoModifier,
        Qt.MouseEventSynthesizedByApplication,
        QPointingDevice.primaryPointingDevice(),
    )
    QApplication.sendEvent(view.viewport(), event)
    QApplication.processEvents()
    assert scene.hovered_hop_row == target_row
    assert all(diagonal.isVisible() == (int(proxy.data(1)) == target_row)
               and horizontal.isVisible() == (int(proxy.data(1)) == target_row)
               for diagonal, horizontal, _mx, _my, proxy in links)


def test_dense_edit_mode_hides_details_until_explicitly_revealed():
    """Dense editing is structure-first; NNN remains an explicit detail layer."""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document(
        "Kagome", connectivity="最近邻+次近邻", boundary_kind="obc", nx=4, ny=4,
    ))
    win.resize(1200, 800)
    win.show()
    QApplication.processEvents()
    win.lattice_mode_btn.setChecked(True)
    QApplication.processEvents()
    scene = win.lattice_scene
    nn = [item for item in scene.items()
          if item.data(0) == "physical-edge-nn"]
    assert nn
    assert not any(item.data(0) == "physical-edge-nnn" for item in scene.items())
    assert all(item.opacity() == pytest.approx(1.0) for item in nn)

    scene.set_show_edit_details(True)
    QApplication.processEvents()
    nnn = [item for item in scene.items()
           if item.data(0) == "physical-edge-nnn"]
    assert nnn
    assert all(item.opacity() == pytest.approx(0.24) for item in nnn)

    # Leaving edit mode restores the normal presentation preference without
    # mutating the actual model or hiding long-range physics from inspection.
    win.lattice_mode_btn.setChecked(False)
    QApplication.processEvents()
    assert all(item.opacity() == pytest.approx(0.38)
               for item in scene.items() if item.data(0) == "physical-edge-nnn")


def test_reentering_lattice_edit_mode_resets_previous_dense_coefficient_layer():
    """Each edit session should reopen compact, not inherit stale overlays."""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document(
        "Kagome", connectivity="最近邻+次近邻", boundary_kind="obc", nx=4, ny=4,
    ))
    win.resize(1200, 800)
    win.show()
    QApplication.processEvents()
    win.lattice_mode_btn.setChecked(True)
    QApplication.processEvents()
    scene = win.lattice_scene
    win.lattice_coeff_btn.setChecked(True)
    QApplication.processEvents()
    assert scene.show_all_hop_editors
    assert win.lattice_coeff_btn.isChecked()

    # Finish and start a genuinely new edit session.  The old visual detail
    # preference must not silently return with the next session.
    win.lattice_mode_btn.setChecked(False)
    QApplication.processEvents()
    win.lattice_mode_btn.setChecked(True)
    QApplication.processEvents()
    assert not scene.show_all_hop_editors
    assert not win.lattice_coeff_btn.isChecked()
    assert win.lattice_coeff_btn.text() == "系数：点击编辑"


def test_all_default_templates_keep_edit_layer_and_detail_layer_consistent():
    """Every template gets the same readable-editing contract, not Kagome alone."""
    _app, win, ctrl = _window()
    win.resize(1200, 800)
    win.show()
    QApplication.processEvents()
    for template_name in TEMPLATE_NAMES:
        ctrl.apply_document(template_document(
            template_name, connectivity="最近邻+次近邻", boundary_kind="obc",
            nx=3, ny=3,
        ))
        if win.lattice_mode_btn.isChecked():
            win.lattice_mode_btn.setChecked(False)
        win.lattice_mode_btn.setChecked(True)
        QApplication.processEvents()
        scene = win.lattice_scene
        editable = scene._editable_hops()
        primary = scene._primary_editable_rows(editable)
        assert {guide.row for guide in scene._edit_guides} == primary, template_name
        # The global display preference remains checked, but the edit-layer
        # filter suppresses NNN until the explicit toolbar command.
        assert not any(item.data(0) == "physical-edge-nnn" for item in scene.items()), template_name
        assert not scene.show_edit_details

        win.lattice_details_btn.setChecked(True)
        QApplication.processEvents()
        assert scene.show_edit_details
        assert {guide.row for guide in scene._edit_guides} == {
            int(hop["row"]) for hop in scene._editor_representative_hops(editable)
        }, template_name
        if len(editable) > len(primary):
            assert any(item.data(0) == "physical-edge-nnn" for item in scene.items()), template_name
        win.lattice_mode_btn.setChecked(False)


@pytest.mark.parametrize("shape", ("rectangle", "triangle", "disk", "hexagon"))
def test_every_default_template_builds_each_finite_shape_without_detached_editing(shape):
    """All shipped templates obey finite-mask and editing invariants.

    This guards against a tempting but incorrect "Kagome-only" fix: the
    finite boundary builder, render DTO, central edit anchor and matrix order
    are shared infrastructure and must work for every preset.
    """
    _app, win, ctrl = _window()
    for template_name in TEMPLATE_NAMES:
        ctrl.apply_document(template_document(
            template_name, connectivity="最近邻+次近邻", boundary_kind="obc",
            nx=4, ny=4, shape=shape,
        ))
        boundary = ctrl._build_boundary()
        result = ctrl._state[0]
        data = win.lattice_scene._data
        sites_per_cell = len(win.panel.get_site_rows())
        active_cells = boundary.active_cells()
        assert result.Ncells == len(active_cells) > 0, (template_name, shape)
        assert len(data.sites) == sites_per_cell * len(active_cells), (template_name, shape)
        matrix = np.asarray(result.H, dtype=complex)
        expected_dimension = sites_per_cell * len(active_cells)
        assert matrix.shape == (expected_dimension, expected_dimension), (template_name, shape)
        assert np.allclose(matrix, matrix.conj().T, atol=1e-10), (template_name, shape)

        if win.lattice_mode_btn.isChecked():
            win.lattice_mode_btn.setChecked(False)
        win.lattice_mode_btn.setChecked(True)
        QApplication.processEvents()
        rendered_positions = [(float(x), float(y)) for x, y, *_ in data.sites]
        for item in win.lattice_scene._edit_items.values():
            handle = (float(item.pos().x()), float(-item.pos().y()))
            assert any(
                math.isclose(handle[0], point[0], abs_tol=1e-8)
                and math.isclose(handle[1], point[1], abs_tol=1e-8)
                for point in rendered_positions
            ), (template_name, shape)
        win.lattice_mode_btn.setChecked(False)


@pytest.mark.parametrize("shape", ("rectangle", "triangle", "disk", "hexagon"))
def test_kagome_each_finite_shape_keeps_topology_and_edit_details_separate(shape):
    """Kagome disks are real finite masks, not a rectangular rendering shortcut."""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document(
        "Kagome", connectivity="最近邻+次近邻", boundary_kind="obc",
        nx=4, ny=4, shape=shape,
    ))
    boundary = ctrl._build_boundary()
    result = ctrl._state[0]
    data = win.lattice_scene._data
    active_cells = boundary.active_cells()
    assert result.Ncells == len(active_cells) > 0
    assert len(data.sites) == len(win.panel.get_site_rows()) * len(active_cells)
    assert len(data.cell_polygons) == len(active_cells)

    win.lattice_mode_btn.setChecked(True)
    QApplication.processEvents()
    scene = win.lattice_scene
    assert not any(item.data(0) == "physical-edge-nnn" for item in scene.items())
    # Primitive-cell edit handles must be placed on a real active finite
    # cell.  In a disk/hexagon the historical (0, 0) copy can be masked out,
    # leaving controls detached from the sample at a screen corner.
    rendered_positions = [(float(x), float(y)) for x, y, *_ in data.sites]
    for item in scene._edit_items.values():
        handle = (float(item.pos().x()), float(-item.pos().y()))
        assert any(
            math.isclose(handle[0], point[0], abs_tol=1e-8)
            and math.isclose(handle[1], point[1], abs_tol=1e-8)
            for point in rendered_positions
        ), shape
    win.lattice_details_btn.setChecked(True)
    QApplication.processEvents()
    assert any(item.data(0) == "physical-edge-nnn" for item in win.lattice_scene.items())


def test_combined_matrix_lattice_page_has_draggable_splitter():
    _app, win, _ctrl = _window()
    assert isinstance(win.combined_splitter, QSplitter)
    assert win.combined_splitter.orientation() == Qt.Horizontal
    win.resize(1200, 800)
    win.show()
    QApplication.processEvents()
    win.combined_splitter.setSizes([300, 700])
    assert win.combined_splitter.sizes()[0] > 0
    handle = win.combined_splitter.handle(1)
    assert handle is not None and handle.isEnabled()
    before = win.combined_splitter.sizes()[0]
    win.combined_splitter.moveSplitter(before + 80, 1)
    after = win.combined_splitter.sizes()[0]
    assert after > before


def test_close_with_unsaved_changes_asks_even_when_autosave_is_enabled(monkeypatch):
    _app, win, _ctrl = _window()
    win._dirty = True
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Cancel,
    )
    event = QCloseEvent()
    win.closeEvent(event)
    assert not event.isAccepted()


def test_matrix_text_is_lazy_until_zoomed():
    QApplication.instance() or QApplication([])
    n = 10
    matrix = np.eye(n, dtype=complex)
    matrix[0, 1] = matrix[1, 0] = -1
    scene = MatrixView()
    scene.set_data(MatrixSceneData(n=n, values=matrix, matrix=matrix, mode="numeric"))
    assert not scene._text_items
    scene.set_zoom_level(2.0)
    text_items = list(scene._text_items)
    assert text_items and all(item.isVisible() for item in text_items)


def test_matrix_smart_labels_accept_k_x_symbol_alias():
    QApplication.instance() or QApplication([])
    k_x = sp.Symbol("k_x", real=True)
    raw = sp.Matrix([[0, sp.exp(sp.I * k_x)], [sp.exp(-sp.I * k_x), 0]])
    scene = MatrixView()
    scene.set_data(MatrixSceneData(
        n=2,
        values=np.array([[0, 1], [1, 0]], dtype=complex),
        matrix=raw,
        mode="smart",
        formatter=ElementFormatter(t=1.0),
    ))
    assert "k_{x}" in scene._cell_text_at(0, 1)


def test_zoom_view_scales_and_notifies_scene():
    QApplication.instance() or QApplication([])
    scene = MatrixView()
    view = ZoomGraphicsView(scene)
    view.resize(500, 500)
    scene.set_data(MatrixSceneData(n=2, values=np.eye(2), matrix=np.eye(2)))
    view.fitInView(scene.sceneRect())
    before = abs(view.transform().m11())
    view._scale_by(2.0)
    assert view.user_zoomed
    assert abs(view.transform().m11()) > before * 1.9
    assert scene.text_visible


def test_matrix_rulers_follow_the_visible_viewport():
    """行列号由视口覆盖层绘制，放大到矩阵内部仍返回当前行列。"""
    QApplication.instance() or QApplication([])
    n = 30
    matrix = np.eye(n, dtype=complex)
    scene = MatrixView()
    view = ZoomGraphicsView(scene)
    view.resize(600, 480)
    view.show()
    scene.set_data(MatrixSceneData(
        n=n,
        values=matrix,
        matrix=matrix,
        sites=tuple(f"site-{i}" for i in range(n)),
    ))
    QApplication.processEvents()
    view.fitInView(scene.sceneRect())
    view._scale_by(5.0)
    center = scene._matrix_rect.center()
    view.centerOn(center)
    QApplication.processEvents()

    rows, cols = scene.visible_axis_indices(view)
    assert rows and cols
    assert min(rows) > 0 and max(rows) < n - 1
    assert min(cols) > 0 and max(cols) < n - 1
    assert any(abs(i - n // 2) <= 2 for i in rows)
    assert any(abs(j - n // 2) <= 2 for j in cols)


def test_matrix_edge_labels_are_hidden_as_whole_items_under_frozen_rulers():
    """边缘长表达式不应被四边冻结标尺裁成半个字符。"""
    QApplication.instance() or QApplication([])
    n = 6
    matrix = np.eye(n, dtype=complex)
    kx = sp.Symbol("k_x", real=True)
    raw = sp.zeros(n)
    for i in range(n):
        raw[i, i] = sp.Symbol("omega")
        if i + 1 < n:
            raw[i, i + 1] = -sp.exp(sp.I * kx)
            raw[i + 1, i] = -sp.exp(-sp.I * kx)
    scene = MatrixView()
    view = ZoomGraphicsView(scene)
    view.resize(520, 420)
    view.show()
    scene.set_data(MatrixSceneData(
        n=n,
        values=matrix,
        matrix=raw,
        mode="smart",
    ))
    QApplication.processEvents()
    view.fitInView(scene.sceneRect())
    view._scale_by(5.0)
    view.centerOn(scene._matrix_rect.center())
    QApplication.processEvents()
    scene._mask_text_under_rulers(view, 42, 24)
    visible = [item for item in scene._text_items if item.isVisible()]
    assert visible
    viewport = view.viewport().rect()
    for item in visible:
        rect = item.mapToScene(item.boundingRect()).boundingRect()
        top_left = view.mapFromScene(rect.topLeft())
        bottom_right = view.mapFromScene(rect.bottomRight())
        pixel_rect = QRectF(top_left, bottom_right).normalized()
        assert pixel_rect.left() >= 42 or pixel_rect.right() <= viewport.width() - 42
        assert pixel_rect.top() >= 24 or pixel_rect.bottom() <= viewport.height() - 24


def test_large_matrix_overlay_text_scales_with_zoom():
    QApplication.instance() or QApplication([])
    n = 70  # rasterized path (> TEXT_ITEM_MAX)
    scene = MatrixView()
    view = ZoomGraphicsView(scene)
    view.resize(700, 520)
    view.show()
    scene.set_data(MatrixSceneData(n=n, values=np.eye(n), matrix=np.eye(n)))
    QApplication.processEvents()
    view.fitInView(scene.sceneRect())
    before = scene._overlay_text_pixel_size(view)
    view._scale_by(8.0)
    after = scene._overlay_text_pixel_size(view)
    assert after > before
    assert scene.text_visible_in_view(view)


def test_large_matrix_overlay_keeps_bloch_subscript_inside_the_exponent():
    """Rasterized matrices must not let x fall below the exponent baseline."""
    QApplication.instance() or QApplication([])
    scene = MatrixView()
    layout, font = scene._overlay_math_layout("-t·e^{-ik_{x}}", 24)
    outer = next(node for node in layout.lines[0].children
                 if node.__class__.__name__ == "_Script")
    assert outer.superscript is not None
    inner = next(node for node in outer.superscript.children
                 if node.__class__.__name__ == "_Script")
    assert inner.subscript is not None
    # The subscript is lowered only relative to k's exponent baseline.  Its
    # combined baseline stays above the primary line, unlike nested HTML.
    outer_raise = outer._offsets(font)[0]
    inner_drop = inner._offsets(outer._script_font(font))[1]
    assert outer_raise + inner_drop < 0
    assert layout.metrics(font).width > 0
    # Repaints reuse the parsed layout instead of reparsing every cell.
    assert scene._overlay_math_layout("-t·e^{-ik_{x}}", 24)[0] is layout
    scene.set_theme(True)
    dark_layout, _dark_font = scene._overlay_math_layout("-t·e^{-ik_{x}}", 24)
    assert dark_layout is not layout


def test_lattice_zoom_uses_fit_relative_limits():
    """晶格坐标只有几个单位时，第一次放大不能被绝对上限夹回小尺寸。"""
    QApplication.instance() or QApplication([])
    scene = LatticeView()
    scene.setSceneRect(0, 0, 2, 2)
    view = ZoomGraphicsView(scene)
    view.resize(800, 600)
    view.show()
    QApplication.processEvents()
    view.fitInView(scene.sceneRect())
    before = abs(view.transform().m11())
    view._scale_by(view.WHEEL_STEP)
    after = abs(view.transform().m11())
    assert after > before * 1.1
    assert view._reference_scale == pytest.approx(before, rel=1e-6)


def test_lattice_labels_scale_with_view_and_sites_stay_compact():
    QApplication.instance() or QApplication([])
    scene = LatticeView()
    scene.set_data(LatticeSceneData(
        sites=((0.0, 0.0, "12", "A"), (1.0, 0.0, "13", "B")),
        edges=((0, 1, "NN"),),
    ))
    circles = [item for item in scene.items()
               if isinstance(item, QGraphicsEllipseItem)]
    labels = [item for item in scene.items()
              if isinstance(item, QGraphicsTextItem)]
    assert circles and max(item.rect().width() for item in circles) <= 0.45
    assert labels
    assert all(not (item.flags() & QGraphicsItem.ItemIgnoresTransformations)
               for item in labels)


def test_edit_canvas_draws_snap_grid_nodes_and_toolbar_spacing_updates_them():
    """编辑态背景网格与吸附间距一致，改间距立即刷新且不改拓扑。"""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document(
        "空白自定义", boundary_kind="semi", connectivity="最近邻",
    ))
    win.resize(1200, 800)
    win.show()
    QApplication.processEvents()
    win.lattice_mode_btn.setChecked(True)
    QApplication.processEvents()
    scene = win.lattice_scene
    assert scene.grid_visible
    assert "细实环=当前位置" in win.statusBar().currentMessage()
    assert "虚线环=编辑前位置" in win.statusBar().currentMessage()
    assert "虚线环表示进入编辑前可恢复的位置" in win.lattice_grid_btn.toolTip()
    assert win.lattice_grid_btn.text() == "网格点"
    assert win.lattice_snap_step_widget.findChildren(QLabel)[0].text() == "间距"
    dots = [item for item in scene.items()
            if item.data(0) == "snap-grid-node"]
    assert dots and len(dots) <= 1800
    before = len(dots)
    win.lattice_snap_step_spin.setValue(0.5)
    QApplication.processEvents()
    assert scene.snap_step == pytest.approx(0.5)
    after = [item for item in scene.items()
             if item.data(0) == "snap-grid-node"]
    assert after and len(after) < before
    win.lattice_grid_btn.setChecked(False)
    QApplication.processEvents()
    assert not scene.grid_visible
    assert not any(item.data(0) == "snap-grid-node" for item in scene.items())
    win.lattice_grid_btn.setChecked(True)
    assert scene.grid_visible
    assert any(item.data(0) == "snap-grid-node" for item in scene.items())


def test_snap_grid_stays_in_the_editable_cell_work_area_not_the_coefficient_rail():
    """网格点只覆盖当前原始元胞周围，避免淹没周期虚影和引线。"""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document(
        "蜂窝", boundary_kind="semi", connectivity="最近邻", nx=4, ny=4,
    ))
    win.resize(1440, 920)
    win.show()
    QApplication.processEvents()
    win.lattice_mode_btn.setChecked(True)
    ctrl.fit_all(force=True)
    QApplication.processEvents()
    scene = win.lattice_scene
    bounds = scene._snap_grid_bounds()
    grid = [item for item in scene.items()
            if item.data(0) == "snap-grid-node"]
    assert grid
    assert all(bounds.contains(item.sceneBoundingRect().center()) for item in grid)
    # The scene includes periodic ghost columns and a right-hand editor rail;
    # the compact work area must be strictly smaller than that full canvas.
    assert bounds.width() < scene.sceneRect().width()
    assert bounds.height() < scene.sceneRect().height()


@pytest.mark.parametrize("template_name", ("蜂窝", "三角", "Kagome"))
def test_oblique_snap_grid_contains_only_real_bravais_cell_targets(template_name):
    """斜元胞包围框外的装饰点不能伪装成可吸附目标。"""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document(
        template_name, boundary_kind="obc", shape="rectangle",
        connectivity="最近邻", nx=4, ny=4,
    ))
    win.resize(1200, 800)
    win.show()
    QApplication.processEvents()
    win.lattice_mode_btn.setChecked(True)
    ctrl.fit_all(force=True)
    QApplication.processEvents()
    scene = win.lattice_scene
    assert scene._edit_cell_constrained
    (a1x, a1y), (a2x, a2y) = scene._cell_vectors
    det = a1x * a2y - a1y * a2x
    assert abs(det) > 1e-12
    anchor_x, anchor_y = scene._edit_anchor_offset
    grid = [item for item in scene.items()
            if item.data(0) == "snap-grid-node"]
    assert grid
    for item in grid:
        point = item.sceneBoundingRect().center()
        local_x = point.x() - anchor_x
        local_y = -point.y() - anchor_y
        u = (local_x * a2y - local_y * a2x) / det
        v = (a1x * local_y - a1y * local_x) / det
        assert -1e-8 <= u < 1.0 + 1e-8
        assert -1e-8 <= v < 1.0 + 1e-8


def test_oblique_edit_sites_have_visible_snap_targets_at_their_exact_positions():
    """斜晶格的实际格点必须落在可见网格目标上，而不是悬空。"""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document(
        "蜂窝", boundary_kind="obc", shape="rectangle",
        connectivity="最近邻", nx=4, ny=4,
    ))
    win.resize(1200, 800)
    win.show()
    QApplication.processEvents()
    win.lattice_mode_btn.setChecked(True)
    ctrl.fit_all(force=True)
    QApplication.processEvents()
    scene = win.lattice_scene
    grid = [item for item in scene.items()
            if item.data(0) == "snap-grid-node"]
    assert grid
    # Every editable site, including one that happens to coincide with a
    # regular Cartesian dot, gets the same compact alignment ring.  This is
    # what makes the visible atom center and the advertised snap target
    # unambiguous instead of relying on the atom to hide the background dot.
    anchors = [item for item in grid if item.data(1) == "site-anchor"]
    assert len(anchors) == len(scene._edit_sites)
    centers = [item.sceneBoundingRect().center() for item in grid]
    for site in scene._edit_items.values():
        target = site.scenePos()
        # Either a Cartesian target (for A) or the exact-site halo (for B)
        # must sit at the same scene coordinate.  A loose numerical tolerance
        # covers QGraphicsEllipseItem's bounding-rect quantization only.
        assert min(
            math.hypot(target.x() - center.x(), target.y() - center.y())
            for center in centers
        ) <= 1e-6


def test_oblique_snap_grid_and_rounding_share_bravais_fractional_coordinates():
    """斜元胞背景点与新增/拖动吸附必须沿同一组 a1/a2 分数网格。"""
    QApplication.instance() or QApplication([])
    root3 = math.sqrt(3.0)
    a1, a2 = (root3, 0.0), (root3 / 2.0, 1.5)
    scene = LatticeView()
    scene.set_edit_context(
        [(0.0, 0.0, "A"), (root3 / 2.0, 0.5, "B")],
        cell_vectors=(a1, a2), snap_step=0.25,
    )
    scene.set_edit_mode(True)
    scene.set_data(LatticeSceneData(
        sites=((0.0, 0.0, "1", "A"), (root3 / 2.0, 0.5, "2", "B"),
               (root3, 0.0, "3", "A"), (root3 / 2.0, 1.5, "4", "B")),
    ))
    n1, n2 = scene._bravais_grid_counts()
    grid = [item for item in scene.items()
            if item.data(1) == "bravais-grid"]
    assert grid
    det = a1[0] * a2[1] - a1[1] * a2[0]
    anchor_x, anchor_y = scene.edit_anchor_offset
    for item in grid:
        center = item.sceneBoundingRect().center()
        x = center.x() - anchor_x
        y = -center.y() - anchor_y
        u = (x * a2[1] - y * a2[0]) / det
        v = (a1[0] * y - a1[1] * x) / det
        assert u == pytest.approx(round(u * n1) / n1, abs=1e-8)
        assert v == pytest.approx(round(v * n2) / n2, abs=1e-8)

    # The helper used by both dragging and the explicit site tool lands on
    # the same fractional net, rather than independently rounding x/y.
    snapped_x, snapped_y = scene._snap_local_point(0.61, 0.49, scene.snap_step)
    u = (snapped_x * a2[1] - snapped_y * a2[0]) / det
    v = (a1[0] * snapped_y - a1[1] * snapped_x) / det
    assert u == pytest.approx(round(u * n1) / n1, abs=1e-8)
    assert v == pytest.approx(round(v * n2) / n2, abs=1e-8)


def test_edit_grid_marks_original_geometry_as_a_visible_restore_target():
    """拖动后原始位置仍有可见回归目标，并即时刷新。"""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document(
        "蜂窝", boundary_kind="obc", shape="rectangle",
        connectivity="最近邻", nx=3, ny=3,
    ))
    win.resize(1200, 800)
    win.show()
    QApplication.processEvents()
    win.lattice_mode_btn.setChecked(True)
    ctrl.fit_all(force=True)
    QApplication.processEvents()
    scene = win.lattice_scene
    baseline = list(scene._edit_sites)
    assert baseline
    moved = [(x + 0.2, y, sub) for x, y, sub in baseline]
    scene.set_edit_context(
        moved, cell_vectors=scene._cell_vectors,
        anchor_offset=scene.edit_anchor_offset,
    )
    scene.set_snap_reference_sites([(x, y) for x, y, _sub in baseline])
    QApplication.processEvents()
    markers = [item for item in scene.items()
               if item.data(0) == "snap-grid-node"
               and item.data(1) == "baseline-anchor"]
    assert markers
    anchor_x, anchor_y = scene.edit_anchor_offset
    # The A site at (0, 0) already coincides with a regular 0.25 grid dot;
    # use graphene's irrational B basis coordinate to exercise the explicit
    # dashed baseline marker path.
    expected = QPointF(float(baseline[1][0]) + anchor_x,
                       -float(baseline[1][1]) - anchor_y)
    assert any(
        math.hypot(item.sceneBoundingRect().center().x() - expected.x(),
                   item.sceneBoundingRect().center().y() - expected.y()) <= 1e-6
        for item in markers
    )


def test_user_added_long_range_hop_stays_visible_in_compact_edit_layer():
    """新加的长程跃迁不能因默认降噪而从画布上消失。"""
    _app, win, ctrl = _window()
    ctrl.apply_document(template_document(
        "蜂窝", boundary_kind="semi", connectivity="最近邻", nx=4, ny=3,
    ))
    win.lattice_mode_btn.setChecked(True)
    QApplication.processEvents()
    # Model-facing endpoints are zero-based; append_hop converts them once
    # into the human-facing one-based table and marks the row as authored.
    win.panel.append_hop(
        ["t", 0, 1, 2, 0, "-t", "none", "0", 1],
        reveal_relation=True,
    )
    ctrl.rebuild()
    QApplication.processEvents()
    scene = win.lattice_scene
    assert any(item.data(0) == "ghost-edge-nnn" for item in scene.items())
    assert any(
        item.data(0) == "hopping-editor"
        and int(item.data(1)) == len(win.panel.get_hop_rows()) - 1
        for item in scene.items()
    )


def test_site_creation_respects_snap_toggle_and_rejects_invalid_or_duplicate_points():
    """添加格点不应绕过吸附开关、元胞边界或重复坐标校验。"""
    QApplication.instance() or QApplication([])

    scene = LatticeView()
    scene.set_edit_context(
        [(0.1, 0.1, "A")],
        cell_vectors=((1.0, 0.0), (0.0, 1.0)),
        snap_step=0.25,
    )
    scene.set_edit_mode(True)
    scene.set_data(LatticeSceneData(
        sites=((0.1, 0.1, "1", "A"),),
    ))
    added = []
    messages = []
    scene.siteAddRequested.connect(lambda x, y: added.append((x, y)))
    scene.editSelectionChanged.connect(messages.append)

    # With snapping disabled, preserve the actual pointer coordinate instead
    # of silently rounding it to the toolbar step.
    scene.set_snap_enabled(False)
    scene._append_site_at(QPointF(0.37, -0.41))
    assert added[-1] == pytest.approx((0.37, 0.41))

    # Re-arm the tool for a snapped point.  The same click now lands on the
    # visible 0.25 grid, proving the toggle controls creation as well as drag.
    scene.set_snap_enabled(True)
    scene._append_site_at(QPointF(0.37, -0.41))
    assert added[-1] == pytest.approx((0.25, 0.5))

    before = len(added)
    scene._append_site_at(QPointF(1.2, -0.4))
    scene.set_snap_enabled(False)
    scene._append_site_at(QPointF(0.1, -0.1))
    assert len(added) == before
    assert any("元胞范围内" in message for message in messages)
    assert any("已有格点过近" in message for message in messages)


def test_panel_append_site_is_a_defensive_finite_duplicate_gate():
    """直接调用面板槽也不能把 NaN 或重复格点写进表格。"""
    _app, win, _ctrl = _window()
    panel = win.panel
    rows_before = panel.site_table.rowCount()
    panel.append_site(float("nan"), 0.2)
    assert panel.site_table.rowCount() == rows_before
    assert "有限数值" in panel.error_label.text()
    panel.set_error("")
    first = panel.get_site_rows()[0]
    panel.append_site(first[0], first[1])
    assert panel.site_table.rowCount() == rows_before
    assert "不能重复" in panel.error_label.text()


def test_small_custom_editor_keeps_longer_authored_bond_visible():
    """小型自定义模型不因最近邻渐进显示而丢第三条边。"""
    _app, win, ctrl = _window()
    document = template_document(
        "空白自定义", boundary_kind="semi", connectivity="仅格点",
    )
    document["sites"] = [
        {"x": 0.2, "y": 0.0, "sublattice": "A"},
        {"x": 0.6, "y": 0.4, "sublattice": "A"},
        {"x": 0.2, "y": 0.4, "sublattice": "A"},
    ]
    document["hops"] = [
        {"name": "t", "from_site": 0, "to_site": 1, "off_x": 0,
         "off_y": 0, "amplitude": "-t", "phase": "0",
         "phase_mode": "none", "phase_sign": 1},
        {"name": "t", "from_site": 1, "to_site": 2, "off_x": 0,
         "off_y": 0, "amplitude": "-t", "phase": "0",
         "phase_mode": "none", "phase_sign": 1},
        {"name": "t", "from_site": 2, "to_site": 0, "off_x": 0,
         "off_y": 0, "amplitude": "-t", "phase": "0",
         "phase_mode": "none", "phase_sign": 1},
    ]
    ctrl.apply_document(document)
    win.lattice_mode_btn.setChecked(True)
    QApplication.processEvents()
    assert sum(item.data(0) == "physical-edge-nnn" for item in win.lattice_scene.items()) >= 1


def test_wavefunction_markers_follow_site_spacing_without_covering_neighbors():
    """波函数圆点按几何间距缩放，不应遮住相邻格点。"""
    QApplication.instance() or QApplication([])
    view = WavefunctionView()
    view.set_data(WfSceneData(
        energies=np.array([0.0]),
        wf=np.array([[1.0], [0.5], [0.25], [0.75]]),
        positions=((0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.5, 0.0)),
        title="SSH",
    ))
    circles = [item for item in view.scene.items()
               if isinstance(item, QGraphicsEllipseItem)]
    assert len(circles) == 4
    assert max(item.rect().width() for item in circles) <= 0.34


def test_vector_editor_stays_within_control_rail_at_large_ui_scale():
    _app, win, _ctrl = _window()
    win.resize(1440, 920)
    win.show()
    QApplication.processEvents()
    win._set_ui_scale(1.5, persist=False)
    QApplication.processEvents()
    assert win.panel_scroll.horizontalScrollBar().maximum() == 0
    assert win.panel.a1x_spin.width() <= win.panel.a1x_spin.maximumWidth()
    assert win.panel.a2y_spin.width() <= win.panel.a2y_spin.maximumWidth()


def test_lattice_bond_shells_follow_model_scale_not_absolute_length():
    lattice = Lattice(
        [Site(0, 0.0, 0.0, "A"), Site(1, 2.0, 0.0, "B")],
        Lx=4.0, Ly=1.0,
    )
    hops = [
        HoppingTerm("t", 0, 1, (0, 0), -1.0),   # length 2: NN
        HoppingTerm("t2", 0, 0, (1, 0), -0.2), # length 4: NNN
    ]
    boundary = Boundary(BoundaryKind.OBC, NX=2, NY=1)
    result = HamiltonianBuilder(lattice, hops, boundary).build()
    data = _build_lattice_scene(lattice, hops, boundary, result)
    assert sum(edge[2] == "NN" for edge in data.edges) == 2
    assert sum(edge[2] == "NNN" for edge in data.edges) == 1


def test_lattice_edit_mode_only_labels_unit_cell_handles():
    QApplication.instance() or QApplication([])
    scene = LatticeView()
    scene.set_edit_context([
        (0.0, 0.0, "A"),
        (1.0, 0.0, "B"),
        (1.5, np.sqrt(3) / 2, "A"),
        (2.5, np.sqrt(3) / 2, "B"),
    ])
    scene.set_edit_mode(True)
    scene.set_data(LatticeSceneData(
        sites=tuple((float(i), 0.0, str(i + 13), "A") for i in range(12)),
        ghost=tuple((float(i), 1.0, str(i + 25)) for i in range(8)),
    ))
    handles = list(scene._edit_items.values())
    # Expanded labels such as 13..32 are suppressed in edit mode; unit-cell
    # indices are vector-painted by the four handles themselves.
    assert not any(isinstance(item, QGraphicsTextItem) for item in scene.items())
    assert sorted(item.site_index for item in handles) == [0, 1, 2, 3]
    assert len(handles) == 4
    assert max(item.rect().width() for item in handles) <= 0.47


def test_band_axis_is_fixed_physical_kx():
    QApplication.instance() or QApplication([])
    scene = BandView()
    kx = np.linspace(-np.pi, np.pi, 21)
    scene.set_data(type("D", (), {
        "kx": kx,
        "energies": np.column_stack((np.sin(kx), np.cos(kx))),
        "kx_mark": np.pi / 2,
        "title": "",
    })())
    # 物理绘图区固定为 [-π, π]；sceneRect 额外含固定文字的安全边距。
    rect = scene._plot_rect
    assert rect.left() == pytest.approx(-np.pi)
    assert rect.width() == pytest.approx(2 * np.pi)
    assert scene.sceneRect().left() < rect.left()
    assert scene.sceneRect().right() > rect.right()
    labels = [item.toPlainText() for item in scene.items()
              if isinstance(item, QGraphicsTextItem)]
    assert "k_x" in labels
    assert "kx/π" not in labels
    from hamivisualizer.view.math_text import MathTextItem
    math_axis = [item for item in scene.items()
                 if isinstance(item, MathTextItem) and item.data(1) == "axis"]
    assert any(item.layout.source == "k_x" and item.isVisible()
               for item in math_axis)
    assert scene._mark_item is not None
    assert scene._mark_item.line().x1() == pytest.approx(np.pi / 2)


def test_all_default_finite_shapes_keep_regular_physical_outlines():
    """Every built-in lattice gets a non-empty, metrically correct finite mask."""
    for template_name in TEMPLATE_NAMES:
        for shape in ("triangle", "disk", "hexagon"):
            document = template_document(
                template_name, nx=6, ny=6, boundary_kind="obc",
                connectivity="最近邻", shape=shape,
            )
            cell = document["cell"]
            if isinstance(cell, dict) and "a1" in cell:
                vectors = (tuple(cell["a1"]), tuple(cell["a2"]))
            else:
                vectors = ((float(cell["Lx"]), 0.0),
                           (0.0, float(cell["Ly"])))
            boundary = Boundary(
                BoundaryKind.OBC, NX=6, NY=6, shape=shape,
                shape_vectors=vectors,
            )
            assert boundary.active_cells(), template_name
            outline = boundary.shape_outline()
            expected_vertices = {"triangle": 3, "disk": 96, "hexagon": 6}[shape]
            assert len(outline) == expected_vertices, (template_name, shape)
            side_lengths = []
            for i, (x, y) in enumerate(outline):
                xx, yy = outline[(i + 1) % len(outline)]
                side_lengths.append(math.hypot(xx - x, yy - y))
            assert max(side_lengths) - min(side_lengths) <= 1e-6, (
                template_name, shape, side_lengths[:6]
            )


def test_phi_fraction_and_wavefunction_energy_selection():
    _app, win, ctrl = _window()
    row = next(r for r in range(win.panel.param_table.rowCount())
               if win.panel.param_table.item(r, 0).text() == "phi")
    win.panel.param_table.item(row, 1).setText("1/13")
    assert float(win.panel.param_table.item(row, 1).text()) == pytest.approx(1 / 13)
    assert win.panel.get_params()["phi"] == pytest.approx(np.pi / 13)

    win.panel.boundary_combo.setCurrentIndex(1)
    ctrl.rebuild()
    win.tabs.setCurrentIndex(4)
    assert win.panel.energy_group.isEnabled()
    win.panel.energy_edit.setText("1/4")
    win.panel._energy_changed()
    target = 0.25
    energies = np.asarray(win.wf_view._data.energies)
    assert win.wf_view.selected_energy == pytest.approx(
        energies[np.argmin(np.abs(energies - target))]
    )
    assert win.wf_view.combo.itemText(0).startswith("#1 ")
    info = win.wf_view.info.text()
    assert "目标 E" in info and "ΔE" in info and "边界" in info
    assert f"#{win.wf_view.combo.currentIndex() + 1}" in info
