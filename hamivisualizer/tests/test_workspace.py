"""Regression tests for the persistent multi-model workspace."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMessageBox, QSlider
from PySide6.QtTest import QTest
import pytest

from hamivisualizer.controller import ViewController
from hamivisualizer.model.templates import TEMPLATE_NAMES, template_document
from hamivisualizer.model.lattice import Lattice, Site
from hamivisualizer.model.hopping import HoppingTerm
from hamivisualizer.model.workspace import (
    DocumentHistory,
    MAX_WORKSPACE_BYTES,
    Preferences,
    load_preferences,
    load_workspace,
    save_preferences,
)
from hamivisualizer.model.persistence import save_model
from hamivisualizer.view.main_window import MainWindow


def test_preferences_roundtrip(tmp_path):
    prefs = Preferences(
        ui_scale=180, undo_limit=23, debounce_ms=450, snap_step=0.5,
        snap_enabled=False,
    )
    save_preferences(prefs, tmp_path)
    back = load_preferences(tmp_path)
    assert back.ui_scale == 180
    assert back.undo_limit == 23
    assert back.debounce_ms == 450
    assert back.snap_step == 0.5
    assert back.snap_enabled is False


def test_malformed_workspace_and_preferences_fall_back_to_defaults(tmp_path):
    (tmp_path / "preferences.json").write_text("[]", encoding="utf-8")
    (tmp_path / "workspace.json").write_text("[]", encoding="utf-8")
    assert load_preferences(tmp_path) == Preferences()
    assert load_workspace(tmp_path).sessions == []


def test_workspace_rejects_string_booleans_for_split_state(tmp_path):
    (tmp_path / "workspace.json").write_text(
        '{"version": 1, "split_enabled": "false"}', encoding="utf-8"
    )
    assert load_workspace(tmp_path).split_enabled is False


def test_preferences_reject_string_booleans_for_safety(tmp_path):
    (tmp_path / "preferences.json").write_text(
        '{"version": 1, "autosave": "false", "check_updates": "false"}',
        encoding="utf-8",
    )
    prefs = load_preferences(tmp_path)
    assert prefs.autosave is True
    assert prefs.check_updates is True


def test_oversized_workspace_is_rejected_without_allocating_json(tmp_path):
    path = tmp_path / "workspace.json"
    with path.open("wb") as stream:
        stream.write(b"0" * (MAX_WORKSPACE_BYTES + 1))
    assert load_workspace(tmp_path).sessions == []


def test_bounded_document_history():
    history = DocumentHistory(limit=2)
    history.seed({"value": 0})
    history.push({"value": 1})
    history.push({"value": 2})
    history.push({"value": 3})
    assert history.undo() == {"value": 2}
    assert history.undo() == {"value": 1}
    assert history.undo() is None
    assert history.redo() == {"value": 2}


def test_document_history_keeps_labels_with_undo_and_redo_snapshots():
    history = DocumentHistory(limit=2)
    history.seed({"value": 0})
    history.push({"value": 1}, "移动格点")
    history.push({"value": 2}, "调整元胞")
    assert history.undo_label == "调整元胞"
    assert history.undo() == {"value": 1}
    assert history.redo_label == "调整元胞"
    assert history.redo() == {"value": 2}
    assert history.undo_label == "调整元胞"


def test_all_templates_create_valid_documents():
    for name in TEMPLATE_NAMES:
        document = template_document(name, connectivity="最近邻+次近邻")
        assert document["version"] == 1
        assert document["sites"]
        assert document["boundary"]["kind"] == "semi"


def test_honeycomb_template_uses_two_site_oblique_primitive_cell():
    document = template_document("蜂窝", connectivity="最近邻")
    assert document["cell"]["a1"] == pytest.approx([3 ** 0.5, 0.0])
    assert document["cell"]["a2"] == pytest.approx([3 ** 0.5 / 2, 1.5])
    assert len(document["sites"]) == 2
    assert len(document["hops"]) == 3
    lattice = Lattice(
        [Site(i, s["x"], s["y"], s["sublattice"])
         for i, s in enumerate(document["sites"])],
        a1=tuple(document["cell"]["a1"]), a2=tuple(document["cell"]["a2"]),
    )
    degree = [0] * 2
    for row in document["hops"]:
        hop = HoppingTerm(
            row["name"], row["from_site"], row["to_site"],
            tuple(row["cell_offset"]), 1.0,
        )
        dx, dy = hop.displacement(lattice)
        assert (dx * dx + dy * dy) ** 0.5 == pytest.approx(1.0)
        degree[hop.from_site] += 1
        degree[hop.to_site] += 1
    assert degree == [3, 3]


def test_recent_models_are_deduplicated_and_persisted(tmp_path):
    _app, window, controller = _workspace(tmp_path)
    first = tmp_path / "first.hvisual"
    second = tmp_path / "second.hvisual"
    save_model(first, controller.current_document())
    save_model(second, controller.current_document())
    window._remember_recent_model(first)
    window._remember_recent_model(second)
    window._remember_recent_model(first)
    assert [Path(p).name for p in window._recent_model_paths] == ["first.hvisual", "second.hvisual"]
    window._refresh_recent_models_menu()
    assert [a.text() for a in window.recent_models_menu.actions()][:2] == ["first.hvisual", "second.hvisual"]
    assert [Path(p).name for p in load_workspace(tmp_path).recent_models] == ["first.hvisual", "second.hvisual"]


def test_recent_menu_disambiguates_duplicate_basenames(tmp_path):
    """同名最近模型在菜单中显示父目录，避免用户打开错误文件。"""
    _app, window, controller = _workspace(tmp_path)
    first_dir = tmp_path / "experiment-a"
    second_dir = tmp_path / "experiment-b"
    first_dir.mkdir()
    second_dir.mkdir()
    first = first_dir / "model.hvisual"
    second = second_dir / "model.hvisual"
    save_model(first, controller.current_document())
    save_model(second, controller.current_document())
    window._remember_recent_model(first)
    window._remember_recent_model(second)
    window._refresh_recent_models_menu()

    actions = window.recent_models_menu.actions()
    assert [action.text() for action in actions[:2]] == [
        "model.hvisual  ·  experiment-b",
        "model.hvisual  ·  experiment-a",
    ]
    assert actions[0].toolTip() == str(second.resolve())
    assert actions[1].toolTip() == str(first.resolve())


def test_clearing_recent_models_refreshes_open_menu_immediately(tmp_path):
    app, window, controller = _workspace(tmp_path)
    model_path = tmp_path / "recent.hvisual"
    save_model(model_path, controller.current_document())
    window._remember_recent_model(model_path)
    window._refresh_recent_models_menu()
    assert window.recent_models_menu.actions()[0].text() == "recent.hvisual"

    # This is the same slot invoked by the menu's "清空最近模型记录" action.
    # The placeholder must replace the stale entry without requiring another
    # menu-open cycle.
    window._clear_recent_models()
    app.processEvents()
    actions = window.recent_models_menu.actions()
    assert [action.text() for action in actions] == ["暂无最近模型"]
    assert actions[0].isEnabled() is False
    assert window._recent_model_paths == []


def test_open_recent_model_adds_one_tab_and_reuses_it(tmp_path):
    app, window, controller = _workspace(tmp_path)
    model_path = tmp_path / "opened.hvisual"
    save_model(model_path, controller.current_document())
    window._remember_recent_model(model_path)

    window._open_recent_model(model_path)
    app.processEvents()
    assert len(window._sessions) == 2
    assert window._sessions[window._active_index].meta.path == str(model_path.resolve())
    active = window._active_index

    # Opening the same recent entry again selects the existing tab instead of
    # creating a duplicate session.
    window._open_recent_model(model_path)
    app.processEvents()
    assert len(window._sessions) == 2
    assert window._active_index == active


def _workspace(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    controller = ViewController(window, connect_actions=False)
    window.controller = controller
    controller.load_preset("NP")
    window._workspace_root = tmp_path
    window.enable_workspace_mode(controller)
    return app, window, controller


def test_model_tabs_keep_independent_documents(tmp_path):
    app, window, _controller = _workspace(tmp_path)
    window._add_template("SC")
    app.processEvents()
    assert len(window._sessions) == 2
    assert window.model_bar.count() == 2
    assert len(window._sessions[0].document["sites"]) == 4
    assert len(window._sessions[1].document["sites"]) == 2
    window.model_bar.setCurrentIndex(0)
    app.processEvents()
    assert window.panel.site_table.rowCount() == 4


def test_switching_all_preset_tabs_clears_editors_at_large_ui_scale(tmp_path):
    """每个预设切换后只保留当前模型的编辑控件，且高缩放仍可点击。"""
    app, window, _controller = _workspace(tmp_path)
    for name in TEMPLATE_NAMES[1:]:
        window._add_template(name)
    window.resize(1920, 1200)
    window.show()
    window._set_ui_scale(1.8, persist=False)
    app.processEvents()

    for index, name in enumerate(TEMPLATE_NAMES):
        window.model_bar.setCurrentIndex(index)
        app.processEvents()
        window.lattice_mode_btn.setChecked(True)
        window.lattice_coeff_btn.setChecked(True)
        app.processEvents()
        valid_rows = {
            int(hop.get("row", -1))
            for hop in window.lattice_scene._editable_hops()
        }
        # Switching from NP/SC/Haldane to a compact preset used to leave old
        # QSlider cell widgets visible over the current parameter row.  Count
        # only live visible sliders: removed widgets may still be waiting for
        # Qt's deferred deletion, but they must never remain paintable.
        visible_sliders = [
            slider for slider in window.panel.param_table.findChildren(QSlider)
            if slider.isVisible()
        ]
        assert len(visible_sliders) == window.panel.param_table.rowCount(), name
        proxy_rows = {
            int(proxy.data(1)) for proxy in window.lattice_scene._edit_proxies
        }
        assert proxy_rows.issubset(valid_rows), name
        viewport = window.lattice_gv.viewport().rect()
        for proxy in window.lattice_scene._edit_proxies:
            top_left = window.lattice_gv.mapFromScene(proxy.pos())
            right = top_left.x() + proxy.widget().width()
            bottom = top_left.y() + proxy.widget().height()
            assert top_left.x() >= 0 and right <= viewport.width(), name
            assert top_left.y() >= 0 and bottom <= viewport.height(), name
        window.lattice_mode_btn.setChecked(False)
        app.processEvents()


def test_reordering_model_tabs_preserves_active_session_identity(tmp_path):
    """Moving a tab must reorder presentation without switching the model."""
    app, window, _controller = _workspace(tmp_path)
    window._add_template("SC")
    app.processEvents()
    assert window._sessions[window._active_index].meta.name == "SC"

    # This is the same signal path used by a drag on QTabBar.
    window.model_bar.moveTab(1, 0)
    app.processEvents()

    assert [session.meta.name for session in window._sessions] == ["SC", "NP"]
    assert [window.model_bar.tabText(i) for i in range(2)] == ["SC", "NP"]
    assert window._active_index == 0
    assert window._sessions[window._active_index].meta.name == "SC"


def test_closing_background_tab_keeps_current_model(tmp_path):
    """Closing a non-current tab must not jump to its neighbour."""
    app, window, _controller = _workspace(tmp_path)
    window._add_template("SC")
    window._add_template("Kagome")
    app.processEvents()
    window.model_bar.setCurrentIndex(0)  # NP remains the active model.
    app.processEvents()

    window._close_model(2)  # Close the background tab to the right.
    app.processEvents()

    assert [session.meta.name for session in window._sessions] == ["NP", "SC"]
    assert window._active_index == 0
    assert window._sessions[window._active_index].meta.name == "NP"
    assert window.model_bar.currentIndex() == 0


def test_closing_last_model_leaves_one_active_blank_tab(tmp_path):
    """The final close action leaves a usable blank workspace tab."""
    app, window, _controller = _workspace(tmp_path)
    window._close_model(0)
    app.processEvents()

    assert len(window._sessions) == 1
    assert window._active_index == 0
    assert window._sessions[0].meta.name == "空白自定义"
    assert window.model_bar.count() == 1
    assert window.model_bar.currentIndex() == 0


def test_save_status_uses_filename_and_keeps_full_path_in_tooltip(tmp_path):
    """Long workspace paths must not overflow the status bar at high scale."""
    _app, window, _controller = _workspace(tmp_path)
    session = window._sessions[window._active_index]
    assert window._save_session(session, False)

    saved_path = Path(session.meta.path)
    assert window.statusBar().currentMessage() == f"模型已保存：{saved_path.name}"
    assert window.statusBar().toolTip() == str(saved_path)


def test_cancelled_close_keeps_dirty_model_and_tab(tmp_path, monkeypatch):
    """Cancel in the unsaved-changes prompt must leave the tab untouched."""
    app, window, _controller = _workspace(tmp_path)
    window.preferences.autosave = False
    window.set_dirty(True)
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Cancel
    )

    window._close_model(0)
    app.processEvents()

    assert len(window._sessions) == 1
    assert window._active_index == 0
    assert window._sessions[0].meta.dirty is True
    assert window.model_bar.tabText(0).startswith("• ")


def test_kagome_triangle_menu_action_creates_explicit_flat_edge_nanodisk(tmp_path):
    """The File menu exposes the actual Kagome triangle, not a rectangle alias."""
    app, window, _controller = _workspace(tmp_path)
    window._add_kagome_triangle()
    app.processEvents()
    active = window._sessions[window._active_index]
    document = active.document
    assert active.meta.name.startswith("Kagome 三角纳米盘")
    assert document["boundary"]["kind"] == "obc"
    assert document["boundary"]["shape"] == "triangle"
    assert len(document["sites"]) == 3
    assert document["cell"]["a1"] == pytest.approx([2.0, 0.0])
    assert document["cell"]["a2"] == pytest.approx([1.0, 3 ** 0.5])
    assert window.panel.get_shape() == "triangle"


def test_workspace_ui_scaling_covers_full_range(tmp_path):
    _app, window, _controller = _workspace(tmp_path)
    matrix_cell_size = window.matrix_scene._cell_size
    matrix_transform = window.matrix_gv.transform().m11()
    window._set_ui_scale(2.5)
    assert window._ui_scale == 1.8
    large_row = window.panel.site_table.verticalHeader().defaultSectionSize()
    assert window.panel.param_table.columnWidth(0) == 126
    # Column fitting may add a few pixels for the scaled delegate padding;
    # preserve the old baseline while allowing that accessibility headroom.
    assert window.panel.param_table.columnWidth(1) >= 108
    assert window.panel_scroll.minimumWidth() == 630
    assert window.matrix_scene._cell_size == matrix_cell_size
    assert window.matrix_gv.transform().m11() == matrix_transform
    window._set_ui_scale(0.1)
    assert window._ui_scale == 0.8
    assert window.panel.site_table.verticalHeader().defaultSectionSize() < large_row


def test_long_parameter_value_expands_value_column_without_side_overflow(tmp_path):
    """Fraction results remain readable in the parameter table at 180%."""
    app, window, _controller = _workspace(tmp_path)
    window.resize(1440, 920)
    window.show()
    app.processEvents()
    window._set_ui_scale(1.8, persist=False)
    item = window.panel.param_table.item(0, 1)
    item.setText("0.333333333333")
    app.processEvents()

    table = window.panel.param_table
    value_width = table.columnWidth(1)
    text_width = table.fontMetrics().horizontalAdvance(item.text())
    assert value_width >= text_width + max(18, table.fontMetrics().height() // 2)
    assert table.horizontalScrollBar().maximum() == 0


def test_control_rail_has_no_horizontal_overflow_in_dense_edit_state(tmp_path):
    """The hardest supported UI state must scroll vertically, never sideways."""
    app, window, _controller = _workspace(tmp_path)
    window.resize(1440, 920)
    window.show()
    app.processEvents()
    window._set_ui_scale(1.8, persist=False)
    window.panel.sites_group.setExpanded(True)
    window.panel.hops_group.setExpanded(True)
    window.panel.hop_advanced_check.setChecked(True)
    app.processEvents()

    assert window.panel_scroll.horizontalScrollBar().maximum() == 0
    viewport = window.panel_scroll.viewport().rect()
    for widget in (
        window.panel.a1x_spin, window.panel.a1y_spin,
        window.panel.a2x_spin, window.panel.a2y_spin,
        window.panel.apply_vectors_btn, window.panel.refresh_btn,
    ):
        top_left = widget.mapTo(window.panel_scroll.viewport(), widget.rect().topLeft())
        bottom_right = widget.mapTo(window.panel_scroll.viewport(), widget.rect().bottomRight())
        # A vertically scrollable rail may place widgets outside the current
        # viewport, but their horizontal footprint must remain reachable.
        assert top_left.x() >= -1
        assert bottom_right.x() <= viewport.width() + 1


def test_visual_site_move_updates_precise_table(tmp_path):
    _app, window, controller = _workspace(tmp_path)
    window.lattice_scene.siteMoved.emit(0, 0.5, 0.75)
    controller.rebuild()
    assert window.panel.get_site_rows()[0][:2] == (0.5, 0.75)


def test_saved_workspace_restores_all_model_tabs(tmp_path):
    app, first, _controller = _workspace(tmp_path)
    first._add_template("SC")
    app.processEvents()
    for session in first._sessions:
        assert first._save_session(session, False)
    first._save_workspace_state()

    second = MainWindow()
    second_controller = ViewController(second, connect_actions=False)
    second.controller = second_controller
    second_controller.load_preset("NP")
    second._workspace_root = tmp_path
    second.enable_workspace_mode(second_controller)
    assert second.model_bar.count() == 2
    assert {s.meta.name for s in second._sessions} == {"NP", "SC"}


def test_workspace_restores_autosaved_active_tab_and_comparison_state(tmp_path):
    """Autosaved tabs and split comparison selection survive a new window."""
    app, first, _controller = _workspace(tmp_path)
    first._add_template("SC")
    app.processEvents()
    # The autosave timer is deliberately asynchronous in the real UI.  Wait
    # longer than its debounce interval so this test exercises the same path
    # as a user who switches/restarts after the save indicator settles.
    QTest.qWait(900)
    app.processEvents()
    first.action_split.setChecked(True)
    first.comparison.model_combo.setCurrentIndex(0)
    first.comparison.result_combo.setCurrentIndex(2)  # 晶格
    first._save_workspace_state()
    expected_ids = [session.meta.id for session in first._sessions]

    second = MainWindow()
    second_controller = ViewController(second, connect_actions=False)
    second.controller = second_controller
    second_controller.load_preset("NP")
    second._workspace_root = tmp_path
    second.enable_workspace_mode(second_controller)
    second.resize(1400, 900)
    second.show()
    app.processEvents()

    assert [session.meta.name for session in second._sessions] == ["NP", "SC"]
    assert [session.meta.id for session in second._sessions] == expected_ids
    assert second._active_index == 1
    assert second._sessions[second._active_index].meta.name == "SC"
    assert second.action_split.isChecked() is True
    assert second.comparison.selected_model_id == expected_ids[0]
    assert second.comparison.selected_result == "晶格"
    assert second.comparison.lattice_scene._data is not None
    assert second.comparison.lattice_scene.sceneRect().isValid()
