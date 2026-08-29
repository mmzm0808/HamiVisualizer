"""Theme (light/dark/system) regression tests.

Covers the menu-bar white-on-white fix, the light/dark stylesheet switch,
scene theme propagation, preference persistence, and the follow-system
resolution path.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from hamivisualizer.controller import ViewController
from hamivisualizer.model.workspace import Preferences, load_preferences, save_preferences
from hamivisualizer.view.main_window import MainWindow
from hamivisualizer.view.theme import DARK, LIGHT, resolve_theme


def _window(workspace_root):
    """Create a themed window without ever touching the real ~/.hvisual data."""
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    ctrl = ViewController(win)
    ctrl.load_preset("NP")
    win._workspace_root = workspace_root
    win.enable_workspace_mode(ctrl)
    return app, win, ctrl


def test_menu_bar_style_has_explicit_item_color(tmp_path):
    """回归：菜单栏不再白底白字（QMenuBar::item 必须显式设 color）。"""
    app, win, _ctrl = _window(tmp_path)
    qss = app.styleSheet()
    section = qss.split("QMenuBar::item")[1]
    assert "color:#" in section
    # 深色主题同样有显式文字色
    win._set_theme_mode("dark")
    qss_dark = app.styleSheet()
    assert "color:#" in qss_dark.split("QMenuBar::item")[1]


def test_preferences_theme_normalize_and_roundtrip(tmp_path):
    prefs = Preferences(theme="purple")
    assert prefs.normalized().theme == "system"
    save_preferences(prefs, tmp_path)
    loaded = load_preferences(tmp_path)
    assert loaded.theme == "system"
    prefs.theme = "dark"
    save_preferences(prefs, tmp_path)
    assert load_preferences(tmp_path).theme == "dark"


def test_theme_switch_applies_dark_stylesheet_and_scene_flags(tmp_path):
    _app, win, _ctrl = _window(tmp_path)
    win._set_theme_mode("light")
    QApplication.processEvents()
    assert win._dark is False
    assert "background:#ffffff" in _app.styleSheet()

    win._set_theme_mode("dark")
    QApplication.processEvents()
    assert win._dark is True
    assert win._theme_mode == "dark"
    # 深色画布与深色菜单栏
    qss = _app.styleSheet()
    assert DARK.canvas in qss
    assert DARK.menubar_bg in qss
    # 场景下发 dark 标记
    assert win.matrix_scene._dark is True
    assert win.band_scene._dark is True
    assert win.lattice_scene._blend_base == DARK.blend_base
    assert win.comparison.matrix_scene._dark is True
    # 外观菜单选中项同步
    assert win.theme_dark_action.isChecked()
    assert not win.theme_light_action.isChecked()
    assert not win.theme_system_action.isChecked()
    # 偏好已持久化
    assert win.preferences.theme == "dark"


def test_theme_switch_back_to_light_restores_canvas(tmp_path):
    _app, win, _ctrl = _window(tmp_path)
    win._set_theme_mode("dark")
    win._set_theme_mode("light")
    QApplication.processEvents()
    assert win._dark is False
    assert win.lattice_scene._blend_base == LIGHT.blend_base
    assert win.theme_light_action.isChecked()


def test_theme_dark_rerenders_lattice_first_cell_on_canvas_base(tmp_path):
    _app, win, _ctrl = _window(tmp_path)
    # 晶格已有数据；切到深色后 set_data 会按画布底色重新预混首胞填充
    win.lattice_scene.set_data(win.lattice_scene._data)
    light_fill = _first_cell_fill(win.lattice_scene)
    assert light_fill is not None and light_fill.red() > 150  # 亮色预混到白底
    win._set_theme_mode("dark")
    QApplication.processEvents()
    dark_fill = _first_cell_fill(win.lattice_scene)
    assert dark_fill is not None and dark_fill.red() < 150  # 深色预混到画布底色


def _first_cell_fill(scene):
    """返回场景中首个半透明蓝系填充（首胞/虚影框）的 QColor。

    首胞填充是 (0.90,0.90,1.0) 预混结果，r≈g；A 子格圆点 (0.20,0.55,0.80)
    明显 r<g，据此区分。
    """
    for item in scene.items():
        if hasattr(item, "brush") and item.brush().style().name == "SolidPattern":
            col = item.brush().color()
            if abs(col.red() - col.green()) <= 8 and col.blue() >= col.red():
                return col
    return None


def test_resolve_theme_modes():
    app = QApplication.instance() or QApplication([])
    assert resolve_theme("light", app) == "light"
    assert resolve_theme("dark", app) == "dark"
    assert resolve_theme("system", app) in {"light", "dark"}
    assert resolve_theme("bogus", app) in {"light", "dark"}


def test_theme_menu_action_switch_persists_preference(tmp_path):
    _app, win, _ctrl = _window(tmp_path)
    win.theme_system_action.trigger()
    QApplication.processEvents()
    assert win._theme_mode == "system"
    assert load_preferences(tmp_path).theme == "system"
    win.theme_dark_action.trigger()
    QApplication.processEvents()
    assert win._theme_mode == "dark"
    assert load_preferences(tmp_path).theme == "dark"
