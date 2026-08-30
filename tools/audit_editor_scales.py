"""离屏验收：检查所有内置模型的编辑器在各 UI 倍率和主题下可用。

该脚本只写入项目自己的 ``.codex-artifacts/screenshots``，不会触碰用户的
``~/.hvisual``。除了截图，还会把固定像素的系数输入框映射到视口坐标，
检查完整边框、互不重叠和不覆盖可见格点，便于在改 QSS 或布局后快速回归。

除 100% 外的倍率只用于几何回归，不会生成截图；项目验收证据始终固定为
100%，避免把 150%/180% 的历史图误当成当前界面基准。
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QPointF, QRect
from PySide6.QtWidgets import QApplication, QStyleFactory

from hamivisualizer.main import _configure_font
from hamivisualizer.controller import ViewController
from hamivisualizer.model.templates import TEMPLATE_NAMES, template_document
from hamivisualizer.view.main_window import MainWindow


OUT = ROOT / ".codex-artifacts" / "screenshots" / "editor-all-scale-theme-audit-20260830"
SCALES = (0.8, 1.0, 1.5, 1.8)
EVIDENCE_SCREENSHOT_SCALE = 1.0
THEMES = ("light", "dark")


def _slug(name: str) -> str:
    return {"空白自定义": "blank", "一维链": "chain", "蜂窝": "honeycomb",
            "三角": "triangle"}.get(name, name.lower())


def _assert_editor_geometry(window: MainWindow, name: str, scale: float, theme: str) -> None:
    scene = window.lattice_scene
    view = window.lattice_gv
    viewport = view.viewport().rect()
    rects: list[QRect] = []
    for proxy in scene._edit_proxies:
        widget = proxy.widget()
        top_left = view.mapFromScene(proxy.pos())
        rect = QRect(top_left, widget.size())
        # Keep an 8px cushion so the rounded border and focus ring are visible.
        assert rect.left() >= -2, (name, scale, theme, "left", rect)
        assert rect.top() >= -2, (name, scale, theme, "top", rect)
        assert rect.right() <= viewport.width() - 8, (name, scale, theme, "right", rect)
        assert rect.bottom() <= viewport.height() - 8, (name, scale, theme, "bottom", rect)
        if proxy.boundingRect().width() < widget.width() or proxy.boundingRect().height() < widget.height():
            print(
                f"proxy-mismatch name={name} scale={scale:g} theme={theme} "
                f"proxy={proxy.boundingRect().size()} geometry={proxy.geometry().size()} widget={widget.size()} "
                f"min={widget.minimumSize()} max={widget.maximumSize()}",
                flush=True,
            )
        assert proxy.boundingRect().width() >= widget.width()
        assert proxy.boundingRect().height() >= widget.height()
        rects.append(rect)
    for i, first in enumerate(rects):
        for second in rects[i + 1:]:
            assert not first.adjusted(-1, -1, 1, 1).intersects(second), (
                name, scale, theme, "editor-overlap", first, second,
            )
    points = list(scene._data.sites)
    if scene._show_ghosts:
        points.extend(scene._data.ghost)
    if points and rects:
        radius = abs(float(view.transform().m11())) * float(scene._site_radius)
        rightmost = max(
            view.mapFromScene(QPointF(float(x), -float(y))).x() + radius
            for x, y, *_rest in points
        )
        assert all(rect.left() >= rightmost + 1 for rect in rects), (
            name, scale, theme, "node-overlap", rightmost, rects,
        )


def _evidence_path(stem: str, theme: str, scale: float) -> Path | None:
    """Return a screenshot path only for the canonical 100% evidence scale."""
    if not math.isclose(
        float(scale), EVIDENCE_SCREENSHOT_SCALE, rel_tol=0.0, abs_tol=1e-12,
    ):
        return None
    return OUT / f"{stem}-{theme}-100.png"


def main() -> None:
    # Keep the audit's output contract intentionally fixed inside the project,
    # but still provide a conventional, side-effect-free ``--help`` page.
    # Without parsing here, passing ``--help`` would launch the full 88-case
    # Qt audit and surprise users with minutes of work and screenshots.
    parser = argparse.ArgumentParser(
        description=(
            "审计全部默认模型的编辑器几何；80/100/150/180% 只做检查，"
            "仅保存 100% 证据到项目 .codex-artifacts。"
        ),
    )
    parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    app.setStyle(QStyleFactory.create("Fusion"))
    _configure_font(app)
    names = list(TEMPLATE_NAMES)
    # The triangular Kagome disk is a separate faithful primitive-cell path;
    # include it so this audit covers both the ordinary six-site supercell and
    # the three-site oblique nanodisk representation.
    names.append("Kagome三角盘")
    checked = 0
    saved = 0
    for name in names:
        window = MainWindow()
        ctrl = ViewController(window)
        if name == "Kagome三角盘":
            document = template_document("Kagome", nx=6, ny=6, boundary_kind="obc",
                                         connectivity="最近邻", shape="triangle")
            stem = "kagome-triangle"
        else:
            document = template_document(name, nx=4, ny=4, boundary_kind="obc",
                                         connectivity="最近邻")
            stem = _slug(name)
        ctrl.apply_document(document)
        window.resize(1920, 1200)
        window.show()
        app.processEvents()
        window.lattice_mode_btn.setChecked(True)
        app.processEvents()
        window.lattice_scene.set_show_all_hop_editors(True)
        app.processEvents()
        for scale in SCALES:
            for theme in THEMES:
                window._set_ui_scale(scale, persist=False)
                window._set_theme_mode(theme)
                ctrl.fit_all(force=True)
                app.processEvents()
                print(f"checking={name} scale={scale:g} theme={theme}", flush=True)
                window.lattice_scene._refresh_editor_sizes()
                _assert_editor_geometry(window, name, scale, theme)
                checked += 1
                path = _evidence_path(stem, theme, scale)
                if path is not None:
                    assert window.grab().save(str(path)), path
                    saved += 1
        window.close()
        window.deleteLater()
        app.processEvents()
    print(f"checked={checked} saved={saved} evidence_scale=100% output={OUT}")


if __name__ == "__main__":
    main()
