"""Render the long-parameter-value readability audit at exactly 100% UI scale.

The output directory is supplied by the caller and remains under the project's
``.codex-artifacts`` tree.  This focused harness intentionally avoids the
historical 150%/180% modes so current documentation evidence has one clear
scale convention.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtWidgets import QApplication

from hamivisualizer.controller import ViewController
from hamivisualizer.main import _configure_font
from hamivisualizer.model.templates import template_document
from hamivisualizer.view.main_window import MainWindow


def _save(window: MainWindow, path: Path) -> None:
    QApplication.processEvents()
    pixmap = window.grab()
    if not pixmap.save(str(path)):
        raise OSError(f"无法保存截图：{path}")
    print(f"saved {path} ({pixmap.width()}x{pixmap.height()})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--value", default="0.333333333333")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    _configure_font(app)
    window = MainWindow()
    controller = ViewController(window, connect_actions=False)
    controller.apply_document(template_document(
        "NP", boundary_kind="semi", connectivity="最近邻+次近邻",
    ))
    window.resize(1920, 1200)
    window.show()
    window._set_ui_scale(1.0, persist=False)
    window.panel.params_group.setExpanded(True)
    for group in (
        window.panel.boundary_group,
        window.panel.energy_group,
        window.panel.display_group,
        window.panel.sites_group,
        window.panel.hops_group,
    ):
        group.setExpanded(False)
    window.panel.params_group.setExpanded(True)
    value_row = next(
        (row for row in range(window.panel.param_table.rowCount())
         if window.panel.param_table.item(row, 0)
         and window.panel.param_table.item(row, 0).text() == "t"),
        None,
    )
    if value_row is None:
        raise RuntimeError("参数表没有 t 参数")
    item = window.panel.param_table.item(value_row, 1)
    if item is None:
        raise RuntimeError("参数表没有可编辑数值单元格")
    item.setText(str(args.value))
    QApplication.processEvents()

    for theme in ("light", "dark"):
        window._set_theme_mode(theme)
        QApplication.processEvents()
        _save(window, args.output / f"parameter-value-{theme}-100.png")

    window.set_dirty(False)
    window.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
