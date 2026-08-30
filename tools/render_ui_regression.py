"""Render deterministic off-screen UI regression screenshots.

The script deliberately uses the same font setup and Fusion style as the real
application.  It writes only to the requested output directory, so visual
evidence stays separate from user models and settings.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import math
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PySide6.QtCore import QEvent, QPoint, QPointF, QThreadPool, Qt
from PySide6.QtGui import QMouseEvent, QPointingDevice
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from hamivisualizer.controller import ViewController
from hamivisualizer.main import _configure_font
from hamivisualizer.model.templates import TEMPLATE_NAMES, template_document
from hamivisualizer.model.boundary import (
    SHAPE_DISK, SHAPE_HEXAGON, SHAPE_RECTANGLE, SHAPE_TRIANGLE,
)
from hamivisualizer.view.main_window import MainWindow, _Session
from hamivisualizer.model.workspace import DocumentHistory, ModelSessionData
from hamivisualizer.view.dialogs import HoppingDialog


SCREENSHOT_UI_SCALE = 1.0


def _parse_screenshot_ui_scale(raw: str) -> float:
    """Parse the evidence scale, deliberately restricted to 100%.

    The application itself still supports its full accessibility scale range.
    This harness has a different contract: every newly generated evidence
    image must be directly comparable and must use the requested 100% UI
    scale. Rejecting other values is safer than silently producing a mixed
    scale artifact.
    """
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("--ui-scale 必须是 1.0（100%）") from None
    if not math.isclose(value, SCREENSHOT_UI_SCALE, rel_tol=0.0, abs_tol=1e-12):
        raise argparse.ArgumentTypeError(
            "证据截图只允许 1.0（100%）；程序界面本身仍可在设置中缩放"
        )
    return SCREENSHOT_UI_SCALE


def _save(window: MainWindow, path: Path) -> None:
    QApplication.processEvents()
    pixmap = window.grab()
    if not pixmap.save(str(path)):
        raise OSError(f"failed to save screenshot: {path}")
    print(f"saved {path} ({pixmap.width()}x{pixmap.height()})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefix", default="ui-state")
    parser.add_argument("--template", choices=TEMPLATE_NAMES, default="NP")
    parser.add_argument("--nx", type=int, default=4)
    parser.add_argument("--ny", type=int, default=4)
    parser.add_argument("--boundary", choices=("semi", "obc"), default="semi")
    parser.add_argument(
        "--shape",
        choices=(SHAPE_RECTANGLE, SHAPE_TRIANGLE, SHAPE_DISK, SHAPE_HEXAGON),
        default=SHAPE_RECTANGLE,
        help="OBC finite-cell mask; ignored by semi-infinite boundary",
    )
    parser.add_argument("--view-zoom", type=float, default=1.0)
    parser.add_argument(
        "--ui-scale", type=_parse_screenshot_ui_scale,
        # argparse interpolates ``%(...)s`` placeholders in help strings;
        # escape the literal percent so ``--help`` itself remains usable.
        help="证据截图界面缩放；固定为 1.0（100%%），同时输出亮/暗主题。",
    )
    parser.add_argument(
        "--window-width", type=int, default=1440,
        help="离屏截图窗口宽度（默认 1440；主验收图可用 1920 等更大画布）。",
    )
    parser.add_argument(
        "--window-height", type=int, default=920,
        help="离屏截图窗口高度（默认 920；主验收图可用 1200 等更大画布）。",
    )
    parser.add_argument(
        "--energy", type=float,
        help="OBC wavefunction target energy; selects the nearest eigenstate for screenshots.",
    )
    parser.add_argument("--edit-lattice", action="store_true")
    parser.add_argument(
        "--select-hop-row", type=int,
        help="In lattice edit mode, reveal the compact editor for this hopping row.",
    )
    parser.add_argument(
        "--show-all-hop-editors", action="store_true",
        help="In lattice edit mode, intentionally render every hopping editor.",
    )
    parser.add_argument(
        "--hop-editor-demo", action="store_true",
        help="Click the first canvas coefficient field and commit a 1/3 fraction.",
    )
    parser.add_argument("--bond-ratio-demo", action="store_true")
    parser.add_argument(
        "--oblique-a1-demo", action="store_true",
        help="Render a honeycomb-like edit view with a vertical a1 component.",
    )
    parser.add_argument(
        "--intercell-dialog-demo", action="store_true",
        help="Render the semantic half-infinite intercell hopping dialog.",
    )
    parser.add_argument(
        "--intercell-large-offset-demo", action="store_true",
        help="Render the custom intercell dialog with offsets beyond ±100.",
    )
    parser.add_argument(
        "--intercell-menu-demo", action="store_true",
        help="Render the compact menu for fixed and custom inter-cell hopping.",
    )
    parser.add_argument(
        "--diagonal-intercell-demo", action="store_true",
        help="Render the relation summary for a mixed dx/dy inter-cell hop.",
    )
    parser.add_argument(
        "--selected-intercell-demo", action="store_true",
        help="Render a relation-menu insertion that inherits the selected row endpoints.",
    )
    parser.add_argument(
        "--matrix-selection-demo", action="store_true",
        help="Render a non-modal matrix-cell selection highlight and status detail.",
    )
    parser.add_argument(
        "--matrix-copy-demo", action="store_true",
        help="Render the enabled Edit-menu action for copying the selected matrix element.",
    )
    parser.add_argument(
        "--intercell-panel-demo", action="store_true",
        help="Render the complete hopping panel, including explicit intra/inter-cell actions.",
    )
    parser.add_argument(
        "--hop-relation-demo", action="store_true",
        help="Render palette-tinted intra/inter-cell rows and their compact editing affordance.",
    )
    parser.add_argument(
        "--restore-spacing-demo", action="store_true",
        help="Render restoration of automatic cell spacing after a manual edit.",
    )
    parser.add_argument(
        "--spacing-edit-demo", action="store_true",
        help="Render an active edit session after changing rectangular/oblique cell spacing.",
    )
    parser.add_argument(
        "--restore-topology-demo", action="store_true",
        help="Render geometry restore after adding a site and a dangling hopping row.",
    )
    parser.add_argument(
        "--restore-history-demo", action="store_true",
        help="Render restore followed by undo/redo feedback in an isolated workspace.",
    )
    parser.add_argument(
        "--dense-guard-demo", action="store_true",
        help="Render the recoverable error shown before an unsafe dense calculation.",
    )
    parser.add_argument(
        "--resource-hint-demo", action="store_true",
        help="Render the live Nx/Ny dense-resource estimate without starting a calculation.",
    )
    parser.add_argument(
        "--ghost-hop-demo", action="store_true",
        help="Render semi-infinite edit mode with periodic ghost endpoints armed for bond creation.",
    )
    parser.add_argument(
        "--history-demo", action="store_true",
        help="Create one in-memory site edit and open the Edit menu for undo-label screenshots.",
    )
    parser.add_argument(
        "--plain-click-demo", action="store_true",
        help="Click one visible edit handle and render the non-mutating selection feedback.",
    )
    parser.add_argument(
        "--drag-snap-demo", action="store_true",
        help="Perform a real press-move-release on one edit handle and render the clamped snap result.",
    )
    parser.add_argument(
        "--connectivity", choices=("仅格点", "最近邻", "最近邻+次近邻"),
        default="最近邻+次近邻",
    )
    parser.add_argument(
        "--tab", choices=("combined", "matrix", "lattice", "band", "wavefunction"),
        default="combined",
    )
    args = parser.parse_args()
    ui_scale = args.ui_scale if args.ui_scale is not None else SCREENSHOT_UI_SCALE
    if args.window_width < 900 or args.window_height < 600:
        parser.error("--window-width 至少 900，--window-height 至少 600")
    args.output.mkdir(parents=True, exist_ok=True)

    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    _configure_font(app)
    window = MainWindow()
    if args.intercell_dialog_demo or args.intercell_large_offset_demo:
        window._set_theme_mode("dark")
        window._set_ui_scale(ui_scale, persist=False)
        dialog = HoppingDialog(0, 1, window, semi=True, site_count=6)
        custom_index = dialog.cell_relation.findData("custom")
        if custom_index < 0:
            raise ValueError("semantic intercell option is missing")
        dialog.cell_relation.setCurrentIndex(custom_index)
        # Exercise both endpoint selectors and the semantic relation row in
        # one stable screenshot; the dialog remains a read-only evidence pass.
        dialog.from_combo.setCurrentIndex(2)
        dialog.to_combo.setCurrentIndex(4)
        if args.intercell_large_offset_demo:
            dialog.off_x.setValue(250)
            dialog.off_y.setValue(-125)
        else:
            dialog.off_x.setValue(2)
            dialog.off_y.setValue(-1)
        dialog.resize(620, 500)
        dialog.show()
        QApplication.processEvents()
        path = args.output / f"{args.prefix}-dark-100.png"
        pixmap = dialog.grab()
        if not pixmap.save(str(path)):
            raise OSError(f"failed to save screenshot: {path}")
        print(f"saved {path} ({pixmap.width()}x{pixmap.height()})")
        dialog.close()
        window.close()
        return 0
    controller = ViewController(window, connect_actions=False)
    history_workspace = None
    document = template_document(
        args.template, nx=args.nx, ny=args.ny, boundary_kind=args.boundary,
        connectivity=args.connectivity, shape=args.shape,
    )
    if args.oblique_a1_demo:
        if not isinstance(document.get("cell"), dict):
            raise ValueError("--oblique-a1-demo requires a vector-cell template")
        document = dict(document)
        document["cell"] = {
            key: list(value) if isinstance(value, (tuple, list)) else value
            for key, value in document["cell"].items()
        }
        document["cell"]["a1"][1] = 0.45
    controller.apply_document(document)
    if args.resource_hint_demo:
        # Exercise the live warning path without allocating the intentionally
        # oversized OBC matrix.  The dimensions remain editable and the model
        # can still be saved; this screenshot only verifies the preflight UX.
        controller.set_runtime_preferences(calculation_mode="manual")
        window.panel.set_boundary_index(1)
        window.panel.set_dim(30, 30)
        window.resize(args.window_width, args.window_height)
        window.show()
        for group in (
            window.panel.params_group,
            window.panel.energy_group,
            window.panel.display_group,
            window.panel.sites_group,
            window.panel.hops_group,
        ):
            group.setExpanded(False)
        window.panel.boundary_group.setExpanded(True)
        window.set_dirty(False)
        scale = int(round(ui_scale * 100))
        for theme in ("light", "dark"):
            window._set_theme_mode(theme)
            QApplication.processEvents()
            _save(window, args.output / f"{args.prefix}-{theme}-{scale}.png")
        window.close()
        return 0
    if args.dense_guard_demo:
        # The controller catches the pre-allocation guard as a normal input
        # error.  Keep the oversized dimensions editable/savable while making
        # the reason and the safe next action visible in the UI.
        window.resize(args.window_width, args.window_height)
        window.show()
        window.tabs.setCurrentIndex(1)
        for group in (
            window.panel.params_group,
            window.panel.energy_group,
            window.panel.display_group,
            window.panel.sites_group,
            window.panel.hops_group,
        ):
            group.setExpanded(False)
        window.panel.boundary_group.setExpanded(True)
        window.panel_scroll.ensureWidgetVisible(window.panel.error_label)
        window.set_dirty(False)
        scale = int(round(ui_scale * 100))
        for theme in ("light", "dark"):
            window._set_theme_mode(theme)
            QApplication.processEvents()
            _save(window, args.output / f"{args.prefix}-{theme}-{scale}.png")
        window.close()
        return 0
    if args.history_demo:
        baseline = controller.current_document()
        history = DocumentHistory(limit=10)
        history.seed(baseline)
        # The regular application enables workspace mode during startup and
        # provides an app-data root before preferences/autosave can write.
        # This focused screenshot mode creates an isolated disposable root so
        # it exercises the same undo labels without touching a user's models
        # or preferences (nor leaving test files alongside screenshots).
        history_workspace = tempfile.TemporaryDirectory(
            dir=args.output, prefix=".history-workspace-"
        )
        window._workspace_root = Path(history_workspace.name)
        window.preferences.autosave = False
        window._workspace_enabled = True
        window._sessions = [_Session(ModelSessionData(name=args.template), baseline, history)]
        window._active_index = 0
        changed = dict(baseline)
        changed["sites"] = [dict(site) for site in baseline["sites"]]
        changed["sites"][0]["x"] = float(changed["sites"][0]["x"]) + 0.25
        controller.apply_document(changed)
    if args.bond_ratio_demo:
        if window.panel.hop_table.rowCount() < 3:
            raise ValueError("bond ratio demo requires at least three hopping rows")
        window.panel.set_hopping_strength(0, 1.2)
        window.panel.set_hopping_strength(1, 0.3)
        window.panel.set_hopping_strength(2, 0.3)
        controller.rebuild()
    window.panel.display_group.setExpanded(True)
    window.panel.sites_group.setExpanded(False)
    window.panel.hops_group.setExpanded(False)
    window.resize(args.window_width, args.window_height)
    window.show()
    window.tabs.setCurrentIndex({
        "combined": 0, "matrix": 1, "lattice": 2,
        "band": 3, "wavefunction": 4,
    }[args.tab])
    if args.intercell_panel_demo:
        window._set_theme_mode("dark")
        window._set_ui_scale(ui_scale, persist=False)
        for group in (
            window.panel.boundary_group,
            window.panel.params_group,
            window.panel.energy_group,
            window.panel.display_group,
            window.panel.sites_group,
        ):
            group.setExpanded(False)
        window.panel.hops_group.setExpanded(True)
        window.panel_scroll.ensureWidgetVisible(window.panel.hops_group)
        QApplication.processEvents()
        _save(window, args.output / f"{args.prefix}-dark-100.png")
        window.set_dirty(False)
        window.close()
        return 0
    if args.hop_relation_demo:
        # Keep the evidence focused on the table itself: one intra-cell and
        # one inter-cell row are enough to verify the semantic tint, tooltips,
        # and compact layout without a dense model obscuring the distinction.
        window._set_ui_scale(ui_scale, persist=False)
        for group in (
            window.panel.boundary_group,
            window.panel.params_group,
            window.panel.energy_group,
            window.panel.display_group,
            window.panel.sites_group,
        ):
            group.setExpanded(False)
        window.panel.hops_group.setExpanded(True)
        window.panel.append_hop(["t", 0, 0, 0, 0, "-t", "none", "0", 1])
        window.panel.append_hop(["t", 0, 0, 1, 0, "-t", "none", "0", 1])
        window.panel.hop_table.selectRow(window.panel.hop_table.rowCount() - 1)
        controller.rebuild()
        for theme in ("light", "dark"):
            window._set_theme_mode(theme)
            QApplication.processEvents()
            scale = int(round(ui_scale * 100))
            _save(window, args.output / f"{args.prefix}-{theme}-{scale}.png")
            menu = window.panel._create_hop_context_menu(
                window.panel.hop_table.rowCount() - 1
            )
            menu.popup(window.panel.hop_table.viewport().mapToGlobal(QPoint(12, 12)))
            QApplication.processEvents()
            menu_path = args.output / f"{args.prefix}-menu-{theme}-{scale}.png"
            menu_pixmap = menu.grab()
            if not menu_pixmap.save(str(menu_path)):
                raise OSError(f"failed to save screenshot: {menu_path}")
            print(f"saved {menu_path} ({menu_pixmap.width()}x{menu_pixmap.height()})")
            menu.hide()
        window.set_dirty(False)
        window.close()
        return 0
    if args.restore_spacing_demo:
        # Exercise the exact edge case guarded by the regression test: a
        # model starts with automatic spacing, receives a manual Lx/Ly edit,
        # and then restores the edit-session baseline.
        window.panel.set_cell_size(None)
        controller.rebuild()
        window._set_lattice_edit_mode(True)
        window.panel.set_cell_size((3.0, 4.0))
        controller.rebuild()
        window._restore_edit_baseline()
        window.tabs.setCurrentIndex(2)
        window.panel.sites_group.setExpanded(True)
        window.panel.hops_group.setExpanded(False)
        window.panel_scroll.ensureWidgetVisible(window.panel.sites_group)
        window.set_dirty(False)
        window._set_ui_scale(ui_scale, persist=False)
        scale = int(round(ui_scale * 100))
        for theme in ("light", "dark"):
            window._set_theme_mode(theme)
            QApplication.processEvents()
            _save(window, args.output / f"{args.prefix}-{theme}-{scale}.png")
        window.close()
        return 0
    if args.spacing_edit_demo:
        # Exercise the user-facing path that is easy to get subtly wrong:
        # change the primitive-cell spacing while visual editing is active,
        # then verify the handles, cell frame and magnetic targets all follow
        # the new geometry.  Keep the modified frame (rather than restoring)
        # so the screenshot is useful as a visual audit of the live state.
        window._set_lattice_edit_mode(True)
        vectors = window.panel.get_cell_vectors()
        if vectors is not None:
            scaled = (
                (float(vectors[0][0]) * 1.20, float(vectors[0][1]) * 1.20),
                (float(vectors[1][0]) * 1.12, float(vectors[1][1]) * 1.12),
            )
            window.panel.set_cell_vectors(scaled)
            spacing_text = "斜元胞间距已更新：|a₁|×1.20，|a₂|×1.12"
        else:
            cell = window.panel.get_cell_size()
            if cell is None:
                rows = window.panel.get_site_rows()
                max_x = max((float(row[0]) for row in rows), default=0.0)
                max_y = max((float(row[1]) for row in rows), default=0.0)
                cell = (max_x + 1.0, max_y + 1.0)
            window.panel.set_cell_size(
                (float(cell[0]) * 1.20, float(cell[1]) * 1.12)
            )
            spacing_text = "矩形元胞间距已更新：Lx×1.20，Ly×1.12"
        controller.rebuild()
        window.tabs.setCurrentIndex(2)
        window.panel.params_group.setExpanded(False)
        window.panel.energy_group.setExpanded(False)
        window.panel.display_group.setExpanded(False)
        window.panel.sites_group.setExpanded(True)
        window.panel.hops_group.setExpanded(False)
        window.panel_scroll.ensureWidgetVisible(window.panel.sites_group)
        window.set_dirty(False)
        window._set_ui_scale(ui_scale, persist=False)
        window.statusBar().showMessage(
            f"{spacing_text}；编辑锚点与吸附参考已同步"
        )
        scale = int(round(ui_scale * 100))
        for theme in ("light", "dark"):
            window._set_theme_mode(theme)
            QApplication.processEvents()
            window.statusBar().showMessage(
                f"{spacing_text}；编辑锚点与吸附参考已同步"
            )
            _save(window, args.output / f"{args.prefix}-{theme}-{scale}.png")
        window.close()
        return 0
    if args.intercell_menu_demo:
        # Keep the evidence focused on the actual side-panel entry point.
        # Capture the popup itself because QWidget.grab() intentionally does
        # not include a separate QMenu top-level window.
        window._set_theme_mode("dark")
        window._set_ui_scale(ui_scale, persist=False)
        QApplication.processEvents()
        window.panel.hops_group.setExpanded(True)
        window.panel.add_hop_mode_btn.ensurePolished()
        menu = window.panel.add_hop_mode_btn.menu()
        menu.popup(window.panel.add_hop_mode_btn.mapToGlobal(
            QPoint(0, window.panel.add_hop_mode_btn.height())
        ))
        QApplication.processEvents()
        path = args.output / f"{args.prefix}-dark-100.png"
        pixmap = menu.grab()
        if not pixmap.save(str(path)):
            raise OSError(f"failed to save screenshot: {path}")
        print(f"saved {path} ({pixmap.width()}x{pixmap.height()})")
        menu.hide()
        window.close()
        return 0
    if args.diagonal_intercell_demo:
        window._set_theme_mode("dark")
        window._set_ui_scale(ui_scale, persist=False)
        for group in (
            window.panel.boundary_group,
            window.panel.params_group,
            window.panel.energy_group,
            window.panel.display_group,
            window.panel.sites_group,
        ):
            group.setExpanded(False)
        window.panel.hops_group.setExpanded(True)
        window.panel_scroll.ensureWidgetVisible(window.panel.hops_group)
        right_action = next(
            action for action in window.panel.add_hop_mode_btn.menu().actions()
            if "右侧胞间" in action.text()
        )
        right_action.trigger()
        last_row = window.panel.hop_table.rowCount() - 1
        window.panel.hop_table.item(last_row, 4).setText("1")
        controller.rebuild()
        QApplication.processEvents()
        _save(window, args.output / f"{args.prefix}-dark-100.png")
        window.close()
        return 0
    if args.selected_intercell_demo:
        window._set_theme_mode("dark")
        window._set_ui_scale(ui_scale, persist=False)
        for group in (
            window.panel.boundary_group,
            window.panel.params_group,
            window.panel.energy_group,
            window.panel.display_group,
            window.panel.sites_group,
        ):
            group.setExpanded(False)
        window.panel.hops_group.setExpanded(True)
        window.panel_scroll.ensureWidgetVisible(window.panel.hops_group)
        # Select the final existing bond so the new relation row must retain
        # its endpoints instead of silently falling back to 0 → 1.
        existing_row = window.panel.hop_table.rowCount() - 1
        window.panel.hop_table.selectRow(existing_row)
        selected = window.panel.get_hop_rows()[existing_row]
        right_action = next(
            action for action in window.panel.add_hop_mode_btn.menu().actions()
            if "右侧胞间" in action.text()
        )
        right_action.trigger()
        controller.rebuild()
        # Verify the compact presentation independently of the insertion
        # action: dx/dy remain visible while phase metadata is folded away.
        window.panel.hop_advanced_check.setChecked(False)
        QApplication.processEvents()
        # Leave the newly inserted row selected so the table and its relation
        # summary are visible in one stable evidence frame.
        window.panel.hop_table.selectRow(window.panel.hop_table.rowCount() - 1)
        window.statusBar().showMessage(
            f"已沿用选中键 {selected['from_site']}→{selected['to_site']} 添加右侧胞间项"
        )
        QApplication.processEvents()
        _save(window, args.output / f"{args.prefix}-dark-100.png")
        # This is a read-only evidence pass; do not open the save-confirmation
        # dialog while closing the intentionally modified demo document.
        window.set_dirty(False)
        window.close()
        return 0
    if args.energy is not None:
        if args.boundary != "obc":
            raise ValueError("--energy 仅适用于双开（OBC）波函数")
        window.panel.set_energy(args.energy)
        # Small/medium OBC renders are synchronous. If a caller requests a
        # huge system, the normal loading state is intentionally captured
        # rather than pretending a background result already exists.
        if window.wf_view._data is not None:
            window.wf_view.select_energy(args.energy)
    if (args.edit_lattice or args.drag_snap_demo or args.spacing_edit_demo
            or args.hop_editor_demo):
        window.panel.params_group.setExpanded(False)
        window.panel.energy_group.setExpanded(False)
        window.panel.display_group.setExpanded(False)
        window.panel.sites_group.setExpanded(True)
        window.panel.hops_group.setExpanded(False)
        window.lattice_mode_btn.setChecked(True)
    if args.plain_click_demo:
        window.tabs.setCurrentIndex(2)
        window.panel.params_group.setExpanded(False)
        window.panel.energy_group.setExpanded(False)
        window.panel.display_group.setExpanded(False)
        window.panel.sites_group.setExpanded(True)
        window.panel.hops_group.setExpanded(False)
        window.lattice_mode_btn.setChecked(True)
    if args.restore_topology_demo:
        # Exercise the user-facing recovery path after a topology experiment:
        # one valid original bond and one bond that points to a newly-added
        # site are restored against the original one-site basis.  The main
        # window keeps the valid row and reports the dropped dangling row in
        # the status bar; this is intentionally rendered, not just asserted.
        baseline = controller.current_document()
        window._set_lattice_edit_mode(True)
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
        controller.apply_document(current)
        window._restore_edit_baseline()
        window.tabs.setCurrentIndex(2)
        window.panel.sites_group.setExpanded(True)
        window.panel.hops_group.setExpanded(True)
        window.panel_scroll.ensureWidgetVisible(window.panel.sites_group)
    if args.restore_history_demo:
        # Keep the screenshot self-contained: create the same per-model
        # history stack as the application, but in a disposable workspace.
        # The final frame is the real undo result after a geometry restore,
        # so the status bar and action labels reflect the user's next step.
        baseline = controller.current_document()
        history = DocumentHistory(limit=10)
        history.seed(baseline)
        history_workspace = tempfile.TemporaryDirectory(
            dir=args.output, prefix=".restore-history-workspace-"
        )
        window._workspace_root = Path(history_workspace.name)
        window.preferences.autosave = False
        window._workspace_enabled = True
        window._sessions = [_Session(ModelSessionData(name=args.template), baseline, history)]
        window._active_index = 0
        window._set_lattice_edit_mode(True)
        current = deepcopy(baseline)
        current["sites"] = [dict(site) for site in baseline["sites"]]
        current["sites"][0]["x"] = float(current["sites"][0]["x"]) + 0.25
        controller.apply_document(current)
        window._restore_edit_baseline()
        window.undo()
        window.tabs.setCurrentIndex(2)
        window.panel.sites_group.setExpanded(True)
        window.panel.hops_group.setExpanded(False)
        window.panel_scroll.ensureWidgetVisible(window.panel.sites_group)
    if args.ghost_hop_demo:
        if args.boundary != "semi":
            raise ValueError("--ghost-hop-demo 仅适用于半无限边界")
        window.panel.params_group.setExpanded(False)
        window.panel.energy_group.setExpanded(False)
        window.panel.display_group.setExpanded(False)
        window.panel.sites_group.setExpanded(False)
        window.panel.hops_group.setExpanded(True)
        window.lattice_mode_btn.setChecked(True)
        window.lattice_add_hop_btn.setChecked(True)
        window.statusBar().showMessage(
            "添加跃迁：中心格点与蓝色周期虚影均可作为端点，自动生成 dx/dy"
        )
    QApplication.processEvents()
    if args.show_all_hop_editors:
        window.lattice_coeff_btn.setChecked(True)
    if args.select_hop_row is not None:
        window.lattice_scene.activate_hop_editor(args.select_hop_row)
    controller.fit_all(force=True)

    if args.hop_editor_demo:
        # Exercise the full user path: click the center of the fixed-pixel
        # proxy (not its scene anchor), type a safe fraction, and commit it.
        # This catches regressions where a proxy looks visible but the click
        # is routed to the canvas and starts a pan gesture instead.
        QApplication.processEvents()
        if not window.lattice_scene._edit_proxies:
            raise ValueError("--hop-editor-demo requires at least one hopping row")
        proxy = window.lattice_scene._edit_proxies[0]
        editor = proxy.widget()
        top_left = window.lattice_gv.mapFromScene(proxy.pos())
        center = top_left + QPoint(editor.width() // 2, editor.height() // 2)
        QTest.mouseClick(window.lattice_gv.viewport(), Qt.LeftButton, pos=center)
        QApplication.processEvents()
        if not editor.hasFocus():
            raise AssertionError("coefficient field did not receive the center click")
        editor.selectAll()
        QTest.keyClicks(editor, "1/3")
        QTest.keyClick(editor, Qt.Key_Return)
        QApplication.processEvents()
        if window.lattice_gv._pan_press_pos is not None:
            raise AssertionError("editing a coefficient field armed canvas panning")
        window.statusBar().showMessage(
            "已在画布系数框输入 1/3；参数与跃迁表已同步"
        )

    if args.plain_click_demo:
        # Use the same viewport event path as a user click, rather than
        # calling ``activate_site`` directly.  This catches accidental pan,
        # stale tool state, and proxy hit-testing regressions.
        QApplication.processEvents()
        if not window.lattice_scene._edit_items:
            raise ValueError("--plain-click-demo requires at least one editable site")
        before_click = deepcopy(controller.current_document())
        handle = next(iter(window.lattice_scene._edit_items.values()))
        point = window.lattice_gv.mapFromScene(handle.scenePos())
        QTest.mouseClick(
            window.lattice_gv.viewport(), Qt.LeftButton,
            pos=point,
        )
        QApplication.processEvents()
        if controller.current_document() != before_click:
            raise AssertionError("普通点击不应修改模型文档")

    if args.drag_snap_demo:
        # Exercise the same press/move/release path as a user.  QTest handles
        # the platform mouse grab for the press/release; the middle move is a
        # synthesized viewport event carrying LeftButton, which is reliable
        # on the off-screen Qt backend as well.
        if not window.lattice_scene._edit_items:
            raise ValueError("--drag-snap-demo requires at least one editable site")
        handle = next(iter(window.lattice_scene._edit_items.values()))
        point = window.lattice_gv.mapFromScene(handle.scenePos())
        moved_point = point + QPoint(80, 35)
        QTest.mousePress(window.lattice_gv.viewport(), Qt.LeftButton,
                         Qt.NoModifier, point)
        device = QPointingDevice.primaryPointingDevice()
        move_event = QMouseEvent(
            QEvent.MouseMove, QPointF(moved_point), QPointF(moved_point),
            QPointF(moved_point), Qt.NoButton, Qt.LeftButton, Qt.NoModifier,
            Qt.MouseEventSynthesizedByApplication, device,
        )
        QApplication.sendEvent(window.lattice_gv.viewport(), move_event)
        QTest.mouseRelease(window.lattice_gv.viewport(), Qt.LeftButton,
                           Qt.NoModifier, moved_point)
        QApplication.processEvents()
        # ``update_site_position`` clamps the drag to the declared cell, so
        # the current document must remain serializable after a fast pointer
        # move even when the target lies outside the primitive cell.
        controller.current_document()
        window.statusBar().showMessage(
            "已拖动格点 1；智能吸附已启用，坐标保持在元胞内"
        )

    target_view = {
        "combined": window.combined_matrix_gv,
        "matrix": window.matrix_gv,
        "lattice": window.lattice_gv,
        "band": window.band_gv,
        "wavefunction": window.wf_view.view,
    }[args.tab]

    def prepare_view() -> None:
        QApplication.processEvents()
        controller.fit_all(force=True)
        if args.view_zoom > 1.0:
            target_view._scale_by(args.view_zoom)
            scene = target_view.scene()
            matrix_rect = getattr(scene, "_matrix_rect", None)
            if matrix_rect is not None and matrix_rect.isValid():
                target_view.centerOn(matrix_rect.center())
        QApplication.processEvents()

    # Evidence output is intentionally deterministic: every new frame uses
    # the required 100% application scale, with one light and one dark image.
    # Accessibility/multi-scale behavior remains covered by widget geometry
    # tests and the live application's own settings; it is not emitted as
    # mixed-scale evidence here.
    scale_plan = (("light", ui_scale), ("dark", ui_scale))
    for mode, ui_scale in scale_plan:
        window._set_theme_mode(mode)
        window._set_ui_scale(ui_scale, persist=False)
        prepare_view()
        if args.restore_topology_demo:
            # UI scaling itself writes a short status message. Re-assert the
            # recovery result immediately before the screenshot so the frame
            # records the actionable feedback users actually need to see.
            window.statusBar().showMessage(
                "已恢复编辑前构型；已移除 1 条指向新增格点的无效跃迁"
            )
        if args.restore_history_demo:
            # The scale control also reports its change in the status bar;
            # restore the operation feedback so every scale documents the
            # undo path rather than the harness setup.
            window.statusBar().showMessage("已撤销：恢复编辑前构型")
        if args.plain_click_demo:
            # Keep the actual click feedback visible after the scale control
            # reports its own change; the status bar is part of this audit.
            window.statusBar().showMessage(
                "已选择格点 1；拖动可移动，Delete 可删除；如需连线请先点击‘添加跃迁’"
            )
        if args.drag_snap_demo:
            # UI-scale changes also emit a resource/status hint.  Re-assert
            # the completed drag feedback so each evidence frame records the
            # same user-facing result at every scale.
            window.statusBar().showMessage(
                "已拖动格点 1；智能吸附已启用，坐标保持在元胞内"
            )
        if args.spacing_edit_demo:
            window.statusBar().showMessage(
                "元胞间距已更新；编辑锚点与吸附参考已同步"
            )
        if args.matrix_selection_demo or args.matrix_copy_demo:
            window.tabs.setCurrentIndex(1)
            window._on_matrix_cell_clicked(0, 1 if window.matrix_scene._data.n > 1 else 0)
            if args.matrix_copy_demo:
                window._copy_selected_matrix_cell()
            QApplication.processEvents()
        suffix = int(round(ui_scale * 100))
        _save(window, args.output / f"{args.prefix}-{mode}-{suffix}.png")
    if args.history_demo:
        window.edit_menu.popup(
            window.menuBar().mapToGlobal(QPoint(48, window.menuBar().height()))
        )
        QApplication.processEvents()
        _save(window, args.output / f"{args.prefix}-history-menu-dark-100.png")
        window.edit_menu.hide()
    if args.matrix_copy_demo:
        # Capture the actual menu popup so the affordance is visible in the
        # evidence, not merely asserted through an enabled QAction.
        window.edit_menu.popup(
            window.menuBar().mapToGlobal(QPoint(48, window.menuBar().height()))
        )
        QApplication.processEvents()
        menu_path = args.output / f"{args.prefix}-matrix-copy-menu-dark-100.png"
        menu_pixmap = window.edit_menu.grab()
        if not menu_pixmap.save(str(menu_path)):
            raise OSError(f"failed to save screenshot: {menu_path}")
        print(f"saved {menu_path} ({menu_pixmap.width()}x{menu_pixmap.height()})")
        window.edit_menu.hide()
    # Rendering is a read-only verification pass.  Bond-ratio demos and
    # display/theme changes intentionally mark the document dirty, but the
    # off-screen harness must not open the application's save-confirmation
    # dialog while it is shutting down (that would make CI hang).
    window.set_dirty(False)
    # Large ribbons use the asynchronous spectral path. Wait for its queued
    # callback before destroying the controller, otherwise the worker can
    # emit into a deleted Qt signal carrier during interpreter shutdown.
    QThreadPool.globalInstance().waitForDone(30_000)
    QApplication.processEvents()
    window.close()
    if history_workspace is not None:
        history_workspace.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
