"""Main application window with model workspaces and independent result views."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
import re

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QFont, QFontMetrics, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QAbstractSpinBox, QDialog, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QInputDialog,
    QLabel, QMainWindow, QMessageBox, QScrollArea, QSplitter, QTabBar,
    QTabWidget, QToolButton, QVBoxLayout, QWidget, QSizePolicy,
)

from ..model.persistence import load_model, save_model
from ..model.templates import template_document
from ..model.workspace import (
    DocumentHistory, ModelSessionData, Preferences, WorkspaceData,
    app_data_dir, load_preferences, load_workspace, save_preferences,
    save_workspace,
)
from .band_view import BandView
from .comparison_pane import ComparisonPane
from .control_panel import ControlPanel
from .dialogs import PreferencesDialog, TemplateDialog
from .lattice_view import LatticeView
from .matrix_view import MatrixView
from .theme import DARK, LIGHT, app_palette, app_stylesheet, resolve_theme
from .wavefunction_view import WavefunctionView
from .zoom_view import ZoomGraphicsView


def _wrap(scene) -> ZoomGraphicsView:
    view = ZoomGraphicsView(scene)
    view.setToolTip("滚轮缩放 · 左键拖拽平移 · +/- 缩放 · 0 适应窗口")
    return view


@dataclass
class _Session:
    meta: ModelSessionData
    document: dict
    history: DocumentHistory
    cache: dict = field(default_factory=dict)
    view_state: dict = field(default_factory=dict)


class MainWindow(QMainWindow):
    """HamiVisualizer window with a backward-compatible single editor."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._base_title = "HamiVisualizer — 晶格哈密顿量可视化器"
        self._dirty = False
        self._status_flash_token = 0
        self._ui_scale = 1.0
        self._theme_mode = "system"
        self._dark = False
        # Keep one canonical application font per QApplication.  Every
        # MainWindow applies its own scale to the process-wide font; reading
        # QApplication.font() directly here would therefore compound scales
        # when several model windows/tests are opened in one process (e.g.
        # 100% -> 180% -> 150%), eventually making the control rail wider than
        # its viewport and exposing an unwanted horizontal scrollbar.
        app = QApplication.instance()
        base_font = getattr(app, "_hvisualizer_base_ui_font", None) if app else None
        if base_font is None and app is not None:
            base_font = QFont(app.font())
            setattr(app, "_hvisualizer_base_ui_font", QFont(base_font))
        self._base_ui_font = QFont(base_font or QApplication.font())
        self._workspace_enabled = False
        self._sessions: list[_Session] = []
        self._active_index = -1
        self._switching = False
        self._history_replay = False
        # One-shot semantic label for compound UI actions whose meaning is
        # more precise than a structural diff (for example restoring the
        # edit-session geometry).  It is consumed by the next committed
        # snapshot and cleared even when applying that snapshot fails.
        self._pending_history_label: str | None = None
        # Snapshot taken when lattice editing starts.  The explicit restore
        # action below only brings back geometry (sites + cell vectors), so
        # users can undo a bad snap without losing parameter/hopping edits.
        self._edit_baseline_geometry: dict | None = None
        self._workspace_root = app_data_dir()
        self._recent_model_paths: list[str] = []
        self.preferences = Preferences()
        self.setWindowTitle(self._base_title)

        self.matrix_scene = MatrixView()
        self.lattice_scene = LatticeView()
        self.band_scene = BandView()
        self.wf_view = WavefunctionView()
        self.matrix_gv = _wrap(self.matrix_scene)
        self.lattice_gv = _wrap(self.lattice_scene)
        self.band_gv = _wrap(self.band_scene)
        self.combined_matrix_gv = _wrap(self.matrix_scene)
        self.combined_lattice_gv = _wrap(self.lattice_scene)

        self.tabs = QTabWidget()
        self.combined = QWidget()
        cl = QHBoxLayout(self.combined)
        cl.setContentsMargins(2, 2, 2, 2)
        self.combined_splitter = QSplitter(Qt.Horizontal)
        self.combined_splitter.setChildrenCollapsible(False)
        self.combined_splitter.addWidget(self.combined_matrix_gv)
        self.combined_splitter.addWidget(self.combined_lattice_gv)
        self.combined_splitter.setStretchFactor(0, 1)
        self.combined_splitter.setStretchFactor(1, 1)
        cl.addWidget(self.combined_splitter)
        self.tabs.addTab(self.combined, "矩阵+晶格")
        self.tabs.addTab(self.matrix_gv, "矩阵")

        lattice_page = QWidget()
        lattice_layout = QVBoxLayout(lattice_page)
        lattice_layout.setContentsMargins(0, 0, 0, 0)
        # A grid keeps the edit actions usable at large UI scales.  A single
        # horizontal row makes the whole result pane claim an ever-growing
        # minimum width (at 180% it could force a 2,000 px window); the grid
        # is reflowed to 9/6/4 columns by ``_relayout_lattice_edit_toolbar``.
        edit_bar = QGridLayout()
        edit_bar.setContentsMargins(0, 0, 0, 0)
        edit_bar.setHorizontalSpacing(4)
        edit_bar.setVerticalSpacing(4)
        self.lattice_undo_btn = QToolButton()
        self.lattice_undo_btn.setText("撤销")
        self.lattice_undo_btn.setToolTip("撤销（Ctrl+Z）")
        self.lattice_redo_btn = QToolButton()
        self.lattice_redo_btn.setText("重做")
        self.lattice_redo_btn.setToolTip("重做刚撤销的修改（Ctrl+Shift+Z）")
        self.lattice_restore_btn = QToolButton()
        self.lattice_restore_btn.setText("恢复编辑前构型")
        self.lattice_restore_btn.setToolTip(
            "只恢复进入编辑模式前的格点位置和元胞矢量；"
            "参数与仍然有效的跃迁保留。需要完整回退请使用 Ctrl+Z；"
            "拖动时靠近原位置也会自动吸附回去。"
        )
        self.lattice_snap_btn = QToolButton()
        self.lattice_snap_btn.setText("吸附")
        self.lattice_snap_btn.setCheckable(True)
        self.lattice_snap_btn.setChecked(True)
        self.lattice_snap_btn.setToolTip(
            "拖动格点时吸附到网格和精确几何位置；按住 Alt 可临时自由移动"
        )
        self.lattice_coeff_btn = QToolButton()
        self.lattice_coeff_btn.setText("系数：点击编辑")
        self.lattice_coeff_btn.setCheckable(True)
        self.lattice_coeff_btn.setToolTip(
            "默认仅点击某条跃迁后显示该键的系数输入框；开启后同时显示全部系数。"
            "为避免引线遮挡，密集模型只在悬停或聚焦时显示当前系数的引线；"
            "不会自动打开次近邻物理连线。"
        )
        self.lattice_details_btn = QToolButton()
        self.lattice_details_btn.setText("显示次近邻")
        self.lattice_details_btn.setCheckable(True)
        self.lattice_details_btn.setToolTip(
            "编辑态默认只显示最近邻骨架；勾选后显示次近邻、长程键及其可编辑线。"
        )
        self.lattice_add_hop_btn = QToolButton()
        self.lattice_add_hop_btn.setText("添加跃迁")
        self.lattice_add_hop_btn.setCheckable(True)
        self.lattice_add_hop_btn.setToolTip(
            "显式进入连线工具后，再依次点击两个元胞格点创建跃迁。"
            "普通单击格点只选择，不会创建或隐藏任何连线。"
            "半无限模式下，创建对话框可选择胞内、左/右胞间或上下胞间；"
            "左右虚影格点在该工具开启时也可直接点击，程序会自动推导 dx/dy；"
            "左右胞间会进入 Bloch 的 H1/kx 项。"
        )
        self.lattice_add_site_btn = QToolButton()
        self.lattice_add_site_btn.setText("添加格点")
        self.lattice_add_site_btn.setCheckable(True)
        self.lattice_add_site_btn.setToolTip(
            "显式进入添加工具后，在晶格空白处单击新增一个格点。"
            "普通点击与双击都不会意外新增格点。"
        )
        self.lattice_undo_btn.hide()
        self.lattice_redo_btn.hide()
        self.lattice_restore_btn.hide()
        self.lattice_snap_btn.hide()
        self.lattice_coeff_btn.hide()
        self.lattice_details_btn.hide()
        self.lattice_add_hop_btn.hide()
        self.lattice_add_site_btn.hide()
        self.lattice_mode_btn = QToolButton()
        self.lattice_mode_btn.setText("编辑晶格")
        self.lattice_mode_btn.setCheckable(True)
        self.lattice_mode_btn.setToolTip(
            "拖动格点（默认吸附；按 Alt 自由移动）；Delete 删除；"
            "通过“添加格点”和“添加跃迁”工具明确修改拓扑。"
        )
        self._lattice_edit_buttons = (
            self.lattice_undo_btn,
            self.lattice_redo_btn,
            self.lattice_restore_btn,
            self.lattice_snap_btn,
            self.lattice_add_site_btn,
            self.lattice_add_hop_btn,
            self.lattice_details_btn,
            self.lattice_coeff_btn,
            self.lattice_mode_btn,
        )
        self._lattice_edit_bar = edit_bar
        self._relayout_lattice_edit_toolbar()
        lattice_layout.addLayout(edit_bar)
        lattice_layout.addWidget(self.lattice_gv, 1)
        self.tabs.addTab(lattice_page, "晶格")
        self.tabs.addTab(self.band_gv, "能带")
        self.tabs.addTab(self.wf_view, "波函数")
        self.tabs.setTabToolTip(0, "矩阵和晶格均支持滚轮缩放与左键拖拽平移")
        self.tabs.setTabToolTip(1, "放大到可读尺寸后自动显示矩阵元文字")
        self.tabs.setTabToolTip(2, "浏览/编辑模式可切换；表格保留为精确检查入口")

        self.panel = ControlPanel()
        # The rail is intentionally vertically scrollable.  Ignore the
        # panel's size hint horizontally so expanded tables and high-DPI
        # fonts reflow inside the available width instead of moving the whole
        # content behind a horizontal scrollbar.
        self.panel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.panel_scroll = QScrollArea()
        self.panel_scroll.setObjectName("controlRail")
        self.panel_scroll.setWidgetResizable(True)
        self.panel_scroll.setFrameShape(QFrame.NoFrame)
        self.panel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.panel_scroll.setWidget(self.panel)
        self.panel_scroll.setMinimumWidth(350)

        self.result_banner = QLabel("")
        self.result_banner.setWordWrap(True)
        self.result_banner.setVisible(False)
        self.result_banner.setContentsMargins(10, 7, 10, 7)
        self.primary_right = QWidget()
        right_lay = QVBoxLayout(self.primary_right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(4)
        right_lay.addWidget(self.result_banner)
        right_lay.addWidget(self.tabs, 1)

        self.comparison = ComparisonPane()
        self.comparison.hide()
        self.result_splitter = QSplitter(Qt.Horizontal)
        self.result_splitter.addWidget(self.primary_right)
        self.result_splitter.addWidget(self.comparison)
        self.result_splitter.setStretchFactor(0, 1)
        self.result_splitter.setStretchFactor(1, 1)

        self.editor_splitter = QSplitter(Qt.Horizontal)
        self.editor_splitter.addWidget(self.panel_scroll)
        self.editor_splitter.addWidget(self.result_splitter)
        self.editor_splitter.setStretchFactor(1, 1)
        self.editor_splitter.setSizes([420, 940])

        self.model_bar = QTabBar()
        self.model_bar.setDocumentMode(True)
        self.model_bar.setMovable(True)
        self.model_bar.setTabsClosable(True)
        self.model_bar.setExpanding(False)
        self.add_model_btn = QToolButton()
        self.add_model_btn.setText("+")
        self.add_model_btn.setToolTip("从模板新建模型")
        model_row = QHBoxLayout()
        model_row.setContentsMargins(4, 2, 4, 0)
        model_row.addWidget(self.model_bar, 1)
        model_row.addWidget(self.add_model_btn)
        self.model_row_widget = QWidget()
        self.model_row_widget.setObjectName("modelBarContainer")
        self.model_row_widget.setLayout(model_row)
        self.model_row_widget.hide()

        central = QWidget()
        central_lay = QVBoxLayout(central)
        central_lay.setContentsMargins(0, 0, 0, 0)
        central_lay.setSpacing(0)
        central_lay.addWidget(self.model_row_widget)
        central_lay.addWidget(self.editor_splitter, 1)
        self.setCentralWidget(central)

        self.tabs.currentChanged.connect(self._on_tab_changed)
        # A matrix click is a lightweight inspection action: keep it
        # non-modal, highlight the selected cell in both matrix views, and
        # report the exact logical row/column in the status bar.  The scene
        # owns the hit-testing so the combined and dedicated tabs stay in
        # sync automatically.
        self.matrix_scene.cellClicked.connect(self._on_matrix_cell_clicked)
        self.matrix_scene.selectionChanged.connect(self._on_matrix_selection_changed)
        self.lattice_mode_btn.toggled.connect(self._set_lattice_edit_mode)
        self.lattice_snap_btn.toggled.connect(self._set_snap_enabled)
        self.lattice_add_site_btn.toggled.connect(self._set_site_creation_mode)
        self.lattice_add_hop_btn.toggled.connect(self._set_hop_creation_mode)
        self.lattice_details_btn.toggled.connect(self._set_lattice_edit_details)
        self.lattice_coeff_btn.toggled.connect(self._set_show_all_hop_editors)
        self.lattice_scene.hopCreationModeChanged.connect(self._sync_hop_creation_mode)
        self.lattice_scene.siteCreationModeChanged.connect(self._sync_site_creation_mode)
        self.statusBar().showMessage("就绪")
        self._build_menu()
        self.lattice_undo_btn.clicked.connect(self.action_undo.trigger)
        self.lattice_redo_btn.clicked.connect(self.action_redo.trigger)
        self.lattice_restore_btn.clicked.connect(self._restore_edit_baseline)
        self._apply_style(1.0)
        self._sync_theme_actions()
        self._autosave_timer = QTimer(self)
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(650)
        self._autosave_timer.timeout.connect(self._autosave_current)
        # UI-scale changes need one deferred fit after Qt has laid out the
        # resized rail.  Keep that timer parented to the window so a preview
        # or test window cannot leave a free-floating callback that later
        # dereferences an already-destroyed ZoomGraphicsView.
        self._fit_after_scale_timer = QTimer(self)
        self._fit_after_scale_timer.setSingleShot(True)
        self._fit_after_scale_timer.timeout.connect(self._fit_after_scale)
        # Esc is a window-level cancellation affordance.  Relying only on
        # QGraphicsScene.keyPressEvent made cancellation fail when focus was
        # still in the toolbar or a side-table editor, leaving a topology tool
        # armed when the user believed it had been dismissed.
        self._cancel_lattice_tools_shortcut = QShortcut(
            QKeySequence(Qt.Key_Escape), self,
        )
        self._cancel_lattice_tools_shortcut.setContext(Qt.WindowShortcut)
        self._cancel_lattice_tools_shortcut.activated.connect(
            self._cancel_lattice_tools
        )

    # UI surface consumed by ViewController.
    def _theme(self):
        return DARK if self._dark else LIGHT

    def set_result_state(self, state: str, message: str = "", *, show_banner: bool = True):
        if state == "ready":
            self.result_banner.setVisible(False)
            self.result_banner.setText("")
            self.action_export.setEnabled(True)
            return
        theme = self._theme()
        if state == "busy":
            self.result_banner.setStyleSheet(
                f"background:{theme.banner_info_bg};color:{theme.banner_info_text};"
                f"border:1px solid {theme.banner_info_border};border-radius:6px;"
            )
            self.result_banner.setText(message or "正在计算…")
            self.action_export.setEnabled(False)
        else:
            self.result_banner.setStyleSheet(
                f"background:{theme.banner_err_bg};color:{theme.banner_err_text};"
                f"border:1px solid {theme.banner_err_border};border-radius:6px;"
            )
            self.result_banner.setText(message or "当前视图已过期，请修正输入。")
            self.action_export.setEnabled(False)
        self.result_banner.setVisible(bool(show_banner))

    def flash_status(self, message: str, duration_ms: int = 1800):
        self._status_flash_token += 1
        token = self._status_flash_token
        bar = self.statusBar()
        bar.setStyleSheet(f"QStatusBar {{ color: {self._theme().error_text}; font-weight: 600; }}")
        bar.showMessage(str(message), max(0, int(duration_ms)))

        def restore():
            if token == self._status_flash_token:
                # A delayed flash may outlive a test/preview window.  PySide
                # wrappers then still exist while the C++ status bar is gone.
                try:
                    bar.setStyleSheet("")
                except RuntimeError:
                    return
        QTimer.singleShot(max(0, int(duration_ms)), restore)

    def set_boundary_mode(self, semi: bool):
        self.tabs.setTabEnabled(3, semi)
        self.tabs.setTabEnabled(4, not semi)
        current = self.tabs.currentIndex()
        if (current == 3 and not semi) or (current == 4 and semi):
            self.tabs.setCurrentIndex(0)
        self.panel.set_wavefunction_active((not semi) and self.tabs.currentIndex() == 4)

    def set_dirty(self, dirty: bool = True):
        self._dirty = bool(dirty)
        if self._workspace_enabled and 0 <= self._active_index < len(self._sessions):
            self._sessions[self._active_index].meta.dirty = bool(dirty)
            self._refresh_model_tab(self._active_index)
        self.setWindowTitle(("* " if self._dirty else "") + self._base_title)

    def _on_tab_changed(self, idx: int):
        self.panel.set_wavefunction_active((not self.panel.is_semi()) and idx == 4)
        if self._workspace_enabled and 0 <= self._active_index < len(self._sessions):
            self._sessions[self._active_index].meta.result_tab = idx
        c = getattr(self, "controller", None)
        if c is not None:
            c.fit_all(force=False)

    def _on_matrix_cell_clicked(self, row: int, col: int):
        """Show a precise, non-blocking matrix-element inspection result."""
        try:
            self.matrix_scene.select_cell(row, col)
            index_label, logical_label, value = self.matrix_scene.cell_details(row, col)
        except (IndexError, ValueError):
            return
        value = str(value).replace("\n", " ")
        self.statusBar().showMessage(
            f"已选中 {logical_label}（{index_label}） = {value}", 6000
        )

    def _on_matrix_selection_changed(self, selected):
        """Keep the copy action valid only for the current matrix revision."""
        action = getattr(self, "action_copy_matrix_cell", None)
        if action is not None:
            action.setEnabled(selected is not None)

    def _copy_selected_matrix_cell(self):
        """Copy the selected matrix element in a compact, reusable form."""
        selected = self.matrix_scene.selected_cell
        if selected is None:
            self.flash_status("请先点击一个矩阵元", 1800)
            return
        try:
            index_label, logical_label, value = self.matrix_scene.cell_details(*selected)
        except (IndexError, ValueError):
            # A rebuild can invalidate the selection between the menu event
            # and this slot.  The scene's selection signal normally disables
            # the action, but this guard keeps keyboard/menu races harmless.
            self._on_matrix_selection_changed(None)
            self.flash_status("矩阵已更新，请重新选择矩阵元", 2200)
            return
        display_value = str(value).replace(chr(10), " ")
        try:
            latex_value = self.matrix_scene.cell_latex(*selected)
        except (IndexError, ValueError):
            self._on_matrix_selection_changed(None)
            self.flash_status("矩阵已更新，请重新选择矩阵元", 2200)
            return
        text = (
            f"{logical_label} = {display_value}\n"
            f"LaTeX: {index_label} = {latex_value}"
        )
        clipboard = QApplication.clipboard()
        if clipboard is None:
            self.flash_status("当前环境不支持剪贴板", 2200)
            return
        clipboard.setText(text)
        self.statusBar().showMessage(f"已复制：{index_label} = {display_value}", 2600)

    def _relayout_lattice_edit_toolbar(self):
        """Reflow lattice-edit actions so UI scaling never widens the window.

        The editor is a compact action strip at the default scale.  At 120%
        and above it becomes a small two/three-row palette, keeping every
        button fully clickable inside the result pane instead of letting a
        long label impose a huge minimum window width.  Tooltips retain the
        full descriptions, so the compact layout never removes functionality.
        """
        layout = getattr(self, "_lattice_edit_bar", None)
        buttons = getattr(self, "_lattice_edit_buttons", ())
        if layout is None or not buttons:
            return
        columns = (
            9 if self._ui_scale <= 1.0
            else 6 if self._ui_scale <= 1.3
            else 4 if self._ui_scale <= 1.6
            else 3
        )
        while layout.count():
            layout.takeAt(0)
        for index, button in enumerate(buttons):
            layout.addWidget(button, index // columns, index % columns)
        layout.setAlignment(Qt.AlignRight)
        layout.invalidate()
        layout.activate()
        parent = layout.parentWidget()
        if parent is not None:
            parent.updateGeometry()

    def _set_lattice_edit_mode(self, enabled: bool):
        if enabled and self._edit_baseline_geometry is None:
            self._capture_edit_baseline()
        if not enabled:
            self._edit_baseline_geometry = None
            self.lattice_scene.set_snap_reference_sites(())
        self.lattice_scene.set_edit_mode(enabled)
        self._relayout_lattice_edit_toolbar()
        self.lattice_mode_btn.setText("完成编辑" if enabled else "编辑晶格")
        self.lattice_undo_btn.setVisible(bool(enabled))
        self.lattice_redo_btn.setVisible(bool(enabled))
        self.lattice_restore_btn.setVisible(bool(enabled and self._edit_baseline_geometry))
        self.lattice_snap_btn.setVisible(bool(enabled))
        self.lattice_add_site_btn.setVisible(bool(enabled))
        self.lattice_add_hop_btn.setVisible(bool(enabled))
        self.lattice_details_btn.setVisible(bool(enabled))
        self.lattice_coeff_btn.setVisible(bool(enabled))
        if enabled:
            self.lattice_add_site_btn.blockSignals(True)
            self.lattice_add_site_btn.setChecked(False)
            self.lattice_add_site_btn.blockSignals(False)
            self.lattice_add_hop_btn.blockSignals(True)
            self.lattice_add_hop_btn.setChecked(False)
            self.lattice_add_hop_btn.blockSignals(False)
            self.lattice_details_btn.blockSignals(True)
            self.lattice_details_btn.setChecked(False)
            self.lattice_details_btn.blockSignals(False)
            # A new editing session starts with the quiet, progressive
            # disclosure layer.  Keeping the previous session's
            # "show-all coefficients" state made dense Kagome/NP models
            # reopen covered by dozens of fields, even though the toolbar
            # had just presented this as a fresh edit operation.
            self.lattice_coeff_btn.blockSignals(True)
            self.lattice_coeff_btn.setChecked(False)
            self.lattice_coeff_btn.setText("系数：点击编辑")
            self.lattice_coeff_btn.blockSignals(False)
            self.lattice_scene.set_show_all_hop_editors(False)
            # Geometry and bond tables are the reliable fallback for precise
            # edits. Open them automatically when entering visual edit mode so
            # cell spacing and relative coefficients are discoverable without
            # making users hunt through a collapsed sidebar.
            self.panel.sites_group.setExpanded(True)
            self.panel.hops_group.setExpanded(True)
            self.tabs.setCurrentIndex(2)
            self.statusBar().showMessage(
                f"编辑模式：{'智能吸附' if self.lattice_snap_btn.isChecked() else '自由移动'}；"
                "单击格点仅选择；拖动可移动；用显式工具添加格点或跃迁；点击跃迁线改系数"
            )
        else:
            self.lattice_add_site_btn.blockSignals(True)
            self.lattice_add_site_btn.setChecked(False)
            self.lattice_add_site_btn.blockSignals(False)
            self.lattice_add_hop_btn.blockSignals(True)
            self.lattice_add_hop_btn.setChecked(False)
            self.lattice_add_hop_btn.blockSignals(False)
        if self._workspace_enabled and 0 <= self._active_index < len(self._sessions):
            self._sessions[self._active_index].meta.edit_mode = bool(enabled)

    def _set_snap_enabled(self, enabled: bool):
        self.lattice_scene.set_snap_enabled(enabled)
        if self.lattice_mode_btn.isChecked():
            mode = "智能吸附" if enabled else "自由移动"
            self.statusBar().showMessage(
                f"编辑模式：{mode}；Alt 临时自由移动；Esc 取消拖动；点击元胞跃迁线改系数"
            )

    def _set_hop_creation_mode(self, enabled: bool):
        """Enable the explicit two-click bond tool; normal selection is inert."""
        if enabled and not self.lattice_mode_btn.isChecked():
            self.lattice_add_hop_btn.blockSignals(True)
            self.lattice_add_hop_btn.setChecked(False)
            self.lattice_add_hop_btn.blockSignals(False)
            return
        self.lattice_scene.set_hop_creation_mode(enabled)

    def _set_site_creation_mode(self, enabled: bool):
        """Enable the explicit blank-canvas site tool; selection stays inert."""
        if enabled and not self.lattice_mode_btn.isChecked():
            self.lattice_add_site_btn.blockSignals(True)
            self.lattice_add_site_btn.setChecked(False)
            self.lattice_add_site_btn.blockSignals(False)
            return
        self.lattice_scene.set_site_creation_mode(enabled)

    def _cancel_lattice_tools(self):
        """Cancel topology tools regardless of which editor currently owns focus."""
        cancelled = False
        if self.lattice_scene.hop_creation_mode:
            self.lattice_scene.set_hop_creation_mode(False)
            cancelled = True
        if self.lattice_scene.site_creation_mode:
            self.lattice_scene.set_site_creation_mode(False)
            cancelled = True
        if cancelled:
            self.statusBar().showMessage(
                "已取消当前晶格工具；单击格点仅选择，拖动可移动", 1800
            )

    def _sync_site_creation_mode(self, enabled: bool):
        """Reflect completion/Esc and keep topology tools mutually exclusive."""
        self.lattice_add_site_btn.blockSignals(True)
        self.lattice_add_site_btn.setChecked(bool(enabled))
        self.lattice_add_site_btn.blockSignals(False)
        self.lattice_add_site_btn.setText("选择空白位置…" if enabled else "添加格点")

    def _sync_hop_creation_mode(self, enabled: bool):
        """Keep the toolbar honest when Esc or a completed bond exits the tool."""
        self.lattice_add_hop_btn.blockSignals(True)
        self.lattice_add_hop_btn.setChecked(bool(enabled))
        self.lattice_add_hop_btn.blockSignals(False)
        self.lattice_add_hop_btn.setText("选择两个格点…" if enabled else "添加跃迁")

    def _set_show_all_hop_editors(self, enabled: bool):
        """Expose dense coefficient fields only on explicit user request."""
        self.lattice_scene.set_show_all_hop_editors(enabled)
        self.lattice_coeff_btn.setText("系数：全部显示" if enabled else "系数：点击编辑")
        if self.lattice_mode_btn.isChecked():
            text = (
                "已显示全部跃迁系数；次近邻层可单独切换，悬停或聚焦查看对应引线"
                if enabled else "已切换为点击跃迁线后编辑"
            )
            self.statusBar().showMessage(text, 1800)

    def _set_lattice_edit_details(self, enabled: bool):
        """Toggle long-range visual detail without changing model display prefs."""
        self.lattice_scene.set_show_edit_details(enabled)
        if self.lattice_mode_btn.isChecked():
            self.statusBar().showMessage(
                "已显示次近邻与长程键" if enabled else "已隐藏次近邻与长程键，保留最近邻骨架",
                1800,
            )

    def _capture_edit_baseline(self, document: dict | None = None):
        """Capture one model's geometry for the current edit session."""
        if document is None:
            controller = getattr(self, "controller", None)
            if controller is None:
                return
            try:
                document = controller.current_document()
            except (TypeError, ValueError, AttributeError):
                return
        if isinstance(document, dict):
            self._edit_baseline_geometry = {
                "sites": deepcopy(document.get("sites", [])),
                "cell": deepcopy(document.get("cell")),
            }
            self.lattice_scene.set_snap_reference_sites(
                self._edit_baseline_geometry["sites"]
            )

    def _restore_edit_baseline(self):
        """Restore only geometry captured when the current edit session began."""
        baseline = self._edit_baseline_geometry
        controller = getattr(self, "controller", None)
        if not baseline or controller is None:
            return
        try:
            current = controller.current_document()
        except (TypeError, ValueError, AttributeError) as exc:
            self.flash_status(f"当前模型无效，无法恢复位置：{exc}")
            return
        if not isinstance(current, dict):
            return
        restored = deepcopy(current)
        restored["sites"] = deepcopy(baseline.get("sites", []))
        # ``cell=None`` is meaningful: it means the model was using automatic
        # spacing.  Omitting the key in that case used to leave a user-entered
        # Lx/Ly override behind, so restoring geometry was only half-effective.
        # Restore the explicit None as well as vector/rectangular definitions.
        restored["cell"] = deepcopy(baseline.get("cell"))
        # A geometry-only restore intentionally keeps parameter and hopping
        # edits.  Topology edits can, however, contain a newly-added site
        # index that no longer exists after the baseline sites are restored.
        # Passing that dangling row to the controller used to make the
        # restore fail wholesale, which is especially confusing after a
        # snap/drag experiment.  Keep every still-valid hopping definition
        # (including user-edited offsets) and discard only rows whose local
        # endpoints cannot exist in the restored basis.
        hops = restored.get("hops")
        if isinstance(hops, list):
            site_count = len(restored["sites"])
            valid_hops = []
            dropped = 0
            for hop in hops:
                if not isinstance(hop, dict):
                    dropped += 1
                    continue
                try:
                    from_site = int(hop.get("from_site", -1))
                    to_site = int(hop.get("to_site", -1))
                except (TypeError, ValueError):
                    dropped += 1
                    continue
                if 0 <= from_site < site_count and 0 <= to_site < site_count:
                    valid_hops.append(hop)
                else:
                    dropped += 1
            restored["hops"] = valid_hops
        else:
            dropped = 0
        try:
            previous_label = self._pending_history_label
            self._pending_history_label = "恢复编辑前构型"
            try:
                controller.apply_document(restored)
            finally:
                # ``document_committed`` consumes the pending label on a
                # successful rebuild.  If validation fails before commit,
                # restore the prior pending state rather than leaking this
                # one-shot label into a later unrelated edit.
                if self._pending_history_label == "恢复编辑前构型":
                    self._pending_history_label = previous_label
        except (TypeError, ValueError) as exc:
            self.flash_status(f"恢复位置失败：{exc}")
            return
        self.set_dirty(True)
        self.lattice_scene.set_snap_reference_sites(baseline.get("sites", []))
        if dropped:
            self.statusBar().showMessage(
                f"已恢复编辑前构型；已移除 {dropped} 条指向新增格点的无效跃迁"
            )
        else:
            self.statusBar().showMessage("已恢复编辑前的格点位置和元胞矢量")

    def showEvent(self, event):
        super().showEvent(event)
        c = getattr(self, "controller", None)
        if c is not None:
            c.fit_all(force=False)

    def _build_menu(self):
        mb = self.menuBar()
        fm = mb.addMenu("文件")
        self.action_new = fm.addAction("新建模型…")
        self.action_new.setShortcut(QKeySequence.New)
        self.action_np = fm.addAction("新建 NP 模型")
        self.action_sc = fm.addAction("新建 SC 模型")
        # Kagome 的三角纳米盘不是普通矩形 Kagome 的别名：它使用 3 格点
        # 斜原胞和 OBC 三角掩膜。将它作为明确入口，避免用户看到矩形
        # Kagome 后误以为那就是三角纳米盘。
        self.action_kagome_triangle = fm.addAction("新建 Kagome 三角纳米盘")
        self.action_kagome_triangle.setToolTip(
            "6×6 双开边界、3 格点斜原胞、最近邻连接；三条边为平直等边边界。"
        )
        fm.addSeparator()
        self.action_open = fm.addAction("打开模型…")
        self.action_open.setShortcut(QKeySequence.Open)
        self.recent_models_menu = fm.addMenu("打开最近模型")
        self.recent_models_menu.setToolTipsVisible(True)
        self.recent_models_menu.aboutToShow.connect(self._refresh_recent_models_menu)
        self.action_save = fm.addAction("保存模型")
        self.action_save.setShortcut(QKeySequence.Save)
        self.action_save_as = fm.addAction("模型另存为…")
        self.action_save_as.setShortcut(QKeySequence.SaveAs)
        self.action_export = fm.addAction("导出当前视图 PNG…")
        self.action_export.setShortcut("Ctrl+E")
        fm.addSeparator()
        quit_action = fm.addAction("退出", self.close)
        quit_action.setShortcut(QKeySequence.Quit)

        em = mb.addMenu("编辑")
        self.edit_menu = em
        self.action_undo = em.addAction("撤销")
        self.action_undo.setShortcut(QKeySequence.Undo)
        self.action_redo = em.addAction("重做")
        self.action_redo.setShortcuts([QKeySequence.Redo, QKeySequence("Ctrl+Shift+Z")])
        em.addSeparator()
        self.action_duplicate = em.addAction("复制当前模型")
        self.action_copy_matrix_cell = em.addAction("复制选中矩阵元")
        self.action_copy_matrix_cell.setShortcut(QKeySequence("Ctrl+Shift+C"))
        self.action_copy_matrix_cell.setToolTip(
            "复制当前选中的矩阵元及其逻辑行列标签；先在矩阵中点击一个单元格。"
        )
        self.action_copy_matrix_cell.setEnabled(False)
        # This action is useful even in the historical single-document mode,
        # so bind it when the menu is built rather than only when the
        # multi-model workspace layer is enabled.
        self.action_copy_matrix_cell.triggered.connect(self._copy_selected_matrix_cell)
        self.action_rename = em.addAction("重命名当前模型…")
        self.action_preferences = em.addAction("偏好设置…")
        self.action_undo.setEnabled(False)
        self.action_redo.setEnabled(False)

        vm = mb.addMenu("视图")
        fit_action = vm.addAction("适应窗口", self._fit_views)
        fit_action.setShortcut("F")
        self.action_split = vm.addAction("分屏比较")
        self.action_split.setCheckable(True)
        vm.addSeparator()
        self.action_ui_zoom_in = vm.addAction("放大界面")
        self.action_ui_zoom_in.setShortcuts([QKeySequence("Ctrl++"), QKeySequence("Ctrl+=")])
        self.action_ui_zoom_in.triggered.connect(lambda: self._change_ui_scale(0.10))
        self.action_ui_zoom_out = vm.addAction("缩小界面")
        self.action_ui_zoom_out.setShortcuts([QKeySequence("Ctrl+-"), QKeySequence("Ctrl+_")])
        self.action_ui_zoom_out.triggered.connect(lambda: self._change_ui_scale(-0.10))
        self.action_ui_zoom_reset = vm.addAction("恢复界面大小")
        self.action_ui_zoom_reset.setShortcut("Ctrl+0")
        self.action_ui_zoom_reset.triggered.connect(lambda: self._set_ui_scale(1.0))
        vm.addSeparator()
        self.appearance_menu = vm.addMenu("外观")
        self.theme_light_action = self.appearance_menu.addAction("浅色")
        self.theme_light_action.setCheckable(True)
        self.theme_dark_action = self.appearance_menu.addAction("深色")
        self.theme_dark_action.setCheckable(True)
        self.theme_system_action = self.appearance_menu.addAction("跟随系统")
        self.theme_system_action.setCheckable(True)
        self.theme_light_action.triggered.connect(lambda: self._set_theme_mode("light"))
        self.theme_dark_action.triggered.connect(lambda: self._set_theme_mode("dark"))
        self.theme_system_action.triggered.connect(lambda: self._set_theme_mode("system"))

        hm = mb.addMenu("帮助")
        self.action_update = hm.addAction("检查更新…")
        self.action_update.setEnabled(False)
        self.action_update.setToolTip("将在发布 GitHub Releases 后启用")
        hm.addAction("关于", lambda: QMessageBox.about(
            self, "关于 HamiVisualizer",
            "<b>HamiVisualizer 0.4</b><br/>晶格哈密顿量的多模型交互式可视化工作区。"
            "<br/><br/>支持模型标签、分屏比较、撤销/重做、自动保存和可视化晶格编辑。",
        ))

    # Persistent multi-model layer.
    def enable_workspace_mode(self, controller):
        if self._workspace_enabled:
            return
        self._workspace_enabled = True
        self.controller = controller
        self.preferences = load_preferences(self._workspace_root)
        self._theme_mode = self.preferences.theme
        self._dark = self._resolve_dark(self._theme_mode)
        self._sync_theme_actions()
        self._set_ui_scale(self.preferences.ui_scale / 100.0, persist=False)
        self.lattice_snap_btn.blockSignals(True)
        self.lattice_snap_btn.setChecked(self.preferences.snap_enabled)
        self.lattice_snap_btn.blockSignals(False)
        controller.set_runtime_preferences(
            debounce_ms=self.preferences.debounce_ms,
            calculation_mode=self.preferences.calculation_mode,
            snap_step=self.preferences.snap_step,
            snap_enabled=self.preferences.snap_enabled,
        )
        self.model_row_widget.show()
        state = load_workspace(self._workspace_root)
        self._recent_model_paths = self._normalized_recent_paths(state.recent_models)
        restored: list[_Session] = []
        for meta in state.sessions:
            if not meta.path:
                continue
            try:
                doc = load_model(meta.path)
            except (OSError, ValueError, TypeError):
                continue
            history = DocumentHistory(self.preferences.undo_limit)
            history.seed(doc)
            meta.dirty = False
            restored.append(_Session(meta, doc, history))
        if not restored:
            doc = controller.current_document()
            meta = ModelSessionData(name="NP")
            history = DocumentHistory(self.preferences.undo_limit)
            history.seed(doc)
            restored = [_Session(meta, doc, history)]
        self._sessions = restored
        self.model_bar.blockSignals(True)
        for session in self._sessions:
            self.model_bar.addTab(session.meta.name)
        target = min(max(0, state.current_index), len(self._sessions) - 1)
        self.model_bar.setCurrentIndex(target)
        self.model_bar.blockSignals(False)
        self._active_index = target
        self._switching = True
        controller.apply_document(self._sessions[target].document)
        self._switching = False
        self.tabs.setCurrentIndex(self._sessions[target].meta.result_tab)
        self.lattice_mode_btn.setChecked(self._sessions[target].meta.edit_mode)

        self.model_bar.currentChanged.connect(self._switch_model)
        self.model_bar.tabCloseRequested.connect(self._close_model)
        self.model_bar.tabBarDoubleClicked.connect(self._rename_model)
        self.model_bar.tabMoved.connect(self._move_model)
        self.add_model_btn.clicked.connect(self._new_from_dialog)
        self.action_new.triggered.connect(self._new_from_dialog)
        self.action_np.triggered.connect(lambda: self._add_template("NP"))
        self.action_sc.triggered.connect(lambda: self._add_template("SC"))
        self.action_kagome_triangle.triggered.connect(self._add_kagome_triangle)
        self.action_open.triggered.connect(self._open_as_tab)
        self.action_save.triggered.connect(lambda: self._save_active(False))
        self.action_save_as.triggered.connect(lambda: self._save_active(True))
        self.action_export.triggered.connect(controller.export_png)
        self.action_undo.triggered.connect(self.undo)
        self.action_redo.triggered.connect(self.redo)
        self.action_duplicate.triggered.connect(self.duplicate_current)
        self.action_rename.triggered.connect(lambda: self._rename_model(self._active_index))
        self.action_preferences.triggered.connect(self._show_preferences)
        app = QApplication.instance()
        if app is not None and hasattr(app.styleHints(), "colorSchemeChanged"):
            app.styleHints().colorSchemeChanged.connect(self._on_system_color_scheme_changed)
        self.action_split.toggled.connect(self._set_split_enabled)
        self.comparison.selectionChanged.connect(self._refresh_comparison)
        self.action_split.setChecked(state.split_enabled)
        self._refresh_comparison_models(state.split_model_id)
        self.comparison.set_selected_result(state.split_result)
        if len(state.splitter_sizes) == 2:
            self.result_splitter.setSizes(state.splitter_sizes)
        self.document_committed(controller.current_document())
        self._autosave_current()

    def document_committed(self, document: dict):
        if not self._workspace_enabled or not (0 <= self._active_index < len(self._sessions)):
            return
        session = self._sessions[self._active_index]
        previous_document = session.document
        session.document = deepcopy(document)
        session.cache = {
            "matrix": self.matrix_scene._data, "lattice": self.lattice_scene._data,
            "band": self.band_scene._data, "wavefunction": self.wf_view._data,
        }
        if not self._switching and not self._history_replay:
            label = self._pending_history_label
            self._pending_history_label = None
            session.history.push(
                document, label or self._describe_history_change(previous_document, document)
            )
        self._update_history_actions()
        self._refresh_comparison()
        if self.preferences.autosave and not self._switching:
            self._autosave_timer.start()

    def _switch_model(self, index: int):
        if self._switching or not 0 <= index < len(self._sessions):
            return
        if 0 <= self._active_index < len(self._sessions):
            previous = self._sessions[self._active_index]
            self._capture_view_state(previous)
            try:
                previous.document = self.controller.current_document()
            except ValueError:
                pass
            if self.preferences.autosave:
                self._save_session(previous, False)
        self._active_index = index
        target = self._sessions[index]
        # A restore snapshot belongs to one model and one edit session.  Clear
        # it before applying the target even when both tabs share edit mode and
        # Qt therefore emits no second toggled signal.
        self._edit_baseline_geometry = None
        self._switching = True
        self.controller.apply_document(target.document)
        self.tabs.setCurrentIndex(target.meta.result_tab)
        self.lattice_mode_btn.setChecked(target.meta.edit_mode)
        if target.meta.edit_mode:
            self._capture_edit_baseline(target.document)
        self._switching = False
        self._dirty = target.meta.dirty
        self.setWindowTitle(("* " if self._dirty else "") + self._base_title)
        QTimer.singleShot(0, lambda s=target: self._restore_view_state(s))
        self._update_history_actions()
        self._refresh_comparison_models()
        self._save_workspace_state()
        if self.preferences.autosave:
            self._autosave_timer.start()

    def _capture_view_state(self, session: _Session):
        states = []
        for view in self.all_views():
            t = view.transform()
            states.append({
                "transform": (t.m11(), t.m12(), t.m21(), t.m22(), t.dx(), t.dy()),
                "h": view.horizontalScrollBar().value(),
                "v": view.verticalScrollBar().value(),
                "zoomed": view.user_zoomed,
            })
        session.view_state = {"views": states}

    def _restore_view_state(self, session: _Session):
        from PySide6.QtGui import QTransform
        for view, state in zip(self.all_views(), session.view_state.get("views", [])):
            a, b, c, d, dx, dy = state["transform"]
            view.setTransform(QTransform(a, b, c, d, dx, dy))
            view.horizontalScrollBar().setValue(state["h"])
            view.verticalScrollBar().setValue(state["v"])
            view._user_zoomed = bool(state["zoomed"])
            view._notify_scene_zoom()

    # Model tab actions.
    def _new_from_dialog(self):
        dialog = TemplateDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        values = dialog.values()
        doc = template_document(**{k: v for k, v in values.items() if k != "replace"})
        if values["replace"]:
            answer = QMessageBox.question(
                self, "替换当前模型？", "当前模型的格点、跃迁和参数将被模板替换。",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if answer == QMessageBox.Yes:
                self._replace_current(values["name"], doc)
        else:
            self._add_session(values["name"], doc)

    def _add_template(self, name: str):
        connectivity = "最近邻+次近邻" if name in {"NP", "SC"} else "最近邻"
        self._add_session(name, template_document(name, connectivity=connectivity))

    def _add_kagome_triangle(self):
        """Create the physically explicit Kagome triangular nanodisk preset.

        Keep the ordinary ``Kagome`` template available for rectangular and
        semi-infinite ribbons.  This action is intentionally separate so the
        shape, boundary and primitive-cell convention are never implicit.
        """
        document = template_document(
            "Kagome", nx=6, ny=6, boundary_kind="obc",
            connectivity="最近邻", shape="triangle",
        )
        self._add_session("Kagome 三角纳米盘", document)

    def _add_session(self, name: str, document: dict, path: str = ""):
        meta = ModelSessionData(name=self._unique_name(name), path=path)
        history = DocumentHistory(self.preferences.undo_limit)
        history.seed(document)
        self._sessions.append(_Session(meta, deepcopy(document), history))
        idx = self.model_bar.addTab(meta.name)
        self._refresh_comparison_models()
        self.model_bar.setCurrentIndex(idx)

    def _replace_current(self, name: str, document: dict):
        session = self._sessions[self._active_index]
        session.meta.name = self._unique_name(name, exclude=self._active_index)
        session.meta.path = ""
        session.meta.dirty = True
        session.history.seed(document)
        self._refresh_model_tab(self._active_index)
        self._edit_baseline_geometry = None
        self._switching = True
        self.controller.apply_document(document)
        self._switching = False
        if self.lattice_mode_btn.isChecked():
            self._capture_edit_baseline(document)
        self.set_dirty(True)

    def duplicate_current(self):
        if self._workspace_enabled:
            source = self._sessions[self._active_index]
            self._add_session(f"{source.meta.name} 副本", deepcopy(source.document))

    def _close_model(self, index: int):
        if not 0 <= index < len(self._sessions):
            return
        session = self._sessions[index]
        if session.meta.dirty and not self.preferences.autosave:
            answer = QMessageBox.question(
                self, "关闭模型？", f"“{session.meta.name}”有未保存的更改。",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save,
            )
            if answer == QMessageBox.Cancel:
                return
            if answer == QMessageBox.Save and not self._save_session(session, False):
                return
        active_session = (
            self._sessions[self._active_index]
            if 0 <= self._active_index < len(self._sessions)
            else None
        )
        closing_active = session is active_session or len(self._sessions) == 1
        if len(self._sessions) == 1:
            self._add_session("空白自定义", template_document("空白自定义", connectivity="仅格点"))
            # The fallback tab is now the only session that must remain
            # active after removing the original final tab.
            active_session = self._sessions[-1]
        self.model_bar.blockSignals(True)
        self.model_bar.removeTab(index)
        del self._sessions[index]
        self.model_bar.blockSignals(False)
        if closing_active:
            # Closing the current tab selects the next tab at the same
            # position, or the previous tab when the closed tab was last.
            target = min(index, len(self._sessions) - 1)
            self._active_index = -1
            self.model_bar.setCurrentIndex(target)
            self._switch_model(target)
        else:
            # Closing a background tab must not change the calculation/model
            # shown in the editor.  Its object identity survives index shifts
            # caused by removing a tab to its left.
            target = self._sessions.index(active_session)
            self._active_index = target
            if self.model_bar.currentIndex() != target:
                self.model_bar.setCurrentIndex(target)
        self._refresh_comparison_models()

    def _rename_model(self, index: int):
        if not 0 <= index < len(self._sessions):
            return
        old = self._sessions[index].meta.name
        name, ok = QInputDialog.getText(self, "重命名模型", "模型名称", text=old)
        if ok and name.strip():
            self._sessions[index].meta.name = self._unique_name(name.strip(), exclude=index)
            self._refresh_model_tab(index)
            self._refresh_comparison_models()
            self._save_workspace_state()

    def _move_model(self, from_index: int, to_index: int):
        if not (0 <= from_index < len(self._sessions) and 0 <= to_index < len(self._sessions)):
            return
        # ``QTabBar`` moves the visual tab before emitting ``tabMoved``.  The
        # active document is identified by its session object, not by the
        # transient tab index: moving a background tab must not silently make
        # it the active calculation, and moving the active tab must preserve
        # its identity after the index changes.
        active_session = (
            self._sessions[self._active_index]
            if 0 <= self._active_index < len(self._sessions)
            else None
        )
        session = self._sessions.pop(from_index)
        self._sessions.insert(to_index, session)
        if active_session is not None:
            self._active_index = self._sessions.index(active_session)
        else:
            self._active_index = max(0, self.model_bar.currentIndex())
        self._refresh_comparison_models()
        self._save_workspace_state()

    def _unique_name(self, base: str, exclude: int = -1) -> str:
        used = {s.meta.name for i, s in enumerate(self._sessions) if i != exclude}
        if base not in used:
            return base
        n = 2
        while f"{base} {n}" in used:
            n += 1
        return f"{base} {n}"

    def _refresh_model_tab(self, index: int):
        if 0 <= index < len(self._sessions):
            session = self._sessions[index]
            self.model_bar.setTabText(index, ("• " if session.meta.dirty else "") + session.meta.name)
            self.model_bar.setTabToolTip(index, session.meta.path or "自动保存时创建模型文件")

    # Undo/redo snapshots are per model.
    @staticmethod
    def _describe_history_change(previous: dict, current: dict) -> str:
        """Return a concise user-facing name for one snapshot transition."""
        if not isinstance(previous, dict) or not isinstance(current, dict):
            return "编辑模型"
        old_boundary, new_boundary = previous.get("boundary", {}), current.get("boundary", {})
        if old_boundary != new_boundary:
            if old_boundary.get("shape") != new_boundary.get("shape"):
                return "调整盘形状"
            if (old_boundary.get("NX"), old_boundary.get("NY")) != (
                new_boundary.get("NX"), new_boundary.get("NY"),
            ):
                return "调整系统尺寸"
            return "切换边界条件"
        old_sites, new_sites = previous.get("sites", []), current.get("sites", [])
        if old_sites != new_sites:
            if len(new_sites) > len(old_sites):
                return "添加格点"
            if len(new_sites) < len(old_sites):
                return "删除格点"
            return "移动或编辑格点"
        if previous.get("cell") != current.get("cell"):
            return "调整元胞"
        old_hops, new_hops = previous.get("hops", []), current.get("hops", [])
        if old_hops != new_hops:
            if len(new_hops) > len(old_hops):
                return "添加跃迁"
            if len(new_hops) < len(old_hops):
                return "删除跃迁"
            return "调整跃迁"
        if previous.get("params") != current.get("params"):
            return "调整模型参数"
        if previous.get("kx") != current.get("kx"):
            return "调整 kₓ"
        if previous.get("lattice_display") != current.get("lattice_display"):
            return "调整晶格显示"
        if previous.get("order") != current.get("order"):
            return "调整矩阵排序"
        return "编辑模型"

    def undo(self):
        if not self._workspace_enabled:
            return
        history = self._sessions[self._active_index].history
        label = history.undo_label
        doc = history.undo()
        if doc is not None:
            self._history_replay = True
            try:
                self.controller.apply_document(doc)
            finally:
                # A failed replay must never suppress future history pushes.
                # Without this guard one malformed snapshot could leave the
                # window permanently in replay mode until restart.
                self._history_replay = False
            self.set_dirty(True)
            self.flash_status(f"已撤销：{label or '上一步编辑'}")
        self._update_history_actions()

    def redo(self):
        if not self._workspace_enabled:
            return
        history = self._sessions[self._active_index].history
        label = history.redo_label
        doc = history.redo()
        if doc is not None:
            self._history_replay = True
            try:
                self.controller.apply_document(doc)
            finally:
                self._history_replay = False
            self.set_dirty(True)
            self.flash_status(f"已重做：{label or '上一步编辑'}")
        self._update_history_actions()

    def _update_history_actions(self):
        if self._workspace_enabled and self._sessions:
            history = self._sessions[self._active_index].history
            self.action_undo.setEnabled(history.can_undo)
            self.action_redo.setEnabled(history.can_redo)
            self.lattice_undo_btn.setEnabled(history.can_undo)
            self.lattice_redo_btn.setEnabled(history.can_redo)
            undo_label = history.undo_label
            redo_label = history.redo_label
            self.action_undo.setText(f"撤销：{undo_label}" if undo_label else "撤销")
            self.action_redo.setText(f"重做：{redo_label}" if redo_label else "重做")
            self.action_undo.setToolTip(
                f"撤销：{undo_label}（Ctrl+Z）" if undo_label else "撤销（Ctrl+Z）"
            )
            self.action_redo.setToolTip(
                f"重做：{redo_label}（Ctrl+Shift+Z）" if redo_label else "重做（Ctrl+Shift+Z）"
            )
            self.lattice_undo_btn.setToolTip(self.action_undo.toolTip())
            self.lattice_redo_btn.setToolTip(self.action_redo.toolTip())

    # Files and autosave.
    @staticmethod
    def _safe_filename(text: str) -> str:
        value = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", text).strip(" .")
        return value or "未命名模型"

    def _default_model_path(self, session: _Session) -> Path:
        return self._workspace_root / "models" / (
            f"{self._safe_filename(session.meta.name)}-{session.meta.id[:8]}.hvisual"
        )

    def _save_active(self, choose: bool = False):
        if self._workspace_enabled:
            return self._save_session(self._sessions[self._active_index], choose)
        return self.controller.save_model()

    def _save_session(self, session: _Session, choose: bool = False, *, remember: bool = True) -> bool:
        path = Path(session.meta.path) if session.meta.path else self._default_model_path(session)
        if choose:
            chosen, _ = QFileDialog.getSaveFileName(
                self, "模型另存为", str(path), "HamiVisualizer 模型 (*.hvisual)"
            )
            if not chosen:
                return False
            path = Path(chosen)
            if path.suffix.lower() != ".hvisual":
                path = path.with_suffix(".hvisual")
        try:
            save_model(path, session.document)
        except (OSError, TypeError, ValueError) as exc:
            QMessageBox.critical(self, "保存失败", str(exc))
            return False
        session.meta.path = str(path)
        if remember:
            self._remember_recent_model(path)
        session.meta.dirty = False
        if session is self._sessions[self._active_index]:
            self._dirty = False
            self.setWindowTitle(self._base_title)
        self._refresh_model_tab(self._sessions.index(session))
        # A workspace path can be deeply nested (and may be very long on
        # Windows).  Keep the transient status message compact so it remains
        # readable at 150–180% UI scale; the full path is still available on
        # hover for users who need it.
        self.statusBar().setToolTip(str(path))
        self.statusBar().showMessage(f"模型已保存：{path.name}")
        self._save_workspace_state()
        return True

    def _autosave_current(self):
        if self._workspace_enabled and self.preferences.autosave and self._sessions:
            self._save_session(self._sessions[self._active_index], False, remember=False)

    def _open_as_tab(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开模型", str(self._workspace_root / "models"),
            "HamiVisualizer 模型 (*.hvisual *.json)",
        )
        if not path:
            return
        self._open_recent_model(path)

    @staticmethod
    def _recent_path_key(path: str | Path) -> str:
        """Stable, case-insensitive key for de-duplicating recent paths."""
        try:
            resolved = Path(path).expanduser().resolve(strict=False)
        except (OSError, RuntimeError):
            resolved = Path(path).expanduser().absolute()
        return str(resolved)

    def _normalized_recent_paths(self, paths) -> list[str]:
        recent: list[str] = []
        seen: set[str] = set()
        for value in paths:
            if not isinstance(value, str) or not value.strip():
                continue
            path = self._recent_path_key(value)
            key = path.casefold()
            if key not in seen:
                recent.append(path)
                seen.add(key)
            if len(recent) == 10:
                break
        return recent

    def _remember_recent_model(self, path: str | Path) -> None:
        normalized = self._recent_path_key(path)
        key = normalized.casefold()
        self._recent_model_paths = [
            candidate for candidate in self._recent_model_paths
            if candidate.casefold() != key
        ]
        self._recent_model_paths.insert(0, normalized)
        del self._recent_model_paths[10:]
        self._save_workspace_state()

    def _refresh_recent_models_menu(self) -> None:
        menu = self.recent_models_menu
        menu.clear()
        if not self._recent_model_paths:
            empty = menu.addAction("暂无最近模型")
            empty.setEnabled(False)
            return
        for path_text in list(self._recent_model_paths):
            path = Path(path_text)
            action = menu.addAction(path.name)
            action.setToolTip(str(path))
            action.triggered.connect(
                lambda _checked=False, selected=path_text: self._open_recent_model(selected)
            )
        menu.addSeparator()
        clear = menu.addAction("清空最近模型记录")
        clear.triggered.connect(self._clear_recent_models)

    def _clear_recent_models(self) -> None:
        self._recent_model_paths.clear()
        self._save_workspace_state()
        # The action is triggered from the open menu itself.  Rebuild it now
        # instead of waiting for the next ``aboutToShow`` signal; otherwise
        # the just-cleared entries remain visible until the user closes and
        # reopens the menu, which makes the command feel ineffective.
        self._refresh_recent_models_menu()
        self.statusBar().showMessage("已清空最近模型记录")

    def _open_recent_model(self, path: str | Path) -> None:
        """Open one model as a tab, or select its existing tab when already open."""
        normalized = self._recent_path_key(path)
        file_path = Path(normalized)
        if not file_path.is_file():
            self._recent_model_paths = [
                candidate for candidate in self._recent_model_paths
                if candidate.casefold() != normalized.casefold()
            ]
            self._save_workspace_state()
            QMessageBox.warning(self, "文件不可用", f"最近模型文件不存在，已从列表移除：\n{file_path}")
            return
        for index, session in enumerate(self._sessions):
            if session.meta.path and self._recent_path_key(session.meta.path).casefold() == normalized.casefold():
                self.model_bar.setCurrentIndex(index)
                self._remember_recent_model(normalized)
                self.statusBar().showMessage(f"已切换到已打开的模型：{file_path.name}")
                return
        try:
            doc = load_model(file_path)
        except (OSError, TypeError, ValueError) as exc:
            self._recent_model_paths = [
                candidate for candidate in self._recent_model_paths
                if candidate.casefold() != normalized.casefold()
            ]
            self._save_workspace_state()
            QMessageBox.critical(self, "打开失败", f"模型无法读取，已从最近列表移除。\n{exc}")
            return
        self._add_session(file_path.stem, doc, str(file_path))
        self._remember_recent_model(file_path)
        self.statusBar().showMessage(f"模型已打开：{file_path}")

    def _save_workspace_state(self):
        if not self._workspace_enabled:
            return
        data = WorkspaceData(
            sessions=[s.meta for s in self._sessions],
            recent_models=list(self._recent_model_paths),
            current_index=max(0, self._active_index),
            split_enabled=self.action_split.isChecked(),
            split_model_id=self.comparison.selected_model_id,
            split_result=self.comparison.selected_result,
            splitter_sizes=self.result_splitter.sizes(),
        )
        try:
            save_workspace(data, self._workspace_root)
        except OSError as exc:
            self.statusBar().showMessage(f"工作区保存失败：{exc}")

    # Comparison pane.
    def _set_split_enabled(self, enabled: bool):
        self.comparison.setVisible(bool(enabled))
        if enabled:
            self.result_splitter.setSizes([1, 1])
            self._refresh_comparison()
        self._save_workspace_state()

    def _refresh_comparison_models(self, selected_id: str = ""):
        self.comparison.set_models([(s.meta.id, s.meta.name) for s in self._sessions], selected_id)
        self._refresh_comparison()

    def _refresh_comparison(self):
        if not self.comparison.isVisible():
            return
        selected = self.comparison.selected_model_id
        session = next((s for s in self._sessions if s.meta.id == selected), None)
        if session is not None and not session.cache:
            self._build_preview_cache(session)
        self.comparison.set_cache(session.cache if session is not None else None)
        self._save_workspace_state()

    def _build_preview_cache(self, session: _Session):
        """Lazily render an unopened model in an isolated, hidden editor."""
        from ..controller import ViewController

        preview = MainWindow()
        preview._theme_mode = self._theme_mode
        preview._dark = self._dark
        controller = ViewController(preview, connect_actions=False)
        preview.controller = controller
        controller.apply_document(session.document)
        session.cache = {
            "matrix": preview.matrix_scene._data,
            "lattice": preview.lattice_scene._data,
            "band": preview.band_scene._data,
            "wavefunction": preview.wf_view._data,
        }
        preview.deleteLater()
        # 预览窗口构造时会重新应用系统主题，这里恢复主窗口当前主题，
        # 避免用户显式选择的明暗被临时覆盖。
        self._apply_style(self._ui_scale)

    # Preferences and complete UI scaling.
    def _show_preferences(self):
        dialog = PreferencesDialog(self.preferences, self)
        if dialog.exec() != QDialog.Accepted:
            return
        self.preferences = dialog.values()
        save_preferences(self.preferences, self._workspace_root)
        self._theme_mode = self.preferences.theme
        self._dark = self._resolve_dark(self._theme_mode)
        self._sync_theme_actions()
        self._set_ui_scale(self.preferences.ui_scale / 100.0, persist=False)
        self.lattice_snap_btn.blockSignals(True)
        self.lattice_snap_btn.setChecked(self.preferences.snap_enabled)
        self.lattice_snap_btn.blockSignals(False)
        for session in self._sessions:
            session.history.set_limit(self.preferences.undo_limit)
        self.controller.set_runtime_preferences(
            debounce_ms=self.preferences.debounce_ms,
            calculation_mode=self.preferences.calculation_mode,
            snap_step=self.preferences.snap_step,
            snap_enabled=self.preferences.snap_enabled,
        )
        self.statusBar().showMessage("偏好设置已应用")

    def _change_ui_scale(self, delta: float):
        self._set_ui_scale(self._ui_scale + float(delta))

    def _fit_after_scale(self):
        """Fit views after a UI-scale layout pass while the window is alive."""
        controller = getattr(self, "controller", None)
        if controller is None:
            return
        try:
            controller.fit_all(force=False)
        except RuntimeError:
            # The parented timer normally prevents this path; keep teardown
            # harmless on platforms that destroy child views first.
            return

    def _set_ui_scale(self, value: float, *, persist: bool = True):
        scale = min(1.80, max(0.80, round(float(value) / 0.10) * 0.10))
        self._ui_scale = scale
        self._relayout_lattice_edit_toolbar()
        font = QFont(self._base_ui_font)
        base_size = font.pointSizeF() if font.pointSizeF() > 0 else 10.0
        font.setPointSizeF(base_size * scale)
        app = QApplication.instance()
        if app is not None:
            app.setFont(font)
        self._apply_style(scale)
        metrics = QFontMetrics(font)
        row_height = max(round(24 * scale), metrics.height() + round(9 * scale))
        for table in (self.panel.site_table, self.panel.hop_table, self.panel.param_table):
            table.verticalHeader().setDefaultSectionSize(row_height)
            # The QSS header padding is scaled too; leave an explicit
            # headroom budget so CJK headers never touch/crop against the top
            # rule at 150–180% UI scale.
            table.horizontalHeader().setMinimumHeight(
                max(row_height, metrics.height() + round(12 * scale))
            )
        # Parameter tables with a handful of symbols should expand with the
        # new row metrics; otherwise the last row can remain hidden behind a
        # nested scrollbar after switching to 150–180% UI scale.
        self.panel._fit_param_table_height()
        # The parameter table keeps two fixed columns; leaving their widths at
        # the 100% defaults makes ``omg``/long custom names elide at 150–180%
        # even though the surrounding layout has enough room. Scale these
        # columns with the application font just like row heights.
        self.panel.param_table.setColumnWidth(0, round(70 * scale))
        self.panel.param_table.setColumnWidth(1, round(60 * scale))
        for group in (
            self.panel.boundary_group, self.panel.params_group, self.panel.energy_group,
            self.panel.display_group, self.panel.sites_group, self.panel.hops_group,
        ):
            if hasattr(group, "setExpanded"):
                group.setExpanded(group._expanded)
        self.panel.kx_edit.setMaximumWidth(round(82 * scale))
        # Vector/length editors sit in two compact rows.  Without an explicit
        # cap QDoubleSpinBox may claim its unconstrained style hint (hundreds
        # of pixels), forcing the entire control rail to scroll horizontally
        # at 150–180% UI scale.
        for spin in (
            self.panel.lx_spin, self.panel.ly_spin,
            self.panel.a1x_spin, self.panel.a1y_spin,
            self.panel.a2x_spin, self.panel.a2y_spin,
        ):
            # 112 px at 100% keeps two vector components and the action
            # button on one rail without forcing a horizontal scroll bar.
            spin.setMaximumWidth(round(112 * scale))
            spin.setMinimumWidth(round(78 * scale))
            spin.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
            # Geometry values are deliberately shown with high precision.
            # Hide the native button strip so the final digits remain visible
            # at 150–180% UI scale; wheel and keyboard stepping still work.
            spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        # Keep group-box titles and long field labels readable at the upper
        # UI-scale limit.  The minimum remains 350 px at the default scale;
        # the result pane still has ample room in the supported window size.
        self.panel_scroll.setMinimumWidth(max(350, round(350 * scale)))
        if self._workspace_enabled:
            self.preferences.ui_scale = round(scale * 100)
            if persist:
                save_preferences(self.preferences, self._workspace_root)
        # 字号和控制栏宽度会改变图形视口尺寸。布局完成后重新适配尚未
        # 被用户手动缩放的视图；已平移/缩放的研究位置保持不动。
        controller = getattr(self, "controller", None)
        if controller is not None:
            self._fit_after_scale_timer.start(0)
        self.statusBar().showMessage(f"界面缩放 {round(scale * 100)}%", 1200)

    def _apply_style(self, scale: float):
        """按当前主题与缩放生成 QSS，并应用到应用级（含所有对话框）。"""
        app = QApplication.instance()
        self._dark = self._resolve_dark(self._theme_mode)
        theme = DARK if self._dark else LIGHT
        if app is not None:
            app.setPalette(app_palette(theme))
            app.setStyleSheet(app_stylesheet(theme, scale))
        self._propagate_theme(self._dark)

    def _resolve_dark(self, mode: str) -> bool:
        app = QApplication.instance()
        return resolve_theme(mode, app) == "dark"

    def _propagate_theme(self, dark: bool):
        """把明暗标记下发给主窗口与比较窗格的所有场景。"""
        for scene in (self.matrix_scene, self.lattice_scene, self.band_scene):
            setter = getattr(scene, "set_theme", None)
            if setter is not None:
                setter(dark)
        wf_setter = getattr(self.wf_view, "set_theme", None)
        if wf_setter is not None:
            wf_setter(dark)
        comparison = getattr(self, "comparison", None)
        if comparison is not None:
            for scene in (comparison.matrix_scene, comparison.lattice_scene, comparison.band_scene):
                setter = getattr(scene, "set_theme", None)
                if setter is not None:
                    setter(dark)
            cwf = getattr(comparison.wf_view, "set_theme", None)
            if cwf is not None:
                cwf(dark)

    def _set_theme_mode(self, mode: str):
        mode = mode if mode in {"light", "dark", "system"} else "system"
        self._theme_mode = mode
        if self._workspace_enabled:
            self.preferences.theme = mode
            save_preferences(self.preferences, self._workspace_root)
        self._apply_style(self._ui_scale)
        self._sync_theme_actions()
        label = {"light": "浅色", "dark": "深色", "system": "跟随系统"}[mode]
        self.statusBar().showMessage(f"外观主题：{label}", 1500)

    def _sync_theme_actions(self):
        for mode, name in (("light", "theme_light_action"),
                           ("dark", "theme_dark_action"),
                           ("system", "theme_system_action")):
            action = getattr(self, name, None)
            if action is not None:
                action.setChecked(self._theme_mode == mode)

    def _on_system_color_scheme_changed(self, *_args):
        if self._theme_mode == "system":
            self._apply_style(self._ui_scale)

    def closeEvent(self, event):
        if self._workspace_enabled:
            if any(s.meta.dirty for s in self._sessions):
                answer = QMessageBox.question(
                    self, "退出并保存？", "部分模型有未保存的更改。是否保存后退出？",
                    QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                    QMessageBox.Save,
                )
                if answer == QMessageBox.Cancel:
                    event.ignore(); return
                if answer == QMessageBox.Save:
                    for session in self._sessions:
                        if session.meta.dirty and not self._save_session(session, False):
                            event.ignore(); return
            elif self.preferences.autosave:
                self._autosave_current()
            self._save_workspace_state()
            event.accept(); return
        if not self._dirty:
            event.accept(); return
        answer = QMessageBox.question(
            self, "保存更改？", "当前模型有尚未保存的更改。",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save,
        )
        if answer == QMessageBox.Cancel:
            event.ignore(); return
        if answer == QMessageBox.Save:
            controller = getattr(self, "controller", None)
            if controller is None or not controller.save_model():
                event.ignore(); return
        event.accept()

    def _fit_views(self):
        c = getattr(self, "controller", None)
        if c is not None:
            c.fit_all(force=True)
        if self.comparison.isVisible():
            self.comparison.fit_current()

    def showEvent(self, event):
        """Finish deferred workspace views after the window has a real viewport.

        Workspace restoration runs before ``main()`` shows the window.  A
        comparison cache cannot be fitted while its graphics view has a
        zero-sized viewport, so the initial refresh may intentionally be
        skipped.  Retry once on the first visible frame; this prevents a
        restored split-comparison pane from appearing blank until the user
        changes tabs or models.
        """
        super().showEvent(event)
        if self._workspace_enabled and self.action_split.isChecked():
            QTimer.singleShot(0, self._refresh_comparison)

    def all_views(self) -> list:
        return [
            self.matrix_gv, self.lattice_gv, self.band_gv,
            self.combined_matrix_gv, self.combined_lattice_gv, self.wf_view.view,
        ]
