"""Compact dialogs for preferences, template creation, and visual hopping edits."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSlider,
    QSpinBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..model.expression import parse_expression
from ..model.templates import TEMPLATE_NAMES
from ..model.workspace import Preferences
from ..model.persistence import MAX_NX, MAX_NY
from ..model.boundary import (
    SHAPE_DISK, SHAPE_HEXAGON, SHAPE_RECTANGLE, SHAPE_TRIANGLE,
)

# Keep the dialog aligned with model/persistence validation.  The field is a
# bounded integer editor for safety, but ±1000 is wide enough for long-range
# Bloch harmonics and matches the portable model format.
CELL_OFFSET_INPUT_MAX = 1000


class PreferencesDialog(QDialog):
    def __init__(self, prefs: Preferences, parent=None):
        super().__init__(parent)
        self.setWindowTitle("偏好设置")
        self.setMinimumWidth(420)
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.ui_scale = QSpinBox()
        self.ui_scale.setRange(80, 180)
        self.ui_scale.setSingleStep(10)
        self.ui_scale.setSuffix(" %")
        self.ui_scale.setValue(prefs.ui_scale)
        form.addRow("界面缩放", self.ui_scale)
        self.theme = QComboBox()
        self.theme.addItem("跟随系统", "system")
        self.theme.addItem("浅色", "light")
        self.theme.addItem("深色", "dark")
        idx = self.theme.findData(prefs.theme)
        self.theme.setCurrentIndex(idx if idx >= 0 else 0)
        form.addRow("外观主题", self.theme)
        self.undo_limit = QSpinBox()
        self.undo_limit.setRange(1, 100)
        self.undo_limit.setValue(prefs.undo_limit)
        form.addRow("最大撤销步数", self.undo_limit)
        self.autosave = QCheckBox("编辑后自动保存")
        self.autosave.setChecked(prefs.autosave)
        form.addRow("自动保存", self.autosave)
        self.calc_mode = QComboBox()
        self.calc_mode.addItem("自动计算（推荐）", "automatic")
        self.calc_mode.addItem("手动计算", "manual")
        self.calc_mode.setCurrentIndex(0 if prefs.calculation_mode == "automatic" else 1)
        form.addRow("重新计算", self.calc_mode)
        self.debounce = QSpinBox()
        self.debounce.setRange(100, 3000)
        self.debounce.setSingleStep(50)
        self.debounce.setSuffix(" ms")
        self.debounce.setValue(prefs.debounce_ms)
        form.addRow("自动计算等待", self.debounce)
        self.snap = QDoubleSpinBox()
        self.snap.setRange(0.001, 10.0)
        self.snap.setDecimals(3)
        self.snap.setValue(prefs.snap_step)
        form.addRow("格点吸附间隔", self.snap)
        self.snap_enabled = QCheckBox("拖动格点时启用智能吸附")
        self.snap_enabled.setChecked(prefs.snap_enabled)
        self.snap_enabled.setToolTip(
            "吸附到网格、同列/同行和相邻元胞的精确几何位置；按住 Alt 可临时自由移动。"
        )
        form.addRow("格点吸附", self.snap_enabled)
        self.check_updates = QCheckBox("启动时检查 GitHub Releases（预留）")
        self.check_updates.setChecked(prefs.check_updates)
        self.check_updates.setEnabled(False)
        form.addRow("版本更新", self.check_updates)
        lay.addLayout(form)
        note = QLabel("界面缩放只影响菜单、标签、侧栏和控件，不改变矩阵或晶格的视图缩放。")
        note.setWordWrap(True)
        note.setObjectName("dialogNote")
        lay.addWidget(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("确定")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    def values(self) -> Preferences:
        return Preferences(
            ui_scale=self.ui_scale.value(),
            undo_limit=self.undo_limit.value(),
            autosave=self.autosave.isChecked(),
            calculation_mode=self.calc_mode.currentData(),
            debounce_ms=self.debounce.value(),
            snap_step=self.snap.value(),
            snap_enabled=self.snap_enabled.isChecked(),
            check_updates=self.check_updates.isChecked(),
            theme=self.theme.currentData(),
        ).normalized()


class TemplateDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("新建模型")
        self.setMinimumWidth(440)
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.template = QComboBox()
        self.template.addItems(TEMPLATE_NAMES)
        form.addRow("晶格模板", self.template)
        self.boundary = QComboBox()
        self.boundary.addItem("半无限（x-Bloch）", "semi")
        self.boundary.addItem("双开边界（OBC）", "obc")
        form.addRow("边界", self.boundary)
        self.shape = QComboBox()
        for label, value in (
            ("矩形 / 方形盘", SHAPE_RECTANGLE),
            ("三角形盘", SHAPE_TRIANGLE),
            ("圆盘", SHAPE_DISK),
            ("六边形盘", SHAPE_HEXAGON),
        ):
            self.shape.addItem(label, value)
        self.shape.setToolTip("双开边界下选择有限盘的形状；半无限边界忽略此项。")
        form.addRow("盘形状", self.shape)
        dims = QWidget()
        dims_lay = QVBoxLayout(dims)
        dims_lay.setContentsMargins(0, 0, 0, 0)
        self.nx = QSpinBox(); self.nx.setRange(1, MAX_NX); self.nx.setValue(4)
        self.ny = QSpinBox(); self.ny.setRange(1, MAX_NY); self.ny.setValue(4)
        self.nx.setToolTip("x 方向元胞数；可输入大于滑块常用范围的精确值。")
        self.ny.setToolTip("y 方向元胞数；可输入大于滑块常用范围的精确值。")
        self.nx_slider = self._dimension_slider(4)
        self.ny_slider = self._dimension_slider(4)
        self.nx.valueChanged.connect(lambda value: self._sync_dimension_slider(self.nx_slider, value))
        self.ny.valueChanged.connect(lambda value: self._sync_dimension_slider(self.ny_slider, value))
        self.nx_slider.valueChanged.connect(lambda value: self._sync_dimension_spin(self.nx, value))
        self.ny_slider.valueChanged.connect(lambda value: self._sync_dimension_spin(self.ny, value))
        dims_lay.addLayout(self._dimension_row("NX", self.nx, self.nx_slider))
        dims_lay.addLayout(self._dimension_row("NY", self.ny, self.ny_slider))
        form.addRow("系统尺寸", dims)
        self.connectivity = QComboBox()
        self.connectivity.addItems(["仅格点", "最近邻", "最近邻+次近邻"])
        self.connectivity.setCurrentText("最近邻")
        form.addRow("自动生成跃迁", self.connectivity)
        self.replace = QCheckBox("替换当前模型（默认新建标签）")
        form.addRow("创建方式", self.replace)
        lay.addLayout(form)
        self.preview = QLabel()
        self.preview.setWordWrap(True)
        self.preview.setObjectName("templatePreview")
        lay.addWidget(self.preview)
        self.template.currentTextChanged.connect(self._update_preview)
        self.template.currentTextChanged.connect(self._sync_kagome_defaults)
        self.template.currentTextChanged.connect(self._sync_haldane_defaults)
        self.connectivity.currentTextChanged.connect(self._update_preview)
        self.shape.currentTextChanged.connect(self._update_preview)
        self.boundary.currentIndexChanged.connect(self._sync_shape_enabled)
        self.boundary.currentIndexChanged.connect(self._sync_kagome_defaults)
        self._sync_shape_enabled()
        self._sync_kagome_defaults()
        self._sync_haldane_defaults()
        self._update_preview()
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("创建模型")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    @staticmethod
    def _dimension_slider(value: int) -> QSlider:
        slider = QSlider(Qt.Horizontal)
        slider.setRange(1, 64)
        slider.setValue(value)
        slider.setSingleStep(1)
        slider.setPageStep(1)
        slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        return slider

    @staticmethod
    def _dimension_row(label: str, spin: QSpinBox, slider: QSlider) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        title = QLabel(label)
        title.setMinimumWidth(28)
        row.addWidget(title)
        spin.setMinimumWidth(64)
        row.addWidget(spin)
        row.addWidget(slider, 1)
        return row

    @staticmethod
    def _sync_dimension_slider(slider: QSlider, value: int):
        slider.blockSignals(True)
        if value > slider.maximum():
            slider.setMaximum(value)
        slider.setValue(value)
        slider.blockSignals(False)

    @staticmethod
    def _sync_dimension_spin(spin: QSpinBox, value: int):
        spin.blockSignals(True)
        spin.setValue(value)
        spin.blockSignals(False)

    def _update_preview(self):
        template_name = self.template.currentText()
        if (
            template_name == "Kagome"
            and self.boundary.currentData() == "obc"
            and self.shape.currentData() == SHAPE_TRIANGLE
        ):
            details = "Kagome 三角纳米盘使用 3 格点斜原胞，边界是三条平直等边；"
        elif template_name == "SSH":
            details = (
                "SSH 含胞内 t1 与胞间 t2；双开边界下调小 t1、调大 t2 可观察端点态；"
            )
        elif template_name == "Haldane":
            details = (
                "Haldane 含复数次近邻相位 φ 与子格质量 m；调节 m/φ 可观察能带开隙；"
            )
        else:
            details = ""
        self.preview.setText(
            f"将创建“{self.template.currentText()}”模型，"
            f"包含{self.connectivity.currentText()}，"
            f"{self.shape.currentText()}。{details}"
            "创建后可在晶格页直接拖动格点；"
            f"需要添加跃迁时，先点击工具栏“添加跃迁”，再选择两个格点。"
        )

    def _sync_shape_enabled(self):
        self.shape.setEnabled(self.boundary.currentData() == "obc")

    def _sync_kagome_defaults(self):
        """Make the intended Kagome nanodisk discoverable without hiding choices.

        The shape remains editable: selecting another shape afterwards is
        respected.  Only the transition into the Kagome+OBC combination
        applies the safe, physically explicit default.
        """
        if (self.template.currentText() == "Kagome"
                and self.boundary.currentData() == "obc"
                and self.shape.currentData() == SHAPE_RECTANGLE):
            index = self.shape.findData(SHAPE_TRIANGLE)
            if index >= 0:
                self.shape.blockSignals(True)
                self.shape.setCurrentIndex(index)
                self.shape.blockSignals(False)
                self._update_preview()

    def _sync_haldane_defaults(self):
        """Make Haldane complete by default without hiding user choices.

        The generic wizard starts on nearest-neighbour-only connections for
        lightweight templates.  Haldane's defining complex terms are
        next-nearest neighbours, so selecting the template opts into that
        family once; a later explicit nearest-only choice is left untouched.
        """
        if (self.template.currentText() == "Haldane"
                and self.connectivity.currentText() == "最近邻"):
            self.connectivity.setCurrentText("最近邻+次近邻")

    def values(self) -> dict:
        return {
            "name": self.template.currentText(),
            "nx": self.nx.value(),
            "ny": self.ny.value(),
            "boundary_kind": self.boundary.currentData(),
            "shape": self.shape.currentData(),
            "connectivity": self.connectivity.currentText(),
            "replace": self.replace.isChecked(),
        }


class HoppingDialog(QDialog):
    def __init__(
        self,
        from_site: int,
        to_site: int,
        parent=None,
        *,
        semi: bool = False,
        site_count: int | None = None,
        cell_offset: tuple[int, int] = (0, 0),
    ):
        super().__init__(parent)
        self.setWindowTitle("添加跃迁")
        self._semi = bool(semi)
        try:
            initial_offset = (int(cell_offset[0]), int(cell_offset[1]))
        except (IndexError, TypeError, ValueError):
            initial_offset = (0, 0)
        if any(abs(value) > CELL_OFFSET_INPUT_MAX for value in initial_offset):
            initial_offset = (0, 0)
        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.from_combo = None
        self.to_combo = None
        if site_count is not None and int(site_count) > 0:
            # The side-panel entry has no pre-selected canvas endpoints. Give
            # users an explicit endpoint choice instead of silently creating
            # another 0→1 row and forcing a second table edit.
            count = int(site_count)
            self.from_combo = QComboBox()
            self.to_combo = QComboBox()
            for index in range(count):
                label = f"格点 {index + 1}"
                self.from_combo.addItem(label, index)
                self.to_combo.addItem(label, index)
            self.from_combo.setCurrentIndex(max(0, min(count - 1, int(from_site))))
            self.to_combo.setCurrentIndex(max(0, min(count - 1, int(to_site))))
            form.addRow("起点", self.from_combo)
            form.addRow("终点", self.to_combo)
        else:
            form.addRow("连接", QLabel(f"格点 {from_site} → {to_site}"))
        self.name = QLineEdit("t")
        self.amplitude = QLineEdit("-t")
        self.phase = QLineEdit("0")
        self.off_x = QSpinBox(); self.off_x.setRange(-CELL_OFFSET_INPUT_MAX, CELL_OFFSET_INPUT_MAX)
        self.off_y = QSpinBox(); self.off_y.setRange(-CELL_OFFSET_INPUT_MAX, CELL_OFFSET_INPUT_MAX)
        self.off_x.setToolTip("目标元胞相对起点的 x 偏移，范围 −1000…1000")
        self.off_y.setToolTip("目标元胞相对起点的 y 偏移，范围 −1000…1000")
        self.off_x.valueChanged.connect(self._update_relation_effect)
        self.off_y.valueChanged.connect(self._update_relation_effect)
        self.cell_relation = QComboBox()
        # QComboBox's QVariant bridge does not preserve Python tuple equality
        # consistently across PySide6 versions.  Store a tiny string token
        # and decode it locally so the semantic choices remain deterministic.
        relation_items = (
            ("胞内（同一元胞，0, 0）", "0,0"),
            ("右侧胞间（+1, 0）", "1,0"),
            ("左侧胞间（−1, 0）", "-1,0"),
            ("上方胞间（0, +1）", "0,1"),
            ("下方胞间（0, −1）", "0,-1"),
            ("自定义偏移…", "custom"),
        )
        for label, offset in relation_items:
            self.cell_relation.addItem(label, offset)
        self.cell_relation.setToolTip(
            "选择目标格点位于同一元胞还是相邻元胞；半无限模式的左右胞间对应 Bloch 方向。"
        )
        self.cell_relation.currentIndexChanged.connect(self._sync_cell_relation)
        form.addRow("位置", self.cell_relation)
        form.addRow("参数名", self.name)
        form.addRow("幅度", self.amplitude)
        form.addRow("相位", self.phase)
        self.off_x_label = QLabel("元胞偏移 x")
        self.off_y_label = QLabel("元胞偏移 y")
        form.addRow(self.off_x_label, self.off_x)
        form.addRow(self.off_y_label, self.off_y)
        lay.addLayout(form)
        self.relation_effect = QLabel()
        self.relation_effect.setWordWrap(True)
        self.relation_effect.setObjectName("dialogNote")
        self.relation_effect.setToolTip(
            "说明该跃迁在当前边界条件下进入哪一个哈密顿量块。"
        )
        lay.addWidget(self.relation_effect)
        hint_text = (
            "半无限模式：左右胞间（±1, 0）是 Bloch 方向，会在矩阵中产生跨元胞相位；"
            "上下胞间用于有限方向连接。\n"
            if semi else ""
        ) + "相位为 0 时生成实跃迁；输入 phi 等符号时自动采用复相位。反向共轭由程序补全。"
        hint = QLabel(hint_text)
        hint.setWordWrap(True)
        hint.setObjectName("dialogNote")
        lay.addWidget(hint)
        self.error_label = QLabel("")
        self.error_label.setWordWrap(True)
        self.error_label.setObjectName("errorLabel")
        self.error_label.setVisible(False)
        lay.addWidget(self.error_label)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("确定")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)
        token = f"{initial_offset[0]},{initial_offset[1]}"
        initial_index = self.cell_relation.findData(token)
        if initial_index < 0:
            initial_index = self.cell_relation.findData("custom")
        if initial_index >= 0:
            self.cell_relation.setCurrentIndex(initial_index)
        if self.cell_relation.currentData() == "custom":
            self.off_x.setValue(initial_offset[0])
            self.off_y.setValue(initial_offset[1])
        self._sync_cell_relation(self.cell_relation.currentIndex())

    def _accept_if_valid(self) -> None:
        """Validate dialog expressions before closing.

        The main panel still performs the authoritative model validation, but
        rejecting malformed input here keeps a failed edit from looking like
        an inter-cell offset problem.  The parser is the same restricted
        expression language used by the controller, so the dialog cannot
        silently accept syntax that will later be rejected during rebuild.
        """
        try:
            parse_expression(self.amplitude.text().strip() or "1.0")
            parse_expression(self.phase.text().strip() or "0", empty_value=0)
        except ValueError as exc:
            self.error_label.setText(f"表达式无效：{exc}")
            self.error_label.setVisible(True)
            self.amplitude.setFocus()
            return
        self.error_label.clear()
        self.error_label.setVisible(False)
        self.accept()

    def _sync_cell_relation(self, _index: int = -1):
        """Keep semantic presets and the numeric fallback synchronized."""
        token = self.cell_relation.currentData()
        custom = token == "custom"
        self.off_x.setEnabled(custom)
        self.off_y.setEnabled(custom)
        self.off_x_label.setEnabled(custom)
        self.off_y_label.setEnabled(custom)
        if not custom:
            ox, oy = (int(value) for value in str(token).split(",", 1))
            self.off_x.blockSignals(True)
            self.off_y.blockSignals(True)
            self.off_x.setValue(int(ox))
            self.off_y.setValue(int(oy))
            self.off_x.blockSignals(False)
            self.off_y.blockSignals(False)
        self._update_relation_effect()

    def _update_relation_effect(self, *_args):
        """Explain the selected offset's matrix consequence in plain language."""
        if not hasattr(self, "relation_effect"):
            return
        try:
            dx, dy = int(self.off_x.value()), int(self.off_y.value())
        except (TypeError, ValueError):
            self.relation_effect.setText("关系预览：请输入整数元胞偏移。")
            return
        if not self._semi:
            if dx == 0 and dy == 0:
                text = "关系预览：胞内项（同一元胞，进入有限矩阵主块）。"
            else:
                text = (
                    f"关系预览：胞间项（dx={dx:+d}, dy={dy:+d}）；"
                    "双开边界下目标超出样品时会按边界规则截断。"
                )
        elif dx == 0 and dy == 0:
            text = "关系预览：胞内项，写入 H₀（不产生 Bloch 相位）。"
        elif dx != 0:
            order = abs(dx)
            harmonic = "H₁" if order == 1 else f"H{order}"
            text = (
                f"关系预览：胞间项（dx={dx:+d}, dy={dy:+d}），"
                f"进入 x-Bloch {harmonic} 谐波 e^{{±i{order}kₓ}}；"
                "dy 方向仍按有限元胞编号处理。"
            )
        else:
            text = (
                f"关系预览：有限方向胞间项（dy={dy:+d}），"
                "连接相邻 y 元胞；靠近边界时会被截断。"
            )
        self.relation_effect.setText(text)

    def row(self, from_site: int | None = None, to_site: int | None = None) -> list:
        if self.from_combo is not None and self.to_combo is not None:
            from_site = int(self.from_combo.currentData())
            to_site = int(self.to_combo.currentData())
        if from_site is None or to_site is None:
            raise ValueError("必须提供跃迁起点和终点")
        phase = self.phase.text().strip() or "0"
        return [
            self.name.text().strip() or "t", from_site, to_site,
            self.off_x.value(), self.off_y.value(),
            self.amplitude.text().strip() or "-t",
            "none" if phase in {"0", "0.0"} else "phase",
            phase, 1,
        ]
