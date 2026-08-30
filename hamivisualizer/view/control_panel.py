"""控制面板: 边界模式 / 格点表 / 跃迁表 / 参数 / 符号切换 / 错误标签.

面板是用户编辑模型的唯一入口, 通过 changed 信号通知 Controller 重建。
表格列:
  格点表: x | y | 子格
  跃迁表: name | from | to | off_x | off_y | amplitude | phase_mode | phase | sign
  参数表: 名称 | 数值 | 滑块   (自动由跃迁表达式中的符号生成, 如 t/φ/ω/任意自定义名)
          φ 的数值列显示 φ/π (与 MATLAB 一致), 内部换算回弧度。
"""

from __future__ import annotations

import math
import re
from fractions import Fraction

from ..model.expression import evaluate_expression, parse_expression
from ..model.persistence import MAX_NX, MAX_NY
from ..model.boundary import (
    BoundaryKind, SHAPE_DISK, SHAPE_HEXAGON, SHAPE_RECTANGLE, SHAPE_TRIANGLE,
)
from ..model.hamiltonian import (
    DENSE_WORKING_SET_LIMIT_BYTES,
    estimate_dense_working_set_bytes,
)

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush, QColor, QFontMetrics, QKeyEvent, QPainter, QPalette, QPen, QPolygonF,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QDialog,
    QPushButton,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from .dialogs import HoppingDialog

SITE_COLS = ["x", "y", "子格"]
# 表格显示名保持简短；解析仍按固定列序读取，JSON/API 语义不变。
HOP_COLS = ["name", "from", "to", "dx", "dy", "amp", "mode", "phase", "sign"]
# Human-facing labels are intentionally separate from the stable internal
# column keys above.  This keeps the JSON/API schema unchanged while making
# the compact table and its optional inter-cell columns understandable to
# users who are not reading the implementation vocabulary.
# The compact rail must never clip its table headers.  The detailed meaning
# stays in the table tooltip and context help; these short, conventional
# labels remain legible at 80–180% application scale.
HOP_HEADERS = ["名称", "从", "到", "Δx", "Δy", "幅度", "相位模式", "相位", "符号"]
# Endpoints are deliberately one-based in the visible editor.  The model and
# persisted JSON remain zero-based, so the conversion is kept at this single
# UI boundary instead of leaking into the Hamiltonian builder.
HOP_ENDPOINT_COLUMNS = (1, 2)
PARAM_COLS = ["参数", "数值", "滑块"]
# Keep the unabridged cell text separate from the semantic relation marker
# stored in ``Qt.UserRole``.  Qt may render long coordinates as an ellipsis at
# 150–180% UI scale; this role lets the tooltip always expose the exact value.
RAW_VALUE_ROLE = Qt.UserRole + 100

# 参数滑块配置: name → (int_min, int_max, scale)  value = 滑块整数 × scale
# φ 的数值列按 φ/π 显示 (MATLAB §2.1), scale 仍以弧度计。
PARAM_SLIDERS = {
    "phi": (-100, 100, math.pi / 100.0),
    "t": (-300, 300, 0.01),
    "t1": (-300, 300, 0.01),
    "t2": (-300, 300, 0.01),
    "tc": (-300, 300, 0.01),
    "omg": (-200, 200, 0.01),
    "omega": (-200, 200, 0.01),
}
_INTEGER_RE = re.compile(r"^[+-]?\d+$")
# Dimensions are physical model inputs, not a rendering-size preset.  Keep the
# editor generous so that band/ribbon studies are not artificially capped at
# the old 10/12-cell limits.  The slider starts with a practical range and is
# expanded automatically when a larger value is entered precisely.
DIM_INPUT_MAX = max(MAX_NX, MAX_NY)
DIM_SLIDER_DEFAULT_MAX = 64


class CollapsibleGroupBox(QGroupBox):
    """A clear, clickable accordion card with an in-widget title bar.

    ``QGroupBox::title`` lives in the style's outer margin.  That makes it
    look like text inserted through the card border, and it can be visibly
    sliced when a scroll area stops on the next section.  The title here is
    painted inside a full-width hit target instead, so every left-rail card
    has a stable label, a disclosure affordance and an immediate hover state.
    """

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._base_title = title
        self._expanded = True
        self._header_hovered = False
        self._header_pressed = False
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAccessibleName(title)
        self.setAccessibleDescription("点击标题栏展开或收起此参数分组")
        # The visual title is painted by ``paintEvent``.  Keep QGroupBox's own
        # title empty so Qt never places a second label in its border margin.
        self.setTitle("")
        self.setMouseTracking(True)

    def _header_height(self) -> int:
        return self.fontMetrics().height() + 12

    def _header_rect(self) -> QRectF:
        margin = 7
        return QRectF(
            margin, 6,
            max(0, self.width() - margin * 2),
            self._header_height(),
        )

    def _set_header_hovered(self, hovered: bool) -> None:
        hovered = bool(hovered)
        if hovered == self._header_hovered:
            return
        self._header_hovered = hovered
        self.setCursor(Qt.PointingHandCursor if hovered else Qt.ArrowCursor)
        self.update(self._header_rect().toAlignedRect())

    def setExpanded(self, expanded: bool):  # noqa: N802
        self._expanded = bool(expanded)
        for child in self.findChildren(QWidget, options=Qt.FindDirectChildrenOnly):
            child.setVisible(self._expanded)
        if self._expanded:
            self.setMaximumHeight(16777215)
            self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        else:
            self.setMaximumHeight(self._header_height() + 12)
            self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.updateGeometry()
        self.update()

    def paintEvent(self, event):  # noqa: N802
        """Paint an accessible card header and font-independent arrow."""
        super().paintEvent(event)
        header = self._header_rect()
        palette = self.palette()
        accent = palette.color(QPalette.Highlight)
        if self._header_pressed:
            bg = QColor(accent)
            bg.setAlpha(235)
            fg = palette.color(QPalette.HighlightedText)
            outline = accent
        elif self._header_hovered or self.hasFocus():
            bg = QColor(accent)
            bg.setAlpha(34)
            fg = palette.color(QPalette.WindowText)
            outline = accent
        else:
            # AlternateBase is deliberately used instead of Base: the title
            # is a real toolbar-like strip, not text floating in the card.
            bg = palette.color(QPalette.AlternateBase)
            fg = palette.color(QPalette.WindowText)
            outline = palette.color(QPalette.Mid)
        fm = self.fontMetrics()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(QPen(outline, 1.0))
        painter.setBrush(bg)
        painter.drawRoundedRect(header, 6.0, 6.0)
        # A narrow accent rail makes the active/hovered state legible even on
        # low-contrast displays and gives the six groups a consistent visual
        # hierarchy at every application scale.
        rail = QColor(accent)
        if not (self._header_hovered or self._header_pressed or self.hasFocus()):
            rail.setAlpha(115)
        painter.setPen(Qt.NoPen)
        painter.setBrush(rail)
        painter.drawRoundedRect(
            QRectF(header.left() + 1.0, header.top() + 4.0, 3.0,
                   max(1.0, header.height() - 8.0)),
            1.5, 1.5,
        )
        size = max(5.0, min(8.0, fm.height() * 0.34))
        cx = header.left() + 15.0
        cy = header.center().y()
        if self._expanded:
            points = (
                QPointF(cx - size, cy - size * 0.45),
                QPointF(cx + size, cy - size * 0.45),
                QPointF(cx, cy + size * 0.65),
            )
        else:
            points = (
                QPointF(cx - size * 0.45, cy - size),
                QPointF(cx - size * 0.45, cy + size),
                QPointF(cx + size * 0.65, cy),
            )
        painter.setPen(Qt.NoPen)
        painter.setBrush(fg)
        painter.drawPolygon(QPolygonF(points))
        painter.setPen(fg)
        painter.setFont(self.font())
        text_rect = header.adjusted(28, 0, -9, 0)
        painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, self._base_title)
        if self.hasFocus() and not self._header_pressed:
            focus_rect = header.adjusted(2, 2, -2, -2)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(accent, 1.2, Qt.DashLine))
            painter.drawRoundedRect(focus_rect, 5.0, 5.0)

    def enterEvent(self, event):  # noqa: N802
        self._set_header_hovered(self._header_rect().contains(self.mapFromGlobal(
            self.cursor().pos()
        )))
        super().enterEvent(event)

    def leaveEvent(self, event):  # noqa: N802
        self._header_pressed = False
        self._set_header_hovered(False)
        super().leaveEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802
        self._set_header_hovered(self._header_rect().contains(event.position()))
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._header_rect().contains(event.position()):
            self._header_pressed = True
            self.setFocus(Qt.MouseFocusReason)
            self.update(self._header_rect().toAlignedRect())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if self._header_pressed:
            inside = self._header_rect().contains(event.position())
            self._header_pressed = False
            self.update(self._header_rect().toAlignedRect())
            if event.button() == Qt.LeftButton and inside:
                self.setExpanded(not self._expanded)
                event.accept()
                return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent):  # noqa: N802
        if event.key() in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
            self.setExpanded(not self._expanded)
            event.accept()
            return
        super().keyPressEvent(event)


class ControlPanel(QWidget):
    """参数控制面板 (左侧)."""

    changed = Signal()  # 任一参数变化 → Controller 重建
    displayChanged = Signal(str)  # 仅刷新呈现，不重建哈密顿量
    energyChanged = Signal(float)  # 波函数页能量选择 (不重建哈密顿量)
    recalculateRequested = Signal()
    cancelRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._param_names: list[str] = []
        self._cell_vectors: tuple[tuple[float, float], tuple[float, float]] | None = None
        # ``get_site_rows`` exposes a compact, contiguous model index, while
        # the editable table may temporarily contain blank rows.  Keep the
        # reverse mapping so canvas edits never write a compact index into a
        # different physical QTableWidget row.
        self._site_table_rows: list[int] = []
        self._updating_cell = False
        # Styling relation tooltips writes item data.  QTableWidget reports
        # those writes through cellChanged, so guard the presentation pass
        # and block table signals to prevent a recursive refresh storm.
        self._updating_hop_relation = False
        self._build_ui()
        self._connect()
        self._sync_boundary_controls()
        self._update_resource_hint()

    # ---- UI ----

    def _build_ui(self):
        lay = QVBoxLayout(self)

        # 边界模式
        g_bnd = CollapsibleGroupBox("边界")
        self.boundary_group = g_bnd
        b_lay = QVBoxLayout(g_bnd)
        self.boundary_combo = QComboBox()
        self.boundary_combo.addItems(["半无限 (x-Bloch)", "双开 (OBC)"])
        b_lay.addWidget(self.boundary_combo)
        # NX/NY each have a coarse slider and a precise integer editor.  Keep
        # the two rows separate: on narrow/high-DPI windows this prevents the
        # input boxes from being squeezed into a few pixels, while still
        # leaving the slider enough travel for comfortable mouse control.
        self.nx_spin, self.nx_slider = self._make_dimension_row(
            b_lay,
            "NX",
            "x 方向元胞数；双开边界生效，半无限模式下折叠为 1 个周期胞宽。",
            2,
        )
        self.ny_spin, self.ny_slider = self._make_dimension_row(
            b_lay,
            "NY",
            "y 方向元胞数；半无限和双开边界均生效。",
            2,
        )
        self.resource_hint = QLabel()
        self.resource_hint.setObjectName("panelHint")
        self.resource_hint.setWordWrap(True)
        self.resource_hint.setToolTip(
            "按当前格点数、NX/NY 和边界条件估算稠密 NumPy 后端的工作集。"
            "这是保守提示，不会限制模型保存；超出预算时计算前会给出可恢复错误。"
        )
        b_lay.addWidget(self.resource_hint)
        shape_row = QHBoxLayout()
        shape_row.addWidget(QLabel("盘形状"))
        self.shape_combo = QComboBox()
        for label, value in (
            ("矩形 / 方形盘", SHAPE_RECTANGLE),
            ("正三角形盘", SHAPE_TRIANGLE),
            ("圆盘", SHAPE_DISK),
            ("六边形盘", SHAPE_HEXAGON),
        ):
            self.shape_combo.addItem(label, value)
        self.shape_combo.setToolTip(
            "仅双开边界生效；非矩形盘按元胞真实物理尺度绘制，形状会同时影响有限矩阵、格点显示和边界跃迁。"
        )
        shape_row.addWidget(self.shape_combo, 1)
        b_lay.addLayout(shape_row)
        kx = QHBoxLayout()
        # Use the same mathematical glyph as the band axis and matrix
        # formulas.  A plain ``kx`` makes the x look like a peer character;
        # the Unicode subscript keeps this compact control readable without
        # introducing a heavyweight rich-text label.
        self.kx_label = QLabel("kₓ/π")
        self.kx_label.setToolTip("横向 Bloch 波矢 kₓ（以 π 为单位）")
        kx.addWidget(self.kx_label)
        self.kx_slider = QSlider(Qt.Horizontal)
        self.kx_slider.setRange(-100, 100)
        self.kx_slider.setValue(0)
        kx.addWidget(self.kx_slider)
        self.kx_edit = QLineEdit("0")
        self.kx_edit.setMaximumWidth(50)
        kx.addWidget(self.kx_edit)
        b_lay.addLayout(kx)
        lay.addWidget(g_bnd)

        # 参数 (由跃迁表达式自动生成: t/φ/ω/自定义名)
        g_params = CollapsibleGroupBox("参数（φ 以 π 为单位）")
        self.params_group = g_params
        p_lay = QVBoxLayout(g_params)
        self.param_table = QTableWidget(0, len(PARAM_COLS))
        self.param_table.setHorizontalHeaderLabels(PARAM_COLS)
        self.param_table.verticalHeader().setVisible(False)
        self.param_table.setColumnWidth(0, 70)
        self.param_table.setColumnWidth(1, 60)
        self.param_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.param_table.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        # Parameter names/values are compact scientific data, not prose.  Do
        # not replace them with an ellipsis when Qt paints a cell; the fit
        # helper below owns the width budget and keeps the complete value
        # readable at every UI scale.
        self.param_table.setTextElideMode(Qt.ElideNone)
        self.param_table.setAlternatingRowColors(True)
        p_lay.addWidget(self.param_table)
        hint = QLabel("φ 按 π 计（例如 1/4 = π/4），支持直接输入分数；其余参数为原值。")
        hint.setObjectName("panelHint")
        hint.setWordWrap(True)
        p_lay.addWidget(hint)
        lay.addWidget(g_params)

        # 波函数能量选择。它不是模型表达式参数，不参与哈密顿量重建；
        # 只有 OBC 且当前切到“波函数”页时才启用。
        self.energy_group = CollapsibleGroupBox("波函数（能量选择，仅波函数页生效）")
        e_lay = QHBoxLayout(self.energy_group)
        e_lay.addWidget(QLabel("能量 E"))
        self.energy_edit = QLineEdit("0")
        self.energy_edit.setToolTip("输入目标本征能量；程序会选择能量最接近的本征态。")
        self.energy_edit.setPlaceholderText("例如 0.25")
        e_lay.addWidget(self.energy_edit, 1)
        self.energy_group.setEnabled(False)
        lay.addWidget(self.energy_group)

        # 显示选项
        g_disp = CollapsibleGroupBox("显示")
        self.display_group = g_disp
        d_lay = QVBoxLayout(g_disp)
        self.symbolic_check = QCheckBox("符号模式")
        self.smart_check = QCheckBox("智能识别标签")
        self.smart_check.setChecked(True)
        d_lay.addWidget(self.symbolic_check)
        d_lay.addWidget(self.smart_check)
        bond_row = QHBoxLayout()
        self.show_nn_check = QCheckBox("最近邻键")
        self.show_nnn_check = QCheckBox("次近邻 / 长程键")
        self.show_nn_check.setChecked(True)
        self.show_nnn_check.setChecked(True)
        self.show_nnn_check.setToolTip(
            "绘制第二及更远距离壳层的跃迁；普通浏览态已降低对比度，"
            "关闭后只保留最近邻骨架。"
        )
        bond_row.addWidget(self.show_nn_check)
        bond_row.addWidget(self.show_nnn_check)
        bond_row.addStretch()
        d_lay.addLayout(bond_row)
        structure_row = QHBoxLayout()
        self.show_ghosts_check = QCheckBox("周期虚影")
        self.show_cells_check = QCheckBox("元胞轮廓")
        self.show_ghosts_check.setChecked(True)
        self.show_cells_check.setChecked(True)
        self.show_ghosts_check.setToolTip(
            "半无限模式下显示左右相邻周期像及跨元胞键。"
        )
        structure_row.addWidget(self.show_ghosts_check)
        structure_row.addWidget(self.show_cells_check)
        structure_row.addStretch()
        d_lay.addLayout(structure_row)
        ord_row = QHBoxLayout()
        ord_row.addWidget(QLabel("排序"))
        self.order_combo = QComboBox()
        self.order_combo.addItems(["胞优先", "列优先"])
        ord_row.addWidget(self.order_combo)
        ord_row.addStretch()
        d_lay.addLayout(ord_row)
        label_row = QHBoxLayout()
        label_row.addWidget(QLabel("元胞编号"))
        self.cell_number_combo = QComboBox()
        self.cell_number_combo.addItems(["自下而上", "自上而下"])
        self.cell_number_combo.setToolTip("半无限晶格的元胞编号方向；左右虚影始终复用中间编号。")
        label_row.addWidget(self.cell_number_combo)
        label_row.addStretch()
        d_lay.addLayout(label_row)
        lay.addWidget(g_disp)

        # 格点表
        g_sites = CollapsibleGroupBox("格点（精确编辑 / 检查）")
        self.sites_group = g_sites
        s_lay = QVBoxLayout(g_sites)
        cell_row = QHBoxLayout()
        # Keep the visible labels short enough for a narrow/high-DPI rail;
        # the tooltip carries the precise meaning (distance between adjacent
        # cells along the corresponding lattice vector).
        cell_row.addWidget(QLabel("a₁ 长度"))
        self.lx_spin = QDoubleSpinBox()
        self.lx_spin.setRange(0.0, 1000.0)
        # Preserve irrational preset cell lengths (e.g. 2√3 in Kagome).
        # Six decimals moved boundary samples across the triangle half-plane
        # at exactly the wrong side of a lattice point, changing a 6→4→2
        # mask into a visibly broken 6→2 mask.  The value remains compact in
        # the field but the model keeps enough precision for geometry tests.
        # Eight decimals keep the displayed value within the narrow two-column
        # rail (the site table itself also uses eight decimals), while still
        # resolving the irrational presets far below the geometry tolerance.
        # This is an intentional input/display precision choice: the geometry
        # model itself continues to carry the vector values independently.
        self.lx_spin.setDecimals(8)
        self.lx_spin.setMaximumWidth(140)
        self.lx_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.lx_spin.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.lx_spin.setSpecialValueText("自动")
        self.lx_spin.setToolTip(
            "调整相邻元胞沿 a₁ 的间距；斜原胞会保持方向，仅缩放长度。"
            "0 表示根据格点坐标自动推断。"
        )
        cell_row.addWidget(self.lx_spin, 1)
        cell_row.addWidget(QLabel("a₂ 长度"))
        self.ly_spin = QDoubleSpinBox()
        self.ly_spin.setRange(0.0, 1000.0)
        self.ly_spin.setDecimals(8)
        self.ly_spin.setMaximumWidth(140)
        self.ly_spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.ly_spin.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.ly_spin.setSpecialValueText("自动")
        self.ly_spin.setToolTip(
            "调整相邻元胞沿 a₂ 的间距；斜原胞会保持方向，仅缩放长度。"
        )
        cell_row.addWidget(self.ly_spin, 1)
        s_lay.addLayout(cell_row)
        vector_row1 = QHBoxLayout()
        vector_row1.addWidget(QLabel("a₁ ="))
        self.a1x_spin = QDoubleSpinBox()
        self.a1y_spin = QDoubleSpinBox()
        vector_row1.addWidget(self.a1x_spin, 1)
        vector_row1.addWidget(self.a1y_spin, 1)
        vector_row2 = QHBoxLayout()
        vector_row2.addWidget(QLabel("a₂ ="))
        self.a2x_spin = QDoubleSpinBox()
        self.a2y_spin = QDoubleSpinBox()
        vector_row2.addWidget(self.a2x_spin, 1)
        vector_row2.addWidget(self.a2y_spin, 1)
        vector_btn_row = QHBoxLayout()
        vector_btn_row.addStretch()
        # Length fields are magnitudes, never signed vector components.  The
        # previous shared range made it possible to type a negative ``a₁/a₂``
        # length; ``get_cell_size`` then interpreted two negatives as
        # ``自动`` and the next rebuild failed much later in the Hamiltonian
        # layer.  Keep signs available only on the four vector components.
        for spin in (self.lx_spin, self.ly_spin):
            spin.setRange(0.0, 1000.0)
        for spin in (self.a1x_spin, self.a1y_spin, self.a2x_spin, self.a2y_spin):
            spin.setRange(-1000.0, 1000.0)
            spin.setDecimals(8)
            spin.setSingleStep(0.05)
            # These fields carry long coordinates.  Native up/down buttons
            # consume a fixed strip and can cover the last digit at 150–180%
            # UI scale; wheel and keyboard stepping remain available without
            # that visual collision.
            spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
            spin.setMinimumWidth(0)
            spin.setMaximumWidth(180)
            # QDoubleSpinBox's native size hint reserves room for a very
            # long 8-decimal value (300+ px).  Treat the field as shrinkable
            # so the editor rail follows the viewport at high UI scales;
            # the horizontal input remains scrollable when the value itself
            # is long, while the full border stays inside the panel.
            spin.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.apply_vectors_btn = QPushButton("应用元胞矢量")
        self.apply_vectors_btn.setToolTip(
            "精确设置相邻元胞的两个平移矢量；可编辑斜原胞，不会再退化为矩形。"
        )
        vector_btn_row.addWidget(self.apply_vectors_btn)
        s_lay.addLayout(vector_row1)
        s_lay.addLayout(vector_row2)
        s_lay.addLayout(vector_btn_row)
        self.site_table = QTableWidget(0, len(SITE_COLS))
        self.site_table.setHorizontalHeaderLabels(SITE_COLS)
        self.site_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.site_table.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        self.site_table.setAlternatingRowColors(True)
        self.site_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.site_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.site_table.setToolTip(
            "格点坐标在窄窗口中可紧凑显示；右键单元格可复制未省略的完整值。"
        )
        s_lay.addWidget(self.site_table)
        btn = QHBoxLayout()
        self.add_site_btn = QPushButton("+ 格点")
        self.del_site_btn = QPushButton("− 格点")
        btn.addWidget(self.add_site_btn)
        btn.addWidget(self.del_site_btn)
        btn.addStretch()
        s_lay.addLayout(btn)
        lay.addWidget(g_sites)

        # 跃迁表
        g_hops = CollapsibleGroupBox("跃迁项（精确编辑 / 检查）")
        self.hops_group = g_hops
        h_lay = QVBoxLayout(g_hops)
        self.hop_table = QTableWidget(0, len(HOP_COLS))
        self.hop_table.setHorizontalHeaderLabels(HOP_HEADERS)
        self.hop_table.setAlternatingRowColors(True)
        self.hop_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.hop_table.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        self.hop_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.hop_table.setToolTip(
            "每行是一条 Hermitian 物理键；反向共轭由程序自动补全。"
            "dx=0 且 dy=0 为胞内；任一非零为胞间。phase_mode 仅允许 none / phase。"
        )
        self.hop_relation_hint = QLabel()
        self.hop_relation_hint.setWordWrap(True)
        self.hop_relation_hint.setObjectName("dialogNote")
        self.hop_relation_hint.setToolTip(
            "dx=0 且 dy=0 表示胞内跃迁；任一偏移非零表示胞间跃迁。"
        )
        h_lay.addWidget(self.hop_relation_hint)
        h_lay.addWidget(self.hop_table)
        hop_view = QHBoxLayout()
        self.hop_advanced_check = QCheckBox("显示高级列（相位 / 相位模式 / 符号）")
        self.hop_advanced_check.setToolTip(
            "dx/dy 是每条跃迁的核心关系，在紧凑模式也始终可见；"
            "需要编辑相位、相位模式或符号方向时再展开高级列。"
        )
        hop_view.addWidget(self.hop_advanced_check)
        hop_view.addStretch()
        h_lay.addLayout(hop_view)
        btn2 = QHBoxLayout()
        # The legacy one-click action remains a deliberately safe intra-cell
        # starter.  Say that explicitly in the label: otherwise a user in the
        # semi-infinite editor can reasonably assume it creates a Bloch bond.
        # Keep the action labels short enough to remain fully clickable at the
        # user's 150–180% UI scale.  The tooltips and relation summary carry
        # the complete semantics; the buttons themselves should never turn
        # into an ellipsis or a clipped hit target.
        self.add_hop_btn = QPushButton("+ 胞内")
        self.add_inter_hop_btn = QPushButton("+ 胞间")
        self.add_inter_hop_btn.setToolTip(
            "新增一条右侧相邻元胞的跃迁（dx=+1, dy=0）。"
            "半无限模式下它进入 x-Bloch 的 H₁；其他方向请用“其他关系…”。"
        )
        self.del_hop_btn = QPushButton("− 跃迁")
        # Keep the one-click legacy action, but expose the physically
        # important cell relation beside it.  This avoids forcing users to
        # discover raw dx/dy columns just to create a Bloch/inter-cell bond.
        self.add_hop_mode_btn = QToolButton()
        # Leave the menu indicator to the native QToolButton style instead of
        # embedding a Unicode triangle that some CJK fonts render as tofu.
        # A compact label keeps the popup reachable at high UI scales; the
        # menu entries themselves spell out each physical relation.
        self.add_hop_mode_btn.setText("其他关系…")
        self.add_hop_mode_btn.setPopupMode(QToolButton.InstantPopup)
        self.add_hop_mode_btn.setToolTip(
            "选择新增跃迁位于胞内还是哪个相邻元胞；表格仍可精确修改 dx/dy。"
        )
        hop_menu = QMenu(self.add_hop_mode_btn)
        for label, offset in (
            ("胞内（dx=0, dy=0）", (0, 0)),
            ("右侧胞间（dx=+1, dy=0）", (1, 0)),
            ("左侧胞间（dx=-1, dy=0）", (-1, 0)),
            ("上方胞间（dx=0, dy=+1）", (0, 1)),
            ("下方胞间（dx=0, dy=-1）", (0, -1)),
        ):
            action = hop_menu.addAction(label)
            action.triggered.connect(
                lambda _checked=False, dx=offset[0], dy=offset[1]:
                self._add_hop_with_offset(dx, dy)
            )
        hop_menu.addSeparator()
        custom_action = hop_menu.addAction("自定义胞间偏移…")
        custom_action.setToolTip(
            "打开对话框，可输入任意整数 dx/dy、幅度与相位；适合长程 Bloch 谐波。"
        )
        custom_action.triggered.connect(self._add_custom_hop_dialog)
        self.add_hop_mode_btn.setMenu(hop_menu)
        btn2.addWidget(self.add_hop_btn)
        btn2.addWidget(self.add_inter_hop_btn)
        btn2.addWidget(self.add_hop_mode_btn)
        btn2.addWidget(self.del_hop_btn)
        btn2.addStretch()
        h_lay.addLayout(btn2)
        lay.addWidget(g_hops)

        # 计算控制：自动模式下“立即计算”可跳过防抖；手动模式下它是
        # 唯一触发入口。取消会使后台结果失效，不阻塞界面。
        calc_row = QHBoxLayout()
        self.refresh_btn = QPushButton("立即计算")
        self.cancel_btn = QPushButton("取消计算")
        self.cancel_btn.setEnabled(False)
        calc_row.addWidget(self.refresh_btn, 1)
        calc_row.addWidget(self.cancel_btn)
        lay.addLayout(calc_row)

        # 错误标签 (构建失败提示)
        self.error_label = QLabel("")
        self.error_label.setWordWrap(True)
        self.error_label.setObjectName("errorLabel")
        self.error_label.setVisible(False)
        lay.addWidget(self.error_label)

        for group, expanded in (
            (self.boundary_group, True),
            (self.params_group, True),
            (self.energy_group, True),
            (self.display_group, False),
            (self.sites_group, False),
            (self.hops_group, False),
        ):
            self._make_collapsible(group, expanded)

    @staticmethod
    def _make_dimension_row(
        parent_layout: QVBoxLayout,
        label: str,
        tooltip: str,
        value: int,
    ) -> tuple[QSpinBox, QSlider]:
        """Create one dimension's slider + exact input pair.

        The spin box deliberately has a very large integer range.  A slider
        is excellent for exploring the common 1–64 range, but it is not a
        suitable representation of an arbitrary study size; its upper bound
        is therefore extended on demand by the sync handlers below.
        """
        row = QHBoxLayout()
        title = QLabel(label)
        title.setMinimumWidth(26)
        row.addWidget(title)
        spin = QSpinBox()
        spin.setRange(1, DIM_INPUT_MAX)
        spin.setValue(value)
        spin.setMinimumWidth(62)
        spin.setToolTip(tooltip)
        row.addWidget(spin)
        slider = QSlider(Qt.Horizontal)
        slider.setRange(1, DIM_SLIDER_DEFAULT_MAX)
        slider.setValue(value)
        slider.setPageStep(1)
        slider.setSingleStep(1)
        slider.setToolTip(tooltip)
        slider.setMinimumWidth(90)
        slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row.addWidget(slider, 1)
        parent_layout.addLayout(row)
        return spin, slider

    def _connect(self):
        self.boundary_combo.currentIndexChanged.connect(self._boundary_changed)
        self.order_combo.currentIndexChanged.connect(lambda *_: self._emit_changed())
        self.cell_number_combo.currentIndexChanged.connect(
            lambda *_: self.displayChanged.emit("labels")
        )
        self.nx_spin.valueChanged.connect(self._nx_from_spin)
        self.nx_slider.valueChanged.connect(self._nx_from_slider)
        self.ny_spin.valueChanged.connect(self._ny_from_spin)
        self.ny_slider.valueChanged.connect(self._ny_from_slider)
        self.shape_combo.currentIndexChanged.connect(lambda *_: self._emit_changed())
        self.symbolic_check.toggled.connect(lambda *_: self._emit_changed())
        self.smart_check.toggled.connect(lambda *_: self.displayChanged.emit("smart"))
        for checkbox in (
            self.show_nn_check, self.show_nnn_check,
            self.show_ghosts_check, self.show_cells_check,
        ):
            checkbox.toggled.connect(
                lambda *_: self.displayChanged.emit("lattice_style")
            )
        self.kx_slider.valueChanged.connect(self._kx_changed)
        self.kx_edit.editingFinished.connect(self._kx_from_edit)
        self.site_table.cellChanged.connect(self._site_cell_changed)
        self.site_table.customContextMenuRequested.connect(
            self._show_site_context_menu
        )
        self.lx_spin.valueChanged.connect(self._cell_size_changed)
        self.ly_spin.valueChanged.connect(self._cell_size_changed)
        self.apply_vectors_btn.clicked.connect(self._apply_cell_vectors)
        self.hop_table.cellChanged.connect(self._hop_cell_changed)
        self.hop_table.customContextMenuRequested.connect(
            self._show_hop_context_menu
        )
        self.hop_advanced_check.toggled.connect(self._toggle_hop_columns)
        self.param_table.cellChanged.connect(self._param_cell_changed)
        self.energy_edit.editingFinished.connect(self._energy_changed)
        self.add_site_btn.clicked.connect(lambda: self._add_row(self.site_table, [0.0, 0.0, "A"]))
        self.del_site_btn.clicked.connect(lambda: self._del_row(self.site_table))
        self.add_hop_btn.setToolTip(
            "新增一条同一元胞内的实跃迁（dx=0, dy=0）。"
            "需要半无限 Bloch/胞间连接时，请使用“+ 胞间”或“其他关系…”，"
            "也可以直接编辑 dx/dy。"
        )
        self.add_hop_btn.clicked.connect(self._add_default_hop)
        self.add_inter_hop_btn.clicked.connect(
            lambda: self._add_hop_with_offset(1, 0)
        )
        self.del_hop_btn.clicked.connect(lambda: self._del_row(self.hop_table))
        self.refresh_btn.clicked.connect(self.recalculateRequested)
        self.cancel_btn.clicked.connect(self.cancelRequested)
        self._toggle_hop_columns(False)
        self._update_hop_relation_hint()

    @staticmethod
    def _make_collapsible(group: QGroupBox, expanded: bool):
        if isinstance(group, CollapsibleGroupBox):
            group.setExpanded(expanded)

    def set_calculating(self, active: bool):
        self.cancel_btn.setEnabled(bool(active))
        self.refresh_btn.setText("重新计算" if active else "立即计算")

    def _cell_size_changed(self, *_args):
        if self._updating_cell:
            return
        if self._cell_vectors is not None:
            (a1x, a1y), (a2x, a2y) = self._cell_vectors
            old1, old2 = math.hypot(a1x, a1y), math.hypot(a2x, a2y)
            new1, new2 = self.lx_spin.value(), self.ly_spin.value()
            if min(old1, old2, new1, new2) <= 1e-12:
                # In oblique-vector mode, zero is not a meaningful vector
                # length.  Previously the spin box stayed at the special
                # “自动” value while ``_cell_vectors`` silently retained the
                # old vectors, so the value shown to the user disagreed with
                # the Hamiltonian geometry.  Restore the last valid lengths
                # atomically and explain the distinction.
                blocked1 = self.lx_spin.blockSignals(True)
                blocked2 = self.ly_spin.blockSignals(True)
                try:
                    self.lx_spin.setValue(old1)
                    self.ly_spin.setValue(old2)
                finally:
                    self.lx_spin.blockSignals(blocked1)
                    self.ly_spin.blockSignals(blocked2)
                self.set_error("斜原胞的 a₁/a₂ 长度必须为正；“自动”仅适用于矩形元胞")
                return
            self._cell_vectors = (
                (a1x * new1 / old1, a1y * new1 / old1),
                (a2x * new2 / old2, a2y * new2 / old2),
            )
            self._sync_vector_spins()
            self.set_error("")
        else:
            self._sync_vector_spins(rectangular=True)
        self._emit_changed()

    def _apply_cell_vectors(self):
        a1 = (self.a1x_spin.value(), self.a1y_spin.value())
        a2 = (self.a2x_spin.value(), self.a2y_spin.value())
        det = a1[0] * a2[1] - a1[1] * a2[0]
        if abs(det) < 1e-10:
            self.set_error("a₁ 与 a₂ 不可共线；请调整元胞矢量")
            return
        if math.hypot(*a1) <= 1e-12 or abs(a2[1]) <= 1e-12:
            # ``Lattice`` uses |a₂.y| as the finite-direction period.  Catch
            # this at the point of entry so an invalid vector never reaches
            # a rebuild and leaves the user with a generic scene error.
            self.set_error("a₁ 必须非零，且 a₂ 的 y 分量必须非零")
            return
        if not all(math.isfinite(value) for value in (*a1, *a2)):
            self.set_error("元胞矢量必须是有限数值")
            return
        self._cell_vectors = (a1, a2)
        self._updating_cell = True
        try:
            self.lx_spin.setValue(math.hypot(*a1))
            self.ly_spin.setValue(math.hypot(*a2))
        finally:
            self._updating_cell = False
        self.set_error("")
        self._emit_changed()

    def _sync_vector_spins(self, *, rectangular: bool = False):
        if rectangular or self._cell_vectors is None:
            vectors = ((self.lx_spin.value(), 0.0), (0.0, self.ly_spin.value()))
        else:
            vectors = self._cell_vectors
        for spin, value in zip(
            (self.a1x_spin, self.a1y_spin, self.a2x_spin, self.a2y_spin),
            (*vectors[0], *vectors[1]),
        ):
            spin.blockSignals(True)
            spin.setValue(float(value))
            spin.blockSignals(False)

    # ---- 错误提示 ----

    def set_error(self, msg: str):
        self.error_label.setText(msg)
        self.error_label.setVisible(bool(msg))

    # ---- 参数表 ----

    @staticmethod
    def _slider_cfg(name: str) -> tuple:
        return PARAM_SLIDERS.get(name, (-1000, 1000, 0.01))

    @staticmethod
    def _display_scale(name: str, scale: float) -> float:
        """数值列显示缩放: φ 显示 φ/π, 其余原值."""
        return scale / math.pi if name == "phi" else scale

    @staticmethod
    def _actual_scale(name: str) -> float:
        """数值列读数 → 实际值的乘数 (φ 乘 π)."""
        return math.pi if name == "phi" else 1.0

    def set_params(self, values: dict, force: bool = False):
        """用 {name: 实际值} 填充参数表.

        同名集合未变时默认跳过 (保留用户已编辑的数值); force=True 强制重填
        (预设加载 / 模型打开时使用)。
        """
        names = sorted(values)
        if not force and names == self._param_names:
            return
        self._param_names = names
        tbl = self.param_table
        tbl.blockSignals(True)
        # ``QTableWidget.setRowCount`` removes table cells but does not
        # reliably dispose embedded widgets on every Qt/platform path.  The
        # old parameter sliders can therefore remain parented and visible
        # after switching from a multi-parameter model (NP/SC/Haldane) to a
        # compact one (Kagome/SSH), painting blue grooves over the current
        # rows—especially obvious at 180% UI scale.  Detach every old slider
        # before changing the row count so a model switch leaves exactly one
        # live control per current parameter.
        for row in range(tbl.rowCount()):
            old_slider = tbl.cellWidget(row, 2)
            if old_slider is not None:
                tbl.removeCellWidget(row, 2)
                old_slider.hide()
                old_slider.setParent(None)
                old_slider.deleteLater()
        tbl.setRowCount(len(names))
        for r, name in enumerate(names):
            v = float(values.get(name, 1.0))
            if not math.isfinite(v):
                raise ValueError(f"参数 {name} 必须是有限数值")
            imin, imax, scale = self._slider_cfg(name)
            dscale = self._display_scale(name, scale)
            vint = int(round(v / scale))
            # 文本值是真实来源；滑块范围按需扩展，绝不夹断模型参数。
            imin = min(imin, vint)
            imax = max(imax, vint)
            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            if name == "phi":
                name_item.setToolTip("φ（以 π 为单位；输入 1/4 即 π/4）")
            tbl.setItem(r, 0, name_item)
            displayed = v / self._actual_scale(name)
            tbl.setItem(r, 1, QTableWidgetItem(f"{displayed:.12g}"))
            sl = QSlider(Qt.Horizontal)
            sl.setRange(imin, imax)
            sl.setValue(vint)
            tbl.setCellWidget(r, 2, sl)
            sl.valueChanged.connect(lambda val, row=r: self._param_slider_changed(row, val))
        tbl.blockSignals(False)
        self._fit_param_table_height()
        self._fit_param_table_columns()

    def _fit_param_table_height(self) -> None:
        """Keep ordinary parameter sets visible without a nested scrollbar.

        The control rail already has one deliberate vertical scrollbar.  A
        second scrollbar inside a four-row parameter table is hard to notice
        at 150--180% UI scale and hides the last symbol (for example Haldane's
        ``t2``).  Grow the table for small/normal parameter sets so the rail
        owns scrolling; retain an internal scrollbar only for unusually large
        user-defined symbol sets.
        """
        table = self.param_table
        rows = table.rowCount()
        if rows <= 8:
            header = max(table.horizontalHeader().sizeHint().height(), 1)
            row_height = max(table.verticalHeader().defaultSectionSize(), 1)
            frame = 2 * max(table.frameWidth(), 1)
            table.setMinimumHeight(header + rows * row_height + frame + 2)
            table.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        else:
            table.setMinimumHeight(0)
            table.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

    def _fit_param_table_columns(self) -> None:
        """Keep long parameter values readable without creating side scrolling.

        The value column starts at a compact fixed width so the slider keeps
        most of the rail.  A fraction or a high-precision value can be wider
        than that default, though; Qt then elides it as ``0....`` even when
        the panel still has spare room.  Grow the name/value columns from
        their actual font metrics and reserve a modest minimum for the
        slider.  If a user-defined symbol is genuinely too long, shrink the
        name column first but never let the table exceed its viewport.
        """
        table = self.param_table
        rows = table.rowCount()
        if rows <= 0:
            return
        metrics = QFontMetrics(table.font())
        # The stylesheet contributes scaled left/right item padding (5 px at
        # 100%, 9 px at 180%) in addition to the delegate's frame.  Reserve
        # one full line-height as a conservative budget; using only half a
        # line caused the final character of a 12-digit value to be elided at
        # 180% even though the raw font metric appeared to fit.
        horizontal_pad = max(24, metrics.height())

        def needed(column: int, header: str) -> int:
            texts = [header]
            for row in range(rows):
                item = table.item(row, column)
                if item is not None:
                    texts.append(item.text())
            return max(metrics.horizontalAdvance(text) for text in texts) + horizontal_pad

        name_needed = max(70, needed(0, PARAM_COLS[0]))
        value_needed = max(60, needed(1, PARAM_COLS[1]))
        current_name = table.columnWidth(0)
        current_value = table.columnWidth(1)
        target_name = max(current_name, name_needed)
        target_value = max(current_value, value_needed)

        viewport_width = table.viewport().width()
        if viewport_width > 0:
            slider_min = max(100, metrics.horizontalAdvance(PARAM_COLS[2]) + horizontal_pad)
            fixed_budget = max(0, viewport_width - slider_min)
            overflow = target_name + target_value - fixed_budget
            if overflow > 0:
                # Preserve enough room for a normal symbol name first; long
                # numeric values get priority because they are the editable
                # data users most often need to inspect exactly.
                reducible_name = max(0, target_name - 70)
                reduce_name = min(reducible_name, overflow)
                target_name -= reduce_name
                overflow -= reduce_name
                if overflow > 0:
                    target_value = max(60, target_value - overflow)

        table.setColumnWidth(0, int(target_name))
        table.setColumnWidth(1, int(target_value))

    def get_params(self) -> dict:
        """读参数表 → {name: 实际值} (φ 换算回弧度)."""
        out: dict = {}
        tbl = self.param_table
        for r in range(tbl.rowCount()):
            name_item = tbl.item(r, 0)
            val_item = tbl.item(r, 1)
            if name_item is None or val_item is None:
                raise ValueError(f"参数表第 {r + 1} 行不完整")
            try:
                v = self._parse_scalar(val_item.text())
            except ValueError:
                raise ValueError(
                    f"参数表第 {r + 1} 行“数值”不是有效数字或分数"
                ) from None
            if not math.isfinite(v):
                raise ValueError(f"参数表第 {r + 1} 行“数值”必须有限")
            out[name_item.text()] = v * self._actual_scale(name_item.text())
        return out

    def _param_slider_changed(self, row: int, val: int):
        tbl = self.param_table
        name = tbl.item(row, 0).text()
        _imin, _imax, scale = self._slider_cfg(name)
        dscale = self._display_scale(name, scale)
        tbl.blockSignals(True)
        tbl.item(row, 1).setText(f"{val * dscale:.4g}")
        tbl.blockSignals(False)
        # Slider edits also replace the text cell.  Reflow immediately so a
        # larger display scale never leaves the freshly formatted value
        # elided behind a stale column width.
        self._fit_param_table_columns()
        self._emit_changed()

    def _param_cell_changed(self, row: int, col: int):
        if col != 1:
            return
        tbl = self.param_table
        # ``cellChanged`` is emitted for every keystroke, including a
        # temporarily invalid intermediate value while the user types a
        # fraction.  Resize before parsing so both valid and intermediate
        # text remain visible and the editor never appears to jump/crop.
        self._fit_param_table_columns()
        name = tbl.item(row, 0).text()
        try:
            v = self._parse_scalar(tbl.item(row, 1).text())
        except ValueError:
            self._emit_changed()
            return
        if not math.isfinite(v):
            self._emit_changed()
            return
        # 分数输入接受后立即规范化为小数，用户无需再次手动转换；
        # phi 这一列本来就是 φ/π 单位，因此这里直接显示 v。
        normalized = f"{v:.12g}"
        if tbl.item(row, 1).text().strip() != normalized:
            tbl.blockSignals(True)
            tbl.item(row, 1).setText(normalized)
            tbl.blockSignals(False)
            self._fit_param_table_columns()
        _imin, _imax, scale = self._slider_cfg(name)
        actual = v * self._actual_scale(name)
        vint = int(round(actual / scale))
        sl = tbl.cellWidget(row, 2)
        if isinstance(sl, QSlider):
            sl.blockSignals(True)
            if vint < sl.minimum():
                sl.setMinimum(vint)
            if vint > sl.maximum():
                sl.setMaximum(vint)
            sl.setValue(vint)
            sl.blockSignals(False)
        self._fit_param_table_columns()
        self._emit_changed()

    # ---- 输入读取 (Controller 消费) ----

    @staticmethod
    def _parse_scalar(text: str) -> float:
        """解析普通小数或安全的有理数输入（如 ``1/13``）。

        参数表不执行任意 Python/数学表达式，避免把编辑框变成代码执行
        入口；分数只交给 :class:`fractions.Fraction` 解析。
        """
        raw = str(text).strip()
        if not raw:
            raise ValueError
        try:
            return float(raw)
        except ValueError:
            try:
                # Fraction 对带空格的 ``1 / 13`` 不同 Python 版本行为不一，
                # 这里仅去除空白，仍不允许其它运算符。
                value = float(Fraction(raw.replace(" ", "")))
            except (ValueError, ZeroDivisionError, OverflowError):
                raise ValueError from None
            return value

    def get_energy(self) -> float:
        try:
            value = self._parse_scalar(self.energy_edit.text())
        except ValueError:
            raise ValueError("波函数能量 E 不是有效数字或分数") from None
        if not math.isfinite(value):
            raise ValueError("波函数能量 E 必须是有限数值")
        return value

    def set_energy(self, value: float):
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("波函数能量 E 必须是有限数值")
        self.energy_edit.blockSignals(True)
        self.energy_edit.setText(f"{value:.12g}")
        self.energy_edit.blockSignals(False)
        self.energy_edit.setStyleSheet("")

    def set_wavefunction_active(self, active: bool):
        """只在 OBC 的波函数标签页启用能量选择。"""
        self.energy_group.setEnabled(bool(active))

    def _energy_changed(self):
        try:
            value = self.get_energy()
        except ValueError:
            self.energy_edit.setStyleSheet("border:1px solid #b00020;")
            return
        self.energy_edit.setStyleSheet("")
        self.energyChanged.emit(value)

    def is_semi(self) -> bool:
        return self.boundary_combo.currentIndex() == 0

    def _sync_boundary_controls(self):
        semi = self.is_semi()
        self.nx_spin.setEnabled(not semi)
        self.nx_slider.setEnabled(not semi)
        self.ny_spin.setEnabled(True)
        self.ny_slider.setEnabled(True)
        self.shape_combo.setEnabled(not semi)
        self.kx_slider.setEnabled(semi)
        self.kx_edit.setEnabled(semi)

    @staticmethod
    def _format_resource_bytes(value: int) -> str:
        """Format a byte count compactly for the narrow parameter rail."""
        value = float(max(0, int(value)))
        for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
            if value < 1024.0 or unit == "TiB":
                if unit == "B":
                    return f"{value:.0f} {unit}"
                return f"{value:.1f} {unit}"
            value /= 1024.0
        return f"{value:.1f} TiB"

    def _update_resource_hint(self):
        """Show a conservative dense-backend estimate before the user computes.

        The estimate is deliberately a hint, not an input validator: large
        studies remain editable and saveable for a future sparse backend.  OBC
        uses the full NX×NY upper bound because a non-rectangular mask is only
        known after geometry construction; SEMI uses its folded one-column
        ribbon (NY cells).
        """
        label = getattr(self, "resource_hint", None)
        if label is None:
            return
        try:
            nx, ny = self.get_dim()
            # Blank rows are a table-editing convenience, not physical basis
            # sites.  Counting them here made the preflight estimate look
            # larger than the model the user would actually calculate.
            basis_sites = max(1, len(self._refresh_site_table_rows()))
            semi = self.is_semi()
            cells = ny if semi else nx * ny
            nat = basis_sites * cells
            kind = BoundaryKind.SEMI if semi else BoundaryKind.OBC
            estimate = estimate_dense_working_set_bytes(nat, kind)
            amount = self._format_resource_bytes(estimate)
            limit = self._format_resource_bytes(DENSE_WORKING_SET_LIMIT_BYTES)
            mode = "半无限" if semi else "双开上界"
            if estimate > DENSE_WORKING_SET_LIMIT_BYTES:
                label.setText(
                    f"稠密预估：{mode} {nat}×{nat}，约 {amount}；"
                    f"超过安全预算 {limit}。仍可保存，计算前会提示。"
                )
            else:
                label.setText(
                    f"稠密预估：{mode} {nat}×{nat}，约 {amount}（安全预算 {limit}）"
                )
        except (AttributeError, TypeError, ValueError, OverflowError):
            # During early widget construction tables/spins may not exist yet.
            label.setText("稠密资源预估将在模型参数就绪后显示")

    def _boundary_changed(self, *_):
        self._sync_boundary_controls()
        self._update_resource_hint()
        self._update_hop_relation_hint()
        self._emit_changed()

    def set_boundary_index(self, idx: int):
        self.boundary_combo.setCurrentIndex(idx)

    def get_shape(self) -> str:
        return str(self.shape_combo.currentData() or SHAPE_RECTANGLE)

    def set_shape(self, shape: str):
        index = self.shape_combo.findData(shape)
        self.shape_combo.setCurrentIndex(index if index >= 0 else 0)

    def get_dim(self) -> tuple[int, int]:
        return self.nx_spin.value(), self.ny_spin.value()

    def set_dim(self, nx: int, ny: int):
        values = ((self.nx_spin, self.nx_slider, nx),
                  (self.ny_spin, self.ny_slider, ny))
        for spin, slider, raw in values:
            value = int(raw)
            if not 1 <= value <= DIM_INPUT_MAX:
                raise ValueError(f"尺寸必须在 1～{DIM_INPUT_MAX} 之间")
            spin.blockSignals(True)
            slider.blockSignals(True)
            spin.setValue(value)
            self._set_dimension_slider(slider, value)
            slider.blockSignals(False)
            spin.blockSignals(False)
        self._update_resource_hint()

    @staticmethod
    def _set_dimension_slider(slider: QSlider, value: int):
        """Keep a slider useful for common values while accepting larger ones."""
        value = int(value)
        if value > slider.maximum():
            slider.setMaximum(value)
        slider.setValue(value)

    def _nx_from_spin(self, value: int):
        self.nx_slider.blockSignals(True)
        self._set_dimension_slider(self.nx_slider, value)
        self.nx_slider.blockSignals(False)
        self._emit_changed()

    def _nx_from_slider(self, value: int):
        self.nx_spin.blockSignals(True)
        self.nx_spin.setValue(int(value))
        self.nx_spin.blockSignals(False)
        self._emit_changed()

    def _ny_from_spin(self, value: int):
        self.ny_slider.blockSignals(True)
        self._set_dimension_slider(self.ny_slider, value)
        self.ny_slider.blockSignals(False)
        self._emit_changed()

    def _ny_from_slider(self, value: int):
        self.ny_spin.blockSignals(True)
        self.ny_spin.setValue(int(value))
        self.ny_spin.blockSignals(False)
        self._emit_changed()

    def get_kx(self) -> float:
        try:
            value = float(self.kx_edit.text())
        except ValueError:
            raise ValueError("kₓ/π 不是有效数字") from None
        if not math.isfinite(value):
            raise ValueError("kₓ/π 必须是有限数值")
        return value * math.pi

    def set_kx(self, v: float):
        self.kx_slider.blockSignals(True)
        self.kx_slider.setValue(int(round(v / 3.141592653589793 * 100)))
        self.kx_slider.blockSignals(False)
        self.kx_edit.setText(f"{v / 3.141592653589793:.3f}")

    def is_symbolic(self) -> bool:
        return self.symbolic_check.isChecked()

    def set_symbolic(self, v: bool):
        self.symbolic_check.setChecked(v)

    def is_smart(self) -> bool:
        return self.smart_check.isChecked()

    def set_smart(self, v: bool):
        self.smart_check.setChecked(v)

    def get_order(self) -> str:
        return "cell" if self.order_combo.currentIndex() == 0 else "site"

    def set_order(self, order: str):
        if order not in {"cell", "site"}:
            raise ValueError(f"未知排序方式: {order!r}")
        self.order_combo.setCurrentIndex(0 if order == "cell" else 1)

    def labels_bottom_up(self) -> bool:
        return self.cell_number_combo.currentIndex() == 0

    def set_labels_bottom_up(self, enabled: bool):
        self.cell_number_combo.setCurrentIndex(0 if enabled else 1)

    def lattice_display_options(self) -> dict[str, bool]:
        return {
            "nn": self.show_nn_check.isChecked(),
            "nnn": self.show_nnn_check.isChecked(),
            "ghosts": self.show_ghosts_check.isChecked(),
            "cells": self.show_cells_check.isChecked(),
        }

    def set_lattice_display_options(self, options: dict | None):
        values = {"nn": True, "nnn": True, "ghosts": True, "cells": True}
        if options:
            values.update({key: bool(value) for key, value in options.items()
                           if key in values})
        for checkbox, key in (
            (self.show_nn_check, "nn"),
            (self.show_nnn_check, "nnn"),
            (self.show_ghosts_check, "ghosts"),
            (self.show_cells_check, "cells"),
        ):
            checkbox.setChecked(values[key])

    def get_site_rows(self) -> list[tuple]:
        rows = []
        table_rows: list[int] = []
        for r in range(self.site_table.rowCount()):
            vals = [self._cell(self.site_table, r, c) for c in range(len(SITE_COLS))]
            if not any(vals):
                continue
            try:
                x = float(vals[0]); y = float(vals[1])
            except ValueError:
                raise ValueError(f"格点表第 {r + 1} 行 x/y 必须是有效数字") from None
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError(f"格点表第 {r + 1} 行 x/y 必须有限")
            sub = vals[2].strip() or None
            rows.append((x, y, sub))
            table_rows.append(r)
        self._site_table_rows = table_rows
        return rows

    def _refresh_site_table_rows(self) -> list[int]:
        """Return compact-site → physical table-row mapping.

        This intentionally only inspects whether a row is populated.  Full
        numeric validation remains centralized in :meth:`get_site_rows`, so
        a canvas edit can still report the original invalid row on the next
        rebuild instead of raising a surprising mapping exception here.
        """
        self._site_table_rows = [
            row for row in range(self.site_table.rowCount())
            if any(self._cell(self.site_table, row, col)
                   for col in range(len(SITE_COLS)))
        ]
        return list(self._site_table_rows)

    def get_cell_size(self) -> tuple[float, float] | None:
        if self._cell_vectors is not None:
            return None
        lx, ly = self.lx_spin.value(), self.ly_spin.value()
        if lx < 0 or ly < 0:
            raise ValueError("元胞 Lx 与 Ly 不能为负数")
        if lx == 0 and ly == 0:
            return None
        if lx == 0 or ly == 0:
            raise ValueError("元胞 Lx 与 Ly 必须同时设置，或同时选择自动")
        return float(lx), float(ly)

    def get_cell_vectors(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        return self._cell_vectors

    def set_cell_size(self, cell: tuple[float, float] | None):
        if cell is not None:
            try:
                lx, ly = (float(cell[0]), float(cell[1]))
            except (IndexError, TypeError, ValueError):
                raise ValueError("元胞尺寸必须是 (Lx, Ly) 数值对") from None
            if not (math.isfinite(lx) and math.isfinite(ly)):
                raise ValueError("元胞尺寸必须是有限数值")
            if lx < 0 or ly < 0 or ((lx == 0) != (ly == 0)):
                raise ValueError("元胞 Lx 与 Ly 必须为正数，或同时选择自动")
        self._cell_vectors = None
        self.lx_spin.blockSignals(True)
        self.ly_spin.blockSignals(True)
        if cell is None:
            self.lx_spin.setValue(0.0)
            self.ly_spin.setValue(0.0)
        else:
            self.lx_spin.setValue(lx)
            self.ly_spin.setValue(ly)
        self.lx_spin.blockSignals(False)
        self.ly_spin.blockSignals(False)
        self._sync_vector_spins(rectangular=True)

    def set_cell_vectors(self, vectors: tuple[tuple[float, float], tuple[float, float]] | None):
        if vectors is None:
            self._cell_vectors = None
            self._sync_vector_spins(rectangular=True)
            return
        try:
            a1 = (float(vectors[0][0]), float(vectors[0][1]))
            a2 = (float(vectors[1][0]), float(vectors[1][1]))
        except (IndexError, TypeError, ValueError):
            raise ValueError("元胞矢量必须是两个二维数值向量") from None
        det = a1[0] * a2[1] - a1[1] * a2[0]
        if not all(math.isfinite(value) for value in (*a1, *a2)):
            raise ValueError("元胞矢量必须是有限数值")
        if abs(det) < 1e-12:
            raise ValueError("a₁ 与 a₂ 不可共线；请调整元胞矢量")
        if math.hypot(*a1) <= 1e-12 or abs(a2[1]) <= 1e-12:
            raise ValueError("a₁ 必须非零，且 a₂ 的 y 分量必须非零")
        # Store an immutable, normalized copy.  A caller passing mutable lists
        # must not be able to alter the active geometry behind the panel's
        # change/rebuild signature.
        self._cell_vectors = (a1, a2)
        (a1x, a1y), (a2x, a2y) = self._cell_vectors
        old_lx = self.lx_spin.blockSignals(True)
        old_ly = self.ly_spin.blockSignals(True)
        try:
            self.lx_spin.setValue(math.hypot(a1x, a1y))
            self.ly_spin.setValue(math.hypot(a2x, a2y))
            tip = f"斜原胞：a₁=({a1x:g}, {a1y:g})，a₂=({a2x:g}, {a2y:g})"
            self.lx_spin.setToolTip(tip)
            self.ly_spin.setToolTip(tip)
        finally:
            self.lx_spin.blockSignals(old_lx)
            self.ly_spin.blockSignals(old_ly)
        self._sync_vector_spins()

    def set_hopping_strength(self, row: int, strength: float):
        """Apply an absolute bond strength and normalize its named group.

        For strengths 1.2 and 0.3 in the ``t`` group this writes ``-4*t`` and
        ``-t`` while setting the numeric parameter ``t`` to 0.3.  Thus symbolic
        and numerical views remain two representations of the same physics.
        """
        if not (0 <= int(row) < self.hop_table.rowCount()):
            return
        strength = float(strength)
        if not math.isfinite(strength) or strength <= 0:
            self.set_error("跃迁强度必须是正的有限数值")
            return
        parsed_rows = self._get_hop_rows_with_indices()
        selected_entry = next(
            ((table_row, hop) for table_row, hop in parsed_rows
             if table_row == int(row)),
            None,
        )
        if selected_entry is None:
            self.set_error("所选跃迁行为空或已失效，请重新选择跃迁线")
            return
        _selected_table_row, selected = selected_entry
        group_name = selected["name"]
        params = self.get_params()
        selected_expr = parse_expression(selected["amplitude"])
        free = sorted(str(symbol) for symbol in selected_expr.free_symbols)
        parameter = free[0] if len(free) == 1 else (
            group_name if group_name.isidentifier() else "t"
        )
        group_entries = [
            (table_row, hop) for table_row, hop in parsed_rows
            if hop["name"] == group_name
        ]
        physical: dict[int, Fraction] = {}
        signs: dict[int, int] = {}
        for table_row, hop in group_entries:
            value = evaluate_expression(hop["amplitude"], params)
            if abs(value.imag) > 1e-9:
                self.set_error("带复振幅的跃迁请使用 phase 列编辑，不能只改强度")
                return
            signs[table_row] = -1 if value.real < 0 else 1
            magnitude = strength if table_row == int(row) else abs(value.real)
            physical[table_row] = Fraction(f"{magnitude:.12g}").limit_denominator(100000)
        denominator = 1
        for value in physical.values():
            denominator = math.lcm(denominator, value.denominator)
        integer_values = [value.numerator * (denominator // value.denominator)
                          for value in physical.values()]
        common = 0
        for value in integer_values:
            common = math.gcd(common, abs(value))
        base = Fraction(common, denominator)
        if base <= 0:
            self.set_error("无法从跃迁强度得到有效比例")
            return
        self.hop_table.blockSignals(True)
        try:
            for table_row, _hop in group_entries:
                ratio = int(physical[table_row] / base)
                factor = "" if ratio == 1 else f"{ratio}*"
                sign = "-" if signs[table_row] < 0 else ""
                self.hop_table.setItem(
                    table_row, 5, self._table_item(f"{sign}{factor}{parameter}")
                )
        finally:
            self.hop_table.blockSignals(False)
        params[parameter] = float(base)
        self.set_params(params, force=True)
        self.set_error("")
        self._emit_changed()

    def _get_hop_rows_with_indices(self) -> list[tuple[int, dict]]:
        """Parse hopping rows while retaining their physical table indices.

        Empty rows are intentionally ignored by the public model payload, but
        their presence must not renumber an on-canvas coefficient editor.  The
        editor emits the actual QTableWidget row, so keep that identity until
        after all row-local validation and normalization is complete.
        """
        rows = []
        for r in range(self.hop_table.rowCount()):
            vals = [self._cell(self.hop_table, r, c) for c in range(len(HOP_COLS))]
            if not any(vals):
                continue
            def integer(col: int, label: str, default: str | None = None) -> int:
                text = vals[col].strip() or default
                if text is None or not _INTEGER_RE.fullmatch(text):
                    raise ValueError(
                        f"跃迁表第 {r + 1} 行 {label} 必须是整数，得到 {vals[col]!r}"
                    )
                return int(text)

            phase_mode = vals[6].strip() or "none"
            if phase_mode == "directional":
                raise ValueError(
                    f"跃迁表第 {r + 1} 行使用了尚未支持的方向依赖相位模式 "
                    "directional；请改用 none（实跃迁）或 phase（固定相位）"
                )
            if phase_mode not in {"none", "phase"}:
                raise ValueError(
                    f"跃迁表第 {r + 1} 行 phase_mode 无效：{phase_mode!r}"
                )
            sign = integer(8, "sign", "1")
            if sign not in {-1, 1}:
                raise ValueError(f"跃迁表第 {r + 1} 行 sign 必须是 +1 或 -1")
            endpoint_count = len(self._refresh_site_table_rows())
            def endpoint(col: int, label: str) -> int:
                display_value = integer(col, label)
                if not 1 <= display_value <= endpoint_count:
                    upper = max(1, endpoint_count)
                    raise ValueError(
                        f"跃迁表第 {r + 1} 行 {label} 必须是 1..{upper} 的格点编号，"
                        f"得到 {display_value}"
                    )
                return display_value - 1

            rows.append((r, {
                "name": vals[0].strip() or "t",
                "from_site": endpoint(1, "from"),
                "to_site": endpoint(2, "to"),
                "off_x": integer(3, "off_x", "0"),
                "off_y": integer(4, "off_y", "0"),
                "amplitude": vals[5].strip() or "1.0",
                "phase_mode": phase_mode,
                "phase": vals[7].strip() or "0",
                "phase_sign": sign,
            }))
        return rows

    def get_hop_rows(self) -> list[dict]:
        """Return validated hopping payloads without blank table rows."""
        return [hop for _table_row, hop in self._get_hop_rows_with_indices()]

    # ---- 填充 (预设加载 / 模型打开) ----

    def set_lattice_rows(self, sites):
        self._fill_table(self.site_table, SITE_COLS, [(x, y, s or "") for x, y, s in sites])
        self._site_table_rows = list(range(self.site_table.rowCount()))
        self._update_resource_hint()

    def set_hop_rows(self, hops):
        # ``hops`` is the model-facing representation (zero-based endpoints).
        # Keep the table friendly to humans by displaying conventional labels
        # 1…N while preserving the zero-based payload on the next read.
        display_rows = [self._hop_row_for_display(row) for row in hops]
        self._fill_table(self.hop_table, HOP_COLS, display_rows)
        self._update_hop_relation_tooltips()
        self._update_hop_relation_hint()

    def _hop_cell_changed(self, *_args):
        """Keep the compact relation summary useful while a row is edited."""
        self._update_hop_relation_tooltips()
        self._update_hop_relation_hint()
        self._emit_changed()

    def _site_cell_changed(self, row: int, column: int) -> None:
        """Refresh exact-value help after an inline table edit.

        Coordinate columns stay compact so the single control rail remains
        usable at high UI scale.  Qt may therefore elide the rendered text,
        but hovering the cell must still reveal the exact parsed value.
        """
        self._set_raw_value_tooltip(self.site_table, row, column)
        self._emit_changed()

    def _update_hop_relation_tooltips(self) -> None:
        """Attach per-row relation explanations without adding schema columns.

        The JSON/table schema stays nine columns wide, while every visible
        ``dx``/``dy`` cell tells the user exactly how it participates in the
        semi-infinite construction.  This is useful when several rows share
        the same endpoints but differ only in their cell offset.
        """
        if self._updating_hop_relation:
            return
        table = self.hop_table
        self._updating_hop_relation = True
        previous_blocked = table.blockSignals(True)
        try:
            for row in range(table.rowCount()):
                try:
                    dx = int(self._cell(table, row, 3).strip() or "0")
                    dy = int(self._cell(table, row, 4).strip() or "0")
                except ValueError:
                    relation = "正在编辑：请输入整数 dx / dy"
                    relation_kind = "invalid"
                else:
                    if dx == 0 and dy == 0:
                        relation = "胞内跃迁（同一元胞）"
                        relation_kind = "intra"
                    else:
                        relation = (
                            f"胞间跃迁（目标元胞偏移 dx={dx:+d}, dy={dy:+d}）"
                        )
                        relation_kind = "inter"
                # Refresh the raw role from the current visible text first;
                # users may have edited a cell since the last presentation
                # pass, and the tooltip must never show a stale value.
                for col in range(table.columnCount()):
                    item = table.item(row, col)
                    if item is not None:
                        item.setData(RAW_VALUE_ROLE, item.text())
                for col in (3, 4):
                    item = table.item(row, col)
                    if item is not None:
                        item.setToolTip(relation)
                self._style_hop_row(row, relation, relation_kind)
        finally:
            table.blockSignals(previous_blocked)
            self._updating_hop_relation = False

    @staticmethod
    def _mix_table_colors(base: QColor, accent: QColor, weight: float) -> QColor:
        """Blend a semantic accent into the current table base color.

        Using the palette instead of fixed light/dark RGB values keeps the
        relation cue readable for both themes and for user-defined Qt styles.
        The resulting tint is intentionally quiet so row selection remains the
        strongest visual state.
        """
        weight = max(0.0, min(1.0, float(weight)))
        return QColor(
            round(base.red() * (1.0 - weight) + accent.red() * weight),
            round(base.green() * (1.0 - weight) + accent.green() * weight),
            round(base.blue() * (1.0 - weight) + accent.blue() * weight),
        )

    def _style_hop_row(self, row: int, relation: str, relation_kind: str) -> None:
        """Make intra/inter-cell rows distinguishable without adding a column.

        ``dx``/``dy`` remain the source of truth and the JSON schema is
        untouched.  A subtle palette-derived tint plus a row tooltip gives
        users an immediate visual cue while preserving the compact layout at
        80–180% UI scales.  Invalid in-progress values use a warning tint.
        """
        table = self.hop_table
        palette = table.palette()
        base_role = (
            QPalette.ColorRole.Base
            if row % 2 == 0
            else QPalette.ColorRole.AlternateBase
        )
        base = palette.color(base_role)
        if relation_kind == "inter":
            color = self._mix_table_colors(
                base, palette.color(QPalette.ColorRole.Highlight), 0.16
            )
        elif relation_kind == "invalid":
            color = self._mix_table_colors(
                base, QColor(210, 65, 65), 0.18
            )
        else:
            color = base
        brush = QBrush(color)
        for col in range(table.columnCount()):
            item = table.item(row, col)
            if item is None:
                continue
            item.setData(Qt.BackgroundRole, brush)
            item.setData(Qt.UserRole, relation_kind)
            raw = item.data(RAW_VALUE_ROLE) or item.text()
            item.setData(Qt.ToolTipRole, f"{relation}\n完整值：{raw}")

    def _set_selected_hop_offset(self, off_x: int, off_y: int) -> None:
        """Set the selected row's cell relation through a compact UI action."""
        row = self.hop_table.currentRow()
        if not 0 <= row < self.hop_table.rowCount():
            return
        self.hop_table.blockSignals(True)
        try:
            self.hop_table.setItem(row, 3, self._table_item(str(int(off_x))))
            self.hop_table.setItem(row, 4, self._table_item(str(int(off_y))))
        finally:
            self.hop_table.blockSignals(False)
        self._update_hop_relation_tooltips()
        self._update_hop_relation_hint()
        self._emit_changed()

    def _copy_table_value(self, table: QTableWidget, row: int, column: int) -> str:
        """Copy the unabridged source text of one table cell."""
        item = table.item(int(row), int(column))
        if item is None:
            return ""
        raw = item.data(RAW_VALUE_ROLE)
        text = str(raw if raw is not None else item.text())
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)
        return text

    def _value_copy_action(
        self, menu: QMenu, table: QTableWidget, row: int, column: int,
    ):
        item = table.item(int(row), int(column))
        action = menu.addAction("复制完整值")
        action.setToolTip("复制该单元格未省略的原始文本到剪贴板")
        action.setEnabled(item is not None)
        action.triggered.connect(
            lambda _checked=False: self._copy_table_value(table, row, column)
        )
        return action

    def _show_site_context_menu(self, position) -> None:
        """Show a small, cell-focused menu for coordinate inspection."""
        row = self.site_table.rowAt(position.y())
        column = self.site_table.columnAt(position.x())
        if row < 0 or column < 0:
            return
        self.site_table.setCurrentCell(row, column)
        menu = QMenu(self.site_table)
        self._value_copy_action(menu, self.site_table, row, column)
        menu.exec(self.site_table.viewport().mapToGlobal(position))

    def _create_hop_context_menu(self, row: int, column: int = 0) -> QMenu:
        """Build the row relation menu without coupling it to pointer events."""
        menu = QMenu(self.hop_table)
        menu.setToolTipsVisible(True)
        self._value_copy_action(menu, self.hop_table, row, column)
        menu.addSeparator()
        intra = menu.addAction("设为胞内（dx=0, dy=0）")
        intra.setToolTip("同一元胞内的跃迁，不产生 Bloch 跨胞相位。")
        right = menu.addAction("设为右侧胞间（dx=+1, dy=0）")
        right.setToolTip("半无限模式下进入 x-Bloch 的一阶 H₁。")
        left = menu.addAction("设为左侧胞间（dx=-1, dy=0）")
        left.setToolTip("半无限模式下进入 x-Bloch 的反向胞间关系。")
        menu.addSeparator()
        edit = menu.addAction("在表格中编辑 dx / dy…")
        edit.setToolTip("保留任意整数偏移，适合长程 Bloch 谐波或 y 方向连接。")
        intra.triggered.connect(lambda: self._set_selected_hop_offset(0, 0))
        right.triggered.connect(lambda: self._set_selected_hop_offset(1, 0))
        left.triggered.connect(lambda: self._set_selected_hop_offset(-1, 0))
        edit.triggered.connect(lambda: self._edit_hop_offset_cells(row))
        return menu

    def _edit_hop_offset_cells(self, row: int) -> None:
        """Focus the first relation field for precise manual editing."""
        if not 0 <= row < self.hop_table.rowCount():
            return
        self.hop_table.setCurrentCell(row, 3)
        item = self.hop_table.item(row, 3)
        if item is not None:
            self.hop_table.editItem(item)

    def _show_hop_context_menu(self, position) -> None:
        """Offer common intra/inter-cell edits on the row under the pointer."""
        row = self.hop_table.rowAt(position.y())
        column = self.hop_table.columnAt(position.x())
        if row < 0 or column < 0:
            return
        self.hop_table.setCurrentCell(row, column)
        self.hop_table.selectRow(row)
        menu = self._create_hop_context_menu(row, column)
        menu.exec(self.hop_table.viewport().mapToGlobal(position))

    def _update_hop_relation_hint(self):
        """Show an always-visible, low-noise summary of cell relations."""
        if not hasattr(self, "hop_relation_hint"):
            return
        try:
            rows = self.get_hop_rows()
        except ValueError:
            self.hop_relation_hint.setText(
                "关系：正在编辑（展开高级列检查 dx / dy；dx=0 且 dy=0 为胞内）"
            )
            return
        intra = sum(
            int(row["off_x"]) == 0 and int(row["off_y"]) == 0 for row in rows
        )
        inter = len(rows) - intra
        if self.is_semi():
            x_inter = sum(int(row["off_x"]) != 0 for row in rows)
            y_inter = sum(int(row["off_y"]) != 0 for row in rows)
            diagonal_inter = sum(
                int(row["off_x"]) != 0 and int(row["off_y"]) != 0
                for row in rows
            )
            # The Bloch factor carries the *integer* cell displacement.  A
            # simple "e^(±ikₓ)" label is only correct for |dx|=1 and used to
            # silently misdescribe dx=2/3 rows as nearest-cell couplings.
            x_offsets = sorted({abs(int(row["off_x"])) for row in rows
                                if int(row["off_x"]) != 0})
            if x_offsets:
                harmonics = "、".join(
                    "e^(±ikₓ)" if order == 1 else f"e^(±i{order}kₓ)"
                    for order in x_offsets
                )
                detail = (
                    f"x 方向胞间 {x_inter} 条，Bloch 谐波 {harmonics}"
                )
            else:
                detail = "x 方向胞间 0 条"
            if y_inter:
                detail += f"；y 方向胞间 {y_inter} 条为有限方向连接"
            if diagonal_inter:
                detail += f"；其中斜向胞间 {diagonal_inter} 条同时含 x/y 偏移"
        else:
            detail = "胞间项在有限边界外会按边界规则截断"
        self.hop_relation_hint.setText(
            f"关系：胞内 {intra} 条 · 胞间 {inter} 条　|　{detail}"
        )

    def update_site_position(self, index: int, x: float, y: float):
        table_rows = self._refresh_site_table_rows()
        if not 0 <= int(index) < len(table_rows):
            self.set_error("所选格点行为空或已失效，请重新选择格点")
            return
        # Keep the panel as a second, model-facing validity gate.  The canvas
        # normally prevents this case, but signals can also come from plug-ins
        # or synthetic input events; never write a duplicate into the table.
        try:
            candidate_x, candidate_y = float(x), float(y)
        except (TypeError, ValueError):
            self.set_error("格点坐标必须是有限数值")
            return
        if not (math.isfinite(candidate_x) and math.isfinite(candidate_y)):
            self.set_error("格点坐标必须是有限数值")
            return
        for other_index, other_row in enumerate(table_rows):
            if other_index == int(index):
                continue
            try:
                other_x = float(self._cell(self.site_table, other_row, 0))
                other_y = float(self._cell(self.site_table, other_row, 1))
            except (TypeError, ValueError):
                continue
            if math.hypot(candidate_x - other_x, candidate_y - other_y) <= 1e-9:
                self.set_error("格点坐标不能重复；已保留拖动前位置")
                return
        table_row = table_rows[int(index)]
        self.site_table.blockSignals(True)
        self.site_table.setItem(
            table_row, 0, self._table_item(candidate_x)
        )
        self.site_table.setItem(
            table_row, 1, self._table_item(candidate_y)
        )
        self.site_table.blockSignals(False)
        self._emit_changed()

    def append_site(self, x: float, y: float, sublattice: str = "A"):
        """Append a site with the same finite/duplicate guard as canvas edits.

        The canvas normally validates before emitting ``siteAddRequested``;
        this second gate protects integrations and synthetic events that call
        the panel slot directly, so a malformed coordinate cannot enter the
        table and make the next rebuild appear to drop the lattice.
        """
        try:
            x_value, y_value = float(x), float(y)
        except (TypeError, ValueError, OverflowError):
            self.set_error("格点坐标必须是有限数值")
            return
        if not (math.isfinite(x_value) and math.isfinite(y_value)):
            self.set_error("格点坐标必须是有限数值")
            return
        for row in self._refresh_site_table_rows():
            try:
                old_x = float(self._cell(self.site_table, row, 0))
                old_y = float(self._cell(self.site_table, row, 1))
            except (TypeError, ValueError):
                continue
            if math.hypot(x_value - old_x, y_value - old_y) <= 1e-9:
                self.set_error("格点坐标不能重复；请换一个位置")
                return
        self._add_row(self.site_table, [x_value, y_value, sublattice])

    def remove_site(self, index: int):
        """Remove a site and all incident hops; reindex remaining endpoints."""
        table_rows = self._refresh_site_table_rows()
        if not 0 <= int(index) < len(table_rows):
            self.set_error("所选格点行为空或已失效，请重新选择格点")
            return
        index = int(index)
        table_row = table_rows[index]
        self.site_table.blockSignals(True)
        self.hop_table.blockSignals(True)
        self.site_table.removeRow(table_row)
        self._site_table_rows = [
            row - 1 if row > table_row else row
            for row in table_rows if row != table_row
        ]
        for row in range(self.hop_table.rowCount() - 1, -1, -1):
            try:
                # Table endpoints are one-based; reindex in compact internal
                # coordinates, then write the result back as one-based.
                fr_display = int(self._cell(self.hop_table, row, 1))
                to_display = int(self._cell(self.hop_table, row, 2))
                if fr_display < 1 or to_display < 1:
                    continue
                fr = fr_display - 1
                to = to_display - 1
            except ValueError:
                continue
            if index in (fr, to):
                self.hop_table.removeRow(row)
                continue
            if fr > index:
                self.hop_table.setItem(row, 1, self._table_item(str(fr)))
            if to > index:
                self.hop_table.setItem(row, 2, self._table_item(str(to)))
        self.site_table.blockSignals(False)
        self.hop_table.blockSignals(False)
        self._update_hop_relation_tooltips()
        self._emit_changed()

    def append_hop(self, row, *, reveal_relation: bool = False):
        # Dialogs and canvas tools emit model-facing zero-based endpoints;
        # format them once here for the one-based visible table.
        self._add_row(self.hop_table, self._hop_row_for_display(row))
        # Canvas-created bonds arrive through this common path, rather than
        # through the side-panel relation menu.  Keep the physical relation
        # immediately discoverable and the summary synchronized in both
        # cases; otherwise the row is correct internally but appears to have
        # remained an intra-cell bond until the next unrelated edit.
        try:
            off_x, off_y = int(row[3]), int(row[4])
        except (IndexError, TypeError, ValueError):
            off_x = off_y = 0
        if reveal_relation and (self.is_semi() or off_x != 0 or off_y != 0):
            self.hop_advanced_check.setChecked(True)
        self._update_hop_relation_hint()

    def _default_hop_sites(self) -> tuple[int, int]:
        """Return a valid endpoint pair for a newly inserted hopping row.

        When a user is inspecting a particular bond, the relation menu should
        preserve that bond's endpoints and only change its cell relation.  The
        old unconditional ``0 → 1`` fallback was technically valid but made a
        multi-site model appear to ignore the user's selection.  Invalid or
        absent selections still use the deterministic first-pair fallback.
        """
        # The canvas/model use compact valid-site indices.  A temporarily
        # blank table row must not become a phantom endpoint (particularly in
        # one-site models, where it used to turn the safe inter-cell starter
        # into an invalid 0→1 intra-cell bond).
        site_count = len(self._refresh_site_table_rows())
        if site_count <= 1:
            return 0, 0
        selected = self.hop_table.currentRow()
        if 0 <= selected < self.hop_table.rowCount():
            try:
                from_site = int(self._cell(self.hop_table, selected, 1).strip()) - 1
                to_site = int(self._cell(self.hop_table, selected, 2).strip()) - 1
            except (TypeError, ValueError):
                pass
            else:
                if 0 <= from_site < site_count and 0 <= to_site < site_count:
                    return from_site, to_site
        return 0, 1

    def _add_hop_with_offset(self, off_x: int, off_y: int):
        """Add a real-valued starter bond with an explicit cell relation."""
        from_site, to_site = self._default_hop_sites()
        self._add_row(
            self.hop_table,
            ["t", from_site + 1, to_site + 1, int(off_x), int(off_y), "-t", "none", "0", 1],
        )
        row = self.hop_table.rowCount() - 1
        self.hop_table.selectRow(row)
        self.hop_table.scrollToItem(
            self.hop_table.item(row, 0), QAbstractItemView.PositionAtCenter,
        )
        # dx/dy are the essential distinction for a semi-infinite model and
        # for any explicit inter-cell term in OBC.  Make them visible after a
        # relation-menu action rather than hiding the user's newly created
        # physical connection behind the compact table.
        if (self.is_semi() or off_x != 0 or off_y != 0) and not self.hop_advanced_check.isChecked():
            self.hop_advanced_check.setChecked(True)
        self._update_hop_relation_hint()

    def _add_custom_hop_dialog(self):
        """Create a hopping row without forcing users through raw table columns.

        The table remains the authoritative fallback for bulk edits, while this
        dialog provides a discoverable path for arbitrary inter-cell offsets
        (including ``dx=2``/higher Bloch harmonics) and phase expressions.
        """
        from_site, to_site = self._default_hop_sites()
        dialog = HoppingDialog(
            from_site, to_site, self, semi=self.is_semi(),
            site_count=len(self._refresh_site_table_rows()),
        )
        custom_index = dialog.cell_relation.findData("custom")
        if custom_index >= 0:
            dialog.cell_relation.setCurrentIndex(custom_index)
        if dialog.exec() != QDialog.Accepted:
            return
        row = dialog.row()
        self.append_hop(row)
        if self.is_semi() or int(row[3]) != 0 or int(row[4]) != 0:
            self.hop_advanced_check.setChecked(True)
        last = self.hop_table.rowCount() - 1
        self.hop_table.selectRow(last)
        self.hop_table.scrollToItem(
            self.hop_table.item(last, 0), QAbstractItemView.PositionAtCenter,
        )
        self._update_hop_relation_hint()

    def _add_default_hop(self):
        """Add a valid, discoverable starter bond from the table entry point.

        The former hard-coded ``0 -> 1`` row was invalid for one-site models
        (blank, chain, square, etc.) and silently introduced an undefined
        ``phi`` symbol.  A one-site model instead starts with a right
        inter-cell bond, which is meaningful in both OBC and x-Bloch modes;
        multi-site models start with the obvious intra-cell 0 -> 1 bond.
        """
        # One-site models have no distinct intra-cell pair, so the default is
        # a right inter-cell bond; multi-site models start with the intuitive
        # intra-cell 0→1 bond.  The menu next to this button exposes all
        # relations explicitly.
        if len(self._refresh_site_table_rows()) <= 1:
            self._add_hop_with_offset(1, 0)
        else:
            self._add_hop_with_offset(0, 0)

    # ---- 内部 ----

    @staticmethod
    def _cell(table: QTableWidget, row: int, col: int) -> str:
        it = table.item(row, col)
        return it.text() if it is not None else ""

    @staticmethod
    def _hop_row_for_display(row) -> list:
        """Convert one model hop row to the human-facing table numbering.

        The conversion is intentionally tolerant while a row is being built:
        malformed endpoint text is left untouched so the normal table parser
        can report a precise validation error instead of failing during a UI
        refresh.
        """
        values = list(row)
        for col in HOP_ENDPOINT_COLUMNS:
            if col >= len(values) or values[col] is None:
                continue
            text = str(values[col]).strip()
            if not text:
                continue
            try:
                values[col] = int(text) + 1
            except (TypeError, ValueError):
                pass
        return values

    @staticmethod
    def _fill_table(table: QTableWidget, cols, data):
        table.blockSignals(True)
        table.setRowCount(len(data))
        for r, row in enumerate(data):
            for c, val in enumerate(row):
                if c >= len(cols):
                    break
                table.setItem(r, c, ControlPanel._table_item(val))
        table.blockSignals(False)

    @staticmethod
    def _table_item(value) -> QTableWidgetItem:
        """Create a compact item while preserving the exact value on hover.

        Long floating-point coordinates are the main source of ``0...`` in
        the 180% rail.  Show eight significant digits in the cell (enough for
        visual inspection and editing), while the tooltip keeps Python's full
        model-facing representation for precision-sensitive checks.
        """
        raw_text = str(value)
        text = raw_text
        if isinstance(value, float) and math.isfinite(value) and len(raw_text) > 10:
            text = f"{value:.8g}"
        item = QTableWidgetItem(text)
        item.setData(RAW_VALUE_ROLE, raw_text)
        item.setToolTip(f"完整值：{raw_text}")
        return item

    @staticmethod
    def _set_raw_value_tooltip(table: QTableWidget, row: int, column: int) -> None:
        item = table.item(int(row), int(column))
        if item is None:
            return
        text = item.text()
        item.setData(RAW_VALUE_ROLE, text)
        item.setToolTip(f"完整值：{text}")

    def _add_row(self, table: QTableWidget, defaults):
        table.blockSignals(True)
        r = table.rowCount()
        table.insertRow(r)
        for c, d in enumerate(defaults):
            table.setItem(r, c, self._table_item(d))
        table.blockSignals(False)
        if table is self.hop_table:
            self._update_hop_relation_tooltips()
        self._emit_changed()

    def _del_row(self, table: QTableWidget):
        r = table.currentRow()
        if r >= 0:
            table.removeRow(r)
            if table is self.hop_table:
                self._update_hop_relation_hint()
            self._emit_changed()

    def _kx_changed(self, val: int):
        self.kx_edit.setText(f"{val / 100:.3f}")
        self._emit_changed()

    def _kx_from_edit(self):
        try:
            v = float(self.kx_edit.text())
            self.set_kx(v * 3.141592653589793)
        except ValueError:
            self.kx_edit.setStyleSheet("border:1px solid #b00020;")
        else:
            self.kx_edit.setStyleSheet("")
        self._emit_changed()

    def _emit_changed(self):
        self._update_resource_hint()
        self.changed.emit()

    def _toggle_hop_columns(self, advanced: bool):
        """Keep cell relation visible; disclose only genuinely advanced fields.

        ``dx``/``dy`` determine whether a row is intra-cell or inter-cell and
        are therefore not optional decoration.  Hiding them was especially
        confusing in the half-infinite editor: a newly inserted row looked
        identical to an intra-cell bond even though changing its relation was
        already supported by the model.  The compact layout keeps these two
        short columns visible and hides only phase metadata, which can still
        be edited by enabling the advanced columns.
        """
        for col in (6, 7, 8):
            self.hop_table.setColumnHidden(col, not bool(advanced))
        if not advanced:
            header = self.hop_table.horizontalHeader()
            header.setSectionResizeMode(QHeaderView.Interactive)
            for col, width in (
                (0, 52), (1, 42), (2, 42), (3, 34), (4, 34),
                (5, 64),
            ):
                self.hop_table.setColumnWidth(col, width)
