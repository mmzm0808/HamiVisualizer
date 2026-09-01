"""矩阵视图: 两遍绘制 (先色块后文字, 避免遮挡) + 轴标号.

对应 MATLAB draw_matrix §2.1/§6.3: 色块 z=0 + 文字 z=1，矩阵采用
纯单元格排版，不额外绘制包住整张矩阵的外层括号，
格点 >8 纯热图 (不画文字), 点击 cellClicked 供详情弹窗。
配色与判色同 MATLAB §5.2 (零灰/对角黄/NNN暖灰褐/x-Bloch和橙/复跃迁浅蓝)。
文字源保持 TeX 语义；小矩阵与大矩阵视口叠加层共用轻量数学脚本布局，
保证两条路径的数学排版、基线和缩放一致。
"""

from __future__ import annotations

import math

import sympy as sp

from PySide6.QtCore import QPointF, QRectF, Signal, Qt
from PySide6.QtGui import (
    QColor, QBrush, QFont, QImage, QPen, QPixmap,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSceneHelpEvent,
    QToolTip,
)

from ..model.symbolic import (
    format_bloch_elem,
    format_elem,
    sym_pretty,
    wrap_tex,
)
from .math_text import MathLayout, MathTextItem, math_font
from .rendermodel import (
    DARK_PALETTE,
    MatrixSceneData,
    Palette,
    resolve_cell_class,
)

TEXT_MAX = 8   # 首次绘制直接显示的阶数上限
TEXT_ITEM_MAX = 64  # 放大后懒加载文字的安全上限（与栅格化阈值对齐）
TEXT_CELL_PX_THRESHOLD = 34.0  # 单元屏幕像素达到此值才显示文字
MARGIN = 36    # 轴标号留白；也避免多字符逻辑坐标贴到矩阵边缘
RASTER_THRESHOLD = 64
RASTER_SIZE = 900


def _rgb(rgb: tuple, alpha: int = 255) -> QColor:
    return QColor(*[int(c * 255) for c in rgb], alpha)


def _cell_pen(rgb: tuple) -> QPen:
    """cosmetic 画笔: 宽度恒为像素 (与缩放无关)."""
    pen = QPen(_rgb(rgb), 0.4)
    pen.setCosmetic(True)
    return pen


class MatrixView(QGraphicsScene):
    """哈密顿量矩阵热图 (QGraphicsScene).

    每个矩阵元独立绘制在色块中；“矩阵 + 晶格”组合页直接复用同一
    Scene，因此不会在晶格右侧额外生成巨大 ``[H]``/``(H)`` 包围层。
    """

    cellClicked = Signal(int, int)  # (i, j) 0-based
    # ``None`` is emitted whenever a rebuild clears the previous selection;
    # a tuple carries the new zero-based (row, column).  Keeping this signal
    # on the scene makes the dedicated and combined matrix tabs share one
    # source of truth without inspecting private view state.
    selectionChanged = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: MatrixSceneData | None = None
        self._cell_size = 0.0
        self._matrix_rect = QRectF()
        self._texts: list[list[str]] | None = None
        self._text_items: list[QGraphicsItem] = []
        self._view_zoom: dict[int, float] = {}
        self._zoom_level = 1.0
        self._axis_labels: tuple[str, ...] = ()
        self._dark = False
        self._syncing_edge_text = False
        self._selected_cell: tuple[int, int] | None = None
        self._selection_item: QGraphicsRectItem | None = None
        # Parsed layout is used only by the large-matrix viewport overlay.
        # Cache the small set of repeated labels (zero/onsite/NN terms) so
        # repainting while panning does not parse a formula per visible cell.
        self._overlay_layout_cache: dict[tuple[str, int, bool], tuple[MathLayout, QFont]] = {}

    def set_theme(self, dark: bool):
        """切换矩阵视图的明暗配色（单元格底色 + 冻结标尺 + 矩阵文字）。

        已有数据时按新调色板重绘色块，并保留文字层的缩放状态。
        """
        self._dark = bool(dark)
        if self._data is not None:
            selected = self._selected_cell
            level = max(self._view_zoom.values(), default=self._zoom_level)
            self.set_data(self._data)
            self.set_zoom_level(level)
            if selected is not None:
                self.select_cell(*selected)

    def _text_color(self) -> QColor:
        return QColor(232, 238, 246) if self._dark else QColor(20, 20, 20)

    def set_data(self, data: MatrixSceneData):
        self._data = data
        self.clear()
        self._selected_cell = None
        self._selection_item = None
        self.selectionChanged.emit(None)
        self._texts = None
        self._text_items = []
        self._view_zoom.clear()
        self._zoom_level = 1.0
        self._overlay_layout_cache.clear()
        n = data.n
        self._axis_labels = (
            tuple(str(v) for v in data.sites)
            if len(data.sites) == n else tuple(str(i + 1) for i in range(n))
        )
        if n < 0:
            raise ValueError("矩阵维度不能为负")
        values_shape = getattr(data.values, "shape", None)
        if n and values_shape != (n, n):
            raise ValueError(f"矩阵 values 形状应为 {(n, n)}, 得到 {values_shape}")
        if n == 0:
            self.setSceneRect(QRectF())
            self._matrix_rect = QRectF()
            return
        cell = RASTER_SIZE / n if n > RASTER_THRESHOLD else max(28.0, min(48.0, 800.0 / n))
        self._cell_size = cell
        self._matrix_rect = QRectF(MARGIN, MARGIN, n * cell, n * cell)
        self.setSceneRect(QRectF(0, 0, n * cell + 2 * MARGIN, n * cell + 2 * MARGIN))
        pal = DARK_PALETTE if self._dark else Palette()

        if n > RASTER_THRESHOLD:
            image = QImage(n, n, QImage.Format_RGB32)
            for i in range(n):
                for j in range(n):
                    cls = resolve_cell_class(data.values, i, j, data.t, data.phi)
                    image.setPixelColor(j, i, _rgb(pal.color(cls)))
            pixmap = QPixmap.fromImage(image).scaled(
                round(n * cell), round(n * cell),
                Qt.IgnoreAspectRatio, Qt.FastTransformation,
            )
            item = QGraphicsPixmapItem(pixmap)
            item.setPos(MARGIN, MARGIN)
            item.setZValue(0)
            self.addItem(item)
            return

        texts = self._cell_texts(data)
        self._texts = texts

        # 第一遍: 全部色块 (z=0)
        for i in range(n):
            for j in range(n):
                cls = resolve_cell_class(data.values, i, j, data.t, data.phi)
                rgb = pal.color(cls)
                x, y = MARGIN + j * cell, MARGIN + i * cell
                item = QGraphicsRectItem(x, y, cell, cell)
                item.setBrush(QBrush(_rgb(rgb)))
                item.setPen(_cell_pen(pal.edge_map[cls]))
                item.setZValue(0)
                item.setData(0, (i, j))
                t = texts[i][j]
                # Keep a tooltip on zero cells as well.  Their canvas text is
                # intentionally omitted to reduce clutter, but users must
                # still be able to inspect the exact element under the mouse.
                item.setToolTip(self._cell_tooltip_from_text(i, j, t or "0"))
                self.addItem(item)

        # 小矩阵直接创建；中等矩阵等用户放大到足够阅读时再懒加载。
        if n <= TEXT_MAX:
            self._ensure_text_items()
            self._set_text_visibility(self._text_is_readable())

        # 行列号不再作为场景图元绘制；它们由每个 QGraphicsView 在视口
        # 四边独立叠加，因此放大并平移到矩阵中部时仍持续可见。

    def select_cell(self, i: int, j: int) -> None:
        """Highlight one matrix cell without changing the view transform.

        Selection is a presentation-only layer.  It deliberately uses a
        cosmetic pen and does not accept mouse buttons, so a highlighted cell
        cannot steal the next click or turn a simple click into a pan gesture.
        """
        if self._data is None:
            return
        n = int(self._data.n)
        if not (0 <= int(i) < n and 0 <= int(j) < n):
            return
        self._selected_cell = (int(i), int(j))
        if self._selection_item is not None:
            self.removeItem(self._selection_item)
            self._selection_item = None
        item = QGraphicsRectItem(
            MARGIN + int(j) * self._cell_size,
            MARGIN + int(i) * self._cell_size,
            self._cell_size,
            self._cell_size,
        )
        pen = QPen(QColor("#62a8ff") if self._dark else QColor("#1769d1"), 2.0)
        pen.setCosmetic(True)
        item.setPen(pen)
        item.setBrush(Qt.NoBrush)
        item.setZValue(3.0)
        item.setAcceptedMouseButtons(Qt.NoButton)
        item.setData(0, "matrix-selection")
        self.addItem(item)
        self._selection_item = item
        self.selectionChanged.emit(self._selected_cell)

    @property
    def selected_cell(self) -> tuple[int, int] | None:
        """Currently selected zero-based cell, if any."""
        return self._selected_cell

    def is_interactive_position(self, scene_pos: QPointF) -> bool:
        """Return whether a left click at ``scene_pos`` belongs to a matrix.

        ``ZoomGraphicsView`` uses this small scene-level hit-test to keep a
        matrix-cell selection from arming the canvas pan gesture.  It is
        deliberately independent of the rendered item type: small matrices
        contain one rect per cell, while large matrices use a single raster
        pixmap and both must have identical click semantics.
        """
        if self._data is None or self._cell_size <= 0:
            return False
        return self._matrix_rect.contains(QPointF(scene_pos))

    def _cell_at_scene_pos(self, scene_pos: QPointF) -> tuple[int, int] | None:
        """Map a scene position to one zero-based matrix cell."""
        if self._data is None or self._cell_size <= 0:
            return None
        pos = QPointF(scene_pos)
        if not self._matrix_rect.contains(pos):
            return None
        col = math.floor((pos.x() - MARGIN) / self._cell_size)
        row = math.floor((pos.y() - MARGIN) / self._cell_size)
        n = int(self._data.n)
        if 0 <= row < n and 0 <= col < n:
            return int(row), int(col)
        return None

    def _cell_coordinate_labels(self, row: int, col: int) -> tuple[str, str]:
        """Return one-based array and logical ruler labels for one cell."""
        row_label = (
            self._axis_labels[row] if row < len(self._axis_labels) else str(row + 1)
        )
        col_label = (
            self._axis_labels[col] if col < len(self._axis_labels) else str(col + 1)
        )
        return f"H[{row + 1},{col + 1}]", f"H[{row_label},{col_label}]"

    def _cell_tooltip_from_text(self, row: int, col: int, value: str) -> str:
        """Compose the shared small/raster matrix tooltip."""
        index_label, logical_label = self._cell_coordinate_labels(row, col)
        coordinate = (
            logical_label
            if logical_label == index_label
            else f"{logical_label}（{index_label}）"
        )
        return f"{coordinate} = {str(value).replace(chr(10), ' ')}"

    def cell_tooltip(self, i: int, j: int) -> str:
        """Return the exact user-facing tooltip for any matrix cell."""
        _, _, value = self.cell_details(i, j)
        return self._cell_tooltip_from_text(int(i), int(j), value)

    def helpEvent(self, event: QGraphicsSceneHelpEvent) -> None:
        """Provide per-cell help even when the matrix is one raster pixmap."""
        cell = self._cell_at_scene_pos(event.scenePos())
        if cell is not None:
            QToolTip.showText(
                event.screenPos(), self.cell_tooltip(*cell), event.widget(),
            )
            event.accept()
            return
        QToolTip.hideText()
        super().helpEvent(event)

    def cell_details(self, i: int, j: int) -> tuple[str, str, str]:
        """Return index labels and formatted text for a selected cell."""
        if self._data is None:
            raise ValueError("矩阵尚未生成")
        n = int(self._data.n)
        if not (0 <= int(i) < n and 0 <= int(j) < n):
            raise IndexError(f"矩阵下标越界: ({i}, {j})")
        row, col = int(i), int(j)
        index_label, logical_label = self._cell_coordinate_labels(row, col)
        text = self._cell_text_at(row, col) or "0"
        # Numerical arrays remain zero-based internally, but all visible
        # matrix coordinates follow the one-based lattice numbering used by
        # the canvas and rulers.
        return index_label, logical_label, text

    def cell_latex(self, i: int, j: int) -> str:
        """Return a compilable LaTeX fragment for one matrix cell.

        The on-canvas representation intentionally uses compact Unicode
        symbols (``φ``, ``ω`` and a middle dot) for visual readability.  A
        clipboard consumer usually wants source text instead, so this method
        converts the same current cell into a conservative LaTeX subset while
        preserving the exact symbolic/Bloch expression whenever available.
        """
        if self._data is None:
            raise ValueError("矩阵尚未生成")
        n = int(self._data.n)
        row, col = int(i), int(j)
        if not (0 <= row < n and 0 <= col < n):
            raise IndexError(f"矩阵下标越界: ({i}, {j})")
        data = self._data
        raw = data.matrix[row, col] if data.matrix is not None else data.values[row, col]
        if data.mode == "symbolic":
            expression = sym_pretty(raw) if raw != 0 else "0"
        elif data.mode == "numeric":
            expression = format_elem(data.values[row, col], "numeric")
        elif data.smart_labels and (row, col) in data.smart_labels:
            expression = sym_pretty(data.smart_labels[(row, col)])
        elif isinstance(raw, sp.Basic) and any(
            getattr(s, "name", str(s)) in {"kx", "k_x"}
            for s in raw.free_symbols
        ):
            expression = format_bloch_elem(raw, data.formatter, row == col)
        else:
            expression = format_elem(data.values[row, col], "smart", row == col, data.formatter)
        expression = str(expression).replace("\n", " ").strip() or "0"
        # Keep this conversion deliberately small and reversible: the source
        # formatter already emits braces for exponents/subscripts, so broad
        # HTML/TeX escaping here would corrupt valid expressions.
        return (
            expression.replace("−", "-")
            .replace("·", r"\,")
            .replace("φ", r"\phi")
            .replace("ω", r"\omega")
        )

    def matrix_latex(self, *, max_elements: int = 4096) -> str:
        """Return the current matrix as a paste-ready LaTeX environment.

        The on-canvas matrix intentionally has no enclosing bracket layer;
        this export is a document-oriented artifact, so ``bmatrix`` supplies
        the conventional delimiters expected in a paper.  Each entry goes
        through :meth:`cell_latex`, keeping symbolic Bloch factors and the
        numeric/smart formatter identical to what the user inspected.
        """
        if self._data is None:
            raise ValueError("矩阵尚未生成")
        n = int(self._data.n)
        if n <= 0:
            raise ValueError("矩阵必须是非空方阵")
        if n * n > int(max_elements):
            raise ValueError(
                f"矩阵为 {n}×{n}，超过一次复制的安全上限 {int(max_elements)} 个元素；"
                "请减少格点数后再复制。"
            )
        rows = []
        for i in range(n):
            values = [self.cell_latex(i, j) for j in range(n)]
            rows.append(" & ".join(values))
        return "\\begin{bmatrix}\n" + " \\\\\n".join(rows) + "\n\\end{bmatrix}"

    def set_zoom_level(self, scale: float, source=None) -> None:
        """按视口缩放级别切换文字层。

        同一 scene 可能被“矩阵”与“矩阵+晶格”两个视图共享，因此以所有
        已注册视图中的最大缩放级别决定文字是否创建/显示，避免后一次 fit
        把前一个视图已放大的文字隐藏。
        """
        try:
            level = max(0.01, float(scale))
        except (TypeError, ValueError):
            return
        self._zoom_level = level
        self._view_zoom[id(source) if source is not None else 0] = level
        if self._data is None or self._cell_size <= 0:
            return
        readable = self._text_is_readable()
        if readable and self._data.n <= TEXT_ITEM_MAX:
            self._ensure_text_items()
        self._set_text_visibility(readable)

    @property
    def text_visible(self) -> bool:
        return bool(self._text_items and any(item.isVisible() for item in self._text_items))

    def text_visible_in_view(self, view) -> bool:
        """Return whether cells are readable in one concrete viewport."""
        if self._data is None or self._cell_size <= 0 or view is None:
            return False
        return self._cell_size * abs(float(view.transform().m11())) >= TEXT_CELL_PX_THRESHOLD

    def _overlay_text_pixel_size(self, view) -> int:
        """Choose a readable raster-path font size from the current view scale.

        Large matrices use one raster image plus a viewport text overlay rather
        than O(n²) graphics text items.  The overlay is painted in viewport
        pixels, so it must explicitly follow the cell's on-screen size; a
        fixed font made zooming appear to change the grid but not the labels.
        """
        if view is None or self._cell_size <= 0:
            return 8
        pixels = self._cell_size * abs(float(view.transform().m11()))
        # Do not freeze the label at 32 px: at high zoom the grid keeps
        # growing, so the math glyphs must grow with it as well.  A generous
        # cap only protects against pathological transforms and is far above
        # the normal interactive range.
        return max(8, min(160, round(pixels * 0.28)))

    @staticmethod
    def _math_font(pixel_size: int) -> QFont:
        """Return the shared mathematics face used by both render paths."""
        return math_font(pixel_size)

    def _overlay_math_layout(self, text: str, pixel_size: int) -> tuple[MathLayout, QFont]:
        """Build/cache one parsed TeX-subset layout for a matrix-cell label."""
        key = (str(text), int(pixel_size), bool(self._dark))
        cached = self._overlay_layout_cache.get(key)
        if cached is not None:
            return cached
        font = self._math_font(pixel_size)
        value = (MathLayout(wrap_tex(str(text), 12)), font)
        if len(self._overlay_layout_cache) >= 512:
            self._overlay_layout_cache.pop(next(iter(self._overlay_layout_cache)))
        self._overlay_layout_cache[key] = value
        return value

    def _draw_overlay_math_text(self, painter, text: str, rect: QRectF,
                                pixel_size: int) -> None:
        """Paint one centered, fitted math label in viewport coordinates."""
        if rect.width() <= 2 or rect.height() <= 2 or not text:
            return
        layout, font = self._overlay_math_layout(text, pixel_size)
        size = layout.metrics(font)
        width, height = float(size.width), float(size.height)
        if width <= 0 or height <= 0:
            return
        scale = min((rect.width() - 2) / width, (rect.height() - 2) / height, 1.0)
        if scale <= 0:
            return
        painter.save()
        painter.translate(rect.center())
        painter.scale(scale, scale)
        painter.translate(-width / 2, -height / 2)
        layout.draw(painter, 0.0, 0.0, font, self._text_color())
        painter.restore()

    def _text_is_readable(self) -> bool:
        if self._data is None or self._data.n > TEXT_ITEM_MAX:
            return False
        level = max(self._view_zoom.values(), default=self._zoom_level)
        return self._cell_size * level >= TEXT_CELL_PX_THRESHOLD

    def _set_text_visibility(self, visible: bool) -> None:
        for item in self._text_items:
            item.setVisible(bool(visible))

    def _ensure_text_items(self) -> None:
        if self._text_items or self._data is None or self._texts is None:
            return
        n = self._data.n
        if n > TEXT_ITEM_MAX:
            return
        fs = max(8, min(20, 85 / n))
        font = self._math_font(max(8, round(fs * 1.333)))
        for i in range(n):
            for j in range(n):
                t = self._texts[i][j]
                if not t:
                    continue
                x, y = MARGIN + j * self._cell_size, MARGIN + i * self._cell_size
                ti = MathTextItem(wrap_tex(t, 12), font, self._text_color())
                ti.setZValue(1)
                br = ti.boundingRect()
                # 文字与视角同比缩放，但先按单元格宽高约束 scene-scale，
                # 因而放大后字号会自然变大且不会溢出相邻单元。
                fit = min(0.86 * self._cell_size / max(br.width(), 1e-9),
                          0.72 * self._cell_size / max(br.height(), 1e-9), 1.0)
                ti.setScale(max(0.05, fit))
                ti.setPos(x + (self._cell_size - br.width() * fit) / 2,
                          y + (self._cell_size - br.height() * fit) / 2)
                ti.setData(0, ("matrix-text", i, j))
                ti.setToolTip(self._cell_tooltip_from_text(i, j, t))
                self.addItem(ti)
                self._text_items.append(ti)

    def _mask_text_under_rulers(self, view, strip_w: int, strip_h: int) -> None:
        """Hide whole edge-cell labels that the frozen rulers would cut.

        The viewport rulers are deliberately painted after the scene, so a
        long edge expression could previously remain visible only as a
        misleading fragment.  Mask complete text items instead of clipping
        glyphs; when a matrix is so small that every label touches a ruler,
        keep them all visible rather than producing a blank matrix.
        """
        if self._syncing_edge_text or not self._text_items or view is None:
            return
        readable = self._text_is_readable()
        viewport = view.viewport().rect()
        candidates: list[tuple[QGraphicsItem, bool]] = []
        for item in self._text_items:
            scene_rect = item.mapToScene(item.boundingRect()).boundingRect()
            top_left = view.mapFromScene(scene_rect.topLeft())
            bottom_right = view.mapFromScene(scene_rect.bottomRight())
            pixel_rect = QRectF(top_left, bottom_right).normalized()
            covered = (
                pixel_rect.right() < 0 or pixel_rect.left() > viewport.width()
                or pixel_rect.bottom() < 0 or pixel_rect.top() > viewport.height()
                or pixel_rect.left() < strip_w
                or pixel_rect.right() > viewport.width() - strip_w
                or pixel_rect.top() < strip_h
                or pixel_rect.bottom() > viewport.height() - strip_h
            )
            candidates.append((item, covered))
        has_inner_text = any(not covered for _item, covered in candidates)
        self._syncing_edge_text = True
        try:
            for item, covered in candidates:
                item.setVisible(bool(readable and (not covered or not has_inner_text)))
        finally:
            self._syncing_edge_text = False

    def _add_axes(self, data: MatrixSceneData, cell: float, step: int) -> None:
        # Kept as a compatibility no-op for callers from older integrations.
        return None

    def visible_axis_indices(self, view) -> tuple[list[int], list[int]]:
        """Return readable visible row/column indices for a specific viewport."""
        if self._data is None or self._cell_size <= 0:
            return [], []
        visible = view.mapToScene(view.viewport().rect()).boundingRect()
        clipped = visible.intersected(self._matrix_rect)
        if clipped.isEmpty():
            return [], []
        n = self._data.n
        j0 = max(0, int(math.floor((clipped.left() - MARGIN) / self._cell_size)))
        j1 = min(n - 1, int(math.floor((clipped.right() - MARGIN) / self._cell_size)))
        i0 = max(0, int(math.floor((clipped.top() - MARGIN) / self._cell_size)))
        i1 = min(n - 1, int(math.floor((clipped.bottom() - MARGIN) / self._cell_size)))
        pixels = max(1e-6, self._cell_size * abs(view.transform().m11()))
        step = max(1, int(math.ceil(34.0 / pixels)))
        return list(range(i0, i1 + 1, step)), list(range(j0, j1 + 1, step))

    def draw_viewport_overlay(self, painter, view) -> None:
        """Draw frozen row/column rulers on all four viewport edges."""
        rows, cols = self.visible_axis_indices(view)
        if not rows and not cols:
            return
        viewport = view.viewport().rect()
        font = QFont()
        font.setPointSizeF(8.5)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        strip_h = metrics.height() + 6
        widest = max((metrics.horizontalAdvance(self._axis_labels[i]) for i in rows), default=24)
        strip_w = min(max(34, widest + 10), max(34, viewport.width() // 4))
        if self._dark:
            strip_pen, strip_bg, label_color = (
                QColor(58, 71, 87), QColor(26, 34, 44, 232), QColor(213, 221, 230),
            )
        else:
            strip_pen, strip_bg, label_color = (
                QColor(205, 211, 220), QColor(255, 255, 255, 232), QColor(45, 55, 72),
            )
        painter.setPen(QPen(strip_pen, 1))
        painter.setBrush(strip_bg)
        painter.drawRect(0, 0, viewport.width() - 1, strip_h)
        painter.drawRect(0, viewport.height() - strip_h, viewport.width() - 1, strip_h - 1)
        painter.drawRect(0, strip_h, strip_w, max(0, viewport.height() - 2 * strip_h))
        painter.drawRect(viewport.width() - strip_w, strip_h, strip_w - 1,
                         max(0, viewport.height() - 2 * strip_h))
        painter.setPen(label_color)
        # Do this after calculating the exact ruler footprint but before the
        # labels are painted.  The scene items are then either fully visible
        # or fully hidden, so a zoom/pan frame cannot expose a half glyph.
        self._mask_text_under_rulers(view, strip_w, strip_h)
        for j in cols:
            center = view.mapFromScene(QPointF(
                MARGIN + (j + 0.5) * self._cell_size, self._matrix_rect.top()
            )).x()
            label = self._axis_labels[j]
            width = metrics.horizontalAdvance(label)
            # 四个角属于行标尺，列号不进入侧边条，避免角落文字叠印。
            if strip_w + width / 2 <= center <= viewport.width() - strip_w - width / 2:
                painter.drawText(round(center - width / 2), metrics.ascent() + 3, label)
                painter.drawText(round(center - width / 2),
                                 viewport.height() - metrics.descent() - 3, label)
        for i in rows:
            center = view.mapFromScene(QPointF(
                self._matrix_rect.left(), MARGIN + (i + 0.5) * self._cell_size
            )).y()
            label = self._axis_labels[i]
            y = round(center + (metrics.ascent() - metrics.descent()) / 2)
            # 同理，行号不进入上下列标尺。
            if strip_h <= center <= viewport.height() - strip_h:
                painter.drawText(4, y, label)
                width = metrics.horizontalAdvance(label)
                painter.drawText(viewport.width() - width - 4, y, label)

        # 栅格化大矩阵不创建 O(n²) 个场景文字图元。放大到可读
        # 阈值后，仅在当前视口逐格绘制可见文字，平移时自然按需更新。
        if (
            self._data is not None
            and self._data.n > TEXT_ITEM_MAX
            and self.text_visible_in_view(view)
        ):
            # Keep the rasterized large-matrix path visually consistent with
            # the scene-item path above.  QPainter's default UI font makes
            # ``e``/``k`` look like ordinary sans-serif text, while the
            # dedicated script layout preserves the optical math spacing and
            # explicit pixel-size scaling at every zoom level.
            overlay_pixel_size = self._overlay_text_pixel_size(view)
            painter.save()
            painter.setClipRect(QRectF(
                strip_w, strip_h,
                max(0, viewport.width() - 2 * strip_w),
                max(0, viewport.height() - 2 * strip_h),
            ))
            visible = view.mapToScene(viewport).boundingRect().intersected(self._matrix_rect)
            if not visible.isEmpty():
                n = self._data.n
                j0 = max(0, int(math.floor((visible.left() - MARGIN) / self._cell_size)))
                j1 = min(n - 1, int(math.floor((visible.right() - MARGIN) / self._cell_size)))
                i0 = max(0, int(math.floor((visible.top() - MARGIN) / self._cell_size)))
                i1 = min(n - 1, int(math.floor((visible.bottom() - MARGIN) / self._cell_size)))
                for i in range(i0, i1 + 1):
                    for j in range(j0, j1 + 1):
                        value = self._cell_text_at(i, j)
                        if not value:
                            continue
                        top_left = view.mapFromScene(QPointF(
                            MARGIN + j * self._cell_size,
                            MARGIN + i * self._cell_size,
                        ))
                        bottom_right = view.mapFromScene(QPointF(
                            MARGIN + (j + 1) * self._cell_size,
                            MARGIN + (i + 1) * self._cell_size,
                        ))
                        cell_rect = QRectF(top_left, bottom_right).normalized().adjusted(2, 1, -2, -1)
                        if (
                            cell_rect.left() < strip_w
                            or cell_rect.right() > viewport.width() - strip_w
                            or cell_rect.top() < strip_h
                            or cell_rect.bottom() > viewport.height() - strip_h
                        ):
                            continue
                        # Paint through the same parsed script layout as
                        # small-matrix cells.  Fitting is performed in layout
                        # space, so long expressions are bounded by the cell
                        # instead of exposing clipped braces at its edge.
                        self._draw_overlay_math_text(
                            painter, value, cell_rect, overlay_pixel_size,
                        )
            painter.restore()

    # ---- 内部 ----

    def _cell_texts(self, data: MatrixSceneData) -> list[list[str]]:
        n = data.n
        texts = [[""] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                texts[i][j] = self._cell_text_at(i, j)
        return texts

    def _cell_text_at(self, i: int, j: int) -> str:
        """Format one cell on demand, including raster-only matrices."""
        data = self._data
        if data is None:
            return ""
        raw = data.matrix[i, j] if data.matrix is not None else data.values[i, j]
        if data.mode == "symbolic":
            value = format_elem(raw, "symbolic")
        elif data.mode == "numeric":
            value = format_elem(data.values[i, j], "numeric")
        elif data.smart_labels and (i, j) in data.smart_labels:
            value = sym_pretty(data.smart_labels[(i, j)])
        elif isinstance(raw, sp.Basic) and any(
            getattr(s, "name", str(s)) in {"kx", "k_x"}
            for s in raw.free_symbols
        ):
            value = format_bloch_elem(raw, data.formatter, i == j)
        else:
            value = format_elem(data.values[i, j], "smart", i == j, data.formatter)
        return "" if value == "0" else value

    def mousePressEvent(self, event):
        if self._data is not None and event.button() == Qt.LeftButton:
            cell = self._cell_at_scene_pos(event.scenePos())
            if cell is not None:
                i, j = cell
                # Selection is useful both when text is hidden and when it is
                # readable: it provides a stable visual anchor and lets the
                # main window report the exact row/column without opening a
                # modal dialog or affecting panning.
                self.select_cell(i, j)
                self.cellClicked.emit(i, j)
        super().mousePressEvent(event)
