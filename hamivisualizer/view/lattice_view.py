"""晶格结构视图: 格点 / 键 / 元胞框 / 半无限虚影.

对应 MATLAB draw_lattice §2.1/§6.3:
  - 首胞高亮 (黑实线框 + 浅蓝填充), 其余元胞灰色虚线框
  - NN 键实线红 / NNN 键虚线绿; 半无限虚影键黯淡 (cNN*0.5+0.5 / cNNN*0.4+0.6)
  - 格点圆 (A/B 子格配色) + 白字序号; 虚影格点半透明并带序号
z 分层: 元胞框(0) < 键(1) < 虚影键(1.5) < 虚影格点(2) < 格点(3) < 序号(4)。
"""

from __future__ import annotations

import math
from fractions import Fraction

from PySide6.QtCore import (
    QCoreApplication, QPointF, QRectF, QSizeF, QEvent, QTimer, Qt, Signal,
)
from PySide6.QtGui import (
    QColor, QBrush, QFont, QPainterPath, QPen, QPolygonF,
)
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsRectItem,
    QGraphicsLineItem,
    QGraphicsEllipseItem,
    QGraphicsTextItem,
    QGraphicsPolygonItem,
    QGraphicsProxyWidget,
    QApplication,
    QLineEdit,
    QStyle,
    QStyleOptionGraphicsItem,
)

from ..model.expression import classify_strength_expression
from .rendermodel import LatticeSceneData, Palette


def _q(rgb: tuple, alpha: int = 255) -> QColor:
    return QColor(*[int(c * 255) for c in rgb], alpha)


def _pen(rgb, width: float, style=None) -> QPen:
    """cosmetic 画笔: 宽度恒为像素 (MATLAB LineWidth 语义), 与缩放无关."""
    pen = QPen(_q(rgb), width)
    if style is not None:
        pen.setStyle(style)
    pen.setCosmetic(True)
    return pen


def _blend(rgb: tuple, k: float, base: float) -> tuple:
    """MATLAB 虚影键配色: rgb*k + base (标量加), 返回 0-1 元组."""
    return tuple(
        min(1.0, c * k + base) for c in rgb
    )


def _alpha_over_white(rgb: tuple, alpha: int) -> QColor:
    """把半透明色预先混合到画布底色 (规避部分平台 QBrush alpha 渲染成黑的 bug).

    base 为画布 RGB（亮色=白，深色=深画布色），由调用方传入；保留旧
    函数名以便外部引用，内部按白底预混。
    """
    return _alpha_over(rgb, alpha, (255, 255, 255))


def _alpha_over(rgb: tuple, alpha: int, base: tuple) -> QColor:
    t = alpha / 255.0
    return QColor(*[int(c * 255 * t + b * (1 - t)) for c, b in zip(rgb, base)])


def _fit_text_to_circle(item: QGraphicsTextItem, x: float, y: float,
                        diameter: float, fill: float = 0.78) -> None:
    """Size a label in scene units so it scales naturally with the lattice."""
    br = item.boundingRect()
    if br.width() <= 0 or br.height() <= 0:
        return
    scale = min(diameter * fill / br.width(), diameter * fill / br.height())
    item.setScale(scale)
    item.setPos(x - br.width() * scale / 2, y - br.height() * scale / 2)


def _parse_positive_strength(text: str) -> float:
    """Parse a positive hopping coefficient, including exact fractions.

    The side-panel parameter editor already accepts values such as ``1/3``.
    The on-canvas coefficient field must use the same low-surprise grammar;
    otherwise users can configure a ratio in one place but receive a cryptic
    error after clicking a bond.  Fractions are deliberately delegated to
    :class:`fractions.Fraction`, so arbitrary Python expressions never become
    executable input.
    """
    raw = str(text).strip()
    if not raw:
        raise ValueError("跃迁强度不能为空")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        try:
            value = float(Fraction(raw.replace(" ", "")))
        except (ValueError, ZeroDivisionError, OverflowError):
            raise ValueError("请输入正数或分数，例如 1/3") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError("跃迁强度必须是正的有限数值")
    return value


def _strength_editor_width(editor: QLineEdit, text: str) -> int:
    """Return a compact fixed-pixel width that can actually show ``text``.

    Coefficient editors are embedded in a scene with
    ``ItemIgnoresTransformations``.  Their width therefore cannot be left to
    scene scaling, and the old constant-width field clipped values such as
    ``0.33333333`` immediately after a fraction was committed.  Measure the
    rendered font and reserve room for the border/padding while keeping a
    bounded rail width for dense lattices.
    """
    metrics = editor.fontMetrics()
    # The stylesheet contributes 7 px horizontal padding (6 px while focused)
    # on each side, plus a two-pixel border.  A small extra cushion prevents
    # the final digit from touching the rounded frame on high-DPI backends.
    measured = metrics.horizontalAdvance(str(text)) + 24
    # Keep a generous upper bound for large UI scales; the rail still uses a
    # single column, so a wider field does not add visual clutter or extra
    # leader lines.  The 260 px cap is intentional: at 180% UI scale a
    # 9-character value can measure about 220 px on the bundled fallback font.
    return max(68, min(260, int(math.ceil(measured))))


class _EditableSiteItem(QGraphicsEllipseItem):
    """A unit-cell handle that snaps unless Alt is held."""

    def __init__(self, scene, index: int, x: float, y: float, radius: float = 0.23):
        super().__init__(-radius, -radius, 2 * radius, 2 * radius)
        self.editor_scene = scene
        self.site_index = index
        self.setPos(x, -y)
        self.setFlags(
            QGraphicsItem.ItemIsMovable
            | QGraphicsItem.ItemIsSelectable
            | QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setCursor(Qt.OpenHandCursor)
        self.setZValue(20)
        self._press_pos = QPointF()

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and self.scene() is not None:
            p = QPointF(value)
            if self.editor_scene.snap_enabled and not (
                QApplication.keyboardModifiers() & Qt.AltModifier
            ):
                p = self.editor_scene.snap_position(self.site_index, p, self._press_pos)
            # Snapping is optional, but model validity is not.  Route both
            # magnetic and free-drag paths through the same collision guard so
            # Alt/disabled-snap drags cannot stack two sites at one coordinate.
            return self.editor_scene.safe_edit_position(
                self.site_index, p, fallback=self._press_pos,
            )
        return super().itemChange(change, value)

    def paint(self, painter, option, widget=None):
        """Paint a font-independent vector index that follows the view scale."""
        # QGraphicsEllipseItem's default selected-state decoration is a large
        # white halo on some Fusion/Windows styles.  It visually overwhelms
        # the atom and can be mistaken for a second node.  Suppress that
        # style decoration and draw a small, explicit blue focus ring below.
        clean_option = QStyleOptionGraphicsItem(option)
        clean_option.state &= ~QStyle.State_Selected
        super().paint(painter, clean_option, widget)
        if self.isSelected():
            ring = QPen(QColor("#7dd3fc"), 1.4)
            ring.setCosmetic(True)
            painter.save()
            painter.setPen(ring)
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(self.rect().adjusted(0.035, 0.035, -0.035, -0.035))
            painter.restore()
        # The offscreen Qt font backend can return tofu boxes even for ASCII
        # digits.  A tiny seven-segment vector alphabet is deterministic on
        # every platform and remains sharp while zooming.
        segment_points = {
            "a": ((0.15, 0.0), (0.85, 0.0)),
            "b": ((1.0, 0.15), (1.0, 0.85)),
            "c": ((1.0, 1.15), (1.0, 1.85)),
            "d": ((0.15, 2.0), (0.85, 2.0)),
            "e": ((0.0, 1.15), (0.0, 1.85)),
            "f": ((0.0, 0.15), (0.0, 0.85)),
            "g": ((0.15, 1.0), (0.85, 1.0)),
        }
        digit_segments = {
            "0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg",
            "4": "bcfg", "5": "acdfg", "6": "acdefg", "7": "abc",
            "8": "abcdefg", "9": "abcdfg",
        }
        # The numerical site index stays zero-based internally; visible edit
        # handles follow the same one-based convention as normal lattice
        # labels, matrix rulers, and status messages.
        text = str(self.site_index + 1)
        path = QPainterPath()
        for column, digit in enumerate(text):
            offset = column * 1.35
            for segment in digit_segments.get(digit, ""):
                (x1, y1), (x2, y2) = segment_points[segment]
                path.moveTo(offset + x1, y1)
                path.lineTo(offset + x2, y2)
        bounds = path.boundingRect()
        if bounds.height() <= 0:
            return
        diameter = self.rect().width()
        scale = min(diameter * 0.58 / max(bounds.width(), 0.5),
                    diameter * 0.58 / bounds.height())
        painter.save()
        painter.translate(self.rect().center())
        painter.scale(scale, scale)
        painter.translate(-bounds.center())
        pen = QPen(QColor("white"), 0.15)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)
        painter.restore()

    def shape(self):
        """Use the visible disc as the hit target, not the cosmetic pen box.

        ``QGraphicsEllipseItem.shape()`` includes the pen when resolving scene
        hits.  On Qt 6 a cosmetic two-pixel pen can inflate that shape by about
        one *scene unit* at a zoomed view, even though the painted outline is
        only two device pixels wide.  Nearby sites then overlap in the hit
        tester: clicking one handle may select or drag the neighbour.  The
        outline is presentation-only, so the interaction shape should be the
        actual disc and remain stable under zoom.
        """
        path = QPainterPath()
        path.addEllipse(self.rect())
        return path

    def mousePressEvent(self, event):
        self._press_pos = self.pos()
        self.setFlag(QGraphicsItem.ItemIsFocusable, True)
        self.setFocus(Qt.MouseFocusReason)
        self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.setCursor(Qt.OpenHandCursor)
        # QGraphicsItem may receive a sub-pixel position correction during a
        # plain click (especially after a view zoom). Treat that as a click,
        # not as a site edit; otherwise merely selecting a site can trigger a
        # rebuild and make the clicked element appear to disappear.
        moved = (self.pos() - self._press_pos).manhattanLength() >= 0.02
        if not moved:
            self.setPos(self._press_pos)
        if moved:
            anchor_x, anchor_y = self.editor_scene.edit_anchor_offset
            # Alt temporarily disables magnetic snapping, not the model's
            # validity rules.  Clamp the final coordinates before handing
            # them to the panel so a fast drag cannot create an invalid cell.
            constrained = self.editor_scene.safe_edit_position(
                self.site_index, self.pos(), fallback=self._press_pos,
            )
            if constrained != self.pos():
                self.setPos(constrained)
            self.editor_scene.siteMoved.emit(
                self.site_index,
                float(self.pos().x() - anchor_x),
                float(-self.pos().y() - anchor_y),
            )
        else:
            self.editor_scene.activate_site(self.site_index)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            if self.editor_scene.hop_creation_mode:
                self.editor_scene.set_hop_creation_mode(False)
                event.accept()
                return
            self.setPos(self._press_pos)
            self.editor_scene.editSelectionChanged.emit("已取消本次拖动，格点已回到拖动前位置")
            event.accept()
            return
        super().keyPressEvent(event)


class _GhostSiteItem(QGraphicsEllipseItem):
    """A read-only periodic image that becomes a bond endpoint on demand.

    Ghosts deliberately do not accept mouse buttons during normal browsing,
    so they never steal a pan/selection click.  The explicit bond tool toggles
    their hit targets on; a click then carries the primitive-site identity and
    relative cell offset to :class:`LatticeView` for an unambiguous inter-cell
    row.
    """

    def __init__(self, scene, x: float, y: float, radius: float,
                 source_site: int, cell_dx: int, cell_dy: int):
        super().__init__(-radius, -radius, 2 * radius, 2 * radius)
        self.editor_scene = scene
        self.source_site = int(source_site)
        self.cell_dx = int(cell_dx)
        self.cell_dy = int(cell_dy)
        self.setPos(float(x), -float(y))
        self.setBrush(QBrush(_q((0.55, 0.60, 0.65))))
        self.setPen(_pen((0.42, 0.47, 0.53), 0.6))
        self.setOpacity(0.45)
        self.setZValue(2.1)
        # Let the view distinguish this deliberate endpoint hit from empty
        # canvas before arming its manual pan gesture.  The item remains
        # inert in normal browsing because its accepted mouse buttons are
        # still ``NoButton`` until the explicit bond tool is enabled.
        self.setData(0, "ghost-endpoint")
        # The cursor advertises affordance only while the explicit tool is
        # armed; ordinary browsing should keep the neutral arrow over
        # read-only periodic images.
        self.setCursor(Qt.ArrowCursor)
        self.setAcceptedMouseButtons(Qt.NoButton)

    def set_interaction_enabled(self, enabled: bool) -> None:
        self.setAcceptedMouseButtons(Qt.LeftButton if enabled else Qt.NoButton)
        if enabled:
            self.setCursor(Qt.CrossCursor)
            self.setPen(_pen((0.25, 0.68, 0.95), 1.2))
            self.setOpacity(0.72)
            self.setToolTip(
                f"周期像：格点 {self.source_site + 1}，"
                f"相对元胞偏移 ({self.cell_dx:+d}, {self.cell_dy:+d})；"
                "添加跃迁工具开启时可作为端点"
            )
        else:
            self.setCursor(Qt.ArrowCursor)
            self.setPen(_pen((0.42, 0.47, 0.53), 0.6))
            self.setOpacity(0.45)
            self.setToolTip("")

    def mousePressEvent(self, event):  # noqa: N802 - Qt override
        if self.editor_scene.hop_creation_mode and event.button() == Qt.LeftButton:
            self.editor_scene.activate_ghost(
                self.source_site, self.cell_dx, self.cell_dy,
            )
            event.accept()
            return
        event.ignore()


class _EditableHopGuide(QGraphicsLineItem):
    """Wide, quiet hit target for one editable hopping definition.

    Dense lattices should not turn every bond into a permanent text field.
    This guide follows the source-cell bond and exposes a clear hover state;
    clicking it opens the single coefficient editor for that hop.
    """

    def __init__(self, scene, row: int, x1: float, y1: float,
                 x2: float, y2: float):
        super().__init__(x1, y1, x2, y2)
        self.editor_scene = scene
        self.row = int(row)
        self.setAcceptedMouseButtons(Qt.LeftButton)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setZValue(14)
        self.setData(0, "hopping-guide")
        self.set_selected(False)

    def set_selected(self, selected: bool) -> None:
        if selected:
            self.setPen(_pen((0.20, 0.68, 0.98), 1.6, Qt.DashLine))
            self.setOpacity(0.92)
            return
        # Keep a generous cosmetic hit area while making the guide almost
        # invisible. The physical NN/NNN bond below stays the visual source
        # of truth; this layer exists only to make direct editing discoverable.
        pen = _pen((0.32, 0.58, 0.76), 8.0)
        # Hit testing uses the line item's shape, not its alpha. Keep the
        # idle target truly invisible so overlapping Kagome bonds do not
        # create a translucent fan around the editable cell.
        pen.setColor(QColor(82, 148, 194, 0))
        self.setPen(pen)
        self.setOpacity(1.0)

    def hoverEnterEvent(self, event):  # noqa: N802 - Qt override
        self.setPen(_pen((0.20, 0.68, 0.98), 1.5, Qt.DashLine))
        self.setOpacity(0.95)
        # In dense coefficient mode the leader is progressive-disclosure
        # UI: hovering the physical bond should reveal the matching field
        # without requiring a click (and without rebuilding the scene).
        self.editor_scene._set_hovered_hop_row(self.row)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):  # noqa: N802 - Qt override
        self.set_selected(self.row == self.editor_scene.active_hop_row)
        self.editor_scene._set_hovered_hop_row(None)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):  # noqa: N802 - Qt override
        if event.button() == Qt.LeftButton:
            self.editor_scene.activate_hop_editor(self.row)
            event.accept()
            return
        super().mousePressEvent(event)


class _HopStrengthEdit(QLineEdit):
    """Fixed-pixel bond editor that asks its scene to reveal it on focus."""

    focusEntered = Signal()
    focusLeft = Signal()
    hoverEntered = Signal()
    hoverLeft = Signal()
    owner_scene = None

    def enterEvent(self, event):  # noqa: N802 - Qt override
        super().enterEvent(event)
        self.hoverEntered.emit()

    def leaveEvent(self, event):  # noqa: N802 - Qt override
        super().leaveEvent(event)
        self.hoverLeft.emit()

    def focusInEvent(self, event):  # noqa: N802 - Qt override
        super().focusInEvent(event)
        self.focusEntered.emit()

    def focusOutEvent(self, event):  # noqa: N802 - Qt override
        super().focusOutEvent(event)
        self.focusLeft.emit()

    def resizeEvent(self, event):  # noqa: N802 - Qt override
        super().resizeEvent(event)
        # UI-scale changes resize the fixed-pixel field after the view's
        # transform has already been fitted. Reflow on the next event-loop
        # turn so the new full width/height participates in collision and
        # boundary calculations (including the bottom border).
        scene = self.owner_scene
        if scene is not None:
            QTimer.singleShot(0, scene._reflow_editors)


class LatticeView(QGraphicsScene):
    """晶格结构 (QGraphicsScene 自绘)."""

    siteMoved = Signal(int, float, float)
    siteAddRequested = Signal(float, float)
    siteDeleteRequested = Signal(int)
    hoppingRequested = Signal(int, int)
    hoppingRequestedWithOffset = Signal(int, int, int, int)
    hoppingStrengthEdited = Signal(int, float)
    editSelectionChanged = Signal(str)
    hopCreationModeChanged = Signal(bool)
    siteCreationModeChanged = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: LatticeSceneData | None = None
        self.edit_mode = False
        self.snap_step = 0.25
        self.snap_enabled = True
        self._edit_sites: list[tuple[float, float, str]] = []
        # Immutable geometry captured on entry to an edit session.  The
        # current editable sites are replaced after every rebuild, so they
        # alone cannot provide a magnetic way back to the original graphene
        # construction once the user has completed one drag.
        self._snap_reference_sites: list[tuple[float, float]] = []
        # Physical translation of the real finite-cell copy used as the
        # editor's visual anchor. Site-table coordinates always stay local to
        # the primitive cell; this only prevents the handles from floating at
        # (0, 0) when a disk/hexagon mask excludes that cell.
        self._edit_anchor_offset = (0.0, 0.0)
        # When the controller supplies the actual Bravais vectors, interactive
        # drags are kept inside that primitive cell.  Stand-alone scenes used
        # by plug-ins/tests may omit vectors and intentionally remain free.
        self._edit_cell_constrained = False
        self._edit_items: dict[int, _EditableSiteItem] = {}
        self._ghost_items: list[_GhostSiteItem] = []
        self._edit_hops: list[dict] = []
        self._show_all_hop_editors = False
        # Dense models default to a nearest-neighbour editing layer.  The
        # physical long-range terms remain in the model and can be explicitly
        # revealed from the edit toolbar, rather than turning every editing
        # session into a green crossing-line mesh.
        self._show_edit_details = False
        self._active_hop_row: int | None = None
        # A dense right-hand coefficient rail keeps all values editable while
        # showing only the leader belonging to the bond under the pointer.
        # This transient row is intentionally separate from the committed
        # active row so moving the pointer never rebuilds or mutates a model.
        self._hovered_hop_row: int | None = None
        self._focused_hop_row: int | None = None
        # QGraphicsProxyWidget has QObject ownership semantics in PySide6;
        # keeping explicit references prevents an input proxy (and its guide)
        # from being garbage-collected after the first mouse event.
        self._edit_proxies: list[QGraphicsProxyWidget] = []
        self._edit_guides: list[QGraphicsLineItem] = []
        self._edit_leaders: list[QGraphicsLineItem] = []
        # Compact models can contain adjacent collinear bonds (for example
        # SSH's intracell and intercell links).  A small relation badge above
        # each right-hand editor makes those two coefficients unambiguous
        # without adding another table column or a second input field for the
        # same relation.
        self._edit_relation_badges: list[tuple[QGraphicsTextItem,
                                                 QGraphicsProxyWidget]] = []
        # Magnetic edit-grid dots are a presentation layer, not lattice
        # sites.  Keep explicit references so changing the toolbar spacing
        # can redraw only this layer without rebuilding editors or topology.
        self._grid_items: list[QGraphicsEllipseItem] = []
        self._grid_visible = True
        # Two-segment dashed leaders connect each displaced editor to its bond
        # midpoint. Keep this separate from the public anchor tuples so
        # existing callers continue to receive (proxy, x, y).
        self._edit_leader_links: list[tuple[QGraphicsLineItem, QGraphicsLineItem,
                                              float, float, QGraphicsProxyWidget]] = []
        self._edit_proxy_anchors: list[tuple[QGraphicsProxyWidget, float, float]] = []
        self._last_editor_layout: tuple[float, list] = (1.0, [])
        self._cell_vectors = ((1.0, 0.0), (0.0, 1.0))
        self._hop_start: int | None = None
        self._hop_start_endpoint: tuple[int, int, int] | None = None
        # Creating a bond is deliberately an explicit tool, not a side
        # effect of selecting a site.  The old two-click overload was easy
        # to trigger while inspecting/dragging a lattice and made ordinary
        # clicks look as though geometry or bonds had disappeared.
        self._hop_creation_mode = False
        # Adding a site also changes topology and is therefore explicit.  A
        # double click on a busy lattice is far too easy to perform by
        # accident; the tool uses one intentional click on blank canvas.
        self._site_creation_mode = False
        self._dark = False
        self._blend_base: tuple = (255, 255, 255)
        self._show_nn = True
        self._show_nnn = True
        self._show_ghosts = True
        self._show_cells = True
        self._site_radius = 0.18

    def set_display_options(self, *, nn: bool = True, nnn: bool = True,
                            ghosts: bool = True, cells: bool = True,
                            redraw: bool = True):
        """Set view-only layers without changing the lattice calculation."""
        new = tuple(map(bool, (nn, nnn, ghosts, cells)))
        old = (self._show_nn, self._show_nnn, self._show_ghosts, self._show_cells)
        self._show_nn, self._show_nnn, self._show_ghosts, self._show_cells = new
        if redraw and new != old and self._data is not None:
            self.set_data(self._data)

    def set_theme(self, dark: bool):
        """切换晶格视图明暗：首胞/虚影填充按画布底色预混，重新渲染。"""
        self._dark = bool(dark)
        self._blend_base = (16, 22, 29) if dark else (255, 255, 255)
        if self._data is not None:
            self.set_data(self._data)

    def set_snap_enabled(self, enabled: bool):
        """Enable/disable magnetic snapping without rebuilding the model."""
        self.snap_enabled = bool(enabled)

    @property
    def grid_visible(self) -> bool:
        """Whether the editable canvas shows magnetic grid nodes."""
        return self._grid_visible

    def set_grid_visible(self, enabled: bool) -> None:
        """Show/hide the edit-grid dots without changing snap behaviour."""
        enabled = bool(enabled)
        if enabled == self._grid_visible:
            return
        self._grid_visible = enabled
        self._redraw_snap_grid()

    def set_snap_step(self, step: float) -> None:
        """Update snap spacing and redraw the visible edit grid immediately."""
        try:
            value = float(step)
        except (TypeError, ValueError, OverflowError):
            return
        if not math.isfinite(value):
            return
        self.snap_step = max(0.001, min(10.0, value))
        self._redraw_snap_grid()

    def _clear_snap_grid(self) -> None:
        for item in self._grid_items:
            self.removeItem(item)
        self._grid_items.clear()

    def _bravais_grid_counts(self, step: float | None = None) -> tuple[int, int]:
        """Return fractional subdivisions for an oblique editing grid.

        ``snap_step`` is a physical target spacing, while an oblique cell has
        no single global x/y interval.  Subdivide each primitive vector by
        approximately that spacing instead.  The same counts are consumed by
        drawing and snapping, so a visible dot is always a valid magnetic
        target rather than a decorative point the editor cannot reach.
        """
        value = max(0.001, float(self.snap_step if step is None else step))
        (a1x, a1y), (a2x, a2y) = self._cell_vectors
        n1 = max(1, int(round(math.hypot(a1x, a1y) / value)))
        n2 = max(1, int(round(math.hypot(a2x, a2y) / value)))
        # A nearest integer count is not enough for a basis with rational
        # fractional coordinates.  For example, graphene's B site is at
        # ``(u, v)=(1/2, 1/3)``.  A visually plausible 7×7 net would put that
        # site between targets, which is exactly the ``格点不在网格点上``
        # impression users reported.  Choose the nearest multiples of the
        # simple basis denominators, so the visible dots and the snap helper
        # share an exact target for every shipped lattice.
        required_1, required_2 = self._required_bravais_denominators()

        def nearest_multiple(target: int, divisor: int) -> int:
            divisor = max(1, int(divisor))
            lower = max(divisor, (int(target) // divisor) * divisor)
            upper = lower + divisor
            return lower if target - lower <= upper - target else upper

        n1 = nearest_multiple(n1, required_1)
        n2 = nearest_multiple(n2, required_2)
        # Keep the presentation bounded on an accidentally tiny spacing while
        # retaining a deterministic, near-square number of targets.  Reduce
        # one whole denominator-sized block at a time; otherwise the cap could
        # silently destroy the alignment guarantee just established above.
        while n1 * n2 > 1800:
            if n1 >= n2:
                n1 = max(required_1, n1 - required_1)
            else:
                n2 = max(required_2, n2 - required_2)
        return n1, n2

    def _required_bravais_denominators(self) -> tuple[int, int]:
        """Find small rational basis fractions that the edit net must contain.

        The grid density remains controlled by ``snap_step``; this helper only
        protects exact, simple fractions already present in the editable basis
        (halves, thirds, quarters).  Arbitrary user-dragged coordinates
        are deliberately ignored unless they are within a tight tolerance of a
        small rational, so one free-form move cannot unexpectedly change the
        entire grid density.  The same counts are then consumed by
        :meth:`_snap_local_point` and :meth:`_draw_snap_grid`.
        """
        if not self._edit_cell_constrained or not self._edit_sites:
            return 1, 1
        (a1x, a1y), (a2x, a2y) = self._cell_vectors
        det = a1x * a2y - a1y * a2x
        if abs(det) < 1e-12:
            return 1, 1
        required_1 = required_2 = 1
        # Denominators beyond 4 describe user-specific fine geometry rather
        # than a built-in basis and would make a coarse grid unexpectedly
        # dense.  They remain reachable through the exact-site alignment ring.
        max_denominator = 4
        tolerance = 1.0e-7
        for sx, sy, _sub in self._edit_sites:
            x, y = float(sx), float(sy)
            u = (x * a2y - y * a2x) / det
            v = (a1x * y - a1y * x) / det
            if not (-tolerance <= u < 1.0 + tolerance
                    and -tolerance <= v < 1.0 + tolerance):
                continue
            fu = Fraction(u).limit_denominator(max_denominator)
            fv = Fraction(v).limit_denominator(max_denominator)
            if abs(float(fu) - u) <= tolerance:
                required_1 = math.lcm(required_1, fu.denominator)
            if abs(float(fv) - v) <= tolerance:
                required_2 = math.lcm(required_2, fv.denominator)
        return required_1, required_2

    def _scene_to_local(self, point: QPointF) -> tuple[float, float]:
        """Convert a scene point into the editable cell's local coordinates.

        The editor draws the selected primitive cell at ``_edit_anchor_offset``
        and flips the physical y axis for Qt. Keeping that pair of operations
        in one helper prevents the grid, drag constraint and site-creation
        tools from silently adopting different origins or signs.
        """
        anchor_x, anchor_y = self._edit_anchor_offset
        point = QPointF(point)
        return (
            float(point.x() - anchor_x),
            float(-point.y() - anchor_y),
        )

    def _local_to_scene(self, x: float, y: float) -> QPointF:
        """Convert editable local physical coordinates into scene space."""
        anchor_x, anchor_y = self._edit_anchor_offset
        return QPointF(
            float(anchor_x + x),
            float(-(anchor_y + y)),
        )

    def _local_to_fractional(self, x: float, y: float) -> tuple[float, float] | None:
        """Return Bravais ``(u, v)`` coordinates for a local point.

        ``None`` is returned for unconstrained/degenerate stand-alone scenes.
        Callers can therefore share this conversion without applying oblique
        cell math to plug-in scenes that intentionally use Cartesian values.
        """
        if not self._edit_cell_constrained:
            return None
        (a1x, a1y), (a2x, a2y) = self._cell_vectors
        det = a1x * a2y - a1y * a2x
        if abs(det) < 1e-12:
            return None
        return (
            (float(x) * a2y - float(y) * a2x) / det,
            (a1x * float(y) - a1y * float(x)) / det,
        )

    def _fractional_to_local(self, u: float, v: float) -> tuple[float, float]:
        """Map Bravais fractional coordinates to local physical coordinates."""
        (a1x, a1y), (a2x, a2y) = self._cell_vectors
        return (
            float(u) * a1x + float(v) * a2x,
            float(u) * a1y + float(v) * a2y,
        )

    def _snap_local_point(self, x: float, y: float, step: float) -> tuple[float, float]:
        """Round a local physical point in the same coordinates as the grid."""
        fractional = self._local_to_fractional(x, y)
        if fractional is not None:
            u, v = fractional
            n1, n2 = self._bravais_grid_counts(step)
            u = round(u * n1) / n1
            v = round(v * n2) / n2
            return self._fractional_to_local(u, v)
        return round(x / step) * step, round(y / step) * step

    def _snap_grid_bounds(self) -> QRectF:
        """Return the compact visual work area for snap targets.

        Only the primitive-cell copy represented by the editable handles can
        be changed.  Drawing dots over the entire periodic canvas (including
        the coefficient rail and read-only ghost copies) makes a dense model
        look like an unrelated second lattice.  Keep a small scene margin
        around the actual Bravais parallelogram; unconstrained stand-alone
        scenes retain the historical full-scene fallback for plug-ins.
        """
        scene_rect = QRectF(self.sceneRect())
        if not self._edit_cell_constrained or scene_rect.isEmpty():
            return scene_rect
        (a1x, a1y), (a2x, a2y) = self._cell_vectors
        corners = [
            self._local_to_scene(0.0, 0.0),
            self._local_to_scene(a1x, a1y),
            self._local_to_scene(a2x, a2y),
            self._local_to_scene(a1x + a2x, a1y + a2y),
        ]
        left = min(point.x() for point in corners)
        right = max(point.x() for point in corners)
        top = min(point.y() for point in corners)
        bottom = max(point.y() for point in corners)
        # The margin keeps a row of targets visible near each cell edge while
        # remaining small enough that the read-only neighbouring copies stay
        # visually quiet.  It scales with the chosen density but is bounded
        # for very fine/coarse user-entered spacing.
        padding = max(0.35, min(1.5, 2.0 * max(0.001, self.snap_step)))
        focus = QRectF(
            left - padding, top - padding,
            max(0.0, right - left) + 2.0 * padding,
            max(0.0, bottom - top) + 2.0 * padding,
        )
        # ``QRectF.contains`` treats its right/bottom edge as closed in
        # principle, but values produced by the Bravais-vector arithmetic can
        # land a few ulps below that edge (for example ``-0.9999999999999999``
        # versus ``-1.0``).  Keep a tiny numerical cushion so the visual dot
        # and the advertised work area agree at the boundary without making
        # the grid meaningfully larger.
        bounded = focus.intersected(scene_rect)
        epsilon = 1.0e-9
        return bounded.adjusted(-epsilon, -epsilon, epsilon, epsilon)

    def _draw_snap_grid(self) -> None:
        """Draw quiet, clickable-looking snap targets behind the lattice.

        The points share the same Cartesian origin and interval used by
        :meth:`snap_position`, so a user can see exactly where a dragged site
        will land.  They are limited to the compact editable-cell work area;
        very large scenes coarsen *only the decoration* to keep Qt responsive,
        and the actual snap interval remains unchanged.
        """
        self._clear_snap_grid()
        if not self.edit_mode or not self._grid_visible or self._data is None:
            return
        rect = self._snap_grid_bounds()
        if not rect.isValid() or rect.isEmpty():
            return
        step = max(0.001, float(self.snap_step))
        anchor_x, anchor_y = self._edit_anchor_offset
        dark = self._dark
        color = QColor(72, 105, 132, 120) if dark else QColor(115, 145, 168, 115)
        if self._edit_cell_constrained:
            (a1x, a1y), (a2x, a2y) = self._cell_vectors
            cell_det = a1x * a2y - a1y * a2x
        else:
            cell_det = 0.0
        if abs(cell_det) >= 1e-12:
            # Oblique cells use their own fractional coordinate net.  A
            # Cartesian dot field inside a slanted parallelogram is visually
            # misleading: its rows do not follow either Bravais direction.
            n1, n2 = self._bravais_grid_counts(step)
            radius = min(0.035, max(0.012, step * 0.10))
            for i in range(n1):
                u = i / n1
                for j in range(n2):
                    v = j / n2
                    point = self._local_to_scene(
                        *self._fractional_to_local(u, v)
                    )
                    x, y = point.x(), point.y()
                    item = QGraphicsEllipseItem(x - radius, y - radius,
                                                2 * radius, 2 * radius)
                    item.setBrush(QBrush(color))
                    item.setPen(Qt.NoPen)
                    item.setZValue(-1.0)
                    item.setAcceptedMouseButtons(Qt.NoButton)
                    item.setData(0, "snap-grid-node")
                    item.setData(1, "bravais-grid")
                    self.addItem(item)
                    self._grid_items.append(item)
        else:
            nx = int(math.floor((rect.right() - anchor_x) / step)
                     - math.ceil((rect.left() - anchor_x) / step) + 1)
            # Scene y is inverted relative to physical y.  The interval below
            # is nevertheless identical to snap_position's ``point.y + anchor_y``.
            y_min = math.ceil((rect.top() + anchor_y) / step)
            y_max = math.floor((rect.bottom() + anchor_y) / step)
            ny = max(0, y_max - y_min + 1)
            render_step = step
            point_count = max(0, nx) * ny
            if point_count > 1800:
                multiplier = int(math.ceil(math.sqrt(point_count / 1800.0)))
                render_step = step * max(1, multiplier)
                nx = int(math.floor((rect.right() - anchor_x) / render_step)
                         - math.ceil((rect.left() - anchor_x) / render_step) + 1)
                y_min = math.ceil((rect.top() + anchor_y) / render_step)
                y_max = math.floor((rect.bottom() + anchor_y) / render_step)
                ny = max(0, y_max - y_min + 1)
            if nx <= 0 or ny <= 0:
                return
            radius = min(0.035, max(0.012, render_step * 0.10))
            x_start = math.ceil((rect.left() - anchor_x) / render_step)
            x_stop = math.floor((rect.right() - anchor_x) / render_step)
            for ix in range(x_start, x_stop + 1):
                x = anchor_x + ix * render_step
                for iy in range(y_min, y_max + 1):
                    y = iy * render_step - anchor_y
                    item = QGraphicsEllipseItem(x - radius, y - radius,
                                                2 * radius, 2 * radius)
                    item.setBrush(QBrush(color))
                    item.setPen(Qt.NoPen)
                    item.setZValue(-1.0)
                    item.setAcceptedMouseButtons(Qt.NoButton)
                    item.setData(0, "snap-grid-node")
                    item.setData(1, "cartesian-grid")
                    self.addItem(item)
                    self._grid_items.append(item)

        # The Cartesian fallback grid is intentionally simple and predictable
        # for custom rectangular cells.  Oblique presets (honeycomb, Kagome,
        # triangular) contain basis sites such as ``(sqrt(3)/2, 1/2)`` that
        # are not multiples of a decimal Cartesian interval.  Showing only
        # the fallback dots therefore makes a real lattice site look as if it
        # floats between targets, even though the magnetic snap code already
        # knows the exact basis coordinate.  Add a quiet, non-interactive
        # halo at each current editable site that is not already on a regular
        # dot.  It is a visual snap target, not a second site and never enters
        # hit testing; the editable circle remains on top of it.
        if self._edit_cell_constrained and (self._edit_sites or self._snap_reference_sites):
            halo_color = QColor(72, 105, 132, 150) if dark else QColor(82, 113, 137, 135)
            halo_pen = QPen(halo_color, 0.9)
            halo_pen.setCosmetic(True)
            reference_pen = QPen(halo_color, 0.8, Qt.DashLine)
            reference_pen.setCosmetic(True)
            # Keep the alignment cue just outside the atom instead of using a
            # large selection-like halo.  The old minimum radius (0.215) was
            # enough for an irrational honeycomb basis, but drawing no cue at
            # all for sites that happened to land on a Cartesian dot made the
            # regular sites look detached from the visible grid.  Every exact
            # target now gets the same compact ring; the site itself remains
            # the only interactive item.
            halo_radius = max(0.205, min(0.285, self._site_radius + 0.03))
            # Current sites and the immutable geometry captured on entry to
            # edit mode are both legitimate magnetic targets.  The latter is
            # intentionally dashed: after a drag it gives the user a visible
            # route back to the original graphene/Kagome position, instead of
            # requiring them to remember coordinates or guess where to drop.
            targets = [
                (float(sx), float(sy), "site-anchor")
                for sx, sy, _sub in self._edit_sites
            ] + [
                (float(sx), float(sy), "baseline-anchor")
                for sx, sy in self._snap_reference_sites
            ]
            seen_targets: set[tuple[int, int]] = set()
            for sx, sy, target_kind in targets:
                target = self._local_to_scene(float(sx), float(sy))
                tx, ty = target.x(), target.y()
                if not rect.contains(target):
                    continue
                if abs(cell_det) >= 1e-12:
                    local_x, local_y = self._scene_to_local(target)
                    u, v = self._local_to_fractional(local_x, local_y)
                    if not (-1e-9 <= u < 1.0 - 1e-9
                            and -1e-9 <= v < 1.0 - 1e-9):
                        continue
                target_key = (round(tx * 1.0e8), round(ty * 1.0e8))
                if target_key in seen_targets:
                    continue
                seen_targets.add(target_key)
                halo = QGraphicsEllipseItem(
                    tx - halo_radius, ty - halo_radius,
                    2 * halo_radius, 2 * halo_radius,
                )
                halo.setBrush(Qt.NoBrush)
                halo.setPen(reference_pen if target_kind == "baseline-anchor" else halo_pen)
                halo.setZValue(-0.8)
                halo.setAcceptedMouseButtons(Qt.NoButton)
                halo.setData(0, "snap-grid-node")
                halo.setData(1, target_kind)
                self.addItem(halo)
                self._grid_items.append(halo)

    def _redraw_snap_grid(self) -> None:
        """Refresh only grid dots, preserving selection and editor focus."""
        if self._data is not None:
            self._draw_snap_grid()

    def set_snap_reference_sites(self, sites) -> None:
        """Set immutable local-coordinate snap targets for this edit session.

        ``sites`` deliberately accepts both persisted site dictionaries and
        lightweight ``(x, y, ...)`` rows, so restoring a saved model and
        programmatic scenes share the same behaviour.  These references are
        visual assistance only; they never alter the Hamiltonian by
        themselves.
        """
        references: list[tuple[float, float]] = []
        for site in sites or ():
            try:
                if isinstance(site, dict):
                    x, y = site["x"], site["y"]
                else:
                    x, y = site[0], site[1]
                x, y = float(x), float(y)
            except (IndexError, KeyError, TypeError, ValueError):
                continue
            if math.isfinite(x) and math.isfinite(y):
                references.append((x, y))
        self._snap_reference_sites = references
        # Restoring a baseline updates this list after the model rebuild has
        # already painted the scene.  Refresh only the decorative target
        # layer so the dashed return-to-baseline markers appear immediately;
        # no Hamiltonian or editor proxy is rebuilt.
        if self._data is not None and self.edit_mode:
            self._redraw_snap_grid()

    @property
    def active_hop_row(self) -> int | None:
        return self._active_hop_row

    @property
    def hovered_hop_row(self) -> int | None:
        """Row whose physical bond/editor is currently under the pointer."""
        return self._hovered_hop_row

    def handle_viewport_hover(self, view, point) -> None:
        """Resolve a rail field from the *screen* pointer position.

        ``QGraphicsProxyWidget`` editors intentionally ignore the view
        transform so their controls stay a comfortable, fixed pixel size.
        Consequently their scene bounding rectangles are much larger than
        their on-screen footprints and Qt's normal scene hover hit-test can
        report the wrong overlapping proxy on a dense rail.  The view knows
        the actual device position, so use that as a small, deterministic
        hit map for hover disclosure.  This is only a transient presentation
        state: it never changes the model or rebuilds the scene.
        """
        if not self._edit_proxy_anchors:
            return
        hit_row = None
        px, py = int(point.x()), int(point.y())
        for proxy, _x, _y in self._edit_proxy_anchors:
            if not proxy.isVisible():
                continue
            widget = proxy.widget()
            if widget is None:
                continue
            top_left = view.mapFromScene(proxy.pos())
            if (top_left.x() <= px < top_left.x() + widget.width()
                    and top_left.y() <= py < top_left.y() + widget.height()):
                row = proxy.data(1)
                if row is not None:
                    hit_row = int(row)
                break
        self._set_hovered_hop_row(hit_row)

    @property
    def show_all_hop_editors(self) -> bool:
        return self._show_all_hop_editors

    @property
    def show_edit_details(self) -> bool:
        """Whether NNN/long-range bonds are shown while editing."""
        return self._show_edit_details

    @property
    def edit_anchor_offset(self) -> tuple[float, float]:
        """Physical translation of the active editable primitive-cell copy."""
        return self._edit_anchor_offset

    def set_show_all_hop_editors(self, enabled: bool) -> None:
        """Toggle dense all-bond editors; compact mode is the default."""
        enabled = bool(enabled)
        if enabled == self._show_all_hop_editors:
            return
        self._show_all_hop_editors = enabled
        # The coefficient rail and the physical long-range layer are
        # deliberately independent controls.  Showing every editable value
        # must not unexpectedly turn on a mesh of NNN/long-range bonds; users
        # can opt into that visual detail with the separate toolbar toggle.
        if self._data is not None:
            self.set_data(self._data)

    def _set_hovered_hop_row(self, row: int | None) -> None:
        """Reveal one dense-rail leader without rebuilding the scene.

        Editors are embedded widgets, so rebuilding on every hover would
        steal focus and make the pointer feel sticky.  Toggle only the two
        existing line items instead; geometry remains owned by
        :meth:`set_zoom_level`.
        """
        normalized = None if row is None else int(row)
        if normalized == self._hovered_hop_row:
            return
        self._hovered_hop_row = normalized
        self._update_edit_leader_visibility()

    def _update_edit_leader_visibility(self) -> None:
        """Apply progressive disclosure to coefficient leader lines."""
        links = self._edit_leader_links
        if not links:
            return
        # Compact mode has at most one active editor, or at most three small
        # model editors. Keeping those leaders visible preserves immediate
        # discoverability. Dense all-fields mode is the only case where
        # crossing lines become a problem, so reveal just the hovered/focused
        # row there.
        dense = self._show_all_hop_editors and len(links) > 3
        reveal_rows = {row for row in (
            self._active_hop_row, self._hovered_hop_row, self._focused_hop_row,
        ) if row is not None}
        for diagonal, horizontal, _mx, _my, proxy in links:
            row = proxy.data(1)
            visible = (not dense) or (row in reveal_rows)
            diagonal.setVisible(visible)
            horizontal.setVisible(visible)

    def set_show_edit_details(self, enabled: bool) -> None:
        """Reveal/hide non-primary bonds only for the lattice edit session."""
        enabled = bool(enabled)
        if enabled == self._show_edit_details:
            return
        self._show_edit_details = enabled
        if not enabled:
            self._active_hop_row = None
        if self._data is not None:
            self.set_data(self._data)

    def activate_hop_editor(self, row: int) -> None:
        """Expose the canvas field for any row in an equivalent relation."""
        representative = self._editor_representative_for_row(row)
        if representative is None:
            return
        representative_row = int(representative.get("row", row))
        self._active_hop_row = representative_row
        if self._data is not None:
            self.set_data(self._data)
        merged_rows = tuple(representative.get("_editor_rows", (row,)))
        suffix = (
            f"（等价表格行：{', '.join(str(value + 1) for value in merged_rows)}）"
            if len(merged_rows) > 1 else ""
        )
        self.editSelectionChanged.emit(
            f"已选跃迁 {representative_row + 1}{suffix}；"
            "在键旁输入强度，或在左侧表格精确编辑"
        )

    def update_hop_strength(self, row: int, strength: float) -> None:
        """Keep the transient edit context in sync with a committed field.

        Theme/UI-scale changes rebuild the embedded editor layer from the
        latest :attr:`_edit_hops` snapshot before the debounced Hamiltonian
        rebuild necessarily runs.  Updating this view-only snapshot after the
        controller accepts a value prevents a freshly applied stylesheet from
        restoring the old coefficient on screen.
        """
        representative = self._editor_representative_for_row(row)
        rows = (
            {int(value) for value in representative.get("_editor_rows", ())}
            if representative is not None else {int(row)}
        )
        value = float(strength)
        for hop in self._edit_hops:
            if int(hop.get("row", -1)) in rows:
                hop["strength"] = value

    def set_edit_context(self, sites, *, hops=(), cell_vectors=None,
                         snap_step: float | None = None,
                         anchor_offset: tuple[float, float] | None = None):
        self._edit_sites = [(float(x), float(y), str(sub or "A")) for x, y, sub in sites]
        self._edit_hops = [dict(h) for h in hops]
        valid_rows = {int(h.get("row", -1)) for h in self._editable_hops()}
        if self._active_hop_row not in valid_rows:
            self._active_hop_row = None
        if cell_vectors is not None:
            self._cell_vectors = tuple(tuple(map(float, vector)) for vector in cell_vectors)
            self._edit_cell_constrained = True
        else:
            self._edit_cell_constrained = False
        if anchor_offset is not None:
            try:
                ax, ay = map(float, anchor_offset)
            except (TypeError, ValueError) as exc:
                raise ValueError("编辑元胞锚点必须是 (x, y) 坐标") from exc
            if not (math.isfinite(ax) and math.isfinite(ay)):
                raise ValueError("编辑元胞锚点必须是有限坐标")
            self._edit_anchor_offset = (ax, ay)
        else:
            # set_edit_context is also used by isolated scenes in tests and
            # plug-ins.  Do not accidentally keep an anchor belonging to a
            # previous finite sample when such a caller does not provide one.
            self._edit_anchor_offset = (0.0, 0.0)
        if snap_step is not None:
            # Use the validated setter so plug-ins and restored documents get
            # the same finite range as the toolbar, and so an already-visible
            # edit grid cannot retain a stale interval.
            self.set_snap_step(snap_step)
        if self._data is not None:
            self.set_data(self._data)

    def constrain_edit_position(self, point: QPointF) -> QPointF:
        """Keep a dragged primitive-cell site inside its Bravais cell.

        The persistence layer quite correctly rejects coordinates outside the
        declared cell.  A canvas drag, however, should never leave the model
        in that invalid intermediate state.  Project the pointer into the
        parallelogram in fractional ``(u, v)`` coordinates and leave
        stand-alone scenes unconstrained when no vectors were supplied.
        """
        point = QPointF(point)
        if not self._edit_cell_constrained:
            return point
        fractional = self._scene_to_local(point)
        u_v = self._local_to_fractional(*fractional)
        if u_v is None:
            return point
        u, v = u_v
        # Keep the upper edge strictly inside: validation uses u/v < 1.
        # Keep a visible decimal margin as well as a mathematical one.  The
        # coordinate table intentionally displays eight significant digits;
        # an ``1-1e-9`` clamp would round back to ``1`` in that field and
        # immediately fail the strict ``u/v < 1`` persistence check.
        eps = 1e-7
        cu = min(1.0 - eps, max(0.0, u))
        cv = min(1.0 - eps, max(0.0, v))
        if abs(cu - u) <= 1e-15 and abs(cv - v) <= 1e-15:
            return point
        return self._local_to_scene(*self._fractional_to_local(cu, cv))

    def _edit_collision_tolerance(self) -> float:
        """Return the minimum visual clearance between editable sites."""
        # This is intentionally smaller than a typical site diameter: nearby
        # legitimate Kagome/oblique sites remain reachable, while a dragged
        # circle can never be written on top of another circle.  The value is
        # independent of view zoom because coordinates live in scene units.
        return max(0.03, min(0.12, max(0.03, self.snap_step * 0.8)))

    def _edit_live_positions(self, skip_index: int | None = None):
        for owner, (x, y, _sub) in enumerate(self._edit_sites):
            if skip_index is not None and owner == int(skip_index):
                continue
            yield owner, self._local_to_scene(float(x), float(y))

    def _edit_position_is_clear(self, index: int, point: QPointF,
                                tolerance: float | None = None) -> bool:
        """Check a scene-space candidate against every other live site."""
        if not 0 <= int(index) < len(self._edit_sites):
            return True
        radius = self._edit_collision_tolerance() if tolerance is None else float(tolerance)
        return all(
            math.hypot(point.x() - live.x(), point.y() - live.y()) > radius
            for _owner, live in self._edit_live_positions(int(index))
        )

    def safe_edit_position(self, index: int, point: QPointF,
                           *, fallback: QPointF | None = None) -> QPointF:
        """Constrain a drag and guarantee it does not overlap another site.

        The old implementation protected only magnetic snap candidates.  A
        free drag (Alt or a disabled snap toggle) could therefore emit a
        duplicate coordinate and leave the controller with an invalid model.
        This final gate is deliberately view-local and deterministic: it first
        nudges away from the nearest occupied site, then tries a small radial
        set of alternatives, and finally returns the previous valid position.
        """
        index = int(index)
        candidate = self.constrain_edit_position(QPointF(point))
        if self._edit_position_is_clear(index, candidate):
            return candidate
        clearance = self._edit_collision_tolerance() + 0.02
        fallback_point = None
        if fallback is not None:
            fallback_point = self.constrain_edit_position(QPointF(fallback))

        def clear(value: QPointF) -> bool:
            return self._edit_position_is_clear(index, value)

        # Push out from each colliding site.  Iterating makes the result stable
        # even when a dense model has two neighbours around the pointer.
        adjusted = QPointF(candidate)
        for _ in range(max(2, len(self._edit_sites) + 1)):
            changed = False
            for _owner, live in self._edit_live_positions(index):
                dx = adjusted.x() - live.x()
                dy = adjusted.y() - live.y()
                distance = math.hypot(dx, dy)
                if distance > self._edit_collision_tolerance():
                    continue
                if distance <= 1e-12:
                    if fallback_point is not None:
                        dx = fallback_point.x() - live.x()
                        dy = fallback_point.y() - live.y()
                        distance = math.hypot(dx, dy)
                    if distance <= 1e-12:
                        # Deterministic direction for a pathological already
                        # duplicated input; valid models normally use fallback.
                        angle = (index + 1) * 1.61803398875
                        dx, dy, distance = math.cos(angle), math.sin(angle), 1.0
                adjusted = self.constrain_edit_position(QPointF(
                    live.x() + dx / distance * clearance,
                    live.y() + dy / distance * clearance,
                ))
                changed = True
            if not changed or clear(adjusted):
                break
        if clear(adjusted):
            return adjusted

        # A radial fallback preserves as much of the user's pointer intent as
        # possible when the first nudge is clipped by an oblique cell edge.
        for radius in (clearance, clearance * 1.8, clearance * 2.6):
            for step in range(16):
                angle = 2.0 * math.pi * step / 16.0
                probe = self.constrain_edit_position(QPointF(
                    candidate.x() + radius * math.cos(angle),
                    candidate.y() + radius * math.sin(angle),
                ))
                if clear(probe):
                    return probe
        # The drag origin is a valid, non-overlapping position for a valid
        # model, so it is the safest final fallback instead of emitting a
        # duplicate that would invalidate the next rebuild.
        if fallback_point is not None and clear(fallback_point):
            return fallback_point
        return adjusted

    def snap_position(self, index: int, point: QPointF,
                      drag_origin: QPointF | None = None) -> QPointF:
        """Grid snap plus magnetic restoration/alignment candidates.

        Rectangular cells use the Cartesian interval; oblique cells use
        fractional subdivisions along ``a₁``/``a₂``.  Exact model coordinates
        and the drag origin remain higher-priority magnetic candidates, so
        graphene/Kagome basis sites stay reachable even when their fractions
        are not an integer subdivision of the chosen interval.  Alt still
        means completely free movement.
        """
        interactive_drag = drag_origin is not None
        if not self.snap_enabled:
            result = QPointF(point)
            return (
                self.safe_edit_position(index, result, fallback=drag_origin)
                if interactive_drag else result
            )
        step = self.snap_step
        # Snap in the primitive cell's local coordinates.  A central anchor
        # for a slanted disk/hexagon cell is generally not a multiple of the
        # grid (e.g. sqrt(3)/2); snapping in global scene coordinates would
        # silently write those fractional offsets back into the site table.
        local_x, local_y = self._scene_to_local(point)
        snapped_x, snapped_y = self._snap_local_point(local_x, local_y, step)
        grid = self._local_to_scene(snapped_x, snapped_y)
        tolerance = max(0.08, step * 0.55)
        collision_tolerance = max(0.03, min(0.12, tolerance * 0.8))
        # Snap as a complete 2-D point.  The previous independent x/y
        # candidates could combine coordinates from two different atoms and
        # create a position that was not part of any lattice geometry.
        # During a real drag, never magnetically snap one site onto another
        # site's current position.  Besides being visually confusing, that
        # creates duplicate rows and can make the topology appear to lose a
        # bond after the next rebuild.  Programmatic callers (without a drag
        # origin) still get the old complete-geometry candidate behaviour;
        # this keeps the helper useful for previews and preserves its 2-D
        # snapping semantics outside the interactive editor.
        live_positions = [
            (self._local_to_scene(float(x), float(y)), site_index)
            for site_index, (x, y, _s) in enumerate(self._edit_sites)
        ]
        current_candidates: list[tuple[QPointF, int]] = []
        for site_index, (x, y, _s) in enumerate(self._edit_sites):
            if not interactive_drag or site_index == index:
                current_candidates.append(
                    (self._local_to_scene(float(x), float(y)), site_index)
                )
        candidates: list[QPointF] = [candidate for candidate, _owner in current_candidates]
        for ref_index, (x, y) in enumerate(self._snap_reference_sites):
            candidate = self._local_to_scene(float(x), float(y))
            if interactive_drag:
                # A baseline target is useful for restoring a moved site, but
                # not if that target is already occupied by a different live
                # site.  Keep the ownership check in scene coordinates so it
                # also works for oblique anchor offsets.
                occupied = any(
                    owner != index
                    and math.hypot(candidate.x() - live.x(), candidate.y() - live.y())
                    <= collision_tolerance
                    for live, owner in live_positions
                )
                if occupied:
                    continue
            candidates.append(candidate)
        if drag_origin is not None:
            candidates.append(QPointF(drag_origin))
        nearest = min(
            candidates,
            key=lambda candidate: math.hypot(
                candidate.x() - point.x(), candidate.y() - point.y()
            ),
            default=None,
        )
        if nearest is not None and math.hypot(
            nearest.x() - point.x(), nearest.y() - point.y()
        ) <= tolerance:
            return self.safe_edit_position(index, nearest, fallback=drag_origin)
        if interactive_drag:
            # Keep the pointer responsive while enforcing a small visual
            # clearance.  Returning the raw point on an exact neighbour hit
            # still allowed duplicate coordinates; nudge that point away from
            # the closest live site instead.  The nudge is scene-space and
            # therefore works for oblique cells and non-unit coordinates.
            def nudge_from_collisions(candidate: QPointF) -> QPointF:
                adjusted = QPointF(candidate)
                for live, owner in live_positions:
                    if owner == index:
                        continue
                    dx = adjusted.x() - live.x()
                    dy = adjusted.y() - live.y()
                    distance = math.hypot(dx, dy)
                    if distance >= collision_tolerance:
                        continue
                    if distance <= 1e-12:
                        if 0 <= index < len(live_positions):
                            dx = live_positions[index][0].x() - live.x()
                            dy = live_positions[index][0].y() - live.y()
                        distance = math.hypot(dx, dy)
                    if distance <= 1e-12:
                        dx, dy, distance = 1.0, 0.0, 1.0
                    clearance = collision_tolerance + 0.02
                    adjusted = QPointF(
                        live.x() + dx / distance * clearance,
                        live.y() + dy / distance * clearance,
                    )
                return adjusted

            grid = nudge_from_collisions(grid)
            # If the grid candidate collided, prefer a nudged pointer position
            # so a fast drag does not snap to an unrelated lattice point.
            if any(
                owner != index
                and math.hypot(grid.x() - live.x(), grid.y() - live.y())
                <= collision_tolerance
                for live, owner in live_positions
            ):
                return self.safe_edit_position(
                    index, nudge_from_collisions(QPointF(point)),
                    fallback=drag_origin,
                )
        if interactive_drag:
            return self.safe_edit_position(index, grid, fallback=drag_origin)
        return grid

    def set_edit_mode(self, enabled: bool):
        self.edit_mode = bool(enabled)
        self.set_hop_creation_mode(False, announce=False)
        self.set_site_creation_mode(False, announce=False)
        if self.edit_mode:
            # Each edit session starts structure-first.  This is view-only;
            # the user's persistent normal-view display preferences remain
            # untouched and return as soon as editing ends.
            self._show_edit_details = False
        else:
            self._active_hop_row = None
        if self._data is not None:
            self.set_data(self._data)

    @property
    def hop_creation_mode(self) -> bool:
        return self._hop_creation_mode

    @property
    def site_creation_mode(self) -> bool:
        return self._site_creation_mode

    def set_site_creation_mode(self, enabled: bool, *, announce: bool = True) -> None:
        """Enter/leave the deliberate one-click site creation tool."""
        enabled = bool(enabled) and self.edit_mode
        if enabled:
            self.set_hop_creation_mode(False, announce=False)
        if enabled == self._site_creation_mode:
            return
        self._site_creation_mode = enabled
        self.siteCreationModeChanged.emit(enabled)
        if announce:
            self.editSelectionChanged.emit(
                "添加格点：请在晶格空白处单击；Esc 或再次点击“添加格点”取消"
                if enabled else "已退出添加格点；单击格点仅选择，拖动即可移动"
            )

    def set_hop_creation_mode(self, enabled: bool, *, announce: bool = True) -> None:
        """Enter/leave the deliberate two-site bond creation tool.

        Normal clicks remain harmless selection clicks.  Keeping this state
        in the scene also makes it impossible for an invisible stale start
        site to survive a redraw or a mode switch.
        """
        enabled = bool(enabled) and self.edit_mode
        if enabled:
            self.set_site_creation_mode(False, announce=False)
        if enabled == self._hop_creation_mode and self._hop_start is None:
            return
        self._hop_creation_mode = enabled
        self._hop_start = None
        self._hop_start_endpoint = None
        for ghost in self._ghost_items:
            ghost.set_interaction_enabled(enabled)
        for item in self._edit_items.values():
            item.setOpacity(1.0)
        self.hopCreationModeChanged.emit(enabled)
        if announce:
            message = (
                "添加跃迁：依次点击两个格点（半无限可点左右虚影自动生成胞间偏移）；"
                "再次点击“添加跃迁”或按 Esc 取消"
                if enabled else "已退出添加跃迁；单击格点仅选择，拖动即可移动"
            )
            self.editSelectionChanged.emit(message)

    def activate_site(self, index: int):
        if not self.edit_mode:
            return
        if not self._hop_creation_mode:
            # Do not overload selection as a destructive / modal operation.
            self.editSelectionChanged.emit(
                f"已选择格点 {index + 1}；拖动可移动，Delete 可删除；"
                "如需连线请先点击“添加跃迁”"
            )
            return
        self._activate_hop_endpoint(index, 0, 0)

    def activate_ghost(self, index: int, cell_dx: int, cell_dy: int):
        """Use a visible periodic image as an explicit bond endpoint."""
        if not self.edit_mode or not self._hop_creation_mode:
            return
        self._activate_hop_endpoint(index, int(cell_dx), int(cell_dy))

    def _activate_hop_endpoint(self, index: int, cell_dx: int, cell_dy: int):
        """Collect two logical endpoints and emit their relative cell offset."""
        index = int(index)
        if not 0 <= index < len(self._edit_sites):
            self.editSelectionChanged.emit("该格点不属于当前可编辑元胞")
            return
        if self._hop_start is None:
            self._hop_start = index
            self._hop_start_endpoint = (index, int(cell_dx), int(cell_dy))
            where = (
                "中心元胞" if (cell_dx, cell_dy) == (0, 0)
                else f"元胞偏移 ({cell_dx:+d}, {cell_dy:+d})"
            )
            self.editSelectionChanged.emit(
                f"已选择起点 格点 {index + 1}（{where}）；请选择终点"
            )
            return
        start = self._hop_start
        start_endpoint = self._hop_start_endpoint or (start, 0, 0)
        self._hop_start = None
        self._hop_start_endpoint = None
        relative = (
            int(cell_dx) - int(start_endpoint[1]),
            int(cell_dy) - int(start_endpoint[2]),
        )
        # A self-hop with a non-zero cell offset is a valid Bloch term; only
        # reject an identical primitive site in the identical cell.
        if start != index or relative != (0, 0):
            if relative == (0, 0):
                self.hoppingRequested.emit(start, index)
            else:
                self.hoppingRequestedWithOffset.emit(
                    start, index, relative[0], relative[1],
                )
            # A single bond is one explicit action.  Returning immediately
            # to normal selection avoids a latent second creation operation.
            self.set_hop_creation_mode(False, announce=False)
            self.editSelectionChanged.emit("已提交跃迁；单击格点仅选择，拖动即可移动")
        else:
            self._hop_start = start
            self._hop_start_endpoint = start_endpoint
            self.editSelectionChanged.emit(
                "起点与终点是同一元胞格点；请选择另一格点或周期像，或取消添加跃迁"
            )

    def delete_selected(self) -> bool:
        if not self.edit_mode:
            return False
        selected = [item for item in self.selectedItems() if isinstance(item, _EditableSiteItem)]
        if not selected:
            return False
        self.siteDeleteRequested.emit(selected[0].site_index)
        return True

    def _append_site_at(self, scene_point: QPointF) -> None:
        """Emit one valid local coordinate for the explicit site tool.

        Site creation used to round every click, even after the user had
        turned ``吸附`` off, and it forwarded clicks outside an oblique
        primitive cell to the panel.  The next rebuild then rejected the
        model, which looked like a newly added point (or an existing bond)
        had mysteriously disappeared.  Keep creation on the same validity
        contract as dragging: optional grid rounding, strict cell bounds,
        and a small duplicate/overlap guard before emitting the signal.
        """
        scene_point = QPointF(scene_point)
        x, y = self._scene_to_local(scene_point)
        if self.snap_enabled:
            step = max(0.001, float(self.snap_step))
            x, y = self._snap_local_point(x, y, step)

        # A site table stores local primitive-cell coordinates.  Reject an
        # invalid click with an actionable status message instead of silently
        # clamping it to an edge or letting the later Hamiltonian rebuild fail.
        fractional = self._local_to_fractional(x, y)
        if fractional is not None:
            u, v = fractional
            if not (-1e-9 <= u < 1.0 - 1e-9
                    and -1e-9 <= v < 1.0 - 1e-9):
                self.editSelectionChanged.emit(
                    "添加失败：请在蓝色原始元胞范围内点击；元胞外坐标不会写入模型"
                )
                return

        candidate = self._local_to_scene(x, y)
        clearance = self._edit_collision_tolerance()
        if any(
            math.hypot(candidate.x() - live.x(), candidate.y() - live.y())
            <= clearance
            for _owner, live in self._edit_live_positions()
        ):
            self.editSelectionChanged.emit(
                "添加失败：该位置与已有格点过近，请换一个网格节点"
            )
            return
        self.set_site_creation_mode(False, announce=False)
        self.siteAddRequested.emit(float(x), float(y))
        mode = "吸附" if self.snap_enabled else "自由"
        self.editSelectionChanged.emit(
            f"已添加格点 ({x:g}, {y:g})（{mode}位置）；拖动可调整，表格可精确检查"
        )

    def mousePressEvent(self, event):
        if (self.edit_mode and self._site_creation_mode
                and event.button() == Qt.LeftButton):
            # A creation click must target an actual blank position, not
            # consume a site or a coefficient input that the user meant to
            # inspect.  Cell outlines and bonds are allowed here: they are
            # visual context rather than editable controls.
            hit = self.items(event.scenePos())
            if any(isinstance(item, (_EditableSiteItem, QGraphicsProxyWidget))
                   for item in hit):
                self.editSelectionChanged.emit(
                    "请在没有格点或输入框的空白处单击；Esc 可取消添加格点"
                )
            else:
                self._append_site_at(event.scenePos())
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        # Double click deliberately has no mutation semantics.  It used to
        # add a site on a best-effort "blank" hit test, which felt random on
        # dense lattices and could be mistaken for a disappearing connection.
        if self.edit_mode:
            self.editSelectionChanged.emit(
                "双击不会修改晶格；需要新增格点请先启用“添加格点”"
            )
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self._hop_creation_mode:
            self.set_hop_creation_mode(False)
            event.accept()
            return
        if event.key() == Qt.Key_Escape and self._site_creation_mode:
            self.set_site_creation_mode(False)
            event.accept()
            return
        super().keyPressEvent(event)

    def set_data(self, data: LatticeSceneData):
        self._data = data
        # A rebuild invalidates the old hover target.  Keeping it would make
        # an unrelated new field inherit a leader highlight after a parameter
        # change or theme switch.
        self._hovered_hop_row = None
        self._focused_hop_row = None
        # QGraphicsProxyWidget embeds a real QWidget into the scene.  Merely
        # dropping our Python references before ``scene.clear()`` is not
        # sufficient on the offscreen/Windows paint paths: the old child can
        # keep one stale backing-store frame at its previous position.  After
        # a fit or a tab switch that frame looks like a second row of editors,
        # often clipped by the bottom edge of the viewport.  Hide and detach
        # every embedded widget first, then let Qt dispose it after the
        # current event is finished.  This keeps rebuilds and theme changes
        # visually atomic without changing the editor's public signals.
        old_proxies = tuple(self._edit_proxies)
        for proxy in old_proxies:
            widget = proxy.widget()
            if widget is not None:
                widget.hide()
                proxy.setWidget(None)
                widget.deleteLater()
                # ``deleteLater`` is intentionally deferred during normal
                # interaction, but a rebuild can be immediately followed by
                # a grab/tab switch before the event loop gets another turn.
                # Flush only this widget's deferred delete, never the global
                # queue, so dialogs and unrelated controls keep their normal
                # Qt lifecycle.
                QCoreApplication.sendPostedEvents(widget, QEvent.DeferredDelete)
        self._edit_proxies.clear()
        self._grid_items.clear()
        self._ghost_items.clear()
        self._edit_guides.clear()
        self._edit_leaders.clear()
        for badge, _proxy in self._edit_relation_badges:
            self.removeItem(badge)
        self._edit_relation_badges.clear()
        self._edit_leader_links.clear()
        self._edit_proxy_anchors.clear()
        self.clear()
        pal = Palette()
        if not data.sites:
            self.setSceneRect(QRectF())
            return

        # 坐标范围 → 场景尺寸
        # Qt 场景 y 轴向下；统一用 -physical_y，保证物理图像 y 轴向上。
        all_pts = [(s[0], -s[1]) for s in data.sites]
        if self._show_ghosts:
            all_pts += [(g[0], -g[1]) for g in data.ghost]
        # Keep the explicit finite-mask outline inside the scene rect.  This
        # matters for a coarse disk/triangle where the outer polygon can sit a
        # little beyond the centre of the boundary cells.
        all_pts += [(point[0], -point[1]) for point in data.boundary_outline]
        xs = [p[0] for p in all_pts]
        ys = [p[1] for p in all_pts]
        pad = 0.7
        x0, x1 = min(xs) - pad, max(xs) + pad
        y0, y1 = min(ys) - pad, max(ys) + pad
        # Editing uses a dedicated right-hand coefficient rail. Reserve a
        # scene-space pocket for it before the view fits the lattice; without
        # this margin the rail is forced on top of the rightmost physical
        # nodes in finite OBC samples (especially NP/Kagome).
        if self.edit_mode and self._editable_hops():
            # Keep a generous scene-space pocket: the proxy is fixed in
            # screen pixels, so a percentage-only margin becomes too small
            # for compact semi-infinite samples after a 180% UI scale.  The
            # absolute floor covers the widest styled editor plus its gap,
            # node radius and edge cushion at the smallest normal zoom.
            rail_reserve = max(2.4, 0.28 * (x1 - x0))
            x1 += rail_reserve
        self.setSceneRect(QRectF(x0, y0, x1 - x0, y1 - y0))

        # Keep the snap targets visually behind cell outlines and bonds.  It
        # is intentionally drawn only while editing; browsing remains clean.
        if self.edit_mode:
            self._draw_snap_grid()

        # A non-rectangular sample should identify itself in the canvas.  The
        # badge is deliberately anchored to the scene margin (not to a cell
        # or a node), so it remains visible after zooming/panning without
        # covering the physical Kagome skeleton.  Keep it as a scene item so
        # exports and the combined view carry the same unambiguous context.
        if (data.boundary_outline and len(data.boundary_outline) == 3
                and getattr(data, "title", "")):
            badge = QGraphicsTextItem("正三角纳米盘 · 平直等边边界")
            badge_font = QFont("Cambria Math")
            badge_font.setPointSizeF(10.0)
            badge.setFont(badge_font)
            badge_outline = (0.38, 0.68, 0.92) if self._dark else (0.18, 0.42, 0.68)
            badge.setDefaultTextColor(_q(badge_outline))
            bounds = badge.boundingRect()
            if bounds.width() > 0:
                # Keep the physical badge compact across models with very
                # different cell scales; it follows view zoom like the
                # matrix/lattice labels instead of becoming a giant overlay.
                target_width = min(3.4, max(1.8, 0.28 * (x1 - x0)))
                badge.setScale(target_width / bounds.width())
            badge.setPos(x0 + 0.18, y0 + 0.16)
            badge.setZValue(5.0)
            badge.setData(0, "finite-shape-badge")
            badge.setToolTip("双开边界正三角形纳米盘；三条边为平直等边边界")
            self.addItem(badge)

        # 元胞框 (z=0): 首胞实线高亮, 其余灰虚线 (MATLAB §6.3)
        outline = (0.38, 0.68, 0.92) if self._dark else (0.18, 0.42, 0.68)
        secondary_outline = (0.48, 0.53, 0.60) if self._dark else (0.55, 0.58, 0.62)
        for k, (bx, by, bw, bh) in enumerate(
            data.cell_boxes if self._show_cells else ()
        ):
            item = QGraphicsRectItem(bx - 0.5, -(by - 0.5 + bh), bw, bh)
            if k == 0:
                item.setPen(_pen(outline, 1.35))
                item.setBrush(QBrush(_alpha_over((0.72, 0.78, 0.95), 24, self._blend_base)))
            else:
                item.setPen(_pen(secondary_outline, 0.75, Qt.DashLine))
                item.setBrush(Qt.NoBrush)
            item.setZValue(0)
            self.addItem(item)
        # A complete grid of oblique primitive-cell outlines is helpful for a
        # tiny sample, but on a 5×5+ triangular/hexagonal disk it becomes a
        # second, dashed lattice that competes with the physical hopping
        # skeleton.  Keep the first cell as an explicit blue "primitive
        # cell" cue and let the unambiguous finite-sample silhouette carry
        # the rest of the geometry.  The DTO still contains every polygon for
        # editing/export; this is presentation-only progressive disclosure.
        polygons = data.cell_polygons if self._show_cells else ()
        # In a finite non-rectangular sample the primitive-cell polygons are
        # a *reference* for editing, not a second outline of the whole disk.
        # Drawing every oblique cell makes the parallelograms extend beyond
        # the actual basis sites (especially the 3-site Kagome triangle),
        # producing dashed lines outside the physical sample.  Keep one
        # clearly highlighted primitive cell and let boundary_outline provide
        # the single, honest physical silhouette.  Semi-infinite ribbons and
        # rectangular OBC samples retain their complete cell grid.
        if data.boundary_outline and not data.semi:
            polygons = polygons[:1]
        elif len(polygons) > 12:
            polygons = polygons[:1]
        for k, polygon in enumerate(polygons):
            item = QGraphicsPolygonItem(QPolygonF([QPointF(x, -y) for x, y in polygon]))
            if k == 0:
                item.setPen(_pen(outline, 1.35))
                item.setBrush(QBrush(_alpha_over((0.72, 0.78, 0.95), 24, self._blend_base)))
                item.setToolTip("原始元胞（一个平移单元）")
            else:
                item.setPen(_pen(secondary_outline, 0.75, Qt.DashLine))
                item.setBrush(Qt.NoBrush)
            item.setZValue(0)
            self.addItem(item)

        # Non-rectangular samples get one clear physical silhouette in
        # addition to the individual primitive-cell outlines.  Without this
        # layer a sparse triangle/disk can look like a rectangular collection
        # of cells, especially when the cell-outline toggle is disabled.
        if data.boundary_outline:
            item = QGraphicsPolygonItem(
                QPolygonF([QPointF(x, -y) for x, y in data.boundary_outline])
            )
            item.setPen(_pen(outline, 1.8))
            item.setBrush(Qt.NoBrush)
            item.setZValue(0.35)
            item.setData(0, "finite-shape-outline")
            self.addItem(item)

        # 键 (z=1): NN 实线红 / NNN 虚线绿。编辑态优先让用户看清
        # 最近邻骨架；次近邻仍保留且可点，但退到背景层，避免密集模型
        # 看起来像所有连接同等重要的一张线网。
        pos = {i: (x, y) for i, (x, y, _l, _s) in enumerate(data.sites)}
        # Hiding long bonds is useful for dense presets, but it makes a small
        # custom triangle look broken when its third edge is merely longer
        # than the nearest shell.  Small scenes keep every authored edge
        # visible; dense scenes retain the progressive-disclosure behaviour.
        compact_scene = len(self._edit_sites) == 3 and len(data.edges) <= 12
        # A relation entered through the table or an explicit add tool is a
        # deliberate user action.  Do not let the compact preset detail layer
        # hide it merely because its geometric length is beyond the nearest
        # shell; otherwise the row is present in the model but appears to
        # have vanished from the canvas.  The flag is transient UI metadata
        # and never changes the physical shell classification.
        user_added_hop = any(
            bool(hop.get("_user_added")) for hop in self._edit_hops
        )
        show_edit_details = (
            (not self.edit_mode) or self._show_edit_details or compact_scene
            or user_added_hop
        )
        for i, j, kind in data.edges:
            if (kind == "NN" and not self._show_nn) or (
                kind != "NN" and (not self._show_nnn or not show_edit_details)
            ):
                continue
            if i not in pos or j not in pos:
                continue
            x1, y1 = pos[i]
            x2, y2 = pos[j]
            y1, y2 = -y1, -y2
            line = QGraphicsLineItem(x1, y1, x2, y2)
            rgb = pal.edge_nn if kind == "NN" else pal.edge_nnn
            pen = _pen(rgb, 2.0 if kind == "NN" else 1.15)
            if kind != "NN":
                pen.setStyle(Qt.DashLine)
            line.setPen(pen)
            # NNN/long-range bonds are useful physical context, but drawing
            # them at the same visual weight as the NN skeleton makes dense
            # honeycomb/Kagome views look like an unstructured green mesh.
            # Keep the layer available (the display toggle can still hide it)
            # while lowering its normal-view contrast; edit mode remains even
            # quieter until the explicit detail switch is enabled.
            line.setOpacity(1.0 if kind == "NN" else (0.24 if self.edit_mode else 0.38))
            line.setZValue(1.0 if kind == "NN" else 0.8)
            line.setData(0, "physical-edge-nn" if kind == "NN" else "physical-edge-nnn")
            self.addItem(line)

        # 虚影键 (z=1.5, 黯淡): MATLAB cNN*0.5+0.5 / cNNN*0.4+0.6
        for x1, y1, x2, y2, kind in (
            data.ghost_edges if self._show_ghosts else ()
        ):
            if (kind == "NN" and not self._show_nn) or (
                kind != "NN" and (not self._show_nnn or not show_edit_details)
            ):
                continue
            y1, y2 = -y1, -y2
            line = QGraphicsLineItem(x1, y1, x2, y2)
            if kind == "NN":
                # A periodic image is a distinct relation from the solid
                # intracell bond even when the two segments are collinear
                # (SSH is the common example).  A quiet dash pattern prevents
                # the two from reading as one long line while retaining the
                # familiar faded ghost colour.
                pen = _pen(_blend(pal.edge_nn, 0.5, 0.5), 1.2, Qt.DashLine)
            else:
                pen = _pen(_blend(pal.edge_nnn, 0.4, 0.6), 1.0, Qt.DashLine)
            line.setPen(pen)
            line.setOpacity(
                0.72 if kind == "NN" else (0.12 if self.edit_mode else 0.20)
            )
            line.setZValue(0.7 if kind == "NN" else 0.6)
            line.setData(0, "ghost-edge-nn" if kind == "NN" else "ghost-edge-nnn")
            self.addItem(line)

        # 格点 (z=3) + 序号 (z=4)
        # 半径相对当前模型最短键长定义，不依赖绝对坐标尺度。
        bond_lengths = []
        for i, j, _kind in data.edges:
            if i in pos and j in pos:
                x1, y1 = pos[i]
                x2, y2 = pos[j]
                length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                if length > 1e-12:
                    bond_lengths.append(length)
        # Keep the node as a readable anchor rather than a giant selection
        # halo. The previous 0.16 ratio made compact honeycomb/Kagome views
        # look like overlapping bubbles once edit handles and fields were on.
        r = 0.12 * min(bond_lengths) if bond_lengths else 0.14
        self._site_radius = r
        for _idx, (x, y, label, sub) in enumerate(data.sites):
            y = -y
            rgb = pal.site_a if sub == "A" else pal.site_b
            circ = QGraphicsEllipseItem(x - r, y - r, 2 * r, 2 * r)
            circ.setBrush(QBrush(_q(rgb)))
            circ.setPen(_pen((0.75, 0.8, 0.85) if self._dark else (0, 0, 0), 0.6))
            circ.setZValue(3)
            self.addItem(circ)
            if not self.edit_mode:
                t = QGraphicsTextItem(str(label))
                t.setDefaultTextColor(QColor(255, 255, 255))
                font = QFont()
                font.setPointSizeF(6.5)
                font.setBold(True)
                t.setFont(font)
                _fit_text_to_circle(t, x, y, 2 * r)
                t.setZValue(4)
                self.addItem(t)

        # 虚影格点 (z=2, 半透明 + 黯淡序号).  Newer scene DTOs carry the
        # logical source/offset metadata so these images can be selected by
        # the explicit bond tool.  Keep the three-column fallback for plug-ins
        # and older callers that construct LatticeSceneData directly.
        rich_ghosts = data.ghost_sites or ()
        if self._show_ghosts and rich_ghosts:
            for x, y, label, source_site, cell_dx, cell_dy in rich_ghosts:
                item = _GhostSiteItem(
                    self, x, y, r, source_site, cell_dx, cell_dy,
                )
                item.set_interaction_enabled(self._hop_creation_mode)
                self.addItem(item)
                self._ghost_items.append(item)
                if not self.edit_mode:
                    t = QGraphicsTextItem(str(label))
                    t.setDefaultTextColor(QColor(255, 255, 255))
                    font = QFont()
                    font.setPointSizeF(6.0)
                    font.setBold(True)
                    t.setFont(font)
                    t.setOpacity(0.6)
                    _fit_text_to_circle(t, x, -y, 2 * r)
                    t.setZValue(2.5)
                    self.addItem(t)
        elif self._show_ghosts:
            for x, y, label in data.ghost:
                y = -y
                circ = QGraphicsEllipseItem(x - r, y - r, 2 * r, 2 * r)
                circ.setBrush(QBrush(_q(pal.site_ghost)))
                circ.setPen(_pen((0, 0, 0), 0.6))
                circ.setOpacity(0.45)
                circ.setZValue(2)
                self.addItem(circ)
                if not self.edit_mode:
                    t = QGraphicsTextItem(str(label))
                    t.setDefaultTextColor(QColor(255, 255, 255))
                    font = QFont()
                    font.setPointSizeF(6.0)
                    font.setBold(True)
                    t.setFont(font)
                    t.setOpacity(0.6)
                    _fit_text_to_circle(t, x, y, 2 * r)
                    t.setZValue(2.5)
                    self.addItem(t)

        if self.edit_mode:
            self._draw_edit_handles()
            self._draw_edit_hop_controls()
            # Rebuilds triggered by “显示全部系数” happen after the view has
            # already been fitted.  Reflow immediately against the active
            # view transform; otherwise the newly created fixed-pixel proxies
            # briefly remain at their bond-local anchors instead of the right
            # rail until the next resize/zoom event.
            self._reflow_editors()

    def _draw_edit_handles(self):
        """Overlay only the unit-cell sites; expanded copies stay read-only."""
        self._edit_items.clear()
        anchor_x, anchor_y = self._edit_anchor_offset
        for index, (x, y, sub) in enumerate(self._edit_sites):
            item = _EditableSiteItem(
                self, index, x + anchor_x, y + anchor_y,
                radius=self._site_radius * 1.1,
            )
            color = QColor("#1677ff") if sub == "A" else QColor("#f26b38")
            item.setBrush(QBrush(color))
            # Cosmetic pen means a true two-pixel outline.  A regular width=2
            # pen is two *scene units* wide and becomes a giant white halo at
            # the fit scale used by small-coordinate lattices.
            item.setPen(_pen((1.0, 1.0, 1.0), 2.0))
            self.addItem(item)
            self._edit_items[index] = item

    def _editable_hops(self) -> list[dict]:
        """Return valid non-onsite hopping rows eligible for canvas editing."""
        eligible: list[dict] = []
        for hop in self._edit_hops:
            fr, to = int(hop.get("from_site", -1)), int(hop.get("to_site", -1))
            ox, oy = int(hop.get("off_x", 0)), int(hop.get("off_y", 0))
            if not (0 <= fr < len(self._edit_sites) and 0 <= to < len(self._edit_sites)):
                continue
            if fr == to and ox == 0 and oy == 0:
                continue
            eligible.append(hop)
        return eligible

    @staticmethod
    def _editor_geometry_key(hop: dict) -> tuple:
        """Return the canonical geometric part of an editable relation.

        The reverse of ``(fr, to, dx, dy)`` is
        ``(to, fr, -dx, -dy)``.  Choosing the lexicographically smaller tuple
        handles positive as well as negative offsets; a sign-only heuristic
        misses reverse rows such as ``1 → 0, dy=+1``.
        """
        fr, to = int(hop.get("from_site", -1)), int(hop.get("to_site", -1))
        ox, oy = int(hop.get("off_x", 0)), int(hop.get("off_y", 0))
        forward = (fr, to, ox, oy)
        reverse = (to, fr, -ox, -oy)
        return min(forward, reverse)

    @staticmethod
    def _editor_parameter_key(hop: dict) -> tuple:
        """Identify the editable parameter family carried by a hopping row.

        Rows with the same geometry but different named amplitudes (for
        example ``t`` versus ``t2``) are independent physical contributions
        and must keep separate controls.  The numeric multiplier in an
        amplitude expression is deliberately ignored: the ratio editor turns
        ``-t`` into ``-4*t`` while it is still the same editable ``t`` family.
        Reverse Hermitian rows can still share one control: ``phase_sign`` is
        intentionally omitted because it describes direction, not a second
        user-editable magnitude.
        """
        def normalized(value, default: str) -> str:
            if value is None:
                value = default
            return str(value).strip().replace(" ", "")

        classified = classify_strength_expression(hop.get("amplitude", "1.0"))
        return (
            normalized(hop.get("name"), "t"),
            normalized(hop.get("phase_mode"), "none"),
            normalized(hop.get("phase"), "0"),
            classified.kind,
            classified.parameter,
        )

    @classmethod
    def _editor_relation_key(cls, hop: dict) -> tuple:
        """Normalize one physical relation and its parameter family.

        One rendered line gets one field *per independent parameter family*:
        duplicate/reverse rows belonging to the same amplitude remain quiet,
        while distinct terms on an overlapping bond stay editable and
        discoverable instead of silently disappearing from the canvas.
        """
        return cls._editor_geometry_key(hop) + cls._editor_parameter_key(hop)

    def _editor_representative_hops(self, hops: list[dict]) -> list[dict]:
        """Return one editor per geometric relation/parameter family."""
        representatives: list[dict] = []
        grouped: dict[tuple, list[int]] = {}
        for hop in hops:
            key = self._editor_relation_key(hop)
            row = int(hop.get("row", -1))
            if key in grouped:
                grouped[key].append(row)
                # Keep a short audit trail on the representative tooltip;
                # table rows remain untouched and are still the exact fallback.
                representatives[ next(
                    i for i, item in enumerate(representatives)
                    if int(item.get("row", -1)) == grouped[key][0]
                ) ]["_editor_rows"] = tuple(grouped[key])
                continue
            grouped[key] = [row]
            item = dict(hop)
            item["_editor_rows"] = (row,)
            representatives.append(item)
        return representatives

    def _editor_representative_for_row(self, row: int) -> dict | None:
        """Resolve any table row to the representative canvas editor.

        The table intentionally keeps every directed/repeated row. Canvas
        controls may collapse an equivalent group, so callers must not assume
        that the selected table row is itself the proxy's row identity.
        """
        target = int(row)
        for representative in self._editor_representative_hops(
            self._editable_hops()
        ):
            if target in tuple(representative.get("_editor_rows", ())):
                return representative
        return None

    def _primary_editable_rows(self, hops: list[dict]) -> set[int]:
        """Rows in the shortest non-onsite geometrical hopping shell.

        This is only an edit-layer visibility decision.  It intentionally
        does not infer, filter or mutate the physical Hamiltonian terms.
        """
        (a1x, a1y), (a2x, a2y) = self._cell_vectors
        lengths: list[tuple[int, float]] = []
        for hop in hops:
            fr, to = int(hop["from_site"]), int(hop["to_site"])
            ox, oy = int(hop.get("off_x", 0)), int(hop.get("off_y", 0))
            x1, y1, _ = self._edit_sites[fr]
            x2, y2, _ = self._edit_sites[to]
            length = math.hypot(x2 + ox * a1x + oy * a2x - x1,
                                y2 + ox * a1y + oy * a2y - y1)
            if length > 1e-12:
                lengths.append((int(hop.get("row", -1)), length))
        if not lengths:
            return set()
        nearest = min(length for _row, length in lengths)
        return {row for row, length in lengths if length <= nearest * 1.05 + 1e-12}

    def _draw_edit_hop_controls(self):
        """Draw quiet click targets and only the editors that are needed."""
        editable = self._editable_hops()
        editor_hops = self._editor_representative_hops(editable)
        if not editor_hops:
            return
        geometry_counts: dict[tuple, int] = {}
        for hop in editor_hops:
            geometry = self._editor_geometry_key(hop)
            geometry_counts[geometry] = geometry_counts.get(geometry, 0) + 1
        rows = {int(hop.get("row", -1)) for hop in editor_hops}
        if self._active_hop_row not in rows:
            self._active_hop_row = None
        # Small models remain pleasantly direct: all up to three strengths
        # are visible. Dense Kagome/NP/long-range models use progressive
        # disclosure by default: click a source-cell bond to open one field.
        if self._show_all_hop_editors or len(editor_hops) <= 3:
            visible_rows = rows
        elif self._active_hop_row is None:
            visible_rows = set()
        else:
            visible_rows = {self._active_hop_row}
        # Hidden detail bonds must not leave invisible mouse targets above
        # nearest-neighbour lines.  That used to make a visible red bond
        # unexpectedly open a distant NNN coefficient in dense Kagome views.
        user_added_hop = any(
            bool(hop.get("_user_added")) for hop in editor_hops
        )
        guide_rows = (
            rows if self._show_edit_details or self._show_all_hop_editors
            or user_added_hop
            or (len(self._edit_sites) == 3
                and self._data is not None and len(self._data.edges) <= 12)
            else self._primary_editable_rows(editor_hops)
        )
        if not self._show_edit_details and not self._show_all_hop_editors:
            visible_rows &= guide_rows
        if user_added_hop:
            visible_rows |= {
                int(hop.get("row", -1)) for hop in editor_hops
                if hop.get("_user_added")
            }

        (a1x, a1y), (a2x, a2y) = self._cell_vectors
        # Keep the physical translation of the editable cell immutable for
        # the whole loop.  ``editor_anchor_*`` below is only the temporary
        # screen-space offset for one coefficient field; reusing the same
        # variable used to shift every subsequent bond after the first one.
        base_anchor_x, base_anchor_y = self._edit_anchor_offset
        midpoint_slots: dict[tuple[int, int], int] = {}
        for hop in editor_hops:
            row = int(hop.get("row", -1))
            fr, to = int(hop["from_site"]), int(hop["to_site"])
            ox, oy = int(hop.get("off_x", 0)), int(hop.get("off_y", 0))
            x1, y1, _ = self._edit_sites[fr]
            x2, y2, _ = self._edit_sites[to]
            x1 += base_anchor_x
            y1 += base_anchor_y
            x2 += base_anchor_x + ox * a1x + oy * a2x
            # Both primitive vectors contribute to the endpoint.  The old
            # expression omitted ``off_x * a1y``, so a user-edited oblique
            # a1 vector kept the guide/editor on the wrong horizontal level
            # even though the Hamiltonian used the correct displacement.
            y2 += base_anchor_y + ox * a1y + oy * a2y
            mx, my = (x1 + x2) / 2, -(y1 + y2) / 2

            if row in guide_rows:
                guide = _EditableHopGuide(self, row, x1, -y1, x2, -y2)
                guide.set_selected(row == self._active_hop_row)
                self.addItem(guide)
                self._edit_guides.append(guide)
            if row not in visible_rows:
                continue

            editor = _HopStrengthEdit(f"{float(hop.get('strength', 1.0)):.8g}")
            editor.setObjectName("hopStrengthEditor")
            editor.owner_scene = self
            # Resolve the active stylesheet before fixing the proxy size so
            # the input's complete border and hit target always agree.
            editor.ensurePolished()
            metrics = editor.fontMetrics()
            width = _strength_editor_width(editor, editor.text())
            styled_height = max(
                editor.sizeHint().height(), editor.minimumSizeHint().height(),
                metrics.height() + 12,
            )
            height = max(26, min(36, int(styled_height)))
            editor.setFixedSize(width, height)
            editor.setAlignment(Qt.AlignCenter)
            # A newly created QLineEdit puts its cursor at the end of the
            # constructor text.  When the field is center-aligned, Qt can
            # retain that end-of-text horizontal scroll offset even while
            # the field is unfocused.  After a fraction commit this made a
            # perfectly wide ``0.33333333`` field paint as ``).3333333`` at
            # high UI scale.  Start rebuilt editors at the beginning so the
            # complete value is visible; user typing still controls the
            # cursor once the field receives focus.
            editor.setCursorPosition(0)
            # Do not install QDoubleValidator here: it rejects the perfectly
            # valid ``1/3`` form while the side-panel parameter editor accepts
            # it.  Validation is performed on commit by the shared safe
            # scalar parser below; keeping the field permissive also allows a
            # user to type the slash in two keystrokes without the validator
            # fighting the intermediate text.
            editor.setPlaceholderText("例如 1/3")
            from_site_label = fr + 1
            to_site_label = to + 1
            boundary_label = (
                "胞内" if (ox == 0 and oy == 0)
                else f"胞间 ({ox:+d}, {oy:+d})"
            )
            editor.setToolTip(
                f"跃迁 {row + 1}：格点 {from_site_label} → {to_site_label}，"
                f"{boundary_label}\n"
                "输入绝对跃迁强度（支持小数或分数，如 1/3）；"
                "同名键自动化为最简整数比。悬停物理键可查看对应引线。"
                + (
                    f"\n已合并等价表格行：{', '.join(str(value + 1) for value in hop.get('_editor_rows', (row,)))}"
                    if len(hop.get("_editor_rows", (row,))) > 1 else ""
                )
            )
            editor.editingFinished.connect(
                lambda r=row, field=editor: self._commit_hop_strength(r, field)
            )
            editor.setProperty("hvisualizer-original-strength", float(hop.get("strength", 1.0)))
            # Keep the last accepted display string separately from the live
            # line-edit text.  During editing ``editor.text()`` is already the
            # user's pending value, so it cannot serve as the rollback
            # snapshot when the controller rejects that value.
            editor.setProperty("hvisualizer-accepted-text", editor.text())
            proxy = QGraphicsProxyWidget()
            proxy.setWidget(editor)
            proxy.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
            proxy.setData(0, "hopping-editor")
            proxy.setData(1, row)
            editor.setProperty("hvisualizer-hop-row", row)
            editor.hoverEntered.connect(
                lambda selected_row=row: self._set_hovered_hop_row(selected_row)
            )
            editor.hoverLeft.connect(
                lambda: self._set_hovered_hop_row(None)
            )
            editor.focusEntered.connect(
                lambda selected_proxy=proxy: self._set_editor_focus_state(
                    selected_proxy, True,
                )
            )
            editor.focusLeft.connect(
                lambda selected_proxy=proxy: self._set_editor_focus_state(
                    selected_proxy, False,
                )
            )

            dx, dy = x2 - x1, -(y2 - y1)
            length = math.hypot(dx, dy)
            nx, ny = (-dy / length, dx / length) if length > 1e-12 else (0.0, -1.0)
            slot_key = (round(mx * 1000), round(my * 1000))
            slot = midpoint_slots.get(slot_key, 0)
            midpoint_slots[slot_key] = slot + 1
            side = 1.0 if slot % 2 == 0 else -1.0
            offset = 0.20 + 0.14 * (slot // 2)
            editor_anchor_x = mx + side * nx * offset
            editor_anchor_y = my + side * ny * offset

            # The coefficient rail uses a two-segment leader: a diagonal
            # pointer leaves the physical bond, then a short horizontal run
            # enters the right-hand input column.  Keeping the segments
            # separate makes the intended routing obvious and avoids the
            # misleading single bent-looking line from the old layout.
            leader_pen = _pen(
                (0.28, 0.58, 0.72) if self._dark else (0.28, 0.48, 0.68),
                0.9, Qt.DashLine,
            )
            diagonal = QGraphicsLineItem(mx, my, mx, my)
            horizontal = QGraphicsLineItem(mx, my, mx, my)
            for leader, role in ((diagonal, "diagonal"), (horizontal, "horizontal")):
                leader.setPen(leader_pen)
                leader.setOpacity(0.62)
                leader.setZValue(14.5)
                leader.setAcceptedMouseButtons(Qt.NoButton)
                leader.setData(0, f"hopping-editor-leader-{role}")
                self.addItem(leader)
                self._edit_leaders.append(leader)

            proxy.setPos(editor_anchor_x, editor_anchor_y)
            proxy.setZValue(25)
            self.addItem(proxy)
            self._edit_proxies.append(proxy)
            # Adjacent collinear bonds otherwise look like one continuous
            # line with two anonymous numeric boxes.  Keep the badge in the
            # same right-hand rail, but only for compact models where it adds
            # clarity instead of turning a dense coefficient list into a
            # second annotation layer.
            if len(editor_hops) <= 3:
                relation_text = (
                    "胞内" if (ox == 0 and oy == 0)
                    else f"胞间 {ox:+d},{oy:+d}"
                )
                # If multiple independent amplitudes share the same geometric
                # line, keep the compact relation badge but identify the
                # parameter family so the two fields are not anonymous.
                if geometry_counts.get(self._editor_geometry_key(hop), 0) > 1:
                    relation_text += f" · {hop.get('name', 't')}"
                badge = QGraphicsTextItem(relation_text)
                badge_font = QFont("Segoe UI")
                badge_font.setPointSizeF(8.0)
                badge_font.setBold(True)
                badge.setFont(badge_font)
                badge.setDefaultTextColor(
                    QColor("#1769aa") if not self._dark
                    else QColor("#8fd3ff")
                )
                badge.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
                badge.setOpacity(0.9)
                badge.setZValue(26)
                badge.setData(0, "hopping-editor-relation")
                badge.setToolTip(
                    f"跃迁 {row + 1}：{relation_text}；"
                    f"格点 {from_site_label} → {to_site_label}"
                )
                self.addItem(badge)
                self._edit_relation_badges.append((badge, proxy))
            self._edit_leader_links.append((diagonal, horizontal, mx, my, proxy))
            self._edit_proxy_anchors.append((proxy, editor_anchor_x, editor_anchor_y))

    def _ensure_editor_visible(self, proxy: QGraphicsProxyWidget) -> None:
        """Reveal a focused fixed-pixel editor in every visible lattice view.

        A proxy can sit at the edge after the user pans or changes tabs.  Qt's
        default focus handling does not always scroll a QGraphicsView for a
        child widget, so the focused field may be only half visible even when
        its stored anchor is valid.  Use the scene's own view list and a small
        pixel margin; hidden comparison views are deliberately skipped.
        """
        for view in self.views():
            if not view.isVisible() or view.viewport().size().isEmpty():
                continue
            view.ensureVisible(proxy, 12, 12)

    def _reflow_editors(self):
        """Re-run the fixed-pixel layout after a widget/DPI size change."""
        if not self._edit_proxy_anchors:
            return
        for view in self.views():
            if view.isVisible() and not view.viewport().size().isEmpty():
                self.set_zoom_level(abs(float(view.transform().m11())), view)
                return

    def _refresh_editor_sizes(self) -> None:
        """Re-measure embedded fields after a global font/UI-scale change.

        Applying a new application stylesheet changes font metrics without
        necessarily emitting a child ``resizeEvent`` when the editor already
        has a fixed size.  Re-measuring here closes that gap: the next layout
        pass grows both width and height before the proxy is positioned.
        """
        for proxy, _x, _y in self._edit_proxy_anchors:
            editor = proxy.widget()
            if editor is None:
                continue
            editor.ensurePolished()
            width = _strength_editor_width(editor, editor.text())
            metrics = editor.fontMetrics()
            height = max(26, min(44, int(metrics.height() + 12)))
            # ``QGraphicsProxyWidget`` can retain a stale fixed height while
            # the stylesheet has already enlarged the child.  Use the
            # child's actual post-polish size as the authoritative value;
            # this branch also covers style engines that add a few pixels
            # beyond QFontMetrics.height().
            height = max(height, int(editor.height()))
            if editor.width() != width or editor.height() != height:
                editor.setFixedSize(width, height)
                # QGraphicsProxyWidget normally follows its child through
                # QWidget::resizeEvent, but a global application-font swap
                # can update the fixed child before Qt delivers that event.
                # Explicitly mirror the size on the proxy as well; otherwise
                # the painted editor may be taller than its proxy bounds and
                # its lower border becomes clipped at 150–180% UI scale.
                proxy.setMinimumSize(QSizeF(float(width), float(height)))
                proxy.setMaximumSize(QSizeF(float(width), float(height)))
                proxy.resize(editor.size())
                proxy.setGeometry(QRectF(
                    proxy.geometry().topLeft(),
                    QSizeF(float(width), float(height)),
                ))
            else:
                # The child may have been resized by QStyle after the last
                # pass even though our requested dimensions are unchanged.
                # Mirror the observed widget size unconditionally so proxy
                # geometry cannot lag by a few style pixels.
                proxy.setMinimumSize(QSizeF(float(editor.width()), float(editor.height())))
                proxy.setMaximumSize(QSizeF(float(editor.width()), float(editor.height())))
                proxy.resize(editor.size())
                proxy.setGeometry(QRectF(
                    proxy.geometry().topLeft(),
                    QSizeF(float(editor.width()), float(editor.height())),
                ))

    def _set_editor_focus_state(self, proxy: QGraphicsProxyWidget, focused: bool) -> None:
        """Keep the focused field above handles and reveal it in the view."""
        # The right-hand rail keeps fields away from the physical handles.
        # Once a field owns focus, its active border and text get an
        # additional z-order lift so a nearby handle can never cover it.
        proxy.setZValue(30.0 if focused else 25.0)
        if focused:
            row = proxy.data(1)
            if row is not None:
                self._focused_hop_row = int(row)
                self._set_hovered_hop_row(int(row))
            self._ensure_editor_visible(proxy)
        else:
            row = proxy.data(1)
            widget = proxy.widget()
            if row is not None and int(row) == self._focused_hop_row:
                self._focused_hop_row = None
            # If focus moved away while the pointer is still over the field,
            # keep its leader until the normal leave event; otherwise clear it
            # immediately so a stale line does not linger after tabbing out.
            if (row is not None and int(row) == self._hovered_hop_row
                    and not (widget is not None and widget.underMouse())):
                self._set_hovered_hop_row(None)

    def set_zoom_level(self, scale: float, source=None):
        """Keep fixed-pixel editors in the right-hand rail while zooming.

        Editors deliberately stay in one predictable column rather than
        jumping among collision-search slots. The two-segment leaders retain
        each true bond midpoint, so panning/zooming changes only the view
        transform and never the logical association between a field and its
        hopping term.
        """
        if not self._edit_proxy_anchors:
            return
        # One lattice scene is shared by the dedicated lattice tab and the
        # combined matrix+lattice tab.  Their view transforms are different;
        # letting a hidden view reposition the shared proxy widgets after the
        # visible view has been fitted is what previously reintroduced
        # overlaps and clipped-looking borders.  Only the active visible view
        # is allowed to drive fixed-pixel layout.
        if source is not None and not source.isVisible():
            return
        factor = max(1e-6, abs(float(scale)))
        self._refresh_editor_sizes()
        rect = self.sceneRect()
        # The proxy ignores the view transform, so convert its *full* pixel
        # dimensions to scene units only for the scene-rect fallback.  The
        # stored anchor is the proxy's top-left (QGraphicsProxyWidget keeps
        # that convention), not its centre.  Clamping by half the size leaves
        # the lower/right half outside the scene and makes the editor appear
        # clipped at the bottom edge of the canvas.
        # Arrange fixed-pixel editors in screen-aware scene coordinates. A
        # midpoint-only layout is technically deterministic but becomes
        # unreadable when several bonds meet in one node: fields overlap one
        # another, cover handles, and hide the very line they edit. Search a
        # small set of nearby slots, keeping each field close to its bond
        # while avoiding already placed fields and editable nodes.
        node_rects = [item.sceneBoundingRect() for item in self._edit_items.values()
                      if item.isVisible()]
        entries = []
        for proxy, x, y in self._edit_proxy_anchors:
            if not proxy.isVisible():
                continue
            widget = proxy.widget()
            width_px = float(widget.width() if widget is not None else 70.0)
            height_px = float(widget.height() if widget is not None else 28.0)
            width_scene, height_scene = width_px / factor, height_px / factor
            entries.append((proxy, x, y, width_px, height_px,
                            width_scene, height_scene))

        # The coefficient rail is deliberately deterministic: every field is
        # placed in a right-hand column and connected to its bond by a
        # diagonal + horizontal dashed leader.  The previous local collision
        # search could fall back to arbitrary scene-grid slots (including the
        # bottom edge), which made editors appear detached and caused the
        # lower border to be clipped in sparse custom models.
        link_by_proxy = {
            linked_proxy: (diagonal, horizontal, mx, my)
            for diagonal, horizontal, mx, my, linked_proxy
            in self._edit_leader_links
        }
        entries_with_midpoint = []
        for proxy, x, y, width_px, height_px, width_scene, height_scene in entries:
            link = link_by_proxy.get(proxy)
            if link is None:
                continue
            entries_with_midpoint.append((
                proxy, x, y, width_px, height_px, width_scene, height_scene,
                link[2], link[3], link[0], link[1],
            ))
        badge_by_proxy = {
            proxy: badge for badge, proxy in self._edit_relation_badges
        }
        # Keep the user's row order stable; focus only changes z-order, not
        # the visual order of the right-hand coefficient list.
        self._last_editor_layout = (factor, [])
        gap_scene = 8.0 / factor
        edge_margin = 14.0 / factor
        min_x = rect.left() + edge_margin
        max_x = rect.right() - edge_margin
        # Prefer a column just outside the editable central copy.  Clamp to
        # the scene's right edge so the complete widget remains clickable.
        data_right = max(
            (float(point[0]) for point in self._data.sites),
            default=rect.left(),
        )
        if self._show_ghosts:
            ghost_xs = [float(point[0]) for point in self._data.ghost]
            if ghost_xs:
                data_right = max(data_right, max(ghost_xs))
        edit_rights = [
            item.sceneBoundingRect().right()
            for item in self._edit_items.values()
            if item.isVisible()
        ]
        node_right = max(
            [data_right + self._site_radius, *edit_rights],
            default=rect.left(),
        )
        panel_gap = 0.42
        rail_width = max(
            (item[5] for item in entries_with_midpoint),
            default=70.0 / factor,
        )
        # ``panel_right`` is the rail's right edge.  Keep the entire widget
        # (not just its right edge) outside the physical lattice; otherwise
        # the left half of a 70 px editor can still cover the last column.
        panel_left = max(node_right + panel_gap, min_x)
        panel_right = panel_left + rail_width
        if panel_right > max_x:
            panel_right = max_x
            panel_left = panel_right - rail_width
        available_height = max(1e-9, rect.height() - 2 * edge_margin)
        # Badges occupy a small fixed-pixel pocket immediately above their
        # editor.  Include that pocket in the rail calculation so the first
        # badge cannot be clipped by the scene top and the last one cannot
        # fall below the viewport after a fit or zoom.
        badge_metrics = {}
        for item in entries_with_midpoint:
            badge = badge_by_proxy.get(item[0])
            if badge is None:
                badge_metrics[item[0]] = (0.0, 0.0)
                continue
            bounds = badge.boundingRect()
            badge_metrics[item[0]] = (
                max(0.0, float(bounds.width()) / factor),
                max(0.0, float(bounds.height()) / factor),
            )
        badge_pad = 3.0 / factor
        total_height = sum(
            item[6] + (
                badge_metrics[item[0]][1] + badge_pad
                if badge_metrics[item[0]][1] > 0 else 0.0
            )
            for item in entries_with_midpoint
        )
        if entries_with_midpoint:
            natural_gap = gap_scene
            needed = total_height + natural_gap * (len(entries_with_midpoint) - 1)
            if needed <= available_height:
                rail_gap = natural_gap
                start_y = rect.center().y() - needed / 2.0
            else:
                # Dense “show all” mode gets a compact single rail rather
                # than scattered fields.  Keep a small gap whenever possible
                # and let the focused field be revealed by ensureVisible().
                rail_gap = max(1.0 / factor,
                               (available_height - total_height)
                               / max(1, len(entries_with_midpoint) - 1))
                start_y = rect.top() + edge_margin
        else:
            rail_gap = gap_scene
            start_y = rect.center().y()

        placed: list[QRectF] = []
        cursor_y = start_y
        for (proxy, _x, _y, _width_px, _height_px, width_scene, height_scene,
             mx, my, diagonal, horizontal) in entries_with_midpoint:
            _badge_width, badge_height = badge_metrics.get(proxy, (0.0, 0.0))
            badge_extra = badge_height + badge_pad if badge_height > 0 else 0.0
            max_y = rect.bottom() - edge_margin - height_scene
            py = min(
                max(cursor_y + badge_extra, rect.top() + edge_margin + badge_extra),
                max_y,
            )
            chosen = QRectF(panel_right - width_scene, py,
                            width_scene, height_scene)
            proxy.setPos(chosen.topLeft())
            badge = badge_by_proxy.get(proxy)
            if badge is not None and badge_height > 0:
                badge_width = badge_metrics[proxy][0]
                badge.setPos(
                    chosen.right() - badge_width,
                    chosen.top() - badge_height - 2.0 / factor,
                )
            # Route the diagonal leader to a bend just left of the rail, then
            # run horizontally into the field's left edge.  Both segments
            # use cosmetic dashed pens, so zooming never changes readability.
            # Keep a visible screen-space horizontal dash before the widget;
            # using a raw scene-unit constant here made it collapse to a
            # sub-pixel stub after zooming.
            bend_x = chosen.left() - 10.0 / factor
            bend_y = chosen.center().y()
            diagonal.setLine(mx, my, bend_x, bend_y)
            horizontal.setLine(bend_x, bend_y, chosen.left(), bend_y)
            placed.append(chosen)
            self._last_editor_layout[1].append((proxy, chosen, width_scene, height_scene))
            cursor_y = py + height_scene + rail_gap
        # Apply visibility only after all fields have been laid out.  In dense
        # all-fields mode this hides the crossing mesh while preserving exact
        # geometry for the one row the user is inspecting.
        self._update_edit_leader_visibility()

    def _commit_hop_strength(self, row: int, editor: QLineEdit):
        try:
            value = _parse_positive_strength(editor.text())
        except ValueError as exc:
            editor.setStyleSheet("border:1px solid #b00020;")
            self.editSelectionChanged.emit(str(exc))
            return
        original = editor.property("hvisualizer-original-strength")
        if original is not None and abs(value - float(original)) <= 1e-10:
            # Losing focus while clicking another lattice element must not
            # rebuild the scene when the user did not change the value.
            editor.setStyleSheet("")
            return
        accepted_text = editor.property("hvisualizer-accepted-text")
        if accepted_text is None:
            accepted_text = f"{float(original):.8g}" if original is not None else editor.text()
        old_state = (
            str(accepted_text), editor.width(), editor.styleSheet(), original,
        )
        self._pending_hop_edit = (int(row), editor, old_state, value)
        self.hoppingStrengthEdited.emit(int(row), value)

    def accept_hop_strength(self, row: int, value: float) -> None:
        """Finalize a strength edit after the model/controller accepts it."""
        pending = getattr(self, "_pending_hop_edit", None)
        if pending is None or pending[0] != int(row):
            return
        editor = pending[1]
        editor.setStyleSheet("")
        # Keep the committed representation predictable for subsequent edits
        # and screenshots while still preserving exact physics in the signal.
        # Eight significant digits match the initial editor representation and
        # are ample for an interactive coefficient field.  The model signal
        # still carries the full parsed float, so display shortening never
        # changes the physics.
        display = f"{value:.8g}"
        editor.setText(display)
        # A freshly committed fraction can be wider than the original value
        # used to size this proxy (for example ``1`` → ``0.333333333333``).
        # Resize before the next rail layout pass so the complete text and
        # border remain visible at every UI scale.
        editor.setFixedWidth(_strength_editor_width(editor, display))
        # Keep the beginning (especially the leading ``0.``) visible after a
        # commit.  QLineEdit otherwise places its horizontal viewport at the
        # cursor end, making a valid value look clipped even when the field is
        # wide enough to show it in full.
        editor.setCursorPosition(0)
        QTimer.singleShot(0, self._reflow_editors)
        editor.setProperty("hvisualizer-original-strength", value)
        editor.setProperty("hvisualizer-accepted-text", display)
        self._pending_hop_edit = None

    def reject_hop_strength(self, row: int) -> None:
        """Restore an edit rejected by the model/controller transaction."""
        pending = getattr(self, "_pending_hop_edit", None)
        if pending is None or pending[0] != int(row):
            return
        editor, old_state = pending[1], pending[2]
        old_text, old_width, old_style, old_original = old_state
        editor.setText(old_text)
        editor.setFixedWidth(old_width)
        editor.setStyleSheet(old_style)
        editor.setProperty("hvisualizer-original-strength", old_original)
        editor.setProperty("hvisualizer-accepted-text", old_text)
        editor.setCursorPosition(0)
        self._pending_hop_edit = None
        QTimer.singleShot(0, self._reflow_editors)
