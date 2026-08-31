"""可缩放的 QGraphicsView.

滚轮负责以鼠标位置为锚点缩放，左键拖拽负责平移；场景若提供
``set_zoom_level``，会收到当前视口缩放比例，用于按实际像素大小切换
矩阵单元文字等细节层。
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QGraphicsItem, QGraphicsView


class ZoomGraphicsView(QGraphicsView):
    """带平移、滚轮缩放和快捷键缩放的图形视图。"""

    zoomChanged = Signal(float)

    # 缩放上限/下限按「适应窗口」时的基准比例计算，而不是使用
    # 场景坐标中的绝对比例。晶格场景通常只有几个坐标单位，fit 后
    # 的 Qt 变换可能是 100× 以上；若仍用绝对 MAX_SCALE=20，第一次
    # 滚轮放大就会被夹回 20×，表现为视图突然缩小（矩阵场景因坐标
    # 较大暂时看不出这个问题）。
    MIN_ZOOM_RATIO = 0.10
    MAX_ZOOM_RATIO = 64.0
    # 兼容 0.3 版本中暴露的常量名；实际限制已改为相对 fit 基准。
    MIN_SCALE = 0.15
    MAX_SCALE = 20.0
    WHEEL_STEP = 1.18

    def __init__(self, scene=None, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        # 不使用 Qt 的 ScrollHandDrag：它在按下瞬间就进入“抓手”状态，
        # 矩阵元点击弹出详情框后，轻微鼠标移动也会被误认为平移。
        # 改为带 6px 阈值的手动平移，纯点击始终只触发场景点击。
        self.setDragMode(QGraphicsView.NoDrag)
        self._pan_press_pos = None
        self._pan_press_scroll = None
        self._pan_active = False
        self._pan_button = None
        # Hover-driven disclosures (for example the single coefficient
        # leader in a dense lattice rail) must receive pointer moves even
        # when no mouse button is pressed.  QAbstractScrollArea keeps mouse
        # tracking disabled by default, which made the interaction appear to
        # work only in synthetic tests that targeted child widgets directly.
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setInteractive(True)
        self._user_zoomed = False
        # 最近一次 fitInView 得到的「窗口基准」缩放比例。
        self._reference_scale = 1.0
        self._notify_scene_zoom()

    @property
    def user_zoomed(self) -> bool:
        return self._user_zoomed

    def reset_user_zoom(self) -> None:
        """让下一次 fit 重新成为视口基准。"""
        self._user_zoomed = False

    def fitInView(self, rect, aspectRatioMode=Qt.KeepAspectRatio):  # noqa: N802
        result = super().fitInView(rect, aspectRatioMode)
        scale = abs(float(self.transform().m11()))
        if scale > 1e-12:
            self._reference_scale = scale
        self._notify_scene_zoom()
        return result

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._notify_scene_zoom()

    def paintEvent(self, event):
        super().paintEvent(event)
        scene = self.scene()
        draw_overlay = getattr(scene, "draw_viewport_overlay", None)
        if draw_overlay is not None:
            painter = QPainter(self.viewport())
            painter.setRenderHint(QPainter.TextAntialiasing)
            draw_overlay(painter, self)
            painter.end()

    def mousePressEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.MiddleButton):
            scene = self.scene()
            if event.button() == Qt.LeftButton and scene is not None:
                hit = self.itemAt(event.position().toPoint())
                hit_kind = hit.data(0) if hit is not None else None
                editor_hit = hit_kind == "hopping-editor"
                # Bond guides and semi-infinite ghost endpoints are
                # deliberate click targets, not canvas anchors.  Treating
                # them as interactive prevents a small pointer wobble after
                # a click from starting a pan and swallowing the requested
                # editor/endpoint action.
                topology_hit = hit_kind in {
                    "hopping-guide", "ghost-endpoint", "wavefunction-site",
                }
                movable_hit = (
                    getattr(scene, "edit_mode", False)
                    and hit is not None and (
                        hit.flags() & QGraphicsItem.ItemIsMovable
                        or (hit.parentItem() is not None and
                            hit.parentItem().flags() & QGraphicsItem.ItemIsMovable)
                    )
                )
                matrix_hit_test = getattr(scene, "is_interactive_position", None)
                matrix_hit = False
                # Keep scene-specific hit testing independent from the
                # concrete graphics item type.  MatrixView's large raster
                # path has no per-cell QGraphicsRectItem, yet a click must
                # still select a cell without arming a pan.
                if callable(matrix_hit_test):
                    try:
                        matrix_hit = bool(matrix_hit_test(
                            self.mapToScene(event.position().toPoint())
                        ))
                    except (TypeError, ValueError):
                        matrix_hit = False
                if editor_hit or topology_hit or movable_hit or matrix_hit:
                    # A coefficient field is an interactive child widget, not
                    # empty canvas.  Do not arm the manual pan gesture while
                    # pressing it; a tiny hand movement during text editing
                    # must never drag the lattice underneath the user.
                    self._pan_press_pos = None
                    self._pan_press_scroll = None
                    self._pan_active = False
                    self._pan_button = None
                    super().mousePressEvent(event)
                    return
            self._pan_press_pos = event.position()
            self._pan_press_scroll = (
                self.horizontalScrollBar().value(),
                self.verticalScrollBar().value(),
            )
            self._pan_active = False
            self._pan_button = event.button()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # Embedded lattice editors use fixed-pixel proxy widgets.  Resolve
        # their visible screen footprint before forwarding the event so a
        # dense coefficient rail reveals the field actually under the
        # pointer, even when Qt's scene-space proxy bounds overlap.
        scene = self.scene()
        hover_handler = getattr(scene, "handle_viewport_hover", None)
        if callable(hover_handler) and not event.buttons():
            try:
                hover_handler(self, event.position().toPoint())
            except (AttributeError, TypeError, ValueError):
                # Hover disclosure is auxiliary UI; never let it interfere
                # with normal panning or the underlying scene interaction.
                pass
        if (
            self._pan_press_pos is not None
            and self._pan_button is not None
            and (event.buttons() & self._pan_button)
        ):
            delta = event.position() - self._pan_press_pos
            if not self._pan_active and delta.manhattanLength() >= 6:
                self._pan_active = True
                self.viewport().setCursor(Qt.ClosedHandCursor)
            if self._pan_active:
                h0, v0 = self._pan_press_scroll
                self.horizontalScrollBar().setValue(round(h0 - delta.x()))
                self.verticalScrollBar().setValue(round(v0 - delta.y()))
                event.accept()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() in (Qt.LeftButton, Qt.MiddleButton):
            self._pan_press_pos = None
            self._pan_press_scroll = None
            self._pan_active = False
            self._pan_button = None
            self.viewport().unsetCursor()
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if not delta:
            event.ignore()
            return
        old_scale = self._current_scale()
        factor = self.WHEEL_STEP ** (delta / 120.0)
        target = self._clamp_scale(old_scale * factor)
        factor = target / old_scale
        if abs(factor - 1.0) <= 1e-12:
            event.accept()
            return

        # AnchorUnderMouse keeps the cell beneath the cursor fixed while zooming.
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.scale(factor, factor)
        self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self._user_zoomed = True
        self._notify_scene_zoom()
        event.accept()

    def keyPressEvent(self, event):
        scene = self.scene()
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            delete_selected = getattr(scene, "delete_selected", None)
            if delete_selected is not None and delete_selected():
                event.accept()
                return
        if event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self._scale_by(self.WHEEL_STEP)
            event.accept()
            return
        if event.key() in (Qt.Key_Minus, Qt.Key_Underscore):
            self._scale_by(1.0 / self.WHEEL_STEP)
            event.accept()
            return
        if event.key() == Qt.Key_0:
            self.reset_user_zoom()
            rect = self.scene().sceneRect() if self.scene() is not None else None
            if rect is not None and rect.isValid() and not rect.isEmpty():
                self.fitInView(rect, Qt.KeepAspectRatio)
            event.accept()
            return
        super().keyPressEvent(event)

    def _scale_by(self, factor: float) -> None:
        old_scale = self._current_scale()
        target = self._clamp_scale(old_scale * factor)
        self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.scale(target / old_scale, target / old_scale)
        self._user_zoomed = True
        self._notify_scene_zoom()

    def _current_scale(self) -> float:
        """当前绝对缩放比例，始终返回一个可用于除法的正数。"""
        scale = abs(float(self.transform().m11()))
        return scale if scale > 1e-12 else max(self._reference_scale, 1.0)

    def _clamp_scale(self, scale: float) -> float:
        """按 fit 基准限制缩放，避免小坐标场景触发绝对上限跳变。"""
        base = self._reference_scale if self._reference_scale > 1e-12 else 1.0
        return min(base * self.MAX_ZOOM_RATIO,
                   max(base * self.MIN_ZOOM_RATIO, float(scale)))

    def _notify_scene_zoom(self) -> None:
        scale = abs(float(self.transform().m11()))
        if scale <= 1e-12:
            scale = 1.0
        scene = self.scene()
        if scene is not None and hasattr(scene, "set_zoom_level"):
            scene.set_zoom_level(scale, self)
        self.zoomChanged.emit(scale)
