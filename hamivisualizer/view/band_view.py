"""能带视图: E(kx) 子带绘制 (QPainterPath) + 当前 kx 标记线.

横轴使用物理波矢 ``kx``（弧度），固定为 [-π, π]。能量范围和跃迁参数
只影响纵轴，因此改变 t 不会再改变横坐标的宽度或比例。
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QPen, QFont, QPainterPath
from PySide6.QtWidgets import (
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsTextItem,
)

from .rendermodel import BandSceneData


class BandView(QGraphicsScene):
    """能带 E(kx) 视图."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: BandSceneData | None = None
        self._mark_item: QGraphicsLineItem | None = None
        self._y0 = 0.0
        self._y1 = 0.0
        self._dark = False
        self._plot_rect = QRectF()

    def set_theme(self, dark: bool):
        """切换能带视图明暗配色并重绘。"""
        self._dark = bool(dark)
        if self._data is not None:
            self.set_data(self._data)

    def set_data(self, data: BandSceneData):
        self._data = data
        self._mark_item = None
        self.clear()
        kx = np.asarray(data.kx)
        E = np.asarray(data.energies)
        if kx.size == 0 or E.ndim != 2:
            self.setSceneRect(QRectF())
            return
        if E.shape[0] != kx.size:
            raise ValueError(f"能带行数 {E.shape[0]} 与 kx 点数 {kx.size} 不一致")
        if not (np.all(np.isfinite(kx)) and np.all(np.isfinite(E))):
            raise ValueError("能带数据包含 NaN 或 Inf")
        # 横轴是固定的物理波矢，而不是按能量/参数动态归一化的坐标。
        # 预设能带使用 [-π, π]；这里仍允许少量越界数据绘制，但视图窗口
        # 始终保持固定宽度，保证 t、φ 等参数变化时横轴不跳变。
        x = kx
        x0, x1 = -np.pi, np.pi
        emin, emax = float(E.min()), float(E.max())
        pad = (emax - emin) * 0.12 or 1.0
        # Qt y 向下，绘图使用 -E，使高能量显示在上方。
        self._y0, self._y1 = -(emax + pad), -(emin - pad)
        # plot_rect 是严格的物理坐标范围；sceneRect 只负责视口裁剪。
        # 固定像素字体 (ItemIgnoresTransformations) 不能紧贴 sceneRect，
        # 否则界面字号放大后标题和 ±π 刻度会被裁掉并触发滚动条。
        self._plot_rect = QRectF(x0, self._y0, x1 - x0, self._y1 - self._y0)
        y_margin = max(0.45, (self._y1 - self._y0) * 0.07)
        self._label_y_margin = y_margin
        self.setSceneRect(self._plot_rect.adjusted(-0.55, -y_margin, 0.55, y_margin))

        # 网格 + 轴 (cosmetic: 恒像素宽)
        grid_color = QColor(42, 52, 66) if self._dark else QColor(220, 220, 220)
        axis_color = QColor(90, 107, 128) if self._dark else QColor(150, 150, 150)
        for v in (-np.pi, -np.pi / 2, 0.0, np.pi / 2, np.pi):
            grid = QGraphicsLineItem(v, self._y0, v, self._y1)
            pen = QPen(grid_color, 0.6)
            pen.setCosmetic(True)
            grid.setPen(pen)
            grid.setZValue(0)
            self.addItem(grid)
        zero = QGraphicsLineItem(x0, 0, x1, 0)
        pen = QPen(axis_color, 1.0)
        pen.setCosmetic(True)
        zero.setPen(pen)
        self.addItem(zero)

        # 子带
        curve_color = QColor(106, 160, 240) if self._dark else QColor(25, 80, 170)
        for b in range(E.shape[1]):
            path = QPainterPath()
            path.moveTo(x[0], -E[0, b])
            for k in range(1, len(x)):
                path.lineTo(x[k], -E[k, b])
            item = QGraphicsPathItem(path)
            pen = QPen(curve_color, 1.8)
            pen.setCosmetic(True)
            item.setPen(pen)
            item.setZValue(1)
            self.addItem(item)

        # 轴标签
        # 轴标签明确写成 k_x；刻度同时给出 -π、-π/2、…、π，避免
        # 用户误以为横坐标会随 t 缩放。
        axis_font = QFont()
        axis_font.setPointSizeF(9.0)
        tick_color = QColor(174, 188, 205) if self._dark else QColor(90, 90, 90)
        for value, label in (
            (-np.pi, "−π"), (-np.pi / 2, "−π/2"), (0.0, "0"),
            (np.pi / 2, "π/2"), (np.pi, "π"),
        ):
            tick = QGraphicsTextItem(label)
            tick.setFont(axis_font)
            tick.setDefaultTextColor(tick_color)
            # 坐标单位是弧度/能量，不能把 9pt 字体的像素宽高当作
            # 场景单位，否则 fitInView 后文字会膨胀成巨型黑块。
            tick.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
            tick.setPos(value - 0.12, self._y1 - 0.35)
            tick.setData(1, "axis")
            self.addItem(tick)
        xlabel = QGraphicsTextItem("k_x")
        xlabel.setFont(axis_font)
        xlabel.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        xlabel.setPos(x1 - 0.45, self._y1 - 0.85)
        xlabel.setData(1, "axis")
        self.addItem(xlabel)
        ylabel = QGraphicsTextItem("E")
        ylabel.setFont(axis_font)
        ylabel.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        ylabel.setPos(x0 + 0.08, self._y0 + 0.08)
        ylabel.setData(1, "axis")
        self.addItem(ylabel)
        if data.title:
            t = QGraphicsTextItem(data.title)
            font = QFont()
            font.setPointSizeF(10.0)
            font.setBold(True)
            t.setFont(font)
            t.setDefaultTextColor(
                QColor(220, 230, 242) if self._dark else QColor(25, 35, 48)
            )
            # 标题放在物理绘图区上方的专用留白内，不遮挡最高能带。
            t.setPos(x0 + 0.45, self._y0 - self._label_y_margin * 0.82)
            t.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
            self.addItem(t)

        # 当前 kx 标记线
        if data.kx_mark is not None:
            self.set_kx_mark(float(data.kx_mark))

    def set_message(self, message: str):
        self._data = None
        self._mark_item = None
        self.clear()
        self.setSceneRect(QRectF(0, 0, 8, 5))
        item = QGraphicsTextItem(message)
        font = QFont()
        font.setPointSizeF(10.0)
        item.setFont(font)
        item.setDefaultTextColor(QColor(120, 140, 165) if self._dark else QColor(70, 90, 120))
        item.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        item.setPos(0.5, 2.0)
        self.addItem(item)

    def set_kx_mark(self, kx: float):
        """更新当前 kx 标记线（kx 为弧度）。不重建全谱。"""
        if self._data is None or self._data.energies is None:
            return
        kx = max(-np.pi, min(np.pi, float(kx)))
        if self._mark_item is None:
            self._mark_item = QGraphicsLineItem(kx, self._y0, kx, self._y1)
            pen = QPen(QColor(255, 107, 107) if self._dark else QColor(200, 30, 30), 1.6, Qt.DashLine)
            pen.setCosmetic(True)
            self._mark_item.setPen(pen)
            self._mark_item.setZValue(2)
            self.addItem(self._mark_item)
        else:
            self._mark_item.setLine(kx, self._y0, kx, self._y1)
