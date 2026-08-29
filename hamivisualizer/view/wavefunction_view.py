"""OBC 波函数视图: |ψ|² 实空间热图 + 本征态选择."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRectF
from PySide6.QtGui import QColor, QBrush, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QGraphicsEllipseItem,
    QGraphicsScene,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..model.hamiltonian import edge_mask_for_positions
from .rendermodel import WfSceneData
from .zoom_view import ZoomGraphicsView


def _hot_color(t: float) -> QColor:
    """hot 色图: 蓝→青→黄→红 (0..1)."""
    t = max(0.0, min(1.0, t))
    if t < 0.25:
        k = t / 0.25
        return QColor(int(30 + 100 * k), int(40 * k), int(180 - 100 * k))
    if t < 0.5:
        k = (t - 0.25) / 0.25
        return QColor(int(130 + 100 * k), int(40 + 130 * k), 80)
    if t < 0.75:
        k = (t - 0.5) / 0.25
        return QColor(int(230 + 25 * k), int(170 + 60 * k), 0)
    k = (t - 0.75) / 0.25
    return QColor(255, int(230 - 160 * k), 0)


class WavefunctionView(QWidget):
    """|ψ|² 热图控件 (含态选择 QComboBox)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: WfSceneData | None = None
        self._state = 0
        self._requested_energy: float | None = None
        self._dark = False
        lay = QVBoxLayout(self)
        top = QVBoxLayout()
        selector_row = QHBoxLayout()
        selector_row.addWidget(QLabel("本征态"))
        self.combo = QComboBox()
        self.prev_btn = QPushButton("◀")
        self.next_btn = QPushButton("▶")
        self.info = QLabel("")
        # The physical diagnostic can be long at 150% UI scale. Let it take
        # the remaining width and wrap instead of letting a rigid one-line
        # size hint push its right end outside the view.
        self.info.setWordWrap(True)
        self.info.setMinimumWidth(0)
        self.info.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        selector_row.addWidget(self.combo)
        selector_row.addWidget(self.prev_btn)
        selector_row.addWidget(self.next_btn)
        selector_row.addStretch()
        top.addLayout(selector_row)
        # Keep the diagnostic on its own full-width line. In a single
        # toolbar row it had to compete with the state picker and could lose
        # the final classification text at 150% scale.
        top.addWidget(self.info)
        lay.addLayout(top)
        self.scene = QGraphicsScene()
        # 与矩阵/晶格/能带统一交互：滚轮缩放、左键拖拽、0 适应窗口。
        self.view = ZoomGraphicsView(self.scene)
        self.view.setToolTip("滚轮缩放 · 左键拖拽平移 · +/- 缩放 · 0 适应窗口")
        lay.addWidget(self.view)

        self.combo.currentIndexChanged.connect(self._on_combo)
        self.prev_btn.clicked.connect(lambda: self._step(-1))
        self.next_btn.clicked.connect(lambda: self._step(1))

    def set_data(self, data: WfSceneData):
        energies = np.asarray(data.energies)
        wf = np.asarray(data.wf)
        if energies.ndim != 1 or wf.ndim != 2 or wf.shape[1] != energies.size:
            raise ValueError(
                f"波函数数据形状不一致: energies={energies.shape}, wf={wf.shape}"
            )
        if wf.shape[0] != len(data.positions):
            raise ValueError("波函数行数必须与格点位置数量一致")
        if not np.all(np.isfinite(energies)) or not np.all(np.isfinite(wf)):
            raise ValueError("波函数数据包含 NaN 或 Inf")
        if np.any(wf < -1e-12):
            raise ValueError("波函数概率密度不能为负")
        # 视图内部统一保存 ndarray，避免外部 DTO 传入 list 时在后续
        # ``.size``/``.ndim`` 访问处出现类型分支。
        self._data = WfSceneData(
            energies=np.asarray(energies, dtype=float),
            wf=np.asarray(wf, dtype=float),
            positions=tuple(data.positions),
            title=data.title,
        )
        self._state = 0
        self._requested_energy = None
        self.combo.blockSignals(True)
        self.combo.clear()
        for i, e in enumerate(energies):
            self.combo.addItem(f"#{i}  E={e:.3f}")
        self.combo.blockSignals(False)
        if energies.size:
            self.combo.setCurrentIndex(0)
        self._draw()

    def set_theme(self, dark: bool):
        """切换波函数视图明暗（格点描边）并重绘。"""
        self._dark = bool(dark)
        if self._data is not None:
            self._draw()

    def set_loading(self, message: str):
        self._data = None
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.blockSignals(False)
        self.scene.clear()
        self.info.setText(message)

    # ---- 内部 ----

    def _on_combo(self, idx):
        if idx >= 0:
            self._state = idx
            # A direct state-picker choice is no longer an approximation to
            # a requested energy; omit the ΔE wording from its summary.
            self._requested_energy = None
            self._draw()

    def _step(self, d: int):
        n = self.combo.count()
        if n:
            self.combo.setCurrentIndex((self._state + d) % n)

    def select_energy(self, energy: float) -> int | None:
        """选择能量最接近 ``energy`` 的本征态并重绘。

        对角化得到的本征态按能量升序排列；UI 输入的是物理能量而非
        0-based 索引，因此这里集中处理“能量 → 态编号”的映射，避免
        控制器或绘图层各自重复实现并产生偏差。
        """
        if self._data is None or self._data.energies is None:
            return None
        energies = np.asarray(self._data.energies, dtype=float)
        if energies.size == 0:
            return None
        target = float(energy)
        if not np.isfinite(target):
            raise ValueError("目标能量必须是有限数值")
        idx = int(np.argmin(np.abs(energies - target)))
        self._state = idx
        self._requested_energy = target
        self.combo.blockSignals(True)
        self.combo.setCurrentIndex(idx)
        self.combo.blockSignals(False)
        self._draw()
        return idx

    @property
    def selected_energy(self) -> float | None:
        if self._data is None:
            return None
        energies = np.asarray(self._data.energies)
        if energies.size == 0:
            return None
        return float(energies[self._state])

    def _draw(self):
        self.scene.clear()
        data = self._data
        if data is None or data.wf is None or data.wf.ndim != 2:
            self.info.setText("")
            return
        if self._state >= data.wf.shape[1]:
            self.info.setText("")
            return
        wf = np.asarray(data.wf)  # (Nat, Nat), 列 = 态
        col = wf[:, self._state]
        pos = list(data.positions)
        if not pos:
            self.info.setText(f"能量 E = {data.energies[self._state]:.4f}")
            return
        # 除能量外给出可解释的边界局域指标，避免用户凭“似乎居中”
        # 猜测。边界掩膜与求解器处理严格简并态时使用同一几何定义。
        edge_mask = edge_mask_for_positions(pos)
        total = float(np.sum(col))
        edge_weight = float(np.sum(col[edge_mask]) / total) if total > 1e-15 else 0.0
        baseline = float(np.mean(edge_mask))
        enrichment = edge_weight / baseline if baseline > 1e-15 else 0.0
        if enrichment >= 1.45:
            character = "边界局域态"
        elif enrichment <= 0.70:
            character = "体内富集态"
        else:
            character = "体态 / 未见明显边界局域"
        selected = float(data.energies[self._state])
        if self._requested_energy is None:
            energy_text = f"本征态 #{self._state}，E = {selected:.5g}"
        else:
            delta = abs(selected - self._requested_energy)
            energy_text = (
                f"目标 E = {self._requested_energy:.5g} → #{self._state}，"
                f"E = {selected:.5g}，ΔE = {delta:.2g}"
            )
        summary = (
            f"{energy_text}  |  边界 {edge_weight:.1%}（均匀基线 {baseline:.1%}，"
            f"富集 {enrichment:.2f}×）  |  {character}"
        )
        self.info.setText(summary)
        self.info.setToolTip(summary)
        xs = [p[0] for p in pos]
        ys = [-p[1] for p in pos]
        pad = 0.6
        self.scene.setSceneRect(QRectF(min(xs) - pad, min(ys) - pad,
                                       max(xs) - min(xs) + 2 * pad, max(ys) - min(ys) + 2 * pad))
        r = 0.42
        scale = float(np.max(col)) or 1.0
        for i, (x, y) in enumerate(pos):
            y = -y
            t = min(1.0, float(col[i]) / scale)
            circ = QGraphicsEllipseItem(x - r, y - r, 2 * r, 2 * r)
            circ.setBrush(QBrush(_hot_color(t)))
            pen = QPen(QColor(255, 255, 255, 60) if self._dark else QColor(0, 0, 0, 60), 0.3)
            pen.setCosmetic(True)
            circ.setPen(pen)
            circ.setToolTip(f"site {i}\n|ψ|² = {float(col[i]):.6g}")
            self.scene.addItem(circ)
