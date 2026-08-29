"""Read-only secondary result pane for side-by-side model comparison."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .band_view import BandView
from .lattice_view import LatticeView
from .matrix_view import MatrixView
from .wavefunction_view import WavefunctionView
from .zoom_view import ZoomGraphicsView


RESULTS = ("矩阵+晶格", "矩阵", "晶格", "能带", "波函数")


def _view(scene):
    view = ZoomGraphicsView(scene)
    view.setToolTip("该比较视图拥有独立的缩放与平移状态")
    return view


class ComparisonPane(QWidget):
    selectionChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 0, 0, 0)
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("比较"))
        self.model_combo = QComboBox()
        self.result_combo = QComboBox()
        self.result_combo.addItems(RESULTS)
        toolbar.addWidget(self.model_combo, 1)
        toolbar.addWidget(self.result_combo)
        lay.addLayout(toolbar)

        self.matrix_scene = MatrixView()
        self.lattice_scene = LatticeView()
        self.band_scene = BandView()
        self.wf_view = WavefunctionView()
        self.matrix_view = _view(self.matrix_scene)
        self.lattice_view = _view(self.lattice_scene)
        self.band_view = _view(self.band_scene)
        self.combined_matrix = _view(self.matrix_scene)
        self.combined_lattice = _view(self.lattice_scene)
        combined = QWidget()
        combined_lay = QHBoxLayout(combined)
        combined_lay.setContentsMargins(0, 0, 0, 0)
        combined_lay.addWidget(self.combined_matrix)
        combined_lay.addWidget(self.combined_lattice)

        self.stack = QStackedWidget()
        for widget in (combined, self.matrix_view, self.lattice_view, self.band_view, self.wf_view):
            self.stack.addWidget(widget)
        lay.addWidget(self.stack, 1)
        self.empty = QLabel("选择一个已有计算结果的模型进行比较", self.stack)
        self.empty.setObjectName("comparisonEmpty")
        self.empty.setWordWrap(True)
        self.empty.hide()

        self.result_combo.currentIndexChanged.connect(self.stack.setCurrentIndex)
        self.result_combo.currentIndexChanged.connect(self.selectionChanged)
        self.model_combo.currentIndexChanged.connect(self.selectionChanged)

    def set_models(self, models: list[tuple[str, str]], selected_id: str = ""):
        old = selected_id or self.model_combo.currentData() or ""
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for model_id, name in models:
            self.model_combo.addItem(name, model_id)
        idx = self.model_combo.findData(old)
        self.model_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.model_combo.blockSignals(False)

    @property
    def selected_model_id(self) -> str:
        return str(self.model_combo.currentData() or "")

    @property
    def selected_result(self) -> str:
        return self.result_combo.currentText()

    def set_selected_result(self, name: str):
        idx = self.result_combo.findText(name)
        if idx >= 0:
            self.result_combo.setCurrentIndex(idx)

    def set_cache(self, cache: dict | None):
        if not cache:
            self.empty.setText("该模型尚未计算。切换到该模型或点击“立即计算”后即可比较。")
            self.empty.setGeometry(self.stack.rect())
            self.empty.show()
            return
        self.empty.hide()
        if cache.get("matrix") is not None:
            self.matrix_scene.set_data(cache["matrix"])
        if cache.get("lattice") is not None:
            self.lattice_scene.set_data(cache["lattice"])
        if cache.get("band") is not None:
            self.band_scene.set_data(cache["band"])
        if cache.get("wavefunction") is not None:
            self.wf_view.set_data(cache["wavefunction"])
        self.fit_current()

    def fit_current(self):
        views = {
            0: (self.combined_matrix, self.combined_lattice),
            1: (self.matrix_view,),
            2: (self.lattice_view,),
            3: (self.band_view,),
            4: (self.wf_view.view,),
        }.get(self.stack.currentIndex(), ())
        for view in views:
            rect = view.scene().sceneRect()
            if rect.isValid() and not rect.isEmpty() and view.viewport().width() > 1:
                view.fitInView(rect)

