"""视图渲染数据模型与判色/显示格式测试 (无 Qt 事件循环)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from hamivisualizer.model.symbolic import ElementFormatter
from hamivisualizer.view.lattice_view import LatticeView
from hamivisualizer.view.matrix_view import MatrixView
from hamivisualizer.view.rendermodel import (
    CellClass,
    LatticeSceneData,
    MatrixSceneData,
    Palette,
    resolve_cell_class,
)


def _app():
    return QApplication.instance() or QApplication([])


def test_resolve_cell_class():
    H = np.array([[1 + 0j, 0.5 + 0.5j], [0 + 0j, -1 + 0j]])
    assert resolve_cell_class(H, 0, 0) == CellClass.DIAG   # 对角优先
    assert resolve_cell_class(H, 0, 1) == CellClass.COMPLEX
    assert resolve_cell_class(H, 1, 0) == CellClass.ZERO
    assert resolve_cell_class(H, 1, 1) == CellClass.DIAG


def test_resolve_cell_class_physical():
    """MATLAB §5.2 物理判色: NNN 暖灰褐 / x-Bloch 和橙 / 复跃迁浅蓝."""
    H = np.zeros((3, 3), dtype=complex)
    H[0, 1] = -1.0
    assert resolve_cell_class(H, 0, 1, t=1.0, phi=np.pi / 4) == CellClass.NNN
    H[0, 1] = -2.0 * np.cos(np.pi / 4)
    assert resolve_cell_class(H, 0, 1, t=1.0, phi=np.pi / 4) == CellClass.NNSUM
    H[0, 1] = -np.exp(1j * np.pi / 4)
    assert resolve_cell_class(H, 0, 1, t=1.0, phi=np.pi / 4) == CellClass.COMPLEX
    H[0, 1] = 0.37
    assert resolve_cell_class(H, 0, 1, t=1.0, phi=np.pi / 4) == CellClass.REAL


def test_palette_colors():
    pal = Palette()
    assert pal.color(CellClass.ZERO) == pal.zero
    assert pal.color(CellClass.COMPLEX) == pal.complex_
    assert pal.color(CellClass.NNN) == pal.nnn
    assert pal.color(CellClass.NNSUM) == pal.nnsum
    assert CellClass.REAL in pal.edge_map


def test_element_formatter():
    fm = ElementFormatter(t=1.0, phi=np.pi / 4, omg=1.0)
    assert fm.format(0j) == "0"
    assert fm.format(1 + 0j) == "t"
    assert fm.format(-1 + 0j) == "-t"
    assert fm.format(2 + 0j) == "2t"
    assert fm.format(1 + 0j, is_diag=True) == "ω"
    assert fm.format(2 + 0j, is_diag=True) == "2ω"
    v = -np.exp(1j * np.pi / 4)  # -t·e^{iφ}
    assert fm.format(complex(v)) == "-t·e^{iφ}"
    # trig: e^{iφ}+e^{-iφ}=2cosφ
    c = 2 * np.cos(np.pi / 4)
    assert fm.format(c + 0j) == "2t·cosφ"
    # 兜底数值
    assert fm.format(0.37 + 0j) == "0.37"


def test_matrix_view_smoke():
    _app()
    H = np.array([[1 + 0j, -0.707 - 0.707j], [-0.707 + 0.707j, 2 + 0j]])
    data = MatrixSceneData(
        n=2,
        values=H,
        matrix=H,
        mode="smart",
        formatter=ElementFormatter(t=1.0, phi=np.pi / 4, omg=1.0),
    )
    view = MatrixView()
    view.set_data(data)
    # 2×2 色块 + 文字
    assert len(view.items()) >= 4
    assert view._data.n == 2


def test_matrix_view_numeric_no_text_large():
    """n>8 纯热图: 无单元文字项；轴标号由视口覆盖层延迟绘制."""
    _app()
    n = 10
    H = np.zeros((n, n), dtype=complex)
    np.fill_diagonal(H, 1.0)
    data = MatrixSceneData(n=n, values=H, matrix=H, mode="numeric")
    view = MatrixView()
    view.set_data(data)
    from PySide6.QtWidgets import QGraphicsTextItem

    texts = [it for it in view.items()
             if isinstance(it, QGraphicsTextItem) and it.data(1) != "axis"]
    assert len(texts) == 0
    # 轴号不再放入场景，否则放大到内部后就会离开视口；标签数据由
    # 四边冻结的 viewport overlay 持有并按当前可见区域绘制。
    axis = [it for it in view.items()
            if isinstance(it, QGraphicsTextItem) and it.data(1) == "axis"]
    assert axis == []
    assert len(view._axis_labels) == n


def test_matrix_view_html_exponents():
    """e^{...} 以 HTML 上标渲染 (模仿 MATLAB tex 观感)."""
    from hamivisualizer.model.symbolic import to_html

    assert to_html("- t e^{i\\phi}") == "- t e<sup>iφ</sup>"
    assert to_html("e^{i k_{x}}") == "e<sup>i k<sub>x</sub></sup>"
    assert to_html("e^{ik_x}") == "e<sup>ik<sub>x</sub></sup>"
    assert to_html("\\omega + t") == "ω + t"


def test_to_math_html_uses_math_face_and_unicode_minus():
    from hamivisualizer.model.symbolic import to_math_html

    rendered = to_math_html("-t e^{ik_{x}}")
    assert "Cambria Math" in rendered
    # A nested HTML <sub> inside <sup> is laid out incorrectly by Qt rich
    # text (x drops below the exponent).  The renderer uses the mathematical
    # Unicode subscript inside one raised exponent run instead.
    assert "e<sup>ikₓ</sup>" in rendered
    assert "−t" in rendered


@pytest.mark.parametrize("source", (
    r"e^{ik_x}",
    r"e^{i k_{x}}",
    r"-t\cdot e^{-i\phi}",
    r"t_{12} + \omega_{0}",
    r"e^{2ik_{x}} - e^{-2ik_{x}}",
))
def test_matrix_math_layout_accepts_the_supported_tex_script_subset(source):
    """矩阵渲染器应稳定处理常见上下标与嵌套指数，不依赖 HTML 基线。"""
    from hamivisualizer.view.math_text import MathLayout, math_font

    layout = MathLayout(source)
    metrics = layout.metrics(math_font(24))
    assert metrics.width > 0
    assert metrics.height > 0


def test_nested_bloch_subscript_stays_high_inside_raised_exponent():
    """The x in e^{ik_x} must remain visually attached to k, not the base line."""
    from PySide6.QtGui import QFontMetricsF
    from hamivisualizer.view.math_text import MathLayout, math_font

    font = math_font(24)
    layout = MathLayout(r"e^{ik_{x}}")
    outer = next(node for node in layout.lines[0].children
                 if node.__class__.__name__ == "_Script")
    inner = next(node for node in outer.superscript.children
                 if node.__class__.__name__ == "_Script")
    outer_raise = outer._offsets(font)[0]
    inner_drop = inner._offsets(outer._script_font(font))[1]
    assert inner_drop > 0
    # The complete exponent remains materially above the main e baseline.
    assert outer_raise + inner_drop < -0.30 * QFontMetricsF(font).ascent()


def test_math_script_draw_origin_matches_reserved_compact_width():
    """上下标的实际绘制范围必须与单元格预留宽度一致，避免边界裁切。"""
    from hamivisualizer.view.math_text import MathLayout, math_font

    _app()
    font = math_font(24)
    layout = MathLayout(r"e^{-ik_{x}}")
    outer = next(node for node in layout.lines[0].children
                 if node.__class__.__name__ == "_Script")
    script_font = outer._script_font(font)
    base_width = outer.base.box(font).width
    script_width = outer._script_width(script_font)

    # The compact 0.94 reservation and the draw origin must describe the
    # same right edge.  The old implementation drew at base_width, extending
    # six percent beyond the QGraphicsItem boundingRect.
    assert outer.box(font).width == pytest.approx(
        base_width + script_width * 0.94,
    )
    assert outer._script_origin(font) == pytest.approx(
        base_width - script_width * 0.06,
    )
    assert outer._script_origin(font) + script_width == pytest.approx(
        outer.box(font).width,
    )


def test_lattice_view_smoke():
    _app()
    data = LatticeSceneData(
        sites=((0, 0, "0", "A"), (0, 1, "1", "B"), (1, 1, "2", "A"), (1, 0, "3", "B")),
        edges=((0, 1, "NN"), (1, 2, "NN"), (2, 3, "NN"), (3, 0, "NN")),
        semi=False,
        ghost=(),
        cell_boxes=((0, 0, 2, 2),),
        title="NP",
    )
    view = LatticeView()
    view.set_data(data)
    assert len(view.items()) > 0


def test_lattice_view_semi_ghost():
    """半无限模式: 虚影格点占位 (无内容也可)."""
    _app()
    data = LatticeSceneData(
        sites=((0, 0, "0", "A"), (0, 1, "1", "B")),
        edges=(),
        semi=True,
        ghost=((-1, 0, "0"), (2, 0, "0")),
        cell_boxes=(),
    )
    view = LatticeView()
    view.set_data(data)
    assert len(view.items()) > 0


def test_lattice_scene_data_keeps_historical_positional_field_order():
    """Adding ghost metadata must not remap plug-in positional DTO calls."""
    data = LatticeSceneData(
        ((0, 0, "0", "A"),), (), (), True, (), (), (), "legacy",
    )
    assert data.title == "legacy"
    assert data.ghost_sites == ()
