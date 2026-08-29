"""视图渲染数据模型 (DTO) 与判色单源.

controller 把任意晶格归一化成不可变 DTO, 视图只消费 DTO —— 视图零模型依赖、
可独立单测, 这是泛化的关键。

配色与判色规则忠实移植 MATLAB draw_matrix (§5.2):
  零灰 / 对角黄 / NNN 实跃迁暖灰褐 / x-Bloch 和橙 / 复跃迁浅蓝。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


class CellClass:
    """矩阵元结构/物理分类 (判色)."""

    ZERO = "zero"        # 零: 灰
    DIAG = "diag"        # 对角/onsite: 黄
    NNN = "nnn"          # 实数 = ±n·t (NNN 实跃迁): 暖灰褐
    NNSUM = "nnsum"      # 实数 = ±2t·cosφ (x-Bloch 和): 橙
    REAL = "real"        # 其他实数: 浅蓝
    COMPLEX = "complex"  # 复数 (NN 复跃迁): 浅蓝


@dataclass(frozen=True)
class Palette:
    """配色 (RGB 0-1). 对应 MATLAB §5.2 类型配色."""

    zero: tuple = (0.94, 0.94, 0.94)      # cZero 灰
    diag: tuple = (0.95, 0.95, 0.80)      # cDiag 黄
    nnn: tuple = (0.90, 0.80, 0.75)       # cNNN 暖灰褐
    nnsum: tuple = (1.0, 0.82, 0.60)      # cNNsum 橙
    real: tuple = (0.70, 0.85, 1.0)       # cNN 浅蓝
    complex_: tuple = (0.70, 0.85, 1.0)   # cNN 浅蓝
    edge: tuple = (0.65, 0.65, 0.65)

    site_a: tuple = (0.20, 0.55, 0.80)    # A 子格
    site_b: tuple = (0.90, 0.40, 0.20)    # B 子格
    site_ghost: tuple = (0.55, 0.60, 0.65)
    edge_nn: tuple = (0.80, 0.15, 0.15)   # NN 键 (红)
    edge_nnn: tuple = (0.15, 0.60, 0.20)  # NNN 键 (绿)

    # 逐类边框色覆盖 (深色主题用); None 时回退默认值.
    edge_zero: tuple = None
    edge_diag: tuple = None
    edge_nnn_c: tuple = None
    edge_nnsum: tuple = None
    edge_real: tuple = None
    edge_complex: tuple = None

    @property
    def edge_map(self) -> dict:
        """每类矩阵元的边框色 (MATLAB cEdge 逐类)."""
        return {
            CellClass.ZERO: self.edge_zero or (0.85, 0.85, 0.85),
            CellClass.DIAG: self.edge_diag or (0.75, 0.75, 0.55),
            CellClass.NNN: self.edge_nnn_c or (0.60, 0.50, 0.40),
            CellClass.NNSUM: self.edge_nnsum or (0.80, 0.50, 0.20),
            CellClass.REAL: self.edge_real or (0.50, 0.60, 0.80),
            CellClass.COMPLEX: self.edge_complex or (0.50, 0.60, 0.80),
        }

    def color(self, cls: str) -> tuple:
        return {
            CellClass.ZERO: self.zero,
            CellClass.DIAG: self.diag,
            CellClass.NNN: self.nnn,
            CellClass.NNSUM: self.nnsum,
            CellClass.REAL: self.real,
            CellClass.COMPLEX: self.complex_,
        }[cls]


# 深色主题矩阵配色: 保持各物理类别的色相，整体压暗、以浅色文字可读。
DARK_PALETTE = Palette(
    zero=(0.15, 0.18, 0.23),      # 深灰蓝
    diag=(0.33, 0.30, 0.16),      # 暗金
    nnn=(0.31, 0.25, 0.21),       # 暗暖褐
    nnsum=(0.40, 0.28, 0.14),     # 暗橙
    real=(0.15, 0.27, 0.42),      # 暗蓝
    complex_=(0.15, 0.27, 0.42),  # 暗蓝
    edge_zero=(0.28, 0.32, 0.39),
    edge_diag=(0.46, 0.43, 0.25),
    edge_nnn_c=(0.44, 0.36, 0.30),
    edge_nnsum=(0.58, 0.40, 0.22),
    edge_real=(0.34, 0.45, 0.62),
    edge_complex=(0.34, 0.45, 0.62),
)


@dataclass(frozen=True)
class MatrixSceneData:
    """矩阵视图 DTO.

    values: 数值矩阵 (np.ndarray), 判色/尺寸用。
    matrix: 原始矩阵 (np.ndarray 或 sympy.Matrix), 取元素显示。
    mode:   'numeric' | 'smart' | 'symbolic' 显示模式。
    sites:  行/列标签 (格点序号串)。
    formatter: smart 模式用的 ElementFormatter (含当前参数值)。
    t/phi: 物理判色参数 (MATLAB §5.2); None 时退化为结构分类。
    """

    n: int
    values: Any
    matrix: Any = None
    mode: str = "smart"
    sites: tuple = ()
    formatter: Any = None
    title: str = ""
    t: Any = None
    phi: Any = None


@dataclass(frozen=True)
class LatticeSceneData:
    """晶格视图 DTO.

    sites:     [(x, y, label, sublattice)], 实空间格点。
    edges:     [(i, j, kind)], kind ∈ {'NN','NNN'} 决定实线/虚线。
    ghost_edges: [(x1, y1, x2, y2, kind)] 半无限虚影键 (黯淡)。
    semi:      半无限模式 (画 x-Bloch 虚影)。
    ghost:     [(x, y, label)], 半无限左右虚影格点。
    cell_boxes: 元胞框 [(x, y, w, h)], 首元素 = 首胞 (高亮)。
    boundary_outline: 非矩形有限盘的真实物理外轮廓（闭合点列）。
    """

    sites: tuple = ()
    edges: tuple = ()
    ghost_edges: tuple = ()
    semi: bool = False
    ghost: tuple = ()
    cell_boxes: tuple = ()
    cell_polygons: tuple = ()
    title: str = ""
    # Optional logical metadata for semi-infinite ghost nodes.  Keep this
    # field last so positional construction of the historical DTO remains
    # source-compatible.  Each entry is ``(x, y, label,
    # primitive_site_index, cell_dx, cell_dy)``.  Offsets are relative to the
    # editable central-cell copy (not to the displayed label number).
    ghost_sites: tuple = ()
    # Optional physical outline for a finite non-rectangular mask.  The
    # points are ordered for a closed polyline and stay last for positional
    # compatibility with older plug-ins constructing this DTO directly.
    boundary_outline: tuple = ()


@dataclass(frozen=True)
class BandSceneData:
    """能带视图 DTO."""

    kx: Any = None
    energies: Any = None
    kx_mark: Any = None   # 当前 kx（弧度）的标记线; None 不画
    title: str = ""


@dataclass(frozen=True)
class WfSceneData:
    """OBC 波函数视图 DTO."""

    energies: Any = None       # (Nat,)
    wf: Any = None             # (Nat, Nat) |ψ|², 列 = 态
    positions: tuple = ()      # 每格点实空间坐标
    title: str = ""


def resolve_cell_class(values, i: int, j: int, t=None, phi=None, tol: float = 1e-8) -> str:
    """矩阵元结构/物理分类 (判色单源, MATLAB §5.2 泛化).

    t/phi 提供时做物理分类: 实数 |v|≈n·t → NNN, |v|≈2t·cosφ → NNSUM;
    否则退化为结构分类 (零/对角/实/复)。
    """
    v = complex(values[i, j])
    if abs(v) < tol:
        return CellClass.ZERO
    if i == j:
        return CellClass.DIAG
    if abs(v.imag) < tol:
        if t is not None:
            tt = abs(t)
            is_nnn = any(abs(abs(v) - n * tt) < tol for n in range(1, 5))
            if phi is not None:
                is_nnsum = abs(abs(v) - 2 * tt * math.cos(phi)) < tol
            else:
                is_nnsum = False
            if is_nnn and not is_nnsum:
                return CellClass.NNN
            if is_nnsum:
                return CellClass.NNSUM
        return CellClass.REAL
    return CellClass.COMPLEX


def pick_cell_color(palette: Palette, values, i: int, j: int) -> tuple:
    """色块配色 (结构分类)."""
    return palette.color(resolve_cell_class(values, i, j))
