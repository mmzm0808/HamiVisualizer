"""晶格元胞与索引映射.

把 MATLAB 版的「坐标驱动双循环 + 硬编码 A/B 子格相位表」抽象为通用数据模型:
Lattice 只描述「一个元胞里有哪几个格点、平移向量多大」; 具体跃迁关系由
model.hopping.HoppingTerm 描述。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterator


@dataclass(frozen=True)
class Site:
    """元胞内一个格点.

    r:         胞内索引 0..N-1 (== 用户表格的行序)
    x, y:      胞内绝对坐标, 位于 [0, Lx) × [0, Ly)
    sublattice: 'A'/'B'/None, 仅用于显示配色, 不参与计算
    """

    r: int
    x: float
    y: float
    sublattice: str | None = None


@dataclass
class Lattice:
    """元胞: 格点表 + 平移向量 (Lx, Ly).

    Lx/Ly 为 None 时按「格点落在单位整数网格」自动推断: (max - min) + 1。
    用户可显式指定, 以支持任意实数坐标的平移向量。
    """

    sites: list[Site]
    Lx: float | None = None
    Ly: float | None = None
    # 任意二维 Bravais 元胞。留空时完全兼容旧的正交 (Lx, 0)/(0, Ly) 表示。
    a1: tuple[float, float] | None = None
    a2: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if not self.sites:
            raise ValueError("晶格至少需要一个格点")
        for expected, site in enumerate(self.sites):
            if site.r != expected:
                raise ValueError(
                    f"Site.r 必须连续且与列表位置一致: 第 {expected} 项得到 r={site.r}"
                )
            if not (math.isfinite(float(site.x)) and math.isfinite(float(site.y))):
                raise ValueError(f"格点 {expected} 坐标必须是有限数值")
        if (self.a1 is None) != (self.a2 is None):
            raise ValueError("a1/a2 必须同时提供")
        if self.a1 is None and (self.Lx is None or self.Ly is None):
            lx, ly = self._infer_cell_size()
            if self.Lx is None:
                self.Lx = lx
            if self.Ly is None:
                self.Ly = ly
        if self.a1 is None:
            if not (math.isfinite(float(self.Lx)) and math.isfinite(float(self.Ly))):
                raise ValueError("元胞平移向量必须是有限数值")
            if self.Lx <= 0 or self.Ly <= 0:
                raise ValueError(f"元胞平移向量必须为正, 得到 Lx={self.Lx}, Ly={self.Ly}")
            self.a1 = (float(self.Lx), 0.0)
            self.a2 = (0.0, float(self.Ly))
        else:
            self.a1 = tuple(map(float, self.a1))
            self.a2 = tuple(map(float, self.a2))
            if not all(math.isfinite(v) for v in (*self.a1, *self.a2)):
                raise ValueError("元胞矢量必须是有限数值")
            det = self.a1[0] * self.a2[1] - self.a1[1] * self.a2[0]
            if abs(det) < 1e-12:
                raise ValueError("a1/a2 不可共线")
            # 保留旧接口：半无限 Bloch 方向使用 a1，截断高度使用 a2 的 y 分量。
            self.Lx = math.hypot(*self.a1)
            self.Ly = abs(self.a2[1])
            if self.Lx <= 0 or self.Ly <= 0:
                raise ValueError("半无限显示要求 a1 非零且 a2 具有非零 y 分量")
        coords: set[tuple[float, float]] = set()
        det = self.a1[0] * self.a2[1] - self.a1[1] * self.a2[0]
        for site in self.sites:
            # 用晶格基矢的分数坐标验证胞内位置，而不是强行限定矩形。
            u = (site.x * self.a2[1] - site.y * self.a2[0]) / det
            v = (self.a1[0] * site.y - self.a1[1] * site.x) / det
            if not (-1e-10 <= u < 1 - 1e-10 and -1e-10 <= v < 1 - 1e-10):
                raise ValueError(
                    f"格点 {site.r} 坐标 ({site.x}, {site.y}) 必须位于 a1/a2 原胞内"
                )
            key = (round(float(site.x), 12), round(float(site.y), 12))
            if key in coords:
                raise ValueError(f"存在重复格点坐标: ({site.x}, {site.y})")
            coords.add(key)

    def _infer_cell_size(self) -> tuple[float, float]:
        if not self.sites:
            return 1.0, 1.0
        xs = [s.x for s in self.sites]
        ys = [s.y for s in self.sites]
        # 假设格点落在单位整数网格: 尺寸 = (max - min) + 1
        return max(xs) - min(xs) + 1.0, max(ys) - min(ys) + 1.0

    @property
    def N(self) -> int:
        return len(self.sites)

    def position(self, cx: int, cy: int, r: int) -> tuple[float, float]:
        """实空间绝对坐标: 胞 (cx, cy) 内格点 r 的位置."""
        s = self.sites[r]
        return (
            cx * self.a1[0] + cy * self.a2[0] + s.x,
            cx * self.a1[1] + cy * self.a2[1] + s.y,
        )

    def cells(self, NX: int, NY: int) -> Iterator[tuple[int, int]]:
        """迭代 NX×NY 个胞坐标 (cx, cy)."""
        for cx in range(NX):
            for cy in range(NY):
                yield (cx, cy)

    def count_sites(self, boundary) -> int:
        """按边界形式返回总格点数 (与 MATLAB count_sites 对拍).

        SEMI: N·NY (x-Bloch 不增加格点)
        OBC : N·NX·NY
        """
        from .boundary import BoundaryKind

        if boundary.kind is BoundaryKind.SEMI:
            return self.N * boundary.NY
        if boundary.kind is BoundaryKind.OBC:
            return self.N * len(boundary.active_cells())
        raise ValueError(f"未知边界类型: {boundary.kind!r}")

    def indexer(
        self, order: str = "cell", NX: int = 1, NY: int = 1,
        cells: tuple[tuple[int, int], ...] | None = None,
    ) -> "Indexer":
        return Indexer(self, order=order, NX=NX, NY=NY, cells=cells)


@dataclass
class Indexer:
    """格点 → 矩阵行索引 (smap) 及逆映射 (rmap), 一次构建互逆两表.

    order='cell': 胞优先序, idx = (cx·NY + cy)·N + r
                  —— 与 MATLAB build_H_np 'cell' 序 (for cx, for cy) 一致.
    order='site': 按实空间绝对坐标 (x 主, y 次) 字典序.
    """

    lattice: Lattice
    order: str = "cell"
    NX: int = 1
    NY: int = 1
    cells: tuple[tuple[int, int], ...] | None = None

    def __post_init__(self) -> None:
        if self.order not in {"cell", "site"}:
            raise ValueError(f"order 必须是 'cell' 或 'site', 得到 {self.order!r}")
        if self.NX < 1 or self.NY < 1:
            raise ValueError(f"NX/NY 必须为正整数, 得到 {self.NX}/{self.NY}")
        if self.cells is None:
            cells = tuple(
                (cx, cy) for cx in range(self.NX) for cy in range(self.NY)
            )
        else:
            cells = tuple((int(cx), int(cy)) for cx, cy in self.cells)
            if len(set(cells)) != len(cells):
                raise ValueError("cells 不可包含重复元胞")
            if any(not (0 <= cx < self.NX and 0 <= cy < self.NY)
                   for cx, cy in cells):
                raise ValueError("cells 中存在超出 NX×NY 范围的元胞")
        if not cells:
            raise ValueError("cells 至少需要一个元胞")
        self.cells = cells
        smap: dict[tuple[int, int, int], int] = {}
        rmap: list[tuple[int, int, int]] = []
        if self.order == "cell":
            for cx, cy in self.cells:
                for r in range(self.lattice.N):
                    smap[(cx, cy, r)] = len(rmap)
                    rmap.append((cx, cy, r))
        else:
            items = []
            for cx, cy in self.cells:
                for r in range(self.lattice.N):
                    x, y = self.lattice.position(cx, cy, r)
                    items.append((x, y, cx, cy, r))
            items.sort(key=lambda t: (t[0], t[1], t[2], t[3], t[4]))
            for _x, _y, cx, cy, r in items:
                smap[(cx, cy, r)] = len(rmap)
                rmap.append((cx, cy, r))
        self.smap = smap
        self.rmap = rmap

    def __call__(self, cx: int, cy: int, r: int) -> int:
        return self.smap[(cx, cy, r)]

    def idx(self, cx: int, cy: int, r: int) -> int:
        return self.smap[(cx, cy, r)]

    def coord(self, idx: int) -> tuple[int, int, int]:
        return self.rmap[idx]
