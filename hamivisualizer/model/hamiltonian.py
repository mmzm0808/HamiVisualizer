"""HamiltonianBuilder — 统一建阵入口 (SEMI / OBC 分派).

把用户可见的 Lattice + HoppingTerm + Boundary 适配成半无限引擎
(ribbon.build_ribbon) 需要的输入, 并返回视图层消费的 HResult。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import sympy as sp

from .boundary import Boundary, BoundaryKind
from .hopping import HoppingTerm
from .lattice import Lattice
from .ribbon import (
    RibbonSpec,
    build_ribbon,
    _canonical_rows,
    _conj,
    _mat,
    _validate_onsite,
)


# The current backend deliberately uses dense NumPy/SymPy matrices.  Keep a
# generous but finite working-set guard so an accidental 100000-cell input
# produces a recoverable UI error instead of taking down the Python process.
# This is a computation safety budget, not a model/persistence limit; users
# can still edit and save larger studies for a future sparse backend.
DENSE_WORKING_SET_LIMIT_BYTES = 256 * 1024 * 1024


def estimate_dense_working_set_bytes(nat: int, boundary_kind: BoundaryKind) -> int:
    """Estimate the temporary dense-matrix working set for one calculation.

    The estimate covers the primary matrix blocks plus the usual numerical
    conversion/eigensolver buffers.  It intentionally errs on the safe side
    and is used before any ``nat × nat`` allocation occurs.
    """
    nat = int(nat)
    if nat < 0:
        raise ValueError("矩阵维度不能为负数")
    if boundary_kind is BoundaryKind.SEMI:
        arrays = 3  # H0, H1, evaluated H(kx)
    elif boundary_kind is BoundaryKind.OBC:
        arrays = 4  # H, conversion/eigensolver workspace and eigenvectors
    else:
        raise ValueError(f"未知边界类型: {boundary_kind!r}")
    # complex128 = 16 bytes; use integer arithmetic to avoid overflow/rounding
    return nat * nat * 16 * arrays


def _guard_dense_working_set(nat: int, boundary_kind: BoundaryKind) -> None:
    """Reject an obviously unsafe dense calculation before allocating it."""
    estimated = estimate_dense_working_set_bytes(nat, boundary_kind)
    if estimated <= DENSE_WORKING_SET_LIMIT_BYTES:
        return
    mib = estimated / (1024 * 1024)
    limit_mib = DENSE_WORKING_SET_LIMIT_BYTES // (1024 * 1024)
    mode = "半无限" if boundary_kind is BoundaryKind.SEMI else "双开"
    raise ValueError(
        f"{mode}矩阵规模为 {int(nat)}×{int(nat)}，稠密后端预计需要约 {mib:.0f} MiB，"
        f"超过安全预算 {limit_mib} MiB；请减小 NX/NY，或等待后续稀疏后端支持。"
        "模型仍可保存，输入范围没有被限制。"
    )


@dataclass
class HResult:
    """建阵结果 (视图层消费的唯一接口)."""

    blocks: dict = field(default_factory=dict)  # SEMI: {'H0','H1'}
    extra: dict = field(default_factory=dict)   # SEMI: {(distance_x,i,j): amplitude}
    H: Any = None  # OBC: 有限矩阵
    Ncells: int = 0
    Nsites: int = 0
    Nat: int = 0
    skipped: dict = field(default_factory=dict)
    smap: Any = None
    rmap: Any = None
    positions: Any = None
    origin: Any = None
    provenance: Any = None
    labels: Any = None

    def to_semi(self, kx):
        """Evaluate all x-cell harmonics of the semi-infinite Hamiltonian."""
        H0, H1 = self.blocks["H0"], self.blocks["H1"]
        # 数值模型也允许请求保留 kx 的符号结构，供默认“智能标签”显示。
        # np.exp(sympy.Symbol) 不可用，因此只要 kx 是符号就将数值块提升
        # 为 SymPy Matrix；实际谱计算传入 float，仍走高性能 NumPy 路径。
        if isinstance(H0, sp.MatrixBase) or isinstance(kx, sp.Basic):
            H0 = H0 if isinstance(H0, sp.MatrixBase) else sp.Matrix(H0)
            H1 = H1 if isinstance(H1, sp.MatrixBase) else sp.Matrix(H1)
            phase = sp.exp(sp.I * kx)
            # Keep the Bloch convention explicit.  ``conjugate(exp(I*kx))``
            # is only equivalent when ``kx`` is declared real; callers that
            # request a symbolic matrix often intentionally leave that
            # assumption open for later substitution.
            matrix = H0 + phase * H1 + sp.exp(-sp.I * kx) * H1.T.conjugate()
            for (distance, i, j), value in self.extra.items():
                harmonic = sp.exp(sp.I * distance * kx)
                matrix[i, j] += value * harmonic
                # Hermitian conjugation applies to the coefficient and to the
                # Bloch phase separately.  Conjugating the product would also
                # turn an unconstrained symbolic ``kx`` into ``conjugate(kx)``
                # and make this path disagree with RibbonHamiltonian.H_sym.
                matrix[j, i] += sp.conjugate(value) * sp.exp(-sp.I * distance * kx)
            return matrix
        ph = np.exp(1j * kx)
        matrix = H0 + ph * H1 + np.conj(ph) * H1.conj().T
        for (distance, i, j), value in self.extra.items():
            harmonic = np.exp(1j * distance * kx)
            matrix[i, j] += value * harmonic
            matrix[j, i] += np.conj(value * harmonic)
        return matrix


class HamiltonianBuilder:
    """把任意自定义晶格 (Lattice + [HoppingTerm]) 建成哈密顿量.

    用法:
        b = HamiltonianBuilder(lattice, hops, Boundary(BoundaryKind.SEMI, NY=3))
        res = b.build()          # 按边界模式分派
        res = b.build_semi()     # 或显式
    """

    def __init__(self, lattice: Lattice, hops: Iterable[HoppingTerm], boundary: Boundary, order: str = "cell"):
        self.lattice = lattice
        self.hops = list(hops)
        self.boundary = boundary
        self.order = order
        self._validate()

    def _validate(self) -> None:
        if self.order not in {"cell", "site"}:
            raise ValueError(f"order 必须是 'cell' 或 'site', 得到 {self.order!r}")
        if not isinstance(self.boundary, Boundary):
            raise ValueError("boundary 必须是 Boundary 实例")
        for row, hop in enumerate(self.hops, start=1):
            if not isinstance(hop, HoppingTerm):
                raise ValueError(f"第 {row} 条跃迁不是 HoppingTerm")
            if hop.from_site >= self.lattice.N or hop.to_site >= self.lattice.N:
                raise ValueError(
                    f"第 {row} 条跃迁格点索引越界: {hop.from_site}→{hop.to_site}, "
                    f"当前格点范围为 0..{self.lattice.N - 1}"
                )

    # ---- 适配 ----

    def _spec(self) -> RibbonSpec:
        return RibbonSpec(
            cell_sites=tuple((s.x, s.y) for s in self.lattice.sites),
            Lx=self.lattice.Lx,
            Ly=self.lattice.Ly,
            NY=self.boundary.NY,
            a2=self.lattice.a2,
            order=self.order,
        )

    def _rows(self) -> Iterable:
        """把 HoppingTerm 解析成 (from, to, off, amp) 键表.

        相位解析在此完成: amp = amplitude·exp(i·phase_sign·phase)。
        build_ribbon 只见复数/符号表达式。
        """
        for h in self.hops:
            yield (h.from_site, h.to_site, h.cell_offset, h.evaluate())

    def _uses_symbols(self) -> bool:
        """任一跃迁含符号参数 (amplitude/phase 为 sympy.Basic) 即走符号模式."""
        for h in self.hops:
            if isinstance(h.amplitude, sp.Basic) or isinstance(h.phase, sp.Basic):
                return True
        return False

    # ---- 建阵 ----

    def build_semi(self):
        """半无限: H(kx) = H0 + H1·e^{ikx} + H1†·e^{-ikx}.

        数值/符号自动分派 (见 _uses_symbols)。
        """
        _guard_dense_working_set(
            self.lattice.N * self.boundary.NY, BoundaryKind.SEMI
        )
        use_sympy = self._uses_symbols()
        rb = build_ribbon(self._spec(), self._rows(), use_sympy=use_sympy)
        if rb.stats.get("miss", 0):
            raise ValueError(
                f"{rb.stats['miss']} 条键的目标格点未落在基上 —— 请检查格点坐标与 cell_offset 定义"
            )
        # 半无限显示映射: 绝对坐标 → 序号
        smap = {}
        positions = []
        for i, (x, y) in enumerate(rb.basis):
            smap[(round(float(x), 9), round(float(y), 9))] = i
            positions.append((float(x), float(y)))
        return HResult(
            blocks={"H0": rb.H0, "H1": rb.H1},
            extra=dict(rb.extra),
            Ncells=self.boundary.NY,
            Nsites=rb.Nat,
            Nat=rb.Nat,
            skipped=dict(rb.stats),
            smap=smap,
            positions=positions,
            origin=rb.origin,
            labels=tuple(f"{cy}:{r}" for r, cy in rb.origin),
        )

    def build_obc(self):
        """双开边界: 有限厄米矩阵 (NX×NY 胞, 两方向越界即跳过).

        索引序由 self.order 决定 (Indexer)。每个物理键在每胞重复，目标胞越界
        跳过；负偏移/反向定义先按物理键规范化，绝不使用矩阵索引大小去重。
        """
        NX, NY = self.boundary.NX, self.boundary.NY
        lat = self.lattice
        # Check the full rectangular upper bound before constructing the
        # active-cell tuple.  Non-rectangular masks can only reduce this
        # count, but enumerating a huge NX×NY grid just to discover that the
        # dense backend is unsafe would defeat the purpose of the guard.
        _guard_dense_working_set(lat.N * NX * NY, BoundaryKind.OBC)
        active_cells = tuple(self.boundary.active_cells())
        active_set = set(active_cells)
        ix = lat.indexer(order=self.order, NX=NX, NY=NY, cells=active_cells)
        Nat = lat.N * len(active_cells)
        # Keep a second check for custom masks and future Boundary
        # implementations whose active-cell set may be larger than expected.
        _guard_dense_working_set(Nat, BoundaryKind.OBC)
        use_sympy = self._uses_symbols()
        acc: dict = {}
        skipped: dict = {}
        evaluated = [
            (h.from_site, h.to_site, h.cell_offset, h.evaluate()) for h in self.hops
        ]
        canonical, reverse_duplicates = _canonical_rows(evaluated, use_sympy)
        if reverse_duplicates:
            skipped["dedup_reverse"] = reverse_duplicates
        for from_site, to_site, offset, amp in canonical:
            for cx, cy in active_cells:
                i = ix(cx, cy, from_site)
                tx, ty = cx + offset[0], cy + offset[1]
                if (tx, ty) not in active_set:
                    skipped["oob"] = skipped.get("oob", 0) + 1
                    continue
                j = ix(tx, ty, to_site)
                if j == i:
                    _validate_onsite(amp, use_sympy)
                    acc[(i, i)] = acc.get((i, i), 0) + amp  # on-site / 自键
                else:
                    acc[(i, j)] = acc.get((i, j), 0) + amp
                    acc[(j, i)] = acc.get((j, i), 0) + _conj(amp, use_sympy)
        H = _mat(acc, Nat, use_sympy)
        positions = [lat.position(*c) for c in ix.rmap]
        return HResult(
            H=H,
            blocks={},
            Ncells=len(active_cells),
            Nsites=Nat,
            Nat=Nat,
            skipped=skipped,
            smap=ix.smap,
            rmap=ix.rmap,
            positions=positions,
            labels=tuple(f"{cx},{cy}:{r}" for cx, cy, r in ix.rmap),
        )

    def build(self):
        if self.boundary.kind is BoundaryKind.SEMI:
            return self.build_semi()
        if self.boundary.kind is BoundaryKind.OBC:
            return self.build_obc()
        raise ValueError(f"未知边界类型: {self.boundary.kind!r}")


def build_semi(lattice, hops, boundary, order="cell") -> HResult:
    return HamiltonianBuilder(lattice, hops, boundary, order).build_semi()


def build_obc(lattice, hops, boundary, order="cell") -> HResult:
    return HamiltonianBuilder(lattice, hops, boundary, order).build_obc()


def _checked_numeric_hermitian(H) -> np.ndarray:
    matrix = np.asarray(H, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError(f"矩阵必须是非空方阵, 得到 shape={matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("矩阵包含 NaN 或 Inf")
    error = float(np.max(np.abs(matrix - matrix.conj().T)))
    if error > 1e-9:
        raise ValueError(f"矩阵不是 Hermitian: max|H-H†|={error:.3g}")
    return matrix


def eig(H):
    """对角化有限厄米矩阵 → (E, U), E 升序."""
    return np.linalg.eigh(_checked_numeric_hermitian(H))


def _nearest_site_spacing(points: np.ndarray) -> float:
    """Return a robust physical nearest-neighbour spacing without an N² copy."""
    count = len(points)
    if count < 2:
        return 0.0
    nearest = np.full(count, np.inf, dtype=float)
    # The dense Hamiltonian guard currently limits practical wavefunction
    # samples to about 2048 sites.  Chunking still keeps this helper modest in
    # memory and avoids allocating the much larger ``points[:, None, :]``
    # tensor next to the eigensolver buffers.
    chunk_size = min(256, count)
    for start in range(0, count, chunk_size):
        stop = min(count, start + chunk_size)
        delta = points[start:stop, None, :] - points[None, :, :]
        distance2 = np.einsum("ijk,ijk->ij", delta, delta)
        rows = np.arange(stop - start)
        distance2[rows, np.arange(start, stop)] = np.inf
        # Coincident user points must not collapse the physical shell width.
        distance2[distance2 <= 1e-20] = np.inf
        nearest[start:stop] = np.sqrt(np.min(distance2, axis=1))
    finite = nearest[np.isfinite(nearest)]
    return float(np.median(finite)) if finite.size else 0.0


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Return counter-clockwise hull vertices using the monotonic-chain rule."""
    ordered = sorted({(float(x), float(y)) for x, y in points})
    if len(ordered) <= 2:
        return np.asarray(ordered, dtype=float)

    scale = max(
        max(x for x, _y in ordered) - min(x for x, _y in ordered),
        max(y for _x, y in ordered) - min(y for _x, y in ordered),
        1.0,
    )
    cross_tol = 1e-12 * scale * scale

    def cross(origin, a, b):
        return ((a[0] - origin[0]) * (b[1] - origin[1])
                - (a[1] - origin[1]) * (b[0] - origin[0]))

    lower: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= cross_tol:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= cross_tol:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def edge_mask_for_positions(positions) -> np.ndarray:
    """Return a rotation-invariant one-layer boundary mask for finite sites.

    The former x/y bounding-box shell missed slanted triangle edges and most
    of a disk/hexagon perimeter; its ``15% of sample span`` thickness also
    grew with system size and eventually labelled bulk sites as boundary.
    The mask now follows the physical point-cloud convex hull and uses the
    median nearest-site distance as a fixed one-lattice-layer scale.
    """
    points = np.asarray(positions, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not len(points):
        raise ValueError("positions 必须是非空的 (x, y) 坐标序列")
    if not np.all(np.isfinite(points)):
        raise ValueError("positions 包含 NaN 或 Inf")
    if len(points) == 1:
        return np.ones(1, dtype=bool)

    spacing = _nearest_site_spacing(points)
    if spacing <= 1e-12:
        # Degenerate coincident coordinates have no meaningful interior.
        return np.ones(len(points), dtype=bool)
    shell = max(1e-9, 0.80 * spacing)

    centered = points - np.mean(points, axis=0, keepdims=True)
    _u, singular, axes = np.linalg.svd(centered, full_matrices=False)
    is_line = (
        singular.size < 2
        or singular[1] <= max(1e-10, 1e-9 * singular[0])
    )
    if is_line:
        direction = axes[0]
        coordinate = centered @ direction
        return ((coordinate <= coordinate.min() + shell)
                | (coordinate >= coordinate.max() - shell))

    hull = _convex_hull(points)
    if len(hull) < 3:
        # Numerical near-collinearity fallback.
        direction = axes[0]
        coordinate = centered @ direction
        return ((coordinate <= coordinate.min() + shell)
                | (coordinate >= coordinate.max() - shell))

    min_distance2 = np.full(len(points), np.inf, dtype=float)
    for index, start in enumerate(hull):
        end = hull[(index + 1) % len(hull)]
        segment = end - start
        length2 = float(segment @ segment)
        if length2 <= 1e-20:
            continue
        fraction = np.clip(((points - start) @ segment) / length2, 0.0, 1.0)
        closest = start + fraction[:, None] * segment
        delta = points - closest
        min_distance2 = np.minimum(
            min_distance2, np.einsum("ij,ij->i", delta, delta),
        )
    return min_distance2 <= shell * shell + 1e-12


def _localize_exactly_degenerate_edge_states(E: np.ndarray, U: np.ndarray,
                                              edge_mask: np.ndarray) -> np.ndarray:
    """Choose a deterministic edge-localized basis inside exact degeneracies.

    An eigensolver is free to return any orthonormal basis in an exactly
    degenerate energy subspace.  For a finite lattice that can turn two edge
    modes into visually arbitrary superpositions.  Diagonalising the edge
    projector *inside that exact subspace* preserves the Hamiltonian
    eigenvalue while yielding the least- and most-edge-localized
    representatives.  Near-degenerate states are deliberately not mixed:
    they remain genuine individual eigenstates.
    """
    if U.shape[1] < 2:
        return U
    scale = max(1.0, float(np.max(np.abs(E))))
    tolerance = max(1e-12, 128 * np.finfo(float).eps * scale)
    rotated = np.array(U, copy=True)
    start = 0
    while start < len(E):
        end = start + 1
        while end < len(E) and abs(float(E[end] - E[start])) <= tolerance:
            end += 1
        if end - start > 1:
            basis = rotated[:, start:end]
            projector = basis[edge_mask].conj().T @ basis[edge_mask]
            _weights, rotation = np.linalg.eigh(projector)
            rotated[:, start:end] = basis @ rotation
        start = end
    return rotated


def wavefunctions(H, positions=None):
    """OBC 每个本征态的实空间 |ψ|² 分布.

    ``positions`` is optional for API compatibility.  When supplied, exact
    energy degeneracies are represented by an edge-projector-localized basis,
    so the displayed finite-system modes are physically inspectable rather
    than arbitrary mixtures.  返回 ``(E, wf)``；``wf[:, k]`` 是第 ``k`` 个
    本征态的真实概率密度 |ψ|²，满足每列之和为 1，行序与矩阵索引一致。

    颜色映射若需要把峰值缩放到 1，应只在视图层执行。后端若把每列除以
    最大值，虽然图案形状不变，却会让悬停显示的 ``|ψ_i|²`` 不再是概率，
    也会使不同态之间的单点权重失去可比性。
    """
    E, U = np.linalg.eigh(_checked_numeric_hermitian(H))
    if positions is not None:
        edge_mask = edge_mask_for_positions(positions)
        if edge_mask.shape != (U.shape[0],):
            raise ValueError("positions 数量必须与哈密顿量维度一致")
        U = _localize_exactly_degenerate_edge_states(E, U, edge_mask)
    # ``np.linalg.eigh`` 返回正交归一本征向量，因此这里的平方模已经满足
    # sum_i |psi_i|^2 = 1。保留真实概率；热图在绘制时单独按峰值着色。
    return E, np.abs(U) ** 2
