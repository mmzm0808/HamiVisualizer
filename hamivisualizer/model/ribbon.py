"""半无限 (x-Bloch) 建阵引擎 — 与具体模型无关.

MATLAB 版 build_H_np_ribbon / build_H_sc_ribbon 的核心算法被抽象为:
  - fold_x:      x-Bloch 折叠 (cs 分类)
  - build_basis: 逐 y 胞平移原始胞格点, 顶部截断入基
  - build_ribbon:数据驱动建 H0/H1 (步骤 2, 见 hamiltonian.py)

本文件是纯几何/建基部分, 无 GUI 依赖。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class RibbonSpec:
    """半无限建阵所需的全部几何信息, 由 Lattice + Boundary 适配而来.

    cell_sites: 胞内格点绝对坐标 (x, y), r 序
    Lx, Ly:     元胞平移向量 (x 折叠周期 = Lx)
    NY:         y 方向胞数
    edge_extra: 顶部额外截断行数 (半元胞预留, MVP 恒 0)
    order:      'cell' 胞优先 | 'site' 绝对坐标序
    """

    cell_sites: tuple[tuple[float, float], ...]
    Lx: float
    Ly: float
    NY: int
    a2: tuple[float, float] | None = None
    edge_extra: int = 0
    order: str = "cell"

    @property
    def N(self) -> int:
        return len(self.cell_sites)

    @property
    def Ymax(self) -> float:
        return self.NY * self.Ly + self.edge_extra


def fold_x(xt: float, Lx: float) -> tuple[int, float]:
    """x-Bloch 折叠: xt → (cs, xm), 使 xm ∈ [0, Lx), cs 为跨周期计数.

    用 divmod 的地板 (floor) 语义 —— 与 MATLAB 版 while 累加逐位一致, 负数安全:
      fold_x(3, 2)  = (1, 1)
      fold_x(-1, 2) = (-1, 1)
    """
    cs, xm = divmod(xt, Lx)
    return int(cs), float(xm)


def _q(v: float, ndigits: int = 9) -> float:
    """量化坐标键, 防浮点平移误差."""
    return round(float(v), ndigits)


def build_basis(spec: RibbonSpec):
    """构建半无限基: 逐 y 胞平移原始胞格点, y < Ymax 截断入基.

    返回 (basis, origin, keys):
      basis:  list[(x, y)]       实空间坐标 (格子)
      origin: list[(r, cy)]      basis[i] 的胞内序号 r 与 y 胞号 cy
      keys:   dict[(qx, qy), i]  量化坐标 → 基索引
    """
    pts = []
    for cy in range(spec.NY):
        for r in range(spec.N):
            x, y = spec.cell_sites[r]
            shift_x, shift_y = spec.a2 if spec.a2 is not None else (0.0, spec.Ly)
            xx = x + cy * shift_x
            yy = y + cy * shift_y
            if yy >= spec.Ymax - 1e-12:
                continue
            pts.append((r, cy, float(xx), float(yy)))
    if spec.order == "cell":
        # y 胞主序, 胞内 r 次 —— 与 MATLAB ribbon 'cell' 序一致
        pts.sort(key=lambda p: (p[1], p[0]))
    else:
        # 绝对坐标 x 主序
        pts.sort(key=lambda p: (p[2], p[3], p[1], p[0]))
    basis = [(x, yy) for _, _, x, yy in pts]
    origin = [(r, cy) for r, cy, _, _ in pts]
    keys = {}
    for i, (x, y) in enumerate(basis):
        keys[(_q(x), _q(y))] = i
    return basis, origin, keys


def _conj(v: Any, use_sympy: bool) -> Any:
    return v.conjugate() if use_sympy else np.conj(v)


def _mat(acc: dict, n: int, use_sympy: bool = False):
    """稀疏 dict 累积 → 稠密矩阵."""
    if use_sympy:
        import sympy as sp

        M = sp.zeros(n)
        for (i, j), v in acc.items():
            M[i, j] += v
        return M
    M = np.zeros((n, n), dtype=complex)
    for (i, j), v in acc.items():
        M[i, j] += v
    return M


def _amp_equal(a: Any, b: Any, use_sympy: bool) -> bool:
    if use_sympy:
        import sympy as sp

        return sp.simplify(a - b) == 0
    try:
        return bool(np.isclose(complex(a), complex(b), atol=1e-12, rtol=1e-12))
    except (TypeError, ValueError):
        return a == b


def _canonical_rows(rows: Iterable, use_sympy: bool):
    """Normalize a Hermitian bond template independently of matrix ordering.

    Positive cell offsets are preferred.  Reversing a template swaps its
    endpoints and conjugates its amplitude.  An explicitly supplied reverse
    copy with the same physical amplitude is deduplicated, while multiple
    contributions supplied in the same direction remain additive.
    """

    normalized = []
    seen: dict[tuple, list[tuple[Any, bool]]] = {}
    reverse_duplicates = 0
    for row in rows:
        if len(row) < 4:
            raise ValueError("跃迁行至少需要 from/to/offset/amplitude 四项")
        from_r, to_r, off, amp, *metadata = row
        dx, dy = int(off[0]), int(off[1])
        reverse = dx < 0 or (dx == 0 and dy < 0) or (
            dx == 0 and dy == 0 and from_r > to_r
        )
        if reverse:
            from_r, to_r = to_r, from_r
            dx, dy = -dx, -dy
            amp = _conj(amp, use_sympy)
            if metadata and metadata[0] is not None:
                # The first optional metadata field is the symbolic source
                # expression used by smart labels.  It follows the same
                # Hermitian orientation as the numerical amplitude.
                metadata[0] = metadata[0].conjugate()
        key = (from_r, to_r, dx, dy)
        prior = seen.setdefault(key, [])
        if any(old_reverse != reverse and _amp_equal(old_amp, amp, use_sympy)
               for old_amp, old_reverse in prior):
            reverse_duplicates += 1
            continue
        prior.append((amp, reverse))
        normalized.append((from_r, to_r, (dx, dy), amp, *metadata))
    return normalized, reverse_duplicates


def _add_provenance(acc: dict, key: tuple, expression) -> None:
    """Accumulate a sparse label source, failing closed on missing pieces."""
    if key in acc and acc[key] is None:
        return
    if expression is None:
        acc[key] = None
        return
    acc[key] = acc.get(key, 0) + expression


def _validate_onsite(amp: Any, use_sympy: bool) -> None:
    if use_sympy:
        if getattr(amp, "is_real", None) is not True:
            raise ValueError("Hermitian 模型的 on-site 项必须可证明为实数")
        return
    if abs(complex(amp).imag) > 1e-12:
        raise ValueError("Hermitian 模型的 on-site 项不能含虚部")


def build_ribbon(spec: RibbonSpec, rows: Iterable, *, use_sympy: bool = False):
    """数据驱动建半无限 H0 / H1 (H(kx)=H0 + H1·e^{ikx} + H1†·e^{-ikx}).

    rows: 迭代 (from_site, to_site, cell_offset, amplitude)。
          amplitude 已由调用方计算 (HoppingTerm.evaluate()), 可为复数或 sympy 表达式。

    每行表示一条 Hermitian 物理键。负偏移会交换端点并共轭幅度；等价的显式
    反向副本会去重。该规则与最终矩阵索引无关，因此切换 basis order 不改变物理谱。
    cs==0 写入 H0 并自动补共轭；cs==1 写入 H1；cs>=2 记入 extra。
    """
    basis, origin, keys = build_basis(spec)
    Nat = len(basis)
    acc0: dict = {}
    acc1: dict = {}
    extra: dict = {}
    stats: Counter = Counter()
    logical = {key: i for i, key in enumerate(origin)}
    canonical, reverse_duplicates = _canonical_rows(rows, use_sympy)
    if reverse_duplicates:
        stats["dedup_reverse"] += reverse_duplicates
    provenance: dict = {}
    for row in canonical:
        from_r, to_r, off, amp, *metadata = row
        source = metadata[0] if metadata else None
        cs, dy = off
        for cy in range(spec.NY):
            i = logical.get((from_r, cy))
            j = logical.get((to_r, cy + dy))
            if i is None or j is None:
                stats["y_cut"] += 1
                continue
            if cs == 0:
                if j == i:
                    _validate_onsite(amp, use_sympy)
                    acc0[(i, i)] = acc0.get((i, i), 0) + amp
                    _add_provenance(provenance, (i, i, 0), source)
                else:
                    acc0[(i, j)] = acc0.get((i, j), 0) + amp
                    acc0[(j, i)] = acc0.get((j, i), 0) + _conj(amp, use_sympy)
                    _add_provenance(provenance, (i, j, 0), source)
                    _add_provenance(
                        provenance, (j, i, 0),
                        None if source is None else source.conjugate(),
                    )
            elif cs == 1:
                acc1[(i, j)] = acc1.get((i, j), 0) + amp
                _add_provenance(provenance, (i, j, 1), source)
                _add_provenance(
                    provenance, (j, i, -1),
                    None if source is None else source.conjugate(),
                )
            else:
                stats["x_long_range"] += 1
                extra[(cs, i, j)] = extra.get((cs, i, j), 0) + amp
                _add_provenance(provenance, (i, j, cs), source)
                _add_provenance(
                    provenance, (j, i, -cs),
                    None if source is None else source.conjugate(),
                )
    H0 = _mat(acc0, Nat, use_sympy)
    H1 = _mat(acc1, Nat, use_sympy)
    return RibbonHamiltonian(
        H0, H1, extra, basis, origin, keys, dict(stats), provenance,
    )


class RibbonHamiltonian:
    """半无限哈密顿量: H(kx) = H0 + H1·e^{ikx} + H1†·e^{-ikx}.

    提供 H(kx)、H_sym(kx)、bands(kx_grid) 三种求值入口。
    """

    def __init__(
        self,
        H0,
        H1,
        extra: dict | None = None,
        basis=None,
        origin=None,
        keys=None,
        stats: dict | None = None,
        provenance: dict | None = None,
    ):
        self.H0 = H0
        self.H1 = H1
        self.extra = extra or {}
        self.basis = basis
        self.origin = origin
        self.keys = keys
        self.stats = stats or {}
        self.provenance = provenance or {}
        self.Nat = H0.shape[0]

    def H(self, kx: float):
        """数值 kx → H(kx) 矩阵."""
        ph = np.exp(1j * kx)
        H = self.H0 + ph * self.H1 + np.conj(ph) * self.H1.conj().T
        for (cs, i, j), v in self.extra.items():
            phn = np.exp(1j * cs * kx)
            H[i, j] += v * phn
            H[j, i] += np.conj(v) * np.conj(phn)
        return H

    def H_sym(self, kx_sym):
        """符号 kx → 符号 H(kx) 矩阵."""
        import sympy as sp

        # ``build_ribbon`` may retain immutable SymPy blocks when a model is
        # assembled symbolically.  Convert explicitly to a mutable matrix
        # before adding long-range entries in-place.
        H0 = sp.Matrix(self.H0)
        H1 = sp.Matrix(self.H1)
        ph = sp.exp(sp.I * kx_sym)
        H = sp.MutableDenseMatrix(H0)
        base = ph * H1 + sp.conjugate(H1.T) * sp.exp(-sp.I * kx_sym)
        for i in range(H.rows):
            for j in range(H.cols):
                H[i, j] = H[i, j] + base[i, j]
        for (cs, i, j), v in self.extra.items():
            phn = sp.exp(sp.I * cs * kx_sym)
            H[i, j] += v * phn
            H[j, i] += sp.conjugate(v) / phn
        return H

    def bands(self, kx_grid):
        """扫点批量对角化.

        返回 (kx, E): E 形状 (nk, Nat), 每个 kx 处升序。
        向量化构造 (nk,Nat,Nat) 张量, 一次性 eigvalsh。
        """
        kx = np.atleast_1d(kx_grid)
        ph = np.exp(1j * kx)
        # Chunking avoids allocating nk×Nat×Nat for large scans.
        out = np.empty((len(kx), self.Nat), dtype=float)
        chunk_size = max(1, min(32, 2_000_000 // max(1, self.Nat * self.Nat)))
        for start in range(0, len(kx), chunk_size):
            stop = min(len(kx), start + chunk_size)
            p = ph[start:stop]
            H = (
                self.H0[None]
                + p[:, None, None] * self.H1[None]
                + np.conj(p)[:, None, None] * self.H1.conj().T[None]
            )
            for (cs, i, j), v in self.extra.items():
                phn = np.exp(1j * cs * kx[start:stop])
                H[:, i, j] += v * phn
                H[:, j, i] += np.conj(v) * np.conj(phn)
            out[start:stop] = np.linalg.eigvalsh(H)
        return kx, out

    def hermitian_check(self, tol: float = 1e-8) -> float:
        """返回 max|H − H†|; 0 表示严格厄米."""
        worst = 0.0
        for kx in (0.0, 0.371, np.pi):
            H = self.H(kx)
            if H.size:
                worst = max(worst, float(np.max(np.abs(H - H.conj().T))))
        return worst
