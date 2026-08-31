"""跃迁项 (定向键) 定义."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

KNOWN_PHASE_MODES = frozenset({"none", "phase", "directional"})
SUPPORTED_PHASE_MODES = frozenset({"none", "phase"})


@dataclass(frozen=True)
class HoppingTerm:
    """一条定向跃迁键: 胞内格点 from_site → (胞内 to_site, 元胞偏移 cell_offset).

    反向键 (Hermitian 共轭) 由建阵循环自动补全, 用户只需定义一个方向。

    name:        参数名, 显示用 ('t', 't1', 'tc', 'omg', ...)
    from_site / to_site: 胞内格点索引 r
    cell_offset: to 所在元胞相对 from 所在元胞的偏移 (dx, dy); (0,0)=胞内。
                 实空间位移 = pos(to) + off·(Lx,Ly) − pos(from)。
    amplitude:   实数幅度 (float/complex 或 sympy.Symbol)
    phase_mode:  'none' 实跃迁 | 'phase' 固定相位 | 'directional' 方向依赖 (预留)
    phase:       相位值 (弧度数值或 sympy.Symbol)
    phase_sign:  +1/−1, 该键 from 端的相位符号: e^{i·phase_sign·phase}
    applies_to:  预留 (README §4.2), MVP 不用
    """

    name: str
    from_site: int
    to_site: int
    cell_offset: tuple[int, int] = (0, 0)
    amplitude: Any = 1.0
    phase_mode: str = "none"
    phase: Any = 0.0
    phase_sign: int = 1
    applies_to: Any = None

    def __post_init__(self) -> None:
        for field_name, value in (("from_site", self.from_site), ("to_site", self.to_site)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} 必须是非负整数, 得到 {value!r}")
        if (
            not isinstance(self.cell_offset, tuple)
            or len(self.cell_offset) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in self.cell_offset)
        ):
            raise ValueError(f"cell_offset 必须是两个整数, 得到 {self.cell_offset!r}")
        if self.phase_mode not in KNOWN_PHASE_MODES:
            raise ValueError(f"未知 phase_mode: {self.phase_mode!r}")
        if self.phase_sign not in {-1, 1}:
            raise ValueError(f"phase_sign 必须是 +1 或 -1, 得到 {self.phase_sign!r}")
        for field_name, value in (("amplitude", self.amplitude), ("phase", self.phase)):
            if isinstance(value, (int, float, complex)):
                z = complex(value)
                if not (math.isfinite(z.real) and math.isfinite(z.imag)):
                    raise ValueError(f"{field_name} 必须是有限数值")

    def evaluate(self) -> Any:
        """返回标量复数幅度 amplitude·exp(i·phase_sign·phase).

        数值走 cmath; 符号参数 (sympy.Symbol) 自动识别走 sympy。
        phase_mode='directional' (方向依赖相位) 尚未实现, 显式报错。
        """
        if self.phase_mode == "directional":
            raise ValueError(
                "phase_mode='directional' 尚未实现, 请改用 'none' (实跃迁) 或 'phase' (固定相位)"
            )
        if self.phase_mode == "none" or self.phase == 0:
            return self.amplitude
        import sympy as sp

        if isinstance(self.phase, sp.Basic):
            return self.amplitude * sp.exp(sp.I * self.phase_sign * self.phase)
        import cmath

        return self.amplitude * cmath.exp(1j * self.phase_sign * self.phase)

    def displacement(self, lattice) -> tuple[float, float]:
        """实空间位移向量, 由格点位置 + 元胞偏移唯一确定."""
        fx, fy = lattice.position(0, 0, self.from_site)
        tx, ty = lattice.position(
            self.cell_offset[0], self.cell_offset[1], self.to_site
        )
        return (tx - fx, ty - fy)

    def canonical_pair(self) -> tuple[int, int]:
        """规范化无向胞内键 (from,to 排序), 用于去重/冲突检测.

        只处理 cell_offset=(0,0) 的胞内键; 跨胞键由建阵循环的三去重规则保证不双计数。
        """
        a, b = self.from_site, self.to_site
        return (a, b) if a <= b else (b, a)
