"""内置预设模型: NP / SC (可复现 MATLAB 版数值).

从「代码」到「数据」: NP/SC 不再有独立建阵函数, 而是返回
(Lattice, list[HoppingTerm]) 数据, 由通用 HamiltonianBuilder 消费。
这同时证明了数据驱动建模的泛化能力。
"""

from __future__ import annotations

from .hopping import HoppingTerm
from .lattice import Lattice, Site


def NP(phi, t=1.0, omg=1.0):
    """NP 模型: 4-site 全填充元胞 (README §2.1 的 NP).

    元胞 2×2: r0=(0,0,A) r1=(0,1,B) r2=(1,1,A) r3=(1,0,B)。
    NN 复跃迁 -t·e^{±iφ} (方向/子格依赖) + NNN 对角实跃迁 -t + on-site ω。

    相位方向表 (MATLAB): A 右−φ 左−φ 上+φ 下+φ; B 右+φ 左+φ 上−φ 下−φ。
    """
    lattice = Lattice(
        sites=[
            Site(0, 0.0, 0.0, "A"),
            Site(1, 0.0, 1.0, "B"),
            Site(2, 1.0, 1.0, "A"),
            Site(3, 1.0, 0.0, "B"),
        ],
        Lx=2.0,
        Ly=2.0,
    )
    hops: list[HoppingTerm] = []
    # NN: 幅度 -t, 相位符号按 A/B 方向表编码。
    #   胞内 4 边统一 from<to 定义 (反向由建阵共轭补全, 与 MATLAB 双端相位一致):
    #     r0(A)上 +φ / r1(B)右 +φ / r2(A)下 +φ / r0(A)右 −φ
    #   跨胞 +x 键 (cs=1 → H1): r3(B)右 +φ / r2(A)右 −φ
    #   跨胞 +y 键 (cs=0 → H0, 双向填): r1(B)上 −φ / r2(A)上 +φ
    for fr, to, off, sign in [
        (0, 1, (0, 0), +1),   # r0(A) 上 +φ
        (1, 2, (0, 0), +1),   # r1(B) 右 +φ
        (2, 3, (0, 0), +1),   # r2(A) 下 +φ
        (0, 3, (0, 0), -1),   # r0(A) 右 −φ (下横边)
        (3, 0, (1, 0), +1),   # r3(B) 右跨胞 +φ → H1
        (2, 1, (1, 0), -1),   # r2(A) 右跨胞 −φ → H1
        (1, 0, (0, 1), -1),   # r1(B) 上跨胞 −φ → H0
        (2, 3, (0, 1), +1),   # r2(A) 上跨胞 +φ → H0
    ]:
        hops.append(HoppingTerm("t", fr, to, off, -t, "phase", phi, sign))
    # NNN: 对角实跃迁 -t。
    #   同胞 2 条 (每胞, H0) + 对角胞 2 条 (位移 (±1,±1) 且 cx,cy 都跨胞, cs=1 → H1):
    #     r2→r0 右胞 (位移 +1,+1) / r3→r1 右胞 (位移 +1,−1)
    for fr, to, off in [
        (0, 2, (0, 0)),    # r0→r2 同胞对角 (位移 +1,+1)
        (1, 3, (0, 0)),    # r1→r3 同胞对角 (位移 +1,−1)
        (2, 0, (1, 1)),    # r2→r0 右胞对角 (位移 +1,+1, cs=1 → H1)
        (3, 1, (1, -1)),   # r3→r1 右胞对角 (位移 +1,−1, cs=1 → H1)
    ]:
        hops.append(HoppingTerm("t", fr, to, off, -t))
    # on-site ω
    for r in range(4):
        hops.append(HoppingTerm("omg", r, r, (0, 0), omg))
    return lattice, hops


def SC(phi, t=1.0, omg=1.0):
    """SC 模型: checkerboard 2-site 元胞 (README §2.1 的 SC).

    元胞 2×2: r0=A=(1,0), r1=B=(0,1) (空位即无 Site)。
    NN 对角复跃迁 -t·e^{±iφ} + NNN 轴向步2 实跃迁 -t + on-site ω。
    """
    lattice = Lattice(
        sites=[
            Site(0, 1.0, 0.0, "A"),
            Site(1, 0.0, 1.0, "B"),
        ],
        Lx=2.0,
        Ly=2.0,
    )
    hops: list[HoppingTerm] = []
    # NN 对角: MATLAB hop=-t·exp(-1i·pv), pv=iif(isCW,-phi,phi)
    #   r0→r1 off(0,0)  位移(−1,+1): isCW=true  → −t·e^{+iφ}  sign=+1 (胞内, H0)
    #   r0→r1 off(1,0)  位移(+1,+1): isCW=false → −t·e^{−iφ}  sign=−1 (右胞, H1)
    #   r0→r1 off(1,−1) 位移(+1,−1): isCW=true  → −t·e^{+iφ}  sign=+1 (右下胞, H1)
    #   r1→r0 off(0,1)  位移(+1,+1): isB→isCW=true → −t·e^{+iφ} sign=+1 (上胞, H0)
    hops.append(HoppingTerm("t", 0, 1, (0, 0), -t, "phase", phi, +1))
    hops.append(HoppingTerm("t", 0, 1, (1, 0), -t, "phase", phi, -1))
    hops.append(HoppingTerm("t", 0, 1, (1, -1), -t, "phase", phi, +1))
    hops.append(HoppingTerm("t", 1, 0, (0, 1), -t, "phase", phi, +1))
    # NNN 轴向步2 实跃迁 -t
    hops.append(HoppingTerm("t", 0, 0, (0, 1), -t))  # A 竖直步2 (H0)
    hops.append(HoppingTerm("t", 0, 0, (0, -1), -t))
    hops.append(HoppingTerm("t", 1, 1, (1, 0), -t))  # B 水平步2 (cs=1→H1 自跳)
    # on-site ω
    for r in range(2):
        hops.append(HoppingTerm("omg", r, r, (0, 0), omg))
    return lattice, hops
