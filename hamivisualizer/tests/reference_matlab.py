"""MATLAB build_H_np_ribbon / build_H_sc_ribbon 的忠实直译 (half=false, order='cell').

用作数值对拍的参考实现 —— 保持 MATLAB 的 j<=i 去重、双端独立相位、cs 折叠原样,
与泛化实现 (build_ribbon) 逐元对比以锁定数值正确性。

MATLAB 为 1-based, 本参考实现为 0-based, 其余逻辑逐句对应。
"""

from __future__ import annotations

import math

import numpy as np

from hamivisualizer.model.ribbon import fold_x


def ref_build_np_ribbon(NY, t=1.0, phi=np.pi / 4, omg=1.0):
    """直译 build_H_np_ribbon (half=false, order='cell').

    返回 (H0, H1), 满足 H(kx)=H0 + H1·e^{ikx} + H1†·e^{-ikx}。
    """
    Lx, Ly = 2, 2 * NY
    Nat = 4 * NY
    basis: list = []
    keys: dict = {}
    for cy in range(NY):
        for px, py in [(0, 2 * cy), (0, 2 * cy + 1), (1, 2 * cy + 1), (1, 2 * cy)]:
            keys[(px, py)] = len(basis)
            basis.append((px, py))
    H0 = np.zeros((Nat, Nat), dtype=complex)
    H1 = np.zeros((Nat, Nat), dtype=complex)
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for i, (x0, y0) in enumerate(basis):
        isA0 = (x0 + y0) % 2 == 0
        # === NN: 4 方向, 复跃迁 ===
        for d, (dx, dy) in enumerate(dirs, start=1):
            xt, yt = x0 + dx, y0 + dy
            if yt < 0 or yt >= Ly:
                continue
            cs, xm = fold_x(xt, Lx)
            if (xm, yt) not in keys:
                continue
            j = keys[(xm, yt)]
            if cs == 0 and j <= i:
                continue
            # 相位: A 右-φ左-φ上+φ下+φ; B 右+φ左+φ上-φ下-φ
            if isA0:
                bp = -phi if d in (1, 2) else phi
            else:
                bp = phi if d in (1, 2) else -phi
            hop = -t * np.exp(1j * bp)
            if cs == 0:
                H0[i, j] += hop
                H0[j, i] += np.conj(hop)
            elif cs == 1:
                H1[i, j] += hop
        # === NNN: 对角线 |dx|=|dy|=1, 实跃迁 -t ===
        for ndx, ndy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            xt, yt = x0 + ndx, y0 + ndy
            if yt < 0 or yt >= Ly:
                continue
            cx0, cy0 = math.floor(x0 / 2), math.floor(y0 / 2)
            cxt, cyt = math.floor(xt / 2), math.floor(yt / 2)
            same = (cx0 == cxt) and (cy0 == cyt)
            diag = (cx0 != cxt) and (cy0 != cyt)
            if not (same or diag):
                continue
            cs, xm = fold_x(xt, Lx)
            if cs == -1:
                continue
            if (xm, yt) not in keys:
                continue
            j = keys[(xm, yt)]
            hop = -t
            if cs == 0:
                if j <= i:
                    continue
                H0[i, j] += hop
                H0[j, i] += hop
            else:  # cs == 1
                H1[i, j] += hop
    for i in range(Nat):
        H0[i, i] += omg
    return H0, H1


def ref_build_sc_ribbon(NY, t=1.0, phi=np.pi / 4, omg=1.0):
    """直译 build_H_sc_ribbon (half=false, order='cell').

    MATLAB cell 序: for y, for x (y 主), 每胞 A(蓝) 先于 B(橙)。
    """
    Lx, Ly = 2, 2 * NY
    basis: list = []
    keys: dict = {}
    for y in range(Ly):
        for x in range(Lx):
            isA = (x % 2 == 1) and (y % 2 == 0)
            isB = (x % 2 == 0) and (y % 2 == 1)
            if isA or isB:
                keys[(x, y)] = len(basis)
                basis.append((x, y, isA))
    Nat = len(basis)
    H0 = np.zeros((Nat, Nat), dtype=complex)
    H1 = np.zeros((Nat, Nat), dtype=complex)
    for i, (x0, y0, isA0) in enumerate(basis):
        # === NN: 对角, 复跃迁 ===
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            xt, yt = x0 + dx, y0 + dy
            if yt < 0 or yt >= Ly:
                continue
            cs, xm = fold_x(xt, Lx)
            if cs == -1:
                continue
            if (xm, yt) not in keys:
                continue
            j = keys[(xm, yt)]
            if cs == 0 and j <= i:
                continue
            isCW = (dx * dy == -1) if isA0 else (dx * dy == 1)
            pv = -phi if isCW else phi
            hop = -t * np.exp(-1j * pv)
            if cs == 0:
                H0[i, j] += hop
                H0[j, i] += np.conj(hop)
            else:
                H1[i, j] += hop
        # === NNN: 轴向步 2, 实跃迁 -t ===
        nd = [(0, 2), (0, -2)] if isA0 else [(2, 0), (-2, 0)]
        for dx, dy in nd:
            xt, yt = x0 + dx, y0 + dy
            if yt < 0 or yt >= Ly:
                continue
            cs, xm = fold_x(xt, Lx)
            if cs == -1:
                continue
            if (xm, yt) not in keys:
                continue
            j = keys[(xm, yt)]
            hop = -t
            if cs == 0:
                if j <= i:
                    continue
                H0[i, j] += hop
                H0[j, i] += hop
            else:
                H1[i, j] += hop
    for i in range(Nat):
        H0[i, i] += omg
    return H0, H1


def ref_build_np_obc(NX, NY, t=1.0, phi=np.pi / 4, omg=1.0):
    """直译 build_H_np (双开, half=false, order='cell').

    全填充 2NX×2NY 网格, cell 序 smap (cx 主, 胞内顺时针 r0..r3)。
    """
    Lx, Ly = 2 * NX, 2 * NY
    Nat = Lx * Ly
    smap = np.zeros((Ly, Lx), dtype=int)
    ids = 0
    for cx in range(NX):
        for cy in range(NY):
            for xx, yy in [
                (2 * cx, 2 * cy), (2 * cx, 2 * cy + 1),
                (2 * cx + 1, 2 * cy + 1), (2 * cx + 1, 2 * cy),
            ]:
                if xx < Lx and yy < Ly:
                    smap[yy, xx] = ids
                    ids += 1
    H = np.zeros((Nat, Nat), dtype=complex)
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for yi in range(Ly):
        for xi in range(Lx):
            i = int(smap[yi, xi])
            isA = (xi + yi) % 2 == 0
            # NN
            for d, (dx, dy) in enumerate(dirs, start=1):
                nx, ny = xi + dx, yi + dy
                if nx < 0 or nx >= Lx or ny < 0 or ny >= Ly:
                    continue
                j = int(smap[ny, nx])
                if j <= i:
                    continue
                bp = (-phi if d in (1, 2) else phi) if isA else (phi if d in (1, 2) else -phi)
                H[i, j] = -t * np.exp(1j * bp)
                H[j, i] = np.conj(H[i, j])
            # NNN (同胞或对角胞)
            for ndx, ndy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                nx, ny = xi + ndx, yi + ndy
                if nx < 0 or nx >= Lx or ny < 0 or ny >= Ly:
                    continue
                j = int(smap[ny, nx])
                if j <= i:
                    continue
                same = math.floor(xi / 2) == math.floor(nx / 2) and math.floor(yi / 2) == math.floor(ny / 2)
                diag = math.floor(xi / 2) != math.floor(nx / 2) and math.floor(yi / 2) != math.floor(ny / 2)
                if same or diag:
                    H[i, j] = -t
                    H[j, i] = -t
    for i in range(Nat):
        H[i, i] += omg
    return H, smap


def ref_build_sc_obc(NX, NY, t=1.0, phi=np.pi / 4, omg=1.0):
    """直译 build_H_sc (双开, half=false, order='cell').

    checkerboard: 只有 A(奇x,偶y) / B(偶x,奇y) 格点, 空位即无 Site。
    """
    Lx, Ly = 2 * NX, 2 * NY
    smap = np.full((Ly, Lx), -1, dtype=int)  # -1 = 空位 (MATLAB 0, 1-based)
    sites = []
    for cx in range(NX):
        for cy in range(NY):
            xA, yA = 2 * cx + 1, 2 * cy
            if xA < Lx and yA < Ly and xA % 2 == 1 and yA % 2 == 0:
                smap[yA, xA] = len(sites)
                sites.append((xA, yA, True))
            xB, yB = 2 * cx, 2 * cy + 1
            if xB < Lx and yB < Ly and xB % 2 == 0 and yB % 2 == 1:
                smap[yB, xB] = len(sites)
                sites.append((xB, yB, False))
    Nat = len(sites)
    H = np.zeros((Nat, Nat), dtype=complex)
    for s, (x1, y1, isA1) in enumerate(sites):
        # NN 对角
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            x2, y2 = x1 + dx, y1 + dy
            if x2 < 0 or x2 >= Lx or y2 < 0 or y2 >= Ly:
                continue
            s2 = int(smap[y2, x2])
            if s2 < 0 or s2 <= s:
                continue
            isCW = (dx * dy == -1) if isA1 else (dx * dy == 1)
            pv = -phi if isCW else phi
            H[s, s2] = -t * np.exp(-1j * pv)
            H[s2, s] = np.conj(H[s, s2])
        # NNN 轴向步 2
        nd = [(0, 2), (0, -2)] if isA1 else [(2, 0), (-2, 0)]
        for dx, dy in nd:
            x2, y2 = x1 + dx, y1 + dy
            if x2 < 0 or x2 >= Lx or y2 < 0 or y2 >= Ly:
                continue
            s2 = int(smap[y2, x2])
            if s2 < 0 or s2 <= s:
                continue
            H[s, s2] = -t
            H[s2, s] = -t
    for i in range(Nat):
        H[i, i] += omg
    return H, smap
