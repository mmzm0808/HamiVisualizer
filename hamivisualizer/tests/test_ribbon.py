"""fold_x / build_basis 单元测试 (对拍 MATLAB count_sites 格点计数)."""

import numpy as np

from hamivisualizer.model.presets import NP, SC
from hamivisualizer.model.ribbon import RibbonSpec, build_basis, fold_x


def test_fold_x_floor_semantics():
    assert fold_x(0, 2) == (0, 0)
    assert fold_x(1, 2) == (0, 1)
    assert fold_x(2, 2) == (1, 0)
    assert fold_x(3, 2) == (1, 1)
    assert fold_x(-1, 2) == (-1, 1)   # MATLAB while 折叠同语义, 负数安全
    assert fold_x(-3, 2) == (-2, 1)
    assert fold_x(2.5, 2) == (1, 0.5)


def test_fold_x_closure():
    """折叠后必落回 [0, Lx)."""
    for xt in [-5, -1, 0, 1, 3.7, 7, 99]:
        cs, xm = fold_x(xt, 2)
        assert 0 <= xm < 2
        # 反向: 折叠是无损的 (cs·Lx + xm == xt)
        assert abs(cs * 2 + xm - xt) < 1e-12


def _spec_from(lattice, NY, order="cell", edge_extra=0):
    return RibbonSpec(
        cell_sites=tuple((s.x, s.y) for s in lattice.sites),
        Lx=lattice.Lx,
        Ly=lattice.Ly,
        NY=NY,
        order=order,
        edge_extra=edge_extra,
    )


def test_build_basis_np_count():
    """NP 半无限基个数 = 4·NY (MATLAB count_sites 对拍)."""
    lat, _ = NP(1.0)
    for ny in (1, 2, 3):
        basis, origin, keys = build_basis(_spec_from(lat, ny))
        assert len(basis) == 4 * ny
        assert len(origin) == len(basis)
        assert len(keys) == len(basis)


def test_build_basis_sc_count():
    """SC 半无限基个数 = 2·NY (MATLAB count_sites 对拍)."""
    lat, _ = SC(1.0)
    for ny in (1, 2, 3):
        basis, origin, keys = build_basis(_spec_from(lat, ny))
        assert len(basis) == 2 * ny


def test_build_basis_np_cell_order():
    """NP cell 序: y 胞主序, 胞内 r0..r3 (与 MATLAB build_H_np_ribbon 一致)."""
    lat, _ = NP(1.0)
    basis, origin, keys = build_basis(_spec_from(lat, 2))
    assert origin == [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
    ]
    # 坐标
    assert basis[0] == (0.0, 0.0)   # 胞0 r0
    assert basis[3] == (1.0, 0.0)   # 胞0 r3
    assert basis[4] == (0.0, 2.0)   # 胞1 r0
    assert basis[7] == (1.0, 2.0)   # 胞1 r3
    # keys 量化键
    assert keys[(round(1.0, 9), round(2.0, 9))] == 7


def test_build_basis_sc_cell_order():
    """SC cell 序: r0(A) r1(B) 每胞交替 (与 MATLAB build_H_sc_ribbon 一致)."""
    lat, _ = SC(1.0)
    basis, origin, keys = build_basis(_spec_from(lat, 2))
    assert origin == [(0, 0), (1, 0), (0, 1), (1, 1)]
    assert basis[0] == (1.0, 0.0)  # A
    assert basis[1] == (0.0, 1.0)  # B
    assert basis[2] == (1.0, 2.0)  # 胞1 A
    assert basis[3] == (0.0, 3.0)  # 胞1 B


def test_build_basis_site_order():
    """site 序: 绝对坐标 x 主."""
    lat, _ = NP(1.0)
    basis, origin, keys = build_basis(_spec_from(lat, 1, order="site"))
    # x=0 两格点在前 (r0 y0, r1 y1), 然后 x=1 (r3 y0, r2 y1)
    assert basis == [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]


def test_build_basis_keys_unique_and_total():
    """keys 无冲突: 每个基格点一个唯一量化键."""
    for name, factory in (("np", NP), ("sc", SC)):
        lat, _ = factory(1.0)
        basis, _, keys = build_basis(_spec_from(lat, 4))
        assert len(set(keys.keys())) == len(basis)


# ---------------- build_ribbon ----------------

def _rows(lattice, hops):
    for h in hops:
        yield (h.from_site, h.to_site, h.cell_offset, h.evaluate())


def test_build_ribbon_hermitian_and_shape():
    """构建的 H0 严格厄米, H0/H1 尺寸 = Nat."""
    from hamivisualizer.model.ribbon import RibbonHamiltonian, build_ribbon

    for factory, n in ((NP, 4), (SC, 2)):
        lat, hops = factory(np.pi / 4)
        spec = _spec_from(lat, 3)
        rb = build_ribbon(spec, _rows(lat, hops))
        assert rb.H0.shape == (3 * n, 3 * n)
        assert rb.H1.shape == (3 * n, 3 * n)
        assert rb.hermitian_check() == 0.0
        # 基数量与 H 尺寸一致
        assert len(rb.basis) == rb.Nat


def test_build_ribbon_herm_conj_rule():
    """cs=0 键反向共轭补全: H0[i,j] = conj(H0[j,i])."""
    from hamivisualizer.model.ribbon import build_ribbon

    lat, hops = NP(np.pi / 4)
    rb = build_ribbon(_spec_from(lat, 2), _rows(lat, hops))
    H0 = rb.H0
    for i in range(rb.Nat):
        for j in range(rb.Nat):
            assert abs(H0[i, j] - np.conj(H0[j, i])) < 1e-12


def test_build_ribbon_stats():
    """统计: 底部 y 越界被正确计为 y_cut (NP NY=2 的 NN 下键)."""
    from hamivisualizer.model.ribbon import build_ribbon

    lat, hops = NP(np.pi / 4)
    rb = build_ribbon(_spec_from(lat, 2), _rows(lat, hops))
    assert rb.stats["y_cut"] >= 0
    assert rb.stats.get("miss", 0) == 0  # 所有键目标都在基上


def test_H_hermitian_at_kx():
    """H(kx) 在任意 kx 处厄米 (实数特征值)."""
    from hamivisualizer.model.ribbon import build_ribbon

    lat, hops = SC(np.pi / 4)
    rb = build_ribbon(_spec_from(lat, 3), _rows(lat, hops))
    for kx in (0.0, 1.1, -2.3):
        H = rb.H(kx)
        assert np.max(np.abs(H - H.conj().T)) < 1e-10


def test_long_range_x_harmonic_matches_analytic_chain_dispersion():
    from hamivisualizer.model.boundary import Boundary, BoundaryKind
    from hamivisualizer.model.hamiltonian import HamiltonianBuilder
    from hamivisualizer.model.hopping import HoppingTerm
    from hamivisualizer.model.lattice import Lattice, Site

    lattice = Lattice([Site(0, 0.0, 0.0, "A")], Lx=1.0, Ly=1.0)
    hops = [
        HoppingTerm("t", 0, 0, (1, 0), -1.0),
        HoppingTerm("t2", 0, 0, (2, 0), -0.2),
    ]
    result = HamiltonianBuilder(
        lattice, hops, Boundary(BoundaryKind.SEMI, NY=1),
    ).build()
    assert set(distance for distance, _i, _j in result.extra) == {2}
    for kx in np.linspace(-np.pi, np.pi, 17):
        expected = -2.0 * np.cos(kx) - 0.4 * np.cos(2.0 * kx)
        np.testing.assert_allclose(
            result.to_semi(kx)[0, 0], expected, atol=1e-12,
        )
