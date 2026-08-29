"""★ 半无限 H0/H1 与 MATLAB 直译参考逐元对拍 — 数值正确性锚点.

泛化实现 (build_ribbon) 与 MATLAB 逻辑 (reference_matlab) 对 NP/SC 预设
必须逐矩阵元一致 (阈值 <1e-10), 锁定数值正确性后再接视图层。
"""

import numpy as np

from hamivisualizer.model.boundary import Boundary, BoundaryKind
from hamivisualizer.model.hamiltonian import HamiltonianBuilder
from hamivisualizer.model.presets import NP, SC

from reference_matlab import ref_build_np_ribbon, ref_build_sc_ribbon

TOL = 1e-10


def _semi_result(factory, phi, NY, t=1.0, omg=1.0):
    lat, hops = factory(phi, t, omg)
    b = Boundary(BoundaryKind.SEMI, NY=NY)
    return HamiltonianBuilder(lat, hops, b).build_semi()


# ---------------- NP ----------------

def test_np_semi_H0H1_match_matlab():
    for NY in (1, 2, 3, 4):
        phi = np.pi / 4
        H0_ref, H1_ref = ref_build_np_ribbon(NY, 1.0, phi, 1.0)
        res = _semi_result(NP, phi, NY)
        H0, H1 = res.blocks["H0"], res.blocks["H1"]
        assert np.max(np.abs(H0 - H0_ref)) < TOL, f"NP NY={NY} H0 mismatch"
        assert np.max(np.abs(H1 - H1_ref)) < TOL, f"NP NY={NY} H1 mismatch"


def test_np_semi_phase_variants():
    """不同相位下仍逐元一致 (相位符号是泛化实现的关键)."""
    for phi in (0.0, np.pi / 6, np.pi / 3, -0.7):
        H0_ref, H1_ref = ref_build_np_ribbon(2, 1.0, phi, 1.0)
        res = _semi_result(NP, phi, 2)
        H0, H1 = res.blocks["H0"], res.blocks["H1"]
        assert np.max(np.abs(H0 - H0_ref)) < TOL
        assert np.max(np.abs(H1 - H1_ref)) < TOL


def test_np_semi_hermitian():
    for NY in (1, 2, 3):
        res = _semi_result(NP, np.pi / 4, NY)
        H0, H1 = res.blocks["H0"], res.blocks["H1"]
        assert np.max(np.abs(H0 - H0.conj().T)) < TOL
        # H1 无矩阵对角自项 (NP 无跨胞自跳)
        assert np.max(np.abs(np.diag(H1))) == 0


# ---------------- SC ----------------

def test_sc_semi_H0H1_match_matlab():
    for NY in (1, 2, 3, 4):
        phi = np.pi / 4
        H0_ref, H1_ref = ref_build_sc_ribbon(NY, 1.0, phi, 1.0)
        res = _semi_result(SC, phi, NY)
        H0, H1 = res.blocks["H0"], res.blocks["H1"]
        assert np.max(np.abs(H0 - H0_ref)) < TOL, f"SC NY={NY} H0 mismatch"
        assert np.max(np.abs(H1 - H1_ref)) < TOL, f"SC NY={NY} H1 mismatch"


def test_sc_semi_phase_variants():
    for phi in (0.0, np.pi / 4, -1.1):
        H0_ref, H1_ref = ref_build_sc_ribbon(2, 1.0, phi, 1.0)
        res = _semi_result(SC, phi, 2)
        H0, H1 = res.blocks["H0"], res.blocks["H1"]
        assert np.max(np.abs(H0 - H0_ref)) < TOL
        assert np.max(np.abs(H1 - H1_ref)) < TOL


def test_sc_semi_hermitian():
    for NY in (1, 2, 3):
        res = _semi_result(SC, np.pi / 4, NY)
        H0, H1 = res.blocks["H0"], res.blocks["H1"]
        assert np.max(np.abs(H0 - H0.conj().T)) < TOL
        # SC 有 B-site 跨胞自跳 → H1 对角非零
        assert np.max(np.abs(np.diag(H1))) > 0


# ---------------- 能带 ----------------

def test_bands_shapes_and_real():
    for factory, NY, nat in ((NP, 2, 8), (SC, 2, 4)):
        res = _semi_result(factory, np.pi / 4, NY)
        rb = res.blocks
        from hamivisualizer.model.ribbon import RibbonHamiltonian

        ham = RibbonHamiltonian(rb["H0"], rb["H1"])
        kx = np.linspace(-np.pi, np.pi, 51)
        kxs, E = ham.bands(kx)
        assert kxs.shape == (51,)
        assert E.shape == (51, nat)
        assert np.all(np.isreal(E))


def test_bands_band_count():
    """每个 kx 有 Nat 条带且按能量升序 (NP/SC 的复 NN 相位破缺 TR,
    故不要求 E(k)=E(-k))."""
    for factory, NY, nat in ((NP, 2, 8), (SC, 2, 4)):
        res = _semi_result(factory, np.pi / 4, NY)
        from hamivisualizer.model.ribbon import RibbonHamiltonian

        ham = RibbonHamiltonian(res.blocks["H0"], res.blocks["H1"])
        _, E = ham.bands(np.linspace(-np.pi, np.pi, 21))
        assert E.shape[1] == nat
        assert np.all(np.diff(E, axis=1) >= -1e-9)  # 每 kx 升序


def test_Hkx_matches_eig_of_single_point():
    """bands 与该 kx 点直接 H(kx) 对角化一致."""
    res = _semi_result(NP, np.pi / 4, 2)
    from hamivisualizer.model.ribbon import RibbonHamiltonian

    ham = RibbonHamiltonian(res.blocks["H0"], res.blocks["H1"])
    kx0 = 0.37
    _, E = ham.bands([kx0])
    Hk = ham.H(kx0)
    assert np.allclose(np.sort(np.linalg.eigvalsh(Hk)), E[0], atol=1e-10)


# ---------------- OBC 对拍 ----------------

def _obc_result(factory, phi, NX, NY, t=1.0, omg=1.0):
    lat, hops = factory(phi, t, omg)
    b = Boundary(BoundaryKind.OBC, NX=NX, NY=NY)
    return HamiltonianBuilder(lat, hops, b).build_obc()


def test_np_obc_match_matlab():
    from reference_matlab import ref_build_np_obc

    for NX, NY in [(1, 1), (2, 2), (2, 3), (3, 2), (3, 3)]:
        for phi in (np.pi / 4, 0.0, -0.6):
            H_ref, _ = ref_build_np_obc(NX, NY, 1.0, phi, 1.0)
            res = _obc_result(NP, phi, NX, NY)
            assert np.max(np.abs(res.H - H_ref)) < TOL, f"NP OBC {NX}x{NY} phi={phi:.2f}"


def test_sc_obc_match_matlab():
    from reference_matlab import ref_build_sc_obc

    for NX, NY in [(1, 1), (2, 2), (2, 3), (3, 2), (3, 3)]:
        for phi in (np.pi / 4, 0.0, -1.1):
            H_ref, _ = ref_build_sc_obc(NX, NY, 1.0, phi, 1.0)
            res = _obc_result(SC, phi, NX, NY)
            assert np.max(np.abs(res.H - H_ref)) < TOL, f"SC OBC {NX}x{NY} phi={phi:.2f}"


def test_obc_hermitian_and_sites():
    for factory, nat in ((NP, 4), (SC, 2)):
        res = _obc_result(factory, np.pi / 4, 2, 3)
        assert res.Nat == nat * 6
        assert res.Nsites == nat * 6
        assert np.max(np.abs(res.H - res.H.conj().T)) < TOL
        assert len(res.positions) == res.Nat


def test_obc_eig_wavefunctions():
    from hamivisualizer.model.hamiltonian import eig, wavefunctions

    res = _obc_result(NP, np.pi / 4, 2, 2)
    E, U = eig(res.H)
    assert np.all(np.diff(E) >= -1e-12)  # 升序
    assert U.shape == (res.Nat, res.Nat)
    # 波函数 |ψ|² 归一
    E2, wf = wavefunctions(res.H)
    assert np.allclose(E, E2)
    assert wf.shape == (res.Nat, res.Nat)
    assert np.all(wf >= 0) and np.all(wf <= 1 + 1e-12)
    # 每个态最大 = 1
    assert np.allclose(wf.max(axis=0), 1.0)


def test_wavefunctions_choose_edge_localized_basis_inside_exact_degeneracy():
    """Exact degeneracy must not make an inspectable edge mode arbitrary."""
    from hamivisualizer.model.hamiltonian import edge_mask_for_positions, wavefunctions

    # The first two exactly degenerate states span one interior and one edge
    # orbital. A generic eigensolver may rotate this subspace freely; the
    # presentation basis should expose the two physically useful limits.
    positions = [
        (1.0, 1.0), (0.0, 0.0), (1.0, 0.0),
        (2.0, 0.0), (0.0, 1.0), (2.0, 1.0),
        (0.0, 2.0), (1.0, 2.0), (2.0, 2.0),
    ]
    H = np.diag([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    energies, density = wavefunctions(H, positions=positions)
    assert np.allclose(energies[:2], 0.0)
    edge = edge_mask_for_positions(positions)
    weights = np.array([
        density[edge, state].sum() / density[:, state].sum()
        for state in range(2)
    ])
    assert np.allclose(np.sort(weights), [0.0, 1.0], atol=1e-12)
