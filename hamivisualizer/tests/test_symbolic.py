"""符号模式测试: PrettyLatexPrinter / sym_pretty / combine_semi / 符号构建."""

import numpy as np
import sympy as sp

from hamivisualizer.model.boundary import Boundary, BoundaryKind
from hamivisualizer.model.hamiltonian import HamiltonianBuilder
from hamivisualizer.model.presets import NP, SC
from hamivisualizer.model.hopping import HoppingTerm
from hamivisualizer.model.lattice import Lattice, Site
from hamivisualizer.model.symbolic import combine_semi, param, sym_pretty, wrap_tex


def test_sym_pretty_t_leading():
    """t/tc/ω 必须位于所有 e 指数最左 (README 附录修复规则)."""
    t, phi, kx = param("t"), param("phi"), param("kx")
    for expr in [
        -t * sp.exp(sp.I * phi) * sp.exp(-sp.I * kx),
        sp.exp(sp.I * kx) * sp.exp(-sp.I * phi) * t,
    ]:
        s = sym_pretty(expr)
        assert "t" in s and "e^{" in s
        assert s.index("t") < s.index("e^{"), s


def test_sym_pretty_known_forms():
    t, phi, omg, kx = param("t"), param("phi"), param("omg"), param("kx")
    # 已知 MATLAB 元素 (README §7.2)
    assert sym_pretty(-t * sp.exp(sp.I * phi)) == "- t e^{i\\phi}"
    assert sym_pretty(-t * sp.exp(-sp.I * phi)) == "- t e^{-i\\phi}"
    assert sym_pretty(omg + t) == "\\omega + t"
    assert sym_pretty(-t) == "- t"
    # exp 线性指数
    s = sym_pretty(sp.exp(sp.I * (kx - 2 * phi)))
    assert s.startswith("e^{i")
    assert "k_{x}" in s and "\\phi" in s


def test_sym_pretty_numeric_parameter_suffix_is_a_true_subscript():
    assert sym_pretty(param("t1")) == "t_{1}"
    assert sym_pretty(param("t12")) == "t_{12}"
    assert sym_pretty(param("lambda2")) == "lambda_{2}"


def test_sym_pretty_no_conj():
    """real=True 参数根除 conj (MATLAB 5 条 strrep 的替代)."""
    t, kx = param("t"), param("kx")
    s = sym_pretty(sp.conjugate(t * sp.exp(sp.I * kx)))
    assert "conj" not in s
    assert "\\overline" not in s


def test_wrap_tex():
    long_s = "- t e^{i\\phi} e^{i k_{x}}"
    wrapped = wrap_tex(long_s)
    assert "\n" in wrapped  # 换行
    # 短式不换行
    assert "\n" not in wrap_tex("- t e^{i\\phi}")


def _symbolic_semi(name, NY):
    t, phi, omg = param("t"), param("phi"), param("omg")
    factory = {"np": NP, "sc": SC}[name]
    lat, hops = factory(phi, t, omg)
    res = HamiltonianBuilder(lat, hops, Boundary(BoundaryKind.SEMI, NY=NY)).build_semi()
    assert isinstance(res.blocks["H0"], sp.Matrix)
    return res


def test_symbolic_semi_hermitian():
    """符号模式 H0 严格厄米: H0 == H0†."""
    for name in ("np", "sc"):
        res = _symbolic_semi(name, 2)
        H0 = res.blocks["H0"]
        assert H0 == H0.T.conjugate()


def test_combine_semi_matches_numeric():
    """符号 H(kx) 数值代入 ≡ 数值模式 H(kx) (同一构建器, 仅 provider 不同)."""
    t, phi, omg, kx = param("t"), param("phi"), param("omg"), param("kx")
    for name, factory in (("np", NP), ("sc", SC)):
        res_s = HamiltonianBuilder(
            *factory(phi, t, omg), Boundary(BoundaryKind.SEMI, NY=2)
        ).build_semi()
        Hs = combine_semi(res_s.blocks["H0"], res_s.blocks["H1"], kx)
        kx0 = 0.37
        subs = {t: 1.0, phi: np.pi / 4, omg: 1.0, kx: kx0}
        Hs_num = np.array(
            [[complex(sp.N(e.subs(subs))) for e in row] for row in Hs.tolist()]
        )
        res_n = HamiltonianBuilder(
            *factory(np.pi / 4), Boundary(BoundaryKind.SEMI, NY=2)
        ).build_semi()
        H0, H1 = res_n.blocks["H0"], res_n.blocks["H1"]
        Hn = H0 + np.exp(1j * kx0) * H1 + np.exp(-1j * kx0) * H1.conj().T
        assert np.max(np.abs(Hs_num - Hn)) < 1e-8, name


def test_symbolic_obc_builds():
    """符号 OBC 构建成功."""
    t, phi, omg = param("t"), param("phi"), param("omg")
    lat, hops = NP(phi, t, omg)
    res = HamiltonianBuilder(lat, hops, Boundary(BoundaryKind.OBC, NX=2, NY=2)).build_obc()
    assert isinstance(res.H, sp.Matrix)


def test_symbolic_long_range_x_harmonic_is_preserved():
    t, t2, kx = param("t"), param("t2"), param("kx")
    lattice = Lattice([Site(0, 0.0, 0.0, "A")], Lx=1.0, Ly=1.0)
    result = HamiltonianBuilder(
        lattice,
        [
            HoppingTerm("t", 0, 0, (1, 0), -t),
            HoppingTerm("t2", 0, 0, (2, 0), -t2),
        ],
        Boundary(BoundaryKind.SEMI, NY=1),
    ).build()
    expression = sp.simplify(result.to_semi(kx)[0, 0])
    expected = (
        -t * (sp.exp(sp.I * kx) + sp.exp(-sp.I * kx))
        -t2 * (sp.exp(2 * sp.I * kx) + sp.exp(-2 * sp.I * kx))
    )
    assert sp.simplify(sp.expand(expression - expected)) == 0
