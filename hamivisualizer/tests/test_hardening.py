"""Regression tests for correctness, trust boundaries, and validation."""

from __future__ import annotations

import json

import numpy as np
import pytest
import sympy as sp

from hamivisualizer.model.boundary import Boundary, BoundaryKind
from hamivisualizer.model.expression import (
    ExpressionError, collect_symbols, parse_expression,
)
from hamivisualizer.model.hamiltonian import (
    DENSE_WORKING_SET_LIMIT_BYTES,
    HamiltonianBuilder,
    estimate_dense_working_set_bytes,
)
from hamivisualizer.model.hopping import HoppingTerm
from hamivisualizer.model.lattice import Lattice, Site
from hamivisualizer.model.persistence import MAX_MODEL_BYTES, load_model
from hamivisualizer.model.presets import NP, SC
from hamivisualizer.model.symbolic import sym_pretty


@pytest.mark.parametrize("factory", [NP, SC])
@pytest.mark.parametrize("kind", [BoundaryKind.SEMI, BoundaryKind.OBC])
def test_order_is_only_a_basis_permutation(factory, kind):
    lattice, hops = factory(np.pi / 4)
    boundary = (
        Boundary(kind, NY=3) if kind is BoundaryKind.SEMI
        else Boundary(kind, NX=3, NY=2)
    )
    cell = HamiltonianBuilder(lattice, hops, boundary, "cell").build()
    site = HamiltonianBuilder(lattice, hops, boundary, "site").build()
    if kind is BoundaryKind.SEMI:
        cell_h = cell.to_semi(0.371)
        site_h = site.to_semi(0.371)
    else:
        cell_h, site_h = cell.H, site.H
    assert np.allclose(
        np.linalg.eigvalsh(cell_h), np.linalg.eigvalsh(site_h), atol=1e-11
    )


def test_reverse_or_negative_bond_is_not_dropped():
    lattice = Lattice([Site(0, 0, 0), Site(1, 1, 0)], Lx=2, Ly=1)
    positive = [HoppingTerm("t", 0, 1, (0, 0), 2 + 3j)]
    reverse = [HoppingTerm("t", 1, 0, (0, 0), 2 - 3j)]
    boundary = Boundary(BoundaryKind.OBC, NX=1, NY=1)
    h1 = HamiltonianBuilder(lattice, positive, boundary).build().H
    h2 = HamiltonianBuilder(lattice, reverse, boundary).build().H
    assert np.allclose(h1, h2)
    assert abs(h2[0, 1]) > 0


def test_complex_onsite_is_rejected():
    lattice = Lattice([Site(0, 0, 0)], Lx=1, Ly=1)
    hops = [HoppingTerm("bad", 0, 0, (0, 0), 1 + 0.2j)]
    with pytest.raises(ValueError, match="on-site"):
        HamiltonianBuilder(
            lattice, hops, Boundary(BoundaryKind.OBC, NX=1, NY=1)
        ).build()


def test_expression_language_blocks_python_execution():
    with pytest.raises(ExpressionError):
        parse_expression("__import__('os').getcwd()")
    with pytest.raises(ExpressionError):
        parse_expression("x.__class__")
    expr = parse_expression("-sqrt(2)*t*cos(phi) + 1/3")
    assert {str(s) for s in expr.free_symbols} == {"t", "phi"}


def test_kx_alias_is_not_exposed_as_a_user_parameter():
    assert collect_symbols(["k_x + t", "kx - phi"]) == {"t", "phi"}


def test_symbolic_printer_keeps_numeric_power_factors():
    t = sp.Symbol("t", real=True)
    rendered = sym_pretty(sp.sqrt(2) * t)
    assert "t" in rendered
    assert "sqrt" in rendered or "\\sqrt" in rendered


def test_model_value_validation():
    with pytest.raises(ValueError, match="正整数"):
        Boundary(BoundaryKind.OBC, NX=0, NY=1)
    with pytest.raises(ValueError, match="重复"):
        Lattice([Site(0, 0, 0), Site(1, 0, 0)], Lx=1, Ly=1)


def test_dense_backend_guard_rejects_unsafe_dimensions_before_allocation():
    """Large study inputs fail clearly instead of attempting a huge ndarray."""
    assert estimate_dense_working_set_bytes(10, BoundaryKind.SEMI) == 4_800
    with pytest.raises(ValueError, match="安全预算"):
        HamiltonianBuilder(
            Lattice([Site(0, 0.0, 0.0)], Lx=1.0, Ly=1.0), [],
            Boundary(BoundaryKind.SEMI, NY=100_000),
        ).build()
    with pytest.raises(ValueError, match="安全预算"):
        HamiltonianBuilder(
            Lattice([Site(0, 0.0, 0.0)], Lx=1.0, Ly=1.0), [],
            Boundary(BoundaryKind.OBC, NX=100_000, NY=100_000),
        ).build()
    assert DENSE_WORKING_SET_LIMIT_BYTES > estimate_dense_working_set_bytes(10, BoundaryKind.OBC)


def test_untrusted_json_expression_is_rejected(tmp_path):
    obj = {
        "version": 1,
        "sites": [{"x": 0.0, "y": 0.0, "sublattice": "A"}],
        "hops": [{
            "name": "t", "from_site": 0, "to_site": 0,
            "cell_offset": [0, 0],
            "amplitude": "__import__('os').getcwd()",
            "phase_mode": "none", "phase": "0", "phase_sign": 1,
        }],
        "boundary": {"kind": "obc", "NX": 1, "NY": 1},
        "order": "cell", "params": {}, "kx": 0.0,
        "symbolic": False, "smart": True,
    }
    path = tmp_path / "hostile.json"
    path.write_text(json.dumps(obj), encoding="utf-8")
    with pytest.raises(ValueError, match="表达式"):
        load_model(path)


def test_oversized_model_is_rejected_before_json_parsing(tmp_path):
    path = tmp_path / "oversized.hvisual"
    with path.open("wb") as stream:
        stream.write(b"0" * (MAX_MODEL_BYTES + 1))
    with pytest.raises(ValueError, match="文件过大"):
        load_model(path)
