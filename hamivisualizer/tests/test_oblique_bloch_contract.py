"""Contract tests for x-Bloch ribbons with an oblique first cell vector."""

from __future__ import annotations

import numpy as np
import pytest

from hamivisualizer.model.boundary import Boundary, BoundaryKind
from hamivisualizer.model.hamiltonian import HamiltonianBuilder
from hamivisualizer.model.hopping import HoppingTerm
from hamivisualizer.model.lattice import Lattice, Site


def _lattice() -> Lattice:
    return Lattice(
        [Site(0, 0.0, 0.0, "A"), Site(1, 0.35, 0.2, "B")],
        a1=(1.0, 0.5), a2=(0.0, 1.0),
    )


def _hops() -> list[HoppingTerm]:
    return [
        HoppingTerm("t0", 0, 0, (0, 0), 0.25),
        HoppingTerm("t1", 0, 1, (0, 0), -0.8 + 0.15j),
        HoppingTerm("tx", 1, 0, (1, 0), 0.4 - 0.2j),
    ]


def _build(order: str, kind: BoundaryKind, *, nx=3, ny=3):
    return HamiltonianBuilder(
        _lattice(), _hops(),
        Boundary(kind, NX=nx, NY=ny), order=order,
    ).build()


@pytest.mark.parametrize("order", ("cell", "site"))
def test_oblique_semi_uses_a2_layers_and_has_hermitian_bloch_matrix(order):
    result = _build(order, BoundaryKind.SEMI)
    assert result.Nat == 2 * 3
    assert sorted(cy for _r, cy in result.origin) == [0, 0, 1, 1, 2, 2]
    for kx in (-2.4, -0.1, 0.0, 1.7, 3.0):
        matrix = np.asarray(result.to_semi(kx), dtype=complex)
        np.testing.assert_allclose(matrix, matrix.conj().T, atol=1e-12)


def test_oblique_obc_positions_use_complete_a1_and_matrix_is_hermitian():
    result = _build("cell", BoundaryKind.OBC, nx=3, ny=2)
    lattice = _lattice()
    coords = [lattice.position(*cell_site) for cell_site in result.rmap]
    assert result.H.shape == (12, 12)
    np.testing.assert_allclose(result.H, result.H.conj().T, atol=1e-12)
    for cy in range(2):
        p0 = np.asarray(lattice.position(0, cy, 0))
        p1 = np.asarray(lattice.position(1, cy, 0))
        assert p1[1] - p0[1] == pytest.approx(0.5)
    assert any(abs(y - 0.5) < 1e-12 for _x, y in coords)


def test_oblique_semi_spectrum_is_independent_of_basis_order():
    cell = _build("cell", BoundaryKind.SEMI)
    site = _build("site", BoundaryKind.SEMI)
    for kx in (-1.3, 0.0, 2.1):
        np.testing.assert_allclose(
            np.linalg.eigvalsh(cell.to_semi(kx)),
            np.linalg.eigvalsh(site.to_semi(kx)),
            atol=1e-12,
        )


def test_oblique_cell_vector_is_preserved_in_lattice_metadata():
    lattice = _lattice()
    assert lattice.a1 == (1.0, 0.5)
    assert lattice.a2 == (0.0, 1.0)
    assert lattice.position(1, 0, 0) == pytest.approx((1.0, 0.5))
