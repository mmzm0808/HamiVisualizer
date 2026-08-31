"""Cross-check the semi-infinite x-Bloch harmonics against a periodic lift."""

from __future__ import annotations

import numpy as np
import pytest

from hamivisualizer.model.boundary import Boundary, BoundaryKind
from hamivisualizer.model.hamiltonian import HamiltonianBuilder
from hamivisualizer.model.hopping import HoppingTerm
from hamivisualizer.model.lattice import Lattice, Site


def _periodic_lift(result, nx: int) -> np.ndarray:
    """Lift H0/H1/extra to an explicit Nx-cell periodic block matrix."""
    nat = result.Nat
    matrix = np.zeros((nx * nat, nx * nat), dtype=complex)

    def block(row_cell, col_cell, value):
        rows = slice(row_cell * nat, (row_cell + 1) * nat)
        cols = slice(col_cell * nat, (col_cell + 1) * nat)
        matrix[rows, cols] += np.asarray(value, dtype=complex)

    for cell in range(nx):
        block(cell, cell, result.blocks["H0"])
        right = (cell + 1) % nx
        block(cell, right, result.blocks["H1"])
        block(right, cell, np.asarray(result.blocks["H1"], dtype=complex).conj().T)
        for distance, i, j in result.extra:
            value = result.extra[(distance, i, j)]
            forward = np.zeros((nat, nat), dtype=complex)
            forward[i, j] = value
            target = (cell + distance) % nx
            block(cell, target, forward)
            block(target, cell, forward.conj().T)
    return matrix


def _builder(
    order: str, hops: list[HoppingTerm], kind: BoundaryKind, *, nx=1,
    lattice: Lattice | None = None,
):
    lattice = lattice or Lattice([Site(0, 0.0, 0.0, "A")], Lx=1.0, Ly=1.0)
    return HamiltonianBuilder(
        lattice, hops, Boundary(kind, NX=nx, NY=1), order=order,
    )


@pytest.mark.parametrize("order", ("cell", "site"))
def test_periodic_lift_matches_bloch_spectrum_for_long_range_complex_chain(order):
    hops = [
        HoppingTerm("t1", 0, 0, (1, 0), 0.7 + 0.2j),
        HoppingTerm("t2", 0, 0, (-2, 0), -0.3 + 0.4j),
    ]
    result = _builder(order, hops, BoundaryKind.SEMI).build()
    nx = 7
    periodic = _periodic_lift(result, nx)
    expected = []
    for j in range(nx):
        expected.extend(np.linalg.eigvalsh(result.to_semi(2 * np.pi * j / nx)))
    np.testing.assert_allclose(
        np.linalg.eigvalsh(periodic), np.sort(expected), atol=1e-9,
    )


def test_periodic_lift_matches_haldane_complex_phase_cell():
    root3 = np.sqrt(3.0)
    lattice = Lattice(
        [Site(0, 0.0, 0.0, "A"), Site(1, root3 / 2, 0.5, "B")],
        a1=(root3, 0.0), a2=(root3 / 2, 1.5),
    )
    hops = [
        HoppingTerm("t", 0, 1, (0, 0), -1.0),
        HoppingTerm("t", 0, 1, (-1, 0), -1.0),
        HoppingTerm("t", 0, 1, (0, -1), -1.0),
        *[
            HoppingTerm("t2", 0, 0, offset, -0.2, "phase", np.pi / 2, 1)
            for offset in ((1, 0), (0, 1), (1, -1))
        ],
        *[
            HoppingTerm("t2", 1, 1, offset, -0.2, "phase", np.pi / 2, -1)
            for offset in ((1, 0), (0, 1), (1, -1))
        ],
    ]
    result = _builder(
        "cell", hops, BoundaryKind.SEMI, lattice=lattice,
    ).build()
    nx = 8
    periodic = _periodic_lift(result, nx)
    expected = np.concatenate([
        np.linalg.eigvalsh(result.to_semi(2 * np.pi * j / nx))
        for j in range(nx)
    ])
    np.testing.assert_allclose(
        np.linalg.eigvalsh(periodic), np.sort(expected), atol=1e-9,
    )


def test_obc_long_range_offsets_do_not_wrap_and_count_each_boundary_loss():
    result = _builder(
        "cell",
        [
            HoppingTerm("t1", 0, 0, (1, 0), -1.0),
            HoppingTerm("t2", 0, 0, (-2, 0), -0.2),
        ],
        BoundaryKind.OBC,
        nx=7,
    ).build()
    assert result.skipped.get("oob") == 3
    assert result.H.shape == (7, 7)
    assert result.H[0, -1] == 0
    assert result.H[-1, 0] == 0
    assert result.H[0, 1] != 0
    assert result.H[0, 2] != 0
