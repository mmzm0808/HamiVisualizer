"""All built-in templates must satisfy the same matrix invariants.

The UI exposes the templates as ready-to-use physics examples.  A regression
in one less frequently used preset (or in its oblique-cell representation)
must therefore be caught by the same test as the two historical NP/SC cases.
"""

from __future__ import annotations

import numpy as np
import pytest

from hamivisualizer.model.boundary import Boundary, BoundaryKind
from hamivisualizer.model.expression import evaluate_expression
from hamivisualizer.model.hamiltonian import HamiltonianBuilder
from hamivisualizer.model.hopping import HoppingTerm
from hamivisualizer.model.lattice import Lattice, Site
from hamivisualizer.model.templates import TEMPLATE_NAMES, template_document


def _builder_from_document(document: dict, kind: BoundaryKind) -> HamiltonianBuilder:
    """Reconstruct the public model objects exactly as the controller does."""
    sites = [
        Site(index, row["x"], row["y"], row.get("sublattice") or None)
        for index, row in enumerate(document["sites"])
    ]
    cell = document.get("cell")
    if cell is None:
        lattice = Lattice(sites)
    elif "a1" in cell:
        lattice = Lattice(
            sites,
            a1=tuple(cell["a1"]),
            a2=tuple(cell["a2"]),
        )
    else:
        lattice = Lattice(sites, Lx=cell["Lx"], Ly=cell["Ly"])

    params = document["params"]
    hops = []
    for row in document["hops"]:
        phase_mode = row["phase_mode"]
        amplitude = evaluate_expression(row["amplitude"], params)
        phase = (
            evaluate_expression(row["phase"], params)
            if phase_mode == "phase" else 0.0
        )
        hops.append(HoppingTerm(
            name=row["name"],
            from_site=row["from_site"],
            to_site=row["to_site"],
            cell_offset=tuple(row["cell_offset"]),
            amplitude=amplitude,
            phase_mode=phase_mode,
            phase=phase,
            phase_sign=row["phase_sign"],
        ))
    boundary = Boundary(kind, NX=3, NY=3, shape="rectangle")
    return HamiltonianBuilder(lattice, hops, boundary)


@pytest.mark.parametrize("name", TEMPLATE_NAMES)
@pytest.mark.parametrize("kind", [BoundaryKind.SEMI, BoundaryKind.OBC])
def test_every_builtin_template_is_finite_and_hermitian(name, kind):
    """Every preset remains a valid Hermitian model in both boundary modes."""
    document = template_document(
        name,
        nx=3,
        ny=3,
        boundary_kind="semi" if kind is BoundaryKind.SEMI else "obc",
        connectivity="最近邻+次近邻",
        shape="rectangle",
    )
    result = _builder_from_document(document, kind).build()
    if kind is BoundaryKind.SEMI:
        matrices = [result.to_semi(kx) for kx in (-2.7, -0.4, 0.0, 1.2, 3.1)]
    else:
        matrices = [result.H]
    for matrix in matrices:
        numeric = np.asarray(matrix, dtype=complex)
        assert numeric.ndim == 2 and numeric.shape[0] == numeric.shape[1]
        assert np.all(np.isfinite(numeric))
        np.testing.assert_allclose(numeric, numeric.conj().T, atol=1e-9)
