"""Built-in lattice templates used by the new-model wizard."""

from __future__ import annotations

import math

from .boundary import Boundary, BoundaryKind
from .persistence import model_to_dict


TEMPLATE_NAMES = (
    "NP",
    "SC",
    "空白自定义",
    "方格",
    "一维链",
    "蜂窝",
    "三角",
    "Kagome",
    "SSH",
    "Haldane",
)


def _hop(
    fr, to, ox=0, oy=0, *, name="t", amp="-t", phase="0", phase_sign=1,
):
    mode = "phase" if phase != "0" else "none"
    return {
        "name": name,
        "from_site": fr,
        "to_site": to,
        "off_x": ox,
        "off_y": oy,
        "amplitude": amp,
        "phase_mode": mode,
        "phase": phase,
        "phase_sign": int(phase_sign),
    }


def _periodic_bonds(sites, cell, distance: float, *, name="t", amp="-t"):
    """Generate each undirected bond once in an orthogonal periodic cell."""
    lx, ly = cell
    found = set()
    rows = []
    for i, (xi, yi, _si) in enumerate(sites):
        for j, (xj, yj, _sj) in enumerate(sites):
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    if i == j and ox == 0 and oy == 0:
                        continue
                    dx = xj + ox * lx - xi
                    dy = yj + oy * ly - yi
                    if not math.isclose(math.hypot(dx, dy), distance, rel_tol=1e-7, abs_tol=1e-7):
                        continue
                    key = (i, j, ox, oy)
                    reverse = (j, i, -ox, -oy)
                    canonical = min(key, reverse)
                    if canonical in found:
                        continue
                    found.add(canonical)
                    rows.append(_hop(*canonical, name=name, amp=amp))
    return rows


def _periodic_bonds_vectors(
    sites, vectors, distance: float, *, name="t", amp="-t",
):
    """Generate one copy of each bond for an arbitrary 2-D Bravais cell.

    The original helper intentionally accepts only orthogonal ``(Lx, Ly)``
    cells.  Kagome's three-site primitive cell is oblique, so keep that legacy
    path unchanged and use this vector-aware variant for the triangular
    nanodisk representation.
    """
    (a1x, a1y), (a2x, a2y) = vectors
    found = set()
    rows = []
    for i, (xi, yi, _si) in enumerate(sites):
        for j, (xj, yj, _sj) in enumerate(sites):
            # A nearest/next-nearest bond in this primitive cell cannot span
            # more than one neighbouring cell.  The ±1 stencil also makes the
            # helper deterministic for the compact built-in templates.
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    if i == j and ox == 0 and oy == 0:
                        continue
                    dx = xj + ox * a1x + oy * a2x - xi
                    dy = yj + ox * a1y + oy * a2y - yi
                    if not math.isclose(
                        math.hypot(dx, dy), distance,
                        rel_tol=1e-7, abs_tol=1e-7,
                    ):
                        continue
                    key = (i, j, ox, oy)
                    reverse = (j, i, -ox, -oy)
                    canonical = min(key, reverse)
                    if canonical in found:
                        continue
                    found.add(canonical)
                    rows.append(_hop(*canonical, name=name, amp=amp))
    return rows


def _kagome_primitive(connectivity: str):
    """Return a three-site oblique Kagome primitive cell.

    ``a₁=(2,0)``, ``a₂=(1,√3)`` form a 60° Bravais cell and the three basis
    sites sit at the vertices of one elementary triangle.  A triangular OBC
    mask can therefore select ``cx + cy <= N-1`` cells and remain an exact
    equilateral nanodisk.  The semi-infinite preset intentionally continues to
    use the six-site orthogonal supercell below, which is easier to inspect as
    a vertical ribbon.
    """
    root3 = math.sqrt(3.0)
    vectors = ((2.0, 0.0), (1.0, root3))
    sites = [
        (0.0, 0.0, "A"),
        (1.0, 0.0, "B"),
        (0.5, root3 / 2.0, "C"),
    ]
    nn = _periodic_bonds_vectors(sites, vectors, 1.0)
    nnn = _periodic_bonds_vectors(
        sites, vectors, root3, name="t2", amp="-t2",
    )
    hops = [] if connectivity == "仅格点" else nn
    if connectivity == "最近邻+次近邻":
        hops = nn + nnn
    params = {"t": 1.0}
    if any(h["name"] == "t2" for h in hops):
        params["t2"] = 1.0
    return sites, hops, params, {"a1": vectors[0], "a2": vectors[1]}


def _generic(name: str, connectivity: str):
    nn: list[dict] = []
    nnn: list[dict] = []
    if name == "空白自定义":
        sites = [(0.0, 0.0, "A")]
        cell = (1.0, 1.0)
    elif name == "一维链":
        sites = [(0.0, 0.0, "A")]
        cell = (1.0, 1.0)
        nn = [_hop(0, 0, 1, 0)]
        nnn = [_hop(0, 0, 2, 0, name="t2", amp="-t2")]
    elif name == "SSH":
        # Su–Schrieffer–Heeger dimer chain.  The two sites are separated by
        # half a unit cell; t1 is the intracell bond and t2 crosses the cell
        # boundary.  Keeping the intercell offset explicit is important: the
        # OBC builder can then expose the two terminal edge states instead of
        # silently turning the model into a periodic two-site molecule.
        sites = [(0.0, 0.0, "A"), (0.5, 0.0, "B")]
        cell = (1.0, 1.0)
        nn = [
            _hop(0, 1, name="t1", amp="-t1"),
            _hop(1, 0, 1, 0, name="t2", amp="-t2"),
        ]
        nnn = []
    elif name == "Haldane":
        # Minimal Haldane honeycomb model on the same two-site oblique cell
        # used by the graphene preset.  The three A-sublattice NNN bonds
        # carry +phi and the three B-sublattice bonds carry -phi, so the
        # reverse direction is supplied by the Hermitian builder with the
        # conjugate phase.  The staggered mass m is kept as an explicit pair
        # of onsite rows and therefore remains editable in the table.
        root3 = math.sqrt(3.0)
        cell = {"a1": (root3, 0.0), "a2": (root3 / 2, 1.5)}
        sites = [
            (0.0, 0.0, "A"), (root3 / 2, 0.5, "B"),
        ]
        nearest = [
            _hop(0, 1), _hop(0, 1, -1, 0), _hop(0, 1, 0, -1),
        ]
        nnn = [
            _hop(0, 0, ox, oy, name="t2", amp="-t2",
                 phase="phi", phase_sign=+1)
            for ox, oy in ((1, 0), (0, 1), (1, -1))
        ] + [
            _hop(1, 1, ox, oy, name="t2", amp="-t2",
                 phase="phi", phase_sign=-1)
            for ox, oy in ((1, 0), (0, 1), (1, -1))
        ]
        onsite = [
            _hop(0, 0, name="m", amp="m"),
            _hop(1, 1, name="m", amp="-m"),
        ]
        nn = onsite + nearest
    elif name == "方格":
        sites = [(0.0, 0.0, "A")]
        cell = (1.0, 1.0)
        nn = [_hop(0, 0, 1, 0), _hop(0, 0, 0, 1)]
        nnn = [
            _hop(0, 0, 1, 1, name="t2", amp="-t2"),
            _hop(0, 0, 1, -1, name="t2", amp="-t2"),
        ]
    elif name == "三角":
        # Orthogonal supercell of the triangular Bravais lattice.
        root3 = math.sqrt(3.0)
        cell = (1.0, root3)
        sites = [(0.0, 0.0, "A"), (0.5, root3 / 2, "A")]
        nn = _periodic_bonds(sites, cell, 1.0)
        nnn = _periodic_bonds(sites, cell, root3, name="t2", amp="-t2")
    elif name == "蜂窝":
        # 真正的 graphene primitive cell：两格点 + 斜平行四边形 Bravais 元胞。
        # a1 为半无限 x-Bloch 方向，a2 为有限方向；最近邻距离均为 1。
        root3 = math.sqrt(3.0)
        cell = {"a1": (root3, 0.0), "a2": (root3 / 2, 1.5)}
        sites = [
            (0.0, 0.0, "A"), (root3 / 2, 0.5, "B"),
        ]
        nn = [_hop(0, 1), _hop(0, 1, -1, 0), _hop(0, 1, 0, -1)]
        nnn = [
            _hop(r, r, ox, oy, name="t2", amp="-t2")
            for r in (0, 1) for ox, oy in ((1, 0), (0, 1), (1, -1))
        ]
    elif name == "Kagome":
        # Orthogonal two-primitive-cell representation of Kagome.
        root3 = math.sqrt(3.0)
        cell = (2.0, 2 * root3)
        sites = [
            (0.0, 0.0, "A"), (1.0, 0.0, "B"), (0.5, root3 / 2, "C"),
            (1.0, root3, "A"), (0.0, root3, "B"), (1.5, 3 * root3 / 2, "C"),
        ]
        nn = _periodic_bonds(sites, cell, 1.0)
        nnn = _periodic_bonds(sites, cell, root3, name="t2", amp="-t2")
    else:
        raise ValueError(f"未知模板: {name}")
    hops = [] if connectivity == "仅格点" else nn
    if connectivity == "最近邻+次近邻":
        hops = nn + nnn
    # Every built-in preset starts from a unit hopping scale.  SSH is the
    # only model with two independent nearest-neighbour families, so expose
    # both symbols explicitly instead of first creating a legacy ``t`` entry
    # and overwriting it below.
    if name == "SSH":
        params = {"t1": 1.0, "t2": 1.0}
    elif name == "Haldane":
        params = {"t": 1.0, "m": 0.0}
        if any(h["name"] == "t2" for h in hops):
            params.update({"t2": 1.0, "phi": math.pi / 2})
    else:
        params = {"t": 1.0}
    if name not in {"SSH", "Haldane"} and any(h["name"] == "t2" for h in hops):
        # Built-in presets start from a uniform unit hopping scale.  The
        # second-shell parameter remains independent and editable, but its
        # default should not silently make NNN bonds five times weaker than
        # the nearest-neighbour terms.
        params["t2"] = 1.0
    return sites, hops, params, cell


def template_document(
    name: str,
    *,
    nx: int = 4,
    ny: int = 4,
    boundary_kind: str = "semi",
    connectivity: str = "最近邻",
    shape: str = "rectangle",
) -> dict:
    """Create a validated portable model document from a built-in template."""
    if boundary_kind not in {"semi", "obc"}:
        raise ValueError("boundary_kind 必须为 semi 或 obc")
    if name in {"NP", "SC"}:
        # Keep the exact, regression-tested symbolic definitions in one place.
        from .presets import NP, SC
        from .symbolic import param

        lattice, raw_hops = (NP if name == "NP" else SC)(
            param("phi"), param("t"), param("omg")
        )
        sites = [(s.x, s.y, s.sublattice or "") for s in lattice.sites]
        hops = []
        for h in raw_hops:
            amp = str(h.amplitude)
            phase = str(h.phase)
            hops.append(_hop(
                h.from_site, h.to_site, h.cell_offset[0], h.cell_offset[1],
                name=h.name, amp=amp, phase=phase,
            ))
            hops[-1]["phase_mode"] = h.phase_mode
            hops[-1]["phase_sign"] = h.phase_sign
        if connectivity != "最近邻+次近邻":
            if connectivity == "仅格点":
                hops = []
            else:
                # Infer the first coordination shell geometrically. On-site
                # terms remain part of the model but do not count as a shell.
                distances = []
                for h in raw_hops:
                    if h.from_site == h.to_site and h.cell_offset == (0, 0):
                        continue
                    dx, dy = h.displacement(lattice)
                    distances.append(math.hypot(dx, dy))
                nearest = min(distances) if distances else 0.0
                filtered = []
                for row, h in zip(hops, raw_hops):
                    onsite = h.from_site == h.to_site and h.cell_offset == (0, 0)
                    dx, dy = h.displacement(lattice)
                    if onsite or math.hypot(dx, dy) <= nearest * 1.05 + 1e-12:
                        filtered.append(row)
                hops = filtered
        params = {"t": 1.0, "phi": math.pi / 4, "omg": 1.0}
        cell = (float(lattice.Lx), float(lattice.Ly))
    else:
        # A three-site oblique primitive cell makes the finite triangular
        # Kagome nanodisk geometrically faithful.  Keep the six-site
        # orthogonal supercell for semi-infinite ribbons, where the explicit
        # six-site vertical layout is more readable and remains backwards
        # compatible with existing semi-infinite models.
        if (name == "Kagome" and boundary_kind == "obc"
                and shape == "triangle"):
            sites, hops, params, cell = _kagome_primitive(connectivity)
        else:
            sites, hops, params, cell = _generic(name, connectivity)
    # Non-rectangular masks are evaluated in real-space cell geometry.  Keep
    # the physical proportions so a Kagome (2 × 2√3) cell produces an
    # equilateral triangle and circular/hexagonal masks stay isotropic instead
    # of being stretched by an index-only NX×NY grid.  The ratio also records
    # the finite-sample aspect for compatibility with older model files.
    shape_aspect = 1.0
    shape_vectors = None
    if shape in {"triangle", "disk", "hexagon"} and boundary_kind == "obc":
        if isinstance(cell, dict):
            a1, a2 = tuple(cell["a1"]), tuple(cell["a2"])
            sx, sy = math.hypot(*a1), math.hypot(*a2)
            shape_vectors = (a1, a2)
        else:
            sx, sy = float(cell[0]), float(cell[1])
            shape_vectors = ((sx, 0.0), (0.0, sy))
        shape_aspect = (max(int(ny) - 1, 1) * sy) / (
            max(int(nx) - 1, 1) * max(sx, 1e-12)
        )
    boundary = Boundary(
        BoundaryKind.SEMI if boundary_kind == "semi" else BoundaryKind.OBC,
        NX=int(nx), NY=int(ny), shape=shape, shape_aspect=shape_aspect,
        shape_vectors=shape_vectors,
    )
    if isinstance(cell, dict):
        vectors = (tuple(cell["a1"]), tuple(cell["a2"]))
        return model_to_dict(
            sites, hops, boundary, "cell", params, 0.0, False, True,
            cell_vectors=vectors,
        )
    return model_to_dict(sites, hops, boundary, "cell", params, 0.0, False, True, cell)
