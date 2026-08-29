"""Lattice / Indexer / count_sites 单元测试 (对拍 MATLAB count_sites)."""

import math

from hamivisualizer.model.boundary import Boundary, BoundaryKind
from hamivisualizer.model.lattice import Lattice, Site
from hamivisualizer.model.presets import NP, SC


def test_np_cell_basics():
    lattice, _ = NP(1.0)
    assert lattice.N == 4
    assert lattice.Lx == 2.0 and lattice.Ly == 2.0
    # 位置
    assert lattice.position(0, 0, 0) == (0.0, 0.0)  # r0 左下
    assert lattice.position(1, 0, 0) == (2.0, 0.0)  # 右一胞 r0
    assert lattice.position(0, 1, 2) == (1.0, 3.0)  # 上一胞 r2


def test_sc_cell_basics():
    lattice, _ = SC(1.0)
    assert lattice.N == 2
    assert lattice.Lx == 2.0 and lattice.Ly == 2.0
    assert lattice.position(0, 0, 0) == (1.0, 0.0)  # A
    assert lattice.position(0, 0, 1) == (0.0, 1.0)  # B


def test_infer_cell_size_explicit_sites():
    # 无 Lx/Ly 时自动推断 (max-min+1)
    lat = Lattice(sites=[Site(0, 0.0, 0.0), Site(1, 1.0, 0.0)])
    assert lat.Lx == 2.0
    assert lat.Ly == 1.0


def test_count_sites_semi():
    """MATLAB count_sites 半无限: NP=4·NY, SC=2·NY."""
    b = Boundary(BoundaryKind.SEMI, NY=2)
    assert NP(1.0)[0].count_sites(b) == 8
    assert SC(1.0)[0].count_sites(b) == 4
    b3 = Boundary(BoundaryKind.SEMI, NY=3)
    assert NP(1.0)[0].count_sites(b3) == 12
    assert SC(1.0)[0].count_sites(b3) == 6


def test_count_sites_obc():
    """MATLAB count_sites 双开: NP=Lx·Ly=(2NX)(2NY), SC=checkerboard 半."""
    b = Boundary(BoundaryKind.OBC, NX=2, NY=2)
    assert NP(1.0)[0].count_sites(b) == 16
    assert SC(1.0)[0].count_sites(b) == 8
    b35 = Boundary(BoundaryKind.OBC, NX=3, NY=5)
    assert NP(1.0)[0].count_sites(b35) == 60  # 4·3·5
    assert SC(1.0)[0].count_sites(b35) == 30  # 2·3·5


def test_indexer_cell_order():
    """'cell' 序: idx = (cx·NY + cy)·N + r, 与 MATLAB build_H_np 一致."""
    lat, _ = NP(1.0)
    ix = lat.indexer(order="cell", NX=2, NY=2)
    assert ix(0, 0, 0) == 0
    assert ix(0, 0, 3) == 3
    assert ix(0, 1, 0) == 4   # cy 优先于胞内 r
    assert ix(1, 0, 0) == 8   # cx 主
    assert ix(1, 1, 3) == 15
    assert len(ix.rmap) == 16
    # 互逆性
    for idx, coord in enumerate(ix.rmap):
        assert ix(*coord) == idx


def test_indexer_site_order():
    """'site' 序: 绝对坐标 x 主字典序."""
    lat, _ = NP(1.0)
    ix = lat.indexer(order="site", NX=1, NY=1)
    # r0=(0,0) < r1=(0,1) < r3=(1,0) < r2=(1,1)  (x 主)
    assert ix(0, 0, 0) == 0
    assert ix(0, 0, 1) == 1
    assert ix(0, 0, 3) == 2
    assert ix(0, 0, 2) == 3


def test_obc_shapes_provide_nonempty_cell_masks():
    for shape, expected_count in (("rectangle", 16), ("disk", 12)):
        boundary = Boundary(BoundaryKind.OBC, NX=4, NY=4, shape=shape)
        assert len(boundary.active_cells()) == expected_count
    triangle = Boundary(
        BoundaryKind.OBC, NX=4, NY=4, shape="triangle", shape_aspect=3**0.5,
    )
    cells = triangle.active_cells()
    assert 0 < len(cells) < 16
    # Row widths taper toward the apex; this rules out the historical
    # lower-left right-triangle mask while remaining robust to raster size.
    widths = [sum(cy == row for _cx, cy in cells) for row in range(4)]
    nonzero = [width for width in widths if width]
    assert nonzero == sorted(nonzero, reverse=True)


def test_triangle_mask_uses_real_cell_vectors_for_an_equilateral_kagome_disk():
    """Kagome 三角盘在物理坐标里三边等长，而非索引上的直角切片。"""
    boundary = Boundary(
        BoundaryKind.OBC, NX=6, NY=6, shape="triangle",
        shape_vectors=((2.0, 0.0), (0.0, 2.0 * math.sqrt(3.0))),
    )
    left, right, apex = boundary.triangle_outline()
    lengths = (
        math.dist(left, right), math.dist(right, apex), math.dist(apex, left),
    )
    assert all(math.isclose(length, lengths[0], rel_tol=0.0, abs_tol=1e-10)
               for length in lengths[1:])
    # 6×6 Kagome 元胞形成 6→4→2 的正三角形层数。
    assert tuple(sum(cy == row for _cx, cy in boundary.active_cells())
                 for row in range(6)) == (6, 4, 2, 0, 0, 0)


def test_disk_and_hexagon_masks_use_real_cell_metric_for_non_square_cells():
    """圆盘/六边形不应把索引网格误当成物理欧氏坐标。"""
    vectors = ((2.0, 0.0), (0.0, 4.0))
    disk = Boundary(
        BoundaryKind.OBC, NX=5, NY=5, shape="disk", shape_vectors=vectors,
    )
    hexagon = Boundary(
        BoundaryKind.OBC, NX=5, NY=5, shape="hexagon", shape_vectors=vectors,
    )
    # The bottom centre is eight physical units from the sample centre while
    # the fitted disk/hexagon radius is four; an index-only mask incorrectly
    # retained it because its normalized grid coordinate was merely -1.
    assert (2, 0) not in disk.active_cells()
    assert (2, 0) not in hexagon.active_cells()
    assert len(disk.active_cells()) > 0
    assert len(hexagon.active_cells()) > 0


def test_hexagon_outline_is_regular_and_matches_mask_metric():
    """非矩形盘的可见轮廓应与掩膜同为正六边形。"""
    boundary = Boundary(
        BoundaryKind.OBC, NX=7, NY=7, shape="hexagon",
        shape_vectors=((2.0, 0.0), (0.0, 2.0)),
    )
    outline = boundary.shape_outline()
    assert len(outline) == 6
    side_lengths = tuple(
        math.dist(outline[i], outline[(i + 1) % len(outline)])
        for i in range(len(outline))
    )
    assert all(math.isclose(length, side_lengths[0], rel_tol=0.0, abs_tol=1e-10)
               for length in side_lengths[1:])
    # The pointy corners are exactly one fitted circumradius away from the
    # centre in the orthonormal physical frame.
    center = (
        sum(point[0] for point in outline) / len(outline),
        sum(point[1] for point in outline) / len(outline),
    )
    radii = tuple(math.dist(center, point) for point in outline)
    assert all(math.isclose(radius, radii[0], rel_tol=0.0, abs_tol=1e-10)
               for radius in radii[1:])
