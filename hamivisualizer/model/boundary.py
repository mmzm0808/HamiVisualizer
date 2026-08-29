"""边界形式枚举与参数."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


SHAPE_RECTANGLE = "rectangle"
SHAPE_TRIANGLE = "triangle"
SHAPE_DISK = "disk"
SHAPE_HEXAGON = "hexagon"
BOUNDARY_SHAPES = (SHAPE_RECTANGLE, SHAPE_TRIANGLE, SHAPE_DISK, SHAPE_HEXAGON)


class BoundaryKind(Enum):
    """边界形式."""

    SEMI = "semi"  # 半无限: x 方向 Bloch 周期 (kx), y 方向有限 (NY 胞)
    OBC = "obc"    # 双开边界: x, y 都有限 (NX×NY 盘)


@dataclass(frozen=True)
class Boundary:
    """边界参数.

    SEMI: NY 有效 (y 向胞数), NX 折叠胞宽恒 1
    OBC : NX×NY 有效 (盘尺寸)，shape 可选矩形/三角/圆/六边形掩膜
    """

    kind: BoundaryKind
    NY: int = 2
    NX: int = 2
    # OBC 盘形状；半无限模式保留字段以便模型格式稳定，但始终按整条 ribbon 处理。
    shape: str = SHAPE_RECTANGLE
    # 归一化盘形在物理坐标中的纵横缩放。默认值保持旧版模型兼容；
    # 预设会根据元胞矢量自动填入，使三角盘在 Kagome 等非方形网格上仍为正三角形。
    shape_aspect: float = 1.0
    # 当前原胞的两个实空间基矢。它是由界面/模板从 cell 推导出的显示与
    # 掩膜度量，不写入模型文件；有它时所有非矩形盘均按真实欧氏距离
    # 取样，三角盘不会在斜元胞上退化成索引方格直角裁切。
    shape_vectors: tuple[tuple[float, float], tuple[float, float]] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, BoundaryKind):
            raise ValueError(f"未知边界类型: {self.kind!r}")
        for name, value in (("NX", self.NX), ("NY", self.NY)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} 必须是正整数, 得到 {value!r}")
        if self.shape not in BOUNDARY_SHAPES:
            raise ValueError(
                f"shape 必须是 {BOUNDARY_SHAPES!r} 之一, 得到 {self.shape!r}"
            )
        if (isinstance(self.shape_aspect, bool)
                or not math.isfinite(float(self.shape_aspect))
                or float(self.shape_aspect) <= 0.0):
            raise ValueError(
                f"shape_aspect 必须是正的有限数值, 得到 {self.shape_aspect!r}"
            )
        object.__setattr__(self, "shape_aspect", float(self.shape_aspect))
        if self.shape_vectors is not None:
            try:
                a1, a2 = self.shape_vectors
                a1 = (float(a1[0]), float(a1[1]))
                a2 = (float(a2[0]), float(a2[1]))
            except (TypeError, ValueError, IndexError) as exc:
                raise ValueError("shape_vectors 必须是两个二维元胞矢量") from exc
            if not all(math.isfinite(value) for value in (*a1, *a2)):
                raise ValueError("shape_vectors 必须为有限数值")
            if abs(a1[0] * a2[1] - a1[1] * a2[0]) < 1e-12:
                raise ValueError("shape_vectors 不可共线")
            object.__setattr__(self, "shape_vectors", (a1, a2))

    def triangle_outline(self, margin: float = 0.0) -> tuple[tuple[float, float], ...]:
        """返回真实坐标中的正三角形顶点（底边左、右，顶点）。

        只有双开、具有真实元胞基矢且样品至少横纵各两胞时才有非空轮廓；
        小尺寸或旧版直接构造的 Boundary 继续走兼容的归一化掩膜。
        """
        if (self.kind is not BoundaryKind.OBC
                or self.shape != SHAPE_TRIANGLE
                or self.shape_vectors is None
                or self.NX < 2 or self.NY < 2):
            return ()
        (a1x, a1y), (a2x, a2y) = self.shape_vectors
        length_a1 = math.hypot(a1x, a1y)
        length_a2 = math.hypot(a2x, a2y)
        dot = a1x * a2x + a1y * a2y
        # A 60° primitive cell (honeycomb/Kagome) has a natural triangular
        # finite sample: origins ``cx*a1 + cy*a2`` with cx+cy≤N-1 already form
        # an exact equilateral triangle.  The old bounding-frame fit instead
        # centred a smaller triangle inside the parallelogram and discarded a
        # whole diagonal of valid primitive cells.  Keep the rectangular-cell
        # fallback below for legacy 6-site Kagome ribbons and square models.
        if (length_a1 > 1e-12 and length_a2 > 1e-12
                and math.isclose(length_a1, length_a2,
                                 rel_tol=1e-9, abs_tol=1e-10)
                and math.isclose(dot / (length_a1 * length_a2), 0.5,
                                 rel_tol=0.0, abs_tol=1e-9)):
            span = min(self.NX - 1, self.NY - 1)
            if span <= 0:
                return ()
            # The regular primitive triangle has no arbitrary frame offset.
            # ``margin`` only affects cosmetic callers; membership below uses
            # the unpadded outline so exact edge cells remain deterministic.
            if margin:
                # A uniform outward offset of all three sides is equivalent
                # to extending the side by 2m and lowering the base by m.
                # This path is used solely for optional display metadata.
                try:
                    margin = max(0.0, float(margin))
                except (TypeError, ValueError):
                    margin = 0.0
                side = span * length_a1 + 2.0 * margin
                ux, uy = a1x / length_a1, a1y / length_a1
                vx, vy = -uy, ux
                if a2x * vx + a2y * vy < 0.0:
                    vx, vy = -vx, -vy
                # Centre the expanded base on the original side midpoint.
                cx = 0.5 * span * a1x
                cy = 0.5 * span * a1y
                left = (cx - side * ux / 2.0 - margin * vx,
                        cy - side * uy / 2.0 - margin * vy)
                right = (cx + side * ux / 2.0 - margin * vx,
                         cy + side * uy / 2.0 - margin * vy)
                apex = (
                    cx + (math.sqrt(3.0) * side / 2.0 - margin) * vx,
                    cy + (math.sqrt(3.0) * side / 2.0 - margin) * vy,
                )
                return left, right, apex
            return (
                (0.0, 0.0),
                (span * a1x, span * a1y),
                (span * a2x, span * a2y),
            )
        frame = self._physical_frame()
        if frame is None:
            return ()
        try:
            margin = max(0.0, float(margin))
        except (TypeError, ValueError):
            margin = 0.0
        u_min, u_max, v_min, v_max, ux, uy, vx, vy = frame
        u_min -= margin
        u_max += margin
        v_min -= margin
        v_max += margin
        side = min(u_max - u_min, 2.0 * (v_max - v_min) / math.sqrt(3.0))
        if side <= 1e-12:
            return ()
        u_center = 0.5 * (u_min + u_max)
        apex_v = v_min + math.sqrt(3.0) * side / 2.0
        return (
            (u_center - side / 2.0) * ux + v_min * vx,
            (u_center - side / 2.0) * uy + v_min * vy,
        ), (
            (u_center + side / 2.0) * ux + v_min * vx,
            (u_center + side / 2.0) * uy + v_min * vy,
        ), (
            u_center * ux + apex_v * vx,
            u_center * uy + apex_v * vy,
        )

    def shape_outline(
        self, samples: int = 96, margin: float = 0.0,
    ) -> tuple[tuple[float, float], ...]:
        """Return the physical outline of the selected finite non-rectangular mask.

        The outline is display metadata only; matrix membership continues to be
        decided by :meth:`active_cells`.  Keeping both in the same real-space
        frame prevents a non-square/oblique primitive cell from being rendered
        with a misleading index-grid silhouette.  The polygon is intentionally
        returned in physical coordinates so the scene can draw it without
        knowing anything about boundary semantics.
        """
        if (self.kind is not BoundaryKind.OBC
                or self.shape == SHAPE_RECTANGLE
                or self.shape_vectors is None):
            return ()
        frame = self._physical_frame()
        if frame is None:
            return ()
        try:
            margin = max(0.0, float(margin))
        except (TypeError, ValueError):
            margin = 0.0
        u_min, u_max, v_min, v_max, ux, uy, vx, vy = frame
        u_min -= margin
        u_max += margin
        v_min -= margin
        v_max += margin
        u_center = 0.5 * (u_min + u_max)
        v_center = 0.5 * (v_min + v_max)
        radius = 0.5 * min(u_max - u_min, v_max - v_min)
        if radius <= 1e-12:
            return ()

        def point(u: float, v: float) -> tuple[float, float]:
            return (
                u * ux + v * vx,
                u * uy + v * vy,
            )

        if self.shape == SHAPE_TRIANGLE:
            return self.triangle_outline(margin=margin)
        if self.shape == SHAPE_HEXAGON:
            # A regular flat-top hexagon with circumradius ``radius``.
            return tuple(
                point(
                    u_center + radius * math.cos(k * math.pi / 3.0),
                    v_center + radius * math.sin(k * math.pi / 3.0),
                )
                for k in range(6)
            )

        # A circle is approximated by a sufficiently fine polygon for a
        # resolution-independent cosmetic outline.  Clamp the sample count so
        # malformed callers cannot create an unbounded scene item list.
        count = max(24, min(256, int(samples)))
        return tuple(
            point(
                u_center + radius * math.cos(2.0 * math.pi * k / count),
                v_center + radius * math.sin(2.0 * math.pi * k / count),
            )
            for k in range(count)
        )

    def _physical_frame(self):
        """Return an orthonormal frame covering the finite cell origins.

        The index grid is generally not a Euclidean grid: a primitive cell
        may be rectangular with unequal lengths or oblique.  Non-rectangular
        masks therefore use this frame to measure distances in real space.
        The tuple contains projected bounds followed by the two unit axes.
        ``None`` keeps the historical normalized-index fallback for callers
        that construct a Boundary without cell vectors.
        """
        if self.shape_vectors is None:
            return None
        (a1x, a1y), (a2x, a2y) = self.shape_vectors
        length_a1 = math.hypot(a1x, a1y)
        if length_a1 <= 1e-12:
            return None
        ux, uy = a1x / length_a1, a1y / length_a1
        # Choose the perpendicular pointing toward increasing cell-y so the
        # mask has a stable orientation for both oblique-vector signs.
        vx, vy = -uy, ux
        if a2x * vx + a2y * vy < 0.0:
            vx, vy = -vx, -vy
        corners = tuple(
            (cx * a1x + cy * a2x, cx * a1y + cy * a2y)
            for cx in (0, self.NX - 1)
            for cy in (0, self.NY - 1)
        )
        u_values = tuple(x * ux + y * uy for x, y in corners)
        v_values = tuple(x * vx + y * vy for x, y in corners)
        u_min, u_max = min(u_values), max(u_values)
        v_min, v_max = min(v_values), max(v_values)
        if (u_max - u_min) <= 1e-12 or (v_max - v_min) <= 1e-12:
            return None
        return u_min, u_max, v_min, v_max, ux, uy, vx, vy

    @staticmethod
    def _normalized_physical_point(
        cx: int, cy: int, frame, vectors,
    ) -> tuple[float, float] | None:
        """Map a cell origin to an isotropic real-space mask coordinate."""
        if frame is None or vectors is None:
            return None
        u_min, u_max, v_min, v_max, ux, uy, vx, vy = frame
        (a1x, a1y), (a2x, a2y) = vectors
        px = cx * a1x + cy * a2x
        py = cx * a1y + cy * a2y
        u = px * ux + py * uy
        v = px * vx + py * vy
        u_center = 0.5 * (u_min + u_max)
        v_center = 0.5 * (v_min + v_max)
        radius = 0.5 * min(u_max - u_min, v_max - v_min)
        if radius <= 1e-12:
            return None
        return (u - u_center) / radius, (v - v_center) / radius

    def active_cells(self) -> tuple[tuple[int, int], ...]:
        """返回当前边界实际包含的元胞坐标。

        SEMI 的 x 方向由 Bloch 折叠，始终只有一个显示/计算元胞列；
        OBC 根据 ``shape`` 对 NX×NY 网格做掩膜。掩膜只影响有限矩阵和
        晶格显示，跃迁定义仍保持原有 cell_offset 语义。
        """
        if self.kind is BoundaryKind.SEMI:
            return tuple((0, cy) for cy in range(self.NY))
        if self.shape == SHAPE_RECTANGLE:
            return tuple(
                (cx, cy) for cx in range(self.NX) for cy in range(self.NY)
            )

        # 归一化到 [-1, 1] 的网格中心，NX/NY=1 时取中心点，避免除零。
        def norm(index: int, size: int) -> float:
            return 0.0 if size <= 1 else (2.0 * index - (size - 1)) / (size - 1)

        triangle = self.triangle_outline()
        physical_frame = self._physical_frame()
        cells: list[tuple[int, int]] = []
        for cx in range(self.NX):
            for cy in range(self.NY):
                x, y = norm(cx, self.NX), norm(cy, self.NY)
                if self.shape == SHAPE_TRIANGLE:
                    if triangle and self.shape_vectors is not None:
                        # Resolve the membership in physical coordinates.
                        # The three sides are an exact equilateral triangle;
                        # points on its boundary are retained deterministically.
                        (left, right, apex) = triangle
                        (a1x, a1y), (a2x, a2y) = self.shape_vectors
                        px = cx * a1x + cy * a2x
                        py = cx * a1y + cy * a2y

                        def cross(origin, target) -> float:
                            return ((target[0] - origin[0]) * (py - origin[1])
                                    - (target[1] - origin[1]) * (px - origin[0]))

                        c1 = cross(left, right)
                        c2 = cross(right, apex)
                        c3 = cross(apex, left)
                        keep = ((c1 >= -1e-9 and c2 >= -1e-9 and c3 >= -1e-9)
                                or (c1 <= 1e-9 and c2 <= 1e-9 and c3 <= 1e-9))
                        if keep:
                            cells.append((cx, cy))
                        continue
                    # 以物理坐标构造等边三角形，再映射回归一化网格。
                    # ``shape_aspect`` 是 y/x 的物理缩放比，因此无论
                    # 元胞是方形还是 Kagome 的 2×2√3 矩形，三条边长度一致。
                    # 三角形居中并尽可能填满 [-1, 1] 视图；离散网格只会
                    # 产生必要的台阶，不再出现旧版的直角三角盘。
                    aspect = float(self.shape_aspect)
                    height = min(2.0, math.sqrt(3.0) / aspect)
                    half_width = height * aspect / math.sqrt(3.0)
                    base_y = -0.5 * height
                    apex_y = 0.5 * height
                    # Cell centres are a coarse raster of the continuous
                    # boundary.  A half-cell anti-aliasing margin prevents a
                    # 4×4/6×6 sample from collapsing to one skinny row while
                    # keeping the silhouette and matrix mask consistent.
                    grid_step = max(
                        2.0 / max(self.NX - 1, 1),
                        2.0 / max(self.NY - 1, 1),
                    )
                    margin = 0.55 * grid_step
                    if y < base_y - margin or y > apex_y + margin:
                        keep = False
                    else:
                        # 等边三角形在高度方向的线性半宽。
                        sample_y = min(apex_y, max(base_y, y))
                        fraction = (sample_y - base_y) / max(height, 1e-12)
                        allowed = half_width * (1.0 - fraction) + margin
                        keep = abs(x) <= allowed + 1e-12
                elif self.shape == SHAPE_DISK:
                    if physical_frame and self.shape_vectors is not None:
                        point = self._normalized_physical_point(
                            cx, cy, physical_frame, self.shape_vectors,
                        )
                        if point is None:
                            keep = False
                        else:
                            px, py = point
                            # Inflate only by a fraction of one physical grid
                            # step so coarse samples do not lose the boundary
                            # ring; unlike the old index mask this remains a
                            # circle after changing cell lengths or angles.
                            margin = 0.25 * max(
                                2.0 / max(self.NX - 1, 1),
                                2.0 / max(self.NY - 1, 1),
                            )
                            keep = px * px + py * py <= (1.0 + margin) ** 2
                    else:
                        # A slight discrete-grid inflation keeps a 4×4 disk
                        # from collapsing to only its four central cells while
                        # still excluding the four square corners.
                        keep = x * x + y * y <= 1.2 + 1e-12
                else:  # SHAPE_HEXAGON：正六边形的实空间网格近似掩膜
                    if physical_frame and self.shape_vectors is not None:
                        point = self._normalized_physical_point(
                            cx, cy, physical_frame, self.shape_vectors,
                        )
                        if point is None:
                            keep = False
                        else:
                            px, py = point
                            margin = 0.25 * max(
                                2.0 / max(self.NX - 1, 1),
                                2.0 / max(self.NY - 1, 1),
                            )
                            # Strict regular flat-top hexagon with unit
                            # circumradius.  The previous ``max(abs(y),
                            # abs(x)+0.5*abs(y))`` admitted points such as
                            # (0, 1), producing a non-regular, vertically
                            # stretched mask.  These two half-plane tests are
                            # the exact convex polygon whose vertices are at
                            # angles 0, 60, ..., 300 degrees.
                            sqrt3 = math.sqrt(3.0)
                            keep = (
                                abs(py) <= sqrt3 / 2.0 + margin
                                and sqrt3 * abs(px) + abs(py)
                                <= sqrt3 + 2.0 * margin
                            )
                    else:
                        # Compatibility path for old callers that construct a
                        # Boundary without vectors.  Apply the persisted
                        # aspect to the normalized y coordinate and use the
                        # same regular flat-top metric as the vector path.
                        scaled_y = y * float(self.shape_aspect)
                        sqrt3 = math.sqrt(3.0)
                        keep = (
                            abs(scaled_y) <= sqrt3 / 2.0 + 1e-12
                            and sqrt3 * abs(x) + abs(scaled_y)
                            <= sqrt3 + 1e-12
                        )
                if keep:
                    cells.append((cx, cy))
        # 极小尺寸的离散网格不应出现空盘；至少保留中心最近的一个元胞。
        if not cells:
            cells.append(((self.NX - 1) // 2, (self.NY - 1) // 2))
        return tuple(cells)
