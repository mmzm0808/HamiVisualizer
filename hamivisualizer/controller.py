"""ViewController: 面板 → 模型重建 → 视图刷新的信号槽接线.

数据流:
  面板编辑 → changed(防抖) → 解析 Lattice+HoppingTerm+Boundary + ParameterSet
  → HamiltonianBuilder → HResult → 四视图 DTO → set_data。

关键设计 (v0.3):
  - 参数面板: 自动从跃迁表达式的自由符号生成 (t/φ/ω/任意自定义名), 数值可编辑;
    DEFAULT_PARAMS 仅作默认值。修复了「符号模式在 GUI 下失效」与「无参数控件」两处回归。
  - 签名缓存: 只动 kx 时走轻量快路径 (矩阵数值 + 能带标记线), 能带/晶格不重算。
  - 视口保留: 场景矩形不变时不再 fitInView, 用户缩放/平移不被重置。
  - 高保真 MATLAB 观感: 物理判色 (t/φ 传入), 晶格首胞高亮 + 虚影键 (见 scene builder)。
"""

from __future__ import annotations

import math

import numpy as np
import sympy as sp
from PySide6.QtCore import (
    QMarginsF,
    QObject,
    QPoint,
    QRect,
    QRunnable,
    QSizeF,
    QThreadPool,
    Qt,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import QPageLayout, QPageSize, QPainter, QPdfWriter, QRegion
from PySide6.QtWidgets import QAbstractItemView, QDialog, QFileDialog, QMessageBox, QWidget

from .model.boundary import Boundary, BoundaryKind
from .model.expression import evaluate_expression, parse_expression
from .model.hamiltonian import HamiltonianBuilder, wavefunctions
from .model.hopping import HoppingTerm
from .model.lattice import Lattice, Site
from .model.persistence import (
    hop_dict_to_row,
    load_model,
    model_to_dict,
    save_model,
)
from .model.workspace import app_data_dir
from .model.presets import NP, SC
from .model.ribbon import RibbonHamiltonian, fold_x
from .model.symbolic import (
    ElementFormatter,
    collect_param_names,
    param,
)
from .view.rendermodel import (
    BandSceneData,
    LatticeSceneData,
    MatrixSceneData,
    WfSceneData,
)
from .view.dialogs import HoppingDialog

# 数值模式下的默认参数值 (参数面板可覆盖)
DEFAULT_PARAMS = {"t": 1.0, "phi": np.pi / 4, "omg": 1.0, "tc": 1.0}
SPECTRAL_ASYNC_THRESHOLD = 80
# A long-range Bloch term (for example dx=250) belongs in the matrix, but
# expanding hundreds of physical ghost columns is neither useful nor safe for
# an interactive canvas.  The display keeps a small neighbourhood around the
# central ribbon; the Hamiltonian builder still retains every requested
# harmonic exactly.
MAX_VISIBLE_GHOST_LAYERS = 4


def _finite_shape_outline_margin(lattice, boundary: Boundary) -> float:
    """Return a restrained cosmetic margin for finite-shape outlines.

    The mask is defined on cell origins, whereas the visible sites live inside
    each primitive cell.  The old implementation used the largest *absolute
    site radius* as a scalar margin.  That is overly conservative for an
    anisotropic basis such as Kagome: a 3.46-unit diagonal site radius inflated
    a 10-unit triangle into a huge frame around a much smaller sample.  A
    half-cell diagonal is a stable upper bound for the visual overhang while
    keeping the outline close to the actual finite sample.  The outline remains
    mathematically equilateral; this value only controls its cosmetic padding.
    """
    if getattr(boundary, "shape", "rectangle") == "rectangle":
        return 0.0
    vectors = (getattr(lattice, "a1", (0.0, 0.0)),
               getattr(lattice, "a2", (0.0, 0.0)))
    cell_scale = max((math.hypot(float(x), float(y)) for x, y in vectors),
                     default=0.0)
    if cell_scale <= 1e-12:
        return 0.0
    # Keep enough clearance for a basis site near a cell corner, but avoid
    # multiplying the triangle by the full coordinate radius of that basis.
    site_scale = max(
        (math.hypot(float(site.x), float(site.y))
         for site in getattr(lattice, "sites", ())),
        default=0.0,
    )
    return min(site_scale, 0.5 * cell_scale) if site_scale > 0.0 else 0.0


def _finite_shape_outline(lattice, boundary: Boundary, positions) -> tuple:
    """Build the display outline without letting a basis make a false giant frame.

    ``Boundary`` deliberately owns the cell-level mask.  The canvas, however,
    renders basis sites (which may overhang a cell-origin triangle), so the
    most faithful cosmetic outline is the smallest horizontal equilateral
    triangle containing the actually rendered sites.  This keeps a discrete
    Kagome 6→4→2 sample visually recognisable as a triangular nanodisk while
    retaining the exact cell-level membership used by the Hamiltonian.
    """
    if (getattr(boundary, "shape", "rectangle") == "triangle"
            and boundary.kind is BoundaryKind.OBC and positions):
        points = tuple((float(x), float(y)) for x, y in positions)
        y_min = min(y for _x, y in points)
        # A tiny clearance keeps an exact boundary site from touching the
        # stroke after antialiasing, without recreating the old oversized pad.
        span = max(
            max(x for x, _y in points) - min(x for x, _y in points),
            max(y for _x, y in points) - y_min,
            1.0,
        )
        base_y = y_min - 0.02 * span

        def interval(side: float) -> tuple[float, float]:
            lows, highs = [], []
            for x, y in points:
                half_width = side / 2.0 - (y - base_y) / math.sqrt(3.0)
                lows.append(x - half_width)
                highs.append(x + half_width)
            return max(lows), min(highs)

        # A side equal to the horizontal span is a lower bound.  Increase it
        # until the apex and all upper rows fit, then binary-search the tight
        # feasible side.  The interval is convex, so this is deterministic.
        lower = max(
            max(x for x, _y in points) - min(x for x, _y in points),
            2.0 * (max(y for _x, y in points) - base_y) / math.sqrt(3.0),
            1e-9,
        )
        upper = lower
        for _ in range(32):
            lo, hi = interval(upper)
            if lo <= hi + 1e-10:
                break
            upper *= 1.25
        else:
            return tuple(boundary.shape_outline(
                margin=_finite_shape_outline_margin(lattice, boundary),
            ))
        for _ in range(64):
            mid = 0.5 * (lower + upper)
            lo, hi = interval(mid)
            if lo <= hi + 1e-10:
                upper = mid
            else:
                lower = mid
        lo, hi = interval(upper)
        center = 0.5 * (lo + hi)
        apex_y = base_y + math.sqrt(3.0) * upper / 2.0
        return (
            (center - upper / 2.0, base_y),
            (center + upper / 2.0, base_y),
            (center, apex_y),
        )

    # Disk/hexagon masks are selected from cell origins, while the canvas
    # renders every basis site inside those cells.  A multi-site basis can
    # therefore overhang the origin-level boundary by a full bond length (the
    # most visible example is the six-site Kagome disk).  Reusing
    # ``Boundary.shape_outline`` here produced an outline that cut through
    # real sites.  Fit the cosmetic outline to the actual rendered positions
    # in the same orthonormal physical frame used by the mask.  The matrix
    # membership is deliberately unchanged; this only makes the visible
    # boundary honest about what is drawn.
    shape = getattr(boundary, "shape", "rectangle")
    if (shape in {"disk", "hexagon"}
            and boundary.kind is BoundaryKind.OBC and positions):
        frame_builder = getattr(boundary, "_physical_frame", None)
        frame = frame_builder() if callable(frame_builder) else None
        if frame is not None:
            _u_min, _u_max, _v_min, _v_max, ux, uy, vx, vy = frame
            projected = tuple(
                (float(x) * ux + float(y) * uy,
                 float(x) * vx + float(y) * vy)
                for x, y in positions
            )
            u_center = 0.5 * (
                min(u for u, _v in projected) + max(u for u, _v in projected)
            )
            v_center = 0.5 * (
                min(v for _u, v in projected) + max(v for _u, v in projected)
            )
            span = max(
                max(u for u, _v in projected) - min(u for u, _v in projected),
                max(v for _u, v in projected) - min(v for _u, v in projected),
                1.0,
            )
            # Keep a small clearance for the node stroke without bringing
            # back the historical giant padding around anisotropic bases.
            padding = max(0.04, 0.018 * span)
            if shape == "disk":
                radius = max(
                    math.hypot(u - u_center, v - v_center)
                    for u, v in projected
                ) + padding
                count = 96
                return tuple(
                    (
                        (u_center + radius * math.cos(2.0 * math.pi * k / count)) * ux
                        + (v_center + radius * math.sin(2.0 * math.pi * k / count)) * vx,
                        (u_center + radius * math.cos(2.0 * math.pi * k / count)) * uy
                        + (v_center + radius * math.sin(2.0 * math.pi * k / count)) * vy,
                    )
                    for k in range(count)
                )

            # Regular flat-top hexagon.  For a circumradius R in this frame,
            # the exact half-plane metric is
            #   R >= max(2|v|/sqrt(3), |u| + |v|/sqrt(3)).
            sqrt3 = math.sqrt(3.0)
            radius = max(
                max(2.0 * abs(v - v_center) / sqrt3,
                    abs(u - u_center) + abs(v - v_center) / sqrt3)
                for u, v in projected
            ) + padding
            return tuple(
                (
                    (u_center + radius * math.cos(k * math.pi / 3.0)) * ux
                    + (v_center + radius * math.sin(k * math.pi / 3.0)) * vx,
                    (u_center + radius * math.cos(k * math.pi / 3.0)) * uy
                    + (v_center + radius * math.sin(k * math.pi / 3.0)) * vy,
                )
                for k in range(6)
            )
    return tuple(boundary.shape_outline(
        margin=_finite_shape_outline_margin(lattice, boundary),
    ))


class _SpectralSignals(QObject):
    finished = Signal(int, str, object)
    failed = Signal(int, str, str)


class _SpectralWorker(QRunnable):
    """Pure numerical work; never creates or mutates Qt widgets."""

    def __init__(self, generation: int, kind: str, payload, signal_parent=None):
        super().__init__()
        self.generation = generation
        self.kind = kind
        self.payload = payload
        # The QRunnable is auto-deleted by QThreadPool after ``run``. Parent
        # the signal carrier to the controller so a queued finished/failed
        # delivery cannot target a C++ QObject that has already been deleted.
        self.signals = _SpectralSignals(signal_parent)

    def run(self):
        try:
            if self.kind == "band":
                H0, H1, extra, kxs = self.payload
                result = RibbonHamiltonian(H0, H1, extra).bands(kxs)
            else:
                H, positions = self.payload
                result = wavefunctions(H, positions=positions)
        except Exception as exc:  # delivered to the GUI thread as text
            self.signals.failed.emit(self.generation, self.kind, str(exc))
        else:
            self.signals.finished.emit(self.generation, self.kind, result)


class ViewController(QObject):
    """主控制器."""

    def __init__(self, window, *, connect_actions: bool = True):
        super().__init__(window)
        self.window = window
        # A controller is the authoritative owner of a window's calculation
        # pipeline.  Register it here instead of relying on every launcher,
        # test harness or hidden preview window to remember the assignment.
        # Edit-session baseline capture (restore and magnetic home snapping)
        # depends on this relationship being present.
        self.window.controller = self
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(300)
        self._debounce.timeout.connect(self.rebuild)

        self._sig: tuple | None = None
        self._state: tuple | None = None   # (res, lattice, boundary, params, symbolic)
        self._display_hops: list[HoppingTerm] | None = None
        self._matrix_obj = None            # 符号/数值原始矩阵 (kx 快路径复用)
        self._mode = "smart"
        self._formatter = None
        self._lam: tuple | None = None     # lambdify 缓存: (names, matrix, func)
        self._fit_seen: dict = {}
        self._generation = 0
        self._spectral_context: dict[int, tuple] = {}
        # Keep QRunnable signal owners alive until their callback arrives.
        # Without this strong reference, a large band calculation could emit
        # into a deleted _SpectralSignals object while the app was closing.
        self._spectral_workers: dict[int, _SpectralWorker] = {}
        self._thread_pool = QThreadPool.globalInstance()
        self._calculation_mode = "automatic"

        self.window.panel.changed.connect(self._on_input_changed)
        self.window.panel.displayChanged.connect(self._on_display_changed)
        self.window.panel.recalculateRequested.connect(self.rebuild)
        self.window.panel.cancelRequested.connect(self.cancel_calculation)
        if connect_actions:
            self.window.action_np.triggered.connect(lambda: self.load_preset("NP"))
            self.window.action_sc.triggered.connect(lambda: self.load_preset("SC"))
            self.window.action_open.triggered.connect(self.open_model)
            self.window.action_save.triggered.connect(self.save_model)
            self.window.action_export.triggered.connect(self.export_png)
            self.window.action_export_svg.triggered.connect(self.export_svg)
            self.window.action_export_pdf.triggered.connect(self.export_pdf)
        # Matrix inspection is owned by MainWindow: it is intentionally
        # non-modal and highlights the selected cell in both matrix views.
        # Keeping a second controller-side QMessageBox connection here made
        # every click enter a modal loop and defeated lightweight inspection.
        # 能量只用于 OBC 波函数页的本征态选择，不触发哈密顿量重建。
        self.window.panel.energyChanged.connect(self._on_energy_changed)
        lattice_scene = self.window.lattice_scene
        lattice_scene.siteMoved.connect(self.window.panel.update_site_position)
        lattice_scene.siteAddRequested.connect(self.window.panel.append_site)
        lattice_scene.siteDeleteRequested.connect(self.window.panel.remove_site)
        lattice_scene.hoppingRequested.connect(self._create_hopping)
        lattice_scene.hoppingRequestedWithOffset.connect(
            self._create_hopping_with_offset
        )
        lattice_scene.hoppingStrengthEdited.connect(
            self._on_hopping_strength_edited
        )
        lattice_scene.editSelectionChanged.connect(
            lambda text: self.window.statusBar().showMessage(text or "晶格编辑模式")
        )

    def set_runtime_preferences(self, *, debounce_ms: int = 300,
                                calculation_mode: str = "automatic",
                                snap_step: float = 0.25,
                                snap_enabled: bool = True):
        self._debounce.setInterval(max(100, int(debounce_ms)))
        self._calculation_mode = (
            calculation_mode if calculation_mode in {"automatic", "manual"}
            else "automatic"
        )
        self.window.lattice_scene.snap_step = max(0.001, float(snap_step))
        self.window.lattice_scene.set_snap_enabled(snap_enabled)

    def cancel_calculation(self):
        self._debounce.stop()
        self._generation += 1
        self._spectral_context.clear()
        self.window.panel.set_calculating(False)
        self.window.set_result_state("stale", "计算已取消，当前结果可能不是最新。")
        self.window.statusBar().showMessage("已取消当前计算")

    # ---- 预设 ----

    def load_preset(self, name: str):
        """加载内置预设 (NP/SC)。

        以**符号表达式**写入表格 ("-t" / "phi" / "omg"), 参数面板给数值默认值 ——
        于是符号模式显示 t·e^{iφ}, 数值模式用面板值计算, 两条路径共用同一份表格。
        """
        factory = NP if name == "NP" else SC
        lattice, hops = factory(param("phi"), param("t"), param("omg"))
        self.window.panel.set_lattice_rows(
            [(s.x, s.y, s.sublattice or "") for s in lattice.sites]
        )
        self.window.panel.set_cell_size((lattice.Lx, lattice.Ly))
        rows = []
        for h in hops:
            rows.append([
                h.name, h.from_site, h.to_site,
                h.cell_offset[0], h.cell_offset[1],
                self._expr_str(h.amplitude), h.phase_mode,
                self._expr_str(h.phase), h.phase_sign,
            ])
        self.window.panel.set_hop_rows(rows)
        self.window.panel.set_params(
            {"t": DEFAULT_PARAMS["t"], "phi": DEFAULT_PARAMS["phi"], "omg": DEFAULT_PARAMS["omg"]},
            force=True,
        )
        self.window.statusBar().showMessage(f"已加载 {name} 预设 (符号表达式已填入跃迁表)")
        self.rebuild()
        self.window.set_dirty(False)

    @staticmethod
    def _expr_str(v) -> str:
        if isinstance(v, sp.Basic):
            return str(v)
        if isinstance(v, complex):
            return f"{v.real:g}{v.imag:+g}j"
        return f"{v:g}"

    # ---- 表达式解析 ----

    @staticmethod
    def _eval_sym(s: str):
        """字符串 → 符号表达式; 任意自由符号一律设为 real=True (无 conj)."""
        return parse_expression(s)

    @staticmethod
    def _eval_num(s: str, params: dict):
        """字符串 → 复数数值; 未知符号给出明确报错 (不再静默回退)."""
        return evaluate_expression(s, params)

    # ---- 解析 ----

    def _build_lattice(self, site_rows) -> Lattice:
        cell = self.window.panel.get_cell_size()
        vectors = self.window.panel.get_cell_vectors()
        return Lattice(
            [Site(i, x, y, sub) for i, (x, y, sub) in enumerate(site_rows)],
            Lx=cell[0] if cell is not None else None,
            Ly=cell[1] if cell is not None else None,
            a1=vectors[0] if vectors is not None else None,
            a2=vectors[1] if vectors is not None else None,
        )

    def _build_hops(self, symbolic: bool, params: dict) -> list[HoppingTerm]:
        hops = []
        for h in self.window.panel.get_hop_rows():
            pm = h["phase_mode"]
            if pm == "directional":
                raise ValueError("phase_mode='directional' 尚未实现, 请改用 'none' 或 'phase'")
            if symbolic:
                amp = self._eval_sym(h["amplitude"])
                ph = self._eval_sym(h["phase"]) if pm == "phase" else 0
            else:
                amp = self._eval_num(h["amplitude"], params)
                ph = self._eval_num(h["phase"], params) if pm == "phase" else 0.0
            hops.append(HoppingTerm(
                name=h["name"],
                from_site=h["from_site"],
                to_site=h["to_site"],
                cell_offset=(h["off_x"], h["off_y"]),
                amplitude=amp,
                phase_mode=pm,
                phase=ph,
                phase_sign=h["phase_sign"],
            ))
        return hops

    @staticmethod
    def _same_number(a, b, tol: float = 1e-7) -> bool:
        """Compare editable numeric fields at the panel's display precision.

        Cell lengths are stored in spin boxes with eight decimal places, so
        irrational preset values such as ``2√3`` are rounded by the UI.  A
        strict machine epsilon comparison would incorrectly classify an
        untouched preset as custom and disable the Kagome basis migration.
        """
        try:
            return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tol)
        except (TypeError, ValueError):
            return False

    def _document_matches_kagome_builtin(
        self, document: dict, site_rows: list[tuple], hop_rows: list[dict],
    ) -> bool:
        """Check whether the current editable data is an untouched Kagome preset.

        Shape switching is allowed to migrate the representation only when it
        can prove that the user is still on one of the two built-in Kagome
        layouts.  This guard is deliberately strict: a hand-edited site,
        hopping amplitude, cell vector or phase leaves the user's model alone.
        """
        expected_sites = tuple(
            (float(site["x"]), float(site["y"]), str(site.get("sublattice") or ""))
            for site in document.get("sites", ())
        )
        actual_sites = tuple(
            (float(x), float(y), str(sub or "")) for x, y, sub in site_rows
        )
        if len(actual_sites) != len(expected_sites):
            return False
        if any(
            not (
                self._same_number(actual[0], expected[0])
                and self._same_number(actual[1], expected[1])
                and actual[2] == expected[2]
            )
            for actual, expected in zip(actual_sites, expected_sites)
        ):
            return False

        expected_cell = document.get("cell")
        if expected_cell is None:
            if self.window.panel.get_cell_size() is not None:
                return False
        elif "a1" in expected_cell:
            actual_vectors = self.window.panel.get_cell_vectors()
            if actual_vectors is None:
                return False
            for actual, expected in zip(actual_vectors, (expected_cell["a1"], expected_cell["a2"])):
                if any(not self._same_number(a, b) for a, b in zip(actual, expected)):
                    return False
        else:
            actual_cell = self.window.panel.get_cell_size()
            if actual_cell is None or any(
                not self._same_number(a, expected_cell[key])
                for a, key in zip(actual_cell, ("Lx", "Ly"))
            ):
                return False

        expected_hops = tuple(
            (
                str(hop.get("name", "t")), int(hop.get("from_site", 0)),
                int(hop.get("to_site", 0)),
                int(hop.get("cell_offset", [0, 0])[0]),
                int(hop.get("cell_offset", [0, 0])[1]),
                str(hop.get("amplitude", "1.0")),
                str(hop.get("phase_mode", "none")),
                str(hop.get("phase", "0")), int(hop.get("phase_sign", 1)),
            )
            for hop in document.get("hops", ())
        )
        actual_hops = tuple(
            (
                str(hop.get("name", "t")), int(hop.get("from_site", 0)),
                int(hop.get("to_site", 0)), int(hop.get("off_x", 0)),
                int(hop.get("off_y", 0)), str(hop.get("amplitude", "1.0")),
                str(hop.get("phase_mode", "none")), str(hop.get("phase", "0")),
                int(hop.get("phase_sign", 1)),
            )
            for hop in hop_rows
        )
        return actual_hops == expected_hops

    def _migrate_builtin_kagome_shape(self) -> str | None:
        """Switch untouched Kagome OBC presets to their faithful shape basis.

        The finite triangular nanodisk uses the three-site oblique primitive
        cell; the square/rectangular sample keeps the six-site orthogonal
        supercell.  Migration is intentionally opt-in through the shape
        selector and strictly limited to exact built-in data, so user-authored
        six-site or three-site models remain untouched.
        """
        panel = self.window.panel
        if panel.is_semi():
            return None
        target_shape = panel.get_shape()
        if target_shape not in {"triangle", "rectangle"}:
            return None
        source_shape = "rectangle" if target_shape == "triangle" else "triangle"
        nx, ny = panel.get_dim()
        site_rows = panel.get_site_rows()
        hop_rows = panel.get_hop_rows()
        from .model.templates import template_document

        match = None
        connectivity = None
        for candidate in ("仅格点", "最近邻", "最近邻+次近邻"):
            source = template_document(
                "Kagome", nx=nx, ny=ny, boundary_kind="obc",
                connectivity=candidate, shape=source_shape,
            )
            if self._document_matches_kagome_builtin(source, site_rows, hop_rows):
                match = template_document(
                    "Kagome", nx=nx, ny=ny, boundary_kind="obc",
                    connectivity=candidate, shape=target_shape,
                )
                connectivity = candidate
                break
        if match is None:
            return None

        blocked = panel.blockSignals(True)
        try:
            panel.set_lattice_rows([
                (site["x"], site["y"], site.get("sublattice") or "")
                for site in match["sites"]
            ])
            cell = match.get("cell")
            if cell is not None and "a1" in cell:
                panel.set_cell_vectors((tuple(cell["a1"]), tuple(cell["a2"])))
            elif cell is not None:
                panel.set_cell_size((cell["Lx"], cell["Ly"]))
            panel.set_hop_rows([hop_dict_to_row(hop) for hop in match["hops"]])
            panel.set_shape(target_shape)
            params = panel.get_params()
            for name, value in match.get("params", {}).items():
                params.setdefault(name, value)
            panel.set_params(params, force=True)
        finally:
            panel.blockSignals(blocked)

        if target_shape == "triangle":
            return (
                f"Kagome 已切换为三角纳米盘的 3 格点斜原胞（{connectivity}）；"
                "三条边保持平直正三角形"
            )
        return (
            f"Kagome 已切换为方形盘的 6 格点正交超胞（{connectivity}）"
        )

    def _build_boundary(self) -> Boundary:
        nx, ny = self.window.panel.get_dim()
        if self.window.panel.is_semi():
            return Boundary(
                BoundaryKind.SEMI, NX=nx, NY=ny,
                shape="rectangle",
            )
        shape = self.window.panel.get_shape()
        # Shape masks are defined in a normalized NX×NY grid.  Feed the
        # physical y/x scale into the boundary so an equilateral triangle
        # remains equilateral for rectangular/oblique primitive cells.
        aspect = 1.0
        if shape in {"triangle", "disk", "hexagon"}:
            vectors = self.window.panel.get_cell_vectors()
            if vectors is not None:
                sx = math.hypot(*vectors[0])
                sy = math.hypot(*vectors[1])
                shape_vectors = (tuple(vectors[0]), tuple(vectors[1]))
            else:
                cell = self.window.panel.get_cell_size()
                sx, sy = (cell if cell is not None else (1.0, 1.0))
                shape_vectors = ((float(sx), 0.0), (0.0, float(sy)))
            aspect = (max(ny - 1, 1) * float(sy)) / (
                max(nx - 1, 1) * max(float(sx), 1e-12)
            )
        else:
            shape_vectors = None
        return Boundary(
            BoundaryKind.OBC, NX=nx, NY=ny,
            shape=shape, shape_aspect=aspect, shape_vectors=shape_vectors,
        )

    # ---- 重建 ----

    def _on_input_changed(self):
        self.window.set_dirty(True)
        # 输入提示放到底部状态栏短暂红色高亮，避免顶部横幅遮挡绘图区。
        self.window.set_result_state(
            "stale", "输入已更改，正在等待重新计算…", show_banner=False
        )
        self.window.flash_status("输入已更改，正在等待重新计算…")
        if self._calculation_mode == "automatic":
            self._debounce.start()
        else:
            self._debounce.stop()
            self.window.statusBar().showMessage("输入已更改；点击“立即计算”更新结果")

    def _on_hopping_strength_edited(self, row: int, strength: float) -> None:
        """Apply a canvas coefficient and retain it through UI-only redraws."""
        panel = self.window.panel
        panel.set_hopping_strength(int(row), float(strength))
        # ``set_hopping_strength`` reports validation failures through the
        # panel error label and returns without changing the document.  Only
        # accepted values may update the transient canvas edit context;
        # otherwise a theme switch could display a value the model rejected.
        if panel.error_label.text():
            return
        self.window.lattice_scene.update_hop_strength(int(row), float(strength))
        # A canvas coefficient commit is an explicit, completed edit rather
        # than a stream of keystrokes.  The panel's generic ``changed`` signal
        # normally starts the debounce timer, which leaves a short interval
        # where Ctrl+Z has no snapshot yet.  Record the committed document now
        # so undo is immediately available, but keep the scheduled rebuild:
        # rebuilding synchronously would destroy the focused proxy widget and
        # make the editor appear to jump away under the user's cursor.  The
        # later rebuild compares equal and therefore does not create a second
        # history entry.
        callback = getattr(self.window, "document_committed", None)
        if callback is not None:
            callback(self._current_document())

    def _on_display_changed(self, reason: str):
        """Apply presentation-only options without rebuilding the Hamiltonian."""
        if self._state is None or self._display_hops is None:
            self._on_input_changed()
            return
        res, lattice, boundary, params, symbolic = self._state
        if reason == "smart":
            self._push_matrix(
                res, boundary, params, symbolic, self.window.panel.get_kx()
            )
            detail = "矩阵标签"
        elif reason == "labels":
            self._push_lattice(lattice, self._display_hops, boundary, res)
            detail = "元胞编号方向"
        elif reason == "lattice_style":
            self.window.lattice_scene.set_display_options(
                **self.window.panel.lattice_display_options()
            )
            self._fit(self.window.lattice_gv)
            self._fit(self.window.combined_lattice_gv)
            detail = "晶格显示选项"
        else:
            self._on_input_changed()
            return
        self.window.set_dirty(True)
        if self._generation not in self._spectral_context:
            self.window.set_result_state("ready")
        self.window.statusBar().showMessage(
            f"{detail}已即时更新（未重新计算哈密顿量）"
        )
        callback = getattr(self.window, "document_committed", None)
        if callback is not None:
            callback(self._current_document())

    def rebuild(self):
        self._debounce.stop()
        self.window.panel.set_calculating(True)
        try:
            self._do_rebuild()
        except Exception as e:  # noqa: BLE001
            msg = str(e) or type(e).__name__
            # Any in-flight spectral result belongs to an input revision that
            # failed validation and must never make the UI look current again.
            self._generation += 1
            self._sig = None
            self.window.panel.set_error(msg)
            self.window.set_result_state("stale", f"当前结果已过期：{msg}")
            self.window.statusBar().showMessage(f"构建失败: {e}")
            self.window.panel.set_calculating(False)

    def _do_rebuild(self):
        panel = self.window.panel
        # A built-in Kagome model has two faithful representations: the
        # six-site orthogonal supercell used for ribbons/rectangles and the
        # three-site oblique primitive cell used for an equilateral triangular
        # nanodisk.  Shape changes in the editor must switch the basis as well
        # as the mask; otherwise a rectangular six-site basis is merely
        # clipped by a triangle and produces the sparse/incorrect disk users
        # see.  The migration helper is strictly guarded to leave hand-edited
        # models untouched.
        migration_message = self._migrate_builtin_kagome_shape()
        symbolic = panel.is_symbolic()
        site_rows = panel.get_site_rows()
        if not site_rows:
            raise ValueError("格点表为空 —— 请先添加格点或从“文件”加载预设")
        hop_rows = panel.get_hop_rows()
        boundary = self._build_boundary()
        self.window.set_boundary_mode(boundary.kind is BoundaryKind.SEMI)
        order = panel.get_order()

        # 参数面板同步: 自动收集表达式中的自由符号, 保留用户已编辑数值
        names = collect_param_names(
            [h["amplitude"] for h in hop_rows] + [h["phase"] for h in hop_rows]
        )
        old = panel.get_params()
        merged = {n: old.get(n, DEFAULT_PARAMS.get(n, 1.0)) for n in sorted(names)}
        panel.set_params(merged)
        params = panel.get_params()
        kx = panel.get_kx()

        # Geometry is part of the Hamiltonian/display revision.  Omitting it
        # from this signature makes a shape or primitive-cell spacing edit
        # look like a cheap kx-only refresh: the old matrix/scene then stays
        # on screen even though the user explicitly changed the model.
        cell_vectors = panel.get_cell_vectors()
        if cell_vectors is not None:
            cell_signature = (
                "vectors",
                tuple(tuple(float(value) for value in vector)
                      for vector in cell_vectors),
            )
        else:
            cell_size = panel.get_cell_size()
            cell_signature = (
                "rect",
                None if cell_size is None
                else tuple(float(value) for value in cell_size),
            )
        # 签名缓存: 只动 kx → 轻量快路径
        sig = (
            symbolic, order,
            boundary.kind.value, boundary.NX, boundary.NY, boundary.shape,
            cell_signature,
            tuple(site_rows),
            tuple(tuple(sorted(h.items())) for h in hop_rows),
            tuple(sorted(params.items())),
        )
        if sig == self._sig and self._state is not None:
            self._refresh_kx_only(kx)
            panel.set_error("")
            if self._generation in self._spectral_context:
                self.window.set_result_state("busy", "谱数据仍在后台计算…", show_banner=False)
                self.window.flash_status("谱数据仍在后台计算…", duration_ms=1800)
            else:
                self.window.set_result_state("ready")
                panel.set_calculating(False)
            callback = getattr(self.window, "document_committed", None)
            if callback is not None:
                callback(self._current_document())
            return

        self.window.set_result_state("busy", "正在构建哈密顿量并刷新视图…", show_banner=False)
        self.window.flash_status("正在构建哈密顿量并刷新视图…", duration_ms=1800)
        lattice = self._build_lattice(site_rows)
        hops = self._build_hops(symbolic, params)
        builder = HamiltonianBuilder(lattice, hops, boundary, order)
        res = builder.build()

        self._sig = sig
        self._state = (res, lattice, boundary, params, symbolic)
        self._display_hops = hops
        self._generation += 1
        generation = self._generation
        self._push_matrix(res, boundary, params, symbolic, kx)
        self._push_lattice(lattice, hops, boundary, res)
        async_spectral = res.Nat >= SPECTRAL_ASYNC_THRESHOLD
        if boundary.kind is BoundaryKind.SEMI:
            if async_spectral:
                H0, H1, extra = self._numeric_ribbon(res, params)
                kxs = np.linspace(-np.pi, np.pi, 201)
                self.window.band_scene.set_message("正在后台计算能带…")
                self._start_spectral(
                    generation, "band", (H0, H1, extra, kxs), res,
                )
            else:
                self._push_band(res, kx)
        else:
            if async_spectral:
                H = res.H
                if hasattr(H, "free_symbols"):
                    H = self._sym_to_num(H, params, 0.0)
                self.window.wf_view.set_loading("正在后台对角化并计算波函数…")
                self._start_spectral(generation, "wf", (H, tuple(res.positions)), res)
            else:
                self._push_wf(res)
        panel.set_error("")
        if async_spectral:
            self.window.set_result_state(
                "busy", "矩阵与晶格已更新，谱数据正在后台计算…", show_banner=False
            )
            self.window.flash_status("矩阵与晶格已更新，谱数据正在后台计算…", duration_ms=2200)
        else:
            self.window.set_result_state("ready")
            panel.set_calculating(False)
        status = self._status_msg(res, boundary, kx)
        if migration_message:
            status = f"{migration_message} | {status}"
        self.window.statusBar().showMessage(status)
        callback = getattr(self.window, "document_committed", None)
        if callback is not None:
            callback(self._current_document())

    def _status_msg(self, res, boundary, kx) -> str:
        mode = "半无限" if boundary.kind is BoundaryKind.SEMI else "双开"
        ktxt = f" | kx/π={kx / np.pi:.3f}" if boundary.kind is BoundaryKind.SEMI else ""
        message = f"{mode} | {res.Nsites} 格点 | 矩阵 {res.Nat}×{res.Nat} | 刷新 OK{ktxt}"
        # Boundary truncation is physically expected at an open edge, but it
        # is otherwise invisible when a user adds an inter-cell row to a
        # one-cell-wide OBC model.  Surface the count in the status bar so a
        # dropped edge bond is explainable instead of looking like a failed
        # add/rebuild operation.
        if boundary.kind is BoundaryKind.SEMI:
            cut = int(res.skipped.get("y_cut", 0))
            if cut:
                message += f" | 有限方向截断 {cut} 条"
        else:
            cut = int(res.skipped.get("oob", 0))
            if cut:
                message += f" | 边界外跳过 {cut} 条"
        return message

    def _refresh_kx_only(self, kx: float):
        """仅 kx 变化: 重算矩阵数值 + 移动能带标记线 (不重算能带/晶格)."""
        res, _lattice, boundary, params, symbolic = self._state
        if boundary.kind is BoundaryKind.SEMI:
            if symbolic:
                vals = self._sym_to_num(self._matrix_obj, params, kx)
                matrix, mode = self._matrix_obj, "symbolic"
            else:
                vals = res.to_semi(kx)
                # 默认智能标签保留 Bloch 相位结构；纯数值模式则明确
                # 显示当前 kx 已代入后的复数。
                matrix = (res.to_semi(param("kx"))
                          if self._mode == "smart" else vals)
                mode = self._mode
            self.window.matrix_scene.set_data(MatrixSceneData(
                n=res.Nat,
                values=np.asarray(vals, dtype=complex),
                matrix=matrix,
                mode=mode,
                sites=_display_matrix_labels(res, boundary),
                formatter=self._formatter if mode == "smart" else None,
                t=params.get("t"), phi=params.get("phi"),
                title=self._matrix_title(res, boundary, mode, kx),
            ))
            self._fit(self.window.matrix_gv)
            self._fit(self.window.combined_matrix_gv)
            # BandView 的横轴统一使用物理 kx（弧度），这里只移动标记线。
            self.window.band_scene.set_kx_mark(kx)
        self.window.statusBar().showMessage(self._status_msg(res, boundary, kx))

    # ---- 矩阵 ----

    def _matrix_title(self, res, boundary, mode, kx) -> str:
        tag = {"symbolic": "[符号]", "smart": "[智能]", "numeric": "[数值]"}[mode]
        if boundary.kind is BoundaryKind.SEMI:
            return (f"{res.Nat}×{res.Nat}  NY={boundary.NY}  Cells={res.Ncells}  "
                    f"Sites={res.Nsites}  kx/π={kx / np.pi:.3f}  {tag}")
        return (f"{res.Nat}×{res.Nat}  NX={boundary.NX} NY={boundary.NY}  "
                f"Cells={res.Ncells}  Sites={res.Nsites}  {tag}")

    def _push_matrix(self, res, boundary, params, symbolic: bool, kx: float):
        panel = self.window.panel
        if boundary.kind is BoundaryKind.SEMI:
            if symbolic:
                matrix = res.to_semi(param("kx"))
                vals = self._sym_to_num(matrix, params, kx)
                mode = "symbolic"
            else:
                mode = "smart" if panel.is_smart() else "numeric"
                vals = res.to_semi(kx)
                matrix = res.to_semi(param("kx")) if mode == "smart" else vals
        else:
            if symbolic:
                matrix = res.H
                vals = self._sym_to_num(matrix, params, 0.0)
                mode = "symbolic"
            else:
                matrix = res.H
                vals = res.H
                mode = "smart" if panel.is_smart() else "numeric"
        self._matrix_obj = matrix
        self._mode = mode
        self._formatter = (
            ElementFormatter(t=params.get("t", 1.0), phi=params.get("phi"), omg=params.get("omg"))
            if mode == "smart" else None
        )
        self.window.matrix_scene.set_data(MatrixSceneData(
            n=res.Nat,
            values=np.asarray(vals, dtype=complex),
            matrix=matrix,
            mode=mode,
            sites=_display_matrix_labels(res, boundary),
            formatter=self._formatter,
            t=params.get("t"), phi=params.get("phi"),
            title=self._matrix_title(res, boundary, mode, kx),
        ))
        self._fit(self.window.matrix_gv)
        self._fit(self.window.combined_matrix_gv)

    def _sym_to_num(self, matrix, params: dict, kx: float = 0.0) -> np.ndarray:
        """符号矩阵 → 数值 (lambdify 向量化 + 按 (符号集, 矩阵) 缓存)."""
        if matrix is None or getattr(matrix, "rows", 0) == 0:
            return np.zeros((0, 0), dtype=complex)
        syms = sorted(matrix.free_symbols, key=lambda s: str(s))
        vals = []
        for s in syms:
            n = str(s)
            if n in {"kx", "k_x"}:
                vals.append(kx)
            else:
                if n not in params:
                    raise ValueError(f"符号 {n!r} 没有对应参数值")
                vals.append(params[n])
        if not syms:
            return np.asarray(matrix, dtype=complex)
        names = tuple(str(s) for s in syms)
        if self._lam is None or self._lam[0] != names or self._lam[1] is not matrix:
            self._lam = (names, matrix, sp.lambdify(syms, matrix, modules="numpy"))
        return np.asarray(self._lam[2](*vals), dtype=complex)

    # ---- 晶格 ----

    @staticmethod
    def _edit_anchor_offset(lattice, boundary) -> tuple[float, float]:
        """Choose a real, central finite cell for primitive-cell handles.

        Disk/hexagon masks are centred in their finite bounding grid, so the
        historical hard-coded (0, 0) editor copy may lie outside the actual
        sample.  Select an active cell closest to the geometric centre and
        translate only the visual handles there; the editable site coordinates
        themselves remain local primitive-cell coordinates.
        """
        if boundary.kind is BoundaryKind.SEMI:
            cell = (0, (boundary.NY - 1) // 2)
        else:
            cells = tuple(boundary.active_cells())
            center_x = (boundary.NX - 1) / 2.0
            center_y = (boundary.NY - 1) / 2.0
            cell = min(
                cells,
                key=lambda c: ((c[0] - center_x) ** 2 + (c[1] - center_y) ** 2,
                               c[1], c[0]),
            )
        a1x, a1y = lattice.a1
        a2x, a2y = lattice.a2
        return (
            float(cell[0] * a1x + cell[1] * a2x),
            float(cell[0] * a1y + cell[1] * a2y),
        )

    def _push_lattice(self, lattice, hops, boundary, res):
        self.window.lattice_scene.set_display_options(
            **self.window.panel.lattice_display_options(), redraw=False,
        )
        data = _build_lattice_scene(
            lattice, hops, boundary, res,
            labels_bottom_up=self.window.panel.labels_bottom_up(),
        )
        hop_rows = self.window.panel.get_hop_rows()
        params = self.window.panel.get_params()
        edit_hops = []
        for row, hop in enumerate(hop_rows):
            try:
                strength = abs(evaluate_expression(hop["amplitude"], params))
            except ValueError:
                continue
            edit_hops.append({**hop, "row": row, "strength": float(strength)})
        self.window.lattice_scene.set_edit_context(
            self.window.panel.get_site_rows(), hops=edit_hops,
            cell_vectors=(lattice.a1, lattice.a2),
            snap_step=self.window.lattice_scene.snap_step,
            anchor_offset=self._edit_anchor_offset(lattice, boundary),
        )
        self.window.lattice_scene.set_data(data)
        self._fit(self.window.lattice_gv)
        self._fit(self.window.combined_lattice_gv)

    # ---- 能带 / 波函数 ----

    def _numeric_ribbon(self, res, params: dict):
        """符号 ribbon 的所有 x 谐波 → 数值块。"""
        H0, H1 = res.blocks["H0"], res.blocks["H1"]
        if hasattr(H0, "free_symbols"):
            H0 = self._sym_to_num(H0, params, 0.0)
            H1 = self._sym_to_num(H1, params, 0.0)
        extra = {}
        for key, value in res.extra.items():
            if isinstance(value, sp.Basic):
                substitutions = {}
                for symbol in value.free_symbols:
                    name = str(symbol)
                    if name not in params:
                        raise ValueError(f"符号 {name!r} 没有对应参数值")
                    substitutions[symbol] = params[name]
                value = complex(value.evalf(subs=substitutions))
            extra[key] = value
        return H0, H1, extra

    def _push_band(self, res, kx: float):
        H0, H1, extra = self._numeric_ribbon(res, self._state[3])
        ham = RibbonHamiltonian(H0, H1, extra)
        kxs = np.linspace(-np.pi, np.pi, 201)
        _, E = ham.bands(kxs)
        self.window.band_scene.set_data(BandSceneData(
            kx=kxs, energies=E, kx_mark=kx,
            title=f"E(kx)  {res.Nsites} 格点"))
        self._fit(self.window.band_gv)

    def _push_wf(self, res):
        H = res.H
        if hasattr(H, "free_symbols"):
            H = self._sym_to_num(H, self._state[3], 0.0)
        E, wf = wavefunctions(H, positions=tuple(res.positions))
        view = self.window.wf_view
        view.set_data(WfSceneData(
            energies=E, wf=wf, positions=tuple(res.positions), title="|ψ|²"))
        # 面板中的 E 是目标值；实际显示选取能量最近的本征态。
        try:
            target = self.window.panel.get_energy()
        except ValueError:
            target = 0.0
        view.select_energy(target)
        self._fit(view.view)

    def _on_energy_changed(self, energy: float):
        """只切换已计算的 OBC 本征态，不重建矩阵或能带。"""
        if self.window.panel.is_semi():
            return
        view = self.window.wf_view
        idx = view.select_energy(energy)
        if idx is not None and view.selected_energy is not None:
            self.window.statusBar().showMessage(
                f"已选择本征态 #{idx}，E={view.selected_energy:.6g}"
            )

    def _start_spectral(self, generation: int, kind: str, payload, res) -> None:
        self._spectral_context[generation] = (kind, res)
        worker = _SpectralWorker(generation, kind, payload, self)
        self._spectral_workers[generation] = worker
        worker.signals.finished.connect(self._on_spectral_finished)
        worker.signals.failed.connect(self._on_spectral_failed)
        self._thread_pool.start(worker)

    @Slot(int, str, object)
    def _on_spectral_finished(self, generation: int, kind: str, result) -> None:
        if generation != self._generation:
            self._spectral_context.pop(generation, None)
            self._spectral_workers.pop(generation, None)
            return
        context = self._spectral_context.pop(generation, None)
        self._spectral_workers.pop(generation, None)
        if context is None:
            return
        _kind, res = context
        if kind == "band":
            kxs, energies = result
            kx = self.window.panel.get_kx()
            self.window.band_scene.set_data(BandSceneData(
                kx=kxs, energies=energies, kx_mark=kx,
                title=f"E(kx)  {res.Nsites} 格点",
            ))
            self._fit(self.window.band_gv)
        else:
            energies, wf = result
            view = self.window.wf_view
            view.set_data(WfSceneData(
                energies=energies, wf=wf, positions=tuple(res.positions), title="|ψ|²",
            ))
            try:
                target = self.window.panel.get_energy()
            except ValueError:
                target = 0.0
            view.select_energy(target)
            self._fit(view.view)
        self.window.set_result_state("ready")
        self.window.panel.set_calculating(False)
        self.window.statusBar().showMessage("谱数据计算完成")
        callback = getattr(self.window, "document_committed", None)
        if callback is not None:
            callback(self._current_document())

    @Slot(int, str, str)
    def _on_spectral_failed(self, generation: int, kind: str, message: str) -> None:
        if generation != self._generation:
            self._spectral_workers.pop(generation, None)
            return
        self._spectral_context.pop(generation, None)
        self._spectral_workers.pop(generation, None)
        label = "能带" if kind == "band" else "波函数"
        self.window.panel.set_error(f"{label}计算失败: {message}")
        self.window.set_result_state("stale", f"{label}计算失败：{message}")
        self.window.panel.set_calculating(False)

    def _create_hopping(self, from_site: int, to_site: int):
        self._create_hopping_with_offset(from_site, to_site, 0, 0)

    def _create_hopping_with_offset(
        self, from_site: int, to_site: int, off_x: int, off_y: int,
    ):
        dialog = HoppingDialog(
            from_site, to_site, self.window, semi=self.window.panel.is_semi(),
            cell_offset=(int(off_x), int(off_y)),
        )
        if dialog.exec() == QDialog.Accepted:
            row = dialog.row(from_site, to_site)
            self.window.panel.append_hop(row, reveal_relation=True)
            # Keep the table and the canvas focused on the physical bond just
            # created.  This is especially important for a semi-infinite
            # inter-cell row: dx/dy are now visible immediately, so users can
            # verify that it entered the intended Bloch relation without
            # hunting through a long hopping list.
            table = self.window.panel.hop_table
            last = table.rowCount() - 1
            if last >= 0:
                table.selectRow(last)
                table.scrollToItem(
                    table.item(last, 0),
                    QAbstractItemView.PositionAtCenter,
                )
            self.window.panel._update_hop_relation_hint()
            relation = (
                "胞内"
                if int(row[3]) == 0 and int(row[4]) == 0
                else f"胞间 dx={int(row[3]):+d}, dy={int(row[4]):+d}"
            )
            self.window.statusBar().showMessage(
                f"已添加{relation}跃迁：格点 {int(row[1]) + 1} → {int(row[2]) + 1}",
                2200,
            )

    # ---- 交互 ----

    def _on_cell_clicked(self, i, j):
        """Compatibility entry point for integrations that call the old slot.

        Matrix inspection used to open a modal information box here.  Route
        old callers through the same non-modal MainWindow handler as real
        clicks, so external integrations do not reintroduce the old blocking
        interaction.
        """
        handler = getattr(self.window, "_on_matrix_cell_clicked", None)
        if handler is not None:
            handler(int(i), int(j))

    # ---- 文件 ----

    def _current_document(self) -> dict:
        panel = self.window.panel
        return model_to_dict(
            panel.get_site_rows(), panel.get_hop_rows(), self._build_boundary(),
            panel.get_order(), panel.get_params(), panel.get_kx(),
            panel.is_symbolic(), panel.is_smart(),
            panel.get_cell_size(),
            panel.get_cell_vectors(),
            panel.labels_bottom_up(),
            panel.lattice_display_options(),
        )

    def current_document(self) -> dict:
        return self._current_document()

    def _apply_document(self, obj: dict) -> None:
        panel = self.window.panel
        b = obj["boundary"]
        was_blocked = panel.blockSignals(True)
        self._debounce.stop()
        try:
            panel.set_lattice_rows(
                [(s["x"], s["y"], s.get("sublattice") or "") for s in obj["sites"]]
            )
            cell = obj.get("cell")
            if cell is not None and "a1" in cell:
                panel.set_cell_vectors((tuple(cell["a1"]), tuple(cell["a2"])))
            else:
                panel.set_cell_size((cell["Lx"], cell["Ly"]) if cell is not None else None)
            panel.set_hop_rows([hop_dict_to_row(h) for h in obj["hops"]])
            panel.set_boundary_index(0 if b["kind"] == "semi" else 1)
            panel.set_dim(b["NX"], b["NY"])
            panel.set_shape(b.get("shape", "rectangle"))
            panel.set_order(obj["order"])
            panel.set_kx(obj["kx"])
            panel.set_symbolic(obj["symbolic"])
            panel.set_smart(obj["smart"])
            panel.set_labels_bottom_up(obj.get("labels_bottom_up", True))
            panel.set_lattice_display_options(obj.get("lattice_display"))
            panel.set_params(obj["params"], force=True)
        finally:
            panel.blockSignals(was_blocked)

    def apply_document(self, obj: dict, *, rebuild: bool = True) -> None:
        self._apply_document(obj)
        if rebuild:
            self.rebuild()

    def save_model(self):
        panel = self.window.panel
        models_dir = app_data_dir() / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self.window, "保存模型", str(models_dir / "未命名模型.hvisual"),
            "HamiVisualizer 模型 (*.hvisual);;JSON (*.json)")
        if not path:
            return False
        try:
            save_model(path, self._current_document())
        except (OSError, TypeError, ValueError) as exc:
            panel.set_error(f"保存失败: {exc}")
            QMessageBox.critical(self.window, "保存失败", str(exc))
            return False
        self.window.set_dirty(False)
        self.window.statusBar().showMessage(f"模型已保存: {path}")
        return True

    def open_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self.window, "打开模型", str(app_data_dir() / "models"),
            "HamiVisualizer 模型 (*.hvisual *.json)")
        if not path:
            return
        previous = None
        try:
            obj = load_model(path)  # 完整验证在任何 UI 修改之前完成
            try:
                previous = self._current_document()
            except (TypeError, ValueError):
                # 当前编辑内容本就无效时，仍允许用一个有效文件恢复工作。
                previous = None
            self._apply_document(obj)
        except (OSError, TypeError, ValueError) as exc:
            self.window.panel.set_error(f"打开失败: {exc}")
            QMessageBox.critical(self.window, "打开失败", str(exc))
            return
        try:
            self.rebuild()
            if self.window.panel.error_label.text():
                raise ValueError(self.window.panel.error_label.text())
        except Exception as exc:  # defensive rollback after a validated document
            if previous is not None:
                self._apply_document(previous)
                self.rebuild()
            QMessageBox.critical(self.window, "打开失败", f"模型未应用，已恢复原状态。\n{exc}")
            return
        self.window.statusBar().showMessage(f"模型已打开: {path}")
        self.window.set_dirty(False)

    def export_png(self):
        path, _ = QFileDialog.getSaveFileName(
            self.window, "导出 PNG", "hamivisualizer.png", "PNG (*.png)")
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        w = self.window.tabs.currentWidget()
        try:
            ok = w.grab().save(path)
        except OSError as exc:
            QMessageBox.critical(self.window, "导出失败", str(exc))
            return
        self.window.statusBar().showMessage(f"已导出: {path}" if ok else "导出失败")

    def export_svg(self):
        """Export the current result tab as a true SVG drawing.

        Rendering the visible QWidget (rather than only the scene) preserves
        the frozen matrix rulers, tab-specific overlays and the two-pane
        comparison layout.  ``QSvgGenerator`` receives the same QPainter
        commands as the screen, so text and geometry remain editable/vector
        instead of becoming a screenshot embedded in an SVG.
        """
        path, _ = QFileDialog.getSaveFileName(
            self.window, "导出 SVG", "hamivisualizer.svg", "SVG (*.svg)",
        )
        if not path:
            return
        if not path.lower().endswith(".svg"):
            path += ".svg"
        try:
            from PySide6.QtSvg import QSvgGenerator

            widget = self.window.tabs.currentWidget()
            size = widget.size()
            if size.width() <= 1 or size.height() <= 1:
                raise ValueError("当前视图尚未完成布局，无法导出")
            generator = QSvgGenerator()
            generator.setFileName(path)
            generator.setSize(size)
            generator.setViewBox(QRect(0, 0, size.width(), size.height()))
            generator.setResolution(96)
            generator.setTitle("HamiVisualizer")
            generator.setDescription("HamiVisualizer 当前视图")
            painter = QPainter(generator)
            try:
                widget.render(
                    painter, QPoint(), QRegion(),
                    QWidget.DrawWindowBackground | QWidget.DrawChildren,
                )
            finally:
                painter.end()
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            QMessageBox.critical(self.window, "导出失败", str(exc))
            return
        self.window.statusBar().showMessage(f"已导出 SVG：{path}")

    def export_pdf(self):
        """Export the current result tab as a single-page vector PDF."""
        path, _ = QFileDialog.getSaveFileName(
            self.window, "导出 PDF", "hamivisualizer.pdf", "PDF (*.pdf)",
        )
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        try:
            widget = self.window.tabs.currentWidget()
            size = widget.size()
            if size.width() <= 1 or size.height() <= 1:
                raise ValueError("当前视图尚未完成布局，无法导出")
            # Use a 96-DPI pixel-to-point conversion so the PDF has the same
            # physical aspect as the visible view instead of silently using
            # the printer's default A4 crop.
            points = QSizeF(size.width() * 72.0 / 96.0,
                            size.height() * 72.0 / 96.0)
            writer = QPdfWriter(path)
            writer.setResolution(96)
            writer.setPageLayout(QPageLayout(
                QPageSize(points, QPageSize.Point),
                QPageLayout.Landscape if points.width() >= points.height()
                else QPageLayout.Portrait,
                QMarginsF(0, 0, 0, 0),
            ))
            writer.setTitle("HamiVisualizer")
            painter = QPainter(writer)
            try:
                widget.render(
                    painter, QPoint(), QRegion(),
                    QWidget.DrawWindowBackground | QWidget.DrawChildren,
                )
            finally:
                painter.end()
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            QMessageBox.critical(self.window, "导出失败", str(exc))
            return
        self.window.statusBar().showMessage(f"已导出 PDF：{path}")

    # ---- 视口 ----

    def _fit(self, gv, force: bool = False):
        """场景矩形与视口尺寸都未变时不 fitInView —— 保留用户缩放/平移.

        不可见 (尺寸 ≤1) 的视图跳过, 等 tab 切换/窗口显示后再由 fit_all 补 fit
        (QTabWidget 隐藏页尺寸为 0, 此时 fitInView 会得到退化变换)。
        """
        rect = gv.scene().sceneRect()
        if not (rect.isValid() and not rect.isEmpty()):
            return
        # 用户滚轮缩放后，窗口 resize / 切 tab 不应偷偷把视图重置。
        # F/“适应窗口”显式 force 时才恢复基准缩放。
        if not force and getattr(gv, "user_zoomed", False):
            notifier = getattr(gv, "_notify_scene_zoom", None)
            if notifier is not None:
                notifier()
            return
        if force:
            reset = getattr(gv, "reset_user_zoom", None)
            if reset is not None:
                reset()
        w, h = gv.viewport().width(), gv.viewport().height()
        if w <= 1 or h <= 1:
            return
        key = id(gv)
        state = (rect, w, h)
        if not force and self._fit_seen.get(key) == state:
            return
        self._fit_seen[key] = state
        gv.fitInView(rect, Qt.KeepAspectRatio)

    def fit_all(self, force: bool = False):
        for gv in self.window.all_views():
            self._fit(gv, force=force)


def _build_lattice_scene_by_coordinates_legacy(
    lattice, hops, boundary, res, *, labels_bottom_up: bool = True,
) -> LatticeSceneData:
    """从构建结果生成晶格视图 DTO (格点/连线/虚影/元胞框).

    高保真移植 MATLAB draw_lattice:
      - 虚影列层数 D = 最大水平跃迁跨度 (NP=1 层, SC=2 层, 与 MATLAB 一致)
      - 虚影格点 = 每列向左/右逐单位偏移直至越出实区域
      - 虚影键 = 实键在 ±D 平移下的像, 至少一端为虚影 (MATLAB 画法)
      - 首胞 = cell_boxes[0], 视图高亮
    """
    positions = list(res.positions)
    q = lambda v: round(float(v), 6)  # noqa: E731
    central_coord_to_idx = {(q(x), q(y)): i for i, (x, y) in enumerate(positions)}
    if res.origin:
        r_of = [o[0] for o in res.origin]
    elif res.rmap is not None:
        r_of = [c[2] for c in res.rmap]
    else:
        r_of = [0] * len(positions)

    a1x, a1y = lattice.a1
    a2x, a2y = lattice.a2
    semi = boundary.kind is BoundaryKind.SEMI

    # 虚影按完整相邻元胞复制，而不是把坐标每次平移 1。后者只对
    # NP/SC 的整数网格碰巧成立，会破坏蜂窝等非单位元胞的几何。
    requested_ghost_layers = max(
        1, max((abs(h.cell_offset[0]) for h in hops), default=1)
    )
    ghost_layers = min(requested_ghost_layers, MAX_VISIBLE_GHOST_LAYERS)

    xs_all = [p[0] for p in positions]
    ys_all = [p[1] for p in positions]
    # ---- 显示层展开 -----------------------------------------------------
    # 周期显示不能把矩阵折叠后的索引直接当成几何节点索引。先构造可见
    # 坐标，再按“有限开边界网格”的列/行顺序编号；矩阵周期性只在
    # HamiltonianBuilder/build_ribbon 中由 H0/H1 处理。
    ghost_coords: set[tuple[float, float]] = set()
    if semi:
        for x, y in positions:
            for layer in range(1, ghost_layers + 1):
                ghost_coords.add((q(x - layer * a1x), q(y - layer * a1y)))
                ghost_coords.add((q(x + layer * a1x), q(y + layer * a1y)))

    central_coords = {(q(x), q(y)) for x, y in positions}
    display_coords = central_coords | ghost_coords
    # 半无限两侧仅是中心 ribbon 的平移像：必须复用中心元胞/格点编号，
    # 不能按展开后的绝对位置重新从 1 编到末尾。
    central_label = {}
    if res.origin:
        for index, (x, y) in enumerate(positions):
            r, cy = res.origin[index]
            cell_no = cy + 1 if labels_bottom_up else boundary.NY - cy
            central_label[(q(x), q(y))] = f"{cell_no}:{r + 1}"
    else:
        central_label = {(q(x), q(y)): str(i + 1) for i, (x, y) in enumerate(positions)}
    display_label = dict(central_label)
    for source, (x, y) in enumerate(positions):
        label = central_label[(q(x), q(y))]
        for layer in range(1, ghost_layers + 1):
            display_label[(q(x - layer * a1x), q(y - layer * a1y))] = label
            display_label[(q(x + layer * a1x), q(y + layer * a1y))] = label

    sites = []
    for i, (x, y) in enumerate(positions):
        r = r_of[i] if i < len(r_of) else 0
        sub = lattice.sites[r].sublattice if r < lattice.N else "A"
        key = (q(x), q(y))
        # User-facing site numbering starts at one.  The matrix builder keeps
        # zero-based indices internally; only this presentation DTO is
        # shifted so editing/calculation semantics remain unchanged.
        label = display_label[key] if semi else str(i + 1)
        sites.append((x, y, label, sub or "A"))

    ghost = [
        (x, y, display_label[(x, y)])
        for x, y in sorted(ghost_coords)
    ]
    ghost_sites = []
    if semi and res.origin:
        # ``ghost`` stays backward-compatible (x, y, label), while this
        # richer DTO retains the logical source and cell displacement needed
        # by the explicit canvas bond tool.
        anchor_cy = (boundary.NY - 1) // 2
        for source, (x, y) in enumerate(positions):
            primitive_site = int(res.origin[source][0]) if res.origin else source
            source_cy = int(res.origin[source][1]) if res.origin else anchor_cy
            label = central_label[(q(x), q(y))]
            for layer in range(1, ghost_layers + 1):
                for cell_dx in (-layer, layer):
                    ghost_sites.append((
                        q(x + cell_dx * a1x),
                        q(y + cell_dx * a1y),
                        label, primitive_site, cell_dx, source_cy - anchor_cy,
                    ))

    # 显示节点到中心 ribbon 节点的映射。虚影只沿 a1 平移，直接保留
    # 源节点即可；无需将斜原胞错误地按笛卡尔 x 坐标折叠。
    display_source: dict[tuple[float, float], int] = {}
    source_exact: dict[tuple[float, float], tuple[float, float]] = {}
    for source, (x, y) in enumerate(positions):
        display_source[(q(x), q(y))] = source
        source_exact[(q(x), q(y))] = (float(x), float(y))
        for layer in range(1, ghost_layers + 1):
            lx = x - layer * a1x; ly = y - layer * a1y
            rx = x + layer * a1x; ry = y + layer * a1y
            display_source[(q(lx), q(ly))] = source
            source_exact[(q(lx), q(ly))] = (float(lx), float(ly))
            display_source[(q(rx), q(ry))] = source
            source_exact[(q(rx), q(ry))] = (float(rx), float(ry))

    edges = set()
    ghost_edges = set()
    for h in hops:
        d = h.displacement(lattice)
        kind = "NN" if math.hypot(*d) <= 1.05 else "NNN"
        for (x0, y0), source_idx in display_source.items():
            if source_idx >= len(r_of) or r_of[source_idx] != h.from_site:
                continue
            # 用未舍入的精确源坐标计算目标，避免源坐标先被 round(.,6)
            # 再叠加位移，在舍入边界处与目标格点 key 差 1e-6 导致匹配失败。
            ex, ey = source_exact[(x0, y0)]
            target = (q(ex + d[0]), q(ey + d[1]))
            if target not in display_source or target == (x0, y0):
                continue
            a = (q(x0), q(y0))
            b = target
            out_a = a not in central_coords
            out_b = b not in central_coords
            if semi and (out_a or out_b):
                left, right = sorted((a, b))
                ghost_edges.add((*left, *right, kind))
            else:
                i = central_coord_to_idx.get(a)
                j = central_coord_to_idx.get(b)
                if i is not None and j is not None and i != j:
                    edges.add(tuple(sorted((i, j))) + (kind,))

    # 元胞框: 首胞在前 (视图高亮), 其余虚线
    cell_boxes = []
    cell_polygons = []
    if semi:
        for cy in range(boundary.NY):
            ox, oy = cy * a2x, cy * a2y
            cell_polygons.append(((ox, oy), (ox + a1x, oy + a1y),
                                  (ox + a1x + a2x, oy + a1y + a2y),
                                  (ox + a2x, oy + a2y)))
    else:
        active_cells = set(boundary.active_cells())
        for cx in range(boundary.NX):
            for cy in range(boundary.NY):
                if (cx, cy) not in active_cells:
                    continue
                ox, oy = cx * a1x + cy * a2x, cx * a1y + cy * a2y
                cell_polygons.append(((ox, oy), (ox + a1x, oy + a1y),
                                      (ox + a1x + a2x, oy + a1y + a2y),
                                      (ox + a2x, oy + a2y)))

    if semi:
        title = f"x: Bloch (∞), y: {boundary.NY} 胞 ({res.Nsites} 格点)"
    else:
        title = f"{boundary.NX}×{boundary.NY} 胞 ({res.Nsites} 格点)"

    return LatticeSceneData(
        sites=tuple(sites),
        edges=tuple(sorted(edges, key=lambda e: (e[2], e[0], e[1]))),
        ghost_edges=tuple(sorted(ghost_edges)),
        semi=semi,
        ghost=tuple(ghost),
        ghost_sites=tuple(ghost_sites),
        cell_boxes=tuple(cell_boxes),
        cell_polygons=tuple(cell_polygons),
        title=title,
        boundary_outline=_finite_shape_outline(lattice, boundary, positions),
    )


def _hopping_shell_kinds(lattice, hops) -> list[str | None]:
    """Classify bonds using the model's nearest nonzero distance as scale."""
    lengths = [math.hypot(*hop.displacement(lattice)) for hop in hops]
    positive = [length for length in lengths if length > 1e-12]
    if not positive:
        return [None] * len(hops)
    nearest = min(positive)
    tolerance = max(1e-9, nearest * 1e-6)
    return [
        None if length <= 1e-12
        else "NN" if abs(length - nearest) <= tolerance
        else "NNN"
        for length in lengths
    ]


def _display_matrix_labels(res, boundary) -> tuple[str, ...]:
    """Return one-based, human-facing labels for matrix rulers.

    ``HResult`` deliberately stores zero-based topology labels because those
    labels are also useful when debugging the numerical backend.  Matrix
    rulers and copy/status text are user-facing, however, and should follow
    the conventional one-based numbering used by the lattice canvas.  Keep
    the conversion topology-aware so cell/site labels such as ``1,2:3`` stay
    unambiguous while third-party custom labels are preserved verbatim.
    """
    n = int(getattr(res, "Nat", 0) or 0)
    origin = getattr(res, "origin", None)
    if boundary.kind is BoundaryKind.SEMI and origin and len(origin) == n:
        return tuple(f"{int(cy) + 1}:{int(r) + 1}" for r, cy in origin)
    rmap = getattr(res, "rmap", None)
    if boundary.kind is BoundaryKind.OBC and rmap is not None and len(rmap) == n:
        return tuple(
            f"{int(cx) + 1},{int(cy) + 1}:{int(r) + 1}"
            for cx, cy, r in rmap
        )
    labels = tuple(getattr(res, "labels", ()) or ())
    if len(labels) == n:
        # Best-effort conversion for legacy/plugin results that expose only
        # the historical textual labels.  Labels outside this exact grammar
        # are intentionally treated as user-authored and left untouched.
        import re

        pattern = re.compile(r"^(\d+)(?:,(\d+))?:(\d+)$")
        converted = []
        for raw in labels:
            match = pattern.match(str(raw))
            if not match:
                converted.append(str(raw))
                continue
            values = [int(token) + 1 for token in match.groups() if token is not None]
            if len(values) == 2:
                converted.append(f"{values[0]}:{values[1]}")
            else:
                converted.append(f"{values[0]},{values[1]}:{values[2]}")
        return tuple(converted)
    return tuple(str(i + 1) for i in range(n))


def _build_lattice_scene(lattice, hops, boundary, res, *,
                         labels_bottom_up: bool = True) -> LatticeSceneData:
    """Build lattice graphics from integer cell/site topology.

    Display connectivity is addressed by ``(cell_x, cell_y, site_r)``.  Real
    coordinates are calculated only after the endpoints are known, so oblique
    cells never depend on rounded-coordinate lookup.  The older coordinate
    implementation remains a compatibility fallback for third-party HResult
    objects that do not expose ``origin``/``rmap``.
    """
    positions = list(res.positions)
    semi = boundary.kind is BoundaryKind.SEMI
    a1x, a1y = lattice.a1
    a2x, a2y = lattice.a2
    kinds = _hopping_shell_kinds(lattice, hops)
    requested_ghost_layers = max(
        1, max((abs(h.cell_offset[0]) for h in hops), default=1)
    )
    ghost_layers = min(requested_ghost_layers, MAX_VISIBLE_GHOST_LAYERS)

    if semi:
        if not res.origin or len(res.origin) != len(positions):
            return _build_lattice_scene_by_coordinates_legacy(
                lattice, hops, boundary, res,
                labels_bottom_up=labels_bottom_up,
            )
        central_index = {
            (0, int(cy), int(r)): index
            for index, (r, cy) in enumerate(res.origin)
        }
        x_cells = range(-ghost_layers, ghost_layers + 1)
        central_x = {0}
        active_cells = {(0, cy) for cy in range(boundary.NY)}
    else:
        if res.rmap is None or len(res.rmap) != len(positions):
            return _build_lattice_scene_by_coordinates_legacy(
                lattice, hops, boundary, res,
                labels_bottom_up=labels_bottom_up,
            )
        central_index = {
            (int(cx), int(cy), int(r)): index
            for index, (cx, cy, r) in enumerate(res.rmap)
        }
        active_cells = set(boundary.active_cells())
        x_cells = range(boundary.NX)
        central_x = set(x_cells)

    node_positions = {
        (cx, cy, r): lattice.position(cx, cy, r)
        for cx in x_cells
        for cy in range(boundary.NY)
        for r in range(lattice.N)
        if semi or (cx, cy) in active_cells
    }

    def label_for(cy: int, r: int) -> str:
        cell_no = cy + 1 if labels_bottom_up else boundary.NY - cy
        return f"{cell_no}:{r + 1}"

    sites = [None] * len(positions)
    for key, index in central_index.items():
        _cx, cy, r = key
        x, y = positions[index]
        sub = lattice.sites[r].sublattice or "A"
        label = label_for(cy, r) if semi else str(index + 1)
        sites[index] = (x, y, label, sub)
    if any(site is None for site in sites):
        raise ValueError("晶格逻辑索引与绘图坐标不完全对应")

    ghost = []
    ghost_sites = []
    if semi:
        anchor_cy = (boundary.NY - 1) // 2
        for (cx, cy, r), (x, y) in sorted(node_positions.items()):
            if cx != 0:
                ghost.append((x, y, label_for(cy, r)))
                ghost_sites.append((
                    x, y, label_for(cy, r), int(r), int(cx),
                    int(cy) - anchor_cy,
                ))

    central_edges: dict[tuple[int, int], str] = {}
    ghost_edge_keys: dict[
        tuple[tuple[int, int, int], tuple[int, int, int]], str
    ] = {}

    def keep_kind(previous: str | None, incoming: str) -> str:
        return "NN" if previous == "NN" or incoming == "NN" else "NNN"

    for hop, kind in zip(hops, kinds):
        if kind is None:
            continue
        ox, oy = hop.cell_offset
        for cx in x_cells:
            for cy in range(boundary.NY):
                source = (cx, cy, hop.from_site)
                target = (cx + ox, cy + oy, hop.to_site)
                if source == target or target not in node_positions:
                    continue
                logical_pair = tuple(sorted((source, target)))
                if semi and (
                    source[0] not in central_x or target[0] not in central_x
                ):
                    ghost_edge_keys[logical_pair] = keep_kind(
                        ghost_edge_keys.get(logical_pair), kind
                    )
                    continue
                i = central_index.get(source)
                j = central_index.get(target)
                if i is None or j is None or i == j:
                    continue
                pair = tuple(sorted((i, j)))
                central_edges[pair] = keep_kind(central_edges.get(pair), kind)

    edges = tuple(sorted(
        ((i, j, kind) for (i, j), kind in central_edges.items()),
        key=lambda edge: (edge[2], edge[0], edge[1]),
    ))
    ghost_edges = []
    for (source, target), kind in ghost_edge_keys.items():
        x1, y1 = node_positions[source]
        x2, y2 = node_positions[target]
        if (x2, y2) < (x1, y1):
            x1, y1, x2, y2 = x2, y2, x1, y1
        ghost_edges.append((x1, y1, x2, y2, kind))
    ghost_edges.sort()

    # A primitive cell is always its true Bravais parallelogram.  Honeycomb
    # therefore stays a two-site oblique cell instead of a four-site box.
    cell_polygons = []
    polygon_x = (0,) if semi else range(boundary.NX)
    for cx in polygon_x:
        for cy in range(boundary.NY):
            if not semi and (cx, cy) not in active_cells:
                continue
            ox = cx * a1x + cy * a2x
            oy = cx * a1y + cy * a2y
            cell_polygons.append((
                (ox, oy),
                (ox + a1x, oy + a1y),
                (ox + a1x + a2x, oy + a1y + a2y),
                (ox + a2x, oy + a2y),
            ))

    title = (
        f"x: Bloch (∞), y: {boundary.NY} 胞 ({res.Nsites} 格点)"
        if semi else
        f"{boundary.NX}×{boundary.NY} 胞 ({res.Nsites} 格点)"
    )
    return LatticeSceneData(
        sites=tuple(sites),
        edges=edges,
        ghost_edges=tuple(ghost_edges),
        semi=semi,
        ghost=tuple(ghost),
        ghost_sites=tuple(ghost_sites),
        cell_polygons=tuple(cell_polygons),
        title=title,
        boundary_outline=_finite_shape_outline(lattice, boundary, positions),
    )
