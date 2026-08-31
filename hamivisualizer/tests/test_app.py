"""Controller 接线冒烟测试 (offscreen, 不启动事件循环)."""

import os
import math

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import (
    QApplication, QGraphicsEllipseItem, QGraphicsLineItem, QGraphicsItem,
    QGraphicsPolygonItem, QFileDialog,
)

from hamivisualizer.controller import ViewController
from hamivisualizer.view.main_window import MainWindow
from hamivisualizer.model.templates import template_document
from hamivisualizer.model.expression import evaluate_expression
from hamivisualizer.model.boundary import BOUNDARY_SHAPES, Boundary, BoundaryKind
from hamivisualizer.model.hamiltonian import HamiltonianBuilder
from hamivisualizer.model.hopping import HoppingTerm
from hamivisualizer.model.lattice import Lattice, Site
from hamivisualizer.model.persistence import validate_model_dict


def _mk():
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    ctrl = ViewController(win)
    return app, win, ctrl


def test_load_np_semi():
    _app, win, ctrl = _mk()
    ctrl.load_preset("NP")
    md = win.matrix_scene._data
    assert md is not None
    assert md.n == 8  # NP 半无限 NY=2 → 8 格点
    assert md.values.shape == (8, 8)
    assert win.lattice_scene._data is not None
    assert len(win.lattice_scene._data.sites) == 8
    # 能带页已更新
    assert win.band_scene._data is not None
    # 参数面板自动生成: t / φ / ω
    params = win.panel.get_params()
    assert set(params) >= {"t", "phi", "omg"}
    assert abs(params["phi"] - np.pi / 4) < 1e-6
    # 预设以符号表达式入表 (数值模式解析, 符号模式重建)
    assert any(win.panel.hop_table.item(r, 5).text() == "-t"
               for r in range(win.panel.hop_table.rowCount()))


def test_vector_export_writes_svg_and_pdf(tmp_path, monkeypatch):
    """The current tab can be exported as real vector SVG and PDF files."""
    _app, win, ctrl = _mk()
    ctrl.apply_document(template_document("NP", nx=2, ny=2, boundary_kind="semi"))
    win.resize(1000, 700)
    win.show()
    QApplication.processEvents()

    svg_base = tmp_path / "current-view"
    pdf_base = tmp_path / "current-view-pdf"
    targets = iter((str(svg_base), str(pdf_base)))
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (next(targets), ""),
    )

    ctrl.export_svg()
    ctrl.export_pdf()

    svg_path = tmp_path / "current-view.svg"
    pdf_path = tmp_path / "current-view-pdf.pdf"
    assert svg_path.read_text(encoding="utf-8").lstrip().startswith("<?xml")
    assert svg_path.stat().st_size > 100
    assert pdf_path.read_bytes().startswith(b"%PDF")
    assert pdf_path.stat().st_size > 100
    assert "PDF" in win.statusBar().currentMessage()
    win.close()


def test_matrix_latex_copy_uses_current_symbolic_cells_and_bounds_size():
    """Matrix LaTeX copy mirrors the inspected cells and has a safe bound."""
    _app, win, ctrl = _mk()
    ctrl.load_preset("NP")
    data = win.matrix_scene._data
    assert data is not None
    latex = win.matrix_scene.matrix_latex()
    assert latex.startswith("\\begin{bmatrix}\n")
    assert latex.endswith("\n\\end{bmatrix}")
    assert latex.count("\\\\") == data.n - 1
    assert "e^{i\\phi}" in latex
    assert "e^{-ik_{x}}" in latex
    win.action_copy_matrix_latex.trigger()
    assert QApplication.clipboard().text() == latex
    assert "矩阵 LaTeX" in win.statusBar().currentMessage()

    with pytest.raises(ValueError, match="安全上限"):
        win.matrix_scene.matrix_latex(max_elements=data.n * data.n - 1)


def test_load_sc_obc():
    _app, win, ctrl = _mk()
    ctrl.load_preset("SC")
    win.panel.boundary_combo.setCurrentIndex(1)  # 双开
    ctrl.rebuild()
    md = win.matrix_scene._data
    assert md is not None
    assert md.n == 8  # SC OBC NX=2 NY=2 → 2·4=8
    # 波函数页已更新
    assert win.wf_view._data is not None
    assert win.wf_view._data.energies.shape == (8,)
    # OBC 晶格连线非空 (回归: 修复 OBC 场景边仅 from_site=0 的缺陷)
    lat_data = win.lattice_scene._data
    assert len(lat_data.edges) > 4


def test_user_facing_lattice_and_matrix_numbering_starts_at_one():
    """Visible site/ruler coordinates use conventional one-based labels."""
    _app, win, ctrl = _mk()
    ctrl.apply_document(template_document(
        "方格", nx=2, ny=2, boundary_kind="obc", connectivity="最近邻",
    ))
    lattice_labels = [site[2] for site in win.lattice_scene._data.sites]
    assert lattice_labels[0] == "1"
    assert "0" not in lattice_labels
    matrix_labels = win.matrix_scene._data.sites
    assert matrix_labels[0] == "1,1:1"
    assert all(not label.startswith("0") for label in matrix_labels)

    ctrl.apply_document(template_document(
        "方格", ny=2, boundary_kind="semi", connectivity="最近邻",
    ))
    assert win.lattice_scene._data.sites[0][2] == "1:1"
    assert win.matrix_scene._data.sites[0] == "1:1"


def test_ssh_preset_keeps_intercell_bond_and_exposes_edge_states():
    """SSH's alternating bonds remain distinct and produce OBC edge modes."""
    _app, win, ctrl = _mk()
    ctrl.apply_document(template_document(
        "SSH", nx=8, ny=1, boundary_kind="obc", connectivity="最近邻",
    ))
    assert len(win.panel.get_site_rows()) == 2
    rows = win.panel.get_hop_rows()
    assert {(row["name"], row["off_x"]) for row in rows} == {
        ("t1", 0), ("t2", 1),
    }
    assert win.panel.get_params() == {"t1": 1.0, "t2": 1.0}

    # Enter the topological dimerization (t1 < t2) and verify that the
    # finite open chain develops the expected pair of near-zero edge modes.
    win.panel.set_params({"t1": 0.3, "t2": 1.0}, force=True)
    ctrl.rebuild()
    result = ctrl._state[0]
    assert result.Nat == 16
    matrix = np.asarray(result.H, dtype=complex)
    assert np.allclose(matrix, matrix.conj().T, atol=1e-10)
    energies = np.linalg.eigvalsh(matrix)
    assert np.count_nonzero(np.abs(energies) < 1e-3) == 2
    win.wf_view.select_energy(0.0)
    assert "边界局域态" in win.wf_view.info.text()


def test_haldane_preset_keeps_opposite_nnn_phases_and_complex_gap():
    """Haldane's mass/phase rows reach both Bloch and finite pipelines."""
    _app, win, ctrl = _mk()
    document = template_document(
        "Haldane", nx=4, ny=4, boundary_kind="semi",
        connectivity="最近邻+次近邻",
    )
    assert len(document["sites"]) == 2
    assert document["params"]["phi"] == pytest.approx(np.pi / 2)
    nnn = [row for row in document["hops"] if row["name"] == "t2"]
    assert len(nnn) == 6
    assert {row["phase_sign"] for row in nnn} == {-1, 1}
    assert {tuple(row["cell_offset"]) for row in nnn} == {
        (1, 0), (0, 1), (1, -1),
    }
    onsite = [row for row in document["hops"] if row["name"] == "m"]
    assert [row["amplitude"] for row in onsite] == ["m", "-m"]

    ctrl.apply_document(document)
    result = ctrl._state[0]
    bloch = np.asarray(result.to_semi(0.37), dtype=complex)
    assert np.allclose(bloch, bloch.conj().T, atol=1e-10)
    assert np.max(np.abs(bloch.imag)) > 1e-6

    obc = template_document(
        "Haldane", nx=3, ny=3, boundary_kind="obc",
        connectivity="最近邻+次近邻",
    )
    ctrl.apply_document(obc)
    finite = np.asarray(ctrl._state[0].H, dtype=complex)
    assert finite.shape == (18, 18)
    assert np.allclose(finite, finite.conj().T, atol=1e-10)
    assert np.min(np.abs(np.linalg.eigvalsh(finite))) > 1e-3


def test_symbolic_mode():
    _app, win, ctrl = _mk()
    ctrl.load_preset("NP")
    win.panel.symbolic_check.setChecked(True)
    ctrl.rebuild()
    md = win.matrix_scene._data
    assert md is not None
    assert md.mode == "symbolic"
    # ★ 回归: GUI 路径下符号模式必须真的显示 t (此前显示数字)
    assert str(md.matrix[0, 1]) != "0"
    assert any("t" in str(md.matrix[i, j])
               for i in range(md.n) for j in range(md.n))


def test_param_change_numeric():
    """参数面板改 t → 数值矩阵随之变化 (MATLAB t 滑块的等价物)."""
    _app, win, ctrl = _mk()
    ctrl.load_preset("NP")
    before = abs(win.matrix_scene._data.values[0, 1])  # -t·e^{iφ} → |·|=t
    # 找到参数表里 't' 的行, 把数值改成 2
    row = next(r for r in range(win.panel.param_table.rowCount())
               if win.panel.param_table.item(r, 0).text() == "t")
    win.panel.param_table.item(row, 1).setText("2")
    ctrl.rebuild()
    after = abs(win.matrix_scene._data.values[0, 1])
    assert abs(before - 1.0) < 1e-9
    assert abs(after - 2.0) < 1e-9


def test_phi_display_in_pi_units():
    """φ 数值列按 φ/π 显示 (MATLAB §2.1), 内部换回弧度."""
    _app, win, ctrl = _mk()
    ctrl.load_preset("NP")
    row = next(r for r in range(win.panel.param_table.rowCount())
               if win.panel.param_table.item(r, 0).text() == "phi")
    disp = float(win.panel.param_table.item(row, 1).text())
    assert abs(disp - 0.25) < 1e-6
    assert abs(win.panel.get_params()["phi"] - np.pi / 4) < 1e-6


def test_kx_fast_path():
    """只动 kx: 能带对象不重建 (缓存), 矩阵数值与标记线更新."""
    _app, win, ctrl = _mk()
    ctrl.load_preset("NP")
    band_before = win.band_scene._data
    mat_before = win.matrix_scene._data.values.copy()
    win.panel.kx_slider.setValue(50)  # kx/π = 0.5 → changed → 防抖(测试无事件循环)
    ctrl.rebuild()
    assert win.band_scene._data is band_before          # 能带未重算
    assert win.matrix_scene._data is not None
    assert not np.allclose(win.matrix_scene._data.values, mat_before)  # 矩阵更新
    assert win.band_scene._mark_item is not None        # 标记线已画


def test_shape_change_invalidates_geometry_cache_instead_of_kx_refresh():
    """Changing a finite-disk mask must rebuild the matrix and lattice."""
    _app, win, ctrl = _mk()
    ctrl.apply_document(template_document(
        "Kagome", nx=4, ny=4, boundary_kind="obc",
        connectivity="最近邻", shape="rectangle",
    ))
    old_result = ctrl._state[0]
    old_cells = old_result.Ncells
    win.panel.set_shape("triangle")
    ctrl.rebuild()
    new_result = ctrl._state[0]
    assert new_result is not old_result
    assert old_cells == 16
    assert 0 < new_result.Ncells < old_cells
    assert win.lattice_scene._data.title.startswith("4×4")


def test_kagome_shape_switch_migrates_builtin_basis_but_preserves_edits():
    """The Kagome shape selector changes the physical basis, not just a mask.

    A pristine rectangular Kagome preset uses the readable six-site
    orthogonal supercell.  Selecting a finite triangle must replace it with
    the three-site oblique primitive so the visible disk has the correct
    equilateral edge layout.  Once a user edits a site, the migration guard
    must stop treating the document as a built-in preset and leave that
    custom six-site model intact.
    """
    _app, win, ctrl = _mk()
    ctrl.apply_document(template_document(
        "Kagome", nx=3, ny=3, boundary_kind="obc",
        connectivity="最近邻", shape="rectangle",
    ))
    assert len(win.panel.get_site_rows()) == 6
    assert ctrl._state[0].Ncells == 9

    win.panel.set_shape("triangle")
    ctrl.rebuild()
    assert len(win.panel.get_site_rows()) == 3
    assert ctrl._state[0].Ncells == 6
    assert ctrl._state[0].Nat == 18
    assert "3 格点斜原胞" in win.statusBar().currentMessage()

    # Switching back is the inverse operation: the rectangular view gets the
    # six-site orthogonal supercell rather than a stretched three-site basis.
    win.panel.set_shape("rectangle")
    ctrl.rebuild()
    assert len(win.panel.get_site_rows()) == 6
    assert ctrl._state[0].Ncells == 9
    assert ctrl._state[0].Nat == 54
    assert "6 格点正交超胞" in win.statusBar().currentMessage()

    # A deliberate site edit opts the document out of preset migration.  A
    # subsequent shape change therefore keeps the user's six-site basis.
    ctrl.apply_document(template_document(
        "Kagome", nx=3, ny=3, boundary_kind="obc",
        connectivity="最近邻", shape="rectangle",
    ))
    first = win.panel.site_table.item(0, 0)
    first.setText("0.125")
    ctrl.rebuild()
    win.panel.set_shape("triangle")
    ctrl.rebuild()
    assert len(win.panel.get_site_rows()) == 6
    assert win.panel.get_site_rows()[0][0] == pytest.approx(0.125)


def test_cell_vector_change_invalidates_geometry_cache():
    """Editing primitive-cell spacing must reach both geometry and hopping displacement."""
    _app, win, ctrl = _mk()
    ctrl.apply_document(template_document(
        "蜂窝", ny=4, boundary_kind="semi", connectivity="最近邻",
    ))
    old_result = ctrl._state[0]
    old_a1 = tuple(ctrl._state[1].a1)
    old_a2 = tuple(ctrl._state[1].a2)
    win.panel.set_cell_vectors((
        (old_a1[0] * 1.2, old_a1[1] * 1.2),
        old_a2,
    ))
    ctrl.rebuild()
    new_result = ctrl._state[0]
    assert new_result is not old_result
    assert tuple(ctrl._state[1].a1) == pytest.approx(
        (old_a1[0] * 1.2, old_a1[1] * 1.2),
    )
    assert tuple(ctrl._state[1].a1) != old_a1


def test_display_options_refresh_immediately_without_rebuild():
    """智能标签/编号方向是显示选项，不应重建哈密顿量或能带。"""
    _app, win, ctrl = _mk()
    ctrl.load_preset("NP")
    generation = ctrl._generation
    band = win.band_scene._data
    first_label = win.lattice_scene._data.sites[0][2]

    win.panel.smart_check.setChecked(False)
    assert ctrl._generation == generation
    assert win.matrix_scene._data.mode == "numeric"
    assert win.band_scene._data is band
    assert "未重新计算" in win.statusBar().currentMessage()

    win.panel.cell_number_combo.setCurrentIndex(1)
    assert ctrl._generation == generation
    assert win.lattice_scene._data.sites[0][2] != first_label
    assert ctrl.current_document()["labels_bottom_up"] is False


def test_honeycomb_scene_has_exact_two_site_cells_and_complete_bonds():
    _app, win, ctrl = _mk()
    ctrl.apply_document(template_document(
        "蜂窝", ny=4, connectivity="最近邻+次近邻",
    ))
    data = win.lattice_scene._data
    assert len(data.sites) == 8              # 2 sites × 4 finite cells
    assert len(data.ghost) == 16             # one complete image cell each side
    assert len(data.cell_polygons) == 4       # true oblique primitive cells
    assert sum(edge[2] == "NN" for edge in data.edges) == 7
    assert sum(edge[2] == "NNN" for edge in data.edges) == 6
    assert {site[2] for site in data.sites} == {ghost[2] for ghost in data.ghost}
    # Periodic matrix terms are still separate from open-style display edges.
    res = ctrl._state[0]
    assert np.count_nonzero(np.asarray(res.blocks["H1"])) > 0
    assert np.allclose(res.to_semi(0.37), res.to_semi(0.37).conj().T)


def test_lattice_layer_toggles_are_immediate_and_persisted():
    _app, win, ctrl = _mk()
    ctrl.apply_document(template_document(
        "蜂窝", ny=4, connectivity="最近邻+次近邻",
    ))
    generation = ctrl._generation
    lines_before = sum(isinstance(item, QGraphicsLineItem)
                       for item in win.lattice_scene.items())
    win.panel.show_nnn_check.setChecked(False)
    lines_nn_only = sum(isinstance(item, QGraphicsLineItem)
                        for item in win.lattice_scene.items())
    assert lines_nn_only < lines_before
    assert ctrl._generation == generation

    win.panel.show_ghosts_check.setChecked(False)
    circles = sum(isinstance(item, QGraphicsEllipseItem)
                  for item in win.lattice_scene.items())
    assert circles == len(win.lattice_scene._data.sites)
    win.panel.show_cells_check.setChecked(False)
    assert not any(isinstance(item, QGraphicsPolygonItem)
                   for item in win.lattice_scene.items())
    assert ctrl.current_document()["lattice_display"] == {
        "nn": True, "nnn": False, "ghosts": False, "cells": False,
    }
    assert ctrl._generation == generation


def test_chain_nnn_template_updates_matrix_and_band_with_second_harmonic():
    _app, win, ctrl = _mk()
    ctrl.apply_document(template_document(
        "一维链", ny=1, connectivity="最近邻+次近邻",
    ))
    assert win.panel.error_label.text() == ""
    np.testing.assert_allclose(
        win.matrix_scene._data.values[0, 0], -4.0, atol=1e-12,
    )
    band = win.band_scene._data
    expected = -2.0 * np.cos(band.kx) - 2.0 * np.cos(2.0 * band.kx)
    assert np.allclose(band.energies[:, 0], expected, atol=1e-12)
    win.panel.set_kx(np.pi)
    ctrl.rebuild()
    np.testing.assert_allclose(
        win.matrix_scene._data.values[0, 0].real, 0.0, atol=1e-12,
    )


def test_default_extended_templates_use_unit_hopping_scale():
    """Every built-in hopping family starts at magnitude one by default."""
    for name in ("一维链", "方格", "三角", "蜂窝", "Kagome", "SSH"):
        document = template_document(
            name, boundary_kind="semi", connectivity="最近邻+次近邻",
        )
        params = document["params"]
        values = [
            abs(evaluate_expression(hop["amplitude"], params))
            for hop in document["hops"]
        ]
        assert values and all(value == pytest.approx(1.0) for value in values), name


def test_every_template_executes_through_both_boundary_pipelines():
    from hamivisualizer.model.templates import TEMPLATE_NAMES

    _app, win, ctrl = _mk()
    for name in TEMPLATE_NAMES:
        for boundary_kind in ("semi", "obc"):
            ctrl.apply_document(template_document(
                name, nx=2, ny=3, boundary_kind=boundary_kind,
                connectivity="最近邻+次近邻",
            ))
            assert win.panel.error_label.text() == "", (name, boundary_kind)
            result = ctrl._state[0]
            matrix = (
                result.to_semi(0.431)
                if boundary_kind == "semi" else np.asarray(result.H)
            )
            assert np.allclose(matrix, matrix.conj().T, atol=1e-10), (
                name, boundary_kind,
            )
            assert win.lattice_scene._data is not None
            if boundary_kind == "semi":
                assert win.band_scene._data.energies.shape[1] == result.Nat
            else:
                assert win.wf_view._data.wf.shape == (result.Nat, result.Nat)


def test_every_template_routes_x_intercell_rows_to_semi_bloch_harmonics():
    """All shipped presets keep dx!=0 distinct from their intra-cell H₀ terms."""
    from hamivisualizer.model.templates import TEMPLATE_NAMES

    _app, win, ctrl = _mk()
    for name in TEMPLATE_NAMES:
        ctrl.apply_document(template_document(
            name, nx=2, ny=3, boundary_kind="semi",
            connectivity="最近邻+次近邻",
        ))
        rows = win.panel.get_hop_rows()
        x_intercell = [row for row in rows if int(row["off_x"]) != 0]
        result = ctrl._state[0]
        if not x_intercell:
            continue
        h1_nnz = int(np.count_nonzero(np.asarray(result.blocks["H1"], dtype=complex)))
        assert h1_nnz + len(result.extra) > 0, name


def test_default_template_nearest_neighbour_coordination_is_physical():
    """Each shipped nearest-neighbour template keeps its intended bulk graph.

    Rendering and Hermiticity alone cannot catch a missing periodic hop: a
    lattice may still look plausible while an edge shell has the wrong
    coordination.  Count a self-hop with a non-zero cell offset twice, since
    it represents the left and right neighbours of the same basis site.
    """
    expected = {
        "NP": [4, 4, 4, 4],
        "SC": [4, 4],
        "空白自定义": [0],
        "方格": [4],
        "一维链": [2],
        "蜂窝": [3, 3],
        "三角": [6, 6],
        "Kagome": [4, 4, 4, 4, 4, 4],
        "SSH": [2, 2],
        "Haldane": [3, 3],
    }
    for name, degrees_expected in expected.items():
        document = template_document(name, connectivity="最近邻")
        degrees = [0] * len(document["sites"])
        for hop in document["hops"]:
            fr, to = hop["from_site"], hop["to_site"]
            ox, oy = hop["cell_offset"]
            if fr == to and ox == 0 and oy == 0:
                continue
            if fr == to:
                degrees[fr] += 2
            else:
                degrees[fr] += 1
                degrees[to] += 1
        assert degrees == degrees_expected, name

def test_edit_site_rebuild():
    """编辑格点表后重建不崩溃."""
    _app, win, ctrl = _mk()
    ctrl.load_preset("NP")
    win.panel.site_table.item(0, 0).setText("0.25")  # 改 x 坐标
    ctrl.rebuild()
    assert win.matrix_scene._data is not None


def test_kx_slider():
    _app, win, ctrl = _mk()
    ctrl.load_preset("NP")
    win.panel.kx_slider.setValue(50)  # kx/π = 0.5
    assert abs(win.panel.get_kx() / np.pi - 0.5) < 0.01
    assert win.panel.kx_label.text() == "kₓ/π"


def test_error_reported():
    """非法 phase_mode 不再静默: 错误出现在面板错误标签."""
    _app, win, ctrl = _mk()
    ctrl.load_preset("NP")
    win.panel.symbolic_check.setChecked(False)
    win.panel.hop_table.item(0, 6).setText("directional")  # 未实现的相位模式
    ctrl.rebuild()
    assert win.panel.error_label.text() != ""
    assert "directional" in win.panel.error_label.text()
    assert "none" in win.panel.error_label.text()


def test_semi_lattice_ghosts_np():
    """NP 半无限: 左右各复制一个完整元胞，形成三元胞展示."""
    _app, win, ctrl = _mk()
    ctrl.load_preset("NP")
    data = win.lattice_scene._data
    gx = sorted({g[0] for g in data.ghost})
    assert gx == [-2.0, -1.0, 2.0, 3.0]
    assert len(data.ghost_edges) > 0


def test_semi_lattice_ghosts_sc():
    """SC 半无限: NNN 水平步2 → 左右各两层虚影列, 与 MATLAB 一致."""
    _app, win, ctrl = _mk()
    ctrl.load_preset("SC")
    data = win.lattice_scene._data
    gx = sorted({g[0] for g in data.ghost})
    assert gx == [-2.0, -1.0, 2.0, 3.0]


def test_semi_display_reuses_center_cell_labels_for_ghosts():
    """虚影是中心 ribbon 的平移像，故应复用元胞:胞内格点编号。"""
    _app, win, ctrl = _mk()
    ctrl.load_preset("NP")
    data = win.lattice_scene._data
    central = {s[2] for s in data.sites}
    ghost = {g[2] for g in data.ghost}
    assert ghost == central
    assert all(":" in label for label in central)
    # 左右虚影仍是展开几何端点，而不是折回中心节点的零长度连线。
    assert any(edge[0] < 0 <= edge[2] for edge in data.ghost_edges)


def test_semi_periodic_cross_cell_term_is_kept_in_h1():
    """周期矩阵层单独通过 H1 保留左右跨元胞耦合。"""
    from hamivisualizer.model.boundary import Boundary, BoundaryKind
    from hamivisualizer.model.hamiltonian import HamiltonianBuilder
    from hamivisualizer.model.hopping import HoppingTerm
    from hamivisualizer.model.lattice import Lattice, Site

    lattice = Lattice([Site(0, 0.0, 0.0, "A")], Lx=1.0, Ly=1.0)
    hop = HoppingTerm("tc", 0, 0, (1, 0), 2.0)
    result = HamiltonianBuilder(
        lattice, [hop], Boundary(BoundaryKind.SEMI, NY=1)
    ).build_semi()
    assert result.blocks["H1"][0, 0] == 2.0
    assert result.to_semi(0.0)[0, 0] == 4.0
    assert result.to_semi(np.pi)[0, 0] == -4.0


def test_status_explains_open_boundary_intercell_truncation():
    """A one-cell OBC must explain why an added right bond is omitted."""
    _app, win, ctrl = _mk()
    ctrl.apply_document(template_document(
        "一维链", nx=1, ny=1, boundary_kind="obc", connectivity="最近邻",
    ))
    assert "边界外跳过" in win.statusBar().currentMessage()
    assert "1 条" in win.statusBar().currentMessage()


def test_obc_shape_mask_reaches_matrix_and_lattice_scene():
    _app, win, ctrl = _mk()
    ctrl.apply_document(template_document(
        "Kagome", nx=4, ny=4, boundary_kind="obc",
        connectivity="最近邻", shape="triangle",
    ))
    boundary = ctrl._build_boundary()
    assert boundary.shape == "triangle"
    active_count = len(boundary.active_cells())
    assert 0 < active_count < 16
    assert ctrl._state[0].Ncells == active_count
    assert len(win.lattice_scene._data.sites) == (
        len(win.panel.get_site_rows()) * active_count
    )
    assert len(win.lattice_scene._data.cell_polygons) == active_count
    assert len(win.lattice_scene._data.boundary_outline) == 3


def test_all_finite_shape_scenes_expose_a_physical_outline():
    """The renderer must make every non-rectangular mask unambiguous."""
    _app, win, ctrl = _mk()
    for shape, minimum_points in (("rectangle", 0), ("triangle", 3),
                                  ("disk", 24), ("hexagon", 6)):
        ctrl.apply_document(template_document(
            "Kagome", nx=5, ny=5, boundary_kind="obc",
            connectivity="最近邻", shape=shape,
        ))
        outline = win.lattice_scene._data.boundary_outline
        assert len(outline) >= minimum_points, shape
        if shape != "rectangle":
            assert len(win.lattice_scene._data.cell_polygons) == len(
                ctrl._build_boundary().active_cells()
            )


def test_kagome_triangle_uses_three_site_oblique_primitive_cell():
    """正三角 Kagome 盘采用 60° 三格点原始元胞，而非方形六格点超胞。"""
    _app, win, ctrl = _mk()
    ctrl.apply_document(template_document(
        "Kagome", connectivity="最近邻", boundary_kind="obc",
        nx=6, ny=6, shape="triangle",
    ))
    data = win.lattice_scene._data
    outline = data.boundary_outline
    assert len(outline) == 3
    assert len(win.panel.get_site_rows()) == 3
    a1, a2 = win.panel.get_cell_vectors()
    assert math.isclose(math.hypot(*a1), math.hypot(*a2), abs_tol=1e-10)
    assert math.isclose(
        (a1[0] * a2[0] + a1[1] * a2[1])
        / (math.hypot(*a1) * math.hypot(*a2)),
        0.5,
        abs_tol=1e-10,
    )
    active = ctrl._build_boundary().active_cells()
    assert tuple(sum(cy == row for _cx, cy in active) for row in range(6)) == (
        6, 5, 4, 3, 2, 1,
    )
    assert len(data.sites) == 3 * len(active)
    side_lengths = tuple(
        math.dist(outline[i], outline[(i + 1) % 3])
        for i in range(3)
    )
    assert all(math.isclose(length, side_lengths[0], rel_tol=0.0, abs_tol=1e-10)
               for length in side_lengths[1:])
    # The lattice boundary itself spans 10 units; the renderer adds a small,
    # symmetric site-clearance margin, while retaining an equilateral shape.
    assert 10.0 < side_lengths[0] < 12.0
    xs = [point[0] for point in outline]
    ys = [point[1] for point in outline]
    sample_x = [site[0] for site in data.sites]
    sample_y = [site[1] for site in data.sites]
    assert min(xs) <= min(sample_x) + 1e-9
    assert max(xs) >= max(sample_x) - 1e-9
    assert min(ys) <= min(sample_y) + 1e-9
    assert max(ys) >= max(sample_y) - 1e-9


def test_kagome_triangle_has_flat_physical_edges_not_a_jagged_mask():
    """Kagome triangular nanodisk keeps every rendered site inside 3 straight sides."""
    _app, win, ctrl = _mk()
    ctrl.apply_document(template_document(
        "Kagome", connectivity="最近邻", boundary_kind="obc",
        nx=6, ny=6, shape="triangle",
    ))
    outline = tuple(win.lattice_scene._data.boundary_outline)
    assert len(outline) == 3
    # The outline is counter-clockwise for the built-in primitive.  Check
    # signed half-plane membership for all basis sites, including the edge
    # sites that used to make a stepped/jagged-looking boundary.
    signs = []
    for index in range(3):
        start = outline[index]
        end = outline[(index + 1) % 3]
        signs.append(
            (end[0] - start[0], end[1] - start[1])
        )
    values = []
    for x, y, _label, _sub in win.lattice_scene._data.sites:
        values.append(tuple(
            dx * (y - outline[index][1])
            - dy * (x - outline[index][0])
            for index, (dx, dy) in enumerate(signs)
        ))
    assert all(
        all(value >= -1e-8 for value in row)
        or all(value <= 1e-8 for value in row)
        for row in values
    )
    # Straight edge segments remain exactly collinear; this guards against
    # accidentally returning a stepped cell-mask polygon later.
    side_lengths = tuple(
        math.dist(outline[index], outline[(index + 1) % 3])
        for index in range(3)
    )
    assert max(side_lengths) - min(side_lengths) <= 1e-8


def test_kagome_triangle_energy_target_exposes_degenerate_edge_state():
    """Kagome 三角盘命中简并能级时应直接展示边界代表，而非体内代表。"""
    _app, win, ctrl = _mk()
    ctrl.apply_document(template_document(
        "Kagome", connectivity="最近邻", boundary_kind="obc",
        nx=6, ny=6, shape="triangle",
    ))
    target = math.sqrt(2.0) - 1.0
    selected = win.wf_view.select_energy(target)

    assert selected is not None
    assert win.wf_view._requested_group_size == 2
    assert win.wf_view.selected_energy == pytest.approx(target, abs=1e-12)
    assert "同能级 2 态中优先边界代表" in win.wf_view.info.text()
    assert "边界局域态" in win.wf_view.info.text()


def test_triangle_scene_has_a_clear_physical_shape_badge():
    """The canvas labels a triangle without adding an interactive obstruction."""
    _app, win, ctrl = _mk()
    ctrl.apply_document(template_document(
        "Kagome", connectivity="最近邻", boundary_kind="obc",
        nx=5, ny=5, shape="triangle",
    ))
    badges = [
        item for item in win.lattice_scene.items()
        if item.data(0) == "finite-shape-badge"
    ]
    assert len(badges) == 1
    assert "正三角纳米盘" in badges[0].toPlainText()
    assert "平直等边边界" in badges[0].toPlainText()
    assert badges[0].zValue() > 1.0
    assert not (badges[0].flags() & QGraphicsItem.ItemIsMovable)


def test_finite_nonrectangular_scene_does_not_draw_cells_outside_the_sample():
    """Finite masks show one primitive-cell cue plus one physical silhouette."""
    _app, win, ctrl = _mk()
    ctrl.apply_document(template_document(
        "Kagome", connectivity="最近邻", boundary_kind="obc",
        nx=4, ny=4, shape="triangle",
    ))
    win.lattice_mode_btn.setChecked(False)
    QApplication.processEvents()
    # One reference primitive-cell polygon and one closed finite-shape
    # silhouette are drawn; the remaining DTO polygons stay available for
    # editing/export but no longer clutter the physical canvas.
    polygons = [
        item for item in win.lattice_scene.items()
        if isinstance(item, QGraphicsPolygonItem)
    ]
    assert len(polygons) == 2
    assert sum(item.data(0) == "finite-shape-outline" for item in polygons) == 1


def test_nontriangular_scene_outlines_enclose_every_rendered_basis_site():
    """Disk/hexagon outlines must include multi-site basis overhangs."""
    _app, win, ctrl = _mk()
    for template_name in (
        "NP", "SC", "空白自定义", "方格", "一维链", "蜂窝", "三角",
        "Kagome", "SSH", "Haldane",
    ):
        for shape in ("disk", "hexagon"):
            ctrl.apply_document(template_document(
                template_name, connectivity="最近邻", boundary_kind="obc",
                nx=6, ny=6, shape=shape,
            ))
            boundary = ctrl._build_boundary()
            frame = boundary._physical_frame()
            assert frame is not None, (template_name, shape)
            _u0, _u1, _v0, _v1, ux, uy, vx, vy = frame

            def project(point):
                x, y = point
                return x * ux + y * uy, x * vx + y * vy

            sample = [project((x, y))
                      for x, y, *_ in win.lattice_scene._data.sites]
            outline = [project(point)
                       for point in win.lattice_scene._data.boundary_outline]
            center = (
                sum(u for u, _v in outline) / len(outline),
                sum(v for _u, v in outline) / len(outline),
            )
            if shape == "disk":
                radius = max(math.hypot(u - center[0], v - center[1])
                             for u, v in outline)
                assert all(
                    math.hypot(u - center[0], v - center[1]) <= radius + 1e-8
                    for u, v in sample
                ), (template_name, shape)
            else:
                sqrt3 = math.sqrt(3.0)
                radius = max(
                    max(2.0 * abs(v - center[1]) / sqrt3,
                        abs(u - center[0]) + abs(v - center[1]) / sqrt3)
                    for u, v in outline
                )
                assert all(
                    max(2.0 * abs(v - center[1]) / sqrt3,
                        abs(u - center[0]) + abs(v - center[1]) / sqrt3)
                    <= radius + 1e-8
                    for u, v in sample
                ), (template_name, shape)


def test_all_presets_and_finite_shapes_build_hermitian_matrices():
    """Pure-model audit covering every shipped preset and boundary shape."""
    from hamivisualizer.model.templates import TEMPLATE_NAMES

    cases = 0
    for template_name in TEMPLATE_NAMES:
        for boundary_kind in ("semi", "obc"):
            shapes = ("rectangle",) if boundary_kind == "semi" else BOUNDARY_SHAPES
            for shape in shapes:
                document = validate_model_dict(template_document(
                    template_name, nx=4, ny=4, boundary_kind=boundary_kind,
                    shape=shape, connectivity="最近邻+次近邻",
                ))
                cell = document["cell"]
                if "a1" in cell:
                    lattice = Lattice(
                        [Site(i, s["x"], s["y"], s.get("sublattice"))
                         for i, s in enumerate(document["sites"])],
                        a1=tuple(cell["a1"]), a2=tuple(cell["a2"]),
                    )
                else:
                    lattice = Lattice(
                        [Site(i, s["x"], s["y"], s.get("sublattice"))
                         for i, s in enumerate(document["sites"])],
                        Lx=cell["Lx"], Ly=cell["Ly"],
                    )
                hops = [HoppingTerm(
                    h["name"], h["from_site"], h["to_site"],
                    tuple(h["cell_offset"]), 1.0,
                    h.get("phase_mode", "none"), 0.0,
                    h.get("phase_sign", 1),
                ) for h in document["hops"]]
                boundary = Boundary(
                    BoundaryKind.SEMI if boundary_kind == "semi" else BoundaryKind.OBC,
                    NX=4, NY=4, shape=shape,
                )
                result = HamiltonianBuilder(lattice, hops, boundary).build()
                if boundary_kind == "semi":
                    for kx in (-1.1, 0.0, 0.73):
                        matrix = np.asarray(result.to_semi(kx), dtype=complex)
                        assert np.allclose(matrix, matrix.conj().T, atol=1e-9)
                    assert result.skipped.get("miss", 0) == 0
                else:
                    matrix = np.asarray(result.H, dtype=complex)
                    assert np.allclose(matrix, matrix.conj().T, atol=1e-9)
                cases += 1
    assert cases == len(TEMPLATE_NAMES) * (1 + len(BOUNDARY_SHAPES))


def test_all_presets_connectivity_and_boundary_combinations_are_hermitian():
    """Regression matrix for every shipped preset/connection/shape path.

    The broader audit intentionally mirrors the controller's real boundary
    construction: non-rectangular OBC masks receive the physical cell vectors
    used by the renderer.  This catches regressions that a single default
    nearest-neighbour rectangle cannot expose (for example a missing
    inter-cell bond in only one connectivity mode).
    """
    from hamivisualizer.model.templates import TEMPLATE_NAMES

    combinations = 0
    for template_name in TEMPLATE_NAMES:
        for boundary_kind in ("semi", "obc"):
            shapes = ("rectangle",) if boundary_kind == "semi" else BOUNDARY_SHAPES
            for shape in shapes:
                for connectivity in ("仅格点", "最近邻", "最近邻+次近邻"):
                    document = validate_model_dict(template_document(
                        template_name, nx=4, ny=4,
                        boundary_kind=boundary_kind, shape=shape,
                        connectivity=connectivity,
                    ))
                    cell = document["cell"]
                    site_rows = document["sites"]
                    sites = [Site(
                        index, row["x"], row["y"], row.get("sublattice") or None,
                    ) for index, row in enumerate(site_rows)]
                    if "a1" in cell:
                        lattice = Lattice(
                            sites, a1=tuple(cell["a1"]), a2=tuple(cell["a2"]),
                        )
                        vectors = (tuple(cell["a1"]), tuple(cell["a2"]))
                    else:
                        lattice = Lattice(
                            sites, Lx=cell["Lx"], Ly=cell["Ly"],
                        )
                        vectors = None
                    params = document["params"]
                    hops = []
                    for row in document["hops"]:
                        amplitude = evaluate_expression(row["amplitude"], params)
                        phase = evaluate_expression(row["phase"], params)
                        hops.append(HoppingTerm(
                            row["name"], row["from_site"], row["to_site"],
                            tuple(row["cell_offset"]), amplitude,
                            row.get("phase_mode", "none"), phase,
                            row.get("phase_sign", 1),
                        ))
                    boundary_data = document["boundary"]
                    boundary = Boundary(
                        BoundaryKind.SEMI if boundary_kind == "semi"
                        else BoundaryKind.OBC,
                        NX=boundary_data["NX"], NY=boundary_data["NY"],
                        shape=shape,
                        shape_aspect=boundary_data.get("shape_aspect", 1.0),
                        shape_vectors=(vectors if boundary_kind == "obc"
                                       and shape != "rectangle" else None),
                    )
                    result = HamiltonianBuilder(
                        lattice, hops, boundary, document["order"],
                    ).build()
                    assert result.Nat > 0, (template_name, boundary_kind, shape, connectivity)
                    if boundary_kind == "semi":
                        matrices = [result.to_semi(kx) for kx in (0.0, 0.371, math.pi)]
                    else:
                        matrices = [result.H]
                    for matrix in matrices:
                        numeric = np.asarray(matrix, dtype=complex)
                        assert numeric.shape == (result.Nat, result.Nat)
                        assert np.isfinite(numeric).all()
                        assert np.allclose(numeric, numeric.conj().T, atol=1e-8), (
                            template_name, boundary_kind, shape, connectivity,
                        )
                    combinations += 1
    expected = len(TEMPLATE_NAMES) * (1 + len(BOUNDARY_SHAPES)) * 3
    assert combinations == expected


def test_kagome_nearest_neighbour_template_has_fourfold_coordination():
    """The visible red Kagome editing skeleton must match the physical graph."""
    document = template_document("Kagome", connectivity="最近邻")
    assert len(document["sites"]) == 6
    assert len(document["hops"]) == 12
    coordination = [0] * len(document["sites"])
    for hop in document["hops"]:
        coordination[int(hop["from_site"])] += 1
        coordination[int(hop["to_site"])] += 1
    assert coordination == [4, 4, 4, 4, 4, 4]
