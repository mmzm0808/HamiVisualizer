"""通过真实控制器调试 _build_lattice_scene 边匹配。"""

import os, math, sys
from pathlib import Path
os.environ["QT_QPA_PLATFORM"] = "offscreen"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from PySide6.QtWidgets import QApplication

app = QApplication([])
app.setStyle("Fusion")

import hamivisualizer.controller as C
from hamivisualizer.view.rendermodel import LatticeSceneData

_orig = C._build_lattice_scene
def traced(lattice, hops, boundary, res, *, labels_bottom_up=True):
    q = lambda v: round(float(v), 6)
    positions = list(res.positions)
    print("positions:", [tuple(round(v, 3) for v in p) for p in positions])
    print("origin:", res.origin)
    r_of = [o[0] for o in res.origin]
    print("r_of:", r_of)
    central_coord_to_idx = {(q(x), q(y)): i for i, (x, y) in enumerate(positions)}
    a1x, a1y = lattice.a1
    semi = boundary.kind is C.BoundaryKind.SEMI
    ghost_layers = max(1, max((abs(h.cell_offset[0]) for h in hops), default=1))
    central_coords = {(q(x), q(y)) for x, y in positions}
    display_source = {}
    for source, (x, y) in enumerate(positions):
        display_source[(q(x), q(y))] = source
        for layer in range(1, ghost_layers + 1):
            display_source[(q(x - layer * a1x), q(y - layer * a1y))] = source
            display_source[(q(x + layer * a1x), q(y + layer * a1y))] = source
    seen = {}
    for source, (x, y) in enumerate(positions):
        keys = [(q(x), q(y))]
        for layer in range(1, ghost_layers + 1):
            keys.append((q(x - layer * a1x), q(y - layer * a1y)))
            keys.append((q(x + layer * a1x), q(y + layer * a1y)))
        for k in keys:
            if k in seen and seen[k] != source:
                print(f"CONFLICT key {k}: source {seen[k]} 与 {source}")
            seen[k] = source
    print("ghost_layers:", ghost_layers, "ghost_coords computed...")
    # A2-B2 / A4-B4 追踪
    A2, B2 = positions[2], positions[3]
    A4, B4 = positions[6], positions[7]
    for name, pa, pb in [("A2-B2", A2, B2), ("A4-B4", A4, B4)]:
        ka, kb = (q(pa[0]), q(pa[1])), (q(pb[0]), q(pb[1]))
        print(f"{name}: a={ka}->src{display_source.get(ka)} b={kb}->src{display_source.get(kb)}")
        print(f"   in_central: a={ka in central_coords} b={kb in central_coords}; idx a={central_coord_to_idx.get(ka)} b={central_coord_to_idx.get(kb)}")
    # A->B 胞内 hop 追踪
    for h in hops:
        if h.cell_offset != (0, 0):
            continue
        d = h.displacement(lattice)
        print(f"hop {h.from_site}->{h.to_site} off={h.cell_offset} d={tuple(round(v,6) for v in d)}:")
        for (x0, y0), source_idx in display_source.items():
            if source_idx >= len(r_of) or r_of[source_idx] != h.from_site:
                continue
            target = (q(x0 + d[0]), q(y0 + d[1]))
            skip1 = target not in display_source
            skip2 = target == (x0, y0)
            a = (q(x0), q(y0)); bb = target
            if source_idx in (2, 6):
                print(f"   [src={source_idx}] a={a} target={bb} target_in_ds={not skip1} selfloop={skip2} r_of={r_of[source_idx]}")
    return _orig(lattice, hops, boundary, res, labels_bottom_up=labels_bottom_up)

C._build_lattice_scene = traced

from hamivisualizer.controller import ViewController
from hamivisualizer.view.main_window import MainWindow
from hamivisualizer.model.templates import template_document

w = MainWindow()
c = ViewController(w, connect_actions=False)
w.controller = c
c.load_preset("NP")
doc = template_document("蜂窝", connectivity="最近邻+次近邻", boundary_kind="semi")
c.apply_document(doc)
w.panel.nx_spin.setValue(4)
w.panel.ny_spin.setValue(4)
c.rebuild()
app.processEvents()
w.set_dirty(False)
w.close()
