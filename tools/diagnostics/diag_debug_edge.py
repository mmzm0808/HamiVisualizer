"""调试：蜂窝 semi NX=4 NY=4 为何 2-3 / 6-7 中心边丢失。"""

import os, math, sys
from pathlib import Path
os.environ["QT_QPA_PLATFORM"] = "offscreen"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from PySide6.QtWidgets import QApplication
app = QApplication([])

from hamivisualizer.model.templates import template_document
from hamivisualizer.model.lattice import Lattice, Site
from hamivisualizer.model.boundary import Boundary, BoundaryKind
from hamivisualizer.model.hamiltonian import HamiltonianBuilder
from hamivisualizer.model.hopping import HoppingTerm

def _hop_from_dict(h):
    return HoppingTerm(
        name=h["name"], from_site=h["from_site"], to_site=h["to_site"],
        cell_offset=tuple(h["cell_offset"]), amplitude=h["amplitude"],
        phase_mode=h.get("phase_mode", "none"), phase=h.get("phase", 0.0),
        phase_sign=h.get("phase_sign", 1),
    )

doc = template_document("蜂窝", connectivity="最近邻+次近邻", boundary_kind="semi")
print("hops:", doc["hops"])
print("cell:", doc["cell"])
lat = Lattice(sites=[Site(i, s["x"], s["y"], s.get("sublattice")) for i, s in enumerate(doc["sites"])],
              a1=tuple(doc["cell"]["a1"]), a2=tuple(doc["cell"]["a2"]))
print("lat.N:", lat.N, "a1:", lat.a1, "a2:", lat.a2)
b = Boundary(BoundaryKind.SEMI, 4, 4)
hops_obj = [_hop_from_dict(h) for h in doc["hops"]]
params = dict(doc["params"])
builder = HamiltonianBuilder(lat, hops=hops_obj, boundary=b, order="cell")
res = builder.build()

positions = list(res.positions)
print("\npositions:", [tuple(round(v, 3) for v in p) for p in positions])
print("origin:", res.origin)
r_of = [o[0] for o in res.origin]
print("r_of:", r_of)

q = lambda v: round(float(v), 6)
central_coord_to_idx = {(q(x), q(y)): i for i, (x, y) in enumerate(positions)}
a1x, a1y = lat.a1
ghost_layers = max(1, max((abs(h.cell_offset[0]) for h in hops_obj), default=1))
print("ghost_layers:", ghost_layers)

display_source = {}
for source, (x, y) in enumerate(positions):
    display_source[(q(x), q(y))] = source
    for layer in range(1, ghost_layers + 1):
        display_source[(q(x - layer * a1x), q(y - layer * a1y))] = source
        display_source[(q(x + layer * a1x), q(y + layer * a1y))] = source

central_coords = {(q(x), q(y)) for x, y in positions}
# 检查 ghost 展开键冲突
seen = {}
for source, (x, y) in enumerate(positions):
    keys = [(q(x), q(y))]
    for layer in range(1, ghost_layers + 1):
        keys.append((q(x - layer * a1x), q(y - layer * a1y)))
        keys.append((q(x + layer * a1x), q(y + layer * a1y)))
    for k in keys:
        if k in seen and seen[k] != source:
            print(f"CONFLICT key {k}: source {seen[k]} 与 {source} 重叠")
        seen[k] = source

A2 = positions[2]; B2 = positions[3]
A4 = positions[6]; B4 = positions[7]
for name, pa, pb in [("A2-B2", A2, B2), ("A4-B4", A4, B4)]:
    ka, kb = (q(pa[0]), q(pa[1])), (q(pb[0]), q(pb[1]))
    print(f"{name}: a={ka}->{display_source.get(ka)}  b={kb}->{display_source.get(kb)}")
    print(f"   a in central={ka in central_coords}, b in central={kb in central_coords}")
    print(f"   idx[a]={central_coord_to_idx.get(ka)}, idx[b]={central_coord_to_idx.get(kb)}")

h0 = hops_obj[0]
d = h0.displacement(lat)
print("\nhop0:", h0.name, "from:", h0.from_site, "to:", h0.to_site, "off:", h0.cell_offset, "d:", d)
print("--- A->B (0,0) 全部匹配 ---")
for (x0, y0), source_idx in display_source.items():
    if r_of[source_idx] != h0.from_site:
        continue
    target = (q(x0 + d[0]), q(y0 + d[1]))
    if target not in display_source or target == (x0, y0):
        continue
    a = (q(x0), q(y0)); bb = target
    print(f"  src={source_idx} a={a} b={bb} out_a={a not in central_coords} out_b={bb not in central_coords}")
