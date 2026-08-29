"""复现用户截图场景：蜂窝 NN+NNN, semi, NX=4, NY=4, t2=0.2, 渲染晶格页。"""

import os
from pathlib import Path
import sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

app = QApplication([])
app.setStyle("Fusion")
for _f in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simsun.ttc"):
    if os.path.exists(_f):
        fid = QFontDatabase.addApplicationFont(_f)
        fam = QFontDatabase.applicationFontFamilies(fid)
        if fam:
            app.setFont(QFont(fam[0], 10))
            break

from hamivisualizer.controller import ViewController
from hamivisualizer.view.main_window import MainWindow
from hamivisualizer.model.templates import template_document

OUTPUT_DIR = Path(__file__).resolve().parents[2] / ".codex-artifacts" / "diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

w = MainWindow()
c = ViewController(w, connect_actions=False)
w.controller = c
c.load_preset("NP")

doc = template_document("蜂窝", connectivity="最近邻+次近邻", boundary_kind="semi")
c.apply_document(doc)
w.panel.nx_spin.setValue(4)
w.panel.ny_spin.setValue(4)
# t2 = 0.2
c.rebuild()
app.processEvents()

d = w.lattice_scene._data
pos = {i: (s[0], s[1]) for i, s in enumerate(d.sites)}
import math
print(f"sites={len(d.sites)} edges={len(d.edges)} ghost_edges={len(d.ghost_edges)} ghost={len(d.ghost)}")
print("--- 中心边 (i,j,kind,dx,dy,len) ---")
for (i, j, k) in sorted(d.edges):
    dx = pos[j][0]-pos[i][0]; dy = pos[j][1]-pos[i][1]
    print(f"  {i}({pos[i][0]:+.2f},{pos[i][1]:+.2f}) -> {j}({pos[j][0]:+.2f},{pos[j][1]:+.2f}) {k} d=({dx:+.2f},{dy:+.2f}) len={math.hypot(dx,dy):.3f}")
print("--- ghost_edges ---")
for (x1, y1, x2, y2, k) in d.ghost_edges:
    print(f"  ({x1:+.2f},{y1:+.2f})->({x2:+.2f},{y2:+.2f}) {k} len={math.hypot(x2-x1, y2-y1):.3f}")

w._set_theme_mode("dark")
w.tabs.setCurrentIndex(2)
app.processEvents()
w.resize(1569, 1079)
w.show()
app.processEvents()
w.grab().save(str(OUTPUT_DIR / "repro_user.png"))
print(f"saved {OUTPUT_DIR / 'repro_user.png'}")
w.set_dirty(False)
w.close()
