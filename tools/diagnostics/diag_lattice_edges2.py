"""第二诊断：组合视图 + 大尺寸蜂窝，检查连线是否出现真实错连。"""

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

# 大尺寸蜂窝 NN+NNN 半无限 (NY=8)
doc = template_document("蜂窝", connectivity="最近邻+次近邻", boundary_kind="semi")
c.apply_document(doc)
w.panel.ny_spin.setValue(8)
c.rebuild()
app.processEvents()
d = w.lattice_scene._data
print(f"蜂窝NY=8: sites={len(d.sites)} edges={len(d.edges)} ghost_edges={len(d.ghost_edges)} ghost={len(d.ghost)}")
# 检查跨胞 NNN 虚影键是否横穿中心
import math
pos = {i: (s[0], s[1]) for i, s in enumerate(d.sites)}
cx = [p[0] for p in pos.values()]; cy = [p[1] for p in pos.values()]
x0, x1, y0, y1 = min(cx), max(cx), min(cy), max(cy)
cross = 0
for (a, b, k) in d.edges:
    dx = pos[b][0]-pos[a][0]; dy = pos[b][1]-pos[a][1]
    if math.hypot(dx, dy) > 1.1 and (dx < -0.5 or dx > 0.5):
        cross += 1
print(f"跨胞长边(>1.1且dx大): {cross}")
w.tabs.setCurrentIndex(0)
app.processEvents()
w.resize(1280, 860); w.show()
w._set_theme_mode("dark")
app.processEvents()
w.grab().save(str(OUTPUT_DIR / "diag_combined_ny8.png"))
print(f"saved {OUTPUT_DIR / 'diag_combined_ny8.png'}")

# 方格 OBC 大尺寸
doc2 = template_document("方格", connectivity="最近邻", boundary_kind="obc")
c.apply_document(doc2)
w.panel.nx_spin.setValue(6); w.panel.ny_spin.setValue(6)
c.rebuild()
app.processEvents()
d2 = w.lattice_scene._data
print(f"方格OBC6x6: sites={len(d2.sites)} edges={len(d2.edges)} ghost_edges={len(d2.ghost_edges)}")
w.tabs.setCurrentIndex(2)
app.processEvents()
w.lattice_gv.grab().save(str(OUTPUT_DIR / "diag_square_obc.png"))
print(f"saved {OUTPUT_DIR / 'diag_square_obc.png'}")
w.set_dirty(False)
w.close()
