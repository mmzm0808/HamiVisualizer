"""诊断晶格跃迁连线：渲染蜂窝/方格/三角/Kagome 的晶格视图 + 检查边数据。"""

import os
import sys
from pathlib import Path

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

OUTPUT_DIR = Path(__file__).resolve().parents[2] / ".codex-artifacts" / "diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def dump_scene(title, data):
    print(f"\n===== {title} =====")
    print(f"sites={len(data.sites)} edges={len(data.edges)} ghost_edges={len(data.ghost_edges)} ghost={len(data.ghost)}")
    # 检查边异常：重复 / 自环 / 距离异常
    from collections import Counter
    from hamivisualizer.view.rendermodel import LatticeSceneData
    pos = {i: (s[0], s[1]) for i, s in enumerate(data.sites)}
    import math
    lens = Counter()
    dup = Counter((a, b) for (a, b, k) in data.edges)
    for (a, b, k) in data.edges:
        d = math.hypot(pos[a][0] - pos[b][0], pos[a][1] - pos[b][1])
        lens[round(d, 2)] += 1
        if d < 1e-6:
            print(f"  SELF-LOOP edge {a}->{b} kind={k}")
    for (ab, c) in dup.items():
        if c > 1:
            print(f"  DUP edge {ab} x{c}")
    print("  edge length histogram:", dict(sorted(lens.items())))
    # ghost_edges 距离分布
    glens = Counter()
    for (x1, y1, x2, y2, k) in data.ghost_edges:
        glens[round(math.hypot(x2 - x1, y2 - y1), 2)] += 1
    print("  ghost edge length histogram:", dict(sorted(glens.items())))


def main():
    w = MainWindow()
    c = ViewController(w, connect_actions=False)
    w.controller = c
    c.load_preset("NP")

    from hamivisualizer.model.templates import template_document
    configs = [
        ("蜂窝-NN", "蜂窝", "最近邻"),
        ("蜂窝-NN+NNN", "蜂窝", "最近邻+次近邻"),
        ("三角", "三角", "最近邻"),
        ("Kagome", "Kagome", "最近邻"),
        ("方格", "方格", "最近邻"),
    ]
    for label, name, conn in configs:
        doc = template_document(name, connectivity=conn)
        c.apply_document(doc)
        app.processEvents()
        data = w.lattice_scene._data
        dump_scene(label, data)
        # 渲染晶格页
        w.tabs.setCurrentIndex(2)
        app.processEvents()
        pix = w.lattice_gv.grab()
        output = OUTPUT_DIR / f"diag_lattice_{label}.png"
        pix.save(str(output))
        print(f"  saved {output}")

    w.set_dirty(False)
    w.close()


main()
