"""离屏渲染浅色/深色主题界面截图（验证菜单栏可见性与深色覆盖）。

用法: python tools/render_theme_shots.py
输出:  .codex-artifacts/diagnostics/theme_light.png / theme_dark.png
"""

import os
import sys
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")

from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication

app = QApplication([])
app.setStyle("Fusion")

# 与 main.py 一致：加载系统中文字体，避免离屏环境缺字渲染成方块
for _f in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyhbd.ttc", "C:/Windows/Fonts/simsun.ttc"):
    if os.path.exists(_f):
        fid = QFontDatabase.addApplicationFont(_f)
        families = QFontDatabase.applicationFontFamilies(fid)
        if families:
            app.setFont(QFont(families[0], 10))
            break

from hamivisualizer.controller import ViewController
from hamivisualizer.view.main_window import MainWindow

# 构建一个带内容的窗口：NP 半无限 + 蜂窝模板，切到矩阵+晶格页
w = MainWindow()
c = ViewController(w, connect_actions=False)
w.controller = c
c.load_preset("NP")
w.enable_workspace_mode(c)
# 加一个蜂窝模型标签，展示模型栏
from hamivisualizer.model.templates import template_document
w._add_session("蜂窝", template_document("蜂窝", connectivity="最近邻+次近邻"))
app.processEvents()
w.tabs.setCurrentIndex(0)
c.fit_all(force=True)
app.processEvents()

w.resize(1280, 860)
w.show()


def snap(path, mode):
    w._set_theme_mode(mode)
    app.processEvents()
    pix = w.grab()
    ok = pix.save(path)
    print(("SAVED " if ok else "FAIL  ") + path, pix.width(), "x", pix.height())


output_dir = Path(__file__).resolve().parents[1] / ".codex-artifacts" / "diagnostics"
output_dir.mkdir(parents=True, exist_ok=True)
snap(str(output_dir / "theme_light.png"), "light")
snap(str(output_dir / "theme_dark.png"), "dark")
