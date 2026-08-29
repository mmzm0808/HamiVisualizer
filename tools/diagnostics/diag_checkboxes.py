"""渲染偏好设置对话框 + 左侧显示区复选框（暗色），检查勾选视觉。"""

import os
from pathlib import Path
import sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import QApplication, QDialog
from PySide6.QtCore import Qt

app = QApplication([])
app.setStyle("Fusion")
for _f in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simsun.ttc"):
    if os.path.exists(_f):
        fid = QFontDatabase.addApplicationFont(_f)
        fam = QFontDatabase.applicationFontFamilies(fid)
        if fam:
            app.setFont(QFont(fam[0], 10))
            break

from hamivisualizer.model.workspace import Preferences
from hamivisualizer.view.dialogs import PreferencesDialog
from hamivisualizer.view.theme import DARK, resolve_theme, app_palette, app_stylesheet
from PySide6.QtWidgets import QCheckBox, QVBoxLayout, QWidget

OUTPUT_DIR = Path(__file__).resolve().parents[2] / ".codex-artifacts" / "diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

theme = DARK
app.setPalette(app_palette(theme))
app.setStyleSheet(app_stylesheet(theme, 1.0))

d = PreferencesDialog(Preferences())
d.resize(560, 420)
d.show()
app.processEvents()
# 勾选一个复选框
for c in d.findChildren(QCheckBox):
    c.setChecked(True)
app.processEvents()
d.grab().save(str(OUTPUT_DIR / "diag_prefs_dark.png"))
print(f"saved {OUTPUT_DIR / 'diag_prefs_dark.png'}")

# 左侧显示区复选框
w = QWidget()
lay = QVBoxLayout(w)
c1 = QCheckBox("符号模式")
c2 = QCheckBox("智能识别标签")
c3 = QCheckBox("显示高级列（元胞偏移 / 相位模式 / 符号）")
c2.setChecked(True)
for c in (c1, c2, c3):
    lay.addWidget(c)
w.resize(360, 160)
w.show()
app.processEvents()
w.grab().save(str(OUTPUT_DIR / "diag_checkboxes_dark.png"))
print(f"saved {OUTPUT_DIR / 'diag_checkboxes_dark.png'}")
d.close(); w.close()
