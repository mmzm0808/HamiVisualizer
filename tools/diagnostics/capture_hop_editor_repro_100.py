from __future__ import annotations
import os
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtWidgets import QApplication, QStyleFactory
from hamivisualizer.main import _configure_font
from hamivisualizer.controller import ViewController
from hamivisualizer.model.templates import template_document
from hamivisualizer.view.main_window import MainWindow

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '.codex-artifacts' / 'screenshots' / 'hop-editor-repro-100-20260831'
OUT.mkdir(parents=True, exist_ok=True)
app = QApplication.instance() or QApplication([])
app.setStyle(QStyleFactory.create('Fusion'))
_configure_font(app)
for label, doc in [
    ('ssh-semi', template_document('SSH', nx=4, ny=4, boundary_kind='semi', connectivity='最近邻')),
    ('blank-two-duplicate', {
        **template_document('空白自定义', nx=4, ny=4, boundary_kind='semi', connectivity='仅格点'),
        'sites': [{'x': 0.2, 'y': 0.0, 'sublattice': 'A'}, {'x': 0.6, 'y': 0.8, 'sublattice': 'A'}],
        'hops': [
            {'name': 't', 'from_site': 0, 'to_site': 1, 'off_x': 0, 'off_y': 0, 'amplitude': 't', 'phase': 'none', 'phase_sign': 1},
            {'name': 't', 'from_site': 0, 'to_site': 1, 'off_x': 0, 'off_y': 0, 'amplitude': 't', 'phase': 'none', 'phase_sign': 1},
        ], 'params': {'t': 2.0},
    }),
]:
    win = MainWindow(); ctrl = ViewController(win); ctrl.apply_document(doc)
    win.resize(1368, 900); win.show(); app.processEvents(); win.lattice_mode_btn.setChecked(True); app.processEvents(); ctrl.fit_all(force=True); app.processEvents()
    win.grab().save(str(OUT / f'{label}-dark-100.png'))
    print(label, 'proxies=', len(win.lattice_scene._edit_proxies), 'links=', len(win.lattice_scene._edit_leader_links), 'hops=', len(win.lattice_scene._edit_hops))
    win.close(); win.deleteLater(); app.processEvents()
print('output=', OUT)
