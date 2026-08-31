"""HamVisualizerPy 主入口.

用法:  python -m hamivisualizer.main   (在项目根目录)
"""

from __future__ import annotations

import sys
from pathlib import Path


def _check_runtime_compatibility():
    """Fail early with an actionable message instead of an ABI traceback."""
    import numpy as np

    major = int(np.__version__.split(".", 1)[0])
    if major >= 2:
        raise RuntimeError(
            "NumPy " + np.__version__ + " is ABI-incompatible with the tested "
            "PySide6 6.6.x stack. Run this command in the active environment:\n"
            "  python -m pip install \"numpy>=1.26,<2\"\n"
            "Alternatively, create a clean environment from requirements.txt."
        )


def _setup_path():
    """脚本方式直接运行 (python main.py) 时把项目根加入 sys.path."""
    if __package__ in (None, ""):
        import os

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if root not in sys.path:
            sys.path.insert(0, root)


def _configure_font(app):
    """Load stable UI and math fonts explicitly when Qt finds no system fonts."""
    from PySide6.QtGui import QFont, QFontDatabase

    # Noto Sans SC is preferred when available because it has complete CJK
    # coverage in Qt's off-screen/Fusion path.  Some Windows installations
    # expose Microsoft YaHei in the font database but still fall back to tofu
    # glyphs when the application runs without a native windowing backend.
    preferred = ("Noto Sans SC", "Microsoft YaHei UI", "Microsoft YaHei", "Noto Sans CJK SC", "SimSun")
    installed = set(QFontDatabase.families())
    family = next((name for name in preferred if name in installed), None)
    if family is None and sys.platform == "win32":
        font_dir = Path("C:/Windows/Fonts")
        # Register all candidates before selecting one; loading only the first
        # file (historically msyh.ttc) can leave CJK glyph fallback unreliable
        # in headless Qt even though the family reports valid metrics.
        for filename in (
            "Noto Sans SC (TrueType).otf",
            "NotoSansSC-VF.ttf",
            "msyh.ttc",
            "msyhbd.ttc",
            "simsun.ttc",
        ):
            path = font_dir / filename
            if not path.exists():
                continue
            font_id = QFontDatabase.addApplicationFont(str(path))
            families = QFontDatabase.applicationFontFamilies(font_id)
            if family is None:
                for candidate in preferred:
                    if candidate in families:
                        family = candidate
                        break
                if family is None and families:
                    family = families[0]
    if family:
        app.setFont(QFont(family, 10))
    # Matrix expressions use a dedicated math face. Registering the system
    # font makes off-screen checks deterministic without copying proprietary
    # font files into the repository.
    if sys.platform == "win32":
        for filename in ("cambria.ttc", "DejaVuSerif.ttf"):
            path = Path("C:/Windows/Fonts") / filename
            if path.exists():
                QFontDatabase.addApplicationFont(str(path))


def main():
    _setup_path()
    try:
        _check_runtime_compatibility()
    except RuntimeError as exc:
        print(f"HamiVisualizer startup blocked:\n{exc}", file=sys.stderr)
        return 2
    from PySide6.QtWidgets import QApplication, QStyleFactory

    if __package__ in (None, ""):
        from hamivisualizer.controller import ViewController
        from hamivisualizer.view.main_window import MainWindow
    else:
        from .controller import ViewController
        from .view.main_window import MainWindow

    app = QApplication(sys.argv)
    # Use Qt's cross-platform Fusion base before applying the app stylesheet;
    # it avoids the dated native-widget mix that makes a PySide UI feel like Tk.
    app.setStyle(QStyleFactory.create("Fusion"))
    app.setApplicationName("HamiVisualizer")
    app.setOrganizationName("HamiVisualizer")
    app.setApplicationVersion("0.4.0")
    _configure_font(app)
    win = MainWindow()
    # The main application routes global menu actions through the workspace;
    # individual controller tests can still use the legacy direct connections.
    win.controller = ViewController(win, connect_actions=False)
    win.controller.load_preset("NP")
    win.enable_workspace_mode(win.controller)
    win.resize(1360, 900)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
