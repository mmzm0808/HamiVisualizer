"""HamVisualizerPy 主入口.

用法:  python -m hamivisualizer.main   (在项目根目录)
"""

from __future__ import annotations

import sys
from importlib import metadata
from pathlib import Path


def _installed_version(distribution: str) -> str | None:
    """Return a package version without importing a native extension.

    Startup diagnostics must run *before* importing PySide6.  Looking up the
    wheel metadata avoids triggering the native Qt/NumPy loader when the
    environment is known to be incomplete or outside the tested range.
    """
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None
    except (ValueError, TypeError, OSError):
        # A partially interrupted installation should produce the same
        # actionable message as a missing distribution rather than a raw
        # importlib traceback.
        return None


def _check_runtime_compatibility():
    """Fail early with actionable dependency guidance.

    The tested desktop stack is intentionally narrow: NumPy 1.x avoids the
    ABI mismatch seen when PySide6/shiboken wheels built against NumPy 1.x are
    loaded in a NumPy 2.x environment, while PySide6/SymPy ranges keep the
    rendering and symbolic paths reproducible.  This function performs only
    metadata checks and therefore remains safe before any Qt native module is
    imported.
    """
    requirements = (
        ("numpy", "NumPy", "numpy>=1.26,<2"),
        ("PySide6", "PySide6", "PySide6>=6.6,<6.7"),
        ("sympy", "SymPy", "sympy>=1.14,<2"),
    )
    versions: dict[str, str | None] = {
        distribution: _installed_version(distribution)
        for distribution, _label, _spec in requirements
    }
    missing = [label for distribution, label, _spec in requirements
               if versions[distribution] is None]
    if missing:
        names = "、".join(missing)
        raise RuntimeError(
            f"缺少运行依赖：{names}。请在当前环境执行：\n"
            "  python -m pip install -r requirements.txt"
        )

    numpy_version = versions["numpy"] or ""
    try:
        numpy_major = int(numpy_version.split(".", 1)[0])
    except (TypeError, ValueError):
        numpy_major = 99
    if numpy_major >= 2:
        raise RuntimeError(
            "NumPy " + numpy_version + " 与已测试的 PySide6 6.6.x 栈存在 ABI 不兼容。"
            "请在当前环境执行：\n"
            "  python -m pip install \"numpy>=1.26,<2\"\n"
            "也可以从 requirements.txt 创建一个全新的环境。"
        )

    # Keep the version check deliberately small and dependency-free.  A
    # malformed or future major version should be blocked rather than allowed
    # to reach a native import whose failure may be an access violation.
    pyside_version = versions["PySide6"] or ""
    try:
        pyside_parts = tuple(int(part) for part in pyside_version.split(".")[:2])
    except (TypeError, ValueError):
        pyside_parts = (99, 99)
    if pyside_parts != (6, 6):
        raise RuntimeError(
            f"当前 PySide6 版本为 {pyside_version}，项目已验证范围为 6.6.x。"
            "请在当前环境执行：\n"
            "  python -m pip install \"PySide6>=6.6,<6.7\""
        )

    sympy_version = versions["sympy"] or ""
    try:
        sympy_major = int(sympy_version.split(".", 1)[0])
    except (TypeError, ValueError):
        sympy_major = 99
    if sympy_major >= 2 or sympy_major < 1:
        raise RuntimeError(
            f"当前 SymPy 版本为 {sympy_version}，项目已验证范围为 1.14.x。"
            "请在当前环境执行：\n"
            "  python -m pip install \"sympy>=1.14,<2\""
        )

    return versions


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
