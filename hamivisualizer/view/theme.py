"""应用主题（浅色 / 深色）与样式表生成。

集中管理 HamiVisualizer 的全部界面颜色与 QSS / QPalette 构建：

  - ``LIGHT`` / ``DARK`` 两套语义颜色（冻结 dataclass）；
  - ``app_stylesheet(theme, scale)`` 生成跟随 UI 缩放的完整 QSS；
  - ``app_palette(theme)`` 生成 Fusion 基础调色板（菜单、弹窗、滚动条等
    原生控件也跟随主题）；
  - ``resolve_theme(mode, app)`` 把 "light" / "dark" / "system" 解析成
    实际明暗。

深色主题同时作用于控件外壳、绘图画布与场景内元素（矩阵冻结标尺、能带
网格、晶格首胞填充等），由 MainWindow 应用时把 dark 标记下发给各场景。
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette


@dataclass(frozen=True)
class Theme:
    """一组语义颜色（十六进制字符串）。"""

    # 窗口 / 面板
    window: str
    rail: str
    card: str
    card_border: str
    text: str
    muted: str
    group_title: str
    group_text: str
    hint: str
    error_text: str
    # 按钮
    button_bg: str
    button_text: str
    button_border: str
    button_hover_bg: str
    button_hover_border: str
    button_hover_text: str
    button_pressed_bg: str
    # 强调色
    accent: str
    accent_text: str
    selection_bg: str
    # 输入 / 复选
    input_bg: str
    input_border: str
    input_focus_border: str
    checkbox_border: str
    # 表格
    table_bg: str
    table_alt: str
    table_border: str
    table_grid: str
    table_selected: str
    table_selected_text: str
    header_bg: str
    header_text: str
    # 标签页
    pane_bg: str
    tab_border: str
    tab_bg: str
    tab_text: str
    tab_hover_bg: str
    tab_hover_text: str
    tab_selected_bg: str
    tab_selected_text: str
    # 模型栏 / 菜单
    model_bar_bg: str
    model_bar_border: str
    menubar_bg: str
    menubar_border: str
    menubar_hover_bg: str
    menubar_hover_text: str
    menu_bg: str
    menu_border: str
    menu_hover_bg: str
    menu_hover_text: str
    # 状态栏 / 画布 / 分隔 / 滚动条
    statusbar_bg: str
    statusbar_text: str
    statusbar_border: str
    canvas: str
    canvas_border: str
    splitter: str
    splitter_hover: str
    scrollbar: str
    scrollbar_hover: str
    # 结果横幅
    banner_info_bg: str
    banner_info_text: str
    banner_info_border: str
    banner_err_bg: str
    banner_err_text: str
    banner_err_border: str
    # 场景元素
    ruler_bg: str
    ruler_border: str
    ruler_text: str
    band_grid: str
    band_axis: str
    band_text: str
    band_curve: str
    band_mark: str
    # 晶格首胞 / 虚影填充预混基准（亮色=白，暗色=画布色）
    blend_base: tuple = (255, 255, 255)


LIGHT = Theme(
    window="#f4f7fb",
    rail="#eef3f9",
    card="#ffffff",
    card_border="#d8e1ec",
    text="#1d2939",
    muted="#627d98",
    group_title="#33506d",
    group_text="#243b53",
    hint="#667085",
    error_text="#b00020",
    button_bg="#ffffff",
    button_text="#25415d",
    button_border="#cbd8e6",
    button_hover_bg="#edf4ff",
    button_hover_border="#72a2e8",
    button_hover_text="#1457ad",
    button_pressed_bg="#dceaff",
    accent="#2f6fed",
    accent_text="#ffffff",
    selection_bg="#2f6fed",
    input_bg="#ffffff",
    input_border="#cbd8e6",
    input_focus_border="#4c8bf5",
    checkbox_border="#b8c6d8",
    table_bg="#ffffff",
    table_alt="#f7faff",
    table_border="#d8e1ec",
    table_grid="#e6edf5",
    table_selected="#dceaff",
    table_selected_text="#153e75",
    header_bg="#f1f5f9",
    header_text="#486581",
    pane_bg="#ffffff",
    tab_border="#d8e1ec",
    tab_bg="#eaf0f7",
    tab_text="#627d98",
    tab_hover_bg="#f6f9fd",
    tab_hover_text="#2f6fed",
    tab_selected_bg="#ffffff",
    tab_selected_text="#1f5fbf",
    model_bar_bg="#ffffff",
    model_bar_border="#d8e1ec",
    menubar_bg="#ffffff",
    menubar_border="#dce4ee",
    menubar_hover_bg="#eaf2ff",
    menubar_hover_text="#1f5fbf",
    menu_bg="#ffffff",
    menu_border="#d5dfeb",
    menu_hover_bg="#eaf2ff",
    menu_hover_text="#1457ad",
    statusbar_bg="#ffffff",
    statusbar_text="#627d98",
    statusbar_border="#dce4ee",
    canvas="#ffffff",
    canvas_border="#d8e1ec",
    splitter="#e4ebf3",
    splitter_hover="#9dbce9",
    scrollbar="#bac8d8",
    scrollbar_hover="#8fa8c3",
    banner_info_bg="#eef5ff",
    banner_info_text="#174a8b",
    banner_info_border="#a9c8ef",
    banner_err_bg="#fff0f0",
    banner_err_text="#9d1c1c",
    banner_err_border="#e5aaaa",
    ruler_bg="#ffffff",
    ruler_border="#cdd3dc",
    ruler_text="#2d3748",
    band_grid="#dcdcdc",
    band_axis="#969696",
    band_text="#5a5a5a",
    band_curve="#1950aa",
    band_mark="#c81e1e",
    blend_base=(255, 255, 255),
)


DARK = Theme(
    window="#141a21",
    rail="#1a222c",
    card="#1e2833",
    card_border="#2e3b4a",
    text="#d7dee7",
    muted="#8496aa",
    group_title="#9fb6cf",
    group_text="#c3d2e0",
    hint="#7e90a3",
    error_text="#ff7b72",
    button_bg="#24303d",
    button_text="#c9d6e4",
    button_border="#3a4757",
    button_hover_bg="#2c3c50",
    button_hover_border="#4f86c9",
    button_hover_text="#9cc2f0",
    button_pressed_bg="#22334a",
    accent="#3d7ff0",
    accent_text="#ffffff",
    selection_bg="#2f6fed",
    input_bg="#1a232e",
    input_border="#38475a",
    input_focus_border="#5b95f5",
    checkbox_border="#46556a",
    table_bg="#19212b",
    table_alt="#202b37",
    table_border="#2e3b4a",
    table_grid="#2c3745",
    table_selected="#24405f",
    table_selected_text="#cfe0f7",
    header_bg="#222d3a",
    header_text="#a9bccf",
    pane_bg="#19212b",
    tab_border="#2e3b4a",
    tab_bg="#232e3b",
    tab_text="#8496aa",
    tab_hover_bg="#2a3745",
    tab_hover_text="#6aa0f0",
    tab_selected_bg="#1e2833",
    tab_selected_text="#9cc2f0",
    model_bar_bg="#1e2833",
    model_bar_border="#2e3b4a",
    menubar_bg="#1e2833",
    menubar_border="#2a3442",
    menubar_hover_bg="#2a3a50",
    menubar_hover_text="#a8c8f0",
    menu_bg="#222c38",
    menu_border="#3a4757",
    menu_hover_bg="#2a3a50",
    menu_hover_text="#a8c8f0",
    statusbar_bg="#1e2833",
    statusbar_text="#8496aa",
    statusbar_border="#2a3442",
    canvas="#10161d",
    canvas_border="#2e3b4a",
    splitter="#2a3442",
    splitter_hover="#4f86c9",
    scrollbar="#45566a",
    scrollbar_hover="#5f7691",
    banner_info_bg="#1d3045",
    banner_info_text="#9cc2f0",
    banner_info_border="#3a5570",
    banner_err_bg="#3a2025",
    banner_err_text="#ff9b94",
    banner_err_border="#5a3338",
    ruler_bg="#1a222c",
    ruler_border="#3a4757",
    ruler_text="#d5dde6",
    band_grid="#2a3442",
    band_axis="#5a6b80",
    band_text="#aebccd",
    band_curve="#6aa0f0",
    band_mark="#ff6b6b",
    blend_base=(16, 22, 29),
)


def resolve_theme(mode: str, app) -> str:
    """把 "light" / "dark" / "system" 解析为实际明暗字符串。

    "system" 时读取 QStyleHints.colorScheme()（Qt 6.5+ 对 Windows/macOS
    系统深浅色模式的暴露）；无法判定时回退到浅色，保证离屏/无头环境稳定。
    """
    if mode == "dark":
        return "dark"
    if mode == "light":
        return "light"
    try:
        scheme = app.styleHints().colorScheme()
    except Exception:
        scheme = None
    if scheme is not None and scheme == Qt.ColorScheme.Dark:
        return "dark"
    return "light"


def app_palette(theme: Theme) -> QPalette:
    """Fusion 基础调色板：覆盖菜单、弹窗、文件对话框等原生控件。"""
    pal = QPalette()
    text = QColor(theme.text)
    muted = QColor(theme.muted)
    pal.setColor(QPalette.Window, QColor(theme.window))
    pal.setColor(QPalette.WindowText, text)
    pal.setColor(QPalette.Base, QColor(theme.input_bg))
    pal.setColor(QPalette.AlternateBase, QColor(theme.table_alt))
    pal.setColor(QPalette.Text, text)
    pal.setColor(QPalette.Button, QColor(theme.button_bg))
    pal.setColor(QPalette.ButtonText, QColor(theme.button_text))
    pal.setColor(QPalette.Highlight, QColor(theme.accent))
    pal.setColor(QPalette.HighlightedText, QColor(theme.accent_text))
    pal.setColor(QPalette.ToolTipBase, QColor(theme.menu_bg))
    pal.setColor(QPalette.ToolTipText, text)
    pal.setColor(QPalette.PlaceholderText, QColor(theme.hint))
    pal.setColor(QPalette.Link, QColor(theme.accent))
    pal.setColor(QPalette.BrightText, QColor(255, 255, 255))
    for role in (QPalette.Text, QPalette.WindowText, QPalette.ButtonText):
        pal.setColor(QPalette.Disabled, role, muted)
    return pal


def app_stylesheet(theme: Theme, scale: float) -> str:
    """生成跟随 UI 缩放的完整 QSS（应用在 QApplication 上）。"""
    p = lambda value: max(1, round(value * scale))  # noqa: E731
    return f"""
        QMainWindow {{ background:{theme.window}; color:{theme.text}; }}
        QWidget {{ selection-background-color:{theme.selection_bg}; selection-color:{theme.accent_text}; }}
        QScrollArea#controlRail {{ background:{theme.rail}; border-right:1px solid {theme.menubar_border}; }}
        QScrollArea#controlRail > QWidget > QWidget {{ background:{theme.rail}; }}
        QGroupBox {{
            font-weight:600; color:{theme.group_text}; background:{theme.card};
            border:1px solid {theme.card_border}; border-radius:{p(9)}px;
            /* Keep the title inside the widget's own geometry.  A title in
               the outer margin gets sliced in half when the left control
               rail is scrolled to the next section. */
            margin-top:{p(7)}px; padding:{p(8)}px; padding-top:{p(29)}px;
        }}
        QGroupBox::title {{
            /* CollapsibleGroupBox paints an in-card header.  Disable Qt's
               border-margin title so it cannot overlap the custom bar. */
            color:transparent; background:transparent;
        }}
        QPushButton, QToolButton {{
            padding:{p(6)}px {p(11)}px; min-height:{p(17)}px;
            border:1px solid {theme.button_border}; border-radius:{p(6)}px;
            background:{theme.button_bg}; color:{theme.button_text};
        }}
        QPushButton:hover, QToolButton:hover {{
            background:{theme.button_hover_bg}; border-color:{theme.button_hover_border}; color:{theme.button_hover_text};
        }}
        QPushButton:pressed, QToolButton:pressed {{ background:{theme.button_pressed_bg}; }}
        QPushButton:disabled, QToolButton:disabled {{ color:{theme.muted}; background:{theme.rail}; }}
        QToolButton:checked {{
            background:{theme.accent}; color:{theme.accent_text}; border-color:{theme.accent};
        }}
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
            padding:{p(4)}px {p(7)}px; min-height:{p(17)}px;
            border:1px solid {theme.input_border}; border-radius:{p(5)}px;
            background:{theme.input_bg}; color:{theme.text};
        }}
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
            border:2px solid {theme.input_focus_border}; padding:{p(3)}px {p(6)}px;
        }}
        QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
            color:{theme.muted}; background:{theme.rail};
        }}
        /* Fixed-pixel coefficient editors sit on top of the lattice. Keep
           them visually compact, opaque, and high-contrast so a line or node
           behind the field can never make the value look clipped. */
        QLineEdit#hopStrengthEditor {{
            /* The editor supplies a polished fixed height.  A scaled
               stylesheet min-height would override it and make the
               QGraphicsProxyWidget bounding rect shorter than its child. */
            min-height:0px; padding:{p(2)}px {p(7)}px;
            border:1px solid {theme.input_border}; border-radius:{p(6)}px;
            background:{theme.input_bg}; color:{theme.text};
        }}
        QLineEdit#hopStrengthEditor:focus {{
            border:2px solid {theme.input_focus_border};
            padding:{p(1)}px {p(6)}px;
        }}
        QComboBox::drop-down {{ border:0; width:{p(22)}px; }}
        QComboBox QAbstractItemView {{
            background:{theme.menu_bg}; color:{theme.text};
            border:1px solid {theme.menu_border}; selection-background-color:{theme.menu_hover_bg};
            selection-color:{theme.menu_hover_text};
        }}
        QCheckBox {{ spacing:{p(7)}px; color:{theme.text}; }}
        QCheckBox:disabled {{ color:{theme.muted}; }}
        QTableWidget {{
            background:{theme.table_bg}; alternate-background-color:{theme.table_alt};
            border:1px solid {theme.table_border}; border-radius:{p(6)}px;
            gridline-color:{theme.table_grid}; color:{theme.text};
        }}
        QTableWidget::item {{ padding:{p(3)}px {p(5)}px; border:0; }}
        QTableWidget::item:selected {{ background:{theme.table_selected}; color:{theme.table_selected_text}; }}
        QHeaderView::section {{
            background:{theme.header_bg}; color:{theme.header_text}; font-weight:600;
            border:0; border-bottom:1px solid {theme.table_border};
            padding:{p(4)}px {p(5)}px;
        }}
        QTabWidget::pane {{
            border:1px solid {theme.tab_border}; border-radius:{p(7)}px;
            background:{theme.pane_bg}; top:-1px;
        }}
        QTabBar::tab {{
            padding:{p(7)}px {p(14)}px; min-height:{p(18)}px;
            color:{theme.tab_text}; background:{theme.tab_bg};
            border:1px solid transparent;
            border-top-left-radius:{p(6)}px; border-top-right-radius:{p(6)}px;
        }}
        QTabBar::tab:hover {{ background:{theme.tab_hover_bg}; color:{theme.tab_hover_text}; }}
        QTabBar::tab:selected {{
            background:{theme.tab_selected_bg}; color:{theme.tab_selected_text}; font-weight:600;
            border-color:{theme.tab_border}; border-bottom-color:{theme.tab_selected_bg};
        }}
        QWidget#modelBarContainer {{ background:{theme.model_bar_bg}; border-bottom:1px solid {theme.model_bar_border}; }}
        QMenuBar {{ background:{theme.menubar_bg}; color:{theme.text}; border-bottom:1px solid {theme.menubar_border}; padding:{p(2)}px; }}
        QMenuBar::item {{ padding:{p(6)}px {p(10)}px; border-radius:{p(4)}px; color:{theme.text}; background:transparent; }}
        QMenuBar::item:selected {{ background:{theme.menubar_hover_bg}; color:{theme.menubar_hover_text}; }}
        QMenu {{ background:{theme.menu_bg}; color:{theme.text}; border:1px solid {theme.menu_border}; border-radius:{p(7)}px; padding:{p(5)}px; }}
        QMenu::item {{ padding:{p(6)}px {p(28)}px {p(6)}px {p(11)}px; border-radius:{p(4)}px; color:{theme.text}; background:transparent; }}
        QMenu::item:selected {{ background:{theme.menu_hover_bg}; color:{theme.menu_hover_text}; }}
        QMenu::item:disabled {{ color:{theme.muted}; }}
        QMenu::separator {{ height:1px; background:{theme.card_border}; margin:{p(3)}px {p(6)}px; }}
        QStatusBar {{ background:{theme.statusbar_bg}; color:{theme.statusbar_text}; border-top:1px solid {theme.statusbar_border}; }}
        QGraphicsView {{ background:{theme.canvas}; border:1px solid {theme.canvas_border}; border-radius:{p(6)}px; }}
        QSplitter::handle {{ background:{theme.splitter}; width:{p(5)}px; height:{p(5)}px; }}
        QSplitter::handle:hover {{ background:{theme.splitter_hover}; }}
        QScrollBar:vertical {{ width:{p(10)}px; background:transparent; margin:{p(3)}px; }}
        QScrollBar::handle:vertical {{ background:{theme.scrollbar}; min-height:{p(26)}px; border-radius:{p(5)}px; }}
        QScrollBar::handle:vertical:hover {{ background:{theme.scrollbar_hover}; }}
        QScrollBar:horizontal {{ height:{p(10)}px; background:transparent; margin:{p(3)}px; }}
        QScrollBar::handle:horizontal {{ background:{theme.scrollbar}; min-width:{p(26)}px; border-radius:{p(5)}px; }}
        QScrollBar::add-line, QScrollBar::sub-line {{ width:0; height:0; }}
        QToolTip {{ background:{theme.menu_bg}; color:{theme.text}; border:1px solid {theme.menu_border}; padding:{p(4)}px; }}
        QDialog {{ background:{theme.window}; }}
        QMessageBox {{ background:{theme.window}; }}
        QSlider::groove:horizontal {{
            height:{p(4)}px; background:{theme.card_border}; border-radius:{p(2)}px;
        }}
        QSlider::handle:horizontal {{
            width:{p(13)}px; margin:-{p(5)}px 0; border-radius:{p(7)}px;
            background:{theme.accent}; border:1px solid {theme.accent};
        }}
        QSlider::sub-page:horizontal {{ background:{theme.accent}; border-radius:{p(2)}px; }}
        QLabel#panelHint {{ color:{theme.hint}; }}
        QLabel#errorLabel {{ color:{theme.error_text}; font-weight:600; }}
        QLabel#dialogNote {{ color:{theme.hint}; }}
        QLabel#templatePreview {{
            background:{theme.card}; color:{theme.text};
            border:1px solid {theme.card_border}; border-radius:{p(6)}px; padding:{p(10)}px;
        }}
        QLabel#comparisonEmpty {{ background:{theme.canvas}; color:{theme.muted}; padding:{p(18)}px; }}
    """
