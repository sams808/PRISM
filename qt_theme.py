"""
qt_theme.py — ONE centralized stylesheet and palette, replacing main.py's
_apply_futuristic_style and ui_dta_processing.py's _apply_modern_style — two
inconsistent, ad hoc ttk style functions, one of which globally mutates
ttk.Style() with no teardown (re-theming the whole app whenever the DTA tool
opened). Every Qt window should apply this once and never define its own.

Palette: a deep teal accent (distinct from matplotlib's default color cycle,
so it never fights with plotted data) on a cool, quiet sage-grey ground —
built to read as a coherent instrument-control tool, not a default-themed
utility window.
"""
from __future__ import annotations

PALETTE = {
    "bg": "#eef1ee",
    "bg_alt": "#e5e9e4",
    "card": "#f7f9f6",
    "ink": "#202b26",
    "muted": "#5b6b62",
    "border": "#d3dbd2",
    "accent": "#3c6e71",
    "accent_hover": "#2f5658",
    "accent_ink": "#ffffff",
    "selection_bg": "#dcebea",
    "warn": "#a8662b",
    "critical": "#a8402f",
    "critical_bg": "#f6e4df",
}

_FONT_FAMILY = '"Segoe UI", -apple-system, sans-serif'
_MONO_FAMILY = '"Cascadia Mono", Consolas, monospace'


def build_stylesheet(palette: dict = PALETTE) -> str:
    p = palette
    return f"""
    * {{
        font-family: {_FONT_FAMILY};
        color: {p['ink']};
    }}
    QMainWindow, QWidget {{
        background: {p['bg']};
    }}
    QWidget#Sidebar {{
        background: {p['bg_alt']};
        border-right: 1px solid {p['border']};
    }}
    QListWidget#NavList {{
        background: transparent;
        border: none;
        font-size: 13px;
        padding: 8px 4px;
    }}
    QListWidget#NavList::item {{
        padding: 9px 12px;
        border-radius: 4px;
        margin: 2px 4px;
    }}
    QListWidget#NavList::item:selected {{
        background: {p['accent']};
        color: {p['accent_ink']};
    }}
    QListWidget#NavList::item:hover:!selected {{
        background: {p['selection_bg']};
    }}
    QWidget#Card {{
        background: {p['card']};
        border: 1px solid {p['border']};
        border-radius: 4px;
    }}
    QLabel#SectionTitle {{
        font-size: 15px;
        font-weight: 600;
        color: {p['ink']};
    }}
    QLabel#SectionNote {{
        font-size: 12px;
        color: {p['muted']};
    }}
    QPushButton {{
        background: {p['card']};
        border: 1px solid {p['border']};
        border-radius: 4px;
        padding: 7px 14px;
        font-size: 13px;
    }}
    QPushButton:hover {{
        background: {p['selection_bg']};
    }}
    QPushButton#Primary {{
        background: {p['accent']};
        color: {p['accent_ink']};
        border: 1px solid {p['accent_hover']};
        font-weight: 600;
    }}
    QPushButton#Primary:hover {{
        background: {p['accent_hover']};
    }}
    QTableView, QTreeView {{
        background: {p['card']};
        border: 1px solid {p['border']};
        gridline-color: {p['border']};
        selection-background-color: {p['selection_bg']};
        selection-color: {p['ink']};
        font-size: 13px;
    }}
    QHeaderView::section {{
        background: {p['bg_alt']};
        border: none;
        border-bottom: 1px solid {p['border']};
        padding: 6px 8px;
        font-size: 11px;
        font-weight: 600;
        color: {p['muted']};
        text-transform: uppercase;
    }}
    QStatusBar {{
        background: {p['bg_alt']};
        border-top: 1px solid {p['border']};
        color: {p['muted']};
        font-size: 12px;
    }}
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
        background: {p['card']};
        border: 1px solid {p['border']};
        border-radius: 3px;
        padding: 4px 6px;
        font-size: 13px;
    }}
    QLineEdit:focus, QComboBox:focus {{
        border: 1px solid {p['accent']};
    }}
    QTabWidget::pane {{
        border: 1px solid {p['border']};
        background: {p['card']};
    }}
    QTabBar::tab {{
        background: {p['bg_alt']};
        border: 1px solid {p['border']};
        border-bottom: none;
        padding: 6px 14px;
        font-size: 12.5px;
    }}
    QTabBar::tab:selected {{
        background: {p['card']};
        font-weight: 600;
    }}
    QSplitter::handle {{
        background: {p['border']};
    }}
    """


def apply_theme(app, palette: dict = PALETTE) -> None:
    app.setStyleSheet(build_stylesheet(palette))
