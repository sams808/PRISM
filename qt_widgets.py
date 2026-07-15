"""
qt_widgets.py — shared Qt widgets used by every tool window, instead of each
one reinventing matplotlib-embedding boilerplate. Today's Tk app embeds
matplotlib independently 12 times across 5 files — exactly the kind of
duplication that produced the 3-way XAS split; one shared PlotWidget here is
the fix for that pattern in the new architecture.

Also implements the two concrete performance fixes named in the rewrite
plan's "Performance" principle: debounced redraws (the historical Simple
Plot + CIF-overlay lag was almost certainly undebounced redraw-on-every-event)
and export-at-a-physical-size (A FAIRE item 14, "figure export size in cm").
"""
from __future__ import annotations

from typing import Callable, Optional

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class PlotWidget(QWidget):
    """Figure + canvas + navigation toolbar, with a debounced redraw helper.

    Update existing artists in place (line.set_data / set_offsets) wherever
    the underlying data hasn't changed, rather than ax.clear() + replot —
    that + undebounced redraws is the likely root cause of the historical
    CIF-overlay slowdown. request_redraw() below handles the debounce half
    of that fix; the "update in place" half is each caller's responsibility
    when it draws.
    """

    def __init__(self, parent: Optional[QWidget] = None, figsize=(6.0, 4.5), dpi: int = 100,
                 debounce_ms: int = 120):
        super().__init__(parent)
        self.figure = Figure(figsize=figsize, dpi=dpi)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # Live cursor readout (deferred M7 item, implemented once here so
        # every workspace gets it): x/y of the data point under the mouse.
        self.coords_label = QLabel("")
        self.coords_label.setObjectName("SectionNote")
        self.coords_label.setAlignment(Qt.AlignRight)
        layout.addWidget(self.coords_label)
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(debounce_ms)
        self._debounce.timeout.connect(self._flush_redraw)
        self._pending: Optional[tuple] = None

    def _on_mouse_move(self, event) -> None:
        if event.inaxes is None or event.xdata is None:
            self.coords_label.setText("")
            return
        self.coords_label.setText(f"x = {event.xdata:.4g}   y = {event.ydata:.4g}")

    def request_redraw(self, fn: Callable, *args, **kwargs) -> None:
        """Coalesce rapid-fire redraw requests (slider drags, live text-entry,
        a burst of selection-changed signals) into ONE redraw shortly after
        the last call, instead of one redraw per event. `fn` should do the
        actual plotting/artist updates; canvas.draw_idle() is called after.
        """
        self._pending = (fn, args, kwargs)
        self._debounce.start()

    def _flush_redraw(self) -> None:
        if self._pending is None:
            return
        fn, args, kwargs = self._pending
        self._pending = None
        fn(*args, **kwargs)
        self.canvas.draw_idle()

    def clear(self, title: str = "") -> None:
        self.ax.clear()
        self.ax.grid(alpha=0.25)
        if title:
            self.ax.set_title(title)
        self.canvas.draw_idle()

    def export_at_size_cm(self, path: str, width_cm: float, height_cm: float, dpi: int = 300) -> None:
        """Export the current figure at an exact physical size (A FAIRE item 14:
        a popup letting the user pick figure size in cm on export)."""
        w_in, h_in = width_cm / 2.54, height_cm / 2.54
        old_size = self.figure.get_size_inches()
        try:
            self.figure.set_size_inches(w_in, h_in)
            self.figure.savefig(path, dpi=dpi)
        finally:
            self.figure.set_size_inches(*old_size)
            self.canvas.draw_idle()
