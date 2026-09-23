"""
qt_worker.py — minimal background-worker helper (QThreadPool + QRunnable)
for the operations heavy enough to freeze the UI thread: Multi-Fit batch
runs and HT-XRD series peak tracking.

Contract for callers: the submitted function must be PURE computation —
read every widget value BEFORE submitting, touch no Qt widgets inside the
worker. Results come back on the main thread via the on_done callback
(signal-delivered), which is where all UI updates belong.

set_synchronous(True) makes run_in_thread execute inline — used by the
test suite so tests stay deterministic without wait-loops; production code
never calls it.
"""
from __future__ import annotations

import traceback
from typing import Any, Callable, Optional

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

_SYNCHRONOUS = False


def set_synchronous(value: bool) -> None:
    global _SYNCHRONOUS
    _SYNCHRONOUS = bool(value)


class WorkerSignals(QObject):
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(int, int, str)


class Worker(QRunnable):
    def __init__(self, fn: Callable, *args, pass_progress: bool = False, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.pass_progress = pass_progress
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            if self.pass_progress:
                result = self.fn(*self.args, progress=self._emit_progress, **self.kwargs)
            else:
                result = self.fn(*self.args, **self.kwargs)
        except Exception:
            self.signals.error.emit(traceback.format_exc())
            return
        self.signals.finished.emit(result)

    def _emit_progress(self, i: int, n: int, message: str = "") -> None:
        """Callable handed to the job as ``progress`` (the (i, n, message) convention of the SAXS 2.0
        batch functions): a queued signal, safe from the worker thread."""
        self.signals.progress.emit(int(i), int(n), str(message))


def run_in_thread(
    fn: Callable, on_done: Callable[[Any], None],
    on_error: Optional[Callable[[str], None]] = None,
    *args, on_progress: Optional[Callable[[int, int, str], None]] = None, **kwargs,
) -> Worker:
    """Run fn(*args, **kwargs) off the UI thread; deliver its return value
    to on_done (or the formatted traceback to on_error) on the main thread.
    With ``on_progress`` the job is called with an extra ``progress(i, n, message)``
    keyword whose calls reach ``on_progress`` on the main thread.
    Keep a reference to the returned Worker only if you need to inspect it —
    the thread pool owns its lifetime."""
    worker = Worker(fn, *args, pass_progress=on_progress is not None, **kwargs)
    worker.signals.finished.connect(on_done)
    if on_error is not None:
        worker.signals.error.connect(on_error)
    if on_progress is not None:
        worker.signals.progress.connect(on_progress)
    if _SYNCHRONOUS:
        worker.run()
    else:
        QThreadPool.globalInstance().start(worker)
    return worker
