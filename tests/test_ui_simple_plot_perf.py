"""Regression test for the Simple Plot / CIF-overlay slowdown: typing in the
axis-title fields used to call plot_selected_spectrum() (which redraws every
CIF Bragg-marker vline/label from scratch) on every keystroke, with no
debounce — reported directly by hands-on use with 5-7 CIF overlays loaded,
where even a single keystroke took multiple seconds to register. Other CIF
controls in this same file already used a self._debounce() helper; the axis
title/range StringVars were simply never wired through it.
"""
from __future__ import annotations

import tkinter as tk

import pytest

import ui_simple_plot as sp
from main import simpleplot_unified_loader, pick_default_xy_for_ta_sdt


@pytest.fixture(scope="module")
def tk_root():
    # One tk.Tk() root shared by every test in this file. Creating and
    # destroying multiple tk.Tk() roots in sequence within one process is
    # intermittently fragile under pytest (observed ~1-in-4 TclError failures
    # even with pytest-qt's plugin disabled) — sharing one root and giving
    # each test its own SimplePlotWindow (a tk.Toplevel child) instead avoids
    # that fragility entirely rather than just reducing its odds.
    root = tk.Tk()
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def simple_plot_window(tk_root, raman_example_path):
    win = sp.SimplePlotWindow(
        tk_root, [str(raman_example_path)], ["Raman_example"],
        load_any_func=simpleplot_unified_loader,
        pick_ta_xy_func=pick_default_xy_for_ta_sdt,
    )
    tk_root.update_idletasks()
    yield tk_root, win
    win.destroy()


def test_rapid_axis_title_edits_collapse_into_one_redraw(simple_plot_window):
    root, win = simple_plot_window

    call_count = {"n": 0}
    original = win.plot_selected_spectrum

    def counting_wrapper(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    win.plot_selected_spectrum = counting_wrapper

    baseline = call_count["n"]  # construction itself may draw once

    # NOTE: we don't assert anything about the call count *during* this loop.
    # root.update() can legitimately fire an already-elapsed after() timer
    # depending on wall-clock variance between iterations (fixture/setup
    # overhead eats into the debounce window unpredictably under pytest) —
    # that's a property of the test harness, not of the fix. What matters,
    # and what actually reproduces the reported bug, is the END state below:
    # without debouncing, N keystrokes would cause N redraws; with it, they
    # collapse into a small, mostly-fixed number regardless of N.
    text = "My Custom Axis Title"
    for i in range(1, len(text) + 1):
        win.x_title.set(text[:i])
        root.update()

    # Let any pending debounce timer fire.
    root.after(250, root.quit)
    root.mainloop()

    # The old, undebounced code would have produced baseline + len(text) (21)
    # calls here. A handful at most (not dozens) proves the debounce collapsed
    # the keystroke burst instead of redrawing on every character.
    assert call_count["n"] <= baseline + 2
    assert win.x_title.get() == text


def test_rapid_axis_range_edits_are_also_debounced(simple_plot_window):
    root, win = simple_plot_window

    call_count = {"n": 0}
    original = win.update_plot_axes

    def counting_wrapper(*args, **kwargs):
        call_count["n"] += 1
        return original(*args, **kwargs)

    win.update_plot_axes = counting_wrapper

    for v in ["1", "10", "100", "1000"]:
        win.xmin.set(v)
        root.update()

    root.after(250, root.quit)
    root.mainloop()

    # Old undebounced code: 4 calls (one per edit). Debounced: a small number.
    assert call_count["n"] <= 2
