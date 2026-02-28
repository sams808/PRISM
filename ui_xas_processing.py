from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from typing import List, Dict

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from xas_processing import compute_mu


class XASProcessingWindow:
    def __init__(self, master, records: List[Dict]):
        self.master = master
        self.records = records
        self.master.title("XAS processing")
        self.master.geometry("1100x700")

        left = ttk.Frame(master, padding=10)
        left.pack(side="left", fill="y")
        right = ttk.Frame(master, padding=10)
        right.pack(side="left", fill="both", expand=True)

        ttk.Label(left, text="XAS dataset").pack(anchor="w")
        self.var_dataset = tk.StringVar(value=records[0]["title"])
        self.cb_dataset = ttk.Combobox(left, textvariable=self.var_dataset, state="readonly", values=[r["title"] for r in records], width=32)
        self.cb_dataset.pack(fill="x", pady=(0, 10))

        self.var_log = tk.StringVar(value="ln")
        ttk.Label(left, text="log base").pack(anchor="w")
        ttk.Combobox(left, textvariable=self.var_log, state="readonly", values=["ln", "log10"], width=10).pack(anchor="w", pady=(0, 8))

        self.var_deglitch = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="Deglitch", variable=self.var_deglitch).pack(anchor="w")

        ttk.Label(left, text="Deglitch z").pack(anchor="w")
        self.ent_deg_z = ttk.Entry(left, width=10)
        self.ent_deg_z.insert(0, "6.0")
        self.ent_deg_z.pack(anchor="w", pady=(0, 6))

        ttk.Label(left, text="Deglitch window").pack(anchor="w")
        self.ent_deg_win = ttk.Entry(left, width=10)
        self.ent_deg_win.insert(0, "21")
        self.ent_deg_win.pack(anchor="w", pady=(0, 6))

        ttk.Label(left, text="Smooth window").pack(anchor="w")
        self.ent_smooth = ttk.Entry(left, width=10)
        self.ent_smooth.insert(0, "1")
        self.ent_smooth.pack(anchor="w", pady=(0, 12))

        ttk.Button(left, text="Compute and plot μ(E)", command=self.compute_and_plot).pack(fill="x")

        self.fig = Figure(figsize=(7, 5), dpi=110)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        self.compute_and_plot()

    def _get_selected_record(self):
        name = self.var_dataset.get()
        for rec in self.records:
            if rec["title"] == name:
                return rec
        return self.records[0]

    def compute_and_plot(self):
        rec = self._get_selected_record()
        try:
            z = float(self.ent_deg_z.get())
            win = int(self.ent_deg_win.get())
            smooth = int(self.ent_smooth.get())
            mu = compute_mu(
                rec["xas"],
                log_base=self.var_log.get(),
                deglitch=self.var_deglitch.get(),
                deglitch_z=z,
                deglitch_window=win,
                smooth_window=smooth,
            )
        except Exception as exc:
            messagebox.showerror("XAS processing", str(exc), parent=self.master)
            return

        self.ax.clear()
        self.ax.plot(rec["xas"].energy, mu, color="tab:blue", lw=1.5, label="μ(E)")
        self.ax.set_xlabel("Energy (eV)")
        self.ax.set_ylabel("μ(E)")
        self.ax.set_title(f"{rec['title']} — μ(E)")
        self.ax.grid(alpha=0.25)
        self.ax.legend(loc="best")
        self.fig.tight_layout()
        self.canvas.draw_idle()
