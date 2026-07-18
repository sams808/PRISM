# PRISM — Platform for Research In Spectroscopy & Materials

A desktop application for importing, processing, and analyzing scientific spectra:
Raman, XAS/XANES/EXAFS, DTA/DSC/TGA thermal analysis, XRD (including phase
identification and high-temperature series), and SAXS.

The app is PySide6/Qt-based, organized as one main window with a left navigation
rail of technique workspaces, color-coded by module — and each module (Raman,
Fitting, XRD, XAS, Thermal, Processing, Figures) can be switched off in the
Modules toolbar, so a single-technique user sees a simple app. (The original Tkinter application was retired after
the Qt migration completed; it remains available in git history.)

## Workspaces

| Workspace | What it does |
|---|---|
| **Library** | Import data files (auto-detected parser, or Custom Import with parser/column override), browse, preview, rename/duplicate/reorder, delete with Undo, Combine/scale (sum, average, weighted subtraction), export as text. Feeds the other workspaces. |
| **Baseline** | Baseline subtraction (arPLS, ALS, polynomial, spline, rubberband) with live preview, drag-to-pick fit regions, per-spectrum settings memory, and batch apply producing `_bl` spectra. |
| **Raman** | Simple Plot: multi-spectrum plotting (separate or stacked), smoothing, color schemes, axis controls, CIF Bragg-peak overlays with a per-CIF manager, PNG/SVG/PDF export. |
| **XAS** | Full XAS/XANES/EXAFS pipeline: EasyXAFS ZIP / CSV / Athena `.prj` import, μ(E) builder (with deglitching), Larch normalization and EXAFS/FT (`pre_edge`, `autobk`, `xftf`), merge/average, difference spectra, linear-combination fitting, edge definer, Athena `.dat`/`.prj` export. Requires `xraylarch` (see `requirements-xas.txt`). |
| **DTA / Thermal** | Tg determination by three methods (double tangent, parallel tangent, \|dY\| max), integration/extrema "Calculs", batch processing with CSV export. |
| **Peak Fitting** | Single-spectrum peak fitting (Gaussian, pseudo-Voigt, true Voigt, EMG via lmfit): classic one-shot LM or Origin-style stepwise LM (one visible parameter update per iteration, converge-to-tolerance), auto peak finding with an adjustable detection limit, click-to-pick peaks on the plot, parameter linking, residual subplot, F-test confidence intervals, per-component CSV export, fit reports with R², ±1σ errors and peak centroids, save/load parameter models. |
| **Multi-Fit** | Batch fitting: apply a saved parameter model ("recipe" — the same JSON files Peak Fitting saves) to many spectra at once; results table + CSV export. |
| **Mineral ID** | RRUFF database match-assist: ranks mineral candidates by Raman peak overlap, with database filters (laser wavelength, oriented/unoriented, high-res vs broad-scan, quality) applied before ranking; shows each candidate's laser excitation wavelength and overlays the reference spectrum — identification is always the user's explicit decision, never automatic. Requires a local RRUFF cache (see below). |
| **HT-XRD** | High-temperature XRD series: import a folder of patterns (temperature from `.rasx` metadata or a Jana-style `???` filename template), waterfall view colored by temperature with peak guide lines, a Maps tab (2D heatmap with linear/log/sqrt/power color scales, signed/absolute difference map and difference waterfall vs a reference, 3D surface, time axis from heating rate), and multi-window peak tracking with sequential seeding, per-window `@` anchors, absence detection, and vanished/appeared + fit-quality transition flags. |
| **Clustering** | KMeans / hierarchical clustering of spectral series with PCA scatter, per-cluster mean spectra, and assignment table. |

Cross-cutting: `.dataapp` project files (everything survives closing the app),
a Python console (View menu) with the live app objects in scope, dark mode,
background threading for batch operations, and keyboard shortcuts
(Ctrl+O import, Ctrl+S save project, Ctrl+E export, Ctrl+Z undo delete).

## Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt      # core science stack
pip install -r requirements-qt.txt   # PySide6, for the Qt app
pip install -r requirements-xas.txt  # xraylarch, optional — XAS workspace
pip install -r requirements-dev.txt  # pytest + pytest-qt, for running the tests
```

## Running

```bash
python qt_main.py
```

Or double-click `Dataapp.bat`. A standalone `Dataapp.exe` (no Python needed)
can be built with PyInstaller — see the exclusion list in the plan/commit
history; the built `dist/Dataapp/` folder is portable (~350 MB; the
Larch-dependent XAS steps require the Python route).

## Building the RRUFF database cache (Mineral ID workspace)

The Mineral ID workspace needs a one-time local ingest of the RRUFF Raman
database (https://rruff.net — please cite: Lafuente, Downs, Yang & Stone (2015),
"The power of databases: the RRUFF project"):

1. Download the category ZIPs from https://www.rruff.net/zipped_data_files/raman/
2. In Python:
   ```python
   import rruff_science as rs
   rs.build_index([
       ("path/to/excellent_oriented.zip", "excellent_oriented"),
       ("path/to/excellent_unoriented.zip", "excellent_unoriented"),
       # ... one entry per downloaded ZIP
   ])
   ```
3. The cache lands in `~/.raman_cache/rruff/` (~1.2 GB for the full database,
   ~28,000 spectra / ~2,500 minerals) and is loaded automatically by the
   Mineral ID workspace.

## Running the tests

```bash
pytest
```

## Repository layout

Science layer (framework-agnostic, fully tested — no GUI imports):
- `io_universal.py` — pluggable parser framework (XY text, TA SDT, SAXS EDF, Rigaku `.rasx`, …)
- `cif_tools.py` — CIF parsing + Bragg peak generation (disk-cached)
- `dta_science.py` — Tg/derivative/integration math
- `fitting_science.py` — lmfit peak models, fitting entry point, peak finding
- `xas_science.py` — XAS/XANES/EXAFS engine (Larch wrappers, data models, I/O)
- `rruff_science.py` — RRUFF database ingest + match ranking
- `htxrd_science.py` — HTXRD series loading + peak tracking + transition flagging

Qt layer:
- `qt_main.py` — entry point; `qt_shell.py` — main window/navigation
- `qt_widgets.py` (shared plot widget with debounced redraws), `qt_theme.py`,
  `qt_models.py` (Spectrum/SpectrumLibrary), `qt_settings_store.py`,
  `qt_exception_hook.py`
- One `qt_*.py` per workspace: `qt_simple_plot`, `qt_xas`, `qt_dta`,
  `qt_single_fit`, `qt_fit_params`, `qt_multi_fit`, `qt_rruff`, `qt_htxrd`

## Troubleshooting

If a file fails to import in the Library, the parser registry may have
misdetected the format — check `io_universal.py`'s parser list; every parser
records its decision in the returned metadata (`selected_parser`).

## Sharing the local databases with colleagues

Both reference databases are single files once built:

| Database | File | Size | Share it? |
|---|---|---|---|
| XRD ID (COD + PDF-2) | `~/.raman_cache/xrd_id/xrdid.sq` | ~840 MB | **Privately only** — it embeds ICDD PDF-2 content, which is licensed; hand it to group members covered by the same institutional license (USB/network drive), never post it publicly. |
| RRUFF Raman | run `rruff_science.pack_rruff_database()` → one `rruff_pack.sq` | ~1 GB | Yes, with attribution (Lafuente et al. 2015). Import on the other machine with `rruff_science.unpack_rruff_database(path)`. |

Neither file can live in this git repository: GitHub hard-rejects files
over 100 MB (and LFS quotas don't fit multi-GB scientific databases).
Rebuild instead from sources (`build_xrd_database()`, `build_index()`), or
copy the single files above directly.
