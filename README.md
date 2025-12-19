# Dataapp

A desktop GUI for quickly inspecting and processing Raman/Thermal spectra.  It is built
on Tkinter and Matplotlib and wraps a universal loader capable of handling simple
XY text files as well as richer exports such as TA SDT ASCII, SAXS EDF files, and
thermo-analysis traces used by the DTA/DSC/TGA tooling.

## Features
- **Quick import** – batch-select files and let the app auto-detect the parser and
  the best X/Y columns without any prompts.
- **Custom import** – open the column/type selector dialog for each file so you can
  override the detected parser or choose different columns.
- **Processing tools** – rename/reorder imports, baseline subtraction, fitting
  helpers, multi-spectrum sums and a "Simple plot" workspace with CIF overlays.
- **Thermal analysis (DTA/DSC/TGA) tools** – pick temperature/time/Y channels, derive
  dY/dt or dY/dT on the fly, compute Tg using multiple methods, run min/max/integration
  "Calculs" with dual Y-axes, and batch-process many text exports.
- **Format-aware defaults** – TA files expose canonical column names so the UI can
  offer temperature vs heat-flow, TG curves, etc.

## Requirements
The Python dependencies are listed in `requirements.txt`.  Install them with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Make sure a Tcl/Tk runtime is available (the Tkinter module bundled with standard
Python builds already provides it on Windows/macOS; on Linux install `python3-tk`).

## Running the app
After installing the dependencies, start the GUI with:

```bash
python main.py
```

On launch you can use **Quick import** to rapidly load a batch of spectra or
**Custom import** when you need to select parser/columns manually.  Imported files
appear in the central list and become available to the processing/plotting tools
on the right-hand side.

### Using the DTA processing tool
1. Import one or more TA text exports (Quick or Custom import).
2. Open **DTA processing** from the right-hand toolbox. The file name becomes the plot title.
3. Pick **X** (time or temperature), **Y** (DSC/TGA/TGA derivative), and optional
   **dY source** columns. Toggle **dY overlay** to see derivatives on a secondary axis.
4. Adjust Tg method (Double, Parallel, or |dY| max). Use the shaded window to focus
   on a region; the view lock keeps zoom/pan stable while you tweak parameters.
5. Under **Calculs**, choose the Y column (or its derivative) to integrate, find min/max,
   or calculate an average over a custom X-range. Alternate-Y calculations are plotted on
   a distinct right axis so they do not hide the main trace.
6. Use **Batch run** for multiple files; optional PNG snapshots help validate results.

## Repository layout
- `main.py` – core Tkinter application and import workflow
- `io_universal.py` – pluggable parser framework that understands TA/SAXS/XY files
- `ui_simple_plot.py` – standalone plotting window with CIF overlays
- `ui_fit_params.py` – fit-parameter management dialogs
- `cif_tools.py` – CIF parsing + Bragg peak helpers
- `io_importers.py` – legacy lightweight loader kept for reference

## Troubleshooting
If a file fails to import, try the **Custom import** button which will open the
selector dialog even when the auto-detected columns look confident.  The dialog
also displays parser hints and available columns to help pinpoint problems.
