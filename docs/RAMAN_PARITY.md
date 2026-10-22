# Spectragryph → PRISM Raman module: feature inventory and mapping

Source of the inventory: the Spectragryph manual (File, Plot/Views, Spectra,
Process, Transform, Analyze, Identify, Automate, Acquire chapters), the
official feature page and the complete version history 1.0 (2016) to
1.2.16.1 (2022). Every function named there is listed once below, grouped
the way Spectragryph groups them.

Status in PRISM after the relayout of the Raman module (2026-10-22):
**done** (available, same meaning), **partial** (available in a weaker or
different form; the gap is named), **later** (worth doing, not in this
pass), **n/a** (outside PRISM's scope or specific to Spectragryph's own
formats and hardware).

Where: **R** = Raman workbench (the relaid-out Raman workspace, tabs
Spectra / Display / Process / Transform / Analyze / Identify / Sequence /
Export), **L** = Library, **ID** = Raman ID, **F** = Peak Fitting, **C** =
Calculations, **H** = Home / settings.

## 1. File: import

| Spectragryph | Status | PRISM |
|---|---|---|
| Open one or several files | done | L, R Spectra |
| Open a whole folder | done | R Spectra › Folder… (every readable file of the folder) |
| Thermo OMNIC `.spa` | done | L (own reader, metadata kept) |
| Thermo OMNIC `.spg` (multi-spectrum) | later | same block structure as `.spa`; add when a file exists |
| Thermo OMNIC `.mapx` maps | done | L (regions, pixels, stage positions) |
| Thermo GRAMS `.spc` | later | reader to write once a file of the group exists |
| JCAMP-DX (AFFN, SQZ/DIF/DUP) | done | L |
| ASCII / `.txt` / `.csv` / `.dat` / `.xy` (any separator, decimal comma, Latin-1, UTF-16) | done | L |
| RRUFF `.txt` | done | L (generic table) |
| Renishaw `.wdf`, Horiba `.ngs`/`.spc`, Bruker OPUS, Agilent, Shimadzu, Avantes, Ocean Optics, Stellarnet, Andor `.sif`, Princeton, Jasco, PerkinElmer, Varian, Beckman, WinAspect, ASD, USGS, Amptek … (the rest of the 89 formats) | n/a | not the group's instruments; `.wdf` on request |
| Spekwin `.spv`, Spectragryph `.sgd`/`.sgas`/`.sgps` | n/a | Spectragryph's own formats |
| Paste data from clipboard | done | File menu (Ctrl+Shift+V) |
| Drag and drop files onto the window | done | main window |
| Double-click a file in Explorer | later | no command-line opening or file association yet; drag and drop onto the window works |
| Recent files / recent folders | later | |
| Open unknown binary | n/a | |

## 2. File: export

| Spectragryph | Status | PRISM |
|---|---|---|
| Save data as `.csv`, `.dat`, `.txt` | done | L › Export selected as text (txt/csv) |
| Save data as JCAMP-DX `.dx`/`.jdx` | done | R Export › JCAMP-DX… (one .jdx per highlighted spectrum) |
| Save data as GRAMS `.spc`, FOSS `.nir`, `.spv`, `.sgd`, RRUFF format | n/a | |
| Batch export (single files or one multi-column file, original or chosen folder, numbered names) | done | R Export › Text files… (one per highlighted spectrum) and Combined CSV… (all shown spectra on a common grid, one column each) |
| Copy data to clipboard (x,y,y… when x is shared) | done | R Export |
| Save plot as png / svg / pdf | done | R Export |
| Save plot as eps / tiff, resolution choice | done | R Export › Export plot… (PNG / SVG / PDF / EPS / TIFF at a chosen dpi) |
| Save plot as bmp / emf / wmf | n/a | Windows metafiles; svg/pdf are the vector route |
| Copy plot to clipboard | done | R Export |
| Print, print preview | done | R Export › Print… (print dialog; no preview) |
| New view types, duplicate tab, detach tab | n/a | PRISM uses one Library and workspaces instead of tabs |

## 3. Spectra: list management

| Spectragryph | Status | PRISM |
|---|---|---|
| Remove last / all / selected spectra | done | R Spectra › Remove / Remove all (undo); L (delete, clear, undo) |
| Hide / show spectra from the legend | done | R spectra list: the tick shows a spectrum, the highlight makes it the target; Show all / Hide all / Only selected |
| Colorize by property (y at x, min, max, average, std, area, peak position, CoG, time stamp) | done | R Spectra › Colour by property (Apply / Reset colours) |
| Spectrum properties: line colour, type, width per spectrum | done | R Spectra › Properties… (also right-click; legend label, offset, scale) |
| Spectrum data: excitation wavelength, temperature, concentration, path length | done | R Spectra › Properties… (laser nm, temperature, concentration, notes; no path length) |
| Information / LogText (instrument metadata) | partial | the OMNIC fields are imported into the spectrum metadata; Properties… shows and edits laser, temperature, concentration, notes |
| Sort A→Z, Z→A, inverse, custom, by temperature, by wavelength | done | R Spectra › Sort (name, laser, time, maximum, both directions); L reorder |
| Legend text editing | done | L rename |
| Undo (multi-step) / Redo | partial | undo done; redo later |
| Duplicate tab, drag-copy between tabs, transfer to Automate/Identify | n/a / done | Raman ID and Sequence read the same Library |
| Flip through N spectra at a time | later | the list selection does this manually |

## 4. Plot and view

| Spectragryph | Status | PRISM |
|---|---|---|
| Zoom, pan, reset | done | matplotlib toolbar |
| Multicursor (values of all spectra at x, copy) | done | R Display › Cursor readout |
| Automated peak labels: x / y / both, threshold %, prominence, digits, angle, one or all spectra | done | R Display › Peak labels (threshold %) + Plot settings (x / y / both, digits, angle); all shown spectra |
| Manual labels: add, edit, move, remove, restore | partial | R Display › Annotate (a click pins an x, y label), Clear removes them all; no edit / move of one label |
| LIBS / XRF / gamma element labels | n/a | |
| Custom axis titles, custom boundaries | done | R Display |
| Stacked spectra (own y axes), offsets | done | R Display: separate panels, or stacked with shift |
| Scale & shift by mouse, spread evenly (waterfall), make permanent | partial | stacked shift (Display), per-spectrum offset / scale in Properties…, Process › Offset / stretch makes them permanent; no mouse dragging |
| Grid, grid line style, plot background colour | done / later | grid done |
| Display mode lines / points, global line thickness | done | R Display |
| 16 colour palettes, custom palettes | done | R Display › Colours: distinct schemes plus Viridis / Plasma / Cividis / Rainbow / Tab10 / Set2; per-spectrum colours in Properties… |
| Axes font, legend font, legend placement (fixed/floating), legend show/hide | done | R Display › Legend inside / outside / none; Plot settings › Font size |
| Flip x axis | done | R Display › Flip x |
| Change x-axis type on the fly (nm, cm⁻¹, Raman shift, eV) | done | R Display › Show x as (the unit of the data and the laser λ are in Plot settings; display only) |
| y-axis types (absorbance, transmittance, counts/s, …), log-intensity display | partial | R Display › Log y; the Raman unit is the measured intensity |
| Spectroscope view (rainbow bar) | n/a | |
| App styles, dark mode | done | View › Dark mode |

## 5. Process

All with Spectragryph's common options: treat all selected spectra, keep
or replace the originals, result naming.

| Spectragryph | Status | PRISM |
|---|---|---|
| Simple baseline (subtract mean of a range) | done | R Process › Baseline › simple |
| Advanced baseline: linear, adaptive, scattering; baseline spectrum output | done (different algorithms) | R Process › Baseline › arPLS, ALS, polynomial, spline, rubberband, linear through two ranges, or through anchor points clicked on the plot; output the baseline itself |
| x offset / stretch, y offset / stretch | done | R Process › Offset / stretch |
| Normalize to peak, area, value at x | done | R Process › Normalize (+ min–max) |
| Simple smoothing (SG order 3) | done | R Process › Smooth |
| Advanced smoothing: moving average (rectangular / triangular), SG (interval, order), percentile filter, baseline-selective | done (baseline-selective later) | R Process › Smooth: Savitzky-Golay, moving / triangular average, median, percentile filter |
| Remove spikes (max width, backward) | done | R Process › Despike (z, window, picked positions) |
| Remove a peak (interpolate) | done | R Process |
| Cut off (crop) | done | R Process |
| Correct system sensitivity (divide by a curve) | done | R Transform › Divide by reference (explicit entry) |
| SNV, MSC, detrending | done | R Process › Scatter corrections (detrend = polynomial baseline) |

## 6. Transform

| Spectragryph | Status | PRISM |
|---|---|---|
| Add, subtract, multiply, divide spectra (one against all) | done | R Transform › Arithmetic |
| Averaging (mean, median; by naming/numbering schemes) | done | R Transform › Average / median of selected |
| Merging (average overlaps, interpolate gaps) | done | R Transform › Merge |
| Scaled subtraction with live slider | done | R Transform › Scaled subtraction (the slider previews A − k·B live) |
| Transmittance / reflectance / absorbance from intensity | later | not a Raman quantity; arithmetic + log10 already allow it |
| X-axis pixel calibration (polynomial from known lines, `calibration_lines.csv`, Jacobian) | done | R Transform › Calibrate x (pairs, order, residuals) |
| Raman shift from wavelength (laser λ, cut Rayleigh, Jacobian) | done | R Transform › Convert x axis (laser λ, begin spectrum at = cut below) |
| log(1/R), Kubelka–Munk | later | one-line transforms; added to Calculations if asked |
| Derivative 1st–4th with smoothing | done | R Transform |
| Interpolate / resample (start, end, step) | done | R Transform |
| Spectrum from image | n/a | |

## 7. Analyze

| Spectragryph | Status | PRISM |
|---|---|---|
| Peaks & FWHM list (threshold, prominence, inverted, copy) | done | R Analyze › Peaks & FWHM (detected, plus peaks pointed at by hand, added or replacing) |
| Integration (range, zero or tilted baseline, average, copy) | done | R Analyze › Integrate (zero or linear baseline; mean area in the results note; copy / save CSV) |
| Centre of gravity (range, baseline) | done | R Analyze › Integrate (centre-of-gravity column) |
| Extract data (properties of every spectrum in a range → table) | done | R Analyze › Extract data (one row per spectrum) and Statistics |
| Concentration from spectrum (Lambert–Beer), from weighing | n/a | UV-Vis; not Raman |
| Film thickness (interference) | n/a | |
| Create output value (DAC hardware) | n/a | |
| Mixture analysis (multilinear regression against components) | done | C › Linear-combination fit; R Analyze shortcut |
| Gaussian deconvolution | done (richer) | F: Gaussian, pseudo-Voigt, Voigt, EMG, one-step iterations |
| Identification (multi-library search, threshold, display refs) | done | ID |

## 8. Identify (libraries)

| Spectragryph | Status | PRISM |
|---|---|---|
| Create a database: name, description, descriptors, methods | partial | reference library: name, laser λ, source, notes; no free descriptors |
| Database content grid, filter, search bar | partial | Manage library dialog (list, rename, remove); filter later |
| Search definitions: algorithm (spectral similarity = Pearson), x range, weight, descriptor restrictions | done / later | Pearson and Euclidean, raw / smoothed / derivatives, x range; weights later |
| Import reference spectra (any format; OMNIC `.lbt`/`.lbd`, GRAMS `.idx` libraries) | done / n/a | from the Library (any imported file); vendor libraries n/a |
| Reference IDs grouping several spectra of one material | later | |
| Load samples, execute search, hit list with rating, show hits | done | ID |
| Save / load search results | later | |
| Excel / PDF search reports with logo | later | CSV now |
| Live library search during acquisition | n/a | |

## 9. Automate

| Spectragryph | Status | PRISM |
|---|---|---|
| Processing sequence: data input (files / folder, batches), Process / Transform / Analyze steps, data output (show / save) | done | R Sequence: every Process / Transform / Analyze operation is added as a step with its parameters (Add to sequence); run on the highlighted spectra, on all shown or on a folder of files; save / load as JSON; the Log is the protocol |
| Run / stop / live processing / protocol | done (run, protocol) | R Sequence; no stop button, no live processing (no acquisition) |
| Command-line switches for sequences | later | |

## 10. Acquire, Fluorescence / EEM, options

| Spectragryph | Status | PRISM |
|---|---|---|
| Spectrometer control (Andor, Avantes, Ocean Optics, Wasatch, …), dark / reference / blank, live processing, live library search, webcams | n/a | the DXR is driven by OMNIC |
| Fluorescence / EEM ribbon | n/a | |
| File › Options: default axis types on open/save, flip-through thresholds | later | PRISM remembers window, modules, peak-search options |
| Keyboard shortcuts for the frequent functions | partial | Ctrl+O/I/E/S/Z exist; no F-keys in the workbench yet |
| Command line: open files, colours, instance handling | later | |

## The workbench, as built

The Raman workspace is a workbench: a ribbon of eight tabs across the top
(Spectra, Display, Process, Transform, Analyze, Identify, Sequence,
Export), the spectra list on the left (tick = shown, highlight = target of
an operation, colour swatches, right-click for properties), the plot in the
centre and a Results / Log pane under it. Under the ribbon, one detail
panel at a time: the parameter form of the chosen operation (Pick fills an
x or a range from a click on the plot, Preview draws the result dashed,
Apply writes new spectra into the Library, undoable, with Treat all shown
spectra and Hide originals), the Plot settings (titles, limits, font, unit
of the data and laser, peak-label options) or the Sequence steps (shown
with the Sequence tab; double-click a step to reopen it). Every tab fits
in two rows at 1500 px wide; the ribbon scrolls sideways on narrower
screens. The operations are declared once with their parameters
(`raman/raman_ops.py`): the forms, the previews and the replayable
sequences come from the same definition.

Peak Fitting keeps its workspace and has the stepwise controls a user of
Origin expects: 1×, 2×, 5×, 10× iterations, Fit until converged, and a
step size (100 / 50 / 25 / 10 % of each Levenberg–Marquardt update) to
watch the model walk towards the data.
