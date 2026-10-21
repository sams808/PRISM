<!-- Generated from qt_help.py by make_user_guide.py. Edit qt_help.py, then regenerate. -->

# PRISM — quick-start guide

**The flow:** bring data in through the **Library**, then switch to the
workspace for your technique in the left rail. Everything you derive
(baseline-subtracted, combined, fitted) lands back in the Library as a new
spectrum. **File › Save project** keeps the whole session in one
`.prism` file (older project files still load).

## Home

The first page. **Modules**: every technique is a card with an
*Enabled* switch — a fresh install starts with everything off, so
switch on what this computer is for (the **Raman station** preset enables
Raman, Fitting, Processing and Figures; **All modules** and **None** do
what they say). Disabled modules disappear from the rail until you switch them
back on here; the choice is remembered per user. **Reference databases**
lists what is installed on this computer (RRUFF, AMCSD, XRD card databases,
CIF folders, your Raman reference library) and packs them with
**Export databases…** into one folder that another PRISM installs with
**Import a database pack…** — only for people who hold the same
rights to those databases. Data files dropped anywhere on the window are
imported into the Library; **File › Paste data from clipboard**
imports a table copied from Excel, Origin or a text editor.

## Library

**Import files…** auto-detects the format (plain XY, TA SDT thermal exports,
SAXS EDF, Rigaku `.rasx`, JCAMP-DX). If it guesses wrong, use
**Custom import…** (Ctrl+I) to pick the parser and the X/Y columns yourself, with a
preview. Right-click spectra to rename, duplicate, reorder, export as text
(Ctrl+E), combine (sum / average / weighted subtraction), or delete.
**Undo (Ctrl+Z)** steps back through library changes: deletes, renames,
duplicates, combine results, applied baselines, and accepted mineral IDs.

## Baseline

Pick a method — **arPLS** is the automatic default; polynomial / spline /
rubberband fit *through regions you choose*. Type regions like
`100-400; 1800-2600` or toggle **Add region by dragging** and sweep them
on the plot. **Preview** shows raw + baseline + subtracted; **Apply** creates a
`_bl` spectrum.

**Tip for glasses:** broad bands need a stiff arPLS — use λ around
**1e7**. At the usual 1e5 the baseline eats the band itself. (Verified against
the ISG pressure series: λ=1e7 beat a hand-drawn baseline.)

## Raman (Simple Plot)

Multi-select spectra to overlay them: separate panels or stacked with an
offset. Load CIF files to overlay predicted Bragg positions (managed per-CIF:
color, label, visibility). Difference mode shows A−B for exactly two selected
spectra. Toggle **Annotate on click** to pin peak-position labels.
**Peak labels** marks every peak above the threshold (% of the intensity
range) with its position, and **Peak table…** lists position, height, FWHM
(full width at half prominence) and area for the selected spectra, to copy or
save as CSV. **Cursor readout** shows the value of every plotted spectrum at
the mouse position. Line width, lines/points and legend placement are in the
style row; **Copy plot** and **Copy data** put the figure (image) or the
selected spectra (tab-separated text) on the clipboard.

## Peak Fitting

Set up components three ways: in the **Fit param.** table, with
**Auto-find peaks** (its *detection limit* spinner sets how strong a
candidate must be, in units of the noise — lower it to catch weak peaks), or
with **Pick peaks on plot** — toggle it and click each peak apex.
Then **Fit !**. Shapes: G (Gaussian), GL (pseudo-Voigt), V (true Voigt),
EMG (asymmetric, signed skew). The `FWHM=#` column links a component's
width to another's, and the **Name** column labels a component (e.g.
"ν1 PO4") — names follow into the legend, reports, and CSV exports.
Reports include R², ±1σ errors, and peak centroids;
**Conf. intervals** runs rigorous F-test profiling. Save a configuration as a
*model* to reuse it — models are also the recipes **Multi-Fit** applies
across many spectra at once.

**Origin-like mode** is a faithful stepwise Levenberg-Marquardt:
*1 iteration* performs exactly one damped parameter update (watch the
curve move), and *Fit until converged* repeats until the relative χ²
change drops below the tolerance — the same behavior as Origin's NLFit
dialog.

**Width convention:** the "FWHM" fields have always been *half*-widths
(HWHM) — a historical convention kept so old saved models stay valid. Double
them when quoting true FWHM.

## XAS

Import EasyXAFS ZIPs, CSVs, or Athena `.prj` from the workspace's own
buttons. Tabs follow the workflow: Pre-processing (smoothing, Bragg
angle→energy correction, click-based tie-point alignment) → μ(E) Builder
(with deglitching) → Normalization / EXAFS (Larch) → Analysis (average or sum
repeat scans, difference, linear-combination fitting, PCA species count) →
Export (Athena formats). Importing an Athena `.prj` needs no Larch.

## DTA / Thermal

Pick X/Y/dY channels, compute Tg three ways (double tangent, parallel
tangent, |dY| max) — the result panel says whether the methods agree.
"Calculs" integrates or finds extrema over a range; Batch processes a folder.

### Cyclic and multi-step programs

A run that heats, holds and cools several times is split into **segments**
— one per ramp or hold — and every analysis is scoped to the segment chosen in
the selector next to the file name. This matters because temperature is not a
usable x-axis across a whole cyclic run: sorting by it interleaves the ramps,
and a Tg computed from the result is meaningless. Aggregate views ("All data
vs time", "All heating segments") plot but do not compute, and say so.

The **Program** tab shows the method the instrument actually ran, read
from the `OrgMethod` lines in the file's own header, beside what the
furnace did. A TA `.mth` method file can be loaded to cross-check it;
the data file's header stays authoritative, and a method that does not match the
data is reported rather than applied. Two diagnostics are worth reading before
quoting a rate: *target gap* is how far a ramp finished from its setpoint
(measured after the soak that follows it, which is where a ramp completes its
approach), and *control lost below* is the temperature under which the
furnace stopped holding the programmed rate — below that point a nominal
"20 °C/min" cool is not what the specimen experienced. A programmed hold the
furnace never settled into is reported as *absorbed* into the adjacent
ramp, which is not the same claim as never executed.

Export writes one row per segment, carrying the segment's rate, span, cycle
and matched program step alongside the Tg columns. Batch runs either the whole
file, every segment, or only segments matching the current one.

### Rate series (Lasocka and Arrhenius)

The **Rate series** tab builds a multi-rate series and fits it. Rows come
from the heating segments of the cyclic run on screen, from separate files (one
specimen per rate — the usual design), from records already imported, or typed
by hand for a literature value. A file's heating rate is read from its own
program, so a set of runs at 5/10/15/20/30 °C/min needs no typing.

Values are blank until **Compute all values**, which measures every row
with one shared window and method; a row that cannot be measured keeps its place
and shows why, rather than dropping silently out of the fit. Available fits are
Lasocka (Tg = A + B·ln q), Moynihan and Kissinger-form activation energies on
Tg, and Kissinger, Flynn–Wall–Ozawa and Augis–Bennett on a peak temperature.
Absolute temperature is used where the model requires it, independently of the
°C/K display switch.

Nothing here refuses to fit. Objections — too few distinct rates, a
duplicated rate, a short rate span, a non-monotonic trend, mixed thermal
history, mass loss along the series, a cooling-segment rate, a negative
activation energy, the same segment added twice, a temperature column too
coarsely rounded to resolve the transition — appear as warnings above the table,
in the figure annotation, and in the exported CSV, and the fit is still
produced. Read them before quoting a slope: successive cycles of one specimen
confound heating rate with thermal history, and on a glass that crystallizes
during its first heat the later cycles are not the same material.

### Reading the clock and the rate

The acquisition clock's unit is taken from its own column name, so an export
whose first column is `Time (sec)` or `Time [h]` yields
the same rates as one in minutes; the series notes which unit it read. Windows
and segment durations are always shown in minutes whatever the file used.

Where a method is present in the file, a programmed rate is believed even
when it is very slow. Where no method is present, segment detection falls back
to the trace's own noise level, and a genuine ramp slower than about
0.2 °C/min is reported as a hold — if a run that slow matters, state the
programmed rate by loading its method file.

## Raman ID (RRUFF)

**Match by** chooses the search: *peak positions* (below) or
*spectrum similarity*, the Pearson correlation (or a normalised Euclidean
distance) of the whole query against each reference on a common grid — raw,
smoothed, or as first/second derivatives (derivatives remove a fluorescence
slope), optionally inside an x range. The sources are RRUFF's measured spectra
(after the database filters) and **your own reference library**: **Add
query to my library…** stores a spectrum you measured (name, laser λ,
source, notes), **Manage library…** renames, removes, exports and imports
it, and the library travels with a database pack (Home). A score of 1 means an
identical shape; hits from the library are marked "my library".

Auto-find (or type) your spectrum's peak positions, then **Find matches**:
candidates are ranked by peak overlap, each showing its **laser wavelength** —
relative intensities vary between wavelengths, so judge the overlay yourself.
The **Database filters** restrict candidates before ranking: laser λ
(matched ±2 nm, so 532 also covers 532.6), oriented/unoriented, high-res vs
broad-scan, and quality category. Shift/Ctrl-click several result rows to
overlay multiple references at once.
Nothing is labeled until you click **Accept** — and Accept works
iteratively for mixtures: the accepted phase is recorded, the peaks it
explains are removed from the query, and the table immediately shows matches
for the REMAINING peaks (already-accepted references are excluded). Repeat
until every peak is explained; each accept is one Ctrl+Z away. One more click
overlays the matched mineral's predicted XRD pattern (AMCSD structure) in
the Raman workspace (needs the AMCSD download below).
**Download RRUFF database…** / **Download AMCSD structures…** fetch
and index those databases directly from rruff.net — no Python needed, runs
in the background, safe to re-run if interrupted.

## HT-XRD

Import a whole folder of patterns; temperatures come from `.rasx`
metadata or a filename template like `scan_???.xy`. The waterfall is
colored by temperature. The **Maps** tab adds a 2D heatmap
(linear/log/sqrt/power color scales), a difference map or difference waterfall
vs a reference pattern ("first", an index, or a temperature), a 3D surface,
an optional time axis from the heating rate, and dashed peak-guide lines
(`{slice:2θ; slice:2θ}` anchors, interpolated).

**Tracking**, three ways. (1) **Pick track guide** on the waterfall:
click the same peak at two (or more) temperatures — bottom and top, say — then
**Track picked guide** follows the peak along that line; one click from an
intermediate temperature also works for a peak that appears or dies.
(2) **Auto-track all peaks** finds every peak of the first pattern and
tracks them all (the generated windows appear in the field for editing).
(3) Typed windows — `28.5-29.5 @ 28.98; 31-32` — where the
`@` anchor picks WHICH peak to track when a window holds more than
one. In every mode the fit is seeded from the previous pattern (so it follows
a drifting peak instead of jumping to a stronger neighbor), and a peak weaker
than the *Absence σ* noise threshold is reported as absent instead of a
garbage fit — vanished/appeared peaks are flagged as transition signatures
alongside fit-quality anomalies.

## Clustering

Select a series (e.g. a multi-point map), choose KMeans or hierarchical and
a cluster count: PCA scatter colored by cluster, per-cluster mean spectra, and
an assignment table.

## Calculations

Every spectrum operation in one place. Between spectra: add, subtract,
multiply, divide, average, weighted sum, and **modulated addition**
(A + w(x)·B with a constant / linear-ramp / Gaussian envelope). Per
spectrum: scale/offset, normalizations (max, area, min-max), log/ln/exp/√/
power/reciprocal/|y|, x-shift and x-scale calibration, crop, resample,
smoothing (Savitzky-Golay, moving average, median), cosmic-ray despiking,
Savitzky-Golay derivatives (1st to 4th), cumulative integral. Raman tools:
**normalize to a value** at x₀, **subtract a range mean** (offset
baseline), **remove a peak** (interpolate across it), **SNV** and
**MSC** (scatter corrections), **median of spectra**, x-axis
**unit conversion** between wavelength, wavenumber, Raman shift (give the
laser λ) and photon energy, with an optional Jacobian, and **x-axis
calibration** against reference lines ("measured -> reference" pairs,
polynomial order chosen by you, residuals reported). Analysis:
**linear-combination fitting** (target ≈ Σ cᵢ·refᵢ, free or non-negative
coefficients, with R² and percentages) and summary statistics.
*Selection order matters*: the first-selected spectrum is A (or the LCF
target). Preview shows inputs faint + result bold; Apply adds derived
spectra to the Library (undoable).

## XRD ID

QualX-style phase identification against YOUR OWN card databases —
PRISM ships none. Download any database you have the rights to use (any
QualX-format .sq works) and register it with **Add database…** (or
**Add folder…** for several at once): a QualX-format file is indexed
once locally; a PRISM-format .sq (e.g. shared by a colleague) is used in
place. Check any number of registered databases to probe them all in one
search — every card keeps its source and code, and results say which
database each hit came from.
A folder of your own CIFs can be registered as a database too
(**Add CIF folder…**): each structure's powder pattern — positions AND
intensities, so a centred lattice contributes none of the reflections it
extinguishes — is calculated once and indexed like a card.
**Auto-find peaks** removes the background first and reads its detection
limit in σ of the LOCAL noise — counting noise scales as √counts, so it is
several times larger on an amorphous halo than at high angle, and a single
global threshold turns that halo into dozens of false "reflections". The
default is 4σ (measured: 3σ left ~9 spurious peaks per pattern, 4σ left 1.4,
for 0.1% of the real ones). **Peak search options…** exposes the rest —
noise window, SNIP background width, smoothing, minimum peak separation, the
Kα2 test, a 2θ search range, an edge margin and the peak cap — applied live
against the plot and remembered between sessions; **show threshold** draws
the background and the detection limit so a found or missed peak can be
explained by looking.
Auto-find (or type) the pattern's 2θ peaks, set λ and the match tolerance,
optionally restrict by chemistry (must-contain / must-exclude elements) or
source, then **Search match**: candidates ranked by figure of merit
(intensity-weighted coverage both ways, weighted by how closely each line
lands), previewed as stick patterns under the query (shift/ctrl-click to
compare several). **Accept** is iterative for mixtures, exactly like
Raman ID: the phase is recorded, its peaks leave the query, the rest is
re-searched; Ctrl+Z undoes. The **Accepted phases** tab is the ledger —
withdraw any phase (its peaks return to the search) and export the
accepted structures as CIFs for a Rietveld refinement.
**Check Raman-identified phases here** overlays the XRD reference lines
of every phase accepted in Raman ID — the Raman→XRD cross-check in one
click. The Card browser tab looks up any card by name/mineral/formula.

## Figures

Publication-figure building (the Origin-inspired module). **XY builder**:
stack library spectra as layers — per-layer plot type (line, scatter,
line+symbols, sticks, filled area, bars, steps), color, vertical offset, and
multi-panel assignment — with Publication/Presentation/Poster style presets,
log axes, and export at exact centimeter size and dpi (PNG/SVG/PDF/TIFF).
**CIF stick overlays** add reference phases from structure files as the
row of short bars under the data that a Rietveld plot uses: each phase gets
its own band so several never overprint, as *bars* (intensity-scaled),
*ticks* (equal-height reflection markers) or *overlay* (over the
data), with editable color, panel and row height and optional hkl labels on
the strongest reflections. Positions and heights are calculated from each
structure at the λ you set, so extinct reflections are absent and a
bar's height means something.
**Point fitting**: Origin's classic models (linear, polynomials,
exponential decay/growth, power law, logarithmic, Boltzmann sigmoid,
Gaussian, Lorentzian, Arrhenius) with parameters ±1σ and R².
**Ternary**: barycentric composition plots from a CSV table, optional
color-mapped value column. **Raman ↔ XRD**: the cross-technique figure —
your Raman spectrum with its accepted Mineral-ID phases above the XRD
pattern with each accepted phase's reference stick pattern.

## SAXS/WAXS

Small/wide-angle scattering (first ported from the author's own POMME suite). Import 1D curves — two/three-column files or named-column tables with q, I,
sigma and optionally dQ (resolution), a statistical sigma and flags; **Reduction** subtracts an empty capillary
from a 1D curve (manual, auto, transmission or physics-based scaling); **Analysis** fits Guinier (Rg), generalized
Porod (slope + background) and pseudo-Bragg peaks with auto region detection; **WAXS** auto-finds and fits
pseudo-Voigt peaks (d-spacings, crystallinity index); curves can be sent to the Library. **Frames** reduces the raw 2D frames of a beamstopless
instrument (inventory, refined beam centres, pairing with the empties, sector reduction with loading classes);
**Fit** is the physical fitting workflow — grain power law + high-q term as the base, a model-free gate
against a reference curve deciding whether a structure exists, then spheres (absolute with a contrast),
Teubner-Strey, Debye and unified levels compared by BIC under explicit physical rules; **Series figures** builds
publication figures and tables from a series table. The same chain runs headless from one JSON project file
(`scripts/saxs_pipeline.py`) — details in the module guide and docs/SAXS_FITTING.md.
The XAS workspace also gained a **Sample mass** tab — the Hephaestus
calculator on oxide compositions (mol%/wt%) or formulas: masses for total
μt = 1 / 2.5 and edge step Δμt = 1 in a pellet of chosen diameter.

## Glass

Composition-based glass property calculation. Paste (or load as CSV) a
composition table — header = oxides, one row per sample, mol% or wt% —
then: **Optical basicity &Lambda;** (oxygen-weighted Duffy mixing with the
recommended per-oxide values of Rodriguez & McCloy, PNNL-20184 Table B.1),
and **GlassNet predictions** (Tg, viscosity, density, refractive index and
~80 more; Cassar 2023, trained on SciGlass — estimates to validate, not
measurements). Compare against SciGlass, INTERGLAD, glassproperties.com.

## Shortcuts

`Ctrl+O` import · `Ctrl+I` custom import ·
`Ctrl+E` export · `Ctrl+S` save project ·
`Ctrl+Shift+O` open project · `Ctrl+Z` undo ·
`Ctrl+Q` quit · `F1` this guide.
View menu: dark mode, Python console (the live app objects are in scope).


---

## PRISM 2.8.0

*Platform for Research In Spectroscopy & Materials*

Import, process, and analyze scientific spectra: Raman, XAS/XANES/EXAFS,
DTA/DSC/TGA, XRD (including phase identification and high-temperature
series), SAXS.

Source: `github.com/sams808/PRISM`

### Please cite the reference databases that contribute to your work

**RRUFF** (Raman reference spectra): Lafuente B, Downs R T, Yang H,
Stone N (2015) "The power of databases: the RRUFF project." In: Highlights
in Mineralogical Crystallography, pp 1–30. Also acknowledge each matched
sample's owner/source shown in the Raman ID workspace.

**AMCSD** (crystal structures used for XRD overlays): Downs R T,
Hall-Wallace M (2003) "The American Mineralogist Crystal Structure
Database." American Mineralogist 88, 247–250.

© 2026 NOME Group, Washington State University.
Licensed under the MIT License — see LICENSE.

### Built on

numpy · scipy · pandas · matplotlib · lmfit · rampy · scikit-learn ·
PySide6 · xraylarch (optional, XAS)
