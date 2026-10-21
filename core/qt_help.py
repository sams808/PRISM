"""
qt_help.py — in-app Help: the F1 quick-start guide (per-workspace, written
for a group member opening the app for the first time) and the About
dialog (versions + the citations the reference databases ask for).

docs/USER_GUIDE.md is generated from this file — run tools/make_user_guide.py
after editing, never edit the markdown directly.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QDialog, QTextBrowser, QVBoxLayout, QWidget

# Single source of the app version: shown in the window title and About,
# and matched by the git tag at release time.
APP_VERSION = "2.8.0"
APP_NAME = "PRISM"
APP_TAGLINE = "Platform for Research In Spectroscopy & Materials"


def asset_path(name: str) -> str:
    """Path to a bundled asset (repo-root assets/ in dev, or the PyInstaller
    bundle dir when frozen)."""
    import os
    import sys
    if hasattr(sys, "_MEIPASS"):
        base = sys._MEIPASS
    else:
        # this file lives in core/, so assets/ is one level up at repo root
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "assets", name)


# Credits — shown from the Modules toolbar and the Help menu.
# <!-- PRISM was imagined, specified, and battle-tested by Sam Souda. -->
CREDITS_HTML = """
<div style='text-align:center'><h2>PRISM</h2>
<p><i>Platform for Research In Spectroscopy &amp; Materials</i></p></div>
<p>Developed in the NOME group, Washington State University.<br>
Supported by the U.S. Department of Energy.<br>
Thanks to Prof. John S. McCloy for the trust.</p>
<p>Reference databases: RRUFF (Lafuente et al. 2015), AMCSD (Downs &amp;
Hall-Wallace 2003), plus whatever XRD card databases you register yourself
&mdash; full citations in About.</p>
<!-- conceived and directed by Sam Souda -->
<p style='font-size:8pt; color:#888'>made with ChatGPT, improved with Claude&nbsp;&nbsp;&middot;&nbsp;&nbsp;s.s.</p>
"""
HELP_HTML = """
<h1>PRISM — quick-start guide</h1>

<p><b>The flow:</b> bring data in through the <b>Library</b>, then switch to the
workspace for your technique in the left rail. Everything you derive
(baseline-subtracted, combined, fitted) lands back in the Library as a new
spectrum. <b>File &rsaquo; Save project</b> keeps the whole session in one
<code>.prism</code> file (older project files still load).</p>

<h2>Home</h2>
<p>The first page. <b>Modules</b>: every technique is a card with an
<i>Enabled</i> switch &mdash; a fresh install starts with everything off, so
switch on what this computer is for (the <b>Raman station</b> preset enables
Raman, Fitting, Processing and Figures; <b>All modules</b> and <b>None</b> do
what they say). Disabled modules disappear from the rail until you switch them
back on here; the choice is remembered per user. <b>Reference databases</b>
lists what is installed on this computer (RRUFF, AMCSD, XRD card databases,
CIF folders, your Raman reference library) and packs them with
<b>Export databases&hellip;</b> into one folder that another PRISM installs with
<b>Import a database pack&hellip;</b> &mdash; only for people who hold the same
rights to those databases. Data files dropped anywhere on the window are
imported into the Library; <b>File &rsaquo; Paste data from clipboard</b>
imports a table copied from Excel, Origin or a text editor.</p>

<h2>Library</h2>
<p><b>Import files…</b> auto-detects the format (plain XY, TA SDT thermal exports,
SAXS EDF, Rigaku <code>.rasx</code>, JCAMP-DX). If it guesses wrong, use
<b>Custom import…</b> (Ctrl+I) to pick the parser and the X/Y columns yourself, with a
preview. Right-click spectra to rename, duplicate, reorder, export as text
(Ctrl+E), combine (sum / average / weighted subtraction), or delete.
<b>Undo (Ctrl+Z)</b> steps back through library changes: deletes, renames,
duplicates, combine results, applied baselines, and accepted mineral IDs.</p>

<h2>Baseline</h2>
<p>Pick a method — <b>arPLS</b> is the automatic default; polynomial / spline /
rubberband fit <i>through regions you choose</i>. Type regions like
<code>100-400; 1800-2600</code> or toggle <b>Add region by dragging</b> and sweep them
on the plot. <b>Preview</b> shows raw + baseline + subtracted; <b>Apply</b> creates a
<code>_bl</code> spectrum.</p>
<p><b>Tip for glasses:</b> broad bands need a stiff arPLS — use λ around
<b>1e7</b>. At the usual 1e5 the baseline eats the band itself. (Verified against
the ISG pressure series: λ=1e7 beat a hand-drawn baseline.)</p>

<h2>Raman (Simple Plot)</h2>
<p>Multi-select spectra to overlay them: separate panels or stacked with an
offset. Load CIF files to overlay predicted Bragg positions (managed per-CIF:
color, label, visibility). Difference mode shows A−B for exactly two selected
spectra. Toggle <b>Annotate on click</b> to pin peak-position labels.
<b>Peak labels</b> marks every peak above the threshold (% of the intensity
range) with its position, and <b>Peak table…</b> lists position, height, FWHM
(full width at half prominence) and area for the selected spectra, to copy or
save as CSV. <b>Cursor readout</b> shows the value of every plotted spectrum at
the mouse position. Line width, lines/points and legend placement are in the
style row; <b>Copy plot</b> and <b>Copy data</b> put the figure (image) or the
selected spectra (tab-separated text) on the clipboard.</p>

<h2>Peak Fitting</h2>
<p>Set up components three ways: in the <b>Fit param.</b> table, with
<b>Auto-find peaks</b> (its <i>detection limit</i> spinner sets how strong a
candidate must be, in units of the noise — lower it to catch weak peaks), or
with <b>Pick peaks on plot</b> — toggle it and click each peak apex.
Then <b>Fit&nbsp;!</b>. Shapes: G (Gaussian), GL (pseudo-Voigt), V (true Voigt),
EMG (asymmetric, signed skew). The <code>FWHM=#</code> column links a component's
width to another's, and the <b>Name</b> column labels a component (e.g.
"ν1 PO4") — names follow into the legend, reports, and CSV exports.
Reports include R², ±1σ errors, and peak centroids;
<b>Conf. intervals</b> runs rigorous F-test profiling. Save a configuration as a
<i>model</i> to reuse it — models are also the recipes <b>Multi-Fit</b> applies
across many spectra at once.</p>
<p><b>Origin-like mode</b> is a faithful stepwise Levenberg-Marquardt:
<i>1 iteration</i> performs exactly one damped parameter update (watch the
curve move), and <i>Fit until converged</i> repeats until the relative χ²
change drops below the tolerance — the same behavior as Origin's NLFit
dialog.</p>
<p><b>Width convention:</b> the "FWHM" fields have always been <i>half</i>-widths
(HWHM) — a historical convention kept so old saved models stay valid. Double
them when quoting true FWHM.</p>

<h2>XAS</h2>
<p>Import EasyXAFS ZIPs, CSVs, or Athena <code>.prj</code> from the workspace's own
buttons. Tabs follow the workflow: Pre-processing (smoothing, Bragg
angle→energy correction, click-based tie-point alignment) → μ(E) Builder
(with deglitching) → Normalization / EXAFS (Larch) → Analysis (average or sum
repeat scans, difference, linear-combination fitting, PCA species count) →
Export (Athena formats). Importing an Athena <code>.prj</code> needs no Larch.</p>

<h2>DTA / Thermal</h2>
<p>Pick X/Y/dY channels, compute Tg three ways (double tangent, parallel
tangent, |dY| max) — the result panel says whether the methods agree.
"Calculs" integrates or finds extrema over a range; Batch processes a folder.</p>
<h3>Cyclic and multi-step programs</h3>
<p>A run that heats, holds and cools several times is split into <b>segments</b>
— one per ramp or hold — and every analysis is scoped to the segment chosen in
the selector next to the file name. This matters because temperature is not a
usable x-axis across a whole cyclic run: sorting by it interleaves the ramps,
and a Tg computed from the result is meaningless. Aggregate views ("All data
vs time", "All heating segments") plot but do not compute, and say so.</p>
<p>The <b>Program</b> tab shows the method the instrument actually ran, read
from the <code>OrgMethod</code> lines in the file's own header, beside what the
furnace did. A TA <code>.mth</code> method file can be loaded to cross-check it;
the data file's header stays authoritative, and a method that does not match the
data is reported rather than applied. Two diagnostics are worth reading before
quoting a rate: <i>target gap</i> is how far a ramp finished from its setpoint
(measured after the soak that follows it, which is where a ramp completes its
approach), and <i>control lost below</i> is the temperature under which the
furnace stopped holding the programmed rate — below that point a nominal
"20 °C/min" cool is not what the specimen experienced. A programmed hold the
furnace never settled into is reported as <i>absorbed</i> into the adjacent
ramp, which is not the same claim as never executed.</p>
<p>Export writes one row per segment, carrying the segment's rate, span, cycle
and matched program step alongside the Tg columns. Batch runs either the whole
file, every segment, or only segments matching the current one.</p>
<h3>Rate series (Lasocka and Arrhenius)</h3>
<p>The <b>Rate series</b> tab builds a multi-rate series and fits it. Rows come
from the heating segments of the cyclic run on screen, from separate files (one
specimen per rate — the usual design), from records already imported, or typed
by hand for a literature value. A file's heating rate is read from its own
program, so a set of runs at 5/10/15/20/30 °C/min needs no typing.</p>
<p>Values are blank until <b>Compute all values</b>, which measures every row
with one shared window and method; a row that cannot be measured keeps its place
and shows why, rather than dropping silently out of the fit. Available fits are
Lasocka (Tg = A + B·ln q), Moynihan and Kissinger-form activation energies on
Tg, and Kissinger, Flynn–Wall–Ozawa and Augis–Bennett on a peak temperature.
Absolute temperature is used where the model requires it, independently of the
°C/K display switch.</p>
<p>Nothing here refuses to fit. Objections — too few distinct rates, a
duplicated rate, a short rate span, a non-monotonic trend, mixed thermal
history, mass loss along the series, a cooling-segment rate, a negative
activation energy, the same segment added twice, a temperature column too
coarsely rounded to resolve the transition — appear as warnings above the table,
in the figure annotation, and in the exported CSV, and the fit is still
produced. Read them before quoting a slope: successive cycles of one specimen
confound heating rate with thermal history, and on a glass that crystallizes
during its first heat the later cycles are not the same material.</p>
<h3>Reading the clock and the rate</h3>
<p>The acquisition clock's unit is taken from its own column name, so an export
whose first column is <code>Time (sec)</code> or <code>Time [h]</code> yields
the same rates as one in minutes; the series notes which unit it read. Windows
and segment durations are always shown in minutes whatever the file used.</p>
<p>Where a method is present in the file, a programmed rate is believed even
when it is very slow. Where no method is present, segment detection falls back
to the trace's own noise level, and a genuine ramp slower than about
0.2 °C/min is reported as a hold — if a run that slow matters, state the
programmed rate by loading its method file.</p>

<h2>Raman ID (RRUFF)</h2>
<p><b>Match by</b> chooses the search: <i>peak positions</i> (below) or
<i>spectrum similarity</i>, the Pearson correlation (or a normalised Euclidean
distance) of the whole query against each reference on a common grid — raw,
smoothed, or as first/second derivatives (derivatives remove a fluorescence
slope), optionally inside an x range. The sources are RRUFF's measured spectra
(after the database filters) and <b>your own reference library</b>: <b>Add
query to my library…</b> stores a spectrum you measured (name, laser λ,
source, notes), <b>Manage library…</b> renames, removes, exports and imports
it, and the library travels with a database pack (Home). A score of 1 means an
identical shape; hits from the library are marked "my library".</p>
<p>Auto-find (or type) your spectrum's peak positions, then <b>Find matches</b>:
candidates are ranked by peak overlap, each showing its <b>laser wavelength</b> —
relative intensities vary between wavelengths, so judge the overlay yourself.
The <b>Database filters</b> restrict candidates before ranking: laser λ
(matched ±2 nm, so 532 also covers 532.6), oriented/unoriented, high-res vs
broad-scan, and quality category. Shift/Ctrl-click several result rows to
overlay multiple references at once.
Nothing is labeled until you click <b>Accept</b> — and Accept works
iteratively for mixtures: the accepted phase is recorded, the peaks it
explains are removed from the query, and the table immediately shows matches
for the REMAINING peaks (already-accepted references are excluded). Repeat
until every peak is explained; each accept is one Ctrl+Z away. One more click
overlays the matched mineral's predicted XRD pattern (AMCSD structure) in
the Raman workspace (needs the AMCSD download below).
<b>Download RRUFF database…</b> / <b>Download AMCSD structures…</b> fetch
and index those databases directly from rruff.net — no Python needed, runs
in the background, safe to re-run if interrupted.</p>

<h2>HT-XRD</h2>
<p>Import a whole folder of patterns; temperatures come from <code>.rasx</code>
metadata or a filename template like <code>scan_???.xy</code>. The waterfall is
colored by temperature. The <b>Maps</b> tab adds a 2D heatmap
(linear/log/sqrt/power color scales), a difference map or difference waterfall
vs a reference pattern ("first", an index, or a temperature), a 3D surface,
an optional time axis from the heating rate, and dashed peak-guide lines
(<code>{slice:2θ; slice:2θ}</code> anchors, interpolated).</p>
<p><b>Tracking</b>, three ways. (1) <b>Pick track guide</b> on the waterfall:
click the same peak at two (or more) temperatures — bottom and top, say — then
<b>Track picked guide</b> follows the peak along that line; one click from an
intermediate temperature also works for a peak that appears or dies.
(2) <b>Auto-track all peaks</b> finds every peak of the first pattern and
tracks them all (the generated windows appear in the field for editing).
(3) Typed windows — <code>28.5-29.5 @ 28.98; 31-32</code> — where the
<code>@</code> anchor picks WHICH peak to track when a window holds more than
one. In every mode the fit is seeded from the previous pattern (so it follows
a drifting peak instead of jumping to a stronger neighbor), and a peak weaker
than the <i>Absence σ</i> noise threshold is reported as absent instead of a
garbage fit — vanished/appeared peaks are flagged as transition signatures
alongside fit-quality anomalies.</p>

<h2>Clustering</h2>
<p>Select a series (e.g. a multi-point map), choose KMeans or hierarchical and
a cluster count: PCA scatter colored by cluster, per-cluster mean spectra, and
an assignment table.</p>

<h2>Calculations</h2>
<p>Every spectrum operation in one place. Between spectra: add, subtract,
multiply, divide, average, weighted sum, and <b>modulated addition</b>
(A + w(x)·B with a constant / linear-ramp / Gaussian envelope). Per
spectrum: scale/offset, normalizations (max, area, min-max), log/ln/exp/√/
power/reciprocal/|y|, x-shift and x-scale calibration, crop, resample,
smoothing (Savitzky-Golay, moving average, median), cosmic-ray despiking,
Savitzky-Golay derivatives (1st to 4th), cumulative integral. Raman tools:
<b>normalize to a value</b> at x₀, <b>subtract a range mean</b> (offset
baseline), <b>remove a peak</b> (interpolate across it), <b>SNV</b> and
<b>MSC</b> (scatter corrections), <b>median of spectra</b>, x-axis
<b>unit conversion</b> between wavelength, wavenumber, Raman shift (give the
laser λ) and photon energy, with an optional Jacobian, and <b>x-axis
calibration</b> against reference lines ("measured -&gt; reference" pairs,
polynomial order chosen by you, residuals reported). Analysis:
<b>linear-combination fitting</b> (target ≈ Σ cᵢ·refᵢ, free or non-negative
coefficients, with R² and percentages) and summary statistics.
<i>Selection order matters</i>: the first-selected spectrum is A (or the LCF
target). Preview shows inputs faint + result bold; Apply adds derived
spectra to the Library (undoable).</p>

<h2>XRD ID</h2>
<p>QualX-style phase identification against YOUR OWN card databases —
PRISM ships none. Download any database you have the rights to use (any
QualX-format .sq works) and register it with <b>Add database…</b> (or
<b>Add folder…</b> for several at once): a QualX-format file is indexed
once locally; a PRISM-format .sq (e.g. shared by a colleague) is used in
place. Check any number of registered databases to probe them all in one
search — every card keeps its source and code, and results say which
database each hit came from.
A folder of your own CIFs can be registered as a database too
(<b>Add CIF folder…</b>): each structure's powder pattern — positions AND
intensities, so a centred lattice contributes none of the reflections it
extinguishes — is calculated once and indexed like a card.
<b>Auto-find peaks</b> removes the background first and reads its detection
limit in σ of the LOCAL noise — counting noise scales as √counts, so it is
several times larger on an amorphous halo than at high angle, and a single
global threshold turns that halo into dozens of false "reflections". The
default is 4σ (measured: 3σ left ~9 spurious peaks per pattern, 4σ left 1.4,
for 0.1% of the real ones). <b>Peak search options…</b> exposes the rest —
noise window, SNIP background width, smoothing, minimum peak separation, the
Kα2 test, a 2θ search range, an edge margin and the peak cap — applied live
against the plot and remembered between sessions; <b>show threshold</b> draws
the background and the detection limit so a found or missed peak can be
explained by looking.
Auto-find (or type) the pattern's 2θ peaks, set λ and the match tolerance,
optionally restrict by chemistry (must-contain / must-exclude elements) or
source, then <b>Search match</b>: candidates ranked by figure of merit
(intensity-weighted coverage both ways, weighted by how closely each line
lands), previewed as stick patterns under the query (shift/ctrl-click to
compare several). <b>Accept</b> is iterative for mixtures, exactly like
Raman ID: the phase is recorded, its peaks leave the query, the rest is
re-searched; Ctrl+Z undoes. The <b>Accepted phases</b> tab is the ledger —
withdraw any phase (its peaks return to the search) and export the
accepted structures as CIFs for a Rietveld refinement.
<b>Check Raman-identified phases here</b> overlays the XRD reference lines
of every phase accepted in Raman ID — the Raman→XRD cross-check in one
click. The Card browser tab looks up any card by name/mineral/formula.</p>

<h2>Figures</h2>
<p>Publication-figure building (the Origin-inspired module). <b>XY builder</b>:
stack library spectra as layers — per-layer plot type (line, scatter,
line+symbols, sticks, filled area, bars, steps), color, vertical offset, and
multi-panel assignment — with Publication/Presentation/Poster style presets,
log axes, and export at exact centimeter size and dpi (PNG/SVG/PDF/TIFF).
<b>CIF stick overlays</b> add reference phases from structure files as the
row of short bars under the data that a Rietveld plot uses: each phase gets
its own band so several never overprint, as <i>bars</i> (intensity-scaled),
<i>ticks</i> (equal-height reflection markers) or <i>overlay</i> (over the
data), with editable color, panel and row height and optional hkl labels on
the strongest reflections. Positions and heights are calculated from each
structure at the &lambda; you set, so extinct reflections are absent and a
bar's height means something.
<b>Point fitting</b>: Origin's classic models (linear, polynomials,
exponential decay/growth, power law, logarithmic, Boltzmann sigmoid,
Gaussian, Lorentzian, Arrhenius) with parameters ±1σ and R².
<b>Ternary</b>: barycentric composition plots from a CSV table, optional
color-mapped value column. <b>Raman ↔ XRD</b>: the cross-technique figure —
your Raman spectrum with its accepted Mineral-ID phases above the XRD
pattern with each accepted phase's reference stick pattern.</p>

<h2>SAXS/WAXS</h2>
<p>Small/wide-angle scattering (first ported from the author's own POMME suite). Import 1D curves &mdash; two/three-column files or named-column tables with q, I,
sigma and optionally dQ (resolution), a statistical sigma and flags; <b>Reduction</b> subtracts an empty capillary
from a 1D curve (manual, auto, transmission or physics-based scaling); <b>Analysis</b> fits Guinier (Rg), generalized
Porod (slope + background) and pseudo-Bragg peaks with auto region detection; <b>WAXS</b> auto-finds and fits
pseudo-Voigt peaks (d-spacings, crystallinity index); curves can be sent to the Library. <b>Frames</b> reduces the raw 2D frames of a beamstopless
instrument (inventory, refined beam centres, pairing with the empties, sector reduction with loading classes);
<b>Fit</b> is the physical fitting workflow &mdash; grain power law + high-q term as the base, a model-free gate
against a reference curve deciding whether a structure exists, then spheres (absolute with a contrast),
Teubner-Strey, Debye and unified levels compared by BIC under explicit physical rules; <b>Series figures</b> builds
publication figures and tables from a series table. The same chain runs headless from one JSON project file
(<code>scripts/saxs_pipeline.py</code>) &mdash; details in the module guide and docs/SAXS_FITTING.md.
The XAS workspace also gained a <b>Sample mass</b> tab &mdash; the Hephaestus
calculator on oxide compositions (mol%/wt%) or formulas: masses for total
&mu;t = 1 / 2.5 and edge step &Delta;&mu;t = 1 in a pellet of chosen diameter.</p>

<h2>Glass</h2>
<p>Composition-based glass property calculation. Paste (or load as CSV) a
composition table &mdash; header = oxides, one row per sample, mol% or wt% &mdash;
then: <b>Optical basicity &Lambda;</b> (oxygen-weighted Duffy mixing with the
recommended per-oxide values of Rodriguez &amp; McCloy, PNNL-20184 Table B.1),
and <b>GlassNet predictions</b> (Tg, viscosity, density, refractive index and
~80 more; Cassar 2023, trained on SciGlass &mdash; estimates to validate, not
measurements). Compare against SciGlass, INTERGLAD, glassproperties.com.</p>

<h2>Shortcuts</h2>
<p><code>Ctrl+O</code> import · <code>Ctrl+I</code> custom import ·
<code>Ctrl+E</code> export · <code>Ctrl+S</code> save project ·
<code>Ctrl+Shift+O</code> open project · <code>Ctrl+Z</code> undo ·
<code>Ctrl+Q</code> quit · <code>F1</code> this guide.
View menu: dark mode, Python console (the live app objects are in scope).</p>
"""

ABOUT_HTML = f"""
<h2>{APP_NAME} {APP_VERSION}</h2>
<p><i>{APP_TAGLINE}</i></p>
<p>Import, process, and analyze scientific spectra: Raman, XAS/XANES/EXAFS,
DTA/DSC/TGA, XRD (including phase identification and high-temperature
series), SAXS.</p>
<p>Source: <code>github.com/sams808/PRISM</code></p>
<h3>Please cite the reference databases that contribute to your work</h3>
<p><b>RRUFF</b> (Raman reference spectra): Lafuente B, Downs R T, Yang H,
Stone N (2015) "The power of databases: the RRUFF project." In: Highlights
in Mineralogical Crystallography, pp 1–30. Also acknowledge each matched
sample's owner/source shown in the Raman ID workspace.</p>
<p><b>AMCSD</b> (crystal structures used for XRD overlays): Downs R T,
Hall-Wallace M (2003) "The American Mineralogist Crystal Structure
Database." American Mineralogist 88, 247–250.</p>
<p style='font-size:8pt; color:#888'>© 2026 NOME Group, Washington State University.
Licensed under the MIT License — see LICENSE.</p>
<h3>Built on</h3>
<p>numpy · scipy · pandas · matplotlib · lmfit · rampy · scikit-learn ·
PySide6 · xraylarch (optional, XAS)</p>
"""


class HelpDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None, *, html: str = HELP_HTML, title: str = "Quick-start guide"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(760, 620)
        layout = QVBoxLayout(self)
        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        self.browser.setHtml(html)
        layout.addWidget(self.browser)


# Per-module in-depth guides (Help > Module guides). Deliberately verbose --
# these are the training documents for new group members.
def _guide(title, body):
    return "<h1>" + title + "</h1>" + body


MODULE_GUIDES = {
    "Raman": _guide("Raman module", """
<p>The Raman block covers plotting (<b>Raman</b>), identification
(<b>Raman ID</b>), and fitting (<b>Peak Fitting</b>/<b>Multi-Fit</b>).</p>
<h2>Workflow</h2><ol>
<li>Import spectra in the Library (auto-detected; Custom import if not).</li>
<li>Baseline-subtract in <b>Baseline</b>: arPLS &lambda;&asymp;1e7 for broad glass
bands &mdash; at 1e5 the baseline eats the band. Verify with Preview before Apply.</li>
<li>Identify in <b>Raman ID</b>: auto-find peaks (lower the &times;&sigma; limit for
weak bands), filter the database by your laser &lambda; and sample type, then judge
the overlay yourself &mdash; the ranking assists, you decide. Accept iteratively
for mixtures; every accept is Ctrl+Z-undoable and excluded from re-searches.</li>
<li>Fit in <b>Peak Fitting</b>: pick components by clicking apexes, auto-find,
or the table. G/GL/V/EMG shapes; widths are HWHM (double for true FWHM).
Classic = one-shot Levenberg-Marquardt; Origin-like = visible step-by-step
iterations. Save models to reuse in Multi-Fit batches.</li></ol>
<h2>What replaced Spectragryph</h2>
<p>Peak list with FWHM and areas (<b>Peak table…</b>) and automated peak
labels; the multi-spectrum cursor readout; line/point styles, legend
placement, copy plot / copy data to the clipboard and paste from it (File
menu); drag-and-drop import; simple and advanced baselines (Baseline and
Calculations); normalisations to peak, area or a value; smoothing,
despiking, peak removal, cut-off (crop); SNV / MSC; arithmetic, scaled
subtraction, averaging and median of spectra; Raman-shift ⇄ wavelength
conversion and x-axis calibration against reference lines; derivatives to
the 4th order; identification by full-spectrum similarity against RRUFF and
your own reference library, with hit lists; mixture analysis (LCF) and
Gaussian deconvolution (Peak Fitting, with Voigt and EMG shapes too). Not
ported: Spectragryph's own .sgd/.spc file formats, the spectroscope view,
concentration from Lambert-Beer, film thickness, hardware acquisition, and
scripted processing sequences (use Multi-Fit, batch Baseline and the
Calculations' multi-selection instead).</p>
<h2>Pitfalls</h2><ul><li>Compare intensities only within one laser wavelength
&mdash; resonance changes relative heights between 532 and 785 nm.</li>
<li>Cosmic-ray spikes bias fits: despike first (Calculations).</li>
<li>Fit R&sup2; &gt; 0.99 with absurd widths means overlapping components:
link widths (FWHM=#) or fix &eta;.</li></ul>"""),
    "XRD": _guide("XRD module", """
<p><b>XRD ID</b> is a QualX-style search-match on card databases YOU
register — PRISM ships none. Any QualX-format .sq you have the rights to
use works; enable several at once and one search probes them all.</p>
<h2>Databases</h2><ol>
<li><b>Add database…</b> registers one .sq (<b>Add folder…</b> scans a
folder). A QualX-format file is indexed once locally — minutes for
hundreds of thousands of cards, then instant forever; a PRISM-format .sq
is used in place.</li>
<li><b>Add CIF folder…</b> turns your own structure files into a database:
every .cif under the folder (subfolders included) has its powder pattern
calculated from the structure and indexed like a card. Intensities come
from the structure factor, so systematically absent reflections are absent
rather than listed. Re-add the same folder after dropping new files in to
recalculate it.</li>
<li>The checkboxes choose which registered databases each search probes;
every hit shows its source and card code.</li>
<li>Respect each database's license: only share converted files with
people covered by the same rights you downloaded it under.</li></ol>
<h2>Search-match, step by step</h2><ol>
<li>Auto-find 2&theta; peaks (or type them). Check &lambda; (default Cu K&alpha; 1.5406 &Aring;).</li>
<li>Restrict chemistry: contains-all with elements you KNOW are present;
excludes with those that cannot be. This is the biggest ranking help.</li>
<li>Read the FoM as intensity-weighted coverage both ways: a phase whose
strong lines are absent from your pattern is penalized even if every one of
your peaks matches it.</li>
<li>Narrow with the card filters when you know the crystallography:
symmetry (crystal system) and card quality are check-droplists, the
space-group field matches by substring. Identical cards carried by
several databases are shown once.</li>
<li>Accept per phase; the accepted phase's reference bars STAY on the
plot (muted color) and the query peaks it explains turn gray while the
remainder is re-searched &mdash; repeat until everything is explained.
<b>Clear</b> starts the session over and withdraws the accepted phases
with it; Ctrl+Z in the Library brings them back.</li></ol>
<h2>Peak finding</h2>
<p><b>Auto-find peaks</b> is XRD-specific, not the generic curvature
finder: the background (amorphous halo, air scatter, fluorescence) is
estimated and removed FIRST, centers are refined below the step size, and
K&alpha;2 satellites are identified from &lambda; and dropped instead of entering
the search as extra reflections. Peak intensities are background-subtracted
too, so the figure of merit is not skewed by whatever sits under the
low-angle peaks.</p>
<p>The detection limit is read in &sigma; of the LOCAL noise. Counting noise
scales as &radic;counts, so it is several times larger on top of an amorphous
halo than out at high angle; measuring it in a sliding window is what makes
"4&sigma;" mean the same thing across the whole scan. With a single global
&sigma; the noise riding on a halo comes back as dozens of "reflections".
The default limit is 4&sigma;: measured over 110 simulated patterns built
from real structures, 3&sigma; left about 9 spurious peaks per pattern and
4&sigma; left 1.4, at a cost of 0.1&#37; of the genuine ones.</p>
<p><b>Peak search options&hellip;</b> opens the rest: how the noise is measured
and over what window, the SNIP background width (blank = estimated from
this pattern's own peak widths), the smoothing width and the minimum
separation of two peaks, the K&alpha;2 position tolerance and intensity bound,
a 2&theta; range to search in, an edge margin, and the cap on how many peaks
are reported. The dialog stays open so a setting can be changed, applied,
and judged against the plot; the setup is remembered between sessions.</p>
<p><b>show threshold</b> draws the estimated background and the detection
limit over the pattern. A peak is found when it clears the upper curve, so
"why was that shoulder not picked up?" and "why did that bump become a
peak?" are questions you answer by looking rather than by guessing.</p>
<h2>Sample displacement</h2>
<p><b>shift &plusmn;</b> lets each card slide by up to that many &deg;2&theta; before
scoring &mdash; the zero/displacement error that shifts every line the same
way. The offset each card used is in the &Delta;2&theta; column: a whole result
list wanting the same large shift is telling you about the diffractometer,
not the phases. A &#9888; on the Matched count means the card's strongest
in-range line is NOT among your peaks, the one disqualifier a healthy FoM
can hide.</p>
<h2>Accepted phases &rarr; Rietveld</h2><ol>
<li>The <b>Accepted phases</b> tab lists what has been identified on the
selected pattern, and survives a restart or a reopened project (the card
is looked up again from the source and code stored on the spectrum).</li>
<li><b>Remove selected phase</b> withdraws one identification: it leaves
the record and the overlay, and the peaks it explained go back into the
search box so the remainder can be re-examined.</li>
<li><b>Export CIFs for Rietveld…</b> collects each accepted phase's
structure into one folder with a phases.csv index, ready to load as
starting models in GSAS-II, FullProf, TOPAS or Profex. A phase matched
against your own CIF folder exports THAT file. A COD-derived card is
fetched by its COD number (cached, so a repeat export is offline) or taken
from a local mirror you point at. A PDF-2-style card is a d&ndash;I line list
with no atomic coordinates: no CIF exists for it, and the export says so
per phase instead of quietly writing fewer files than you asked for.</li></ol>
<h2>HT-XRD</h2><p>Temperature series: waterfall + Maps (a heatmap with
log/sqrt scaling reveals weak reflections; the difference map vs a reference
slice localizes transitions). Track peaks by clicking the same reflection at
two temperatures on the waterfall &mdash; absence detection reports where a peak
genuinely vanishes instead of fitting noise; vanished/appeared flags are the
transition candidates.</p>
<h2>Cross-check</h2><p>Check Raman-identified phases here overlays the XRD
reference lines of everything accepted in Raman ID &mdash; agreement between the
two techniques is strong evidence; disagreement usually means an amorphous
phase (Raman-visible, XRD-silent) or trace crystallites (the reverse).</p>"""),
    "XAS": _guide("XAS module", """
<p>The pipeline follows the tabs left to right.</p>
<ol><li><b>Import</b>: EasyXAFS ZIP (I0/It channels), CSV, Athena .prj
(no Larch needed for import).</li>
<li><b>Pre-processing</b>: smooth only for e0-finding, never the data you
quantify; Bragg-glitch energy correction; tie-point alignment for scan
drift (Mode C: click matching features BEFORE and AFTER).</li>
<li><b>&mu;(E) Builder</b>: ln(I0/It); deglitch with z&asymp;6 removes monochromator
glitches without touching EXAFS oscillations.</li>
<li><b>Normalization / EXAFS</b> (needs Larch &rarr; run via the Python install,
not the portable exe): pre/post-edge ranges are passed EXPLICITLY (a silent
Larch pitfall this app fixes); autobk rbkg &asymp; 1.0; k-weight 2 is the usual
glass choice.</li>
<li><b>Analysis</b>: average repeat scans (or sum partial acquisitions),
difference spectra, linear-combination fitting against reference standards
(coefficients = phase fractions), PCA species count.</li>
<li><b>Sample mass</b>: paste your oxide composition (mol% or wt%), pick
element+edge and pellet diameter &rarr; masses for total &mu;t = 1 / 2.5 and step
&Delta;&mu;t = 1 (Hephaestus rules of thumb). The <b>target &mu;t</b> field is your
own target absorption length (thickness in units of 1/&mu;) &mdash; change it from
the 2.5 default to get the mass for whatever thickness you actually want,
shown alongside the two fixed reference values. If &Delta;&mu;t = 1 needs several
times the &mu;t = 2.5 mass, the absorber is dilute: measure in fluorescence.</li></ol>"""),
    "Thermal": _guide("Thermal module", """
<p><b>DTA / Thermal</b> computes Tg three independent ways &mdash; double tangent,
parallel tangent, |dY| max &mdash; and says whether they agree. Trust the value
only when the spread is small; a disagreement flag usually means a poorly
chosen baseline window (adjust LOW/HIGH) or an overlapping event. Calculs
integrates peaks and finds extrema; Batch processes a whole folder.</p>
<p>Convention: onset temperatures (tangent methods) are the reportable Tg;
|dY| max is the inflection &mdash; systematically higher. State which you quote.</p>
<p><b>Cyclic programs.</b> A run that heats, holds and cools repeatedly is split
into <b>segments</b> and every analysis is scoped to the one you select.
Temperature is not a usable abscissa across a whole cyclic run &mdash; sorting by
it interleaves the ramps &mdash; so aggregate views plot but do not compute. The
<b>Program</b> tab shows the method the instrument ran (from the file's own
<i>OrgMethod</i> header) beside what the furnace did; a TA <i>.mth</i> file can be
loaded to cross-check, and the data file's header stays authoritative. Read
<i>target gap</i> and <i>control lost below</i> before quoting a cooling rate: below
that temperature the furnace was no longer holding the programmed rate. A hold
the furnace never settled into is reported as <i>absorbed</i>, which is not the
same as never executed.</p>
<p><b>Rate series.</b> Builds a Lasocka or Arrhenius series from the heating
segments of one run, from separate files at different rates (the rate is read
from each file's own program &mdash; no typing), from imported records, or by
hand. Values stay blank until <b>Compute all values</b>, and a row that cannot
be measured stays visible with its reason instead of vanishing from the fit.
Nothing refuses to fit: too few rates, a duplicated rate, a non-monotonic
trend, mixed thermal history, mass loss, a cooling rate or a negative activation
energy all appear as warnings on the banner, in the figure and in the CSV, with
the number still produced. Successive cycles of one specimen confound rate with
thermal history &mdash; and a glass that crystallizes on its first heat is not the
same material afterwards.</p>"""),
    "Processing": _guide("Processing module", """
<p><b>Baseline</b>: arPLS (automatic; &lambda; stiffness &mdash; 1e7 for broad glass
bands), ALS, polynomial/spline/rubberband through picked regions. Settings
are remembered per spectrum; batch Apply creates _bl spectra.</p>
<p><b>Calculations</b> is the everything-else toolbox: arithmetic between
spectra (selection ORDER defines A), modulated addition (blend a reference
in over a chosen range via ramp/gaussian envelopes), normalizations (max,
area, min-max, to a value at x&#8320;), log/exp/power transforms,
x-calibration (shift/scale, or a polynomial against reference lines),
x-unit conversion (nm / cm&#8315;&sup1; / Raman shift / eV), subtract a range
mean, remove a peak, SNV, MSC, median of spectra, crop, resample,
smoothing (SG keeps peak shapes; median kills spikes), despiking,
derivatives, integrals, linear-combination fitting (non-negative for
physical mixtures), statistics, and cluster analysis of spectral series
(KMeans/hierarchical over a common grid, PCA scores, per-cluster means &mdash;
for multi-point maps, find which spectra group together before averaging).</p>"""),
    "SAXS/WAXS": _guide("SAXS/WAXS module", """
<p><b>Curves</b>: the import reads named-column tables first (q / I / sigma, plus dQ, a statistical sigma, smearing
flags and text columns, all kept) and falls back to plain two/three-column files. <b>Reduction</b>: subtract the empty
capillary &mdash; auto matches the high-q tail, transmission uses measured T, physics computes both attenuations
from compositions via xraydb; inspect the corrected curve before analyzing.</p>
<p><b>Analysis</b>: Guinier is only valid for qRg &#8818; 1.3 &mdash; the report prints qRg max, take it seriously;
Rg &rarr; sphere-equivalent diameter shown. The generalized Porod slope m reads surface character (4 = smooth
interface, 3&ndash;4 rough, &lt;3 mass fractal). Pseudo-Bragg peaks give d = 2&pi;/q0 and an apparent correlation
length from the width.</p>
<p><b>WAXS</b>: pseudo-Voigt multi-peak fit with amorphous-hump detection &mdash; the crystallinity index is
crystalline area / total area, comparable only between samples measured identically.</p>
<p><b>Frames</b>: the campaign starts from the raw 2D frames of a beamstopless instrument. Scan the session folders:
every frame gets a role (sample / empty / calibrant from regexes), its session, the beam centre refined from the
recorded direct beam (header centres can be off by several pixels), and diagnostics &mdash; capillary-wall lobes on
the horizontal axis (the two bars that average into a fake low-q peak), slit streaks, saturation. Samples are paired
with the empty of the same folder, configuration and capillary. <b>Reduce all pairs</b> integrates the four diagonal
wedges only, scales the empty by the footprint transmission, gives every curve a loading class (measurable &mu;t &lt; 3,
marginal 3&ndash;6, opaque &gt; 6: shapes only, unreliable) and a usable q-range, and normalises per mm of material
when the samples table gives &mu; (or a composition + density). The reduced curves join the curve list;
<b>Export for SasView&hellip;</b> writes three-column Q / I / dI files.</p>
<p><b>Fit</b>: one engine, physically constrained. Select the curves (with their references) and give each curve its
<b>reference</b> &mdash; the same material without the minority component (a parent composition, an untreated
state); a regex rule fills many at once (<code>^(.*)_x&#92;d+$</code> &rarr; <code>&#92;1_x0</code>: every member of a series gets its x = 0 member). For particles on an
absolute scale give the <b>contrast</b> &Delta;&rho; (10<sup>10</sup> cm<sup>-2</sup>), typed or computed from a phase
formula + density against the sample composition + density (the stoichiometric ceiling of the phase is computed
too). What the fit does, for every curve: (1) <b>base</b> = grain power law (exponent 3.7&ndash;4.7: the outer
surfaces of the powder grains, never a structural level) + a flat level, or + an Ornstein-Zernike term with
&xi; &le; 5 &Aring; (sub-nanometre concentration fluctuations), chosen by BIC; (2) <b>gate</b>: the ratio of the
curve to its reference, each with its high-q term subtracted and normalised at low q, must exceed 1 beyond its
uncertainty over a factor &ge; 2 in q, with a peak above the largest ratio between the references themselves (the
scatter between references, e.g. grinding batches). A curve used as a reference gets no structure. A residual run
of the base is shown as a diagnostic only &mdash; on clean featureless curves it fires on the curvature of the
grain law itself, so it cannot decide; (3) <b>candidates</b> on both bases: spheres (Schulz, + hard-sphere S(q)),
Teubner-Strey (spacing d, correlation length &xi;), Debye (&xi;), a unified level with a tied Porod prefactor, each
also with a second small population; (4) <b>selection</b>: a structure is retained only if it converged, has no
parameter at a bound, beats its base by dBIC &gt; 10, is visible (&ge; 10 % of the model over a factor &ge; 2 in q)
and, for spheres, has its Guinier knee inside the window. Every rejection is listed with its reason in the report.
<b>Read the results</b>: sizes and spacings are the robust numbers; amplitudes become volume fractions only with a
contrast you trust; near-degenerate families (dBIC &lt; 10 between them) are stated as such, not resolved.
<b>Robustness</b> refits the retained models on perturbed data (statistical sigma only, q &le; 0.2, last point
removed, unsmeared) and flags any parameter that moves by more than max(20 %, 2&sigma;). Results save / load as
JSON; Tables writes xlsx + csv (Excel-safe ASCII) + md; Curves writes one fit file per curve.</p>
<p><b>Series figures</b>: a series table you import or edit (sample, panel, stacking order, label, colour, marker,
family, x-value, quantified) drives the grid of normalised shifted curves with labels at the curve ends, the
fit-panel figures (data, model, labelled components, normalised residuals, I or I q<sup>4</sup>), the trends against
your x-value (open = not quantified, faded = not robust) and the 2D frame figure; export pdf/svg/png at your dpi, and
the tables as xlsx / csv / md. <b>Headless</b>: <code>scripts/saxs_pipeline.py init / run / fit / quick</code> with a
JSON project file (curves reduced here or elsewhere, references, contrast, figures, tables) regenerates a whole
campaign; see docs/SAXS_FITTING.md.</p>"""),
    "Figures": _guide("Figures module", """
<p>Build the figure ONCE, export at the journal exact size.
<b>XY builder</b>: add spectra as layers; each layer has type, color,
offset, panel; presets set publication-grade fonts/ticks. Export at cm
size + dpi (600 for line art). <b>Point fitting</b>: model library with
&plusmn;1&sigma; and R&sup2; &mdash; fit Tg vs composition, Arrhenius, sigmoids.
<b>Ternary</b>: composition tables from CSV, color-mapped property.
<b>Raman &harr; XRD</b>: the two-panel identification figure with accepted
phases annotated.</p>"""),
    "Fitting": _guide("Fitting module", """
<p><b>Peak Fitting</b> for one spectrum, <b>Multi-Fit</b> for batches &mdash; the
same JSON models serve both, so refine interactively then apply to a series.
Widths are HWHM everywhere (historic convention; double for true FWHM). Use
confidence intervals (F-test) when a parameter matters for the conclusion,
not just the covariance &plusmn;1&sigma;. In Origin-like mode each iteration is one
damped LM update: watch parameters walk, stop when &chi;&sup2; stalls.</p>"""),
}
