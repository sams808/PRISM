"""
qt_help.py — in-app Help: the F1 quick-start guide (per-workspace, written
for a group member opening the app for the first time) and the About
dialog (versions + the citations the bundled databases ask for).
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtWidgets import QDialog, QTextBrowser, QVBoxLayout, QWidget

HELP_HTML = """
<h1>Dataapp — quick-start guide</h1>

<p><b>The flow:</b> bring data in through the <b>Library</b>, then switch to the
workspace for your technique in the left rail. Everything you derive
(baseline-subtracted, combined, fitted) lands back in the Library as a new
spectrum. <b>File &rsaquo; Save project</b> keeps the whole session in one
<code>.dataapp</code> file.</p>

<h2>Library</h2>
<p><b>Import files…</b> auto-detects the format (plain XY, TA SDT thermal exports,
SAXS EDF, Rigaku <code>.rasx</code>, JCAMP-DX). If it guesses wrong, use
<b>Custom import…</b> (Ctrl+I) to pick the parser and the X/Y columns yourself, with a
preview. Right-click spectra to rename, duplicate, reorder, export as text
(Ctrl+E), combine (sum / average / weighted subtraction), or delete —
deletion is undoable (Ctrl+Z).</p>

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
spectra. Toggle <b>Annotate on click</b> to pin peak-position labels.</p>

<h2>Peak Fitting</h2>
<p>Set up components in <b>Fit param.</b> (or let <b>Auto-find peaks</b> seed them),
then <b>Fit&nbsp;!</b>. Shapes: G (Gaussian), GL (pseudo-Voigt), V (true Voigt),
EMG (asymmetric, signed skew). The <code>FWHM=#</code> column links a component's
width to another's. Reports include R², ±1σ errors, and peak centroids;
<b>Conf. intervals</b> runs rigorous F-test profiling. Save a configuration as a
<i>model</i> to reuse it — models are also the recipes <b>Multi-Fit</b> applies
across many spectra at once.</p>
<p><b>Width convention:</b> the "FWHM" fields have always been <i>half</i>-widths
(HWHM) — a historical convention kept so old saved models stay valid. Double
them when quoting true FWHM.</p>

<h2>XAS</h2>
<p>Import EasyXAFS ZIPs, CSVs, or Athena <code>.prj</code> from the workspace's own
buttons. Tabs follow the workflow: Pre-processing (smoothing, Bragg
angle→energy correction, click-based tie-point alignment) → μ(E) Builder
(with deglitching) → Normalization / EXAFS (Larch) → Analysis (merge, difference,
linear-combination fitting, PCA species count) → Export (Athena formats).</p>

<h2>DTA / Thermal</h2>
<p>Pick X/Y/dY channels, compute Tg three ways (double tangent, parallel
tangent, |dY| max) — the result panel says whether the methods agree.
"Calculs" integrates or finds extrema over a range; Batch processes a folder.</p>

<h2>Mineral ID (RRUFF)</h2>
<p>Auto-find (or type) your spectrum's peak positions, then <b>Find matches</b>:
candidates are ranked by peak overlap, each showing its <b>laser wavelength</b> —
relative intensities vary between wavelengths, so judge the overlay yourself.
Nothing is labeled until you click <b>Accept</b>. One more click overlays the
matched mineral's predicted XRD pattern (AMCSD structure) in the Raman
workspace. Requires the one-time local database build (see the README).</p>

<h2>HT-XRD</h2>
<p>Import a whole folder of patterns; temperatures come from <code>.rasx</code>
metadata or a filename template like <code>scan_???.xy</code>. The waterfall is
colored by temperature. Enter a 2θ window around a peak and <b>Track</b>: center /
width / area vs temperature, with automatic flagging of fit-quality anomalies —
candidate phase-transition windows.</p>

<h2>Clustering</h2>
<p>Select a series (e.g. a multi-point map), choose KMeans or hierarchical and
a cluster count: PCA scatter colored by cluster, per-cluster mean spectra, and
an assignment table.</p>

<h2>Shortcuts</h2>
<p><code>Ctrl+O</code> import · <code>Ctrl+I</code> custom import ·
<code>Ctrl+E</code> export · <code>Ctrl+S</code> save project ·
<code>Ctrl+Shift+O</code> open project · <code>Ctrl+Z</code> undo delete ·
<code>Ctrl+Q</code> quit · <code>F1</code> this guide.
View menu: dark mode, Python console (the live app objects are in scope).</p>
"""

ABOUT_HTML = """
<h2>Dataapp</h2>
<p>Import, process, and analyze scientific spectra: Raman, XAS/XANES/EXAFS,
DTA/DSC/TGA, XRD (including high-temperature series), SAXS.</p>
<p>Source: <code>github.com/sams808/Dataapp</code></p>
<h3>Please cite when the bundled databases contribute to your work</h3>
<p><b>RRUFF</b> (Raman reference spectra): Lafuente B, Downs R T, Yang H,
Stone N (2015) "The power of databases: the RRUFF project." In: Highlights
in Mineralogical Crystallography, pp 1–30. Also acknowledge each matched
sample's owner/source shown in the Mineral ID workspace.</p>
<p><b>AMCSD</b> (crystal structures used for XRD overlays): Downs R T,
Hall-Wallace M (2003) "The American Mineralogist Crystal Structure
Database." American Mineralogist 88, 247–250.</p>
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
