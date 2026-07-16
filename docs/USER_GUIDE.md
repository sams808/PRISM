<!-- Generated from qt_help.py (the in-app F1 guide). Edit there, then regenerate. -->

# Dataapp — quick-start guide

**The flow:** bring data in through the **Library**, then switch to the
workspace for your technique in the left rail. Everything you derive
(baseline-subtracted, combined, fitted) lands back in the Library as a new
spectrum. **File > Save project** keeps the whole session in one
`.dataapp` file.

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

## Peak Fitting
Set up components three ways: in the **Fit param.** table, with
**Auto-find peaks** (its *detection limit* spinner sets how strong a
candidate must be, in units of the noise — lower it to catch weak peaks), or
with **Pick peaks on plot** — toggle it and click each peak apex.
Then **Fit !**. Shapes: G (Gaussian), GL (pseudo-Voigt), V (true Voigt),
EMG (asymmetric, signed skew). The `FWHM=#` column links a component's
width to another's. Reports include R², ±1σ errors, and peak centroids;
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
(with deglitching) → Normalization / EXAFS (Larch) → Analysis (merge, difference,
linear-combination fitting, PCA species count) → Export (Athena formats).

## DTA / Thermal
Pick X/Y/dY channels, compute Tg three ways (double tangent, parallel
tangent, |dY| max) — the result panel says whether the methods agree.
"Calculs" integrates or finds extrema over a range; Batch processes a folder.

## Mineral ID (RRUFF)
Auto-find (or type) your spectrum's peak positions, then **Find matches**:
candidates are ranked by peak overlap, each showing its **laser wavelength** —
relative intensities vary between wavelengths, so judge the overlay yourself.
The **Database filters** restrict candidates before ranking: laser λ
(matched ±2 nm, so 532 also covers 532.6), oriented/unoriented, high-res vs
broad-scan, and quality category.
Nothing is labeled until you click **Accept**. One more click overlays the
matched mineral's predicted XRD pattern (AMCSD structure) in the Raman
workspace. Requires the one-time local database build (see the README).

## HT-XRD
Import a whole folder of patterns; temperatures come from `.rasx`
metadata or a filename template like `scan_???.xy`. The waterfall is
colored by temperature. The **Maps** tab adds a 2D heatmap
(linear/log/sqrt/power color scales), a difference map or difference waterfall
vs a reference pattern ("first", an index, or a temperature), a 3D surface,
an optional time axis from the heating rate, and dashed peak-guide lines
(`{slice:2θ; slice:2θ}` anchors, interpolated).

**Tracking** takes several windows at once — `28.5-29.5 @ 28.98; 31-32` —
where the `@` anchor picks WHICH peak to track when a window holds
more than one. Each pattern's fit is seeded from the previous one (so it
follows a drifting peak instead of jumping to a stronger neighbor), and a
peak weaker than the *Absence σ* noise threshold is reported as absent
instead of a garbage fit — vanished/appeared peaks are flagged as transition
signatures alongside fit-quality anomalies.

## Clustering
Select a series (e.g. a multi-point map), choose KMeans or hierarchical and
a cluster count: PCA scatter colored by cluster, per-cluster mean spectra, and
an assignment table.

## Shortcuts
`Ctrl+O` import · `Ctrl+I` custom import ·
`Ctrl+E` export · `Ctrl+S` save project ·
`Ctrl+Shift+O` open project · `Ctrl+Z` undo ·
`Ctrl+Q` quit · `F1` this guide.
View menu: dark mode, Python console (the live app objects are in scope).

---

## Dataapp
Import, process, and analyze scientific spectra: Raman, XAS/XANES/EXAFS,
DTA/DSC/TGA, XRD (including high-temperature series), SAXS.

Source: `github.com/sams808/Dataapp`

### Please cite when the bundled databases contribute to your work
**RRUFF** (Raman reference spectra): Lafuente B, Downs R T, Yang H,
Stone N (2015) "The power of databases: the RRUFF project." In: Highlights
in Mineralogical Crystallography, pp 1–30. Also acknowledge each matched
sample's owner/source shown in the Mineral ID workspace.

**AMCSD** (crystal structures used for XRD overlays): Downs R T,
Hall-Wallace M (2003) "The American Mineralogist Crystal Structure
Database." American Mineralogist 88, 247–250.

### Built on
numpy · scipy · pandas · matplotlib · lmfit · rampy · scikit-learn ·
PySide6 · xraylarch (optional, XAS)
