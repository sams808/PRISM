# PRISM 2.9.0 — usability review of the Raman station (100 checks)

Scope: the configuration deployed on the Raman spectrometer's computer
(Raman station preset: Raman, Fitting, Processing, Figures modules, plus
Home and Library). Method: every page, ribbon tab, panel and dialog was
rendered off-screen at 1366 × 768 and at 1920 × 1080, in the dark and the
light theme, and every visible widget was linted (elided text, clipped
widgets, overlapping siblings, undersized controls, ribbon overflow,
minimum window size, buttons without a tooltip). Renders and the lint
report live outside the repository (`Desktop/PRISM_review/ux_*`).

Result column: **ok** (passed as found), **fixed** (changed in 2.9.0 after
this review), **noted** (acceptable, left as is).

## A. First launch and setup

| # | Check | Result |
|---|---|---|
| 1 | Splash shows the name and the group, not a feature list | fixed |
| 2 | First page is Home; with nothing switched on it says Welcome and offers Choose modules… | ok |
| 3 | Modules are chosen in Settings › Modules… (presets Raman station / All / None, one switch each) | ok |
| 4 | The choice is remembered per user; the installer self-test no longer overwrites it | fixed |
| 5 | Home shows only the switched-on modules, as cards with buttons to their pages | ok |
| 6 | Switched-off modules disappear from the rail; the Library is always there | ok |
| 7 | Window size and last page come back on restart | ok |
| 8 | A newer setup upgrades an installed PRISM in place (verified 2.8.0 → 2.9.0) | ok |
| 9 | The portable zip runs from any folder (self-test of the extracted copy) | ok |
| 10 | Home › Reference databases… shows what is installed and exports / imports packs | ok |

## B. Navigation and global behaviour

| # | Check | Result |
|---|---|---|
| 11 | Rail: Home, Library, then the pages of the active modules, colour-coded by module | ok |
| 12 | Menus: File (import, custom import, paste, export, open / save project, undo, exit), View, Settings, Help | ok |
| 13 | Shortcuts: Ctrl+O, Ctrl+I, Ctrl+Shift+V, Ctrl+E, Ctrl+Shift+O, Ctrl+S, Ctrl+Z, Ctrl+Q, F1 | ok |
| 14 | Files dropped anywhere on the window are imported; the status bar reports the count | ok |
| 15 | Status-bar messages are transient (6–8 s), then Ready | fixed |
| 16 | Every page fits 1366 × 768 (largest minimum: Peak Fitting 979 × 376) | fixed (Figures needed 791 px) |
| 17 | Dark and light themes: muted text on cards keeps a contrast ratio above 4.5 | ok |
| 18 | Plots stay white in both themes, so the screen matches the export | ok |
| 19 | F1 quick-start guide covers every page (enforced by a test); module guides; About cites the databases | ok |
| 20 | Rail narrowed to 164 px, giving the ribbon room on small screens | fixed |

## C. Library

| # | Check | Result |
|---|---|---|
| 21 | Import files…, Custom import…, Combine / scale…, Undo carry tooltips | fixed |
| 22 | Imported spectra are listed with their kind (omnic_spa, omnic_mapx, generic_xy…) | ok |
| 23 | A map imports as region means plus pixel spectra (cap 2000) | ok |
| 24 | Selecting a row previews it | ok |
| 25 | Right-click: rename, duplicate, reorder, export, combine, delete | ok |
| 26 | Undo covers deletes, renames, duplicates, applied operations, accepted identifications | ok |
| 27 | An unreadable file gives a message, not a crash | ok |
| 28 | Decimal comma, Latin-1 and UTF-16 text files are read | ok |
| 29 | Thermo OMNIC .spa and .mapx and OMNIC's .txt / .csv exports are read | ok |
| 30 | Projects (.prism) keep spectra, results and settings | ok |

## D. Raman workbench — structure

| # | Check | Result |
|---|---|---|
| 31 | Tabs in working order: Spectra → Display → Process → Transform → Analyze → Identify → Sequence → Export | ok |
| 32 | Every tab is at most two rows; the ribbon is sized for the tallest tab, nothing clipped | fixed |
| 33 | No tab scrolls sideways at 1366 (Display tab trimmed) | fixed |
| 34 | Group labels hang from the top of the ribbon | fixed |
| 35 | One detail panel at a time under the ribbon: operation, Plot settings or Sequence | ok |
| 36 | Spectra list: tick = shown, highlight = target; "N shown" counter | ok |
| 37 | Name filter above the list for long lists | fixed |
| 38 | Show all / Hide all / Only selected explain themselves (tooltips) | fixed |
| 39 | Results / Log pane under the plot: small while empty, grows when a table arrives | fixed |
| 40 | Every ribbon button carries a tooltip | fixed |

## E. Raman workbench — operations

| # | Check | Result |
|---|---|---|
| 41 | An operation panel shows a title, a one-line explanation and its parameters in three columns | ok |
| 42 | Pick… sessions: ranges and points append; reversed ranges swapped; double click harmless; right-click, Esc or Done ends | fixed |
| 43 | Picked ranges shaded and points marked on the plot while the panel is open | fixed |
| 44 | Preview draws dashed; Apply writes new spectra (undoable); the buttons say so | fixed |
| 45 | Treat all shown spectra / Hide originals / suffix on every operation | ok |
| 46 | A wrong parameter gives a message naming the parameter | ok |
| 47 | Advanced baseline has an anchors method (points clicked on the plot) | fixed |
| 48 | Peaks & FWHM accepts hand-picked peaks, added to or replacing the detected ones; Source column | fixed |
| 49 | Found peaks marked on the plot with a toggle; they replace the automatic labels on that spectrum | fixed |
| 50 | The Peak table… window sits in Analyze › Peaks, next to Peaks & FWHM | fixed |

## F. Raman workbench — display and export

| # | Check | Result |
|---|---|---|
| 51 | Overlay / Stacked (shift) / Panels; difference of two spectra | ok |
| 52 | Colour schemes and colormaps, line width, lines / points, legend placement | ok |
| 53 | Normalize, Zero line, Log y, Smooth, Grid as one-click toggles | ok |
| 54 | Show x as (unit conversion) with the data unit and the laser in Plot settings; tooltip says where | ok |
| 55 | Plot settings panel: titles, limits, font, data unit, laser, peak-label options | ok |
| 56 | Hundreds of spectra: legend off above 25, labels off above 12, thin lines above 100, wait cursor | fixed |
| 57 | Cursor readout and click annotations | ok |
| 58 | Export: text files, combined CSV, JCAMP-DX, copy data; plot PNG / SVG / PDF / EPS / TIFF at a dpi, copy, print | ok |
| 59 | Results table Copy / Save CSV with tooltips | fixed |
| 60 | Sequence panel explains itself while empty; steps listed; double-click reopens a step | fixed |

## G. Raman ID

| # | Check | Result |
|---|---|---|
| 61 | Query combo and Auto-find; the peak list shows from its first value | fixed |
| 62 | Match by peak positions or spectrum similarity; Compare / x range disabled when not relevant | ok |
| 63 | Sources: RRUFF and My library | ok |
| 64 | Database filters in a 2 × 2 grid that fits the panel | fixed |
| 65 | Results: overlay toggle, query / candidate peak toggles, Card · Update plot · Clear on one row, Accept, Overlay XRD | fixed |
| 66 | Hit table: Mineral, Formula, RRUFF ID, λ, Matched, Match %, Quality, Score; no sideways scroll at 1366 | fixed |
| 67 | A summary line above the table gives the verdict (convincing / possible / weak) | fixed |
| 68 | Double-click, Card button or right-click opens the card: composition, locality, source, RRUFF page | fixed |
| 69 | Scoring is two-sided and intensity-weighted on peaks re-detected from the data files; damaged files skipped | fixed |
| 70 | Default tolerance 3 cm⁻¹; laser filter ±2 nm | fixed |

## H. Peak Fitting and Multi-Fit

| # | Check | Result |
|---|---|---|
| 71 | Spectrum selector, Norm., Plot X fields with "x min" / "x max" placeholders | fixed |
| 72 | Fit param., Pick peaks on plot, Fit !, Reset explain themselves (tooltips) | fixed |
| 73 | Classic vs Origin-like; 1× / 2× / 5× / 10× iterations; step size; Fit until converged | ok |
| 74 | Chi², R², Conf. intervals, Quick report, Report as…, Export components…, Export plot with tooltips | fixed |
| 75 | X and Y titles editable above the plot | ok |
| 76 | Log pane under the plot | ok |
| 77 | Multi-Fit: spectra, recipe New / Edit / Refresh, Normalize, Fit X range, Run batch fit, results, CSV export, all with tooltips | fixed |
| 78 | Multi-Fit previews the selected result row | ok |
| 79 | Peak Fitting reachable from the workbench (Analyze › Fit peaks) | ok |
| 80 | Peak Fitting fits 1366 px wide (minimum 979) | ok |

## I. Baseline, Calculations, Figures

| # | Check | Result |
|---|---|---|
| 81 | Baseline: method, λ / ratio, regions, Add region by dragging, Preview / Apply with tooltips | fixed |
| 82 | The glass tip (λ ≈ 1e7) is in the guide | ok |
| 83 | Calculations: selection order explained, operation, Pick on plot, Preview / Apply with tooltips | fixed |
| 84 | Figures: the XY builder's panel scrolls, so the window fits 768 px | fixed |
| 85 | Figures: layers, CIF sticks, panels, presets, titles, log axes, legend, ticks, Render, Export with tooltips | fixed |
| 86 | Figures tabs: XY builder, 2D / Series, Table plots, Point fitting, Ternary, Raman ↔ XRD | ok |
| 87 | Baseline and Calculations label their axes x / y (generic pages) | noted |
| 88 | Mixture analysis and Baseline reachable from the workbench (Analyze › Mixture analysis; Processing module) | ok |
| 89 | Calculations › Pick on plot explains itself | ok |
| 90 | No page needs more than 1366 × 768 | ok |

## J. Dialogs and feedback

| # | Check | Result |
|---|---|---|
| 91 | Settings dialog: Modules tab (presets, cards with switches, "n of 9 modules on") and Reference databases tab; non-modal | ok |
| 92 | Help dialog readable in both themes, one section per page | ok |
| 93 | Properties dialog: colour, line style, width, offset, scale, legend text, laser, temperature, concentration, notes | ok |
| 94 | Peak table window: copy as text, save CSV | ok |
| 95 | Add to library dialog: name, laser, source, notes | ok |
| 96 | Library manager: rename, remove, export, import | ok |
| 97 | RRUFF card: every header field, Open the RRUFF page, Open the data file | ok |
| 98 | Message boxes: titled with the operation, one plain sentence | ok |
| 99 | Long tasks (similarity search, peak search, downloads) run in the background with progress in the status line | fixed |
| 100 | All of the above re-run by the audit script after the fixes: 0 elided labels, 0 overlaps, 0 ribbon overflow, every page fits | ok |

## Known constraints

- At 768 px of height the plot is short while an operation panel or the
  Plot settings are open; close the panel (or drag the splitter) to see
  more. The ribbon's two-row design costs 136 px.
- The Raman ID left panel scrolls at 768 px: its last section (databases on
  this computer, download buttons) is below the fold.
- Dialog buttons such as Close, Cancel, Add and Apply have no tooltip on
  purpose.
