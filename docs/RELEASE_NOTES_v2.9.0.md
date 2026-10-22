PRISM 2.9.0 — the Raman workbench, module settings off the Home page, and the fixes found while deploying 2.8.0 on the spectrometer's computer.

## Home page and modules

- The Home page shows only the modules that are switched on, as cards with buttons to their workspaces. When none is on, it offers **Choose modules…**. Module activation lives in **Settings › Modules…** (presets *Raman station*, *All modules*, *None*; one switch per module) and in **Settings › Module switches**; the choice is remembered per user.
- The installer self-test (`PRISM.exe --distcheck`) now runs on a throw-away settings file (`PRISM_SETTINGS_DIR`). In 2.8.0 it switched every module on in the user's real settings, so a freshly checked installation woke up with all modules enabled and forgot the modules chosen before.
- The splash screen no longer lists what the application does.

## Raman workbench (replaces the Raman workspace)

- A ribbon of eight tabs: **Spectra**, **Display**, **Process**, **Transform**, **Analyze**, **Identify**, **Sequence**, **Export**. The spectra list on the left (tick = shown, highlight = target of an operation, colour swatches, right-click for properties, a filter box for long lists), the plot in the centre, a Results / Log pane underneath.
- One detail panel at a time under the ribbon: the parameter form of the chosen operation, the **Plot settings** (titles, limits, font, unit of the data and laser, peak-label options) or the **Sequence** steps (double-click a step to reopen it).
- **Pick…** on any parameter that is an x value: one click for a value, two clicks per range, as many ranges or points as needed in one session (clicks append; right-click, Esc or Done ends it). Ranges given in the wrong order are swapped, a double click is not two points, and the chosen ranges and points are shaded and marked on the plot while the panel is open.
- **Advanced baseline** gains the *anchors* method: a baseline through the spectrum's own values at points clicked on the plot, joined by straight segments or a smooth spline.
- **Peaks & FWHM** accepts peaks pointed at by hand, added to the detected ones or replacing them; picks snap to the apex within a chosen window; the table says which peaks were detected and which were picked.
- **Found peaks** (Display tab) marks the peaks of the last Peaks & FWHM run on the plot, detected and hand-picked, with their positions; one toggle hides them.
- **Remove all** empties the library (undoable); the spectra list has a name filter.
- Large sets (a map's 2000 pixel spectra) no longer freeze the application: above 25 spectra the legend is left out, above 12 the peak labels, above 100 the lines are thin and marker-free, and the results table fills without repainting per cell.
- The **Peak Fitting** Origin-like panel offers 1×, 2×, 5×, 10× iterations, *Fit until converged* and a *Step size* (100 / 50 / 25 / 10 % of each Levenberg–Marquardt update).

## Raman ID

- Every hit shows the mineral's **ideal formula** (SiO₂, Ca₅(PO₄)₃F …); the plot legend carries it too.
- **Card / RRUFF page…** (or a double-click on a hit, or the right-click menu) opens the hit's card: composition, measured chemistry, locality, source, laser, orientation, data file, and a button to its page on rruff.info.
- **Show the query peaks** and **Show the candidates' peaks** switch the peak marks on the plot.
- The peak-position search is rescored. It used to count how many query peaks fell within the tolerance of any reference peak, so a reference with many peaks of its own floated up on chance coincidences (62 % for an unrelated mineral). The score is now two-sided and intensity-weighted: the share of the query's peak intensity the reference explains, times the share of the reference's own peaks (re-detected from its data file) present in the query, as a geometric mean. The status line says whether the best score is convincing (≥ 0.6), possible (≥ 0.4) or weak.
- A damaged RRUFF data file (a line with only one number) made the similarity search fail with "operands could not be broadcast"; the parser now keeps x and y aligned and the search skips any reference it cannot score.

## Installer

- A newer `PRISM-<version>-setup.exe` upgrades an installed PRISM in place: same application id, same folder and privileges as before, the previous `_internal` folder wiped before the new files are laid down, a running PRISM closed through the Restart Manager. No uninstall is needed first.

## Documentation

- `docs/RAMAN_PARITY.md`: every Spectragryph function, with its status in PRISM and where it sits.
- docs/UX_REVIEW_2.9.0.md: the 100-point usability review of the Raman station (every page rendered and linted at 1366 x 768 and 1920 x 1080, both themes) and what it changed: every page fits a laptop screen, tooltips on every action button, transient status messages, a hits summary line in Raman ID, the Peak table window in Analyze.
