PRISM 2.1.0 (released under the app's former name) — the first hands-on-feedback release, four days after 2.0.0. Everything here came from real use of the app.

## Fixed
- **Athena `.prj` import works everywhere now, including the portable exe.** A new pure-Python reader handles both `.prj` on-disk formats; Larch is only needed for normalization/EXAFS (and the error message now says so). Investigating this bug also revealed that the old import path returned *zero* spectra on modern Larch versions, and that `.prj` *export* had been silently failing since Larch dropped `write_athena` — both fixed.

## Peak Fitting
- **Origin-like mode rebuilt as a faithful stepwise Levenberg-Marquardt**: "1 iteration" performs exactly one damped parameter update (watch the curve move), "Fit until converged" repeats to the χ² tolerance (default 1e-9, Origin's own), redrawing every step.
- **Pick peaks on plot**: toggle, then click each peak apex to add a component — amplitude is read from the data near the click.
- **Auto-find peaks** gained a *detection limit (×σ)* control: lower it to catch weak peaks, raise it to keep only the strongest.

## Mineral ID (RRUFF)
- **Database filters** applied before ranking: laser wavelength (±2 nm, so 532 also covers 532.6), oriented/unoriented, high-res vs broad-scan, quality category.

## HT-XRD
- **Tracking reworked for real series**: several windows at once (`28.5-29.5 @ 28.98; 31-32`), where `@` anchors *which* peak to track in a crowded window; sequential seeding + valley-bounded fit regions keep the fit on a drifting peak next to stronger neighbors; peaks weaker than the *Absence σ* noise threshold are reported **absent** instead of fit garbage, and vanished/appeared peaks are flagged as transition signatures. Validated on a real 57-scan series (the tracked reflection is caught vanishing at 697.5 °C).
- **New Maps tab**: 2D heatmap (linear/log/sqrt/power color scales, true non-uniform temperature axis), signed/absolute difference map and difference waterfall vs a reference ("first", an index, or a temperature), 3D surface, optional time axis from the heating rate, and `{slice:2θ; ...}` peak guide lines on heatmap and waterfall.

## Library
- **Undo (Ctrl+Z) extended beyond deletion**: renames, duplicates, combine results, applied baselines, and accepted mineral identifications are all one undo away.

334 tests, all green. Portable build: download the portable zip, extract, double-click the exe (Larch-based XAS normalization/EXAFS still needs the Python + launcher-.bat route).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
