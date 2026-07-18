PRISM 2.0.0 (released under the app's former name) — the completed PySide6/Qt rebuild of the multi-technique spectra suite.

## Highlights
- **10 workspaces**: Library, Raman (Simple Plot + CIF overlays), XAS (full pipeline incl. Larch normalization/EXAFS, LCF, PCA), DTA/Thermal (3 Tg methods + agreement flag), Peak Fitting (G/GL/true-Voigt/EMG, parameter linking, confidence intervals), Multi-Fit (batch recipes), Mineral ID (RRUFF match-assist, 37,911 spectra), HT-XRD (temperature series + peak tracking + transition flagging), Clustering, Baseline (arPLS/ALS/poly/spline/rubberband).
- **Project files** (format v3): the whole session — spectra, fit models, XAS store with history, HT-XRD series, CIF overlays, baseline settings — survives closing the app. A demo project ships in `EXAMPLES/`.
- **Onboarding**: press F1 in the app for the per-workspace quick-start guide; `docs/USER_GUIDE.md` is the same content.
- **300 tests**, all green; validated end-to-end on real datasets (P5Bi8-12 HT-XRD series, ISG pressure series, PBi0-1 map series).

## Getting started
- **Portable (no Python needed)**: download the portable zip below, extract, double-click the exe. Note: Larch-dependent XAS steps (normalization/EXAFS) are not included in the portable build — use the Python route for those.
- **From a Python 3.11 install**: clone, `pip install -r requirements.txt`, double-click the launcher .bat (or `python qt_main.py`).
- The RRUFF/AMCSD reference databases are built once locally (~1.3 GB under `~/.raman_cache/`); see the README for the two one-line commands.

## Data citations
Mineral identification uses the RRUFF database — Lafuente, Downs, Yang & Stone (2015), "The power of databases: the RRUFF project" — and AMCSD structures — Downs & Hall-Wallace (2003), *American Mineralogist* 88, 247–250. Please cite when these contribute to published work.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
