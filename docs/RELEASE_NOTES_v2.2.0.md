**Dataapp is now PRISM** — *Platform for Research In Spectroscopy & Materials* — with a new logo, a startup splash, and the biggest feature release yet: 13 workspaces, organized into activatable modules.

> **Upgrading from 2.x**: the portable build is renamed — extract `PRISM-portable.zip` and double-click `PRISM\PRISM.exe`. Your projects and local databases are untouched. The GitHub repository moved to `github.com/sams808/PRISM` (old links redirect).

## New identity
- **Modules toolbar**: seven color-coded checkboxes (Raman, Fitting, XRD, XAS, Thermal, Processing, Figures) — uncheck what you don't use and the app shows only your workspaces. Per-user, remembered across sessions. The nav rail is color-coded to match.
- Splash screen, window/taskbar icon, and a Credits page (NOME group · U.S. Department of Energy · Prof. John S. McCloy).

## New workspaces
- **Calculations** — 30 operations: spectrum arithmetic (add/subtract/multiply/divide/average/weighted sum), modulated addition A + w(x)·B, normalizations, log/exp/√/power transforms, x-axis calibration, crop/resample, smoothing, cosmic-ray despiking, derivatives, integrals, linear-combination fitting with R² and percentages, statistics.
- **XRD ID** — QualX-style phase identification, rebuilt inside PRISM: search-match your 2θ peaks against a locally merged **692,665-card database** (COD inorganic + full COD + ICDD PDF-2 — every card keeps its source and code), with chemistry/source filters, figure-of-merit ranking, stick-pattern previews, and iterative Accept for mixtures. Searches take ~1 s. One-time local database build required (F1 help).
- **Figures** — publication figure building: layered XY plots (7 plot types, multi-panel, style presets, export at exact cm size/dpi), Origin's classic point-fit models with ±1σ and R², native ternary composition plots from CSV, and the **Raman ↔ XRD identification figure** (your Raman spectrum with accepted Mineral-ID phases over the XRD pattern with accepted phase reference lines).

## From the second feedback wave
- Peak Fitting: "Pick peaks on plot" now auto-disables the zoom tool (the reported clicks-do-nothing bug) with a crosshair cursor.
- XAS: Sum alongside Average in the Analysis tab; `.prj` import works without Larch (and export works on modern Larch).
- Mineral ID: filters populate on page entry; Accept is iterative for mixtures (matched peaks removed, remainder re-searched, accepted references excluded); shift-click overlays several candidates.
- HT-XRD: **pick-to-track** — click the same peak at two temperatures on the waterfall and track along that guide (works for appearing/dying peaks from a single intermediate click); **Auto-track all peaks**; robust noise estimation.
- Raman → XRD cross-check: one button overlays the XRD reference lines of every Raman-identified phase.

397 tests, all green.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
