PRISM 2.3.0 — the scattering release: a full **SAXS/WAXS module** and the Hephaestus-style **sample-mass calculator** for XAS.

## SAXS/WAXS (new module, 14th workspace)
The author's POMME suite, ported into PRISM as `saxs_core` + a four-tab workspace:
- **Curves**: Xenocs-style 1D ASCII import, log-log overlay, send-to-Library.
- **Reduction**: empty/background subtraction with manual, auto, transmission, or physics-based (xraydb) scaling.
- **Analysis**: Guinier (Rg + sphere-equivalent diameter), generalized Porod (slope + background + partial invariant), pseudo-Bragg peak (d-spacing, apparent correlation length) — all with automatic region detection.
- **WAXS**: pseudo-Voigt multi-peak auto-find + fit, d-spacings, crystallinity index.
sasmodels/bumps model fitting stays in the standalone pomme repo (optional heavy dependency).

## XAS: Sample mass tab (Hephaestus 'formula' tool)
Enter a single formula **or an oxide composition table in mol% or wt%** (the lab's spreadsheet workflow), pick the element/edge and pellet diameter: PRISM reports μ/ρ above the edge, the edge step Δ(μ/ρ), the absorber mass fraction, and the pellet masses for total μt = 1 / 2.5 and edge step Δμt = 1 — with a fluorescence-mode warning for dilute samples.

408 tests, all green.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
