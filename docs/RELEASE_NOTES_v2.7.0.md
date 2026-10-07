PRISM 2.7.0 — SAXS 2.0: from raw 2D frames to physical fits, publication figures and tables, for any sample series.

## SAXS/WAXS — SAXS 2.0 (new)

Three new tabs in the SAXS/WAXS workspace, a single physical fitting engine and a headless pipeline. Everything is sample-agnostic: references, contrasts, grouping, ordering, colours, markers and x-values come from you; nothing in the code knows a sample name.

- **Frames tab**: inventory of raw 2D frames (EDF) from a beamstopless instrument.
  - Each frame gets a role from regexes (sample / empty / calibrant), a session and a configuration.
  - The beam centre is refined from the recorded direct beam. Header centres were off by 0.5–0.8 mm in three of five sessions of a real campaign.
  - Diagnostics: direct-beam level, saturation, and the lobe / streak diagnostics that expose capillary-wall light on the horizontal detector axis (the artefact that averages into a fake low-q maximum).
  - Samples are paired with the empty of the same folder, configuration and capillary (`same_folder`, `nearest_time` or explicit rules).
  - **Reduce all pairs** integrates the four diagonal wedges only and masks the bar zone. It takes the transmission from the beam footprint with a central-disc uniformity check, classifies every loading (measurable / marginal / opaque / unreliable), flags the usable q-range, and normalises per mm of material from a samples table (μ per sample, or composition + density through xraydb at the frame's energy).
  - One `.dat` per sample, with the full provenance in its header; the curves join the workspace.
- **Fit tab**: physical fitting of absolute 1D curves with one engine (`saxs.core.fit_workflow`), described in `docs/SAXS_FITTING.md`.
  - **Base**: grain power law (exponent 3.7–4.7) plus a flat level, or an Ornstein–Zernike term with ξ ≤ 5 Å, chosen by BIC.
  - **Model-free gate**: the ratio of each curve to its reference curve must exceed 1 beyond its uncertainty over a factor ≥ 2 in q. Its peak must also exceed the largest ratio between the references themselves.
  - **Candidates**, each on both bases and with an optional second small population:
    - Schulz-polydisperse spheres, absolute with a particle contrast, also with a Percus–Yevick S(q);
    - Teubner–Strey;
    - Debye;
    - a unified level with a tied Porod prefactor.
  - **Retention rules**: converged; nothing at a bound; ΔBIC > 10 over the base; visible (≥ 10 % of the model over a factor ≥ 2 in q); spheres with their Guinier knee inside the window. Every rejection is listed with its reason.
  - **Derived quantities**: sizes in nm, invariants, volume fractions for the contrast, the minimum contrast able to produce the invariant, and the stoichiometric ceiling of the phase.
  - **Robustness pass**: statistical σ, q ≤ 0.2, last point removed, unsmeared, and an optional calibration offset.
  - Fits are done in ln I with a relative weight floor, with Gaussian resolution smearing from the dQ column. Multistart with prescreening continues a best start that ran out of evaluations, so a reported failure is a real one.
  - References are set one by one or with a regex rule. The contrast is typed or computed from a phase formula and density (`saxs.core.contrast`).
  - Results save and load as JSON; tables export as xlsx / csv / md; one fit file per curve.
- **Series figures tab**: a series table (import / edit) drives:
  - the grid of normalised, shifted curves, with labels at the curve ends, letters and family labels;
  - fit-panel figures (data, model, labelled components, normalised residuals, I or I q⁴);
  - trends against your x-value (open markers = not quantified, faded = not robust);
  - the 2D frame figure.

  Export pdf / svg / png at any dpi. Tables are written as xlsx (frozen header, column widths), csv (utf-8 with BOM, ASCII-only cells so Excel never garbles a dash) and markdown.
- **Named-column curves** (`saxs.core.loader.load_tabular_curve`) keep dQ, the statistical σ, smearing flags and text columns. q in 1/nm is converted to 1/Å.
- **SasView-ready curves**: every reduction also writes three-column Q / I / dI files (usable range, per mm of material, `#` header only), which SasView 6.1.2 opens as 1D data; this was verified through SasView's own loader.
  - They are written to `curves_sasview/` by the pipeline and to `sasview/` next to the Frames tab's output folder, and are available from the tab's **Export for SasView…** button and `saxs_pipeline.py sasview CURVES_DIR`.
  - The ten-column provenance files stay PRISM's own format (SasView rejects them as 2D data).
- **Model-free descriptors** (`saxs.core.descriptors`):
  - local log-log slopes, low-q exponent and Porod constant;
  - high-q plateau and its ratio to the scaled empty;
  - excess invariant over the grain baseline;
  - Porod-plot shoulders and Kratky maximum;
  - an explicit maximum test in I, I q² and I q⁴, whose significance is checked against the propagated noise and the data's own scatter.
- **Headless pipeline**: `py -3.11 scripts/saxs_pipeline.py init project.json`, then:
  - `run project.json [--steps …]` runs inventory → reduce → describe → fit → robustness → figures → tables from one JSON file. Each step reads the previous step's files, so any step reruns alone;
  - a `curves` block reads curves reduced elsewhere;
  - `fit project.json` runs fit → robustness → figures → tables;
  - `quick CURVE --reference REF` fits one curve without a project file;
  - `manifest.json` records the code version, options, inputs, outputs and timings.
- **Validation**:
  - synthetic frames and curves;
  - model forms checked against their analytic limits (sphere forward scattering φΔρ²⟨V²⟩/⟨V⟩ and Porod limit, a dense numerical polydispersity average, the Percus–Yevick compressibility limit, the Teubner–Strey analytic peak, exact Schulz moments);
  - fit tests: recovery of known structures, gate decisions on structure-free curves, selection rules, JSON round trips;
  - UI wiring tests;
  - a real-data regression (env-gated: `PRISM_SAXS_REFERENCE_DIR`, `PRISM_SAXS_REDUCED_V2_DIR`) that reproduces a hand-made 43-frame campaign: pairing identical, reduced curves identical to the last digit, descriptors identical.

## Removed

- The **Composite fit** tab and its staged engine (`composite_*`, `_staged_*`), and the **Unified fit** tab and its two-level engine (`unified_levels`). Both are superseded by the Fit tab's single engine. Earlier fit results in their JSON formats are not read by the new tab.
- Unused legacy modules in `saxs/core` that nothing imported: `modelfit` (sasmodels/bumps bridge), `batch`, `session`, `report`, `transforms`.

## Fixes and internals

- `core/qt_worker.run_in_thread` accepts `on_progress`: the job receives a `progress(i, n, message)` callable whose calls reach the UI thread.
- The 1D loader recognises `I_corr` / `sigma_I` columns. `edf_sectors` gains `sector_anisotropy`, exposes the usable-range and centre options, and tolerates frames without a wavelength header (virtual-detector exports) when a default is given.
- The command line prints progress and flushes its log, so a long run redirected to a file shows where it is.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
