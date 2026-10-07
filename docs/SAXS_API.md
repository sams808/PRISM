# SAXS module API

Reference of PRISM's `saxs/` package, which takes raw 2D frames of a beamstopless instrument, or curves reduced
elsewhere, through to physical fits, publication figures and tables, for any sample series. The fitting method and
how to run it are described in [SAXS_FITTING.md](SAXS_FITTING.md). Every function is sample-agnostic: names,
references, contrasts, grouping, colours, markers and x-values come from the caller, never from code.

Conventions shared by all modules:

- `saxs/core` is pure numpy / pandas / scipy / lmfit / matplotlib, with nothing Qt. `fabio`, `openpyxl` and `xraydb`
  are imported inside the functions that need them.
- Thresholds live in option dataclasses (`ReductionOptions`, `DescriptorOptions`, `WorkflowOptions`) with `to_dict`
  / `from_dict`. They are written into every `.dat` header, result file and pipeline manifest. `WorkflowOptions.from_dict`
  rejects unknown keys.
- Every long step takes an optional `progress(i, n, message)` callback. `core.qt_worker.run_in_thread` forwards it
  to the UI thread through `on_progress`.
- Curves are `saxs.core.curve.Curve`. Per-point extra columns travel in `curve.metadata["columns"]`: lists in the
  reduced curves of `sector_batch`, float arrays from the tabular loader (text columns stay lists). The resolution
  column is in `curve.metadata["dq"]`.
- Units: q in 1/Å, I in 1/cm (absolute curves), lengths in Å inside the fits and in nm in the reported tables. The
  unit is part of every column name.

## loader.py

- `load_curve(path, ...)`: two- or three-column text files.
- `load_tabular_curve(path, q_col=None, i_col=None, sigma_col=None, dq_col=None)`: named-column tables.
  - Columns are picked by name (`Q_NAMES`, `I_NAMES`, `SIGMA_NAMES`, `DQ_NAMES`) or given explicitly; every other
    column is kept.
  - q in 1/nm (`q_nm`, `q_invnm`) is converted to 1/Å, together with an unlabelled dQ.
  - Dropped rows are counted in `metadata["n_dropped"]`.

## edf_sectors.py

- `FrameGeometry.from_header(h, default_wavelength_A=None)`, `read_frame(path, default_wavelength_A=None)`.
- `reduce_arrays(..., refine_centre, max_shift_px, bright_fraction, min_ratio, min_ratio_footprint,
  r_footprint_px, nsigma, min_run)`. The metadata includes `n_hot_sample`, `n_hot_empty` and
  `empty_centre_offset_from_header_px`.
- `sector_anisotropy(data, geom, valid=None, half_deg=25, rmax_px=220, q_window=(2e-3, 1.5e-2), prominence=0.05,
  streak_band_px=(30, 60), lobe_min_ratio=2.0) -> SectorAnisotropy`:
  - horizontal / vertical / diagonal / full profiles;
  - `lobe_q`, a horizontal maximum whose H/V ratio exceeds `lobe_min_ratio`, and `lobe_ratio`;
  - `streak_h`, `streak_v`.

## frames.py

- `FrameRecord` fields:
  - identity: path, rel, folder, session, stem, name, comment_name, tag, collimation, configuration, role;
  - acquisition: date, exposure_s, distance_m, wavelength_A, pixel_m, shape, count_cutoff;
  - centre: header centre, `cx / cy` (the centre used), `cx_own / cy_own`, `centre_source`, `reference_empty`,
    centre offsets;
  - diagnostics: direct-beam level (r < 4 and r < 5 px), max cps, saturated, n_negative, n_hot, lobe_q,
    lobe_prominence, lobe_ratio, streak_h, streak_v;
  - capillary_mm, header.

  Methods: `energy_ev`, `datetime`, `geometry(refined)`.
- `scan_frames(roots, empty_pattern, calibrant_pattern, exclude_pattern, name_from="stem"|"comment",
  diagnostics=True, progress, on_error, max_shift_px, bright_fraction, default_wavelength_A)` works in two passes:
  1. the empties first, each with its own direct-beam centroid;
  2. every other frame, with its session empty as the reference. A frame keeps its own centroid only when the
     centre is bright and within `max_shift_px`; otherwise it takes the empty's centre.
- `frame_record`, `diagnose_frame`, `reference_empty`.
- `pair_frames(records, rule="same_folder"|"nearest_time"|"explicit", explicit=None, distance_tol_m,
  default_capillary_mm, fallback_to_session, prefer_same_collimation) -> PairingResult`. The result has `pairs`,
  `unpaired`, `duplicate_names()`, `describe()` and `table()`.
- `inventory_table(records)`, `write_inventory(records, path)`.

## sector_batch.py

- `ReductionOptions`: sector, footprint, centre refinement, class thresholds, usable-range rule, plateau bands,
  per-mm probes, dat suffix.
- `MuSource.from_table({name: mu_mm}, key_regex)` / `.from_samples_table(csv|df, key_regex, energy_ev, ...)`, with
  `mu_for(name, energy_ev)` and `rho_e_for(name)`.
- `classify_loading(mu_t, heterogeneity, options)` checks in order:
  1. unreliable: T4/T outside 0.3–3;
  2. measurable: μt < 3;
  3. marginal: μt < 6;
  4. opaque otherwise.
- `ReducedSample`: the reduced curve, its transmission, loading class, centre, usable q range, μ, effective
  thickness, provenance and plateaus. Methods: `curve_per_mm()`, `usable_curve(per_mm=True)` (the input of
  descriptors and fits), `header_dict()`.
- `reduce_pairs(pairs, options, mu_source, out_dir, progress, on_error, sasview_dir)`, `write_reduced_dat`,
  `load_reduced_dat`, `load_reduced_folder`, `summary_table`, `export_sasview`.

## descriptors.py

- `DescriptorOptions`.
- Building blocks: `rebin_log`, `local_slopes`, `slope_at`, `value_at`, `lowq_exponent`, `band_median`, `plateau`,
  `porod_constant`, `invariant`, `excess_over_grain`, `porod_shoulders`, `kratky_maximum`.
- `maximum_test(q, I, sigma, weights=("I","Iq2","Iq4"), ...)`. The noise is the larger of the propagated σ and the
  data's own scatter.
- `has_maximum` (a maximum of I: a characteristic spacing), `has_kratky_maximum` (in I q² only: a Guinier knee).
- `describe_curve(curve, options, rho_e) -> DescriptorResult`, `describe_batch(curves, options, rho_e, progress)
  -> (results, DataFrame)`, `descriptors_table`.

## models.py

Components, each with `params()` (name, start, bounds, unit, doc), `eval(q, **values)`, an optional `derived(...)`, a
`title` and its literature `reference`:

| Component | Parameters |
|---|---|
| `FlatBackground` | C |
| `PowerLaw` | B, p |
| `LorentzOZ` | A, xiL |
| `Guinier` | G, Rg |
| `GuinierPorod` | G, Rg, p |
| `BeaucageUnified` | G, Rg, B, p |
| `BeaucageTied` | G, Rg, d, pf; B tied by `beaucage_polydisperse_prefactor` |
| `Dab` | A, xi |
| `TeubnerStrey` | S, d, xi |
| `SphereSchulz` | phi, R, sig, drho, phi_hs |

`TeubnerStrey.derived` returns q_max, a2, c1, c2, fa and has_peak. `ts_classic_from_physical` /
`ts_physical_from_classic` convert between the two parameter sets. `SphereSchulz.derived` returns D_number,
D_vol_weighted, L_vol_scherrer and Rg_ensemble. Helpers: `sphere_amplitude`, `schulz_radii` (Golub–Welsch),
`schulz_moment`, `percus_yevick_sq`. `COMPONENTS` maps names to classes.

## resolution.py

- `smear_points(func, q, dq, n_nodes=20)`: point-by-point Gaussian smearing, Gauss–Hermite, truncated at q = 0.

## contrast.py

- `electron_density_cm3(comp, density)`, `xray_sld_1e10cm2(rho_e)`, `sld_1e10cm2(comp, density)`.
- `phase_contrast_1e10cm2(phase, rho_phase, sample, rho_sample)`.
- `stoichiometric_ceiling(phase, rho_phase, sample, rho_sample, ignore=("O",))`.
- `invariant(func, q_lo=1e-6, q_hi=100, n=20000)`, `two_phase_product_cm4(Q)`, `volume_fraction_from_invariant(Q,
  drho)`, `min_contrast_1e10cm2(Q)`.
- `as_composition`, `element_amounts`, `molar_mass`.
- Constants `R_E_CM` and `N_A` (CODATA 2018).

## physical_fit.py

- `TermSpec(component, prefix, values, bounds, fixed, free)`: one model term (`key`, `to_dict` / `from_dict`).
- `fit_physical(name, q, I, sigma, terms, dq=None, floor=0.03, q_range=None, starts=None, n_nodes=12,
  max_nfev=4000, prescreen=None, polish_rounds=3) -> PhysFitResult`:
  - least squares in ln I with a relative weight floor;
  - multistart, with an optional prescreen of the starts;
  - continuation of a best start that ran out of evaluations;
  - BIC, AIC, standard errors and at-bound flags;
  - excluded points counted, with the reason.
- `PhysFitResult`: `redchi`, `rms_log`, `any_at_bound`, `value`, `values`, `eval(q, dq)` (components + total),
  `to_dict` / `from_dict`.
- `is_at_bound`, `build_parameters`, `model_components`.
- `residual_runs(q, r, threshold) -> [Run]`.
- `reference_ratio(q, I, sigma, q_ref, I_ref, sigma_ref, flat, flat_ref, norm_window, rel_spread, threshold) ->
  RatioResult`.

## fit_workflow.py

- `WorkflowOptions`: all defaults and meanings are listed in SAXS_FITTING.md §5.
- `CurveInput(name, q, I, sigma, sigma_stat, dq, smeared_flag, reference, contrast, contrast_alt, phi_max, tags)`, and
  `CurveInput.from_curve(curve, reference, **kw)`.
- `run_workflow(inputs, options, progress) -> WorkflowResult` runs the bases, the reference null maximum, the gate,
  the structural candidates and the selection.
- `robustness(result, inputs, variants, progress) -> DataFrame`. The table is also stored on the result.
- Steps, usable alone: `fit_bases`, `null_maximum`, `reference_gate`, `fit_structures`, `select`, `derive`,
  `component_visible`, `sphere_in_window`, `parse_key`.
- Results:
  - `WorkflowResult(options, curves, null_max, null_pairs, created, robustness)`, with `save` / `load` (JSON);
  - `CurveResult(name, reference, fits, base, gate, fit_gate_span, retained, reasons, derived, ...)`, with
    `retained_fit` and `structural`;
  - `GateResult`.
- Tables: `summary_table(result, extra)`, `comparison_table`, `parameters_table`, `curve_table(cr, key)`,
  `robust_flags`, `param_robust`, `method_notes`.

## series_figures.py

- `SeriesTable`:
  - required columns: sample, panel_row, panel_col, stack_index, label;
  - optional: colour, marker, linestyle, x_value, family_label, col_title, quantified, note_symbol, group,
    fit_letter, extra_panels, fit_order;
  - methods: `from_csv`, `from_dataframe`, `validate`, plus accessors.
- `apply_style(style, overrides, dpi)`, `light_grid`, `rebinned_curve`, `grid_figure`.
- Fit figures: `component_label`, `draw_fit_panel(ax, axr, cr, table, letter, x_range, iq4, model_key)`,
  `fit_panels_figure(results, samples, table, ncol, ..., iq4)`.
- Trends and frames: `TrendPanel`, `default_trend_panels`, `trends_figure`, `frame_figure`.
- `export_figure(fig, base, formats, dpi)`.

## tables.py

- `ascii_transliterate`, `format_cell`, `dataframe_to_markdown`, `transliterate_frame`.
- `write_tables(sheets, base_path, notes, md_sheet, formats) -> {format: path}`.
- `modelfree_sheet(descriptors, extra)`.
- `workflow_sheets(result, extra, descriptors)`: the summary, models, parameters and robustness sheets.
- `write_workflow_tables(result, base_path, extra, descriptors, formats)`.

## pipeline.py and scripts/saxs_pipeline.py

- `ProjectConfig` (`load` / `save` / `template` / `resolve` / `out`). Its fields cover:
  - raw frames: frames_roots, patterns, pairing;
  - a `curves` block for curves reduced elsewhere;
  - samples_table, output_dir, the reduction / descriptors / fit blocks, series_table, figures, tables.
- `STEPS`, `step_*`, `Manifest`, `run(config, steps, progress, log)`.
- Fit inputs: `references_for`, `contrasts_for`, `tags_for`, `fit_inputs`.
- Command line:
  - `init CONFIG [--frames-root …] [--curves-dir DIR] [--output-dir …] [--force]`;
  - `run CONFIG [--steps …]`;
  - `fit CONFIG [--no-robustness] [--no-figures]`;
  - `quick CURVE [--reference REF] [--contrast DRHO] [-o OUT] [--iq4] [--families …] [--no-second-population]
    [--no-robustness]`;
  - `inventory` / `reduce` (with `--config` or explicit roots);
  - `sasview CURVES_DIR`.

  Exit codes: 0 success, 1 error, 2 unpaired samples.

## GUI (mixed into saxs/qt_saxs.py SaxsWorkspace)

- **Frames** tab (`qt_saxs2.py`):
  - roots, patterns and pairing rule; **Scan** fills the inventory table and the diagnostics plot;
  - samples table, output folder and reduction options; **Reduce all pairs** fills the results table and adds the
    curves to the workspace;
  - summary CSV and SasView export.
- **Fit** tab (`qt_saxs_fit.py`):
  - curve list, reference (Set / regex rule) and contrast (Set / Compute…), options;
  - **Fit selected curves** and **Robustness of the retained models**;
  - results table, model selector, plot (I or I q⁴) with components and residuals, report, parameter table;
  - Save / Load (JSON), Tables (xlsx / csv / md), Curves (one csv per curve).
- **Series figures** tab (`qt_saxs2.py`):
  - series table (import / save / fill from curves / edit);
  - grid, fit panels, trends and 2D frame figure;
  - export pdf / svg / png, and the tables.
