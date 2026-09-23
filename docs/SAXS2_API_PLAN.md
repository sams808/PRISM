# SAXS 2.0 module API

The SAXS 2.0 extension of PRISM's `saxs/` package: from raw 2D frames of a beamstopless instrument
to publication figures and tables, for any sample series. Written as the plan before coding
(2026-09-23) and brought in line with what was built. The design it implements is the author's
`PRISM_SAXS2_DESIGN.md` (kept outside the repo). Every function is sample-agnostic: names,
grouping, colours, markers and x-values come from the caller (a series table), never from code.

Conventions shared by all modules

- `saxs/core` is pure numpy / pandas / scipy / lmfit / matplotlib. `fabio`, `openpyxl`, `xraydb`
  are imported inside the functions that need them; nothing Qt.
- Thresholds live in option dataclasses (`ReductionOptions`, `DescriptorOptions`, `UnifiedOptions`)
  with the September 2026 rethink's values as defaults, `to_dict` / `from_dict`, and are written
  into every `.dat` header, result and pipeline manifest.
- A `progress(i, n, message)` callback is optional on every long step; `core.qt_worker.run_in_thread`
  forwards it to the UI thread through `on_progress`.
- Curves are `saxs.core.curve.Curve`. Per-point extra columns (sample / empty levels, ratio, usable
  flag, per-mm intensity) travel in `curve.metadata["columns"]` as lists (JSON-serialisable).

## edf_sectors.py (existing, extended)

- `FrameGeometry.from_header(h, default_wavelength_A=None)`, `read_frame(path, default_wavelength_A=None)`.
- `reduce_arrays(..., refine_centre, max_shift_px, bright_fraction, min_ratio, min_ratio_footprint,
  r_footprint_px, nsigma, min_run)`; metadata gains `n_hot_sample`, `n_hot_empty`,
  `empty_centre_offset_from_header_px`.
- `sector_anisotropy(data, geom, valid=None, half_deg=25, rmax_px=220, q_window=(2e-3, 1.5e-2),
  prominence=0.05, streak_band_px=(30, 60), lobe_min_ratio=2.0) -> SectorAnisotropy`
  (horizontal / vertical / diagonal / full profiles, `lobe_q` = a horizontal maximum whose H/V ratio
  exceeds `lobe_min_ratio`, `lobe_ratio`, `streak_h`, `streak_v`).

## frames.py

- `FrameRecord`: path, rel, folder, session, stem, name, comment_name, tag, collimation, configuration,
  role, date, exposure_s, distance_m, wavelength_A, pixel_m, shape, count_cutoff, header centre,
  `cx / cy` (the centre used), `cx_own / cy_own`, `centre_source`, `reference_empty`, centre offsets,
  direct-beam level (r < 4 and r < 5 px), max cps, saturated, n_negative, n_hot, lobe_q, lobe_prominence,
  lobe_ratio, streak_h, streak_v, capillary_mm, header. `energy_ev`, `datetime`, `geometry(refined)`.
- `scan_frames(roots, empty_pattern, calibrant_pattern, exclude_pattern, name_from="stem"|"comment",
  diagnostics=True, progress, on_error, max_shift_px, bright_fraction, default_wavelength_A)`:
  two passes — the empties first (their own direct-beam centroid), then every other frame with its
  session empty as the reference (own centroid only when the centre is bright and within
  `max_shift_px`, else the empty's centre: the reduction's rule). `exclude_pattern` is searched in
  the relative path, the stem and the full path.
- `frame_record(path, root, ..., reference=None)`, `diagnose_frame(rec, data, reference)`,
  `reference_empty(rec, empties)`.
- `pair_frames(records, rule="same_folder"|"nearest_time"|"explicit", explicit=None, distance_tol_m,
  default_capillary_mm, fallback_to_session, prefer_same_collimation) -> PairingResult`
  (`pairs: [FramePair]`, `unpaired`, `duplicate_names()`, `describe()`, `table()`).
- `inventory_table(records)`, `write_inventory(records, path)`.

## sector_batch.py

- `ReductionOptions` (sector, footprint, centre refinement, class thresholds, usable-range rule,
  plateau bands, per-mm probes, dat suffix).
- `MuSource.from_table({name: mu_mm}, key_regex)` / `.from_samples_table(csv|df, key_regex, energy_ev,
  ...)`: `mu_for(name, energy_ev)`, `rho_e_for(name)` (composition + density through
  `chemistry.calculate_absorption`; compositions in the chemistry syntax, e.g. `50.6(SiO2)13.6(B2O3)`).
- `classify_loading(mu_t, heterogeneity, options)`: unreliable (T4/T outside 0.3-3) first, then
  measurable (< 3), marginal (< 6), opaque.
- `ReducedSample`: name, curve (I_corr, cps/pixel; columns sample_cps_px, empty_cps_px, ratio, n_pix,
  usable, I_per_mm, sigma_per_mm), transmission, loading_class, centre_source, q range, mu, t_eff,
  session, paths, exposures, geometry, centres and offsets, ratio plateaus, hot pixels, flat-field
  factors, per-mm probes, dat_path, options, reduction (in memory only). `curve_per_mm()`,
  `usable_curve(per_mm=True)` (the input of descriptors and fits), `header_dict()`.
- `reduce_arrays_with_options`, `reduce_pair_records(pair, options, mu_source, cache)`,
  `reduce_pairs(pairs, options, mu_source, out_dir, progress, on_error)` (duplicate names get a
  `__2` suffix), `write_reduced_dat`, `load_reduced_dat`, `load_reduced_folder`, `summary_table`.

## descriptors.py

- `DescriptorOptions`; `rebin_log`, `local_slopes`, `slope_at`, `value_at`, `lowq_exponent`,
  `band_median`, `plateau`, `porod_constant`, `invariant`, `excess_over_grain`, `porod_shoulders`,
  `kratky_maximum`, `maximum_test(q, I, sigma, weights=("I","Iq2","Iq4"), smooth_pts=7,
  min_prominence_sigma=3, min_relative_prominence=0.05)` (noise = max of the propagated sigma and the
  data's own scatter), `has_maximum` (a maximum of I: a characteristic spacing),
  `has_kratky_maximum` (I q^2 only: a Guinier knee).
- `describe_curve(curve, options, rho_e) -> DescriptorResult` (values with the reference column names,
  maxima, shoulders, rebinned arrays, `row()`), `describe_batch(curves, options, rho_e, progress)
  -> (results, DataFrame)`, `descriptors_table`.

## composite_models.py (additions)

- `beaucage_polydisperse_prefactor(G, Rg, d)`, `BeaucageTied` (G, Rg, d, pf in 0.1-1.5),
  `RisingTail` (T, m fixed 2). `BeaucageUnified.eval` is written operation-for-operation as the
  reference script (least-squares paths are sensitive to last-ulp differences near a saddle).
- `composite_fit.PRESETS`: L1, L1_T, L1_L2, L1_L2_T.

## unified_levels.py

- `UnifiedOptions` (q_max, per_decade, sigma_floor, dbic_min, bounds of Rg1 / d1 / Rg2 / d2 / logpf2
  and of the log amplitudes, starts, tail_exponent, max_nfev, min_points, q_min_required,
  amplitude_floor, bound_tol, relaxed_d1_max, relaxed_rg1_max, robust_tol, use_usable; `relaxed()`).
- `components(values, q, l2, tail)`, `model(...)`, `prepare_curve(curve, options)`,
  `make_parameters`, `fit_candidate`, `forward_select(bics, g2, I_max, options)`, `is_at_bound`.
- `UnifiedFitResult` (sample, cls, fitted, skip_reason, l2, tail, n_pts, q range, n_params, chi2r,
  rms_log, base values, bic, dBIC_vs_base, dBIC_L2, dBIC_tail, params {name: ParamValue(value,
  stderr, min, max, at_bound)}, derived (Rg in nm, D_eq, G, B, pf, C, T), candidates, selection_path,
  warnings, covariance_ok, per_mm, q / I / sigma, options, relaxed, resolved; `model()`,
  `components()`, `residual_log()`, `at_bound_names()`, `to_json()` / `from_json()`).
- `fit_unified_levels`, `robustness_pass`, `resolved_flag`, `fit_with_robustness`,
  `batch_fit(curves, options, robustness=True, progress) -> (results, table)`, `results_table`,
  `result_row`, `save_results_json`, `load_results_json`.

## series_figures.py

- `SeriesTable` (required: sample, panel_row, panel_col, stack_index, label; optional: colour, marker,
  linestyle, x_value, family_label, col_title, quantified, note_symbol, group, fit_letter,
  extra_panels "row:col;…", fit_order); `from_csv`, `from_dataframe`, `validate`, `samples`, `has`,
  `get`, `shape`, `panels`, `samples_in_panel`, `groups`, `samples_in_group`, `extra_panels`,
  `colour`, `marker`, `linestyle`, `label`, `quantified`.
- `apply_style(style, overrides, dpi)`, `light_grid`, `rebinned_curve`, `grid_figure`,
  `draw_fit_panel`, `fit_panels_figure`, `TrendPanel`, `default_trend_panels`, `trends_figure`,
  `frame_figure`, `export_figure(fig, base, formats, dpi)`.

## tables.py

- `ascii_transliterate`, `format_cell`, `dataframe_to_markdown`, `transliterate_frame`,
  `write_tables(sheets, base_path, notes, md_sheet, formats) -> {format: path}`,
  `fit_sheet(results, summary, extra, amplitude_unit)`, `modelfree_sheet(descriptors, extra)`,
  `robustness_sheet(results)`, `default_notes(fit_options)`.

## pipeline.py and scripts/saxs_pipeline.py

- `ProjectConfig` (name, frames_roots, patterns, name_from, default_wavelength_A, pairing, samples_table,
  sample_key_regex, energy_ev, output_dir, reduction / descriptors / fit overrides, series_table,
  figures, tables; `load` / `save` / `template` / `resolve` / `out`), `STEPS`, `step_*`, `Manifest`,
  `run(config, steps, progress, log)`.
- CLI: `init CONFIG [--frames-root …] [--output-dir …] [--force]`, `run CONFIG [--steps …]`,
  `inventory` / `reduce` with `--config` or explicit roots. Exit codes 0 / 1 / 2 (unpaired samples).

## GUI (saxs/qt_saxs2.py, mixed into SaxsWorkspace)

- Frames tab: roots, patterns, pairing rule, Scan (worker + progress) → inventory table + diagnostics
  plot (crop about the centre + H / V / diagonal profiles), samples table, output folder, reduction
  options, Reduce all pairs (worker + progress) → results table, curves added to the workspace,
  summary CSV export.
- Unified fit tab: curve picker, options, Fit (worker) → plot with components and residuals,
  parameter table with stderr / at-bound, report with dBIC and candidates; Robustness check; Batch
  with progress; CSV / XLSX / JSON save and load.
- Series figures tab: series table (import / save / fill from curves / edit), Grid / Fit panels /
  Trends / 2D frame figure previews, export pdf / svg / png at a chosen dpi, tables export.
