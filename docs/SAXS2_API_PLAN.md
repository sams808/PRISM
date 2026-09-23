# SAXS 2.0 module API plan

Working plan for the SAXS 2.0 extension of PRISM's `saxs/` package (design: the author's
`PRISM_SAXS2_DESIGN.md`, kept outside the repo). Written before coding and updated as the phases
land. Every function below is sample-agnostic: names, grouping, colours, markers and x-values come
from the caller (a series table), never from code.

Conventions shared by all modules

- `saxs/core` is pure numpy / pandas / scipy / lmfit / matplotlib. `fabio`, `pyFAI`, `openpyxl`,
  `xraydb` are imported inside the functions that need them.
- Thresholds live in option dataclasses with the rethink's values as defaults; every output header
  and the pipeline manifest record them.
- A `progress(i, n, message)` callback is optional on every long step.
- Curves are `saxs.core.curve.Curve`. Per-point extra columns (ratio, usable flag, empty level,
  per-mm intensity) travel in `curve.metadata["columns"]` as lists so sessions stay JSON.

## edf_sectors.py (existing, additions)

```python
@dataclass
class SectorAnisotropy:
    r_px: np.ndarray; q: np.ndarray
    horizontal: np.ndarray; vertical: np.ndarray; diagonal: np.ndarray; full: np.ndarray   # cps/pixel
    lobe_q: Optional[float]; lobe_prominence: float; lobe_ratio: float                  # H/V at lobe_q
    streak_h: float; streak_v: float                                                    # axis / diagonal at 30-60 px

def sector_anisotropy(data, geom, valid=None, half_deg=25.0, rmax_px=220, q_window=(2e-3, 1.5e-2),
                      prominence=0.05, streak_band_px=(30, 60)) -> SectorAnisotropy
```

`reduce_arrays` gains the usable-range parameters (`min_ratio`, `min_ratio_footprint`,
`r_footprint_px`, `nsigma`, `min_run`) and the centre-refinement parameters (`max_shift_px`,
`bright_fraction`) so `ReductionOptions` can drive it; defaults unchanged.

## frames.py

```python
@dataclass
class FrameRecord:
    path: str; rel: str; folder: str; session: str; stem: str
    name: str                      # sample identity (file stem by default, comment name on request)
    comment_name: str; tag: str; collimation: str; configuration: str
    role: str                      # sample | empty | calibrant | other
    date: str; exposure_s: float; distance_m: float; wavelength_A: float; pixel_m: float
    shape: tuple[int, int]; count_cutoff: float
    cx_header: float; cy_header: float
    cx: Optional[float]; cy: Optional[float]           # refined from the recorded direct beam
    centre_offset_px: Optional[tuple]; centre_offset_mm: Optional[float]
    direct_beam_cps: float; max_cps: float; saturated: bool; n_negative: int; n_hot: int
    lobe_q: Optional[float]; lobe_ratio: float; streak_h: float; streak_v: float
    capillary_mm: Optional[float]                    # parsed from the name ("0p5mm", "1.5 mm")
    header: dict[str, str]                           # the raw EDF header

def scan_frames(roots, empty_pattern=r"(?i)empty", calibrant_pattern=r"(?i)agbeh", exclude_pattern=None,
                name_from="stem", diagnostics=True, progress=None) -> list[FrameRecord]
def frame_record(path, root, ...) -> FrameRecord
def pair_frames(records, rule="same_folder", explicit=None, distance_tol_m=0.05,
                default_capillary_mm=1.0, fallback_to_session=True) -> PairingResult
    # PairingResult.pairs: list[FramePair(sample: FrameRecord, empty: FrameRecord)]
    # PairingResult.unpaired: list[FrameRecord]; .describe() -> str
def inventory_table(records) -> pd.DataFrame
def write_inventory(records, path) -> None
```

Pairing rules: `same_folder` (empties of the same immediate folder, same configuration, same
capillary token; nearest header date; optional fall-back to the session = top-level folder under the
root), `nearest_time` (any empty of the same configuration and capillary, nearest date),
`explicit` (`{sample name or path: empty path}` or a CSV with columns `sample, empty`).

## sector_batch.py

```python
@dataclass
class ReductionOptions:
    axis_exclusion_deg=20.0; bar_rect_px=(40.0, 14.0); rmax_px=650
    footprint_r_px=16.0; vertical_half_deg=30.0
    refine_centre=True; max_shift_px=4.0; bright_fraction=0.01
    mu_t_measurable=3.0; mu_t_marginal=6.0; heterogeneity_range=(0.3, 3.0)
    min_ratio=1.25; min_ratio_footprint=5.0; r_footprint_px=30.0; nsigma=3.0; min_run=8
    plateau_bands_px=((6, 12), (12, 20)); per_mm_probe_q=(0.02, 0.05, 0.15)

class MuSource:                     # mu (1/mm) per sample name
    @classmethod from_table(mapping) ; @classmethod from_samples_table(df, energy_ev=None, key_regex=None)
    def mu_for(name, energy_ev) -> Optional[float]; def rho_e_for(name) -> Optional[float]

def classify_loading(mu_t, heterogeneity, options) -> str    # measurable|marginal|opaque|unreliable
@dataclass
class ReducedSample:
    name, session, sample_path, empty_path, curve (Curve, cps/pixel), curve_per_mm (Curve or None),
    transmission (Transmission), mu_mm, mu_t, t_eff_mm, loading_class, centre_source, centre_offset_px,
    empty_centre_offset_px, q_min_use, q_max_use, n_use, plateau ratios, n_hot_sample, n_hot_empty,
    ff_sample, ff_empty, per_mm_probe {q: I}, dat_path
def reduce_pair_records(pair, options, mu_source) -> ReducedSample
def reduce_pairs(pairs, options, mu_source, out_dir=None, progress=None) -> list[ReducedSample]
def summary_table(results) -> pd.DataFrame          # phase1_summary.csv columns
def write_reduced_dat(result, path) -> None          # reduced_v2 column layout, "# key: value" header
def load_reduced_dat(path) -> ReducedSample          # round trip (curve + per-mm curve + columns)
def usable_per_mm_curve(result) -> Curve             # per-mm, usable range only (fits / descriptors input)
```

## descriptors.py

```python
@dataclass
class DescriptorOptions:
    per_decade=40; slope_window_dec=0.15; slopes_at=(6e-3, 1e-2, 2e-2, 4e-2)
    lowq=(4.5e-3, 1.2e-2); hiq=(0.15, 0.20); shoulder_range=(7e-3, 8e-2); excess_qmax=0.1
    min_points=20; maximum_smooth_pts=7; maximum_min_prominence_sigma=3.0
def rebin_log(q, I, sigma, per_decade=40) -> (qb, Ib, sb)
def local_slopes(q, I, window_dec=0.15) -> np.ndarray
def lowq_exponent(q, I, sigma, qrange) -> (n, n_err, K, npts)
def plateau(q, I, qrange) -> (C, mad)
def porod_constant(q, I, qrange) -> K4
def invariant(q, I, qrange) -> Q
def maximum_test(q, I, sigma, weights=("I", "Iq2", "Iq4"), smooth_pts=7, min_prominence_sigma=3) -> dict
def porod_shoulders(q, I, C, qrange) -> list[(q, prominence, size_nm)]
@dataclass
class DescriptorResult: sample, cls, n_use, values (dict of the phase2 columns), has_maximum, maxima, curves (rebinned)
def describe_curve(curve, options=None, rho_e=None) -> DescriptorResult
def describe_batch(curves, options=None, rho_e=None, progress=None) -> pd.DataFrame
```

## composite_models.py (additions)

```python
def beaucage_polydisperse_prefactor(G, Rg, d) -> float          # Beaucage 1996
class BeaucageTied(Component):   name = "beaucage_tied"; params G, Rg, d, pf (0.1-1.5)
class RisingTail(Component):     name = "rising_tail";   params T (>= 0), m (fixed 2.0)
PRESETS += {"L1": [flat_background, beaucage_unified], "L1_T": [+ rising_tail],
            "L1_L2": [flat_background, beaucage_unified, beaucage_tied], "L1_L2_T": [+ rising_tail]}
```

## unified_levels.py

```python
@dataclass
class UnifiedOptions:
    q_max=0.20; per_decade=40; sigma_floor=0.03; dbic_min=10.0
    rg1_bounds=(250.0, 1500.0); d1_bounds=(3.0, 4.5); rg2_bounds=(40.0, 200.0); d2_bounds=(2.0, 4.5)
    logpf2_bounds=(-1.0, 0.18); rg1_starts=(400.0, 700.0); rg2_starts=(80.0, 150.0)
    d1_start=4.0; d2_start=3.5; logpf2_start=-0.3; tail_exponent=2.0; max_nfev=4000
    min_points=30; q_min_required=0.01; amplitude_floor=1e-6
    relaxed_d1_max=5.0; relaxed_rg1_max=3000.0; robust_tol=0.20
@dataclass
class ParamValue: value, stderr, min, max, at_bound
@dataclass
class UnifiedFitResult:
    sample, cls, n_pts, q_min, q_max, l2, tail, n_params, chi2r, rms_log, chi2r_base, rms_base, bic,
    dBIC_vs_base, dBIC_L2, dBIC_tail, params: dict[str, ParamValue], derived: dict, candidates: dict,
    warnings: list[str], covariance_ok: bool, q, I, sigma (rebinned arrays), options: dict,
    relaxed: Optional[UnifiedFitResult], resolved: Optional[bool]
    def model(q) -> array; def components(q) -> dict; def to_json(); @classmethod from_json()
def prepare_curve(curve, options) -> (q, I, sigma)           # usable, q <= q_max, rebinned
def fit_candidate(q, I, sigma, l2, tail, options) -> lmfit MinimizerResult
def fit_unified_levels(curve, options=None) -> UnifiedFitResult
def robustness_pass(curve, result, options) -> UnifiedFitResult   # adds .relaxed and .resolved
def fit_with_robustness(curve, options=None) -> UnifiedFitResult
def batch_fit(curves, options=None, robustness=True, progress=None) -> (list[UnifiedFitResult], DataFrame)
def results_table(results) -> pd.DataFrame        # phase3e columns + robustness columns
def save_results_json(results, path); def load_results_json(path) -> list[UnifiedFitResult]
```

## series_figures.py

```python
class SeriesTable:                # wraps a DataFrame
    REQUIRED = ("sample", "panel_row", "panel_col", "stack_index", "label")
    OPTIONAL = ("colour", "marker", "linestyle", "x_value", "family_label", "col_title", "quantified",
                "note_symbol", "group", "fit_letter")
    @classmethod from_csv(path); @classmethod from_dataframe(df); def validate(); def row(sample)
    def panels() -> list[(row, col)]; def samples_in_panel(row, col) -> ordered list; def groups()
def apply_style(style="Publication", overrides=None) -> rc_context
def grid_figure(curves, table, shift_decades=1.6, normalise=True, x_range=(4e-3, 0.62), q_max=0.22,
                per_decade=60, letters=True, style="Publication", figsize=None) -> Figure
def fit_panels_figure(results, curves, samples, table=None, ncol=3, letters=True, title=None,
                      x_range=(4.5e-3, 0.22), style="Publication") -> Figure
@dataclass
class TrendPanel: key, source ("descriptors"|"fits"), column, ylabel, yscale="linear", ylim=None,
                  transform=None, err_column=None, only_quantified=False, fade_unresolved=False, require=None
def default_trend_panels() -> list[TrendPanel]
def trends_figure(table, descriptors_df, fits_df, panels=None, x_label="x", style="Publication") -> Figure
def frame_figure(sample_frame, empty_frame, crop_px=45, q_marker=None, titles=None, style="Publication") -> Figure
def export_figure(fig, base, formats=("pdf", "svg", "png"), dpi=600) -> list[str]
```

## tables.py

```python
def ascii_transliterate(text: str) -> str
def write_tables(sheets: dict[str, pd.DataFrame], base_path, notes=None, md_sheet=None,
                 formats=("xlsx", "csv", "md")) -> dict[str, str]
def dataframe_to_markdown(df, notes=None) -> str
```

## pipeline.py and scripts/saxs_pipeline.py

```python
@dataclass
class ProjectConfig:
    name; frames_roots; empty_pattern; calibrant_pattern; exclude_pattern; pairing (rule, explicit)
    samples_table (csv path); sample_key_regex; energy_ev (override); output_dir
    reduction (ReductionOptions dict); descriptors (DescriptorOptions dict); fit (UnifiedOptions dict)
    series_table (csv path); figures (grid / fit_panels / trends / frame specs, formats, dpi)
    tables (base name, notes, x_label); 
    @classmethod template(); @classmethod load(path); def save(path)
STEPS = ("inventory", "reduce", "describe", "fit", "robustness", "figures", "tables")
def run(config, steps=STEPS, progress=None, log=print) -> Manifest
```

CLI: `saxs_pipeline.py init CONFIG [--frames-root DIR ...]`, `inventory CONFIG`,
`run CONFIG [--steps ...]`, `reduce CONFIG` (= run --steps inventory reduce).

## GUI (qt_saxs.py)

- "Frames" tab: roots / patterns / pairing rule, Scan (worker), inventory table, diagnostics plot for
  the selected frame, Reduce all (worker) with progress, results table, curves sent to the workspace list.
- "Unified fit" tab: curve picker, options, Fit (worker), components + residual plot, parameter table
  with stderr / at-bound, Robustness check, Batch on selected curves, CSV/XLSX export, JSON save/load.
- "Series figures" tab: series table (import / edit), Grid / Fit panels / Trends / Frame figure previews,
  Export (formats, dpi), Tables export.
