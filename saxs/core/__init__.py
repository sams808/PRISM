"""
saxs.core — the SAXS/WAXS science engine, ported from the author's own
'pomme' project (github.com/sams808/pomme) into PRISM as the SAXS/WAXS
module. Framework-agnostic: Curve model with provenance, Xenocs-style 1D
ASCII loader, physics-based empty/background reduction (xraydb),
Guinier/Porod/peak/unified analysis, WAXS multi-peak fitting, and the
composite-model staged-fitting pipeline (component library, presets,
staged fit, batch runner).

Re-exports the names qt_saxs.py (and most other external callers) need,
so `from saxs.core import Curve, load_curve, fit_staged, ...` works
without reaching into individual submodules.
"""
from saxs.core.analysis import (
    auto_detect_guinier_region,
    auto_detect_peak_window,
    auto_detect_porod_region,
    fit_guinier,
    fit_porod_general,
    fit_pseudo_bragg_peak,
)
from saxs.core.chemistry import CapillaryConfig, SamplePhysicsConfig
from saxs.core.composite_batch import BatchItem, batch_to_csv_rows, run_batch, write_batch_csv
from saxs.core.composite_fit import PRESETS, CompositeModel, build_composite, build_preset
from saxs.core.composite_models import COMPONENTS
from saxs.core.composite_staged import FitResult, fit_staged, propose_windows
from saxs.core.curve import Curve
from saxs.core.edf_sectors import SectorReduction, reduce_arrays, reduce_frame_pair, sector_anisotropy
from saxs.core.frames import FramePair, FrameRecord, PairingResult, inventory_table, pair_frames, scan_frames, write_inventory
from saxs.core.sector_batch import (MuSource, ReducedSample, ReductionOptions, classify_loading, export_sasview, load_reduced_dat,
                                    load_reduced_folder, reduce_pairs, summary_table, write_reduced_dat, write_sasview_dat)
from saxs.core.descriptors import DescriptorOptions, DescriptorResult, describe_batch, describe_curve, maximum_test, rebin_log
from saxs.core.unified_levels import (UnifiedFitResult, UnifiedOptions, batch_fit, fit_unified_levels, fit_with_robustness,
                                      load_results_json, robustness_pass, save_results_json)
from saxs.core.series_figures import SeriesTable, export_figure, fit_panels_figure, frame_figure, grid_figure, trends_figure
from saxs.core.tables import fit_sheet, modelfree_sheet, robustness_sheet, write_tables
from saxs.core.pipeline import ProjectConfig, STEPS as PIPELINE_STEPS, run as run_pipeline
from saxs.core.loader import load_curve
from saxs.core.reduction import CorrectionSettings, correct_sample
from saxs.core.waxs import auto_find_peaks, fit_waxs_peaks

__version__ = "PRISM"

__all__ = [
    "auto_detect_guinier_region", "auto_detect_peak_window", "auto_detect_porod_region",
    "fit_guinier", "fit_porod_general", "fit_pseudo_bragg_peak",
    "CapillaryConfig", "SamplePhysicsConfig",
    "BatchItem", "batch_to_csv_rows", "run_batch", "write_batch_csv",
    "PRESETS", "CompositeModel", "build_composite", "build_preset",
    "COMPONENTS",
    "FitResult", "fit_staged", "propose_windows",
    "Curve",
    "SectorReduction", "reduce_arrays", "reduce_frame_pair", "sector_anisotropy",
    "FramePair", "FrameRecord", "PairingResult", "inventory_table", "pair_frames", "scan_frames", "write_inventory",
    "MuSource", "ReducedSample", "ReductionOptions", "classify_loading", "export_sasview", "load_reduced_dat", "load_reduced_folder",
    "reduce_pairs", "summary_table", "write_reduced_dat", "write_sasview_dat",
    "DescriptorOptions", "DescriptorResult", "describe_batch", "describe_curve", "maximum_test", "rebin_log",
    "UnifiedFitResult", "UnifiedOptions", "batch_fit", "fit_unified_levels", "fit_with_robustness", "load_results_json",
    "robustness_pass", "save_results_json",
    "SeriesTable", "export_figure", "fit_panels_figure", "frame_figure", "grid_figure", "trends_figure",
    "fit_sheet", "modelfree_sheet", "robustness_sheet", "write_tables",
    "ProjectConfig", "PIPELINE_STEPS", "run_pipeline",
    "load_curve",
    "CorrectionSettings", "correct_sample",
    "auto_find_peaks", "fit_waxs_peaks",
]
