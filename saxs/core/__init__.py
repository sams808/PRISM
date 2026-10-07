"""
saxs.core - the SAXS/WAXS science engine of PRISM (framework-agnostic, nothing Qt), first ported from the
author's own pomme project (github.com/sams808/pomme).

Data and reduction: the Curve container with provenance, ASCII loaders (``load_curve`` for generic two/three-column
files, ``load_tabular_curve`` for named-column tables with dQ / statistical sigma / flags), empty subtraction of 1D
curves, sector reduction of beamstopless 2D frames (frames, edf_sectors, sector_batch), model-free descriptors,
Guinier / Porod / pseudo-Bragg analysis and WAXS peak fitting.

Fitting (one engine): the model library (``models``), the single-model fit with per-point resolution smearing
(``physical_fit``), the multi-curve physical workflow with a model-free reference gate, model families, selection
rules and robustness (``fit_workflow``), contrast / invariant / volume-fraction helpers (``contrast``).

Outputs: series figures (``series_figures``), tables (``tables``) and the JSON project pipeline (``pipeline``,
CLI ``scripts/saxs_pipeline.py``). See docs/SAXS_FITTING.md.
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
from saxs.core.contrast import (electron_density_cm3, invariant, min_contrast_1e10cm2, phase_contrast_1e10cm2,
                                sld_1e10cm2, stoichiometric_ceiling, volume_fraction_from_invariant, xray_sld_1e10cm2)
from saxs.core.curve import Curve
from saxs.core.descriptors import DescriptorOptions, DescriptorResult, describe_batch, describe_curve, maximum_test, rebin_log
from saxs.core.edf_sectors import SectorReduction, reduce_arrays, reduce_frame_pair, sector_anisotropy
from saxs.core.fit_workflow import (CurveInput, CurveResult, WorkflowOptions, WorkflowResult, comparison_table,
                                    parameters_table, robustness, run_workflow)
from saxs.core.fit_workflow import summary_table as fit_summary_table
from saxs.core.frames import FramePair, FrameRecord, PairingResult, inventory_table, pair_frames, scan_frames, write_inventory
from saxs.core.loader import load_curve, load_tabular_curve
from saxs.core.models import COMPONENTS
from saxs.core.physical_fit import PhysFitResult, TermSpec, fit_physical, reference_ratio
from saxs.core.pipeline import ProjectConfig, STEPS as PIPELINE_STEPS, run as run_pipeline
from saxs.core.reduction import CorrectionSettings, correct_sample
from saxs.core.resolution import smear_points
from saxs.core.sector_batch import (MuSource, ReducedSample, ReductionOptions, classify_loading, export_sasview, load_reduced_dat,
                                    load_reduced_folder, reduce_pairs, summary_table, write_reduced_dat, write_sasview_dat)
from saxs.core.series_figures import SeriesTable, export_figure, fit_panels_figure, frame_figure, grid_figure, trends_figure
from saxs.core.tables import modelfree_sheet, write_tables, write_workflow_tables
from saxs.core.waxs import auto_find_peaks, fit_waxs_peaks

__all__ = [
    "auto_detect_guinier_region", "auto_detect_peak_window", "auto_detect_porod_region",
    "fit_guinier", "fit_porod_general", "fit_pseudo_bragg_peak",
    "CapillaryConfig", "SamplePhysicsConfig",
    "electron_density_cm3", "invariant", "min_contrast_1e10cm2", "phase_contrast_1e10cm2", "sld_1e10cm2",
    "stoichiometric_ceiling", "volume_fraction_from_invariant", "xray_sld_1e10cm2",
    "Curve",
    "DescriptorOptions", "DescriptorResult", "describe_batch", "describe_curve", "maximum_test", "rebin_log",
    "SectorReduction", "reduce_arrays", "reduce_frame_pair", "sector_anisotropy",
    "CurveInput", "CurveResult", "WorkflowOptions", "WorkflowResult", "comparison_table", "parameters_table",
    "robustness", "run_workflow", "fit_summary_table",
    "FramePair", "FrameRecord", "PairingResult", "inventory_table", "pair_frames", "scan_frames", "write_inventory",
    "load_curve", "load_tabular_curve",
    "COMPONENTS",
    "PhysFitResult", "TermSpec", "fit_physical", "reference_ratio",
    "ProjectConfig", "PIPELINE_STEPS", "run_pipeline",
    "CorrectionSettings", "correct_sample",
    "smear_points",
    "MuSource", "ReducedSample", "ReductionOptions", "classify_loading", "export_sasview", "load_reduced_dat", "load_reduced_folder",
    "reduce_pairs", "summary_table", "write_reduced_dat", "write_sasview_dat",
    "SeriesTable", "export_figure", "fit_panels_figure", "frame_figure", "grid_figure", "trends_figure",
    "modelfree_sheet", "write_tables", "write_workflow_tables",
    "auto_find_peaks", "fit_waxs_peaks",
]
