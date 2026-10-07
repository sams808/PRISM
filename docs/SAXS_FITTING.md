# SAXS physical fitting

PRISM fits reduced 1D SAXS curves with a single, physically constrained engine (`saxs.core.fit_workflow`, built on
`saxs.core.physical_fit`). For every curve it answers three questions, in this order:

1. **What is the featureless baseline?** A grain power law plus a high-q level.
2. **Is there anything else to fit?** Decided model-free, by comparison with a reference curve.
3. **If so, which physical model describes it, with which sizes, and how robust are those numbers?**

A structure is reported only when it passes explicit rules. Every candidate that was tried stays in the result, with
the reason it was rejected. The engine knows nothing about samples: names, references, contrasts and figure layouts
are supplied by the user.

- [1. Input data](#1-input-data)
- [2. Running a fit](#2-running-a-fit) — in the application, one curve from the command line, a series from a project file
- [3. Method](#3-method) — weights, base, gate, candidates, selection, derived quantities, robustness
- [4. Reading the results](#4-reading-the-results)
- [5. Options](#5-options)
- [6. Project file reference](#6-project-file-reference)
- [7. Python API](#7-python-api)
- [8. Limits and good practice](#8-limits-and-good-practice)
- [9. References](#9-references)

## 1. Input data

Curves are named-column text tables (tab, comma or whitespace separated; `#` comment lines before the header). The
loader (`saxs.core.loader.load_tabular_curve`) picks the columns by name, case-insensitively:

| Quantity | Column names recognised | Needed | Notes |
|---|---|---|---|
| q | `q`, `q_A`, `q_invA`, `q_1/A`, `q(1/A)`; `q_nm`, `q_invnm` | yes | 1/Å. Columns in 1/nm are divided by 10 on load, and so is an unlabelled dQ column (recorded in `metadata["q_unit_converted"]`). |
| I | `I`, `I_cm`, `intensity`, `I_abs`, `dsigma_domega`, `corrected` | yes | Absolute units (1/cm) are needed for volume fractions. Sizes do not depend on the scale. |
| σ | `sigma`, `sigma_total_cm`, `sigma_cm`, `dI`, `err`, `error`, `sigma_corrected`, `Idev` | recommended | Total uncertainty. 3 % of I when absent. |
| statistical σ | `sigma_stat…`, `dI_stat…` | optional | Counting statistics only; used by the `stat` robustness variant. |
| dQ | `dQ`, `dQ_A`, `dQ_invA`, `sigma_q`, `Qdev` | optional | Gaussian σ of the q resolution (1/Å). Turns on resolution smearing. |
| smeared flags | `smeared…`, `…smear…flag…` | optional | 1 where a Gaussian dQ under-represents the beam. Those points are left out of the `nosmear` variant. |

Every other column, text included, is kept with the curve. Rows with a non-finite q or I, or with q ≤ 0, are dropped
and counted, never silently. Plain two- or three-column files (q, I, σ) also load in the Curves tab, but they carry
no dQ.

Three inputs make a fit much more informative:

- **A reference curve per sample**: the same material without the minority component, such as a parent
  composition, an untreated state or the matrix alone. The gate (§3.3) compares each curve with its reference. With
  two or more references, the gate threshold is measured from the data. With fewer, it falls back to 1, and the
  result says so.
- **The particle contrast Δρ** (10¹⁰ cm⁻²). It puts spheres on an absolute scale and turns invariants into volume
  fractions. Type it, take it from a samples-table column, or compute it from a phase formula and density against
  the sample's composition and density. The X-ray SLD is r_e·ρ_e (`saxs.core.contrast`). The same computation gives
  the stoichiometric ceiling of the phase.
- **dQ**: without it the steep low-q end of a grain law is fitted unsmeared. The error is small but systematic.

## 2. Running a fit

### 2.1 In the application: SAXS/WAXS → Fit

1. **Curves** tab: load the curves. Named-column files keep dQ, the statistical σ and the flags.
2. **Fit** tab:
   - select the curves to fit, and include their references;
   - *Reference of the selected curves*: choose one and press **Set**, or write a regex rule on the names and press
     **Apply rule**. For example `^(.*)_x\d+$` → `\1_x0` gives every member of a series its x = 0 member;
   - *Particle contrast*: type Δρ and press **Set**, or use **Compute…** with the phase formula and density plus the
     sample composition and density (this also gives the stoichiometric ceiling);
   - *Options*: the defaults are the recommended values (§5);
   - **Fit selected curves** runs in the background with a progress bar.
3. Inspect the results:
   - The table lists every curve with its gate, retained model, χ²_red, main result and robustness verdict.
   - Selecting a row shows the plot: data, model, labelled components and normalised residuals, as I or I q⁴.
   - It also shows the report (the gate numbers, and every candidate with its ΔBIC and verdict) and the parameter
     table (values, standard errors, bounds, at-bound flags).
   - The *model* box shows any other candidate.
4. **Robustness of the retained models** refits the retained models on perturbed data (§3.7). Then:
   - **Save…** writes JSON, which **Load…** reads back;
   - **Tables…** writes xlsx, csv and md;
   - **Curves…** writes one csv per curve with q, I, model, residual and every component.
5. **Series figures** tab: publication figures from a series table (grid of curves, fit panels, trends).

### 2.2 One curve from the command line

```
py -3.11 scripts/saxs_pipeline.py quick sample.dat --reference parent.dat --contrast DRHO -o sample_fit
```

`DRHO` is the particle contrast in 10¹⁰ cm⁻². Leave it out to fit spheres on a relative scale.

This writes `fits.json`, `tables.xlsx/.csv/.md`, `<curve>_fit.csv` and `fits.pdf/.png`. For each curve it prints
the retained model and every rejection reason. Without `--reference` the gate is off: structures are offered and
flagged. Other flags:
- `--families teubner_strey debye` limits the candidates;
- `--no-second-population` drops the +UNI2 candidates;
- `--iq4` draws I q⁴;
- `--no-robustness` skips the refits.

### 2.3 A series from a project file

```
py -3.11 scripts/saxs_pipeline.py init project.json --curves-dir reduced_curves
#   edit project.json: references, contrast, series table, figures, tables
py -3.11 scripts/saxs_pipeline.py fit project.json
```

`fit` runs fit → robustness → figures → tables. `run project.json --steps …` runs any subset of inventory, reduce,
describe, fit, robustness, figures and tables. Each step reads the files of the previous one, so any step reruns
alone. `manifest.json` records the code version, options, inputs, outputs and timings of every step. A project can
also start from raw 2D frames (`init --frames-root …`); the inventory and reduce steps then produce the curves.

A complete example. Paths are relative to the project file, and §6 lists every key. The density is that of corundum
(α-Al₂O₃), calculated from its unit cell (a = 4.759 Å, c = 12.991 Å, Z = 6); use your phase's density and keep its
source.

```json
{
  "name": "Doped series",
  "curves": {"dir": "reduced", "pattern": "*_merged.dat", "name_regex": "^(.*)_merged$"},
  "samples_table": "samples.csv",
  "output_dir": "fits_output",
  "fit": {
    "options": {"dbic_min": 10},
    "references": {"pattern": "^(S\\d)_x\\d+$", "replace": "\\1_x0"},
    "contrast": {"phase": "Al2O3", "density_g_cm3": 3.987,
                 "composition_column": "composition", "density_column": "density_g_cm3"},
    "tags": {"holder": "header:holder (H\\d)"},
    "robustness_variants": ["stat", "qmax", "noedge", "nosmear"]
  },
  "series_table": "series.csv",
  "figures": {
    "formats": ["pdf", "png"], "dpi": 600,
    "fit_panels": [{"name": "Fig_fits_{group}", "group": "*", "ncol": 3},
                   {"name": "Fig_fits_structural_Iq4", "only": "structural", "iq4": true}],
    "trends": {"name": "Fig_trends", "x_label": "x (mol%)"}
  },
  "tables": {"base_name": "fits", "extra_columns": {"x (mol%)": "samples:x_mol:2"}}
}
```

`samples.csv` needs a `sample` column matching the curve names (or `sample_key_regex` to map them), plus whatever
the blocks refer to, here `composition` (e.g. `50(SiO2)25(Na2O)25(B2O3)`), `density_g_cm3` and `x_mol`.

## 3. Method

### 3.1 Fitting

- **Residuals in ln I.** Each residual is r = (ln I − ln M) / w, with w = √((σ/I)² + f²) and a relative floor
  f = 0.03. Thanks to the floor, a few very precise points cannot dominate, and systematic errors of a few per cent
  are acknowledged.
- **Resolution.** When dQ is given, the model is smeared point by point with a Gaussian of width dQ (Gauss–Hermite
  quadrature, 12 nodes).
- **Model comparison.** BIC = χ² + k ln n. A difference above 10 is "very strong" evidence (Kass & Raftery 1995),
  and that is the threshold used everywhere.
- **Optimiser.** lmfit `least_squares` from a grid of starts:
  - for structural candidates, the 10 starts with the lowest initial cost are optimised;
  - a best start that runs out of evaluations is continued, up to 3 times.

  "Did not converge" therefore means a genuine failure, not a spent budget. Multistart least squares on these models
  is sensitive to last-digit changes of the input, and the continuation makes the minimum found reproducible.
- **At-bound flag.** A parameter is flagged when it lies within 1 % of a bound, measured on a log scale for ranges
  wider than ×50. A zero lower bound counts only below 10⁻⁴ of the upper bound.

### 3.2 Base

| Key | Model | Constraint | Meaning |
|---|---|---|---|
| B0 | B q⁻ᵖ + C | p ∈ [3.7, 4.7], C ≥ 0 | Outer surfaces of grains or particles (Porod) + flat level |
| B1 | B q⁻ᵖ + A / (1 + q²ξ²) | ξ ∈ [1, 5] Å | Grain law + sub-nanometre concentration fluctuations (Ornstein–Zernike) |

The grain law is never interpreted as a structural level. ξ is capped so that the Ornstein–Zernike term cannot absorb
nanometre structure. B1 is preferred only when it beats B0 by ΔBIC > 10 with no parameter at a bound.

### 3.3 Gate: is there a structure to fit?

The gate compares each curve with its reference, without a model:

1. The ratio of the curve to its reference is formed after subtracting each curve's fitted high-q term, and
   normalised to 1 at q ≤ 0.006 Å⁻¹, where the grain law dominates both curves.
2. The curve passes when the ratio exceeds 1 by more than its uncertainty over a contiguous run spanning a factor ≥ 2
   in q inside 0.006–0.1 Å⁻¹, **and** its peak exceeds the null maximum.
3. The null maximum is the largest ratio between any two references (all ordered pairs) in the same window. It is the
   scatter between references themselves, for instance grinding or batch differences of the grain law.

A curve that serves as a reference gets no structural term. A curve without a reference is either offered the
candidates with a flag (`unreferenced: "offer"`, the default) or skipped (`"skip"`).

The base residuals are not used to decide. A run of residuals against the base (`fit_gate_span`) is reported as a
diagnostic only, because it fires on clean, featureless curves: a real grain law is not a single power law at the
2–3σ level.

### 3.4 Structural candidates

Every candidate is fitted on both bases. A key without a suffix is on the preferred base; `@B0` / `@B1` marks the
other one. `+UNI2` adds a second, smaller population. Size bounds follow the measured window [q_min, q_max], because a
length outside it cannot be measured.

| Key | Model | Free parameters (bounds) | Describes |
|---|---|---|---|
| SPH | Schulz-polydisperse spheres | φ (0–0.74 with a contrast; a free amplitude without), R (10 Å – 4/q_min), σ_R/R (0.05–0.7) | Dilute particles. φ is a volume fraction when Δρ is known. |
| SPHPY | SPH × Percus–Yevick hard-sphere S(q) | + φ_HS (0–0.5) | Interacting particles. Must beat SPH by ΔBIC > 10. |
| TS | Teubner–Strey | S, d (2π/q_max – 2π/q_min), ξ (5 Å – 2π/q_min) | Quasi-periodic two-phase medium with spacing d and correlation length ξ |
| DAB | Debye–Anderson–Brumberger | A, ξ (2 Å – 1/q_min) | Random two-phase medium with sharp interfaces |
| UNI | Unified level, Porod prefactor tied to G (Beaucage 1996) | G, Rg (5 Å – 1/q_min), d (2–4.5), pf (0.1–1.5) | A population with a Guinier knee |
| +UNI2 | Second unified level | G, Rg (5–100 Å), d (2–4.5), pf (0.1–1.5) | A second, smaller population |

The sphere ensemble uses generalised Gauss–Laguerre quadrature (Golub–Welsch, 64 nodes), which gives the exact
Schulz moments. Past q R = 2.5 / σ_R it hands over smoothly to the Porod asymptote.

### 3.5 Selection

A structural model is retained only if **all** of the following hold:

- the gate passed;
- the fit converged, and no free parameter is at a bound;
- it beats its base by ΔBIC > 10;
- it is visible: the structural component carries ≥ 10 % of the model over a span ≥ ×2 in q, at q ≤ 0.2 Å⁻¹;
- for spheres, the ensemble Rg lies inside the window 1/q_max … 1/q_min. Outside it, the "spheres" are only a q⁻⁴ law,
  i.e. grain-like.

SPHPY must also beat SPH by ΔBIC > 10. A second population must beat the retained single structure by ΔBIC > 10 and
be visible itself. Among the candidates that pass, the lowest BIC wins. Every rejection is recorded with its reason
(`CurveResult.reasons`).

### 3.6 Derived quantities

All lengths are reported in nm.

- **TS**: d, ξ, d/ξ, and the amphiphilicity factor f_a (a maximum of I(q) requires f_a < 0).
- **Spheres**: number-mean R and D, the volume-weighted D, the volume-weighted Scherrer-equivalent length (comparable
  with an XRD crystallite size) and the ensemble Rg. Also φ (absolute), and φ_alt for an alternative contrast.
- **Invariant of the structural component**: Q = ∫ I q² dq, integrated over 10⁻⁶–100 Å⁻¹. From it:
  - the two-phase product φ(1−φ)Δρ² = Q / (2π²);
  - φ for the given Δρ;
  - Δρ_min, the smallest contrast able to produce Q (reached at φ = ½). Any candidate phase must exceed it.
- **Stoichiometric ceiling φ_max**: the largest volume fraction of the phase that the sample's composition allows,
  with oxygen ignored. A φ above φ_max rules the phase out.

### 3.7 Robustness

The retained model is refitted, starting from its own solution, on perturbed data:

| Variant | Perturbation |
|---|---|
| `stat` | Weights from the statistical σ only (skipped when the curve has none) |
| `qmax` | q ≤ 0.2 Å⁻¹ |
| `noedge` | Last point removed |
| `nosmear` | Unsmeared fit without the flagged low-q points (or above `nosmear_qmin`) |
| `offset` | A known calibration offset applied to tagged curves (`offset_variant`): the factor applies below `q_lo` and ramps to 1 at `q_hi` |

A parameter is robust when every variant keeps it within max(20 %, 2σ) of the main value and off its bounds.
`robust_primary` covers the main structure, and `robust_second` covers the second population.

## 4. Reading the results

Outputs of a project run, under `output_dir`:

| File | Content |
|---|---|
| `fits.json` | The full result: every candidate, every parameter, the gate arrays. Reload it with `WorkflowResult.load` or **Load…** in the Fit tab. |
| `fits_summary.csv` | One row per curve: gate, retained model, χ²_red, ΔBIC, next best model, derived quantities, robustness |
| `fits/<curve>_fit.csv` | q, I, model, normalised residual, one column per component |
| `fits_robustness.csv` | One row per (curve, variant, parameter) |
| `tables/<base>.xlsx`, `.csv`, `.md` | Sheets *summary*, *models* (every candidate with its verdict), *parameters*, *robustness*, *notes* (the method in five lines). The csv files are Excel-safe ASCII. |
| `figures/` | Grid of curves, fit panels (I or I q⁴), trends against your x-value |
| `manifest.json` | Code version, options, inputs, outputs and timings of every step |

How to read them:

- **Sizes and spacings** (Rg, R, d, ξ) are the robust numbers.
- **Amplitudes** become volume fractions only through a contrast you trust. When it is uncertain, report Δρ_min and
  the two-phase product instead.
- **Near-degenerate families.** When `dBIC_next_minus_retained` < 10, the data do not distinguish the retained model
  from the next one. State both; do not choose.
- **At a bound, or not converged**: not a result. The models sheet shows these candidates with their status, and
  they are never retained.
- A **non-robust** parameter (`robustness_failures` names the parameter:variant pairs) is reported with that flag.
  Trend figures draw it faded.

## 5. Options

`WorkflowOptions` (the `fit.options` block of a project file; Options box of the Fit tab). The defaults are the
recommended values.

| Option | Default | Meaning |
|---|---|---|
| `floor` | 0.03 | Relative floor of the weights |
| `n_nodes` | 12 | Gauss–Hermite nodes of the resolution smearing |
| `smear` | true | Use dQ when a curve has it |
| `grain_p_bounds` | [3.7, 4.7] | Exponent range of the grain law |
| `highq` | ["flat", "oz"] | High-q terms offered to the base |
| `oz_xi_bounds` | [1, 5] | ξ range of the Ornstein–Zernike term (Å) |
| `highq_seed_qmin` | 0.3 | Start value of the high-q term: median I above this q (1/Å) |
| `gate` | "reference" | `"reference"`, or `"none"` (structures always offered, flagged) |
| `gate_window` | [0.006, 0.1] | q window of the gate (1/Å) |
| `norm_qmax` | 0.006 | The ratio is normalised at q ≤ this value |
| `gate_span` | 2 | Minimum span of the excess run (factor in q); also used for visibility |
| `gate_sigma` | 1 | Excess threshold in units of the ratio's uncertainty |
| `gate_null_max` | null | Peak threshold. null = largest reference/reference ratio |
| `unreferenced` | "offer" | Curves without a reference: `"offer"` (flagged) or `"skip"` |
| `families` | all five | Any of `sphere`, `sphere_py`, `teubner_strey`, `debye`, `unified` |
| `second_population` | true | Offer +UNI2 |
| `second_rg_bounds` | [5, 100] | Rg range of the second population (Å) |
| `dbic_min` | 10 | ΔBIC threshold of every comparison |
| `visibility_fraction` | 0.10 | Minimum share of the model carried by a structural component |
| `visibility_qmax` | 0.2 | Visibility is assessed at q ≤ this value (1/Å) |
| `prescreen` | 10 | Starts optimised per structural candidate |
| `max_nfev` | 3000 | Evaluation budget per start (4000 for the bases) |
| `polish_rounds` | 3 | Continuations of a best start that ran out of evaluations (0 = off) |
| `report_q` | 0.45 | The high-q term is also reported at this q (1/Å) |
| `robust_qmax` | 0.2 | q limit of the `qmax` variant (1/Å) |
| `robust_tol` | 0.20 | Relative tolerance of the robustness test (or 2σ, whichever is larger) |
| `nosmear_qmin` | null | `nosmear` variant: fit above this q (else drop the flagged points) |
| `offset_variant` | null | `{"factor", "q_lo", "q_hi", "tag", "value"}`: the `offset` variant for curves whose tag matches |

Unknown keys and unknown families are rejected with an error, never ignored.

## 6. Project file reference

| Key | Content |
|---|---|
| `name` | Project name |
| `curves` | Reduced curves: `dir`, `pattern` (glob, default `*.dat`), `name_regex` (group 1 = curve name), `exclude_regex`, optional `q_col` / `i_col` / `sigma_col` / `dq_col`. Without this block the curves come from the reduce step. |
| `frames_roots`, `empty_pattern`, `calibrant_pattern`, `exclude_pattern`, `name_from`, `pairing`, `default_wavelength_A`, `reduction` | Raw-frame projects: inventory, pairing and reduction |
| `samples_table`, `sample_key_regex`, `energy_ev` | csv with `sample` plus `mu_mm`, or `composition` + `density_g_cm3`, plus any columns the blocks below refer to |
| `output_dir` | Where everything is written |
| `fit.options` | `WorkflowOptions` overrides (§5) |
| `fit.references` | `{"explicit": {curve: reference}}`, `{"pattern": regex, "replace": replacement}` (re.sub on the full name) and/or `{"column": samples-table column}`. Explicit pairs win over the rule, which wins over the column. A reference must be one of the curves. |
| `fit.contrast` | `{"phase", "density_g_cm3", "alt_density_g_cm3", "composition_column", "density_column", "ceiling"}`, or `{"column": samples-table column}` in 10¹⁰ cm⁻² |
| `fit.tags` | `{tag: "header:<regex with one group>" \| "samples:<column>"}`, used by the `offset` variant |
| `fit.robustness_variants` | Subset of `stat`, `qmax`, `noedge`, `nosmear`, `offset` |
| `series_table` | csv with sample, panel_row, panel_col, stack_index, label (+ colour, marker, linestyle, x_value, family_label, col_title, quantified, note_symbol, group, extra_panels, fit_order). Needed for the grid and trend figures and for table order. |
| `figures` | `formats`, `dpi`, `style`; `grid` (shift_decades, x_range, q_max, per_decade, ylabel); `fit_panels` (a list: `name` with `{group}`, `samples` or `group` (`"*"` = every group), `only: "structural"`, `ncol`, `iq4`, `x_range`, `title`, `per_figure`); `trends` (`name`, `x_label`, optional `panels`); `frame` |
| `tables` | `base_name`, `formats`, `extra_columns` (`{"title": "samples:<column>[:<decimals>]" \| "series:<column>[:<decimals>]"}`) |

`py -3.11 scripts/saxs_pipeline.py init` writes a template with these keys and a comment block.

## 7. Python API

```python
from saxs.core.loader import load_tabular_curve
from saxs.core.fit_workflow import CurveInput, WorkflowOptions, run_workflow, robustness, summary_table
from saxs.core.contrast import phase_contrast_1e10cm2

ref = CurveInput.from_curve(load_tabular_curve("parent.dat"))
# densities in g/cm3: the phase's from its crystal structure, the sample's measured
drho = phase_contrast_1e10cm2("Al2O3", rho_phase, "50(SiO2)25(Na2O)25(B2O3)", rho_sample)
smp = CurveInput.from_curve(load_tabular_curve("sample.dat"), reference=ref.name, contrast=drho)
result = run_workflow([ref, smp], WorkflowOptions())
robustness(result, [ref, smp])
print(summary_table(result).T)
result.save("fits.json")
```

The modules, from the bottom up:

| Module | Content |
|---|---|
| `saxs.core.models` | Components (flat, power law, Ornstein–Zernike, Guinier, Guinier–Porod, unified levels, DAB, Teubner–Strey, Schulz spheres with Percus–Yevick S(q)), each with parameters, bounds, units and its literature source |
| `saxs.core.resolution` | Gaussian pinhole smearing |
| `saxs.core.contrast` | Electron density, X-ray SLD, phase contrast, stoichiometric ceiling, invariant |
| `saxs.core.physical_fit` | `fit_physical` (multistart, prescreen, continuation, BIC, at-bound flags), residual runs, reference ratio |
| `saxs.core.fit_workflow` | Base, gate, candidates, selection, derived quantities, robustness, tables |
| `saxs.core.pipeline`, `scripts/saxs_pipeline.py` | JSON projects, steps, manifest, command line |
| `saxs.core.series_figures`, `saxs.core.tables` | Figures and xlsx / csv / md tables |

## 8. Limits and good practice

- **Reduction quality bounds everything.** Absolute calibration, transmission, background and the merge of
  configurations decide what the fit can say. Check them first: overlaps between configurations should agree within
  their uncertainty, and the plateau should agree with the blank.
- **Choose references carefully.** A reference must differ from the sample only by the component whose structure is
  being sought. With several references, their mutual scatter sets the gate threshold, so a poor reference raises
  the bar for every curve.
- **Contrast is a hypothesis.** Volume fractions inherit its uncertainty. Use an alternative density
  (`alt_density_g_cm3`), the stoichiometric ceiling and Δρ_min to test it.
- **Degeneracy is a result.** Spheres, Teubner–Strey and a unified level can describe the same knee. When ΔBIC < 10
  between them, report the size scale, not the morphology.
- **Run time** grows with the candidates. A curve that passes the gate gets up to 20 structural fits (5 families ×
  2 bases × with or without a second population), each from 10 starts, which takes minutes. A curve that fails the
  gate takes seconds. `families` and `second_population` shorten exploratory runs.
- **Start grids** are written for a window starting near q ≈ 0.0026 Å⁻¹ and rescaled to the curve's q_min.

## 9. References

- Beaucage, G. (1995). *J. Appl. Cryst.* 28, 717–728 (unified exponential/power-law approach).
- Beaucage, G. (1996). *J. Appl. Cryst.* 29, 134–146 (polydispersity and the Porod prefactor).
- Debye, P., Anderson, H. R. & Brumberger, H. (1957). *J. Appl. Phys.* 28, 679–683.
- Golub, G. H. & Welsch, J. H. (1969). *Math. Comp.* 23, 221–230 (Gauss quadrature rules).
- Kass, R. E. & Raftery, A. E. (1995). *J. Am. Stat. Assoc.* 90, 773–795 (Bayes factors; the BIC scale).
- Kinning, D. J. & Thomas, E. L. (1984). *Macromolecules* 17, 1712–1718 (Percus–Yevick S(q) in SAXS).
- Kotlarchyk, M. & Chen, S.-H. (1983). *J. Chem. Phys.* 79, 2461–2469 (Schulz-polydisperse spheres).
- Ornstein, L. S. & Zernike, F. (1914). *Proc. K. Ned. Akad. Wet.* 17, 793–806.
- Porod, G. (1951). *Kolloid-Z.* 124, 83–114.
- Schwarz, G. (1978). *Ann. Stat.* 6, 461–464 (BIC).
- Teubner, M. & Strey, R. (1987). *J. Chem. Phys.* 87, 3195–3200.
