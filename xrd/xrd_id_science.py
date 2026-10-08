"""
xrd_id_science.py — XRD phase identification (framework-agnostic): a
QualX-style search-match engine over the user's OWN reference databases.

PRISM ships no reference data. Users download whichever card database they
have the rights to use — any QualX-format .sq works — and register it in
the XRD ID workspace ("Add database…" / "Add folder…"). Registered
databases live in a small JSON registry (~/.raman_cache/xrd_id/
databases.json); any number can be enabled at once and every search probes
all enabled ones, so results can mix several databases exactly like QualX.

QualX .sq format (reverse-engineered): an `id` table with one row per
card — name / mineralname / chemical_formula / spacegroup / quality / rir
and the reference pattern as comma-separated text in `dvalue` (d-spacings,
Å) and `intensita` (intensities) — plus a `chemical` table (card id →
element) and `infodb` (provenance). Registering such a file converts it
ONCE into PRISM's indexed format below (stored under ~/.raman_cache/
xrd_id/imported/); a .sq that is already in PRISM's format (e.g. handed
over by a colleague) registers in place, no copy.

PRISM indexed .sq schema:
  cards(card_id, source, source_code, name, mineral, formula, spacegroup,
        quality, rir, nd, d, i)          — lines capped at the strongest 60
  elements(card_id, element)             — for chemistry filters
  strong_lines(card_id, d, i)            — top-3 lines, d-indexed, the
                                           search-match prefilter
  sources(tag, date, ncard, origin)      — provenance of each merged input

A folder of CIFs registers as a database too: each structure's powder
pattern is CALCULATED (xrd.cif_patterns — structure factors, so systematic
absences are absent) and written into the same indexed schema, plus a
cif_files table mapping each card back to the file it came from.

Search-match: query peaks (2θ, I) at wavelength λ → d-spacings; candidate
cards are prefiltered by strong-line coincidence, then scored by the
geometric mean of intensity-weighted coverage in both directions (how much
of the card's pattern the measurement explains, and vice versa), each line
weighted by how closely it lands inside the tolerance window, with an
optional bounded global 2θ offset for sample displacement — never an
automatic answer, always a ranked list, same philosophy as the RRUFF
match-assist.

Also here: an XRD-specific peak finder (SNIP background, noise-referenced
detection limit, sub-step centers, Kα2 identification) and structure
export for the accepted phases, which is what a Rietveld refinement needs
and what a d–I card database cannot itself provide.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

XRD_ID_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".raman_cache", "xrd_id")
XRD_ID_DB_PATH = os.path.join(XRD_ID_CACHE_DIR, "xrdid.sq")  # legacy single-db location
XRD_ID_REGISTRY_PATH = os.path.join(XRD_ID_CACHE_DIR, "databases.json")
XRD_ID_IMPORT_DIR = os.path.join(XRD_ID_CACHE_DIR, "imported")

CU_KA1 = 1.5406  # Å — the default lab wavelength

MAX_LINES_PER_CARD = 60  # strongest lines kept; plenty for identification
N_STRONG = 3             # lines per card in the prefilter index


# =============================================================================
# d-spacing / 2θ conversions
# =============================================================================

def two_theta_to_d(two_theta_deg: np.ndarray, wavelength: float = CU_KA1) -> np.ndarray:
    tt = np.asarray(two_theta_deg, float)
    s = np.sin(np.radians(tt / 2.0))
    return np.where(s > 1e-9, wavelength / (2.0 * np.where(s > 1e-9, s, 1.0)), np.inf)


def d_to_two_theta(d: np.ndarray, wavelength: float = CU_KA1) -> np.ndarray:
    d = np.asarray(d, float)
    arg = wavelength / (2.0 * np.where(d > 1e-9, d, np.inf))
    out = np.full_like(d, np.nan)
    ok = (arg > 0) & (arg <= 1.0)
    out[ok] = 2.0 * np.degrees(np.arcsin(arg[ok]))
    return out


# =============================================================================
# Peak detection on a measured diffractogram
#
# The generic 2nd-derivative finder (core.fitting_science.find_peak_candidates)
# was written for Raman and treats a diffractogram's background as signal: on
# a glass + crystallites pattern the amorphous halo dominates the curvature
# budget, so weak Bragg peaks fall under the σ threshold while the halo's own
# shoulders come back as "peaks". This finder is XRD-specific:
#
#   1. SNIP background clipping (the standard XRD/XRF background estimator)
#      removes the halo and the instrumental background FIRST;
#   2. the detection threshold is taken from the noise of the background-
#      SUBTRACTED residual, so "3σ" means the same thing at 15° (on top of a
#      big halo) and at 75° (on a flat, low background);
#   3. centers are refined by parabolic interpolation of the smoothed apex —
#      a 10–80° scan in 4000 points steps 0.0175°, an order of magnitude
#      coarser than the tolerance a search-match can exploit;
#   4. Kα2 satellites are identified geometrically (Bragg's law at λ2) and
#      flagged rather than reported as independent reflections. An
#      unstripped Kα1/Kα2 doublet is the single most common source of
#      spurious extra lines in a hand-picked lab-XRD peak list;
#   5. intensities come back with the positions, background-subtracted, so
#      the search-match ranks on real relative intensities instead of raw
#      heights that still carry the background under each peak.
# =============================================================================

# Kα1 / Kα2 emission wavelengths (Å) of the common sealed-tube anodes.
# Values: Hölzer et al., Phys. Rev. A 56 (1997) 4554 (Cr–Cu) and the
# International Tables Vol. C table 4.2.2.1 (Mo, Ag).
ANODE_KALPHA: Dict[str, Tuple[float, float]] = {
    "Cr": (2.289700, 2.293606),
    "Fe": (1.936042, 1.939980),
    "Co": (1.788965, 1.792850),
    "Cu": (1.540598, 1.544426),
    "Mo": (0.709300, 0.713590),
    "Ag": (0.559408, 0.563813),
}


def ka2_ratio_for(wavelength: float) -> float:
    """λ(Kα2)/λ(Kα1) of the anode whose Kα1 is closest to `wavelength`
    (within 0.02 Å; Cu's ratio otherwise). The doublet separation in °2θ
    depends only on this ratio, so it is all the stripping needs."""
    best, best_gap = "Cu", float("inf")
    for anode, (ka1, _ka2) in ANODE_KALPHA.items():
        gap = abs(float(wavelength) - ka1)
        if gap < best_gap:
            best, best_gap = anode, gap
    if best_gap > 0.02:
        best = "Cu"
    ka1, ka2 = ANODE_KALPHA[best]
    return ka2 / ka1


def ka2_position(two_theta_ka1, ratio: float) -> np.ndarray:
    """Where the Kα2 satellite of a Kα1 reflection sits, in °2θ: same d,
    longer wavelength → sin θ scaled by λ2/λ1 (NaN past back-reflection)."""
    s = np.sin(np.radians(np.asarray(two_theta_ka1, float) / 2.0)) * float(ratio)
    out = np.full(np.shape(s), np.nan, float)
    ok = np.abs(s) <= 1.0
    out[ok] = 2.0 * np.degrees(np.arcsin(s[ok]))
    return out


def snip_background(y: np.ndarray, width_points: int, *, decreasing: bool = True) -> np.ndarray:
    """SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping,
    Ryan et al., NIM B 34 (1988) 396) — the background estimator used by
    essentially every diffraction/fluorescence package.

    The signal is put through the LLS transform (which compresses the
    dynamic range so a strong peak cannot drag the background up), then
    each point is repeatedly replaced by the smaller of itself and the mean
    of the two points `p` channels away. Anything narrower than the widest
    window is clipped away; anything broader survives as background. The
    decreasing-window variant (p from width down to 1) is used because it
    preserves narrow Bragg peaks markedly better than the increasing one.

    width_points sets BOTH the window and the iteration count, so it is an
    aggressiveness dial rather than a hard cutoff: measured against a
    known halo, a window of ~2x the Bragg FWHM leaves 97% of a 9.4°-wide
    amorphous hump standing while recovering sharp peaks in full, and one
    of 2° erodes that same hump to 1% of itself (which silently moves the
    halo's counts into the peak intensities). estimate_background_width()
    picks it from the pattern's own peak widths; the caller converts °2θ.
    """
    y = np.asarray(y, float)
    if len(y) < 3:
        return np.zeros_like(y)
    floor = float(np.nanmin(y))
    v = np.clip(y - floor, 0.0, None)
    v = np.log(np.log(np.sqrt(v + 1.0) + 1.0) + 1.0)  # LLS
    m = int(max(1, min(width_points, (len(y) - 1) // 2)))
    for p in (range(m, 0, -1) if decreasing else range(1, m + 1)):
        left = np.concatenate([np.full(p, v[0]), v[:-p]])
        right = np.concatenate([v[p:], np.full(p, v[-1])])
        v = np.minimum(v, 0.5 * (left + right))
    back = (np.exp(np.exp(v) - 1.0) - 1.0) ** 2 - 1.0  # inverse LLS
    return np.clip(back, 0.0, None) + floor


def _noise_sigma(residual: np.ndarray) -> float:
    """Point-to-point noise of a background-subtracted pattern, from the
    MAD of its second difference: robust to the peaks themselves (a few
    percent of the points) in a way a plain std is not. The √6 divides out
    the variance a second difference of white noise adds."""
    r = np.asarray(residual, float)
    d2 = np.diff(r, n=2) if len(r) > 3 else np.array([])
    if len(d2):
        mad = float(np.median(np.abs(d2 - np.median(d2))))
        sigma = 1.4826 * mad / np.sqrt(6.0)
        if np.isfinite(sigma) and sigma > 0:
            return sigma
    s = float(np.std(r))
    return s if s > 0 else 1.0


def local_noise_sigma(residual: np.ndarray, window_points: int) -> np.ndarray:
    """Point-by-point noise, same definition as _noise_sigma but measured
    in a sliding window instead of over the whole pattern.

    Counting noise is signal-dependent: σ = √counts. On a diffractogram
    whose background runs from a couple of hundred counts at high angle to
    a couple of thousand under an amorphous halo, the real noise varies by
    the square root of that range — a factor of three or four. A single
    global σ is then simultaneously too strict where the background is low
    (weak high-angle reflections are thrown away) and far too permissive
    where it is high, which is how a halo comes back as dozens of
    "reflections" a few counts tall. Measured on simulated glass-ceramic
    patterns, a global σ produced ~28 spurious peaks per pattern; the same
    threshold against a local σ produces almost none, with no loss of
    recall.

    Returned per input point, so it can be handed straight to
    scipy.signal.find_peaks as a prominence array.
    """
    r = np.asarray(residual, float)
    n = len(r)
    if n < 8:
        return np.full(max(n, 1), _noise_sigma(r))
    d2 = np.abs(np.diff(r, n=2))
    # centre the second difference back on its middle sample
    d2 = np.concatenate([d2[:1], d2, d2[-1:]])[:n]
    win = int(max(9, window_points))
    if win >= n:
        return np.full(n, _noise_sigma(r))

    # Block medians, linearly interpolated between block centres, rather
    # than a sliding median filter. The noise envelope varies smoothly
    # (it follows √background), so nothing is lost by sampling it every
    # half-window; and every block median is taken over real samples
    # only, so no edge padding rule can put a replicated value in the
    # window. That matters: padding the ends by repetition made the
    # estimate at the last point of a scan a fraction of the truth.
    half = max(4, win // 2)
    centres, values = [], []
    for start in range(0, n, half):
        stop = min(start + win, n)
        if stop - start < 4:
            if centres:
                break
            stop = n
        block = d2[start:stop]
        centres.append(0.5 * (start + stop - 1))
        values.append(float(np.median(block)))
    if len(centres) == 1:
        med = np.full(n, values[0])
    else:
        med = np.interp(np.arange(n, dtype=float), np.asarray(centres, float),
                        np.asarray(values, float))
    sigma = 1.4826 * med / np.sqrt(6.0)
    floor = 1e-9 * float(np.nanmax(np.abs(r)) or 1.0)
    globally = _noise_sigma(r)
    # a window that lands entirely inside a wide peak sees no noise at all;
    # never let the threshold collapse below a small fraction of the
    # pattern's overall noise
    return np.clip(sigma, max(floor, 0.1 * globally), None)


def _odd_window(n_points: int, total: int) -> int:
    w = int(max(5, n_points)) | 1
    return min(w, total - (1 - total % 2))


BACKGROUND_WIDTH_LIMITS = (0.25, 3.0)  # °2θ — the sane range for the SNIP window


def _local_half_width(smooth: np.ndarray, i: int, step: float,
                      spans: Sequence[float] = (1.0, 2.0, 4.0, 8.0)) -> Optional[float]:
    """FWHM of the peak at index i, measured against a LOCAL floor.

    scipy's peak_widths measures at half the peak's prominence, and
    prominence is global: a sharp reflection sitting on an amorphous hump
    has the whole hump as its prominence, so its "width at half
    prominence" is the hump's width, not the peak's — off by a factor of
    fifty on exactly the patterns that need the estimate most. Here the
    floor is the higher of the two valley floors inside a bounded window,
    and the window grows until the peak fits comfortably inside it, so
    both a 0.1° reflection and a 1.5° nanocrystalline one are measured on
    their own terms. None when the peak fits no span.
    """
    n = len(smooth)
    for span in spans:
        half_span = max(2, int(round(span / step)))
        lo_i, hi_i = max(0, i - half_span), min(n, i + half_span + 1)
        if i - lo_i < 2 or hi_i - i < 3:
            continue
        floor = max(float(smooth[lo_i:i].min()), float(smooth[i + 1:hi_i].min()))
        height = float(smooth[i]) - floor
        if height <= 0:
            continue
        half = floor + 0.5 * height
        left = i
        while left > lo_i and smooth[left] > half:
            left -= 1
        right = i
        while right < hi_i - 1 and smooth[right] > half:
            right += 1
        if smooth[left] > half or smooth[right] > half:
            continue  # never came back down inside this span — widen it
        width = (right - left) * step
        if width <= 0.6 * span:
            return width
    return None


def estimate_background_width(x: Sequence[float], y: Sequence[float], *,
                              factor: float = 2.0) -> float:
    """Pick the SNIP window from the pattern's own peaks, in °2θ.

    A fixed window cannot serve both a well-crystallized powder (Bragg
    FWHM ~0.1°, where a wide window would swallow the amorphous halo into
    the peak intensities) and a nanocrystalline or badly-strained sample
    (FWHM >1°, where a narrow window would clip the reflections away as
    background). So the widths are measured first — on the RAW pattern,
    before any background is removed, since the estimate is what decides
    how to remove it — and the window is set to a small multiple of the
    median, clamped to a sane range.
    """
    x = np.asarray(list(x), float)
    y = np.asarray(list(y), float)
    lo, hi = BACKGROUND_WIDTH_LIMITS
    if len(x) < 7:
        return lo
    step = float(np.median(np.diff(x)))
    if not np.isfinite(step) or step <= 0:
        return lo
    try:
        from scipy.signal import find_peaks, savgol_filter
    except ImportError:  # pragma: no cover - scipy is a hard requirement
        return lo
    win = _odd_window(int(round(0.10 / step)), len(y))
    smooth = savgol_filter(y, window_length=win, polyorder=2, mode="interp") if win >= 5 else y
    sigma = _noise_sigma(y - smooth)
    span = float(np.nanmax(smooth) - np.nanmin(smooth)) or 1.0
    idx, _ = find_peaks(smooth, prominence=max(3.0 * sigma, 1e-3 * span))
    widths = [w for w in (_local_half_width(smooth, int(i), step) for i in idx)
              if w is not None and w > 0]
    if not widths:
        return lo
    return float(np.clip(factor * float(np.median(widths)), lo, hi))


@dataclass
class XrdPeak:
    """A detected reflection. `intensity` is the background-subtracted
    height — what the search-match should weight on."""
    two_theta: float
    intensity: float
    fwhm: float
    snr: float
    is_ka2: bool = False
    parent_two_theta: Optional[float] = None  # the Kα1 line a satellite belongs to


@dataclass
class PeakSearchOptions:
    """Every knob of find_xrd_peaks in one place, so a UI can carry the
    whole search setup around, store it and restore it without growing a
    dozen parallel fields. Defaults are find_xrd_peaks' own."""
    min_prominence_sigma: float = 4.0
    subtract_background: bool = True
    background_width: Optional[float] = None   # None = estimate per pattern
    local_noise: bool = True
    noise_window: float = 3.0
    smooth_width: float = 0.10
    min_separation: float = 0.05
    flag_ka2: bool = True
    ka2_tol: float = 0.03
    ka2_max_ratio: float = 0.85
    max_peaks: int = 40
    edge_margin: float = 0.0
    two_theta_min: Optional[float] = None
    two_theta_max: Optional[float] = None

    def as_kwargs(self) -> Dict[str, Any]:
        rng = None
        if self.two_theta_min is not None or self.two_theta_max is not None:
            rng = (self.two_theta_min if self.two_theta_min is not None else -1e9,
                   self.two_theta_max if self.two_theta_max is not None else 1e9)
        return {
            "min_prominence_sigma": self.min_prominence_sigma,
            "subtract_background": self.subtract_background,
            "background_width": self.background_width,
            "local_noise": self.local_noise,
            "noise_window": self.noise_window,
            "smooth_width": self.smooth_width,
            "min_separation": self.min_separation,
            "flag_ka2": self.flag_ka2,
            "ka2_tol": self.ka2_tol,
            "ka2_max_ratio": self.ka2_max_ratio,
            "max_peaks": self.max_peaks,
            "edge_margin": self.edge_margin,
            "two_theta_range": rng,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "PeakSearchOptions":
        """Tolerant of missing, extra and unparseable entries — a stored
        setup from an older version must never stop the page loading."""
        opts = cls()
        for field_name, spec in cls.__dataclass_fields__.items():
            if not data or field_name not in data:
                continue
            raw = data[field_name]
            current = getattr(opts, field_name)
            try:
                if isinstance(current, bool) or field_name in (
                        "subtract_background", "local_noise", "flag_ka2"):
                    setattr(opts, field_name, str(raw).strip().lower()
                            not in ("false", "0", "none", ""))
                elif field_name == "max_peaks":
                    setattr(opts, field_name, int(float(raw)))
                elif raw is None or str(raw).strip() == "":
                    setattr(opts, field_name, None if current is None else current)
                else:
                    setattr(opts, field_name, float(raw))
            except (TypeError, ValueError):
                continue
        return opts


@dataclass
class PeakSearchTrace:
    """What the finder saw, for a caller that wants to show its working:
    the estimated background, the background-subtracted pattern, and the
    detection threshold actually applied at every point."""
    x: np.ndarray
    background: np.ndarray
    corrected: np.ndarray
    smooth: np.ndarray
    sigma: np.ndarray
    threshold: np.ndarray
    background_width: float


def _find_peaks_impl(
    x: Sequence[float], y: Sequence[float], *, wavelength: float = CU_KA1,
    min_prominence_sigma: float = 4.0, subtract_background: bool = True,
    background_width: Optional[float] = None, smooth_width: float = 0.10,
    min_separation: float = 0.05, flag_ka2: bool = True, ka2_tol: float = 0.03,
    ka2_max_ratio: float = 0.85, max_peaks: int = 40, edge_margin: float = 0.0,
    local_noise: bool = True, noise_window: float = 3.0,
    two_theta_range: Optional[Tuple[float, float]] = None,
) -> Tuple[List[XrdPeak], Optional[PeakSearchTrace]]:
    """Detect Bragg reflections in a measured diffractogram.

    All widths are in °2θ and converted to points with the pattern's own
    step, so the same settings work on a fast 0.05°-step survey and a slow
    0.005°-step scan. `min_prominence_sigma` is the detection limit in
    units of the background-subtracted noise; `background_width` defaults
    to estimate_background_width() on this pattern. Kα2 satellites are
    returned flagged (`is_ka2`), never silently deleted — the caller
    decides. Returns peaks ordered by 2θ; `max_peaks` keeps the strongest
    Kα1 lines.

    local_noise / noise_window: measure the detection threshold in a
    sliding window (see local_noise_sigma) instead of once for the whole
    pattern. On anything with a varying background — every glass or
    glass-ceramic — this is the difference between a clean peak list and
    one padded with counting noise off the top of the amorphous halo.
    two_theta_range: restrict the search to a sub-range of the scan.
    """
    x = np.asarray(list(x), float)
    y = np.asarray(list(y), float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 7:
        return [], None
    order = np.argsort(x, kind="mergesort")
    x, y = x[order], y[order]
    step = float(np.median(np.diff(x)))
    if not np.isfinite(step) or step <= 0:
        return [], None

    if subtract_background:
        if background_width is None:
            background_width = estimate_background_width(x, y)
        corrected = y - snip_background(y, int(round(background_width / step)))
    else:
        corrected = y - float(np.nanmin(y))

    try:
        from scipy.signal import find_peaks, peak_widths, savgol_filter
    except ImportError:  # pragma: no cover - scipy is a hard requirement
        find_peaks = peak_widths = savgol_filter = None

    if savgol_filter is not None and smooth_width > 0:
        win = _odd_window(int(round(smooth_width / step)), len(corrected))
        smooth = (savgol_filter(corrected, window_length=win, polyorder=2, mode="interp")
                  if win >= 5 else corrected)
    else:
        smooth = corrected

    # Noise is measured on the CORRECTED pattern, not on (corrected −
    # smooth). Both _noise_sigma and local_noise_sigma take a second
    # difference and divide by √6, which is the right factor for raw white
    # noise; feeding them a residual the smoother has already flattened
    # applies the smoothing twice and underestimates σ — enough, measured,
    # to let a few dozen noise ripples per pattern past a "3σ" threshold.
    # The median inside the estimators is what makes the peaks themselves
    # harmless here.
    scale_floor = 1e-9 * float(np.nanmax(np.abs(corrected)) or 1.0)
    if local_noise and noise_window > 0:
        sigma_arr = np.clip(local_noise_sigma(corrected, int(round(noise_window / step))),
                            scale_floor, None)
    else:
        sigma_arr = np.full(len(corrected), max(_noise_sigma(corrected), scale_floor))
    threshold = float(min_prominence_sigma) * sigma_arr
    trace = PeakSearchTrace(
        x=x, background=y - corrected, corrected=corrected, smooth=smooth,
        sigma=sigma_arr, threshold=threshold,
        background_width=float(background_width or 0.0))
    distance = max(1, int(round(min_separation / step)))

    if find_peaks is None:  # pragma: no cover
        idx = np.where((smooth[1:-1] > smooth[:-2]) & (smooth[1:-1] >= smooth[2:])
                       & (smooth[1:-1] > threshold[1:-1]))[0] + 1
        widths = np.full(len(idx), 2.0 * distance, float)
    else:
        idx, _props = find_peaks(smooth, prominence=threshold, distance=distance)
        widths = peak_widths(smooth, idx, rel_height=0.5)[0] if len(idx) else np.array([])

    # The window is applied to the DETECTED peaks, not to the data: the
    # background and the noise are still estimated on the whole scan, so
    # narrowing the window never changes where the remaining peaks sit or
    # how strong they look.
    keep = np.ones(len(idx), bool)
    if edge_margin > 0:
        keep &= (x[idx] >= x[0] + edge_margin) & (x[idx] <= x[-1] - edge_margin)
    if two_theta_range is not None:
        lo, hi = float(two_theta_range[0]), float(two_theta_range[1])
        keep &= (x[idx] >= min(lo, hi)) & (x[idx] <= max(lo, hi))
    if not np.all(keep):
        idx, widths = idx[keep], widths[keep]
    if len(idx) == 0:
        return [], None

    peaks: List[XrdPeak] = []
    for k, i in enumerate(idx):
        center = _parabolic_center(x, smooth, int(i))
        height = float(smooth[i])
        sigma_here = float(sigma_arr[int(i)])
        peaks.append(XrdPeak(two_theta=center, intensity=max(height, 0.0),
                             fwhm=float(widths[k]) * step if k < len(widths) else 0.0,
                             snr=height / sigma_here if sigma_here > 0 else 0.0))

    if flag_ka2:
        _flag_ka2_satellites(peaks, ratio=ka2_ratio_for(wavelength),
                             tol=ka2_tol, max_ratio=ka2_max_ratio)

    # Cap on the Kα1 lines only; a satellite rides with its parent so the
    # caller can still report "n doublets stripped" honestly.
    primary = [p for p in peaks if not p.is_ka2]
    if max_peaks and len(primary) > max_peaks:
        keep = {id(p) for p in sorted(primary, key=lambda p: p.intensity, reverse=True)[:max_peaks]}
        parents = {p.two_theta for p in primary if id(p) in keep}
        peaks = [p for p in peaks
                 if id(p) in keep or (p.is_ka2 and p.parent_two_theta in parents)]
    peaks.sort(key=lambda p: p.two_theta)
    return peaks, trace


def find_xrd_peaks(*args, **kwargs) -> List[XrdPeak]:
    """Detect Bragg reflections in a measured diffractogram — see
    _find_peaks_impl for the full parameter list."""
    return _find_peaks_impl(*args, **kwargs)[0]


def find_xrd_peaks_traced(*args, **kwargs) -> Tuple[List[XrdPeak], Optional[PeakSearchTrace]]:
    """find_xrd_peaks, plus what it saw: the background it removed, the
    noise it measured and the threshold it applied at every point. A UI
    can draw those over the pattern so that "why was this peak not
    found?" is answerable by looking instead of by guessing."""
    return _find_peaks_impl(*args, **kwargs)


def _parabolic_center(x: np.ndarray, y: np.ndarray, i: int) -> float:
    """Sub-step apex position from the three points around the maximum —
    exact for a parabola, and a diffraction peak's top is parabolic to
    first order."""
    if not (0 < i < len(y) - 1):
        return float(x[i])
    y0, y1, y2 = float(y[i - 1]), float(y[i]), float(y[i + 1])
    denom = y0 - 2.0 * y1 + y2
    if denom == 0:
        return float(x[i])
    delta = 0.5 * (y0 - y2) / denom
    if abs(delta) > 1.0:
        return float(x[i])
    return float(x[i]) + delta * float(x[i + 1] - x[i - 1]) / 2.0


def _flag_ka2_satellites(peaks: List[XrdPeak], *, ratio: float, tol: float,
                         max_ratio: float) -> None:
    """Mark every peak that sits where the Kα2 satellite of a STRONGER peak
    must be. Strongest-first so a satellite is never taken for a parent,
    and a satellite may not exceed `max_ratio` of its parent (the physical
    ratio is ≈0.5; the bound only rejects accidental coincidences with a
    genuine reflection of comparable strength)."""
    by_strength = sorted(range(len(peaks)), key=lambda k: peaks[k].intensity, reverse=True)
    for k in by_strength:
        parent = peaks[k]
        if parent.is_ka2:
            continue
        expected = float(ka2_position([parent.two_theta], ratio)[0])
        if not np.isfinite(expected):
            continue
        for m, cand in enumerate(peaks):
            if m == k or cand.is_ka2 or cand.intensity > max_ratio * parent.intensity:
                continue
            if abs(cand.two_theta - expected) <= tol:
                cand.is_ka2 = True
                cand.parent_two_theta = parent.two_theta


def peak_intensities_at(
    x: Sequence[float], y: Sequence[float], positions: Sequence[float], *,
    subtract_background: bool = True, background_width: Optional[float] = None,
    window: float = 0.15,
) -> List[float]:
    """Background-subtracted heights at hand-typed / edited peak positions,
    so an edited peak list still carries meaningful relative intensities.
    Falls back to 1.0 for positions outside the measured range."""
    x = np.asarray(list(x), float)
    y = np.asarray(list(y), float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return [1.0] * len(positions)
    order = np.argsort(x, kind="mergesort")
    x, y = x[order], y[order]
    step = float(np.median(np.diff(x))) or 1.0
    if subtract_background:
        if background_width is None:
            background_width = estimate_background_width(x, y)
        corrected = y - snip_background(y, int(round(background_width / step)))
    else:
        corrected = y - float(np.nanmin(y))
    half = max(float(window), 3.0 * step)
    out: List[float] = []
    for p in positions:
        near = np.abs(x - float(p)) <= half
        out.append(float(np.nanmax(corrected[near])) if np.any(near) else 1.0)
    # a flat-zero list would make every card score identically
    return [v if v > 0 else 1.0 for v in out] if any(v > 0 for v in out) else [1.0] * len(out)


# =============================================================================
# Database build (one-time merge of the QualX-format sources)
# =============================================================================

def _parse_lines_text(d_text, i_text) -> Tuple[np.ndarray, np.ndarray]:
    """The dvalue/intensita columns are comma-separated float text (some
    databases declare them BLOB but store text all the same)."""
    def _floats(t):
        if t is None:
            return np.array([])
        if isinstance(t, bytes):
            t = t.decode("ascii", errors="replace")
        vals = []
        for tok in str(t).split(","):
            tok = tok.strip()
            if tok:
                try:
                    vals.append(float(tok))
                except ValueError:
                    pass
        return np.array(vals, float)

    d = _floats(d_text)
    i = _floats(i_text)
    n = min(len(d), len(i))
    return d[:n], i[:n]


def _safe_float(v) -> Optional[float]:
    """Some databases store empty numerics as runs of spaces — not None, not ''."""
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def _clean_name(name: Optional[str]) -> str:
    """Some databases carry $-prefixed typesetting codes in names ('$GB-...')."""
    s = (name or "").strip()
    s = re.sub(r"\$[A-Z0-9]{1,3}[- ]?", "", s)
    return s.strip()


def build_xrd_database(
    sources: Sequence[Tuple[str, str]], *, out_path: str = XRD_ID_DB_PATH,
    min_lines: int = 3, progress_every: int = 50000, log=print,
) -> Dict[str, int]:
    """Convert (and optionally merge) QualX-format .sq files into one
    PRISM-indexed database. sources: [(path, tag), ...] — the tag labels
    every card's provenance in results. Cards keep their original id as
    source_code. Cards with fewer than min_lines usable lines are skipped
    (nothing to match against). Returns {tag: n_cards_ingested}."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)
    out = sqlite3.connect(out_path)
    out.executescript("""
        CREATE TABLE cards (
            card_id INTEGER PRIMARY KEY,
            source TEXT NOT NULL, source_code TEXT NOT NULL,
            name TEXT, mineral TEXT, formula TEXT, spacegroup TEXT,
            quality TEXT, rir REAL, nd INTEGER, d TEXT, i TEXT
        );
        CREATE TABLE elements (card_id INTEGER NOT NULL, element TEXT NOT NULL);
        CREATE TABLE strong_lines (card_id INTEGER NOT NULL, d REAL NOT NULL, i REAL NOT NULL);
        CREATE TABLE sources (tag TEXT, date TEXT, ncard INTEGER, origin TEXT);
    """)

    counts: Dict[str, int] = {}
    for path, tag in sources:
        src = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            for row in src.execute("SELECT * FROM infodb"):
                out.execute("INSERT INTO sources VALUES (?,?,?,?)", (tag, str(row[1]), int(row[2]), str(row[-1])))
        except sqlite3.Error:
            out.execute("INSERT INTO sources VALUES (?,?,?,?)", (tag, "", 0, ""))

        n_in = 0
        cur = src.execute(
            "SELECT id, name, mineralname, chemical_formula, spacegroup, quality, rir, dvalue, intensita FROM id"
        )
        rows_cards, rows_strong = [], []
        for k, (cid, name, mineral, formula, sg, quality, rir, d_text, i_text) in enumerate(cur):
            if progress_every and k and k % progress_every == 0:
                log(f"  [{tag}] scanned {k} cards, kept {n_in}…")
            d, i = _parse_lines_text(d_text, i_text)
            if len(d) < min_lines:
                continue
            imax = float(i.max()) if len(i) else 0.0
            if imax <= 0:
                continue
            i = i / imax * 100.0
            order = np.argsort(i)[::-1][:MAX_LINES_PER_CARD]
            d, i = d[order], i[order]
            rows_cards.append((
                tag, str(cid), _clean_name(name), _clean_name(mineral),
                (formula or "").strip().strip('"'), (sg or "").strip(),
                (quality or "").strip(), _safe_float(rir),
                len(d), ",".join(f"{v:.5f}" for v in d), ",".join(f"{v:.2f}" for v in i),
            ))
            n_in += 1
        out.executemany(
            "INSERT INTO cards (source, source_code, name, mineral, formula, spacegroup, quality, rir, nd, d, i) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows_cards)
        # strong lines need the new card_id: fetch back in insert order
        first_id = out.execute("SELECT MAX(card_id) FROM cards").fetchone()[0] - len(rows_cards) + 1
        for offset, rec in enumerate(rows_cards):
            d = [float(v) for v in rec[9].split(",")[:N_STRONG]]
            i = [float(v) for v in rec[10].split(",")[:N_STRONG]]
            for dv, iv in zip(d, i):
                rows_strong.append((first_id + offset, dv, iv))
        out.executemany("INSERT INTO strong_lines VALUES (?,?,?)", rows_strong)

        # elements (chemistry filter): only for cards we kept
        code_to_id = {rec[1]: first_id + off for off, rec in enumerate(rows_cards)}
        rows_el = []
        try:
            for cid, el in src.execute("SELECT id, chemical_element FROM chemical"):
                card_id = code_to_id.get(str(cid))
                if card_id is not None and el:
                    rows_el.append((card_id, str(el).strip().capitalize()))
        except sqlite3.Error:
            pass
        out.executemany("INSERT INTO elements VALUES (?,?)", rows_el)
        out.commit()
        counts[tag] = n_in
        log(f"[{tag}] ingested {n_in} cards ({len(rows_el)} element links)")
        src.close()

    log("indexing…")
    out.executescript("""
        CREATE INDEX idx_strong_d ON strong_lines(d);
        CREATE INDEX idx_elements ON elements(element, card_id);
        CREATE INDEX idx_cards_mineral ON cards(mineral);
        CREATE INDEX idx_cards_formula ON cards(formula);
    """)
    out.commit()
    out.close()
    return counts


def database_summary(db_path: str = XRD_ID_DB_PATH) -> Optional[Dict[str, Any]]:
    if not os.path.isfile(db_path):
        return None
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    by_source = dict(con.execute("SELECT source, COUNT(*) FROM cards GROUP BY source"))
    total = sum(by_source.values())
    qualities = sorted({(q or "").strip().upper()
                        for (q,) in con.execute("SELECT DISTINCT quality FROM cards")
                        if (q or "").strip()})
    con.close()
    return {"total_cards": total, "by_source": by_source, "qualities": qualities, "path": db_path}


# =============================================================================
# Database registry — use whatever databases you want, several at once
# =============================================================================

def sniff_sq_format(path: str) -> Optional[str]:
    """'prism' (PRISM's indexed schema), 'qualx' (original QualX schema),
    or None (not a recognizable card database)."""
    if not os.path.isfile(path):
        return None
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        tables = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')")}
        if {"cards", "strong_lines"} <= tables:
            con.close()
            return "prism"
        if "id" in tables:
            cols = {r[1] for r in con.execute("PRAGMA table_info(id)")}
            con.close()
            if {"dvalue", "intensita"} <= cols:
                return "qualx"
            return None
        con.close()
    except sqlite3.Error:
        return None
    return None


def load_registry(registry_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """The registered databases: [{'name', 'path', 'enabled', 'origin'?}, …].
    First call migrates a pre-registry unified database (the old fixed
    ~/.raman_cache/xrd_id/xrdid.sq location) into the registry so existing
    setups keep working untouched."""
    registry_path = registry_path or XRD_ID_REGISTRY_PATH
    entries: List[Dict[str, Any]] = []
    if os.path.isfile(registry_path):
        try:
            with open(registry_path, encoding="utf-8") as f:
                entries = list(json.load(f).get("databases", []))
        except (OSError, ValueError):
            entries = []
    if not any(os.path.normcase(e.get("path", "")) == os.path.normcase(XRD_ID_DB_PATH)
               for e in entries) and os.path.isfile(XRD_ID_DB_PATH) \
            and os.path.normcase(registry_path) == os.path.normcase(XRD_ID_REGISTRY_PATH):
        entries.insert(0, {"name": "Local database", "path": XRD_ID_DB_PATH, "enabled": True})
        save_registry(entries, registry_path)
    return entries


def save_registry(entries: Sequence[Dict[str, Any]], registry_path: Optional[str] = None) -> None:
    registry_path = registry_path or XRD_ID_REGISTRY_PATH
    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump({"version": 1, "databases": list(entries)}, f, indent=1)


def _unique_name(base: str, entries: Sequence[Dict[str, Any]]) -> str:
    taken = {e.get("name") for e in entries}
    name, k = base, 2
    while name in taken:
        name = f"{base} ({k})"
        k += 1
    return name


def register_database(path: str, name: Optional[str] = None, *,
                      registry_path: Optional[str] = None,
                      import_dir: Optional[str] = None,
                      log=print) -> Dict[str, Any]:
    """Register a card database for searching. A PRISM-format .sq is
    registered in place (no copy — network drives are fine); a QualX-format
    .sq is converted ONCE into the indexed format under import_dir, then
    the converted file is registered (with the original recorded as
    'origin'). Re-registering an already-registered path is a no-op.
    Raises ValueError when the file isn't a recognizable card database."""
    registry_path = registry_path or XRD_ID_REGISTRY_PATH
    import_dir = import_dir or XRD_ID_IMPORT_DIR
    path = os.path.abspath(path)
    fmt = sniff_sq_format(path)
    if fmt is None:
        raise ValueError(
            f"{os.path.basename(path)} is not a recognizable card database "
            "(expected a QualX-format or PRISM-format .sq SQLite file).")
    entries = load_registry(registry_path)
    for e in entries:
        if os.path.normcase(e.get("path", "")) == os.path.normcase(path) or \
                os.path.normcase(e.get("origin") or "") == os.path.normcase(path):
            return e
    name = _unique_name((name or os.path.splitext(os.path.basename(path))[0]).strip(), entries)
    if fmt == "qualx":
        os.makedirs(import_dir, exist_ok=True)
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", name) or "db"
        out_path = os.path.join(import_dir, f"{slug}.sq")
        log(f"Importing '{name}' (one-time indexing — large databases take a few minutes)…")
        counts = build_xrd_database([(path, name)], out_path=out_path, log=log)
        log(f"Imported {sum(counts.values())} cards from '{name}'.")
        entry = {"name": name, "path": out_path, "enabled": True, "origin": path}
    else:
        entry = {"name": name, "path": path, "enabled": True}
    entries.append(entry)
    save_registry(entries, registry_path)
    return entry


def register_folder(folder: str, *, registry_path: Optional[str] = None,
                    import_dir: Optional[str] = None, log=print) -> List[Dict[str, Any]]:
    """Register every recognizable .sq in a folder (non-recursive first,
    then one level of subfolders — the layout database downloads unpack to).
    Unrecognizable files are skipped with a log line."""
    added: List[Dict[str, Any]] = []
    candidates: List[str] = []
    for root in [folder] + sorted(
            os.path.join(folder, d) for d in os.listdir(folder)
            if os.path.isdir(os.path.join(folder, d))):
        candidates += sorted(
            os.path.join(root, f) for f in os.listdir(root)
            if f.lower().endswith(".sq") and os.path.isfile(os.path.join(root, f)))
    for p in candidates:
        try:
            added.append(register_database(p, registry_path=registry_path,
                                           import_dir=import_dir, log=log))
        except ValueError as exc:
            log(f"Skipped {os.path.basename(p)}: {exc}")
    return added


def unregister_database(name: str, *, registry_path: Optional[str] = None) -> None:
    """Remove a database from the registry (the .sq file itself is never
    deleted — it may be the user's only copy)."""
    entries = [e for e in load_registry(registry_path) if e.get("name") != name]
    save_registry(entries, registry_path)


def set_database_enabled(name: str, enabled: bool, *,
                         registry_path: Optional[str] = None) -> None:
    entries = load_registry(registry_path)
    for e in entries:
        if e.get("name") == name:
            e["enabled"] = bool(enabled)
    save_registry(entries, registry_path)


def enabled_database_paths(registry_path: Optional[str] = None) -> List[str]:
    return [e["path"] for e in load_registry(registry_path)
            if e.get("enabled", True) and os.path.isfile(e.get("path", ""))]


# =============================================================================
# Crystal-system inference (filters): the databases store the Hermann-
# Mauguin symbol as text, so the system is derived from the symbol's
# rotational content — standard H-M reading order (cubic before trigonal
# because m-3m contains a 3; hexagonal before tetragonal is irrelevant as
# 6 and 4 never co-occur as principal axes).
# =============================================================================

CRYSTAL_SYSTEMS = ["cubic", "hexagonal", "trigonal", "tetragonal",
                   "orthorhombic", "monoclinic", "triclinic"]


def crystal_system(spacegroup: str) -> str:
    """Best-effort crystal system from an H-M space-group string; ''
    when the string is empty/unreadable. Decision tree: R lattice ->
    trigonal; primary 6 -> hexagonal; a 3 in the PRIMARY position ->
    trigonal, a 3 anywhere later (the body diagonal) -> cubic (covers
    23, m-3, 432, -43m, m-3m and screw variants like P4132); primary
    4 -> tetragonal; else count 2-fold/mirror directions."""
    s = (spacegroup or "").strip()
    if not s:
        return ""
    body = re.sub(r"\s+", "", s)
    rest = body[1:] if body[:1].upper() in "PABCIFR" and len(body) > 1 else body
    seq = re.sub(r"[-/_:]", "", rest)  # symbol sequence without decorations
    if not seq:
        return ""
    if body[:1].upper() == "R":
        return "trigonal"
    if seq[0] == "6":
        return "hexagonal"
    if "3" in seq:
        return "trigonal" if seq[0] == "3" else "cubic"
    if seq[0] == "4":
        return "tetragonal"
    n_dirs = len(re.findall(r"[2mcdnabe]", seq))
    if n_dirs >= 3:
        return "orthorhombic"
    if seq in ("1", "-1") or rest in ("1", "-1"):
        return "triclinic"
    if n_dirs >= 1:
        return "monoclinic"
    return ""


# =============================================================================
# Search-match
# =============================================================================

@dataclass
class XrdMatch:
    card_id: int
    source: str
    source_code: str
    name: str
    mineral: str
    formula: str
    spacegroup: str
    quality: str
    rir: Optional[float]
    fom: float               # 0..1 figure of merit (geometric-mean coverage)
    cov_card: float          # how much of the card's pattern is in the data
    cov_query: float         # how much of the data the card explains
    n_matched: int
    db: str = ""             # which registered database the card came from
    matched_pairs: List[Tuple[float, float]] = field(default_factory=list)  # (query 2θ, card 2θ)
    d: np.ndarray = field(default_factory=lambda: np.array([]))
    i: np.ndarray = field(default_factory=lambda: np.array([]))
    shift: float = 0.0            # refined global 2θ offset applied to the card (°)
    n_card_lines: int = 0         # card lines inside the measured range
    mean_abs_dev: float = 0.0     # mean |Δ2θ| over the matched lines (°)
    strongest_matched: bool = True  # is the card's strongest in-range line observed?


def _greedy_assign(c_tt: np.ndarray, q_tt: np.ndarray, tol: float) -> Tuple[np.ndarray, np.ndarray]:
    """Assign card lines to measured peaks, at most one each.

    Card lines arrive strongest-first (the index stores them that way), and
    each takes its nearest still-free peak inside `tol` — so a card's 100%
    line is never robbed of its peak by a 3% line that happens to sit
    marginally closer, which is what an unordered nearest-neighbour
    assignment does. Returns (query index per card line, −1 = unmatched;
    |Δ2θ| per card line).

    Vectorized on purpose: the two argsort/take calls replace a per-line
    numpy argmin, and this runs once per candidate card — tens of thousands
    of times per search.
    """
    n_c, n_q = len(c_tt), len(q_tt)
    assign = np.full(n_c, -1, int)
    dev = np.full(n_c, np.inf, float)
    if n_c == 0 or n_q == 0:
        return assign, dev
    dist = np.abs(c_tt[:, None] - q_tt[None, :])
    order = np.argsort(dist, axis=1, kind="stable")
    ranked = np.take_along_axis(dist, order, axis=1)
    used = np.zeros(n_q, bool)
    for k in range(n_c):
        row_j, row_d = order[k], ranked[k]
        for t in range(n_q):
            if row_d[t] > tol:
                break
            j = row_j[t]
            if not used[j]:
                used[j] = True
                assign[k] = j
                dev[k] = row_d[t]
                break
    return assign, dev


def _score_card(c_tt: np.ndarray, c_i: np.ndarray, q_tt: np.ndarray, q_i: np.ndarray,
                tol: float, shift: float) -> Dict[str, Any]:
    """Score one card at one 2θ offset.

    Coverage is weighted by how well each line actually lands: a line at
    the very edge of the tolerance window contributes almost nothing, one
    dead on contributes fully (w = 1 − (Δ/tol)²). Without that weight a
    tolerance generous enough to absorb sample displacement makes a sloppy
    match indistinguishable from an exact one — the complaint behind
    "everything scores 90%".
    """
    assign, dev = _greedy_assign(c_tt + shift, q_tt, tol)
    ok = assign >= 0
    if not np.any(ok):
        return {"fom": 0.0, "cov_card": 0.0, "cov_query": 0.0, "n_matched": 0,
                "pairs": [], "mad": 0.0, "strongest": False, "shift": shift,
                "resid": 0.0}
    w = 1.0 - (dev[ok] / tol) ** 2
    ci_total = float(c_i.sum())
    qi_total = float(q_i.sum())
    cov_card = float((w * c_i[ok]).sum()) / ci_total if ci_total > 0 else 0.0
    cov_query = float((w * q_i[assign[ok]]).sum()) / qi_total if qi_total > 0 else 0.0
    # Signed residual, intensity-weighted: the global offset this card
    # would like. Sign convention: positive = the card sits BELOW the data.
    signed = q_tt[assign[ok]] - (c_tt[ok] + shift)
    weights = c_i[ok]
    resid = float(np.average(signed, weights=weights)) if weights.sum() > 0 else 0.0
    return {
        "fom": float(np.sqrt(cov_card * cov_query)),
        "cov_card": cov_card, "cov_query": cov_query, "n_matched": int(ok.sum()),
        "pairs": [(float(q_tt[assign[k]]), float(c_tt[k] + shift))
                  for k in np.flatnonzero(ok)],
        "mad": float(np.mean(dev[ok])), "strongest": bool(assign[0] >= 0),
        "shift": shift, "resid": resid,
    }


def _dedup_key(source_code: str, name: str, mineral: str, formula: str,
               spacegroup: str, quality: str) -> Tuple[str, str, str, str]:
    """Cards identical in code + composition + symmetry + quality are the
    same reference record, however many databases carry it (user request:
    e.g. the same COD entry present in both an inorganic subset and the
    full COD shows up twice otherwise)."""
    phase = (mineral or name or formula or "").strip().lower()
    return (str(source_code).strip(), phase,
            re.sub(r"\s+", "", (spacegroup or "")).lower(), (quality or "").strip().upper())


def search_match(
    query_two_theta: Sequence[float], query_intensity: Optional[Sequence[float]] = None, *,
    wavelength: float = CU_KA1, tol_two_theta: float = 0.2,
    elements_all: Sequence[str] = (), elements_none: Sequence[str] = (),
    sources: Sequence[str] = (), top_n: int = 40,
    two_theta_range: Optional[Tuple[float, float]] = None,
    max_candidates: int = 20000, db_path: str = XRD_ID_DB_PATH,
    db_paths: Optional[Sequence[str]] = None,
    qualities: Sequence[str] = (), crystal_systems: Sequence[str] = (),
    spacegroup_contains: str = "", shift_tolerance: float = 0.0,
    n_probe_peaks: int = 12, min_matched: int = 1,
) -> List[XrdMatch]:
    """Rank database cards against a measured peak list.

    query_two_theta/intensity: the measured peaks (equal intensities are
    assumed when none given). tol_two_theta: match window in °2θ.
    elements_all: card must contain ALL of these; elements_none: none.
    sources: restrict to these source tags. two_theta_range: only card
    lines inside the measured range count against the card's coverage.
    db_paths: probe SEVERAL registered databases at once (QualX-style) —
    results are merged, DEDUPLICATED (same code+phase+spacegroup+quality
    kept once, best FoM wins), and re-ranked; each match's .db says which
    file it came from. When db_paths is None, the single db_path is probed.
    qualities / crystal_systems / spacegroup_contains: optional card
    filters (quality codes; systems from CRYSTAL_SYSTEMS; substring of
    the H-M symbol).

    shift_tolerance: allow each card a refined global 2θ offset up to this
    many degrees (sample displacement / zero error — the dominant
    systematic in Bragg-Brentano geometry). 0 disables it; the offset
    actually used comes back as XrdMatch.shift.
    n_probe_peaks: how many of the strongest measured peaks seed the
    candidate prefilter. More reaches deeper into minor phases at the cost
    of a longer search.
    min_matched: drop cards explaining fewer than this many measured peaks.
    """
    paths = [str(p) for p in db_paths] if db_paths is not None else [db_path]
    results: List[XrdMatch] = []
    for p in paths:
        results.extend(_search_match_one(
            query_two_theta, query_intensity, wavelength=wavelength,
            tol_two_theta=tol_two_theta, elements_all=elements_all,
            elements_none=elements_none, sources=sources, top_n=top_n,
            two_theta_range=two_theta_range, max_candidates=max_candidates,
            db_path=p, qualities=qualities, crystal_systems=crystal_systems,
            spacegroup_contains=spacegroup_contains,
            shift_tolerance=shift_tolerance, n_probe_peaks=n_probe_peaks,
            min_matched=min_matched))
    results.sort(key=lambda r: (r.fom, r.n_matched), reverse=True)
    seen: set = set()
    unique: List[XrdMatch] = []
    for r in results:
        key = _dedup_key(r.source_code, r.name, r.mineral, r.formula, r.spacegroup, r.quality)
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique[:top_n]


def _search_match_one(
    query_two_theta: Sequence[float], query_intensity: Optional[Sequence[float]] = None, *,
    wavelength: float = CU_KA1, tol_two_theta: float = 0.2,
    elements_all: Sequence[str] = (), elements_none: Sequence[str] = (),
    sources: Sequence[str] = (), top_n: int = 40,
    two_theta_range: Optional[Tuple[float, float]] = None,
    max_candidates: int = 20000, db_path: str = XRD_ID_DB_PATH,
    qualities: Sequence[str] = (), crystal_systems: Sequence[str] = (),
    spacegroup_contains: str = "", shift_tolerance: float = 0.0,
    n_probe_peaks: int = 12, min_matched: int = 1,
) -> List[XrdMatch]:
    if not os.path.isfile(db_path):
        raise FileNotFoundError(
            f"No XRD database at {db_path}. Register one in the XRD ID "
            "workspace (Add database…) or build one with build_xrd_database().")
    q_tt = np.asarray(list(query_two_theta), float)
    if len(q_tt) == 0:
        return []
    q_i = (np.asarray(list(query_intensity), float) if query_intensity is not None
           else np.ones_like(q_tt))
    q_i = np.where(np.isfinite(q_i) & (q_i > 0), q_i, 1.0)
    q_d = two_theta_to_d(q_tt, wavelength)

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    # ---- candidate prefilter: cards whose strong lines coincide with the
    # query's strongest peaks (union, ranked by hit weight). With a flat
    # intensity list (hand-typed peaks, no spectrum) "strongest N" would be
    # an arbitrary subset — probe every peak instead so no phase whose
    # lines happen to sit late in the list is silently missed.
    #
    # Candidates are ranked by intensity-WEIGHTED hits, not a raw count:
    # a card whose strong line sits under the pattern's main peak is a
    # better lead than one that only clips two weak ones, and when the
    # max_candidates cut bites (a 700k-card database routinely returns
    # more leads than that) the raw count threw the better lead away.
    n_probe = len(q_tt) if float(q_i.max()) == float(q_i.min()) \
        else min(len(q_tt), max(1, int(n_probe_peaks)))
    strongest = np.argsort(q_i)[::-1][:n_probe]
    q_weight = q_i / (float(q_i.max()) or 1.0)
    hits: Dict[int, float] = {}
    for qi in strongest:
        tt = q_tt[qi]
        # Δd from Δ2θ: |dd| = d·cot(θ)·Δθ(rad)/1 … evaluated at this line.
        # The window also has to absorb the 2θ offset the scoring stage is
        # allowed to refine away, or the prefilter would never surface the
        # displaced cards that offset exists for.
        theta = np.radians(tt / 2.0)
        window = tol_two_theta + 2.0 * max(0.0, float(shift_tolerance))
        dd = abs(q_d[qi] / np.tan(theta) * np.radians(window / 2.0)) if theta > 0 else 0.01
        dd = max(dd, 1e-4)
        for (cid,) in con.execute("SELECT card_id FROM strong_lines WHERE d BETWEEN ? AND ?",
                                  (q_d[qi] - dd, q_d[qi] + dd)):
            hits[cid] = hits.get(cid, 0.0) + float(q_weight[qi])
    if not hits:
        con.close()
        return []

    # ---- chemistry filters, BEFORE the max_candidates cut.
    # Filtering the truncated pool instead would throw away every
    # chemically valid card that happened to rank past the cut on
    # strong-line coincidence alone — and a chemistry filter is precisely
    # the case where the user has told us those cards are the ones that
    # matter. On a 700k-card database the cut bites on almost every
    # search, so the ordering is not academic.
    if elements_all:
        keep = None
        for el in elements_all:
            got = {cid for (cid,) in con.execute(
                "SELECT card_id FROM elements WHERE element = ?", (el.strip().capitalize(),))}
            keep = got if keep is None else (keep & got)
        hits = {c: w for c, w in hits.items() if c in (keep or set())}
    if elements_none:
        drop = set()
        for el in elements_none:
            drop |= {cid for (cid,) in con.execute(
                "SELECT card_id FROM elements WHERE element = ?", (el.strip().capitalize(),))}
        hits = {c: w for c, w in hits.items() if c not in drop}
    candidates = sorted(hits, key=lambda c: hits[c], reverse=True)[:max_candidates]
    if not candidates:
        con.close()
        return []

    # ---- score
    marks = ",".join("?" * len(candidates))
    src_filter = ""
    params: List[Any] = list(candidates)
    if sources:
        src_filter = f" AND source IN ({','.join('?' * len(sources))})"
        params += list(sources)
    rows = con.execute(
        f"SELECT card_id, source, source_code, name, mineral, formula, spacegroup, quality, rir, d, i "
        f"FROM cards WHERE card_id IN ({marks}){src_filter}", params).fetchall()
    con.close()

    db_label = os.path.splitext(os.path.basename(db_path))[0]
    quality_set = {q.strip().upper() for q in qualities if q.strip()}
    system_set = {c.strip().lower() for c in crystal_systems if c.strip()}
    sg_needle = (spacegroup_contains or "").strip().lower()
    results: List[XrdMatch] = []
    for (cid, source, code, name, mineral, formula, sg, quality, rir, d_text, i_text) in rows:
        if quality_set and (quality or "").strip().upper() not in quality_set:
            continue
        if system_set and crystal_system(sg or "") not in system_set:
            continue
        if sg_needle and sg_needle not in (sg or "").lower():
            continue
        c_d = np.array([float(v) for v in d_text.split(",")], float)
        c_i = np.array([float(v) for v in i_text.split(",")], float)
        c_tt = d_to_two_theta(c_d, wavelength)
        ok = np.isfinite(c_tt)
        if two_theta_range is not None:
            ok &= (c_tt >= two_theta_range[0]) & (c_tt <= two_theta_range[1])
        c_tt, c_i_in = c_tt[ok], c_i[ok]
        if len(c_tt) == 0:
            continue

        best = _score_card(c_tt, c_i_in, q_tt, q_i, tol_two_theta, 0.0)
        if shift_tolerance > 0 and best["n_matched"]:
            # One Gauss-Newton-ish step: the intensity-weighted mean
            # residual IS the offset that best re-centres the matched
            # lines, so a single refinement converges for anything the
            # bound admits. Clamped, then kept only if it really helps —
            # a shift must never invent a match it could not earn.
            trial = float(np.clip(best["resid"], -shift_tolerance, shift_tolerance))
            if abs(trial) > 1e-4:
                cand = _score_card(c_tt, c_i_in, q_tt, q_i, tol_two_theta, trial)
                if cand["fom"] > best["fom"]:
                    best = cand
        if best["n_matched"] < max(1, int(min_matched)):
            continue
        results.append(XrdMatch(
            card_id=cid, source=source, source_code=code, name=name or "", mineral=mineral or "",
            formula=formula or "", spacegroup=sg or "", quality=quality or "", rir=rir,
            fom=best["fom"], cov_card=best["cov_card"], cov_query=best["cov_query"],
            n_matched=best["n_matched"], db=db_label, matched_pairs=best["pairs"],
            d=c_d, i=c_i, shift=best["shift"], n_card_lines=int(len(c_tt)),
            mean_abs_dev=best["mad"], strongest_matched=best["strongest"],
        ))
    results.sort(key=lambda r: (r.fom, r.n_matched), reverse=True)
    return results[:top_n]


def _rows_to_cards(rows, db_label: str = "") -> List[Dict[str, Any]]:
    out = []
    for (cid, source, code, name, mineral, formula, sg, quality, d_text, i_text) in rows:
        out.append({
            "card_id": cid, "source": source, "source_code": code, "name": name,
            "mineral": mineral, "formula": formula, "spacegroup": sg, "quality": quality,
            "db": db_label,
            "d": np.array([float(v) for v in d_text.split(",")], float),
            "i": np.array([float(v) for v in i_text.split(",")], float),
        })
    return out


def find_cards_by_elements(query: str, *, mode: str = "exact", limit: int = 100,
                           db_path: str = XRD_ID_DB_PATH,
                           db_paths: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Element-set card lookup (user report: text-searching 'TiO2' missed
    cards whose formula is written 'O2 Ti'). The query is parsed as a
    chemical formula; cards are matched on their ELEMENT SET, so formula
    spelling/order/stoichiometry writing don't matter.
    mode='exact'  — card contains exactly these elements and no others
    mode='contains' — card contains at least these elements."""
    import xraydb
    if db_paths is not None:
        out: List[Dict[str, Any]] = []
        for p in db_paths:
            out += find_cards_by_elements(query, mode=mode, limit=limit - len(out), db_path=p)
            if len(out) >= limit:
                break
        return out
    if not os.path.isfile(db_path):
        return []
    try:
        elements = sorted({el.capitalize() for el in xraydb.chemparse(query.strip())})
    except (ValueError, KeyError):
        return []  # not a chemical formula — the text search handles it
    if not elements:
        return []
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    marks = ",".join("?" * len(elements))
    if mode == "exact":
        ids = [r[0] for r in con.execute(
            f"SELECT card_id FROM elements GROUP BY card_id "
            f"HAVING SUM(CASE WHEN element IN ({marks}) THEN 1 ELSE 0 END) = COUNT(DISTINCT element) "
            f"AND COUNT(DISTINCT element) = ?", (*elements, len(elements))).fetchall()]
    else:
        keep = None
        for el in elements:
            got = {r[0] for r in con.execute("SELECT card_id FROM elements WHERE element = ?", (el,))}
            keep = got if keep is None else (keep & got)
        ids = sorted(keep or set())
    ids = ids[:limit]
    if not ids:
        con.close()
        return []
    rows = con.execute(
        f"SELECT card_id, source, source_code, name, mineral, formula, spacegroup, quality, d, i "
        f"FROM cards WHERE card_id IN ({','.join('?' * len(ids))})", ids).fetchall()
    con.close()
    return _rows_to_cards(rows, os.path.splitext(os.path.basename(db_path))[0])


def find_cards_by_text(text: str, *, limit: int = 50, db_path: str = XRD_ID_DB_PATH,
                       db_paths: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """Name/mineral/formula substring lookup — the Raman↔XRD bridge uses
    this to pull reference patterns for an already-identified mineral."""
    if db_paths is not None:
        out: List[Dict[str, Any]] = []
        for p in db_paths:
            out += find_cards_by_text(text, limit=limit - len(out), db_path=p)
            if len(out) >= limit:
                break
        return out
    if not os.path.isfile(db_path):
        return []
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    like = f"%{text.strip()}%"
    rows = con.execute(
        "SELECT card_id, source, source_code, name, mineral, formula, spacegroup, quality, d, i "
        "FROM cards WHERE mineral LIKE ? OR name LIKE ? OR formula LIKE ? LIMIT ?",
        (like, like, like, int(limit))).fetchall()
    con.close()
    return _rows_to_cards(rows, os.path.splitext(os.path.basename(db_path))[0])


def find_card_by_code(source: str, source_code: str, *,
                      db_path: str = XRD_ID_DB_PATH,
                      db_paths: Optional[Sequence[str]] = None) -> Optional[Dict[str, Any]]:
    """The one card a recorded identification points at (source tag +
    original code). Lets an accepted phase be reconstructed — stick
    pattern included — after PRISM restarts or a project is reopened,
    from the few fields worth storing in the spectrum's metadata."""
    for p in ([str(q) for q in db_paths] if db_paths is not None else [db_path]):
        if not os.path.isfile(p):
            continue
        con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
        rows = con.execute(
            "SELECT card_id, source, source_code, name, mineral, formula, spacegroup, "
            "quality, d, i FROM cards WHERE source = ? AND source_code = ? LIMIT 1",
            (str(source), str(source_code))).fetchall()
        con.close()
        if rows:
            return _rows_to_cards(rows, os.path.splitext(os.path.basename(p))[0])[0]
    return None


# =============================================================================
# A folder of CIFs as a search-match source
#
# Structure files are the other half of a lab's reference material: papers
# ship CIFs, COD is downloadable as CIFs, and a group's own refinements end
# as CIFs. They are not a card database — they carry no d-I list — so the
# pattern is CALCULATED from each structure (xrd.cif_patterns) and written
# into exactly the same indexed .sq the card databases use. Everything
# downstream then works unchanged: multi-database probing, chemistry and
# symmetry filters, dedup, accept, the card browser.
#
# The one addition is a cif_files table mapping each card back to the file
# it came from, so exporting a CIF for an accepted phase identified this
# way is a copy of the user's own file — no lookup, no download, and the
# exported structure is exactly the one that was matched.
# =============================================================================

def _cif_source_code(path: str) -> str:
    """The card code for a structure file: its filename stem, which is what
    the user recognizes, and which is the COD number for a COD download."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem[:64] or "cif"


def build_cif_database(
    folder: str, *, out_path: str, tag: Optional[str] = None,
    wavelength: float = CU_KA1, two_theta_max: float = 90.0,
    recursive: bool = True, min_lines: int = 3, log=print,
) -> Dict[str, int]:
    """Calculate every CIF in a folder into one PRISM-indexed .sq.

    Files that cannot be read as a structure are skipped with a logged
    reason rather than failing the whole folder — a reference folder
    collected over years always holds a few broken or non-structure
    files. Returns {tag: n_cards}.
    """
    from xrd import cif_patterns as cifp

    tag = tag or os.path.basename(os.path.normpath(folder)) or "CIF"
    paths = cifp.list_cif_files(folder, recursive=recursive)
    if not paths:
        raise ValueError(f"No .cif files found in {folder}.")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)
    out = sqlite3.connect(out_path)
    out.executescript("""
        CREATE TABLE cards (
            card_id INTEGER PRIMARY KEY,
            source TEXT NOT NULL, source_code TEXT NOT NULL,
            name TEXT, mineral TEXT, formula TEXT, spacegroup TEXT,
            quality TEXT, rir REAL, nd INTEGER, d TEXT, i TEXT
        );
        CREATE TABLE elements (card_id INTEGER NOT NULL, element TEXT NOT NULL);
        CREATE TABLE strong_lines (card_id INTEGER NOT NULL, d REAL NOT NULL, i REAL NOT NULL);
        CREATE TABLE sources (tag TEXT, date TEXT, ncard INTEGER, origin TEXT);
        CREATE TABLE cif_files (card_id INTEGER NOT NULL, path TEXT NOT NULL);
    """)

    import datetime
    kept = skipped = 0
    for k, path in enumerate(paths, 1):
        if k % 50 == 0:
            log(f"  {k}/{len(paths)} files, {kept} structures calculated…")
        try:
            pat = cifp.powder_pattern_from_cif(
                path, wavelength=wavelength, two_theta_max=two_theta_max,
                max_lines=MAX_LINES_PER_CARD)
        except Exception as exc:
            skipped += 1
            log(f"  skipped {os.path.basename(path)}: {exc}")
            continue
        if len(pat.d) < min_lines:
            skipped += 1
            log(f"  skipped {os.path.basename(path)}: only {len(pat.d)} reflection(s) in range.")
            continue
        order = np.argsort(pat.intensity)[::-1][:MAX_LINES_PER_CARD]
        d, i = pat.d[order], pat.intensity[order]
        cur = out.execute(
            "INSERT INTO cards (source, source_code, name, mineral, formula, spacegroup, "
            "quality, rir, nd, d, i) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (tag, _cif_source_code(path), pat.name, "", pat.formula, pat.spacegroup,
             "CIF" if pat.calculated_intensities else "cell", None, len(d),
             ",".join(f"{v:.5f}" for v in d), ",".join(f"{v:.2f}" for v in i)))
        card_id = int(cur.lastrowid)
        out.executemany("INSERT INTO strong_lines VALUES (?,?,?)",
                        [(card_id, float(dv), float(iv))
                         for dv, iv in zip(d[:N_STRONG], i[:N_STRONG])])
        out.executemany("INSERT INTO elements VALUES (?,?)",
                        [(card_id, el) for el in pat.elements])
        out.execute("INSERT INTO cif_files VALUES (?,?)", (card_id, os.path.abspath(path)))
        kept += 1

    out.execute("INSERT INTO sources VALUES (?,?,?,?)",
                (tag, datetime.date.today().isoformat(), kept, os.path.abspath(folder)))
    out.executescript("""
        CREATE INDEX idx_strong_d ON strong_lines(d);
        CREATE INDEX idx_elements ON elements(element, card_id);
        CREATE INDEX idx_cards_mineral ON cards(mineral);
        CREATE INDEX idx_cards_formula ON cards(formula);
        CREATE INDEX idx_cif_files ON cif_files(card_id);
    """)
    out.commit()
    out.close()
    log(f"[{tag}] {kept} structures calculated, {skipped} file(s) skipped.")
    return {tag: kept}


def register_cif_folder(folder: str, name: Optional[str] = None, *,
                        registry_path: Optional[str] = None,
                        import_dir: Optional[str] = None,
                        wavelength: float = CU_KA1, two_theta_max: float = 90.0,
                        recursive: bool = True, log=print) -> Dict[str, Any]:
    """Register a folder of structure files as a searchable database. The
    calculated patterns are stored under import_dir; re-registering the
    same folder RECALCULATES it, which is what the user wants after adding
    files to the folder."""
    registry_path = registry_path or XRD_ID_REGISTRY_PATH
    import_dir = import_dir or XRD_ID_IMPORT_DIR
    folder = os.path.abspath(folder)
    entries = load_registry(registry_path)
    existing = next((e for e in entries
                     if os.path.normcase(e.get("cif_folder") or "") == os.path.normcase(folder)),
                    None)
    label = (name or os.path.basename(os.path.normpath(folder)) or "CIF folder").strip()
    if existing is None:
        label = _unique_name(label, entries)
    else:
        label = existing["name"]
    os.makedirs(import_dir, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", label) or "cif"
    out_path = os.path.join(import_dir, f"cif_{slug}.sq")
    log(f"Calculating powder patterns for the CIFs in '{label}'…")
    counts = build_cif_database(folder, out_path=out_path, tag=label, wavelength=wavelength,
                                two_theta_max=two_theta_max, recursive=recursive, log=log)
    entry = {"name": label, "path": out_path, "enabled": True, "cif_folder": folder,
             "kind": "cif", "wavelength": float(wavelength), "recursive": bool(recursive),
             "ncards": int(sum(counts.values()))}
    if existing is None:
        entries.append(entry)
    else:
        existing.update(entry)
    save_registry(entries, registry_path)
    return entry


def cif_path_for_card(source: str, source_code: str, *,
                      db_paths: Sequence[str] = ()) -> Optional[str]:
    """The structure file a CIF-derived card was calculated from, if the
    database that produced it is still registered and the file still
    exists."""
    for p in db_paths:
        if not os.path.isfile(p):
            continue
        try:
            con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
            has = con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='cif_files'").fetchone()
            if not has:
                con.close()
                continue
            row = con.execute(
                "SELECT f.path FROM cif_files f JOIN cards c ON c.card_id = f.card_id "
                "WHERE c.source = ? AND c.source_code = ? LIMIT 1",
                (str(source), str(source_code))).fetchone()
            con.close()
        except sqlite3.Error:
            continue
        if row and os.path.isfile(row[0]):
            return row[0]
    return None


# =============================================================================
# Structure (CIF) export — what a Rietveld refinement actually needs
#
# A card database carries a d-I LINE LIST, not a structure: no cell, no
# atomic coordinates, no occupancies. So a CIF cannot be written from a
# card, only FETCHED for it, and only when the card's code identifies a
# structure in a public archive. COD-derived databases do exactly that —
# their card code IS the COD entry number — so an accepted phase can be
# turned into a refinable starting model. PDF-2-style cards cannot: they
# are d-I lists whose codes are PDF numbers, and no amount of processing
# recovers atom positions from them. Both cases are reported honestly
# rather than one of them silently producing an empty file.
#
# Lookup order per phase: a local mirror the user points at (offline, and
# the only option on an air-gapped machine), then the download cache, then
# crystallography.net. Everything downloaded is cached under
# ~/.raman_cache/xrd_id/cif so a re-export of the same phases is offline.
# =============================================================================

XRD_CIF_CACHE_DIR = os.path.join(XRD_ID_CACHE_DIR, "cif")
COD_CIF_URL = "https://www.crystallography.net/cod/{cod_id}.cif"


def cod_id_for_card(source: str, source_code: str) -> Optional[str]:
    """The COD entry number a card points at, or None when the card is not
    COD-derived. COD numbers are 7-digit (a few legacy ones 6-digit);
    anything else — a PDF-2 number, an ICSD collection code — is not one,
    and guessing would fetch somebody else's structure."""
    tag = (source or "").strip().upper()
    code = (source_code or "").strip()
    if not code.isdigit():
        return None
    if "COD" not in tag:
        return None
    return code if 6 <= len(code) <= 8 else None


def find_local_cif(cod_id: str, folders: Sequence[str]) -> Optional[str]:
    """Look for <cod_id>.cif in the user's own mirror. Both layouts are
    tried: a flat folder of CIFs, and the nested tree the COD bulk
    download unpacks to (cif/1/01/09/1010930.cif)."""
    cid = str(cod_id).strip()
    if not cid.isdigit():
        return None
    padded = cid.zfill(7)
    nested = os.path.join(padded[0], padded[1:3], padded[3:5], f"{cid}.cif")
    for folder in folders:
        folder = str(folder or "").strip()
        if not folder or not os.path.isdir(folder):
            continue
        for rel in (f"{cid}.cif", nested, os.path.join("cif", nested)):
            candidate = os.path.join(folder, rel)
            if os.path.isfile(candidate):
                return candidate
    return None


def download_cod_cif(cod_id: str, *, cache_dir: Optional[str] = None,
                     timeout: float = 30.0) -> str:
    """Fetch one COD entry, caching it. Written to a .part sidecar first so
    a connection that dies mid-stream never leaves a truncated CIF behind
    that the next run would happily reuse."""
    import urllib.request

    cache_dir = cache_dir or XRD_CIF_CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    dest = os.path.join(cache_dir, f"{cod_id}.cif")
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        return dest
    req = urllib.request.Request(COD_CIF_URL.format(cod_id=cod_id),
                                 headers={"User-Agent": "PRISM/1 (github.com/sams808/PRISM)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    if not data.strip():
        raise ValueError(f"COD returned an empty file for entry {cod_id}.")
    tmp = dest + ".part"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, dest)
    return dest


def _phase_filename(phase: Dict[str, Any], cod_id: Optional[str]) -> str:
    label = (phase.get("mineral") or phase.get("name") or phase.get("formula")
             or phase.get("source_code") or "phase")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(label)).strip("_") or "phase"
    if cod_id:
        return f"{slug[:60]}_COD{cod_id}.cif"
    tail = re.sub(r"[^A-Za-z0-9._-]+", "_", str(phase.get("source_code", "")).strip()).strip("_")
    if not tail:
        return f"{slug[:60]}.cif"
    # a CIF-folder card's code IS its filename, which usually already
    # carries the phase name — don't emit 'Quartz_Quartz_1011172.cif'
    if slug.lower() in tail.lower():
        return f"{tail[:70]}.cif"
    return f"{slug[:60]}_{tail[:40]}.cif"


def cif_for_phase(phase: Dict[str, Any], *, local_folders: Sequence[str] = (),
                  allow_download: bool = True, cache_dir: Optional[str] = None,
                  timeout: float = 30.0, db_paths: Sequence[str] = ()) -> Tuple[Optional[str], str]:
    """Locate the structure file for one accepted phase. Returns (path or
    None, a human-readable note on where it came from or why there is
    none).

    A phase identified against the user's OWN CIF folder needs no lookup
    at all: the database remembers which file each card was calculated
    from, so the export hands back exactly the structure that matched."""
    own = cif_path_for_card(phase.get("source", ""), phase.get("source_code", ""),
                            db_paths=db_paths)
    if own:
        return own, f"the CIF this phase was matched against ({own})"
    cod_id = cod_id_for_card(phase.get("source", ""), phase.get("source_code", ""))
    if cod_id is None:
        return None, (f"no structure available - '{phase.get('source', '?')}' card "
                      f"{phase.get('source_code', '?')} is a d-I line list, not a "
                      f"structure entry")
    local = find_local_cif(cod_id, local_folders)
    if local:
        return local, f"local mirror ({local})"
    cached = os.path.join(cache_dir or XRD_CIF_CACHE_DIR, f"{cod_id}.cif")
    if os.path.isfile(cached) and os.path.getsize(cached) > 0:
        return cached, "download cache"
    if not allow_download:
        return None, f"COD {cod_id} not in the local mirror or cache (downloads are off)"
    try:
        return download_cod_cif(cod_id, cache_dir=cache_dir, timeout=timeout), \
            f"downloaded from COD ({cod_id})"
    except Exception as exc:  # network, 404, permissions - all reportable
        return None, f"COD {cod_id} could not be fetched: {type(exc).__name__}: {exc}"


def export_phase_cifs(
    phases: Sequence[Dict[str, Any]], out_dir: str, *,
    local_folders: Sequence[str] = (), allow_download: bool = True,
    cache_dir: Optional[str] = None, write_summary: bool = True,
    db_paths: Sequence[str] = (), log=print,
) -> Dict[str, Any]:
    """Collect the structures of a set of accepted phases into one folder
    ready to load into a Rietveld program (GSAS-II, FullProf, TOPAS,
    Profex...). Each phase dict needs at least 'source' and 'source_code';
    'mineral' / 'name' / 'formula' / 'fom' are used for naming and the
    summary. Returns {'written': [(label, path)], 'skipped': [(label,
    reason)], 'summary': path or None}."""
    os.makedirs(out_dir, exist_ok=True)
    written: List[Tuple[str, str]] = []
    skipped: List[Tuple[str, str]] = []
    rows: List[List[str]] = []
    for phase in phases:
        label = str(phase.get("mineral") or phase.get("name") or phase.get("formula")
                    or phase.get("source_code") or "phase")
        src, note = cif_for_phase(phase, local_folders=local_folders,
                                  allow_download=allow_download, cache_dir=cache_dir,
                                  db_paths=db_paths)
        base = [label, str(phase.get("formula", "")), str(phase.get("source", "")),
                str(phase.get("source_code", "")),
                f"{float(phase.get('fom') or 0.0) * 100:.0f}"]
        if src is None:
            log(f"  {label}: {note}")
            skipped.append((label, note))
            rows.append(base + ["", note])
            continue
        cod_id = cod_id_for_card(phase.get("source", ""), phase.get("source_code", ""))
        dest = os.path.join(out_dir, _phase_filename(phase, cod_id))
        with open(src, "rb") as fin, open(dest, "wb") as fout:
            fout.write(fin.read())
        log(f"  {label}: {os.path.basename(dest)} ({note})")
        written.append((label, dest))
        rows.append(base + [os.path.basename(dest), note])

    summary_path = None
    if write_summary and rows:
        import csv
        summary_path = os.path.join(out_dir, "phases.csv")
        with open(summary_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["phase", "formula", "source", "code", "fom_percent", "cif_file", "note"])
            w.writerows(rows)
    return {"written": written, "skipped": skipped, "summary": summary_path}
