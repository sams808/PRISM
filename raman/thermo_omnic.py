"""
thermo_omnic.py — readers for the files a Thermo Scientific DXR Raman
microscope writes through OMNIC (framework-agnostic).

Formats, as reverse-engineered on the group's own files (2026-10-21) and
checked against OMNIC's text export of the same spectrum:

``.spa`` (single spectrum, binary)
    Starts with the ASCII title "Spectral Data File"; bytes 30.. hold a
    free-text comment (stage position, signal-to-noise ratio). From offset
    304 a directory of 16-byte records: int16 key, int32 offset, int32
    size. Key 2 is the spectrum header: int32 number of points at +4,
    float32 LAST x at +16 and FIRST x at +20 (the data run from high to low
    wavenumber). Key 3 is the intensity block, float32 little-endian, one
    value per point, high wavenumber first. Keys 0x153 / 0x1b carry the
    exposure-time line. The reversed intensities match the OMNIC .txt
    export of the same file to 0.02 % (a float32 vs text rounding level).

``.mapx`` (map, HDF5 container)
    ``FileInfo`` attributes: SpectralRangeStart / SpectralRangeEnd (cm-1),
    SpectralType, SpatialUnitOfMeter (1e-9: nanometres), NumberROIs;
    ``FileInfo/MetaData`` is an XML blob with the acquisition parameters
    (laser power, scans, exposure, objective, grating, time of first frame).
    ``Regions/<name>/Dataset`` is a float32 array shaped (nx, ny, npoints)
    ("X, Y, W" in the group's DataspaceDescript) and ``-StagePosXYZ`` the
    region's bounding box [xmin, ymin, z, xmax, ymax, z] in stage units. The
    wavenumber axis is not stored as a dataset: it is the linear grid from
    SpectralRangeStart to SpectralRangeEnd over npoints, which reproduces
    the 1.9285 cm-1 step of the text exports exactly.

Reading ``.mapx`` needs h5py (a PRISM requirement since 2.8.0).
"""
from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

SPA_MAGIC = b"Spectral Data File"


# ----------------------------------------------------------------------
# .spa
# ----------------------------------------------------------------------

def is_spa(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(len(SPA_MAGIC)) == SPA_MAGIC
    except OSError:
        return False


def _spa_directory(raw: bytes) -> List[Tuple[int, int, int]]:
    """[(key, offset, size), …] from the 16-byte records at offset 304."""
    recs = []
    pos = 304
    while pos + 10 <= len(raw) and len(recs) < 64:
        key, = struct.unpack_from("<h", raw, pos)
        if key == 0:
            break
        off, size = struct.unpack_from("<ii", raw, pos + 2)
        if key not in (1,) and 0 <= off < len(raw) and size >= 0:
            recs.append((key, off, size))
        pos += 16
    return recs


def _text_blocks(raw: bytes, recs) -> List[str]:
    out = []
    for key, off, size in recs:
        if key in (2, 3) or size <= 0:
            continue
        chunk = raw[off:off + size]
        txt = chunk.decode("latin-1", errors="ignore").replace("\0", " ").strip()
        if txt and sum(c.isprintable() for c in txt) > 0.8 * len(txt):
            out.append(txt)
    return out


def read_spa(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """(x ascending in cm-1, y, meta) of an OMNIC .spa spectrum."""
    with open(path, "rb") as f:
        raw = f.read()
    if not raw.startswith(SPA_MAGIC):
        raise ValueError(f"{os.path.basename(path)} is not an OMNIC .spa file (no 'Spectral Data File' signature).")
    recs = _spa_directory(raw)
    hdr = next((off for key, off, size in recs if key == 2), None)
    data = next(((off, size) for key, off, size in recs if key == 3), None)
    if hdr is None or data is None:
        raise ValueError(f"{os.path.basename(path)}: no spectrum header/data block in the .spa directory.")
    npts, = struct.unpack_from("<i", raw, hdr + 4)
    last_x, first_x = struct.unpack_from("<ff", raw, hdr + 16)
    off, size = data
    n_avail = size // 4
    if npts <= 1 or n_avail < npts:
        raise ValueError(f"{os.path.basename(path)}: inconsistent point count ({npts} declared, {n_avail} stored).")
    y_raw = np.frombuffer(raw, dtype="<f4", count=npts, offset=off).astype(float)
    # the file runs from high to low wavenumber: first_x (header +20) is the
    # low end, last_x (header +16) the high end of the data as stored
    x_stored = np.linspace(last_x, first_x, npts)
    order = np.argsort(x_stored, kind="mergesort")
    x = x_stored[order]
    y = y_raw[order]
    comment = raw[30:300].split(b"\0")[0].decode("latin-1", errors="ignore").strip()
    meta: Dict[str, Any] = {"omnic_comment": comment, "n_points": int(npts),
                            "x_first_stored": float(last_x), "x_last_stored": float(first_x)}
    m = re.search(r"Position\s*X\s*=\s*(-?\d+)\s*Y\s*=\s*(-?\d+)", comment)
    if m:
        meta["stage_x"] = float(m.group(1)); meta["stage_y"] = float(m.group(2))
    m = re.search(r"Signal to noise ratio\s*=\s*([\d.]+)", comment)
    if m:
        meta["snr"] = float(m.group(1))
    for txt in _text_blocks(raw, recs):
        m = re.search(r"Exposure time:\s*([\d.]+)\s*seconds", txt)
        if m:
            meta["exposure_s"] = float(m.group(1))
        meta.setdefault("omnic_text", []).append(txt[:200])
    return x, y, meta


# ----------------------------------------------------------------------
# .mapx
# ----------------------------------------------------------------------

@dataclass
class MapRegion:
    name: str
    data: np.ndarray            # (nx, ny, npoints) float32 as stored
    x_positions: np.ndarray     # (nx,) stage x in micrometres (pixel centres)
    y_positions: np.ndarray     # (ny,) stage y in micrometres
    stage_box: Tuple[float, float, float, float]  # xmin, ymin, xmax, ymax in micrometres

    @property
    def shape(self) -> Tuple[int, int]:
        return int(self.data.shape[0]), int(self.data.shape[1])

    def spectrum(self, ix: int, iy: int) -> np.ndarray:
        return np.asarray(self.data[ix, iy, :], float)

    def mean_spectrum(self) -> np.ndarray:
        return np.asarray(self.data, float).reshape(-1, self.data.shape[-1]).mean(axis=0)


@dataclass
class OmnicMap:
    path: str
    x: np.ndarray                       # wavenumber axis (cm-1), ascending
    regions: List[MapRegion]
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_spectra(self) -> int:
        return sum(r.shape[0] * r.shape[1] for r in self.regions)


def is_mapx(path: str) -> bool:
    if not path.lower().endswith(".mapx"):
        return False
    try:
        with open(path, "rb") as f:
            return f.read(8) == b"\x89HDF\r\n\x1a\n"
    except OSError:
        return False


def _decode_attr(v) -> Any:
    if isinstance(v, np.ndarray) and v.dtype == np.uint8:
        b = bytes(v)
        try:
            return b.decode("utf-16-le").rstrip("\0") if b[1:2] == b"\0" else b.decode("utf-8").rstrip("\0")
        except UnicodeDecodeError:
            return b.decode("latin-1", errors="ignore")
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="ignore")
    if isinstance(v, np.generic):
        return v.item()
    return v


_META_VARS = {
    "m_Laser_mW": ("laser_mW", float), "m_NumberOfScans": ("n_scans", int),
    "m_ExposureTimeSec": ("exposure_s", float), "m_ObjectiveDescription": ("objective", str),
    "GratingSpacing": ("grating_lines_mm", int), "m_EMCCDGain": ("emccd_gain", int),
    "m_SpatialResolutionNM": ("spatial_resolution_nm", float), "m_TimeOfFirstFrame": ("time_of_first_frame", str),
    "LaserSerialNumber": ("laser_serial", str), "m_LaserPolarizer": ("laser_polarizer", str),
}


def _parse_metadata_xml(xml: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for var, (key, cast) in _META_VARS.items():
        m = re.search(r'NAME="' + re.escape(var) + r'"[^>]*>([^<]*)<', xml)
        if m and key not in out:
            try:
                out[key] = cast(m.group(1).strip())
            except (TypeError, ValueError):
                out[key] = m.group(1).strip()
    m = re.search(r'NAME="m_LaserWavelength[^"]*"[^>]*>([^<]*)<', xml)
    if m:
        try:
            out["laser_nm"] = float(m.group(1))
        except ValueError:
            pass
    return out


def read_mapx(path: str, *, load_data: bool = True) -> OmnicMap:
    """Read an OMNIC .mapx map. With load_data=False the region arrays are
    left as h5py datasets are NOT kept (the file is closed): only shapes
    and positions are available through the regions' data=np.empty(shape)."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - requirement since 2.8.0
        raise ImportError("Reading .mapx maps needs the h5py package (pip install h5py).") from exc
    regions: List[MapRegion] = []
    with h5py.File(path, "r") as h:
        info = h["FileInfo"]
        attrs = {k: _decode_attr(v) for k, v in info.attrs.items()}
        start = float(attrs.get("SpectralRangeStart", float("nan")))
        end = float(attrs.get("SpectralRangeEnd", float("nan")))
        unit_m = float(attrs.get("SpatialUnitOfMeter", 1e-9) or 1e-9)
        to_um = unit_m / 1e-6
        meta: Dict[str, Any] = {
            "spectral_type": attrs.get("SpectralType", ""), "spectral_unit": attrs.get("SpectralShortLabel", "cm-1"),
            "amplitude": attrs.get("AmplitudeLongLabel", ""), "label": attrs.get("Label", ""),
            "n_rois": int(attrs.get("NumberROIs", 0) or 0), "guid": attrs.get("GUID", ""),
            "spectral_range": (start, end), "spatial_unit_m": unit_m,
        }
        if "MetaData" in info:
            raw = bytes(info["MetaData"][...])
            xml = raw.decode("utf-16-le", errors="ignore") if raw[1:2] == b"\0" else raw.decode("utf-8", errors="ignore")
            meta.update(_parse_metadata_xml(xml))
        npts = None
        if "Regions" in h:
            for name in sorted(h["Regions"], key=lambda s: (len(s), s)):
                g = h["Regions"][name]
                if "Dataset" not in g:
                    continue
                ds = g["Dataset"]
                if ds.ndim != 3:
                    continue
                data = ds[...] if load_data else np.empty(ds.shape, dtype=np.float32)
                npts = int(ds.shape[-1])
                box = g["-StagePosXYZ"][...] if "-StagePosXYZ" in g else np.array([0, 0, 0, ds.shape[0], ds.shape[1], 0], float)
                xmin, ymin, xmax, ymax = float(box[0]) * to_um, float(box[1]) * to_um, float(box[3]) * to_um, float(box[4]) * to_um
                nx, ny = int(ds.shape[0]), int(ds.shape[1])
                # pixel centres across the box (one pixel: the box centre)
                xs = np.linspace(xmin, xmax, nx) if nx > 1 else np.array([(xmin + xmax) / 2])
                ys = np.linspace(ymin, ymax, ny) if ny > 1 else np.array([(ymin + ymax) / 2])
                regions.append(MapRegion(name=name, data=data, x_positions=xs, y_positions=ys,
                                         stage_box=(xmin, ymin, xmax, ymax)))
    if not regions:
        raise ValueError(f"{os.path.basename(path)}: this map holds no spectra ({meta.get('n_rois', 0)} regions of "
                         "interest, camera mosaic only) — an acquisition that was set up but never collected.")
    if not (np.isfinite(start) and np.isfinite(end)) or npts is None:
        raise ValueError(f"{os.path.basename(path)}: the spectral range is missing from FileInfo.")
    x = np.linspace(start, end, npts)
    if x[0] > x[-1]:
        x = x[::-1]
        for r in regions:
            r.data = r.data[..., ::-1]
    return OmnicMap(path=path, x=x, regions=regions, meta=meta)


def mapx_summary(m: OmnicMap) -> str:
    parts = [f"{len(m.regions)} region(s), {m.n_spectra} spectra, {len(m.x)} points "
             f"{m.x[0]:.1f}-{m.x[-1]:.1f} {m.meta.get('spectral_unit', 'cm-1')}"]
    for key, label in (("laser_mW", "laser mW"), ("n_scans", "scans"), ("exposure_s", "exposure s"), ("objective", "objective")):
        if key in m.meta:
            parts.append(f"{label} {m.meta[key]}")
    return ", ".join(parts)
