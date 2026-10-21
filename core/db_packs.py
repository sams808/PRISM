"""
db_packs.py — export and import of the reference databases PRISM keeps in
its per-user cache (framework-agnostic, no Qt).

Why: a colleague with the same database rights (another PC of the group,
the Raman spectrometer's computer) should receive the databases as one
folder PRISM reads directly, instead of re-downloading hundreds of
megabytes or re-converting a card database. A pack is a plain folder:

    prism_databases.json         manifest: what the pack holds, versions, dates
    rruff_pack.sq                RRUFF Raman spectra + index (one SQLite file)
    amcsd_pack.zip               AMCSD CIF structures + their index
    xrd/<slug>.sq                registered XRD card databases (PRISM format)
    cif_folders/<slug>.zip       registered CIF folders (the structure files;
                                 their patterns are recalculated on import)
    raman_library.json           the user's own Raman reference library

Everything lands where the app expects it on import (the same cache
layout the download buttons produce), so nothing else has to change on
the receiving machine. Respect the licence of every database: RRUFF and
AMCSD are redistributable with attribution, a licensed card database only
to people covered by the same rights.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
import zipfile
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence

MANIFEST_NAME = "prism_databases.json"
PACK_FORMAT = "prism-databases"
PACK_VERSION = 1

KEY_RRUFF = "rruff"
KEY_AMCSD = "amcsd"
KEY_XRD = "xrd"
KEY_CIF_FOLDERS = "cif_folders"
KEY_RAMAN_LIBRARY = "raman_library"
ALL_KEYS = (KEY_RRUFF, KEY_AMCSD, KEY_XRD, KEY_CIF_FOLDERS, KEY_RAMAN_LIBRARY)

LABELS = {
    KEY_RRUFF: "RRUFF Raman spectra",
    KEY_AMCSD: "AMCSD crystal structures",
    KEY_XRD: "XRD card databases",
    KEY_CIF_FOLDERS: "CIF structure folders",
    KEY_RAMAN_LIBRARY: "Raman reference library (your own spectra)",
}


@dataclass
class DbItem:
    key: str
    label: str
    installed: bool
    detail: str
    size_bytes: int = 0
    entries: Optional[List[Dict[str, Any]]] = None  # per-database rows (XRD / CIF folders)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _log_noop(msg: str) -> None:
    pass


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "db"


def _dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _file_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


# ----------------------------------------------------------------------
# Paths (resolved at call time so tests can point them at temp folders)
# ----------------------------------------------------------------------

def _rruff_dir(rruff_dir: Optional[str]) -> str:
    if rruff_dir:
        return rruff_dir
    from raman.rruff_science import RRUFF_CACHE_DIR
    return RRUFF_CACHE_DIR


def _amcsd_dir(amcsd_dir: Optional[str]) -> str:
    if amcsd_dir:
        return amcsd_dir
    from raman.rruff_science import AMCSD_CACHE_DIR
    return AMCSD_CACHE_DIR


def _raman_library_path(path: Optional[str]) -> str:
    if path:
        return path
    from raman.raman_library import LIBRARY_PATH
    return LIBRARY_PATH


# ----------------------------------------------------------------------
# Inventory
# ----------------------------------------------------------------------

def inventory(*, rruff_dir: Optional[str] = None, amcsd_dir: Optional[str] = None,
              registry_path: Optional[str] = None, raman_library_path: Optional[str] = None,
              sizes: bool = False) -> List[DbItem]:
    """What is installed on this machine. `sizes=True` walks the cache
    folders (the RRUFF raw folder holds ~38k files: a second or two)."""
    items: List[DbItem] = []

    rd = _rruff_dir(rruff_dir)
    index_path = os.path.join(rd, "index.json")
    if os.path.isfile(index_path):
        try:
            with open(index_path, encoding="utf-8") as f:
                n = len(json.load(f))
            detail = f"{n} spectra indexed"
        except (OSError, ValueError):
            detail = "index unreadable"
        items.append(DbItem(KEY_RRUFF, LABELS[KEY_RRUFF], True, detail,
                            _dir_size(os.path.join(rd, "raw")) + _file_size(index_path) if sizes else 0))
    else:
        items.append(DbItem(KEY_RRUFF, LABELS[KEY_RRUFF], False, "not installed"))

    ad = _amcsd_dir(amcsd_dir)
    cif_index = os.path.join(ad, "cif_index.json")
    if os.path.isfile(cif_index):
        try:
            with open(cif_index, encoding="utf-8") as f:
                idx = json.load(f)
            n = sum(len(v) for v in idx.values())
            detail = f"{n} structures ({len(idx)} minerals)"
        except (OSError, ValueError):
            detail = "index unreadable"
        items.append(DbItem(KEY_AMCSD, LABELS[KEY_AMCSD], True, detail,
                            _dir_size(os.path.join(ad, "cif")) + _file_size(cif_index) if sizes else 0))
    else:
        items.append(DbItem(KEY_AMCSD, LABELS[KEY_AMCSD], False, "not installed"))

    import xrd.xrd_id_science as xid
    entries = xid.load_registry(registry_path)
    cards = [e for e in entries if e.get("kind") != "cif"]
    cifs = [e for e in entries if e.get("kind") == "cif"]
    present = [e for e in cards if os.path.isfile(e.get("path", ""))]
    items.append(DbItem(
        KEY_XRD, LABELS[KEY_XRD], bool(present),
        (", ".join(e["name"] for e in present) if present else "none registered"),
        sum(_file_size(e["path"]) for e in present) if sizes else 0,
        entries=[{"name": e["name"], "path": e.get("path", ""), "enabled": bool(e.get("enabled", True)),
                  "present": os.path.isfile(e.get("path", ""))} for e in cards]))
    present_cif = [e for e in cifs if os.path.isdir(e.get("cif_folder", ""))]
    items.append(DbItem(
        KEY_CIF_FOLDERS, LABELS[KEY_CIF_FOLDERS], bool(present_cif),
        (", ".join(f"{e['name']} ({e.get('ncards', '?')} structures)" for e in present_cif)
         if present_cif else "none registered"),
        sum(_dir_size(e["cif_folder"]) for e in present_cif) if sizes else 0,
        entries=[{"name": e["name"], "folder": e.get("cif_folder", ""), "enabled": bool(e.get("enabled", True)),
                  "wavelength": float(e.get("wavelength", 1.5406)), "recursive": bool(e.get("recursive", True)),
                  "present": os.path.isdir(e.get("cif_folder", ""))} for e in cifs]))

    lp = _raman_library_path(raman_library_path)
    if os.path.isfile(lp):
        try:
            from raman.raman_library import load_library
            n = len(load_library(lp))
            detail = f"{n} reference spectra"
        except Exception:
            detail = "library unreadable"
        items.append(DbItem(KEY_RAMAN_LIBRARY, LABELS[KEY_RAMAN_LIBRARY], True, detail, _file_size(lp) if sizes else 0))
    else:
        items.append(DbItem(KEY_RAMAN_LIBRARY, LABELS[KEY_RAMAN_LIBRARY], False, "empty"))
    return items


# ----------------------------------------------------------------------
# Export
# ----------------------------------------------------------------------

def _zip_dir(folder: str, out_zip: str, *, arc_root: str = "", log: Callable[[str], None] = _log_noop,
             progress: Optional[Callable[[int, int, str], None]] = None) -> int:
    files = []
    for root, _dirs, names in os.walk(folder):
        for n in names:
            files.append(os.path.join(root, n))
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i, p in enumerate(files):
            rel = os.path.relpath(p, folder)
            zf.write(p, os.path.join(arc_root, rel) if arc_root else rel)
            if progress and (i % 2000 == 0 or i == len(files) - 1):
                progress(i + 1, len(files), os.path.basename(out_zip))
    return len(files)


def export_pack(out_dir: str, keys: Optional[Sequence[str]] = None, *,
                log: Callable[[str], None] = _log_noop,
                progress: Optional[Callable[[int, int, str], None]] = None,
                rruff_dir: Optional[str] = None, amcsd_dir: Optional[str] = None,
                registry_path: Optional[str] = None, raman_library_path: Optional[str] = None) -> Dict[str, Any]:
    """Write the selected databases into out_dir as a pack. Returns the
    manifest. Databases that are not installed are skipped with a log line,
    never an error, so one export call can be 'everything I have'."""
    keys = list(keys) if keys else list(ALL_KEYS)
    os.makedirs(out_dir, exist_ok=True)
    manifest: Dict[str, Any] = {
        "format": PACK_FORMAT, "version": PACK_VERSION,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "items": {},
    }
    inv = {it.key: it for it in inventory(rruff_dir=rruff_dir, amcsd_dir=amcsd_dir,
                                          registry_path=registry_path, raman_library_path=raman_library_path)}

    if KEY_RRUFF in keys:
        it = inv[KEY_RRUFF]
        if it.installed:
            from raman.rruff_science import pack_rruff_database
            log("Packing the RRUFF database (one SQLite file)…")
            out = pack_rruff_database(_rruff_dir(rruff_dir), out_path=os.path.join(out_dir, "rruff_pack.sq"))
            manifest["items"][KEY_RRUFF] = {"file": os.path.basename(out), "detail": it.detail}
            log(f"  RRUFF: {it.detail} -> {os.path.basename(out)}")
        else:
            log("RRUFF database not installed here: skipped.")

    if KEY_AMCSD in keys:
        it = inv[KEY_AMCSD]
        if it.installed:
            ad = _amcsd_dir(amcsd_dir)
            out = os.path.join(out_dir, "amcsd_pack.zip")
            log("Packing the AMCSD structures…")
            with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.write(os.path.join(ad, "cif_index.json"), "cif_index.json")
            # append the CIF folder (one zip, index at the root, files under cif/)
            cif_dir = os.path.join(ad, "cif")
            files = sorted(os.listdir(cif_dir)) if os.path.isdir(cif_dir) else []
            with zipfile.ZipFile(out, "a", compression=zipfile.ZIP_DEFLATED) as zf:
                for i, n in enumerate(files):
                    zf.write(os.path.join(cif_dir, n), "cif/" + n)
                    if progress and (i % 2000 == 0 or i == len(files) - 1):
                        progress(i + 1, len(files), "amcsd_pack.zip")
            manifest["items"][KEY_AMCSD] = {"file": "amcsd_pack.zip", "detail": it.detail}
            log(f"  AMCSD: {it.detail} -> amcsd_pack.zip")
        else:
            log("AMCSD structures not installed here: skipped.")

    if KEY_XRD in keys:
        it = inv[KEY_XRD]
        rows = []
        for e in (it.entries or []):
            if not e.get("present"):
                log(f"  XRD database '{e['name']}': file missing, skipped.")
                continue
            sub = os.path.join(out_dir, "xrd")
            os.makedirs(sub, exist_ok=True)
            fname = _slug(e["name"]) + ".sq"
            log(f"Copying XRD database '{e['name']}' ({_file_size(e['path']) / 1e6:.0f} MB)…")
            shutil.copyfile(e["path"], os.path.join(sub, fname))
            rows.append({"name": e["name"], "file": "xrd/" + fname, "enabled": e.get("enabled", True)})
        if rows:
            manifest["items"][KEY_XRD] = {"databases": rows}
        else:
            log("No XRD card database registered here: skipped.")

    if KEY_CIF_FOLDERS in keys:
        it = inv[KEY_CIF_FOLDERS]
        rows = []
        for e in (it.entries or []):
            if not e.get("present"):
                log(f"  CIF folder '{e['name']}': folder missing, skipped.")
                continue
            sub = os.path.join(out_dir, "cif_folders")
            os.makedirs(sub, exist_ok=True)
            fname = _slug(e["name"]) + ".zip"
            log(f"Zipping CIF folder '{e['name']}'…")
            n = _zip_dir(e["folder"], os.path.join(sub, fname), log=log, progress=progress)
            rows.append({"name": e["name"], "file": "cif_folders/" + fname, "enabled": e.get("enabled", True),
                         "wavelength": e.get("wavelength", 1.5406), "recursive": e.get("recursive", True),
                         "n_files": n})
        if rows:
            manifest["items"][KEY_CIF_FOLDERS] = {"folders": rows}
        else:
            log("No CIF folder registered here: skipped.")

    if KEY_RAMAN_LIBRARY in keys:
        it = inv[KEY_RAMAN_LIBRARY]
        if it.installed:
            shutil.copyfile(_raman_library_path(raman_library_path), os.path.join(out_dir, "raman_library.json"))
            manifest["items"][KEY_RAMAN_LIBRARY] = {"file": "raman_library.json", "detail": it.detail}
            log(f"  Raman reference library: {it.detail} -> raman_library.json")
        else:
            log("Raman reference library is empty: skipped.")

    with open(os.path.join(out_dir, MANIFEST_NAME), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    log(f"Pack written to {out_dir} ({len(manifest['items'])} item(s)).")
    return manifest


# ----------------------------------------------------------------------
# Import
# ----------------------------------------------------------------------

def read_pack_manifest(src_dir: str) -> Dict[str, Any]:
    """The manifest of a pack folder (or the folder a manifest file was
    picked from). Raises ValueError when it is not a PRISM pack."""
    if os.path.isfile(src_dir) and os.path.basename(src_dir) == MANIFEST_NAME:
        src_dir = os.path.dirname(src_dir)
    p = os.path.join(src_dir, MANIFEST_NAME)
    if not os.path.isfile(p):
        raise ValueError(f"No {MANIFEST_NAME} in {src_dir}: not a PRISM database pack.")
    with open(p, encoding="utf-8") as f:
        m = json.load(f)
    if m.get("format") != PACK_FORMAT:
        raise ValueError(f"{p} is not a PRISM database pack (format {m.get('format')!r}).")
    if int(m.get("version", 0)) > PACK_VERSION:
        raise ValueError(f"Pack version {m.get('version')} is newer than this PRISM reads ({PACK_VERSION}).")
    m["_dir"] = src_dir
    return m


def import_pack(src_dir: str, keys: Optional[Sequence[str]] = None, *,
                log: Callable[[str], None] = _log_noop,
                progress: Optional[Callable[[int, int, str], None]] = None,
                rruff_dir: Optional[str] = None, amcsd_dir: Optional[str] = None,
                registry_path: Optional[str] = None, import_dir: Optional[str] = None,
                raman_library_path: Optional[str] = None) -> Dict[str, Any]:
    """Install the databases of a pack into this machine's cache. Returns
    {key: summary}. Items absent from the pack are skipped."""
    m = read_pack_manifest(src_dir)
    src_dir = m["_dir"]
    items = m.get("items", {})
    keys = list(keys) if keys else list(items.keys())
    done: Dict[str, Any] = {}

    if KEY_RRUFF in keys and KEY_RRUFF in items:
        from raman.rruff_science import unpack_rruff_database
        sq = os.path.join(src_dir, items[KEY_RRUFF]["file"])
        log("Installing the RRUFF database…")
        n = unpack_rruff_database(sq, cache_dir=_rruff_dir(rruff_dir))
        done[KEY_RRUFF] = f"{n} spectra installed"
        log(f"  RRUFF: {n} spectra.")

    if KEY_AMCSD in keys and KEY_AMCSD in items:
        ad = _amcsd_dir(amcsd_dir)
        os.makedirs(ad, exist_ok=True)
        z = os.path.join(src_dir, items[KEY_AMCSD]["file"])
        log("Installing the AMCSD structures…")
        with zipfile.ZipFile(z) as zf:
            names = zf.namelist()
            for i, n in enumerate(names):
                # only the two things the pack writes: cif_index.json and cif/<file>
                if n == "cif_index.json" or (n.startswith("cif/") and "/" not in n[4:] and ".." not in n):
                    zf.extract(n, ad)
                if progress and (i % 2000 == 0 or i == len(names) - 1):
                    progress(i + 1, len(names), "AMCSD")
        done[KEY_AMCSD] = f"{sum(1 for n in names if n.startswith('cif/'))} structures installed"
        log(f"  AMCSD: {done[KEY_AMCSD]}.")

    if KEY_XRD in keys and KEY_XRD in items:
        import xrd.xrd_id_science as xid
        imp = import_dir or xid.XRD_ID_IMPORT_DIR
        os.makedirs(imp, exist_ok=True)
        names = []
        for row in items[KEY_XRD].get("databases", []):
            src = os.path.join(src_dir, row["file"])
            dst = os.path.join(imp, os.path.basename(row["file"]))
            log(f"Installing XRD database '{row['name']}'…")
            if os.path.abspath(src) != os.path.abspath(dst):
                shutil.copyfile(src, dst)
            entry = xid.register_database(dst, row["name"], registry_path=registry_path, import_dir=imp, log=log)
            if not row.get("enabled", True):
                entries = xid.load_registry(registry_path)
                for e in entries:
                    if e.get("path") == entry.get("path"):
                        e["enabled"] = False
                xid.save_registry(entries, registry_path)
            names.append(entry["name"])
        done[KEY_XRD] = f"{len(names)} database(s) registered: " + ", ".join(names)

    if KEY_CIF_FOLDERS in keys and KEY_CIF_FOLDERS in items:
        import xrd.xrd_id_science as xid
        imp = import_dir or xid.XRD_ID_IMPORT_DIR
        base = os.path.join(imp, "cif_sources")
        names = []
        for row in items[KEY_CIF_FOLDERS].get("folders", []):
            z = os.path.join(src_dir, row["file"])
            folder = os.path.join(base, _slug(row["name"]))
            os.makedirs(folder, exist_ok=True)
            log(f"Extracting CIF folder '{row['name']}'…")
            with zipfile.ZipFile(z) as zf:
                for n in zf.namelist():
                    if ".." in n or n.startswith(("/", "\\")):
                        continue
                    zf.extract(n, folder)
            entry = xid.register_cif_folder(folder, row["name"], registry_path=registry_path, import_dir=imp,
                                            wavelength=float(row.get("wavelength", 1.5406)),
                                            recursive=bool(row.get("recursive", True)), log=log)
            names.append(f"{entry['name']} ({entry.get('ncards', '?')} structures)")
        done[KEY_CIF_FOLDERS] = f"{len(names)} folder(s) registered: " + ", ".join(names)

    if KEY_RAMAN_LIBRARY in keys and KEY_RAMAN_LIBRARY in items:
        from raman.raman_library import load_library, save_library, merge_entries
        src = os.path.join(src_dir, items[KEY_RAMAN_LIBRARY]["file"])
        lp = _raman_library_path(raman_library_path)
        incoming = load_library(src)
        merged, n_new = merge_entries(load_library(lp) if os.path.isfile(lp) else [], incoming)
        save_library(merged, lp)
        done[KEY_RAMAN_LIBRARY] = f"{n_new} new reference spectra ({len(merged)} total)"
        log(f"  Raman reference library: {done[KEY_RAMAN_LIBRARY]}.")

    log(f"Import finished: {len(done)} item(s).")
    return done
