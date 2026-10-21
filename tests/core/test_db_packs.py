"""core.db_packs — database pack export / import on temp caches (RRUFF,
AMCSD and the Raman library; XRD databases and CIF folders through the
registry)."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

import core.db_packs as dbp


def _fake_rruff(root):
    """A two-spectrum RRUFF cache in rruff_science's layout."""
    raw = root / "raw"
    raw.mkdir(parents=True)
    index = []
    for name, rid in (("Quartz", "R040031"), ("Calcite", "R040070")):
        fn = f"{name}__{rid}__Raman__532__0__unoriented__Raman_Data_Processed__12345.txt"
        lines = [f"##NAMES={name}", f"##RRUFFID={rid}", "##LASER_WAVELENGTH=532", "##DATA_KIND=Processed"]
        lines += [f"{100 + i * 10}, {1000 if i == 20 else 10}" for i in range(100)]
        (raw / fn).write_text("\n".join(lines), encoding="utf-8")
        index.append({"mineral": name, "rruff_id": rid, "wavelength_nm": 532.0, "peaks": [300.0],
                      "raw_path": str(raw / fn), "category": "excellent_unoriented", "data_kind": "Processed"})
    (root / "index.json").write_text(json.dumps(index), encoding="utf-8")


def _fake_amcsd(root):
    cif = root / "cif"
    cif.mkdir(parents=True)
    (cif / "Quartz__0000789.cif").write_text("data_quartz\n_cell_length_a 4.913\n", encoding="utf-8")
    (root / "cif_index.json").write_text(json.dumps({"quartz": ["Quartz__0000789.cif"]}), encoding="utf-8")


def test_inventory_reports_installed_and_missing(tmp_path):
    rruff = tmp_path / "rruff"; amcsd = tmp_path / "amcsd"
    _fake_rruff(rruff)
    items = {it.key: it for it in dbp.inventory(rruff_dir=str(rruff), amcsd_dir=str(amcsd),
                                                registry_path=str(tmp_path / "reg.json"),
                                                raman_library_path=str(tmp_path / "lib.json"))}
    assert set(items) == set(dbp.ALL_KEYS)
    assert items["rruff"].installed and "2 spectra" in items["rruff"].detail
    assert not items["amcsd"].installed and not items["xrd"].installed
    assert not items["raman_library"].installed


def test_export_import_round_trip(tmp_path):
    src_rruff = tmp_path / "src" / "rruff"; src_amcsd = tmp_path / "src" / "amcsd"
    _fake_rruff(src_rruff); _fake_amcsd(src_amcsd)
    import raman.raman_library as rl
    lib_src = str(tmp_path / "src" / "library.json")
    x = np.linspace(100, 1000, 91)
    rl.save_library([rl.entry_from_spectrum("standard A", x, np.exp(-0.5 * ((x - 500) / 20) ** 2))], lib_src)

    pack = tmp_path / "pack"
    log = []
    manifest = dbp.export_pack(str(pack), log=log.append, rruff_dir=str(src_rruff), amcsd_dir=str(src_amcsd),
                               registry_path=str(tmp_path / "src" / "reg.json"), raman_library_path=lib_src)
    assert set(manifest["items"]) == {"rruff", "amcsd", "raman_library"}  # nothing XRD registered -> skipped, not failed
    assert (pack / dbp.MANIFEST_NAME).is_file() and (pack / "rruff_pack.sq").is_file() and (pack / "amcsd_pack.zip").is_file()
    assert any("skipped" in m for m in log)

    dst = tmp_path / "dst"
    done = dbp.import_pack(str(pack), log=log.append, rruff_dir=str(dst / "rruff"), amcsd_dir=str(dst / "amcsd"),
                           registry_path=str(dst / "reg.json"), import_dir=str(dst / "imported"),
                           raman_library_path=str(dst / "library.json"))
    assert set(done) == {"rruff", "amcsd", "raman_library"}
    # RRUFF: index rewritten to the new location, raw files present and searchable
    from raman.rruff_science import load_index
    idx = load_index(cache_dir=str(dst / "rruff"))
    assert [r["mineral"] for r in idx] == ["Quartz", "Calcite"]
    assert all(os.path.isfile(r["raw_path"]) and r["raw_path"].startswith(str(dst / "rruff")) for r in idx)
    # AMCSD: the index and the CIF land where find_cifs_for_mineral looks
    from raman.rruff_science import find_cifs_for_mineral
    assert len(find_cifs_for_mineral("Quartz", cache_dir=str(dst / "amcsd"))) == 1
    # Raman library merged, and a second import does not duplicate
    assert len(rl.load_library(str(dst / "library.json"))) == 1
    dbp.import_pack(str(pack), keys=["raman_library"], raman_library_path=str(dst / "library.json"),
                    registry_path=str(dst / "reg.json"))
    assert len(rl.load_library(str(dst / "library.json"))) == 1
    inv = {it.key: it for it in dbp.inventory(rruff_dir=str(dst / "rruff"), amcsd_dir=str(dst / "amcsd"),
                                              registry_path=str(dst / "reg.json"),
                                              raman_library_path=str(dst / "library.json"))}
    assert inv["rruff"].installed and inv["amcsd"].installed and inv["raman_library"].installed


def test_manifest_validation(tmp_path):
    with pytest.raises(ValueError):
        dbp.read_pack_manifest(str(tmp_path))
    (tmp_path / dbp.MANIFEST_NAME).write_text(json.dumps({"format": "other"}), encoding="utf-8")
    with pytest.raises(ValueError):
        dbp.read_pack_manifest(str(tmp_path))
    (tmp_path / dbp.MANIFEST_NAME).write_text(json.dumps({"format": dbp.PACK_FORMAT, "version": 99, "items": {}}), encoding="utf-8")
    with pytest.raises(ValueError):
        dbp.read_pack_manifest(str(tmp_path))
    (tmp_path / dbp.MANIFEST_NAME).write_text(json.dumps({"format": dbp.PACK_FORMAT, "version": 1, "items": {}}), encoding="utf-8")
    m = dbp.read_pack_manifest(str(tmp_path / dbp.MANIFEST_NAME))  # the manifest file itself is accepted too
    assert m["_dir"] == str(tmp_path)
