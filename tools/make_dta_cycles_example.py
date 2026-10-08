"""
make_dta_cycles_example.py — regenerate the committed cyclic-DTA fixtures:
  EXAMPLES/DTA_cycles_example.txt  (decimated copy of a real TA SDT export)
  EXAMPLES/DTA_cycles_example.mth  (byte-verbatim copy of the TA method file)

The source is a real Q600/SDT run, `DTA/BM/BM-0-cycles.txt`: 18996 data rows,
8 heating ramps + 8 programmed isotherms, 890 min, four heat/cool cycles. That
file is 2.5 MB of UTF-16 and cannot be committed, so the repo carries every 4th
row of it, re-encoded UTF-8-with-BOM to match EXAMPLES/DTA_example.txt.

    py -3.11 tools/make_dta_cycles_example.py [BM-0-cycles.txt] [--every 4]

Two rules this script exists to enforce:

1. THE HEADER IS COPIED VERBATIM, through and including the `StartOfData`
   line. Not re-emitted, not normalised, not de-duplicated. The whole point of
   this fixture is the 62-line header: 19 repeated `OrgMethod` lines (the
   thermal program), 3 each of `Xcomment` / `InstCalDate` / `InstCalFile`, and
   `Sig1..Sig7` with a U+00B0 degree sign in two of the column names. Those
   repeated keys are what the program parser reads and what the header-parsing
   tests pin; rewriting any of them would make the fixture test a fiction.

2. DO NOT RAISE `--every` CASUALLY. Decimation is not free here: it sets the
   sampling interval that the segmentation rate estimator sees, and that
   estimator falls off a cliff. Measured on this run (12 true segments:
   4 heats, 4 cools, 4 high-temperature holds):

       --every 4  -> dt 0.1875 min -> 12 segments   (committed fixture)
       --every 5  -> dt 0.2343 min -> 12 segments
       --every 6  -> dt 0.2811 min -> 72 segments, rates to -357 C/min
       --every 8  -> dt 0.3749 min -> 110 segments

   Cause: with ~2 points inside the +/-0.25 min rate window the least-squares
   denominator is small but not zero, so the "slope" is a two-point secant over
   noise. `estimate_rate`'s `rate_min_points >= 5` guard (which widens the
   window in INDEX space) removes the cliff entirely — with it, factors 2..16
   all give the same 12 segments. Factor 4 is only 1.5x away from the factor-6
   cliff, so raising this number tests the guard rather than the science. If
   you change it, re-run tests/dta/test_dta_cycles_fixture.py, which pins the
   fixture's row count and byte size, and expect it to fail loudly.

Why 4 and not 3 or 5: factor 3 is 414 KiB (too heavy for the repo), factor 5
keeps 20 % fewer points for the same order of magnitude of weight, and
18996 = 4 x 4749 exactly, so every stride group is full and the row count is
exact rather than ragged. Note this does NOT mean the fixture ends on the real
final row: `[::4]` keeps indices 0, 4, ... 18992, so the fixture stops at
t = 890.2292 min, 0.1416 min short of the real 890.3708 min. Tests that pin the
end of the run must use the fixture's own value.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

# this file lives in tools/, so EXAMPLES/ is one level up at repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "EXAMPLES"

DEFAULT_EVERY = 4
DEFAULT_TXT_NAME = "DTA_cycles_example.txt"
DEFAULT_MTH_NAME = "DTA_cycles_example.mth"
UTF8_BOM = b"\xef\xbb\xbf"


def default_bm_dir() -> Path:
    """Where the real (uncommitted) BM DTA folder lives on this machine.

    Mirrors `_find_dta_bm_dir` in tests/conftest.py, including the
    PRISM_DTA_BM_DIR override, so the generator and the tests that compare
    the fixture back against reality never disagree about the source.
    """
    override = os.environ.get("PRISM_DTA_BM_DIR")
    if override:
        return Path(override)
    return REPO_ROOT.parents[2] / "DTA" / "BM"


def split_header_and_data(text: str) -> tuple[list[str], list[str]]:
    """Split a TA SDT export into (header-through-StartOfData, data rows).

    The `StartOfData` marker is matched whitespace-insensitively and
    case-insensitively, exactly the way `core.io_universal.parse_ta_sdt_txt`
    matches it — if the two ever disagreed, the fixture would be split at a
    line the parser does not consider the boundary. Blank lines after the
    marker are dropped so that `[::every]` counts real rows only; a blank row
    swallowed into the stride would silently shift the decimation.
    """
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if "startofdata" in re.sub(r"\s+", "", line).lower():
            start = i
            break
    if start is None:
        raise SystemExit("no StartOfData marker found - is this a TA SDT text export?")
    header = lines[: start + 1]
    data = [ln for ln in lines[start + 1:] if ln.strip()]
    return header, data


def build_fixture_bytes(source: Path, every: int) -> tuple[bytes, list[str], list[str]]:
    """Read the real export and return the fixture's exact bytes.

    Returns `(payload, header, kept_rows)` so the caller can report the counts
    it just wrote instead of re-reading and re-deriving them.

    Encoding: the source is UTF-16 (BOM `ff fe`); the output is UTF-8 with a
    BOM and CRLF line endings, byte-for-byte the same opening as the existing
    `EXAMPLES/DTA_example.txt` (`ef bb bf 43 4c 4f 53 45 44 0d 0a`). Both are
    accepted by `_decode_text_autodetect`, but matching the existing example
    keeps one encoding story for everything in EXAMPLES/, and UTF-8 halves the
    committed size of a pure-ASCII-plus-degree-sign file.
    """
    if every < 1:
        raise SystemExit(f"--every must be >= 1, got {every}")
    text = source.read_bytes().decode("utf-16")
    header, data = split_header_and_data(text)
    kept = data[::every]
    body = "\r\n".join(header + kept) + "\r\n"
    return UTF8_BOM + body.encode("utf-8"), header, kept


def copy_method_file(source_mth: Path, dest: Path) -> bool:
    """Copy the TA `.mth` binary method file byte-for-byte.

    `shutil.copyfile`, never a text round-trip: the `.mth` is a binary record
    stream (24-byte records holding float32 rate/target arguments) and the
    fixture's only job is to be bit-identical to the instrument's own file, so
    that the reverse-engineered decoder is tested against real bytes. Returns
    False (with a warning) rather than failing the whole run when the source is
    absent — the text fixture is still worth regenerating on its own.
    """
    if not source_mth.is_file():
        print(f"warning: {source_mth} not found - .mth fixture left untouched", file=sys.stderr)
        return False
    shutil.copyfile(source_mth, dest)
    return True


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, write both fixtures, and report the verifiable numbers."""
    bm_dir = default_bm_dir()
    parser = argparse.ArgumentParser(
        description="Regenerate EXAMPLES/DTA_cycles_example.txt (+ .mth) from the real BM cyclic run.",
        epilog="Raising --every past 5 makes the run unsegmentable without the "
               "estimate_rate min-points guard; see this file's docstring.",
    )
    parser.add_argument(
        "source", nargs="?", type=Path, default=bm_dir / "BM-0-cycles.txt",
        help="the real UTF-16 TA SDT export (default: %(default)s)",
    )
    parser.add_argument(
        "--every", type=int, default=DEFAULT_EVERY, metavar="N",
        help="keep every Nth data row (default: %(default)s - read the docstring before changing)",
    )
    parser.add_argument(
        "--mth", type=Path, default=bm_dir / "cycles.mth",
        help="the TA binary method file to copy verbatim (default: %(default)s)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=EXAMPLES_DIR, metavar="DIR",
        help="where to write the fixtures (default: %(default)s)",
    )
    parser.add_argument(
        "--no-mth", action="store_true",
        help="write only the text fixture, leaving the committed .mth alone",
    )
    args = parser.parse_args(argv)

    if not args.source.is_file():
        print(f"error: source not found: {args.source}", file=sys.stderr)
        print("       pass the path explicitly, or set PRISM_DTA_BM_DIR.", file=sys.stderr)
        return 2

    payload, header, kept = build_fixture_bytes(args.source, args.every)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_txt = args.out_dir / DEFAULT_TXT_NAME
    out_txt.write_bytes(payload)

    n_org = sum(1 for ln in header if ln.startswith("OrgMethod\t"))
    n_sig = sum(1 for ln in header if re.match(r"^Sig\d+\t", ln))
    print(f"{out_txt}: {len(kept)} data rows (every {args.every}), "
          f"{len(header)} header lines, {len(payload)} bytes")
    print(f"  header carries {n_org} OrgMethod lines and {n_sig} signal definitions")

    if not args.no_mth:
        out_mth = args.out_dir / DEFAULT_MTH_NAME
        if copy_method_file(args.mth, out_mth):
            print(f"{out_mth}: {out_mth.stat().st_size} bytes (verbatim copy of {args.mth.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
