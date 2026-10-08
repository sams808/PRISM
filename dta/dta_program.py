"""TA Instruments temperature-program model: OrgMethod text + .mth binary.

One contract: bytes or header lines in, a MethodProgram out. Nothing here
touches measured data -- dta_segments.py maps a program onto a run. Pure
stdlib (re + struct): no numpy, no pandas, no Qt, so a headless batch script
or a test can import it for free.

Two properties are load-bearing, and both are pinned by tests:

* **Nothing in this module raises.** A truncated, empty, random or
  wrong-format .mth, a header with no OrgMethod key, a step line in a
  vocabulary we have never seen -- every one of them comes back as a
  MethodProgram carrying `status`, `warnings` and `error`. That is a product
  rule, not a style preference: the user asked that unusual runs still be
  processable, so a method file we cannot read must degrade the Program tab,
  never block loading the data next to it.
* **Nothing is silently dropped.** An unknown OrgMethod line or an unknown
  .mth opcode is kept in place, verbatim, flagged with
  `confidence != "parsed"`, and counted by `MethodProgram.n_unrecognized`.
  Dropping one would renumber every step after it and silently corrupt the
  program-to-data alignment, which is far worse than a visible
  "step 7 not understood" row.

The .mth format is REVERSE-ENGINEERED FROM ONE FILE (n = 1). Every program
decoded from binary carries `meta["format"] == MTH_FORMAT_TAG`, currently
"ta_mth_reverse_engineered_v1". That tag is pinned by a test so that the day
a second .mth from another module decodes wrongly, the fix is a v2 tag with
its own rules rather than a quiet mutation of what v1 means.
"""
from __future__ import annotations

import math
import re
import struct
from collections.abc import Mapping as _MappingABC
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# =============================================================================
# 1) Vocabulary
# =============================================================================

STEP_RAMP = "ramp"
STEP_ISOTHERMAL = "isothermal"
STEP_EQUILIBRATE = "equilibrate"
STEP_JUMP = "jump"
STEP_GAS = "gas"
STEP_FLOW = "flow"
STEP_OTHER = "other"          # recognized but not thermal: sampling, storage, abort...
STEP_REPEAT = "repeat"
STEP_END = "end_of_method"
STEP_UNKNOWN = "unknown"      # line/record not recognized, kept verbatim
STEP_INVALID = "invalid"      # binary record unreadable, kept verbatim

THERMAL_STEP_KINDS = frozenset({STEP_RAMP, STEP_ISOTHERMAL, STEP_EQUILIBRATE, STEP_JUMP})

PROGRAM_PARSED = "parsed"
PROGRAM_PARTIAL = "partial"
PROGRAM_UNRECOGNIZED = "unrecognized"

CONFIDENCE_PARSED = "parsed"
CONFIDENCE_UNKNOWN = "unknown"
CONFIDENCE_INVALID = "invalid"

SOURCE_ORGMETHOD = "orgmethod"
SOURCE_MTH = "mth"
SOURCE_BOTH = "both"          # merge_programs only: both sources carry this step
SOURCE_USER = "user"

ORGMETHOD_FORMAT_TAG = "ta_orgmethod_text_v1"
MTH_MAGIC = "TA Instruments Method File"
MTH_RECORD_SIZE = 24
MTH_FORMAT_TAG = "ta_mth_reverse_engineered_v1"
MTH_OPCODES: Dict[int, str] = {
    0x2645: STEP_GAS,
    0x26B8: STEP_FLOW,
    0x2635: STEP_RAMP,
    0x2625: STEP_ISOTHERMAL,
}

#: A lone 24-byte slot that happens to match the record shape is noise, not a
#: method: one occurs in the only real file we have (offset 1087 of
#: cycles.mth). A candidate table must therefore hold at least this many
#: records. The cost is that a genuine one-step .mth would read as
#: unrecognized -- a wrong answer we can see and fix, where a one-record
#: false positive would silently describe a furnace program that never ran.
MTH_MIN_RECORDS = 2

#: Consecutive unreadable slots a step table may be bridged across (see
#: _chain_from): a corrupt record in the middle of a table is still part of
#: that table.
MTH_DEFAULT_MAX_GAP = 3

#: Cap on the byte range swept for the step table. A .mth is a few KB; this
#: only stops a mis-picked 500 MB file from freezing the UI thread.
MTH_MAX_SCAN_BYTES = 4 * 1024 * 1024

#: U+00B0 DEGREE SIGN, written as an escape so this source file stays pure
#: ASCII and cannot be corrupted by a re-encoding tool between here and the
#: byte-identical round-trip test.
DEGREE_SIGN = "°"

NAN = float("nan")
_ZERO12 = b"\x00" * 12


# =============================================================================
# 2) Small numeric helpers
# =============================================================================

def _as_float(value: Any) -> float:
    """float(value) or NaN -- never raises.

    Unset numeric fields are NaN and never 0.0 throughout this module,
    because 0.0 is a legal ramp rate, a legal hold duration and a legal mass
    flow: using it as "missing" would make "Mass flow 0.00 mL/min" (the real
    step 19 of BM-0-cycles) indistinguishable from "no value recorded".
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return NAN


def _finite(value: Any) -> bool:
    """True when `value` is a real, finite number."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _fmt2(value: float) -> str:
    """TA's two-decimal number format; '?' when the value is unknown."""
    return f"{float(value):.2f}" if _finite(value) else "?"


def _fmt1(value: float) -> str:
    """One-decimal number for fingerprints; '?' when unknown."""
    return f"{float(value):.1f}" if _finite(value) else "?"


def _nan_to_none(value: float) -> Optional[float]:
    """NaN -> None for JSON.

    json.dumps writes a bare `NaN` token, which is not valid JSON and is
    rejected by strict readers; a .prism project must survive a round trip
    through any of them.
    """
    return float(value) if _finite(value) else None


# =============================================================================
# 3) Step / schedule / comparison records
# =============================================================================

@dataclass(frozen=True)
class MethodStep:
    """One method step. Unset numeric fields are NaN, never 0.

    `value` exists so that a quantity which is NOT a duration never lands in
    `duration_min`. The scratchpad prototype stored "Mass flow 100.00 mL/min"
    as duration_min=100.0, and any program clock summing duration_min then
    injects 100 phantom minutes into an 890-minute run. Gas index, mass flow
    and sampling interval therefore all live in `value`, and
    `MethodProgram.schedule()` budgets time only for THERMAL_STEP_KINDS.
    """

    index: int                           # 1-based WRITTEN step number, as the instrument prints it
    kind: str                            # STEP_*
    rate_C_min: float = NAN              # ramp
    target_C: float = NAN                # ramp / equilibrate / jump
    duration_min: float = NAN            # isothermal
    value: float = NAN                   # gas index (1-based) | flow mL/min | sampling s/pt
    text: str = ""                       # step text WITHOUT the "<n>: " numbering
    source: str = SOURCE_ORGMETHOD       # "orgmethod" | "mth" | "both" | "user"
    opcode: Optional[int] = None         # .mth only
    confidence: str = CONFIDENCE_PARSED  # "parsed" | "unknown" | "invalid"
    repeat_pass: int = 0                 # >0 on expanded repeats
    repeat_target: int = 0               # STEP_REPEAT: written index the block restarts at
    repeat_count: int = 0                # STEP_REPEAT: extra passes (block runs count+1 times)

    @property
    def is_thermal(self) -> bool:
        """True when this step moves or holds the furnace setpoint."""
        return self.kind in THERMAL_STEP_KINDS

    @property
    def is_recognized(self) -> bool:
        """True when the step text / binary record was understood."""
        return self.confidence == CONFIDENCE_PARSED

    def render(self) -> str:
        """Canonical TA step text, WITHOUT the leading '<n>: ' numbering.

        Canonical, not verbatim: a recognized step is re-rendered from its
        parsed numbers in TA's own format, so the same step decoded from the
        binary .mth and read from the OrgMethod header produce one identical
        string -- that equality is exactly what makes a .mth checkable
        against the run's own recorded method. The untouched original stays
        in `.text`, which is also what an unrecognized step renders as: a
        step we did not understand is echoed, never guessed at.
        """
        k = self.kind
        if k == STEP_RAMP:
            return (f"Ramp {_fmt2(self.rate_C_min)} {DEGREE_SIGN}C/min "
                    f"to {_fmt2(self.target_C)} {DEGREE_SIGN}C")
        if k == STEP_ISOTHERMAL:
            return f"Isothermal for {_fmt2(self.duration_min)} min"
        if k == STEP_EQUILIBRATE:
            return f"Equilibrate at {_fmt2(self.target_C)} {DEGREE_SIGN}C"
        if k == STEP_JUMP:
            return f"Jump to {_fmt2(self.target_C)} {DEGREE_SIGN}C"
        if k == STEP_FLOW:
            return f"Mass flow {_fmt2(self.value)} mL/min"
        if k == STEP_GAS:
            return f"Select gas {int(self.value)}" if _finite(self.value) else "Select gas ?"
        if k == STEP_REPEAT:
            return f"Repeat segment {self.repeat_target} {self.repeat_count} times"
        if k == STEP_END:
            return self.text or "End of method"
        return self.text

    def to_dict(self) -> dict:
        """JSON-safe dict (NaN -> None) for .prism persistence."""
        return {
            "index": int(self.index),
            "kind": str(self.kind),
            "rate_C_min": _nan_to_none(self.rate_C_min),
            "target_C": _nan_to_none(self.target_C),
            "duration_min": _nan_to_none(self.duration_min),
            "value": _nan_to_none(self.value),
            "text": str(self.text),
            "source": str(self.source),
            "opcode": None if self.opcode is None else int(self.opcode),
            "confidence": str(self.confidence),
            "repeat_pass": int(self.repeat_pass),
            "repeat_target": int(self.repeat_target),
            "repeat_count": int(self.repeat_count),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "MethodStep":
        """Rebuild from to_dict(). Never raises: bad fields fall back."""
        if not isinstance(d, _MappingABC):
            return cls(index=0, kind=STEP_UNKNOWN, text=str(d),
                       confidence=CONFIDENCE_INVALID)

        def _int(key: str, default: int = 0) -> int:
            try:
                return int(d.get(key, default))
            except (TypeError, ValueError):
                return default

        opcode = d.get("opcode", None)
        try:
            opcode = None if opcode is None else int(opcode)
        except (TypeError, ValueError):
            opcode = None
        return cls(
            index=_int("index"),
            kind=str(d.get("kind") or STEP_UNKNOWN),
            rate_C_min=_as_float(d.get("rate_C_min", NAN)),
            target_C=_as_float(d.get("target_C", NAN)),
            duration_min=_as_float(d.get("duration_min", NAN)),
            value=_as_float(d.get("value", NAN)),
            text=str(d.get("text") or ""),
            source=str(d.get("source") or SOURCE_ORGMETHOD),
            opcode=opcode,
            confidence=str(d.get("confidence") or CONFIDENCE_PARSED),
            repeat_pass=_int("repeat_pass"),
            repeat_target=_int("repeat_target"),
            repeat_count=_int("repeat_count"),
        )


@dataclass(frozen=True)
class ScheduleEntry:
    """One ideal setpoint window: what the controller intended.

    Non-thermal steps (gas, mass flow, sampling interval) still get an entry,
    with t_start == t_end, so that "step index -> schedule entry" stays a
    total mapping and no caller has to re-derive which steps cost no time.
    """

    step_index: int
    t_start: float
    t_end: float
    T_start: float
    T_end: float

    @property
    def duration_min(self) -> float:
        """Programmed length of this window, in minutes (0 for a gas step)."""
        return float(self.t_end) - float(self.t_start)


@dataclass(frozen=True)
class ProgramComparison:
    """Result of compare_programs(): are these two the same furnace program?"""

    verdict: str                        # "identical"|"equivalent"|"mismatch"|"incomparable"
    score: float                        # 0..1 of thermal steps agreeing
    rows: Tuple[Tuple[int, str, str, str], ...] = ()   # (n, a_text, b_text, tag)
    reasons: Tuple[str, ...] = ()

    @property
    def is_same_method(self) -> bool:
        """True only on a positive answer.

        "incomparable" (one side unreadable) is deliberately NOT "equivalent":
        a caller gating "does this .mth belong to this run?" must never be
        talked into yes by a file it could not read.
        """
        return self.verdict in ("identical", "equivalent")


# =============================================================================
# 4) The program
# =============================================================================

@dataclass(frozen=True)
class MethodProgram:
    """An ordered method: what the operator told the furnace to do.

    `status` is the honesty channel of the whole module:
      * "parsed"       every step understood;
      * "partial"      steps recovered, but at least one is unknown/invalid,
                       or the source itself was known to be incomplete;
      * "unrecognized" nothing usable -- `error` says why, `bool(program)`
                       is False, and every downstream consumer falls back to
                       data-only behaviour instead of failing.
    """

    steps: Tuple[MethodStep, ...] = ()
    name: str = ""
    source: str = ""                    # "orgmethod" | "mth" | "orgmethod+mth"
    source_path: Optional[str] = None
    status: str = PROGRAM_UNRECOGNIZED
    proc_name: Optional[str] = None     # header ProcName / .mth string @142; a WEAK identity signal
    warnings: Tuple[str, ...] = ()
    error: Optional[str] = None
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        """False when there is nothing trustworthy to show."""
        return bool(self.steps) and self.status != PROGRAM_UNRECOGNIZED

    def __len__(self) -> int:
        return len(self.steps)

    @property
    def thermal_steps(self) -> Tuple[MethodStep, ...]:
        """Written steps that move or hold the setpoint (no gas/flow/other)."""
        return tuple(s for s in self.steps if s.is_thermal)

    @property
    def ramp_rates(self) -> Tuple[float, ...]:
        """|rate| of every non-zero ramp, in written order.

        The absolute value is deliberate: TA writes a cooling ramp as a
        positive rate with a lower target, so the sign carries no information
        and a signed list would sort and group wrongly in the rate series.
        """
        return tuple(abs(s.rate_C_min) for s in self.steps
                     if s.kind == STEP_RAMP and _finite(s.rate_C_min) and s.rate_C_min != 0.0)

    @property
    def n_unrecognized(self) -> int:
        """How many steps were kept but not understood."""
        return sum(1 for s in self.steps if not s.is_recognized)

    def physical_steps(self, *, max_steps: int = 2000
                       ) -> Tuple[Tuple[MethodStep, ...], Tuple[str, ...]]:
        """Written steps with `Repeat segment` unrolled, plus expansion notes.

        Everything that has to reason about what the furnace ACTUALLY did --
        the schedule, the fingerprint, any alignment onto measured data --
        goes through here rather than through `.steps`, because with a repeat
        in the method the written list and the executed list differ.
        """
        return expand_program(self.steps, max_steps=max_steps)

    def schedule(self, *, t0: float = 0.0, T0: float = 25.0) -> List[ScheduleEntry]:
        """Ideal setpoint trajectory, one entry per physical step.

        A Q600 ramp is TIME-budgeted: it occupies |target - T_setpoint| / rate
        minutes and the clock advances whether or not the furnace arrived.
        This is not an approximation, it is how the controller behaves, and it
        is why the four "Isothermal for 20 min at 50 C" steps of
        BM-0-cycles.txt are on the program clock even though the furnace was
        still passively cooling through 211 -> 102 C at the time.

        Consequences worth knowing before trusting the numbers:
          * A ramp advances the setpoint to its target even when the furnace
            never got there; the gap between this ideal trajectory and the
            measurement is the signal dta_segments reports as `target_gap_C`.
          * Equilibrate and Jump take ZERO programmed time (the instrument
            waits for the furnace, and no duration is written in the method),
            so on a program containing them `total_minutes()` is a LOWER
            BOUND. parse_* attaches a warning saying so.
          * A ramp at rate <= 0 or with an unknown target cannot be budgeted
            at all: it gets a zero-length entry, leaves the setpoint where it
            was, and is reported by the parse warning -- reported, never
            raised, and never silently treated as instantaneous progress.
        """
        steps, _notes = self.physical_steps()
        out: List[ScheduleEntry] = []
        t = _as_float(t0)
        T = _as_float(T0)
        if not _finite(t):
            t = 0.0
        if not _finite(T):
            T = 25.0
        for st in steps:
            if st.kind == STEP_RAMP:
                rate = abs(st.rate_C_min) if _finite(st.rate_C_min) else NAN
                if _finite(rate) and rate > 0.0 and _finite(st.target_C):
                    target = float(st.target_C)
                    dur = abs(target - T) / rate
                    out.append(ScheduleEntry(st.index, t, t + dur, T, target))
                    t += dur
                    T = target
                else:
                    out.append(ScheduleEntry(st.index, t, t, T, T))
            elif st.kind == STEP_ISOTHERMAL:
                dur = float(st.duration_min) if _finite(st.duration_min) else 0.0
                if dur < 0.0:
                    dur = 0.0
                out.append(ScheduleEntry(st.index, t, t + dur, T, T))
                t += dur
            elif st.kind in (STEP_EQUILIBRATE, STEP_JUMP):
                newT = float(st.target_C) if _finite(st.target_C) else T
                out.append(ScheduleEntry(st.index, t, t, T, newT))
                T = newT
            else:
                out.append(ScheduleEntry(st.index, t, t, T, T))
        return out

    def total_minutes(self, *, T0: float = 25.0) -> float:
        """Programmed duration of the whole method, in minutes.

        T0 matters: the first ramp's length is |target - T0| / rate, so the
        starting temperature of the specimen is part of the answer (19.014 C
        on BM-0-cycles gives 890.716 min against 890.371 min measured).
        """
        entries = self.schedule(t0=0.0, T0=T0)
        return float(entries[-1].t_end) if entries else 0.0

    def describe(self) -> str:
        """One-line summary for a status bar or a log."""
        n_ramp = sum(1 for s in self.steps if s.kind == STEP_RAMP)
        n_hold = sum(1 for s in self.steps if s.kind == STEP_ISOTHERMAL)
        return (f"{len(self.steps)} steps ({n_ramp} ramps, {n_hold} holds), "
                f"status={self.status}")

    def to_dict(self) -> dict:
        """JSON-safe dict for .prism persistence."""
        return {
            "steps": [s.to_dict() for s in self.steps],
            "name": str(self.name),
            "source": str(self.source),
            "source_path": None if self.source_path is None else str(self.source_path),
            "status": str(self.status),
            "proc_name": None if self.proc_name is None else str(self.proc_name),
            "warnings": list(self.warnings),
            "error": None if self.error is None else str(self.error),
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "MethodProgram":
        """Rebuild from to_dict(). Never raises; junk becomes unrecognized."""
        if not isinstance(d, _MappingABC):
            return cls(status=PROGRAM_UNRECOGNIZED,
                       error=f"cannot restore a program from {type(d).__name__}")
        raw_steps = d.get("steps") or ()
        steps: List[MethodStep] = []
        try:
            for item in raw_steps:
                steps.append(MethodStep.from_dict(item))
        except TypeError:
            steps = []
        meta = d.get("meta") or {}
        if not isinstance(meta, _MappingABC):
            meta = {}
        warnings = d.get("warnings") or ()
        try:
            warnings = tuple(str(w) for w in warnings)
        except TypeError:
            warnings = ()
        return cls(
            steps=tuple(steps),
            name=str(d.get("name") or ""),
            source=str(d.get("source") or ""),
            source_path=(None if d.get("source_path") is None else str(d.get("source_path"))),
            status=str(d.get("status") or PROGRAM_UNRECOGNIZED),
            proc_name=(None if d.get("proc_name") is None else str(d.get("proc_name"))),
            warnings=warnings,
            error=(None if d.get("error") is None else str(d.get("error"))),
            meta=dict(meta),
        )


def _unrecognized(error: str, *, name: str = "", source: str = "",
                  source_path: Optional[str] = None,
                  warnings: Sequence[str] = (),
                  meta: Optional[Mapping[str, Any]] = None) -> MethodProgram:
    """An empty program that says why it is empty. The module's only failure
    mode -- there is no exception path out of any public function."""
    return MethodProgram(steps=(), name=name, source=source, source_path=source_path,
                         status=PROGRAM_UNRECOGNIZED, warnings=tuple(warnings),
                         error=str(error), meta=dict(meta or {}))


# =============================================================================
# 5) OrgMethod text grammar
# =============================================================================

#: Degree-sign look-alikes that all mean "degrees" in a TA export. They are
#: DELETED from a working copy of the line before matching, so the grammar
#: never has to enumerate them in a dozen patterns.
#:
#: Why so many: the raw UTF-16 file holds a real U+00B0, but the same text
#: reaches us through several lossy paths -- U+FFFD when a byte stream was
#: decoded with errors="replace", "A-circumflex + degree" when UTF-8 bytes
#: were read as latin-1, U+00BA (masculine ordinal) from some locale-aware
#: exporters, U+02DA / U+2218 from copy-paste out of a PDF or a chat client.
#: A line pasted by a user must parse the same as one read from the file, and
#: a unit marker that is missing entirely ("Ramp 10.00 C/min") must too.
_DEGREE_VARIANTS = (
    "°",   # DEGREE SIGN, the real thing
    "�",   # REPLACEMENT CHARACTER, from a lossy decode
    "Â",   # 'A-circumflex', the first byte of UTF-8 0xC2 0xB0 read as latin-1
    "º",   # MASCULINE ORDINAL INDICATOR
    "˚",   # RING ABOVE
    "∘",   # RING OPERATOR
    "·",   # MIDDLE DOT
)

_NUM = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"

_RE_STEP_INDEX = re.compile(r"^\s*(\d+)\s*[:.)]\s*(.*)$", re.S)
_RE_RAMP_RATE_FIRST = re.compile(
    r"^ramp\s+" + _NUM + r"\s*C?\s*/\s*min\s+to\s+" + _NUM + r"\s*C?\s*$", re.I)
_RE_RAMP_TARGET_FIRST = re.compile(
    r"^ramp\s+to\s+" + _NUM + r"\s*C?\s+at\s+" + _NUM + r"\s*C?\s*/\s*min\s*$", re.I)
_RE_ISOTHERMAL = re.compile(r"^isothermal\s*(?:for\s+)?" + _NUM + r"\s*min\b", re.I)
_RE_EQUILIBRATE = re.compile(r"^equilibrate\s*(?:at\s+)?" + _NUM + r"\s*C?\s*$", re.I)
_RE_JUMP = re.compile(r"^jump\s*(?:to\s+)?" + _NUM + r"\s*C?\s*$", re.I)
_RE_FLOW = re.compile(r"^mass\s*flow\s*:?\s*" + _NUM, re.I)
_RE_GAS = re.compile(r"^select\s+gas\s*:?\s*" + _NUM, re.I)
_RE_REPEAT = re.compile(
    r"^repeat\s+segment\s*:?\s*(\d+)\s*(?:for\s+)?(\d+)?\s*(?:time|times)?\s*$", re.I)
_RE_END = re.compile(r"^end(?:\s+of\s+method)?\s*$", re.I)
_RE_SAMPLING = re.compile(r"^sampling\s+interval\s*:?\s*" + _NUM, re.I)

#: Instructions we RECOGNIZE as non-thermal housekeeping. Matching one means
#: "understood, costs no furnace time" (STEP_OTHER); matching none means
#: "not understood" (STEP_UNKNOWN) and is counted and shown. The distinction
#: matters because only the second one is a reason to tell the user that the
#: program on screen may be incomplete.
_OTHER_KEYWORDS = (
    "sampling interval", "sampling", "data storage", "abort",
    "mark end of cycle", "air cool", "furnace cool", "external event",
    "event", "stop equilibrate", "autosampler", "initial temperature",
    "gas flow", "purge", "record", "comment",
)


def _strip_degree_marks(text: str) -> str:
    """Delete every degree-sign look-alike from a working copy of `text`.

    Used for MATCHING only -- the stored `.text` keeps whatever the file
    actually contained, so nothing is rewritten behind the user's back.
    """
    out = text
    for ch in _DEGREE_VARIANTS:
        if ch in out:
            out = out.replace(ch, "")
    return out


def _norm_for_match(text: str) -> str:
    """Normalize a step line for matching: degree marks gone, whitespace
    collapsed, ends trimmed. Case is handled by re.I in the patterns."""
    return re.sub(r"\s+", " ", _strip_degree_marks(text)).strip()


def parse_step_text(text: str, index: Optional[int] = None, *,
                    source: str = SOURCE_ORGMETHOD) -> MethodStep:
    """Parse ONE method line into a MethodStep. Never raises.

    Accepts the line with or without its "<n>: " numbering; when the number
    is present and `index` is None it is used, because the instrument's own
    numbering is what every other TA artefact (the .001 file, the printed
    method, a user's notes) refers to.

    An unrecognized line becomes STEP_UNKNOWN with `text` verbatim and
    `confidence == "unknown"` -- kept in place so the step numbering of
    everything after it stays true.
    """
    raw = "" if text is None else str(text)
    body = raw.strip()
    idx = index
    m = _RE_STEP_INDEX.match(body)
    if m:
        body = m.group(2).strip()
        if idx is None:
            try:
                idx = int(m.group(1))
            except ValueError:      # pragma: no cover - the group is \d+
                idx = None
    if idx is None:
        idx = 0                     # 0 == "unnumbered"; parse_org_method_lines fills it in

    def _step(kind: str, **kw: Any) -> MethodStep:
        kw.setdefault("text", body)
        kw.setdefault("source", source)
        return MethodStep(index=int(idx), kind=kind, **kw)

    if not body:
        return _step(STEP_UNKNOWN, confidence=CONFIDENCE_UNKNOWN)

    norm = _norm_for_match(body)
    low = norm.lower()

    m = _RE_RAMP_RATE_FIRST.match(norm)
    if m:
        return _step(STEP_RAMP, rate_C_min=_as_float(m.group(1)),
                     target_C=_as_float(m.group(2)))
    m = _RE_RAMP_TARGET_FIRST.match(norm)
    if m:
        # "Ramp to 1300 C at 10 C/min" -- same step, target written first.
        return _step(STEP_RAMP, rate_C_min=_as_float(m.group(2)),
                     target_C=_as_float(m.group(1)))
    m = _RE_ISOTHERMAL.match(norm)
    if m:
        return _step(STEP_ISOTHERMAL, duration_min=_as_float(m.group(1)))
    m = _RE_EQUILIBRATE.match(norm)
    if m:
        return _step(STEP_EQUILIBRATE, target_C=_as_float(m.group(1)))
    m = _RE_JUMP.match(norm)
    if m:
        return _step(STEP_JUMP, target_C=_as_float(m.group(1)))
    m = _RE_FLOW.match(norm)
    if m:
        return _step(STEP_FLOW, value=_as_float(m.group(1)))
    m = _RE_GAS.match(norm)
    if m:
        return _step(STEP_GAS, value=_as_float(m.group(1)))
    m = _RE_REPEAT.match(norm)
    if m:
        target = int(m.group(1))
        count = int(m.group(2)) if m.group(2) is not None else 1
        return _step(STEP_REPEAT, repeat_target=target, repeat_count=count)
    if _RE_END.match(norm):
        return _step(STEP_END)
    m = _RE_SAMPLING.match(norm)
    if m:
        return _step(STEP_OTHER, value=_as_float(m.group(1)))
    for kw in _OTHER_KEYWORDS:
        if kw in low:
            return _step(STEP_OTHER)
    return _step(STEP_UNKNOWN, confidence=CONFIDENCE_UNKNOWN)


def _program_warnings(steps: Sequence[MethodStep]) -> List[str]:
    """Warnings that travel with a parsed program.

    Every one of these is a WARNING and not a refusal, per the standing rule
    that unusual programs must still be processable: they describe how far to
    trust `schedule()`, they never gate it.
    """
    out: List[str] = []
    bad = [s for s in steps if not s.is_recognized]
    if bad:
        shown = ", ".join(str(s.index) for s in bad[:8])
        more = "" if len(bad) <= 8 else f" (+{len(bad) - 8} more)"
        out.append(f"{len(bad)} step(s) not understood and kept verbatim: {shown}{more}")
    if any(s.kind == STEP_EQUILIBRATE for s in steps):
        out.append("Equilibrate steps carry no programmed duration (the instrument waits "
                   "for the furnace), so the programmed total time is a lower bound.")
    dead = [s for s in steps
            if s.kind == STEP_RAMP and (not _finite(s.rate_C_min) or s.rate_C_min <= 0.0)]
    if dead:
        shown = ", ".join(str(s.index) for s in dead[:8])
        out.append(f"Ramp step(s) {shown} have no usable rate; they are given zero "
                   f"programmed time and leave the setpoint where it was.")
    holes = [s for s in steps
             if s.kind == STEP_RAMP and _finite(s.rate_C_min) and s.rate_C_min > 0.0
             and not _finite(s.target_C)]
    if holes:
        shown = ", ".join(str(s.index) for s in holes[:8])
        out.append(f"Ramp step(s) {shown} have no usable target; their duration is "
                   f"not budgeted and is counted as zero.")
    if steps and not any(s.is_thermal for s in steps):
        out.append("The method contains no thermal steps; there is no temperature "
                   "program to align to the data.")
    return out


def parse_org_method_lines(lines: Iterable[str], *, name: str = "",
                           source_path: Optional[str] = None) -> MethodProgram:
    """Ordered OrgMethod lines -> MethodProgram. Never raises.

    Lines may keep or drop their "<n>: " numbering; an unnumbered line is
    given its 1-based position, so the two mixed together still produce a
    sane, gap-free table.
    """
    steps: List[MethodStep] = []
    try:
        iterator = list(lines)
    except TypeError:
        return _unrecognized("OrgMethod lines are not iterable",
                             name=name, source=SOURCE_ORGMETHOD, source_path=source_path,
                             meta={"format": ORGMETHOD_FORMAT_TAG})
    for raw in iterator:
        if raw is None:
            continue
        text = str(raw).strip()
        if not text:
            continue
        step = parse_step_text(text, index=None, source=SOURCE_ORGMETHOD)
        if step.index <= 0:
            step = replace(step, index=len(steps) + 1)
        steps.append(step)

    if not steps:
        return _unrecognized("no OrgMethod step lines were supplied",
                             name=name, source=SOURCE_ORGMETHOD, source_path=source_path,
                             meta={"format": ORGMETHOD_FORMAT_TAG, "n_lines": 0})

    warnings = _program_warnings(steps)
    n_bad = sum(1 for s in steps if not s.is_recognized)
    status = PROGRAM_PARSED if n_bad == 0 else PROGRAM_PARTIAL
    return MethodProgram(
        steps=tuple(steps), name=name, source=SOURCE_ORGMETHOD, source_path=source_path,
        status=status, warnings=tuple(warnings),
        meta={"format": ORGMETHOD_FORMAT_TAG, "n_lines": len(steps)},
    )


def _header_pairs(header_lines: Optional[Sequence[Sequence[str]]]) -> List[Tuple[str, str]]:
    """Coerce a [[key, value], ...] structure into clean (key, value) pairs.

    A .prism round trip turns tuples into lists and may drop malformed rows;
    this accepts anything two-element-ish and ignores the rest instead of
    failing the load.
    """
    out: List[Tuple[str, str]] = []
    if not header_lines:
        return out
    try:
        rows = list(header_lines)
    except TypeError:
        return out
    for row in rows:
        try:
            key, value = row[0], row[1]
        except (TypeError, IndexError, KeyError):
            continue
        out.append((str(key).strip(), str(value).strip()))
    return out


def _org_method_lines_with_provenance(
        header: Mapping[str, Any],
        header_lines: Optional[Sequence[Sequence[str]]] = None,
) -> Tuple[List[str], str]:
    """Every OrgMethod line in order, plus WHERE it came from.

    Provenance matters to exactly one caller decision -- whether the single
    line we found is a genuinely one-step method or the wreckage of
    `dict.setdefault` collapsing 19 repeated keys into the first one -- so it
    is computed here and exposed through program_from_header's warning rather
    than guessed at downstream.
    Returns (lines, provenance) with provenance in
    {"header_object", "header_lines", "collapsed_key", "none"}.
    """
    # 1) A header object that kept its repeated keys (core.io_universal.TAHeader).
    getter = getattr(header, "all", None)
    if callable(getter):
        try:
            values = list(getter("OrgMethod"))
        except Exception:
            values = []
        lines = [str(v).strip() for v in values if str(v).strip()]
        if lines:
            return lines, "header_object"
    raw_lines = getattr(header, "lines", None)
    if raw_lines:
        pairs = _header_pairs(raw_lines)
        lines = [v for k, v in pairs if k.lower() == "orgmethod" and v]
        if lines:
            return lines, "header_object"

    # 2) The flattened [[key, value], ...] carried in meta / a .prism project.
    pairs = _header_pairs(header_lines)
    if pairs:
        lines = [v for k, v in pairs if k.lower() == "orgmethod" and v]
        if lines:
            return lines, "header_lines"

    # 3) The legacy plain dict, where repeated keys have already collapsed.
    value: Any = None
    if isinstance(header, _MappingABC):
        for key in header.keys():
            if str(key).strip().lower() == "orgmethod":
                value = header[key]
                break
    if value is None:
        return [], "none"
    if isinstance(value, (list, tuple)):
        lines = [str(v).strip() for v in value if str(v).strip()]
        return (lines, "header_lines") if lines else ([], "none")
    text = str(value).strip()
    return ([text], "collapsed_key") if text else ([], "none")


def org_method_lines_from_header(header: Mapping[str, Any],
                                 header_lines: Optional[Sequence[Sequence[str]]] = None
                                 ) -> List[str]:
    """Every OrgMethod line in order: from `header.lines`, else `header_lines`
    (a .prism round trip), else the single collapsed `header["OrgMethod"]`."""
    lines, _prov = _org_method_lines_with_provenance(header, header_lines)
    return lines


def program_from_header(header: Mapping[str, Any],
                        header_lines: Optional[Sequence[Sequence[str]]] = None,
                        *, name: str = "", source_path: Optional[str] = None
                        ) -> MethodProgram:
    """Build a program from a parsed TA header. Never raises.

    Demotes to PROGRAM_PARTIAL with an explicit warning when only the legacy
    collapsed first line was available: a plain dict keeps one value per key,
    so a 19-step cyclic method arrives here as the single line
    "1: Select gas 1". Reporting that as a parsed one-step program would be a
    confident lie about what the furnace did -- the user must be told the
    steps are missing, not shown a program that looks complete.
    """
    if header is None:
        header = {}
    lines, provenance = _org_method_lines_with_provenance(header, header_lines)
    proc_name = None
    if isinstance(header, _MappingABC):
        for key in ("ProcName", "Method"):
            for have in header.keys():
                if str(have).strip().lower() == key.lower():
                    candidate = str(header[have]).strip()
                    if candidate:
                        proc_name = candidate
                    break
            if proc_name:
                break

    if not lines:
        return _unrecognized(
            "the file header carries no OrgMethod lines",
            name=name, source=SOURCE_ORGMETHOD, source_path=source_path,
            meta={"format": ORGMETHOD_FORMAT_TAG, "provenance": provenance},
        )

    program = parse_org_method_lines(lines, name=name, source_path=source_path)
    meta = dict(program.meta)
    meta["provenance"] = provenance
    warnings = list(program.warnings)
    status = program.status
    if provenance == "collapsed_key":
        warnings.insert(0, (
            "Only one OrgMethod line was available: the header was read into a plain "
            "dict, which keeps the FIRST value of a repeated key and discards the rest. "
            "A multi-step method therefore arrives here as its step 1 alone. Re-read the "
            "file with a header that preserves repeated keys to see the whole program."))
        status = PROGRAM_PARTIAL
    return replace(program, status=status, warnings=tuple(warnings), meta=meta,
                   proc_name=proc_name)


# =============================================================================
# 6) Repeat expansion
# =============================================================================

def expand_program(steps: Sequence[MethodStep], *, max_steps: int = 2000
                   ) -> Tuple[Tuple[MethodStep, ...], Tuple[str, ...]]:
    """`Repeat segment N M times` -> physical step list (block runs M+1 times).

    Semantics, all of them deliberate:
      * The block is written step N up to and including the repeat
        instruction; it executes once normally and then M more times, so
        Equilibrate / Ramp / Iso / Ramp / "Repeat segment 2 3 times" / End
        expands to 1 + 4*3 + 1 = 14 physical steps.
      * The repeat instruction itself is NOT emitted: it is flow control, not
        something the furnace does, and emitting it would corrupt every
        step-count and alignment downstream.
      * Targets are BACKWARDS ONLY. A forward or self-referencing target is a
        malformed method that would loop forever or skip work; it is ignored
        with a note rather than obeyed or raised on.
      * A satisfied repeat FORGETS its counter, so an enclosing loop re-arms
        it -- that is what makes nested cycles execute the inner block once
        per outer pass instead of once in total.
      * Hard caps: `max_steps` emitted steps and 4*max_steps interpreter
        iterations. A runaway method (999999 passes) therefore terminates
        with a truncated list and a note saying so, which is the "report it,
        do not raise" rule applied to a program that cannot be executed as
        written.

      * `End of method` ends the physical program even when written steps
        follow it; those steps are unreachable and are reported as such. The
        interpreter runs on every program, repeats or not, so this rule and
        the cap behave identically whether or not a loop is present.

    Returns (physical_steps, notes). On a program with no repeats and no
    early end, the returned tuple equals the written steps exactly. Never
    raises.
    """
    try:
        written = list(steps)
    except TypeError:
        return (), ("step list is not iterable; nothing to expand",)
    if not written:
        return (), ()

    cap = int(max_steps) if _finite(max_steps) else 2000
    if cap <= 0:
        return (), ("expansion cap is not positive; nothing expanded",)

    # Written index -> position, for the backwards jump. Later duplicates of
    # the same written number are ignored: the FIRST one is what an operator
    # means by "segment 2".
    pos_of_index: Dict[int, int] = {}
    for pos, st in enumerate(written):
        key = int(getattr(st, "index", 0) or 0)
        pos_of_index.setdefault(key, pos)

    out: List[MethodStep] = []
    notes: List[str] = []
    counters: Dict[int, int] = {}
    passes: Dict[int, int] = {}
    pc = 0
    iterations = 0
    max_iterations = 4 * cap
    truncated = False

    while 0 <= pc < len(written):
        iterations += 1
        if iterations > max_iterations:
            notes.append(f"repeat expansion stopped after {max_iterations} interpreter "
                         f"iterations (cap 4*max_steps); the method may loop forever")
            truncated = True
            break
        st = written[pc]
        kind = getattr(st, "kind", STEP_UNKNOWN)

        if kind == STEP_REPEAT:
            target_index = int(getattr(st, "repeat_target", 0) or 0)
            count = int(getattr(st, "repeat_count", 0) or 0)
            target_pos = pos_of_index.get(target_index, -1)
            if target_pos < 0:
                notes.append(f"step {st.index}: repeat target segment {target_index} "
                             f"does not exist; the repeat is ignored")
                pc += 1
                continue
            if target_pos >= pc:
                notes.append(f"step {st.index}: repeat target segment {target_index} is "
                             f"not before it; a forward repeat is ignored")
                pc += 1
                continue
            if count <= 0:
                notes.append(f"step {st.index}: repeat count {count} is not positive; "
                             f"the block runs once")
                pc += 1
                continue
            remaining = counters.get(pc)
            if remaining is None:
                remaining = count
            if remaining > 0:
                counters[pc] = remaining - 1
                pc = target_pos
                continue
            # Satisfied: forget the counter so an enclosing loop re-arms it.
            counters.pop(pc, None)
            pc += 1
            continue

        if len(out) >= cap:
            notes.append(f"repeat expansion capped at {cap} steps (max_steps={cap}); "
                         f"the physical program is longer than this")
            truncated = True
            break

        n_pass = passes.get(pc, 0)
        passes[pc] = n_pass + 1
        out.append(replace(st, repeat_pass=n_pass) if n_pass else st)

        if kind == STEP_END:
            if pc + 1 < len(written):
                notes.append(f"step {st.index}: end of method; "
                             f"{len(written) - pc - 1} written step(s) after it are unreachable")
            break
        pc += 1

    if truncated and len(out) > cap:            # pragma: no cover - defensive
        out = out[:cap]
    return tuple(out), tuple(notes)


# =============================================================================
# 7) The .mth binary
# =============================================================================

def scan_utf16_strings(blob: bytes, *, min_len: int = 3) -> List[Tuple[int, str]]:
    """Printable UTF-16LE runs, scanned on EVEN offsets only.

    Even-only is not an optimization, it is correctness: UTF-16 code units
    are two bytes, so a scanner that steps one byte at a time drifts parity
    after the first odd match and then reads the high byte of one character
    together with the low byte of the next -- which is how ProcName ("Ramp"
    at offset 142 of cycles.mth) goes missing.

    A run is a stretch of code units whose high byte is 0 and whose low byte
    is printable Latin-1; runs shorter than `min_len` characters are dropped
    as noise. Returns [(byte_offset, text)], in file order. Never raises.
    """
    try:
        data = bytes(blob)
    except (TypeError, ValueError):
        return []
    n = len(data)
    try:
        floor = max(1, int(min_len))
    except (TypeError, ValueError):
        floor = 3
    out: List[Tuple[int, str]] = []
    start: Optional[int] = None
    chars: List[str] = []
    i = 0
    while i + 1 < n:
        lo, hi = data[i], data[i + 1]
        if hi == 0 and (0x20 <= lo < 0x7F or lo >= 0xA0):
            if start is None:
                start = i
                chars = []
            chars.append(chr(lo))
        else:
            if start is not None and len(chars) >= floor:
                out.append((start, "".join(chars)))
            start = None
            chars = []
        i += 2
    if start is not None and len(chars) >= floor:
        out.append((start, "".join(chars)))
    return out


def _mth_float_is_plausible(value: float) -> bool:
    """Could this float32 be a real method argument?

    Plausible means finite AND (exactly zero OR 1e-6 < |v| < 1e5). The
    magnitude floor is the load-bearing half: four zero-ish bytes decode as a
    DENORMAL float (1e-45-ish), which is finite and tiny and would sail
    through a naive "is it a number?" test. Offset 110 of the real cycles.mth
    is exactly that -- opcode 0x1207, args 8.4e-45 and 5.6e-45, twelve zero
    bytes after them -- a perfect shape match that is not a method record.
    No rate, temperature, duration or flow a furnace can be asked for is
    smaller than a micro-unit or larger than 1e5.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(v):
        return False
    if v == 0.0:
        return True
    a = abs(v)
    return 1e-6 < a < 1e5


def _scan_mth_records(blob: bytes) -> Tuple[Dict[int, Tuple[int, float, float]], int]:
    """Every byte offset whose 24 bytes have the shape of a TA step record.

    Shape: uint16 opcode @+0 (non-zero), uint16 0 @+2, float32 @+4, float32
    @+8, twelve zero bytes @+12. The opcode is NOT required to be one we
    know: an unknown opcode is a step we must keep and flag, so the locator
    has to see it as part of the table.

    Returns ({offset: (opcode, arg1, arg2)}, n_bytes_scanned).
    """
    data = blob
    n = len(data)
    limit = min(n, MTH_MAX_SCAN_BYTES)
    found: Dict[int, Tuple[int, float, float]] = {}
    last = limit - MTH_RECORD_SIZE
    off = 0
    while off <= last:
        # Cheap rejects first: this loop runs once per byte of the file.
        if (data[off] or data[off + 1]) and not (data[off + 2] or data[off + 3]):
            if data[off + 12:off + 24] == _ZERO12:
                opcode, _pad, a1, a2 = struct.unpack_from("<HHff", data, off)
                if _mth_float_is_plausible(a1) and _mth_float_is_plausible(a2):
                    found[off] = (int(opcode), float(a1), float(a2))
        off += 1
    return found, limit


def _chain_from(valid: Mapping[int, Tuple[int, float, float]], start: int,
                n_bytes: int, max_gap: int) -> List[int]:
    """Stride-24 run of slots beginning at `start`, bridging short gaps.

    Up to `max_gap` consecutive unreadable slots are bridged, because a
    corrupt record in the middle of a table is still part of that table --
    resyncing past it would renumber every step after it. The run never ends
    on a bridged slot, so a table is never padded with phantom steps.
    """
    slots = [start]
    cur = start
    while True:
        nxt = cur + MTH_RECORD_SIZE
        found = -1
        for gap in range(0, max_gap + 1):
            probe = nxt + gap * MTH_RECORD_SIZE
            if probe + MTH_RECORD_SIZE > n_bytes:
                break
            if probe in valid:
                found = probe
                break
        if found < 0:
            break
        probe = nxt
        while probe < found:
            slots.append(probe)
            probe += MTH_RECORD_SIZE
        slots.append(found)
        cur = found
    return slots


def _locate_chains(valid: Mapping[int, Tuple[int, float, float]], n_scanned: int,
                   gap: int, floor: int) -> List[Tuple[int, int, int, int]]:
    """Group already-scanned record slots into candidate tables, best first.

    Split out from locate_mth_step_table so parse_mth_bytes can scan the blob
    ONCE: on a mis-picked multi-megabyte file the byte sweep is the whole
    cost, and doing it twice was the difference between a responsive Load
    dialog and a visible freeze.
    """
    candidates: List[Tuple[int, int, int, int]] = []
    consumed: set = set()
    for start in sorted(valid):
        if start in consumed:
            continue
        slots = _chain_from(valid, start, n_scanned, gap)
        consumed.update(slots)
        n_known = sum(1 for s in slots
                      if s in valid and valid[s][0] in MTH_OPCODES)
        if len(slots) < floor or n_known < 1:
            continue
        candidates.append((start, len(slots), n_known, len(slots) - n_known))
    # Best = most understood records, then longest, then earliest.
    candidates.sort(key=lambda c: (-c[2], -c[1], c[0]))
    return candidates


def _coerce_scan_args(max_gap: Any, min_records: Any) -> Tuple[int, int]:
    """Sanitize the two scan knobs; bad values fall back to the defaults."""
    try:
        gap = max(0, int(max_gap))
    except (TypeError, ValueError):
        gap = MTH_DEFAULT_MAX_GAP
    try:
        floor = max(1, int(min_records))
    except (TypeError, ValueError):
        floor = MTH_MIN_RECORDS
    return gap, floor


def locate_mth_step_table(blob: bytes, *, max_gap: int = MTH_DEFAULT_MAX_GAP,
                          min_records: int = MTH_MIN_RECORDS
                          ) -> List[Tuple[int, int, int, int]]:
    """Find the 24-byte step table by SCAN, never a fixed offset.

    The table sits at 654 in cycles.mth, but that offset is a property of
    that file's preamble (name, comment and calibration strings), not of the
    format: a method saved under a longer name moves it. Anything anchored to
    654 would read another file's padding as a furnace program.

    Returns [(start, n_records, n_known, n_nonstandard)], best first, where
    n_known counts records whose opcode is in MTH_OPCODES and n_nonstandard
    counts the rest (unknown opcodes plus bridged unreadable slots). A
    candidate must hold at least `min_records` records and at least one
    understood opcode. Never raises; an unusable blob gives [].
    """
    try:
        data = bytes(blob)
    except (TypeError, ValueError):
        return []
    if len(data) < MTH_RECORD_SIZE:
        return []
    gap, floor = _coerce_scan_args(max_gap, min_records)
    valid, n_scanned = _scan_mth_records(data)
    if not valid:
        return []
    return _locate_chains(valid, n_scanned, gap, floor)


def _decode_mth_record(position: int, offset: int,
                       record: Optional[Tuple[int, float, float]]) -> MethodStep:
    """One 24-byte slot -> one MethodStep, understood or not.

    `record is None` means the slot failed the shape/plausibility test but
    sits INSIDE the table: it becomes STEP_INVALID, keeps its position, and
    is counted -- never dropped, because dropping it would shift every step
    number after it.
    """
    if record is None:
        return MethodStep(index=position, kind=STEP_INVALID,
                          text=f"<unreadable .mth record at byte {offset}>",
                          source=SOURCE_MTH, confidence=CONFIDENCE_INVALID)
    opcode, a1, a2 = record
    kind = MTH_OPCODES.get(opcode)
    if kind is None:
        return MethodStep(index=position, kind=STEP_UNKNOWN,
                          text=(f"<unknown .mth opcode 0x{opcode:04X} "
                                f"at byte {offset}, args {a1:.6g}, {a2:.6g}>"),
                          source=SOURCE_MTH, opcode=opcode,
                          confidence=CONFIDENCE_UNKNOWN)
    if kind == STEP_RAMP:
        step = MethodStep(index=position, kind=STEP_RAMP, rate_C_min=a1, target_C=a2,
                          source=SOURCE_MTH, opcode=opcode)
    elif kind == STEP_ISOTHERMAL:
        step = MethodStep(index=position, kind=STEP_ISOTHERMAL, duration_min=a1,
                          source=SOURCE_MTH, opcode=opcode)
    elif kind == STEP_FLOW:
        step = MethodStep(index=position, kind=STEP_FLOW, value=a1,
                          source=SOURCE_MTH, opcode=opcode)
    else:
        # Gas: the binary index is 0-based while the text is 1-based
        # ("Select gas 1" is stored as 0.0). Observed on a single value, so
        # it is recorded here rather than trusted anywhere else.
        step = MethodStep(index=position, kind=STEP_GAS, value=float(int(a1) + 1),
                          source=SOURCE_MTH, opcode=opcode)
    return replace(step, text=step.render())


def parse_mth_bytes(blob: bytes, *, name: str = "",
                    source_path: Optional[str] = None) -> MethodProgram:
    """Decode a TA .mth. NEVER raises.

    Returns status parsed / partial / unrecognized. Unknown or unreadable
    records are KEPT and flagged, never dropped and never resynced past.
    `meta["format"]` is MTH_FORMAT_TAG because this decode is reverse
    engineered from one file: see the module docstring.
    """
    try:
        data = bytes(blob)
    except (TypeError, ValueError) as exc:
        return _unrecognized(f"not a byte buffer: {exc}", name=name, source=SOURCE_MTH,
                             source_path=source_path, meta={"format": MTH_FORMAT_TAG})

    meta: Dict[str, Any] = {
        "format": MTH_FORMAT_TAG,
        "record_size": MTH_RECORD_SIZE,
        "n_bytes": len(data),
    }
    warnings: List[str] = []

    # The signature lives at byte 0; scanning the first 64 KiB for it keeps a
    # mis-picked 100 MB file from costing a full string sweep as well as a
    # full record sweep.
    head_strings = scan_utf16_strings(data[:65536])
    magic_found = any(s.startswith(MTH_MAGIC) for _off, s in head_strings)
    meta["magic_found"] = bool(magic_found)
    if not magic_found:
        warnings.append(
            f"the {MTH_MAGIC!r} signature was not found; this may not be a TA method "
            f"file, or it may be a variant this decoder has never seen.")

    if len(data) < MTH_RECORD_SIZE:
        return _unrecognized(
            f"file is {len(data)} bytes, shorter than one {MTH_RECORD_SIZE}-byte step record",
            name=name, source=SOURCE_MTH, source_path=source_path,
            warnings=warnings, meta=meta)

    if len(data) > MTH_MAX_SCAN_BYTES:
        warnings.append(f"only the first {MTH_MAX_SCAN_BYTES} bytes were scanned for the "
                        f"step table ({len(data)} bytes total).")

    valid, n_scanned = _scan_mth_records(data)
    gap, floor = _coerce_scan_args(MTH_DEFAULT_MAX_GAP, MTH_MIN_RECORDS)
    candidates = _locate_chains(valid, n_scanned, gap, floor) if valid else []
    meta["n_candidates"] = len(candidates)
    if not candidates:
        return _unrecognized(
            f"no {MTH_RECORD_SIZE}-byte TA step table found in {len(data)} bytes",
            name=name, source=SOURCE_MTH, source_path=source_path,
            warnings=warnings, meta=meta)

    start, n_records, n_known, n_nonstandard = candidates[0]
    meta.update({"table_offset": start, "n_records": n_records,
                 "n_known": n_known, "n_nonstandard": n_nonstandard})
    if len(candidates) > 1:
        others = ", ".join(f"{c[0]} ({c[1]} records)" for c in candidates[1:4])
        warnings.append(f"{len(candidates)} step-table candidates were found; the one at "
                        f"byte {start} with {n_records} records was used. Others: {others}.")

    steps: List[MethodStep] = []
    for position, offset in enumerate(range(start, start + n_records * MTH_RECORD_SIZE,
                                            MTH_RECORD_SIZE), start=1):
        steps.append(_decode_mth_record(position, offset, valid.get(offset)))

    # ProcName: the first printable string after the magic and before the
    # table. On cycles.mth that is "Ramp" at byte 142. It is a WEAK identity
    # signal -- TA writes the procedure TYPE there, not the method's name --
    # so it is reported and never used to decide whether two methods match.
    proc_name: Optional[str] = None
    magic_end = 2 * len(MTH_MAGIC)
    preamble_strings = (head_strings if start <= 65536
                        else scan_utf16_strings(data[:start]))
    for off, text in preamble_strings:
        if off < magic_end or off >= start:
            continue
        candidate = text.strip()
        if candidate and candidate != MTH_MAGIC:
            proc_name = candidate
            break

    warnings.extend(_program_warnings(steps))
    n_bad = sum(1 for s in steps if not s.is_recognized)
    status = PROGRAM_PARSED if n_bad == 0 else PROGRAM_PARTIAL

    # A step table that runs to the END of the file is the fingerprint of a
    # truncated copy: the real cycles.mth carries 1110 bytes of table inside
    # 6776 bytes of file, because TA writes a long trailer after the steps.
    # When the table reaches the last bytes instead, the chain was cut off by
    # the end of the buffer rather than by TA, and the steps beyond the cut
    # are simply absent -- an 890 min run then silently acquires a 146 min
    # program clock and loses the programmed rate on three of four ramps.
    # `status` is this module's honesty channel, so it must say `partial`: the
    # steps we have are real, the ones we do not have were reported missing by
    # nothing else in the parse.
    #
    # The test is "not even one more record would fit", not "ends exactly at
    # the last byte", because the realistic cause is an interrupted copy and
    # that cuts at an arbitrary byte: the scanner then stops at the last
    # COMPLETE record and leaves 1-23 orphan bytes behind, so 23 of every 24
    # cut positions would slip through an exact-EOF test. The cost of the
    # wider rule is that a .mth variant with a trailer shorter than one record
    # would be called `partial` -- a warning next to a complete step list,
    # which is the cheap direction to be wrong in.
    table_end = start + n_records * MTH_RECORD_SIZE
    orphan_bytes = len(data) - table_end
    meta["table_end"] = int(table_end)
    meta["trailer_bytes"] = int(orphan_bytes)
    meta["table_reaches_eof"] = bool(orphan_bytes < MTH_RECORD_SIZE)
    if orphan_bytes < MTH_RECORD_SIZE:
        warnings.append(
            f"the step table runs to the end of the file (ends at {table_end} of "
            f"{len(data)} bytes, leaving {orphan_bytes} byte(s)); the method file "
            f"may be truncated and steps after #{len(steps)} may be missing.")
        status = PROGRAM_PARTIAL
    return MethodProgram(
        steps=tuple(steps), name=name, source=SOURCE_MTH, source_path=source_path,
        status=status, proc_name=proc_name, warnings=tuple(warnings), meta=meta,
    )


def parse_mth_file(path) -> MethodProgram:
    """parse_mth_bytes on a file; an unreadable path becomes status=unrecognized.

    A missing or unreadable method file must not take the data file down with
    it -- the .mth is a cross-check, and the run is still analysable without
    one.

    `except OSError` alone is one clause too narrow: a path carrying an
    embedded NUL ("\\x00bad") reaches the OS layer as a ValueError, not an
    OSError, and a pasted path from a corrupt recent-files list is exactly how
    that arrives. Both are the same event to this caller -- "the bytes could
    not be read" -- so both are reported, not raised.
    """
    try:
        p = Path(path)
    except TypeError as exc:
        return _unrecognized(f"not a usable path: {exc}", source=SOURCE_MTH,
                             meta={"format": MTH_FORMAT_TAG})
    try:
        blob = p.read_bytes()
    except (OSError, ValueError) as exc:
        return _unrecognized(f"cannot read {p}: {exc}", name=p.stem, source=SOURCE_MTH,
                             source_path=str(p), meta={"format": MTH_FORMAT_TAG})
    return parse_mth_bytes(blob, name=p.stem, source_path=str(p))


# =============================================================================
# 8) Program vs program
# =============================================================================

def _thermal_signature(step: MethodStep) -> str:
    """One thermal step as a canonical token (see program_fingerprint)."""
    if step.kind == STEP_RAMP:
        return f"R{_fmt2(step.rate_C_min)}>{_fmt1(step.target_C)}"
    if step.kind == STEP_ISOTHERMAL:
        return f"I{_fmt2(step.duration_min)}"
    if step.kind == STEP_EQUILIBRATE:
        return f"E{_fmt1(step.target_C)}"
    if step.kind == STEP_JUMP:
        return f"J{_fmt1(step.target_C)}"
    return f"?{step.kind}"          # pragma: no cover - callers filter to thermal


def _physical_thermal(program: MethodProgram) -> Tuple[MethodStep, ...]:
    """Thermal steps of the EXECUTED program (repeats unrolled)."""
    steps, _notes = program.physical_steps()
    return tuple(s for s in steps if s.is_thermal)


def program_fingerprint(program: MethodProgram) -> str:
    """Canonical thermal-only signature, e.g. 'R20.00>1300.0|I20.00|...'.

    Equal iff the furnace does the same thing: gas selection, mass flow,
    sampling interval, step numbering and text formatting are all excluded,
    and repeats are unrolled first, so a method written as a loop and the
    same method written out longhand fingerprint identically.
    """
    if program is None:
        return ""
    return "|".join(_thermal_signature(s) for s in _physical_thermal(program))


def _steps_agree(a: MethodStep, b: MethodStep, *, rate_tol: float,
                 target_tol: float, duration_tol: float) -> Tuple[bool, str]:
    """Do two thermal steps ask the furnace for the same thing?

    Returns (agree, reason_when_not). Tolerances are absolute and in the
    natural units of each quantity, because that is how a user judges them
    ("same 20 C/min ramp", "1 C is nothing on a 1300 C target").
    """
    if a.kind != b.kind:
        return False, f"{a.kind} vs {b.kind}"
    if a.kind == STEP_RAMP:
        if not _finite(a.rate_C_min) or not _finite(b.rate_C_min):
            return False, "a ramp rate is missing"
        if abs(abs(a.rate_C_min) - abs(b.rate_C_min)) > rate_tol:
            return False, (f"rate {_fmt2(a.rate_C_min)} vs {_fmt2(b.rate_C_min)} "
                           f"{DEGREE_SIGN}C/min")
        if not _finite(a.target_C) or not _finite(b.target_C):
            return False, "a ramp target is missing"
        if abs(a.target_C - b.target_C) > target_tol:
            return False, (f"target {_fmt2(a.target_C)} vs {_fmt2(b.target_C)} "
                           f"{DEGREE_SIGN}C")
        return True, ""
    if a.kind == STEP_ISOTHERMAL:
        if not _finite(a.duration_min) or not _finite(b.duration_min):
            return False, "a hold duration is missing"
        if abs(a.duration_min - b.duration_min) > duration_tol:
            return False, f"hold {_fmt2(a.duration_min)} vs {_fmt2(b.duration_min)} min"
        return True, ""
    # Equilibrate / Jump: only the target matters.
    if not _finite(a.target_C) or not _finite(b.target_C):
        return False, "a target temperature is missing"
    if abs(a.target_C - b.target_C) > target_tol:
        return False, f"target {_fmt2(a.target_C)} vs {_fmt2(b.target_C)} {DEGREE_SIGN}C"
    return True, ""


def compare_programs(a: MethodProgram, b: MethodProgram, *, rate_tol: float = 0.05,
                     target_tol: float = 1.0, duration_tol: float = 0.5
                     ) -> ProgramComparison:
    """Compare a user-loaded .mth against the data file's own OrgMethod.

    OrgMethod is authoritative -- it is what the instrument recorded itself
    running, while a .mth on disk is only what someone saved under that name.
    So this answers "may I use b to describe a?", and a "mismatch" must stop
    the .mth from relabelling the run, not stop the run from being analysed.
    """
    if a is None or b is None or not a or not b:
        which = []
        if a is None or not a:
            which.append("the first program")
        if b is None or not b:
            which.append("the second program")
        return ProgramComparison(
            verdict="incomparable", score=0.0,
            reasons=(f"{' and '.join(which)} could not be read; no comparison is possible",))

    ta = _physical_thermal(a)
    tb = _physical_thermal(b)
    if not ta or not tb:
        return ProgramComparison(
            verdict="incomparable", score=0.0,
            reasons=("at least one program has no thermal steps to compare",))

    rows: List[Tuple[int, str, str, str]] = []
    reasons: List[str] = []
    n_same = 0
    for i in range(max(len(ta), len(tb))):
        sa = ta[i] if i < len(ta) else None
        sb = tb[i] if i < len(tb) else None
        if sa is not None and sb is not None:
            agree, why = _steps_agree(sa, sb, rate_tol=rate_tol, target_tol=target_tol,
                                      duration_tol=duration_tol)
            rows.append((i + 1, sa.render(), sb.render(), "same" if agree else "differs"))
            if agree:
                n_same += 1
            elif len(reasons) < 8:
                reasons.append(f"thermal step {i + 1}: {why}")
        elif sa is not None:
            rows.append((i + 1, sa.render(), "", "only_a"))
        else:
            rows.append((i + 1, "", sb.render(), "only_b"))

    score = n_same / float(max(len(ta), len(tb)))
    if len(ta) != len(tb):
        reasons.append(f"{len(ta)} thermal steps vs {len(tb)}")
    if program_fingerprint(a) == program_fingerprint(b):
        verdict = "identical"
    elif len(ta) == len(tb) and n_same == len(ta):
        verdict = "equivalent"
    else:
        verdict = "mismatch"
    return ProgramComparison(verdict=verdict, score=float(score),
                             rows=tuple(rows), reasons=tuple(reasons))


def _merged_pair_agrees(a: MethodStep, b: MethodStep) -> Tuple[bool, str]:
    """Do two steps at the same position describe the same instruction?

    Thermal steps are compared on their numbers with the standard
    tolerances; everything else (gas, flow, sampling, unknown) is compared on
    kind alone, because a mass flow of 100 vs 0 mL/min is a real difference
    the user should see but is not a reason to call the THERMAL program
    wrong.
    """
    if a.is_thermal and b.is_thermal:
        return _steps_agree(a, b, rate_tol=0.05, target_tol=1.0, duration_tol=0.5)
    if a.kind == b.kind:
        return True, ""
    return False, f"{a.kind} vs {b.kind}"


def merge_programs(orgmethod: MethodProgram, mth: MethodProgram) -> MethodProgram:
    """Per-step source tagging for the Program tab's Source column.

    OrgMethod values are NEVER overwritten: the header is the instrument's own
    record of the method it ran, so a .mth can only confirm a step ("both") or
    disagree with it (the step stays as the header wrote it and the
    disagreement becomes a warning). Steps the .mth has beyond the header's
    length are appended, tagged "mth", so a user who loaded a longer method
    can see what is in it without it masquerading as recorded fact.
    """
    org_ok = orgmethod is not None and bool(orgmethod)
    mth_ok = mth is not None and bool(mth)
    if not org_ok and not mth_ok:
        error = "neither the file header nor a .mth produced a usable program"
        if orgmethod is not None and orgmethod.error:
            error = orgmethod.error
        elif mth is not None and mth.error:
            error = mth.error
        return _unrecognized(error, meta={"format": ORGMETHOD_FORMAT_TAG})
    if not org_ok:
        return replace(mth, warnings=tuple(list(mth.warnings) + [
            "No OrgMethod lines were available in the data file; the program shown comes "
            "from the .mth alone and was not confirmed by the instrument's own record."]))
    if not mth_ok:
        return orgmethod

    org_steps = list(orgmethod.steps)
    mth_steps = list(mth.steps)
    pair_by_position = len(org_steps) == len(mth_steps)

    merged: List[MethodStep] = []
    disagreements: List[str] = []
    if pair_by_position:
        for a, b in zip(org_steps, mth_steps):
            agree, why = _merged_pair_agrees(a, b)
            if agree:
                merged.append(replace(a, source=SOURCE_BOTH))
            else:
                merged.append(a)
                if len(disagreements) < 8:
                    disagreements.append(f"step {a.index}: {why}")
    else:
        # Lengths differ: pair thermal steps in order, which is the part that
        # actually describes the furnace, and leave the housekeeping steps
        # tagged by their own source.
        ta = [s for s in org_steps if s.is_thermal]
        tb = [s for s in mth_steps if s.is_thermal]
        agreeing: Dict[int, bool] = {}
        for i, a in enumerate(ta):
            if i >= len(tb):
                break
            ok, why = _steps_agree(a, tb[i], rate_tol=0.05, target_tol=1.0, duration_tol=0.5)
            agreeing[id(a)] = ok
            if not ok and len(disagreements) < 8:
                disagreements.append(f"thermal step {i + 1}: {why}")
        for s in org_steps:
            merged.append(replace(s, source=SOURCE_BOTH) if agreeing.get(id(s)) else s)
        extra = [replace(s, source=SOURCE_MTH) for s in mth_steps[len(org_steps):]]
        merged.extend(extra)

    warnings = list(orgmethod.warnings)
    for w in mth.warnings:
        if w not in warnings:
            warnings.append(w)
    if disagreements:
        warnings.append("The .mth disagrees with the method recorded in the data file at: "
                        + "; ".join(disagreements)
                        + ". The values recorded by the instrument were kept.")
    if len(mth_steps) > len(org_steps):
        warnings.append(f"The .mth has {len(mth_steps) - len(org_steps)} step(s) beyond the "
                        f"{len(org_steps)} recorded in the data file; they are shown, "
                        f"tagged '.mth', and are not part of what ran.")

    meta = dict(orgmethod.meta)
    meta["mth_format"] = dict(mth.meta).get("format", MTH_FORMAT_TAG)
    meta["mth_source_path"] = mth.source_path
    status = orgmethod.status
    if mth.status == PROGRAM_PARTIAL and status == PROGRAM_PARSED:
        status = PROGRAM_PARSED     # the header is complete; the .mth being partial is a note
    return replace(orgmethod, steps=tuple(merged), source="orgmethod+mth", status=status,
                   warnings=tuple(warnings), meta=meta,
                   proc_name=orgmethod.proc_name or mth.proc_name)
