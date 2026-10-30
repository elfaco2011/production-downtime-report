"""Synthetic data for three injection molding machines.

Generates a machine event log and a per-shift production log that look like
a typical MES export, with a fixed seed so the output is reproducible.

Event log model: every shift is split into contiguous, non-overlapping
segments (RUN / STOP / SETUP / PLANNED) that exactly cover the shift.
Unplanned stops arrive as a Poisson process; their durations are lognormal,
which gives many short stops and a few long ones. A stop still running at
shift end usually ends there, but some carry over into the next shift (or
into the next morning), as happens in real logs. IMM-02 has a deliberately
worse mold temperature alarm problem so the analysis has a story to tell.

Usage:
    python -m pdr.synth --out data/sample
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
MACHINES = ("IMM-01", "IMM-02", "IMM-03")
FIRST_MONDAY = date(2026, 8, 31)  # ISO week 2026-W36
N_WEEKS = 8
SHIFTS = {"A": (time(6, 0), time(14, 0)), "B": (time(14, 0), time(22, 0))}
SHIFT_MIN = 480

EVENT_COLUMNS = ["machine", "start", "end", "state", "reason"]
PRODUCTION_COLUMNS = [
    "machine", "date", "shift", "ideal_cycle_s", "good_parts", "scrap_parts",
]

# Column names used in the "foreign" XLSX export, to demonstrate mapping.
XLSX_EVENT_COLUMNS = {
    "machine": "Macchina",
    "start": "Inizio",
    "end": "Fine",
    "state": "Stato",
    "reason": "Causa",
}
XLSX_PRODUCTION_COLUMNS = {
    "machine": "Macchina",
    "date": "Data",
    "shift": "Turno",
    "ideal_cycle_s": "Ciclo ideale (s)",
    "good_parts": "Pezzi buoni",
    "scrap_parts": "Scarti",
}


@dataclass(frozen=True)
class StopCause:
    """Unplanned stop cause: expected count per shift and lognormal duration."""

    per_shift: float
    median_min: float
    sigma: float


BASE_CAUSES = {
    "nozzle clog": StopCause(1.8, 4, 0.5),
    "robot fault": StopCause(1.2, 6, 0.6),
    "quality check hold": StopCause(0.6, 15, 0.5),
    "material shortage": StopCause(0.4, 25, 0.8),
    "no operator": StopCause(0.4, 20, 0.6),
    "mold temperature alarm": StopCause(0.4, 10, 0.6),
}
# IMM-02 has a chronic mold temperature control issue.
MACHINE_CAUSE_OVERRIDES = {
    "IMM-02": {"mold temperature alarm": StopCause(1.8, 20, 0.7)},
}
MAX_STOP_MIN = 240
# Share of stops overrunning shift end that continue into the next shift.
CARRY_OVER_PROBABILITY = 0.5

IDEAL_CYCLE_S = {"IMM-01": 24.0, "IMM-02": 30.0, "IMM-03": 18.0}
PERFORMANCE_MEAN = {"IMM-01": 0.88, "IMM-02": 0.85, "IMM-03": 0.90}
BASE_SCRAP_RATE = 0.015
SCRAP_PER_MOLD_TEMP_ALARM = 0.004

MOLD_CHANGES_PER_WEEK = 2
# Preventive maintenance every other week, Friday shift B, first 3 hours.
PM_WEEK_PARITY = {"IMM-01": 0, "IMM-02": 1, "IMM-03": 0}
PM_DURATION_MIN = 180
# One 30-minute break per shift, centered on mid-shift +/- 30 minutes.
BREAK_MIN = 30
BREAK_JITTER_MIN = 30


def _causes_for(machine: str) -> dict[str, StopCause]:
    causes = dict(BASE_CAUSES)
    causes.update(MACHINE_CAUSE_OVERRIDES.get(machine, {}))
    return causes


def _stop_duration(rng: np.random.Generator, cause: StopCause) -> int:
    minutes = rng.lognormal(np.log(cause.median_min), cause.sigma)
    return int(np.clip(round(minutes), 1, MAX_STOP_MIN))


def _fill_gap(
    rng: np.random.Generator,
    causes: dict[str, StopCause],
    start: int,
    end: int,
) -> tuple[list[tuple[int, int, str, str]], int]:
    """Fill [start, end) minutes with alternating RUN and unplanned STOP.

    Returns the segments and the minutes by which the last stop overran
    `end` (0 if it did not); the overrun part is not in the segments.
    """
    names = list(causes)
    rates = np.array([causes[n].per_shift for n in names]) / SHIFT_MIN
    total_rate = rates.sum()
    probs = rates / total_rate

    segments = []
    overrun = 0
    cursor = start
    while cursor < end:
        run = max(1, round(rng.exponential(1 / total_rate)))
        run_end = min(cursor + run, end)
        segments.append((cursor, run_end, "RUN", ""))
        cursor = run_end
        if cursor >= end:
            break
        name = names[rng.choice(len(names), p=probs)]
        stop_end = cursor + _stop_duration(rng, causes[name])
        overrun = max(0, stop_end - end)
        segments.append((cursor, min(stop_end, end), "STOP", name))
        cursor = stop_end
    return segments, overrun


def _shift_segments(
    rng: np.random.Generator,
    causes: dict[str, StopCause],
    planned_blocks: list[tuple[int, int, str, str]],
    start: int = 0,
) -> tuple[list[tuple[int, int, str, str]], int]:
    """Build the timeline of one shift from `start`, in minutes.

    Returns the segments and the overrun of a stop still running at shift
    end. Overruns into a planned block are dropped: the stop ends there.
    """
    segments = []
    cursor = start
    for block in sorted(planned_blocks):
        gap, _ = _fill_gap(rng, causes, cursor, block[0])
        segments += gap
        segments.append(block)
        cursor = block[1]
    gap, overrun = _fill_gap(rng, causes, cursor, SHIFT_MIN)
    return segments + gap, overrun


def _working_days() -> list[date]:
    return [
        FIRST_MONDAY + timedelta(weeks=w, days=d)
        for w in range(N_WEEKS)
        for d in range(5)
    ]


def _shift_slots() -> list[tuple[date, str]]:
    """All (day, shift) slots in chronological order."""
    return [(day, shift) for day in _working_days() for shift in SHIFTS]


def _can_carry_over(slot: tuple[date, str], next_slot: tuple[date, str] | None) -> bool:
    """A stop may continue into the next slot, but not over a weekend."""
    if next_slot is None:
        return False
    return (next_slot[0] - slot[0]).days <= 1


def _mold_change_slots(rng: np.random.Generator) -> set[tuple[date, str]]:
    """Pick the (day, shift) slots that contain a mold change."""
    slots = set()
    for w in range(N_WEEKS):
        week_slots = [
            (FIRST_MONDAY + timedelta(weeks=w, days=d), s)
            for d in range(5)
            for s in SHIFTS
        ]
        for i in rng.choice(len(week_slots), MOLD_CHANGES_PER_WEEK, replace=False):
            slots.add(week_slots[i])
    return slots


def _planned_blocks(
    rng: np.random.Generator,
    machine: str,
    day: date,
    shift: str,
    mold_change_slots: set[tuple[date, str]],
) -> list[tuple[int, int, str, str]]:
    blocks = []
    week = (day - FIRST_MONDAY).days // 7
    is_pm = (
        day.weekday() == 4
        and shift == "B"
        and week % 2 == PM_WEEK_PARITY[machine]
    )
    if is_pm:
        blocks.append((0, PM_DURATION_MIN, "PLANNED", "preventive maintenance"))
    center = SHIFT_MIN // 2 + int(rng.integers(-BREAK_JITTER_MIN, BREAK_JITTER_MIN + 1))
    break_start = center - BREAK_MIN // 2
    blocks.append((break_start, break_start + BREAK_MIN, "PLANNED", "break"))
    if (day, shift) in mold_change_slots:
        duration = int(np.clip(round(rng.lognormal(np.log(45), 0.3)), 20, 120))
        blocks.append(_place_block(rng, blocks, duration, "SETUP", "mold change"))
    return sorted(blocks)


def _place_block(
    rng: np.random.Generator,
    blocks: list[tuple[int, int, str, str]],
    duration: int,
    state: str,
    reason: str,
    margin: int = 30,
) -> tuple[int, int, str, str]:
    """Place a block at a random offset that does not touch existing blocks."""
    for _ in range(1000):
        start = int(rng.integers(margin, SHIFT_MIN - duration - margin))
        end = start + duration
        if all(end <= s or start >= e for s, e, *_ in blocks):
            return (start, end, state, reason)
    raise RuntimeError(f"no room for a {duration}-minute {reason} block")


def generate(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (events, production) DataFrames with the canonical columns."""
    rng = np.random.default_rng(seed)
    event_rows = []
    production_rows = []

    slots = _shift_slots()
    for machine in MACHINES:
        causes = _causes_for(machine)
        mold_change_slots = _mold_change_slots(rng)
        machine_rows: list[list] = []
        carry = 0  # minutes of the previous shift's last stop still to run
        for i, (day, shift) in enumerate(slots):
            t0 = datetime.combine(day, SHIFTS[shift][0])
            blocks = _planned_blocks(rng, machine, day, shift, mold_change_slots)
            # A carried-over stop extends the previous event and ends at the
            # latest when the first planned block of this shift begins.
            carry = min(carry, blocks[0][0])
            if carry:
                machine_rows[-1][2] = t0 + timedelta(minutes=carry)
            segments, overrun = _shift_segments(rng, causes, blocks, carry)
            for start, end, state, reason in segments:
                machine_rows.append([
                    machine,
                    t0 + timedelta(minutes=start),
                    t0 + timedelta(minutes=end),
                    state,
                    reason,
                ])
            production_rows.append(
                _production_row(rng, machine, day, shift, segments)
            )
            next_slot = slots[i + 1] if i + 1 < len(slots) else None
            carry_over = (
                overrun > 0
                and _can_carry_over((day, shift), next_slot)
                and rng.random() < CARRY_OVER_PROBABILITY
            )
            carry = overrun if carry_over else 0
        event_rows += machine_rows

    events = pd.DataFrame(event_rows, columns=EVENT_COLUMNS)
    production = pd.DataFrame(production_rows, columns=PRODUCTION_COLUMNS)
    validate_events(events)
    return events, production


def _production_row(
    rng: np.random.Generator,
    machine: str,
    day: date,
    shift: str,
    segments: list[tuple[int, int, str, str]],
) -> tuple:
    run_s = 60 * sum(e - s for s, e, state, _ in segments if state == "RUN")
    alarms = sum(1 for *_, r in segments if r == "mold temperature alarm")
    cycle = IDEAL_CYCLE_S[machine]
    performance = np.clip(rng.normal(PERFORMANCE_MEAN[machine], 0.03), 0.75, 0.98)
    total = int(run_s * performance / cycle)
    scrap_rate = BASE_SCRAP_RATE + SCRAP_PER_MOLD_TEMP_ALARM * alarms
    scrap = int(rng.binomial(total, min(scrap_rate, 0.5)))
    return (machine, day, shift, cycle, total - scrap, scrap)


def validate_events(events: pd.DataFrame) -> None:
    """Raise ValueError if the generated event log is inconsistent.

    Checks: no overlaps per machine, every event starts and ends inside a
    shift window, and only unplanned stops may span more than one shift.
    """
    for machine, group in events.groupby("machine"):
        g = group.sort_values("start")
        if (g["end"] <= g["start"]).any():
            raise ValueError(f"{machine}: event with end <= start")
        overlap = g["start"].iloc[1:].to_numpy() < g["end"].iloc[:-1].to_numpy()
        if overlap.any():
            raise ValueError(f"{machine}: overlapping events")

    start_shift = _shift_index(events["start"], closed="left")
    end_shift = _shift_index(events["end"], closed="right")
    outside = start_shift.isna() | end_shift.isna()
    if outside.any():
        raise ValueError(f"{outside.sum()} events outside shift windows")
    spans = (start_shift != end_shift) & (events["state"] != "STOP")
    if spans.any():
        raise ValueError(f"{spans.sum()} non-STOP events span several shifts")


def _shift_index(ts: pd.Series, closed: str) -> pd.Series:
    """Label each timestamp with 'YYYY-MM-DD/shift', or NaN if outside shifts.

    closed='left' matches [shift start, shift end), 'right' matches
    (shift start, shift end], so an event ending at 14:00 belongs to shift A.
    """
    label = pd.Series(pd.NA, index=ts.index, dtype="object")
    day = ts.dt.normalize()
    for name, (shift_start, shift_end) in SHIFTS.items():
        lo = day + pd.Timedelta(hours=shift_start.hour, minutes=shift_start.minute)
        hi = day + pd.Timedelta(hours=shift_end.hour, minutes=shift_end.minute)
        inside = (ts >= lo) & (ts < hi) if closed == "left" else (ts > lo) & (ts <= hi)
        label[inside] = day[inside].dt.strftime("%Y-%m-%d") + "/" + name
    return label


def write_sample(out_dir: Path, seed: int = SEED) -> list[Path]:
    """Write events.csv, production.csv and a renamed-column XLSX export."""
    out_dir.mkdir(parents=True, exist_ok=True)
    events, production = generate(seed)

    events_csv = out_dir / "events.csv"
    production_csv = out_dir / "production.csv"
    xlsx = out_dir / "plant_export.xlsx"

    events.to_csv(events_csv, index=False, date_format="%Y-%m-%d %H:%M")
    production.to_csv(production_csv, index=False)
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        events.rename(columns=XLSX_EVENT_COLUMNS).to_excel(
            writer, sheet_name="Eventi", index=False
        )
        production.rename(columns=XLSX_PRODUCTION_COLUMNS).to_excel(
            writer, sheet_name="Produzione", index=False
        )
    return [events_csv, production_csv, xlsx]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/sample"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    for path in write_sample(args.out, args.seed):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
