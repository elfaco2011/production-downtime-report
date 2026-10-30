"""Downtime and OEE KPIs.

Conventions (also documented in the README):

- Events are split at shift boundaries first, so every minute is assigned
  to exactly one (machine, date, shift). Time outside any shift window
  (e.g. a stop running overnight) is ignored. A shift's date is the day
  on which the shift starts.
- A shift counts for a machine if the machine has at least one event in it.
- PLANNED (breaks, preventive maintenance) is not scheduled time:
  planned production time = shift length - PLANNED minutes.
- SETUP (e.g. mold change) is an availability loss, as in the standard
  OEE definition. In the Pareto it is a separate 'setup' category, next
  to unplanned stops.
- Shift minutes with no event at all are 'unlogged': they count as lost
  availability and are reported separately.

  Availability = run time / planned production time
  Performance  = sum(ideal cycle x total parts) / run time,
                 capped at 1 and flagged when the raw value exceeds 1.
                 Only run time of shifts with production data is used.
  Quality      = good parts / total parts
  OEE          = Availability x Performance x Quality

- MTBF = run time / number of unplanned stops;
  MTTR = unplanned stop time / number of unplanned stops.
  Only STOP events count; SETUP and PLANNED do not.
- A stop spanning several shifts is counted once, in the shift where it
  starts; its minutes are split across the shifts it covers.
- Micro-stops: unplanned stops shorter than Config.micro_stop_min (in
  shift minutes). They stay in availability, MTBF and MTTR and are also
  counted and summed separately.
- All ratios are computed from summed minutes and parts, never by
  averaging ratios.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date

import pandas as pd

from .config import Config
from .io import Dataset

MINUTE = pd.Timedelta(minutes=1)
LOSS_CATEGORIES = {"STOP": "unplanned", "SETUP": "setup"}
PERFORMANCE_TOLERANCE = 1e-9

SUM_COLUMNS = [
    "shift_min", "planned_min", "run_min", "stop_min", "setup_min",
    "unlogged_min", "stop_count", "setup_count", "micro_count", "micro_min",
    "missing_reason_count", "run_min_measured", "ideal_time_s",
    "good_parts", "scrap_parts", "total_parts", "perf_over_1_shifts",
]


@dataclass
class Analysis:
    segments: pd.DataFrame
    shifts: pd.DataFrame
    config: Config
    notes: list[str] = field(default_factory=list)
    has_production: bool = True


def analyze(dataset: Dataset) -> Analysis:
    config = dataset.config
    notes = list(dataset.notes)
    segments = split_by_shift(dataset.events, config)

    logged = (dataset.events["end"] - dataset.events["start"]).sum() / MINUTE
    outside = logged - segments["minutes"].sum()
    if outside > 0:
        notes.append(
            f"{outside:,.0f} logged minutes fall outside the shift windows "
            "and are ignored."
        )

    table = shift_table(segments, dataset.production, config)
    if dataset.production is not None:
        keys = ["machine", "date", "shift"]
        matched = dataset.production.merge(table[keys], on=keys, how="left", indicator=True)
        unmatched = (matched["_merge"] == "left_only").sum()
        if unmatched:
            notes.append(
                f"{unmatched} production row(s) have no matching events "
                "(machine, date, shift) and are ignored."
            )
    unlogged = table["unlogged_min"].sum()
    if unlogged > 0:
        notes.append(
            f"{unlogged:,.0f} shift minutes have no event logged; "
            "they count as lost availability."
        )
    over = int(table["perf_over_1_shifts"].sum())
    if over:
        notes.append(
            f"{over} shift(s) have Performance > 100% (ideal cycle time may be "
            "too high, or parts are counted twice); Performance is capped at 100%."
        )
    return Analysis(segments, table, config, notes, dataset.production is not None)


def split_by_shift(events: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Cut events at shift boundaries: one row per event x shift it covers."""
    ev = events.reset_index(drop=True)
    ev["event_id"] = ev.index
    # A shift starting the day before may still be running at `start`.
    first = ev["start"].dt.normalize() - pd.Timedelta(days=1)
    n_days = ((ev["end"].dt.normalize() - first) / pd.Timedelta(days=1)).astype(int) + 1
    rep = ev.loc[ev.index.repeat(n_days)].copy()
    rep["date"] = first.loc[rep.index] + pd.to_timedelta(
        rep.groupby(level=0).cumcount(), unit="D"
    )
    rep = rep.reset_index(drop=True)

    parts = []
    for shift in config.shifts:
        lo = rep["date"] + pd.Timedelta(minutes=shift.start_min)
        hi = lo + pd.Timedelta(minutes=shift.duration_min)
        seg_start = rep["start"].where(rep["start"] > lo, lo)
        seg_end = rep["end"].where(rep["end"] < hi, hi)
        keep = seg_end > seg_start
        part = rep[keep].copy()
        part["shift"] = shift.name
        part["event_start"] = part["start"]
        part["event_end"] = part["end"]
        part["start"] = seg_start[keep]
        part["end"] = seg_end[keep]
        parts.append(part)
    seg = pd.concat(parts).sort_values(["machine", "start"]).reset_index(drop=True)

    seg["minutes"] = (seg["end"] - seg["start"]) / MINUTE
    seg["event_minutes"] = seg.groupby("event_id")["minutes"].transform("sum")
    seg["is_first"] = ~seg["event_id"].duplicated()
    seg["is_micro"] = (seg["state"] == "STOP") & (
        seg["event_minutes"] < config.micro_stop_min
    )
    seg["week"] = iso_week(seg["date"])
    return seg[[
        "event_id", "machine", "date", "week", "shift", "start", "end",
        "minutes", "state", "reason", "reason_missing", "event_start",
        "event_end", "event_minutes", "is_first", "is_micro", "row",
    ]]


def shift_table(
    segments: pd.DataFrame, production: pd.DataFrame | None, config: Config
) -> pd.DataFrame:
    """One row per machine x date x shift with summed minutes and parts."""
    keys = ["machine", "date", "shift"]
    s = segments
    stop = s["state"] == "STOP"
    flags = pd.DataFrame({
        "logged_min": s["minutes"],
        "planned_min": s["minutes"].where(s["state"] == "PLANNED", 0.0),
        "run_min": s["minutes"].where(s["state"] == "RUN", 0.0),
        "stop_min": s["minutes"].where(stop, 0.0),
        "setup_min": s["minutes"].where(s["state"] == "SETUP", 0.0),
        "stop_count": (s["is_first"] & stop).astype(int),
        "setup_count": (s["is_first"] & (s["state"] == "SETUP")).astype(int),
        "micro_count": (s["is_first"] & s["is_micro"]).astype(int),
        "micro_min": s["minutes"].where(s["is_micro"], 0.0),
        "missing_reason_count": (s["is_first"] & stop & s["reason_missing"]).astype(int),
    })
    table = flags.groupby([s[k] for k in keys]).sum().reset_index()
    durations = {sh.name: sh.duration_min for sh in config.shifts}
    table["shift_min"] = table["shift"].map(durations).astype(float)
    table["unlogged_min"] = table["shift_min"] - table["logged_min"]
    table["week"] = iso_week(table["date"])

    if production is not None and not production.empty:
        p = production.assign(
            total_parts=production["good_parts"] + production["scrap_parts"],
            ideal_time_s=production["ideal_cycle_s"]
            * (production["good_parts"] + production["scrap_parts"]),
        ).groupby(keys)[["good_parts", "scrap_parts", "total_parts", "ideal_time_s"]].sum()
        table = table.merge(p.reset_index(), on=keys, how="left", indicator="_has")
        has = table.pop("_has") == "both"
    else:
        has = pd.Series(False, index=table.index)
        for col in ["good_parts", "scrap_parts", "total_parts", "ideal_time_s"]:
            table[col] = 0.0
    table[["good_parts", "scrap_parts", "total_parts", "ideal_time_s"]] = table[
        ["good_parts", "scrap_parts", "total_parts", "ideal_time_s"]
    ].fillna(0.0)
    table["has_production"] = has
    table["run_min_measured"] = table["run_min"].where(has, 0.0)
    raw = table["ideal_time_s"] / (table["run_min_measured"] * 60)
    table["perf_over_1_shifts"] = (raw > 1 + PERFORMANCE_TOLERANCE).astype(int)
    return table.sort_values(keys).reset_index(drop=True)


def kpi_table(shifts: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """KPIs per group, e.g. by=['machine', 'week']; by=[] gives one total row."""
    if by:
        sums = shifts.groupby(by)[SUM_COLUMNS].sum()
    else:
        sums = shifts[SUM_COLUMNS].sum().to_frame().T
    return _ratios(sums).reset_index(drop=not by)


def _ratios(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    t["planned_production_min"] = t["shift_min"] - t["planned_min"]
    t["downtime_min"] = t["stop_min"] + t["setup_min"]
    t["availability"] = _div(t["run_min"], t["planned_production_min"])
    t["performance_raw"] = _div(t["ideal_time_s"], t["run_min_measured"] * 60)
    t["performance"] = t["performance_raw"].clip(upper=1.0)
    t["performance_capped"] = t["performance_raw"] > 1 + PERFORMANCE_TOLERANCE
    t["quality"] = _div(t["good_parts"], t["total_parts"])
    t["oee"] = t["availability"] * t["performance"] * t["quality"]
    t["mtbf_min"] = _div(t["run_min"], t["stop_count"])
    t["mttr_min"] = _div(t["stop_min"], t["stop_count"])
    return t


def _div(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b.where(b > 0)).astype(float)


def summary(analysis: Analysis) -> dict:
    """Headline KPIs for the whole (filtered) analysis."""
    row = kpi_table(analysis.shifts, []).iloc[0]
    return {k: (None if pd.isna(v) else v.item() if hasattr(v, "item") else v)
            for k, v in row.items()}


def pareto(segments: pd.DataFrame) -> pd.DataFrame:
    """Downtime by cause: unplanned stops and setups, largest first."""
    loss = segments[segments["state"].isin(LOSS_CATEGORIES)]
    columns = ["reason", "category", "minutes", "count", "pct", "cum_pct"]
    if loss.empty:
        return pd.DataFrame(columns=columns)
    t = (
        loss.assign(category=loss["state"].map(LOSS_CATEGORIES))
        .groupby(["reason", "category"])
        .agg(minutes=("minutes", "sum"), count=("is_first", "sum"))
        .reset_index()
        .sort_values(["minutes", "reason"], ascending=[False, True])
        .reset_index(drop=True)
    )
    t["pct"] = 100 * t["minutes"] / t["minutes"].sum()
    t["cum_pct"] = t["pct"].cumsum()
    return t[columns]


def top_stops(segments: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Longest unplanned stops by downtime inside shifts."""
    stops = segments[segments["state"] == "STOP"]
    t = (
        stops.groupby("event_id")
        .agg(
            machine=("machine", "first"),
            start=("event_start", "first"),
            end=("event_end", "first"),
            reason=("reason", "first"),
            downtime_min=("minutes", "sum"),
            row=("row", "first"),
        )
        .sort_values(["downtime_min", "start"], ascending=[False, True])
        .head(n)
        .reset_index(drop=True)
    )
    t["duration_min"] = (t["end"] - t["start"]) / MINUTE
    return t[["machine", "start", "end", "reason", "downtime_min", "duration_min", "row"]]


def filter_analysis(
    analysis: Analysis,
    machines: list[str] | None = None,
    start: date | None = None,
    end: date | None = None,
    shifts: list[str] | None = None,
    week: str | None = None,
) -> Analysis:
    """Restrict to machines, an inclusive date range (shift dates), shifts or an ISO week."""

    def mask(df: pd.DataFrame) -> pd.Series:
        m = pd.Series(True, index=df.index)
        if machines is not None:
            m &= df["machine"].isin(machines)
        if start is not None:
            m &= df["date"] >= pd.Timestamp(start)
        if end is not None:
            m &= df["date"] <= pd.Timestamp(end)
        if shifts is not None:
            m &= df["shift"].isin(shifts)
        if week is not None:
            m &= df["week"] == week
        return m

    return replace(
        analysis,
        segments=analysis.segments[mask(analysis.segments)],
        shifts=analysis.shifts[mask(analysis.shifts)],
    )


def iso_week(dates: pd.Series) -> pd.Series:
    iso = dates.dt.isocalendar()
    return iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)

