"""Downtime and OEE KPIs.

Conventions (also documented in the README):

Time base
- Events are split at shift boundaries first, so every minute belongs to
  exactly one (machine, date, shift). Time outside any shift window (e.g.
  a stop running overnight) is ignored and reported. A shift's date is
  the day on which the shift starts.
- A shift counts for a machine if the machine has at least one event in
  it. Machine-days with no event while other machines ran are reported as
  a data quality issue, not counted (there is no per-machine calendar).
- PLANNED (breaks, preventive maintenance) is not scheduled time:
  planned production time = shift length - PLANNED minutes.
- Unlogged time (shift minutes with no event) is, with unlogged_time:
  'loss' (default), an availability loss; 'exclude', also removed from
  planned production time.

Six big losses
- Breakdowns: unplanned stops (STOP) -> Availability.
- Setup and adjustments: SETUP, e.g. mold change -> Availability, as in
  the standard OEE definition. In the all-losses Pareto it is its own
  category, separate from breakdowns.
- Small stops: micro-stops, i.e. unplanned stops shorter than
  micro_stop_min (default 2 min, measured inside shifts). With
  micro_stops_as='performance' (default) their time counts as run time,
  so they lower Performance; with 'availability' they lower Availability.
  OEE is the same either way. They are always excluded from MTBF, MTTR
  and the stop count, and are reported separately.
- Reduced speed: lower Performance (cycle slower than ideal).
- Startup and production rejects: lower Quality (scrap).

Formulas
  run time     = RUN minutes (+ micro-stop minutes if micro_stops_as is
                 'performance')
  Availability = run time / planned production time
  Performance  = sum(ideal cycle x total parts) / run time, capped at 1
                 and flagged when the raw value exceeds 1. Only run time
                 of shifts with production data is used.
  Quality      = good parts / total parts
  OEE          = Availability x Performance x Quality
  MTBF         = run time / number of unplanned stops (micro excluded)
  MTTR         = unplanned stop minutes / number of unplanned stops
  Downtime     = unplanned stops + setups (+ micro-stops if they count
                 against availability); unlogged time is reported apart.

- A stop spanning several shifts is counted once, in the shift where it
  starts; its minutes are split across the shifts it covers.
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
PERFORMANCE_TOLERANCE = 1e-9
PARETO_VIEWS = ("unplanned", "all_losses")
MICRO_STOPS_LABEL = "Micro-stops"
UNLOGGED_LABEL = "Unlogged"

SUM_COLUMNS = [
    "shift_min", "planned_min", "planned_production_min", "run_min",
    "run_min_measured", "stop_min", "setup_min", "micro_min", "unlogged_min",
    "downtime_min", "stop_count", "setup_count", "micro_count",
    "unplanned_event_count", "missing_reason_count", "ideal_time_s",
    "good_parts", "scrap_parts", "total_parts", "perf_over_1_shifts",
]


@dataclass
class Analysis:
    segments: pd.DataFrame
    shifts: pd.DataFrame
    config: Config
    notes: list[str] = field(default_factory=list)
    has_production: bool = True
    # Logged minutes outside every shift window (whole file, not filtered).
    outside_min: float = 0.0
    # Machine-days with no event while other machines have events.
    missing_machine_days: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Rows skipped by the loader or not matched to any shift.
    dropped_rows: pd.DataFrame = field(default_factory=pd.DataFrame)


def analyze(dataset: Dataset) -> Analysis:
    config = dataset.config
    segments = split_by_shift(dataset.events, config)
    table = shift_table(segments, dataset.production, config)

    logged = (dataset.events["end"] - dataset.events["start"]).sum() / MINUTE
    dropped = [dataset.dropped_rows]
    if dataset.production is not None:
        keys = ["machine", "date", "shift"]
        matched = dataset.production.merge(
            table[keys], on=keys, how="left", indicator=True
        )
        unmatched = matched[matched["_merge"] == "left_only"]
        dropped.append(pd.DataFrame({
            "what": "production",
            "row": unmatched["row"],
            "reason": "no events for this machine, date and shift",
        }))
    return Analysis(
        segments=segments,
        shifts=table,
        config=config,
        notes=list(dataset.notes),
        has_production=dataset.production is not None,
        outside_min=max(0.0, logged - segments["minutes"].sum()),
        missing_machine_days=_missing_machine_days(segments),
        dropped_rows=pd.concat(
            [d for d in dropped if not d.empty], ignore_index=True
        ) if any(not d.empty for d in dropped) else dataset.dropped_rows,
    )


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
    """One row per machine x date x shift, with additive columns only.

    The loss-treatment settings are applied here, so any grouping of this
    table can be summed and turned into KPIs by kpi_table().
    """
    keys = ["machine", "date", "shift"]
    s = segments
    unplanned = s["state"] == "STOP"
    stop = unplanned & ~s["is_micro"]
    first = s["is_first"]
    flags = pd.DataFrame({
        "logged_min": s["minutes"],
        "planned_min": s["minutes"].where(s["state"] == "PLANNED", 0.0),
        "run_min": s["minutes"].where(s["state"] == "RUN", 0.0),
        "stop_min": s["minutes"].where(stop, 0.0),
        "setup_min": s["minutes"].where(s["state"] == "SETUP", 0.0),
        "micro_min": s["minutes"].where(s["is_micro"], 0.0),
        "stop_count": (first & stop).astype(int),
        "setup_count": (first & (s["state"] == "SETUP")).astype(int),
        "micro_count": (first & s["is_micro"]).astype(int),
        "unplanned_event_count": (first & unplanned).astype(int),
        "missing_reason_count": (first & unplanned & s["reason_missing"]).astype(int),
    })
    table = flags.groupby([s[k] for k in keys]).sum().reset_index()
    durations = {sh.name: sh.duration_min for sh in config.shifts}
    table["shift_min"] = table["shift"].map(durations).astype(float)
    # Rounded so that float noise from second timestamps is not reported.
    table["unlogged_min"] = (
        (table["shift_min"] - table["logged_min"]).round(6).clip(lower=0)
    )
    table["week"] = iso_week(table["date"])

    micro_in_run = config.micro_stops_as == "performance"
    if micro_in_run:
        table["run_min"] += table["micro_min"]
    table["downtime_min"] = table["stop_min"] + table["setup_min"]
    if not micro_in_run:
        table["downtime_min"] += table["micro_min"]
    table["planned_production_min"] = table["shift_min"] - table["planned_min"]
    if config.unlogged_time == "exclude":
        table["planned_production_min"] -= table["unlogged_min"]

    parts = ["good_parts", "scrap_parts", "total_parts", "ideal_time_s"]
    if production is not None and not production.empty:
        p = production.assign(
            total_parts=production["good_parts"] + production["scrap_parts"],
            ideal_time_s=production["ideal_cycle_s"]
            * (production["good_parts"] + production["scrap_parts"]),
        ).groupby(keys)[parts].sum()
        table = table.merge(p.reset_index(), on=keys, how="left", indicator="_has")
        has = table.pop("_has") == "both"
    else:
        has = pd.Series(False, index=table.index)
        for col in parts:
            table[col] = 0.0
    table[parts] = table[parts].fillna(0.0)
    table["has_production"] = has
    table["run_min_measured"] = table["run_min"].where(has, 0.0)
    table["performance_raw"] = _div(table["ideal_time_s"], table["run_min_measured"] * 60)
    table["perf_over_1_shifts"] = (
        table["performance_raw"] > 1 + PERFORMANCE_TOLERANCE
    ).astype(int)
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


def pareto(analysis: Analysis, view: str = "unplanned") -> pd.DataFrame:
    """Downtime by cause, largest first.

    view='unplanned': unplanned stops by reason, micro-stops excluded.
    view='all_losses': unplanned stops, setups by reason, micro-stops as one
    bar and unlogged time (when counted as a loss).
    count is the number of events; for Unlogged, the number of shifts.
    """
    if view not in PARETO_VIEWS:
        raise ValueError(f"view must be one of {PARETO_VIEWS}")
    s = analysis.segments
    columns = ["reason", "category", "minutes", "count", "pct", "cum_pct"]

    def by_reason(mask: pd.Series, category: str) -> pd.DataFrame:
        return (
            s[mask].groupby("reason")
            .agg(minutes=("minutes", "sum"), count=("is_first", "sum"))
            .reset_index()
            .assign(category=category)
        )

    parts = [by_reason((s["state"] == "STOP") & ~s["is_micro"], "unplanned")]
    if view == "all_losses":
        parts.append(by_reason(s["state"] == "SETUP", "setup"))
        micro = s[s["is_micro"]]
        parts.append(pd.DataFrame([{
            "reason": MICRO_STOPS_LABEL, "category": "micro",
            "minutes": micro["minutes"].sum(), "count": micro["is_first"].sum(),
        }]))
        if analysis.config.unlogged_time == "loss":
            unlogged = analysis.shifts["unlogged_min"]
            parts.append(pd.DataFrame([{
                "reason": UNLOGGED_LABEL, "category": "unlogged",
                "minutes": unlogged.sum(), "count": (unlogged > 0).sum(),
            }]))
    t = pd.concat(parts, ignore_index=True)
    t = t[t["minutes"] > 0]
    if t.empty:
        return pd.DataFrame(columns=columns)
    t = t.sort_values(["minutes", "reason"], ascending=[False, True]).reset_index(drop=True)
    t["count"] = t["count"].astype(int)
    t["pct"] = 100 * t["minutes"] / t["minutes"].sum()
    t["cum_pct"] = t["pct"].cumsum()
    return t[columns]


def micro_stops(analysis: Analysis, top: int = 5) -> dict:
    """Micro-stop totals and the causes with the most minutes."""
    m = analysis.segments[analysis.segments["is_micro"]]
    causes = (
        m.groupby("reason")
        .agg(count=("is_first", "sum"), minutes=("minutes", "sum"))
        .sort_values(["minutes", "count"], ascending=False)
        .head(top)
        .reset_index()
    )
    return {
        "count": int(m["is_first"].sum()),
        "minutes": float(m["minutes"].sum()),
        "threshold_min": analysis.config.micro_stop_min,
        "counted_as": analysis.config.micro_stops_as,
        "top_causes": causes,
    }


def stop_events(analysis: Analysis, include_micro: bool = False) -> pd.DataFrame:
    """Unplanned stops as whole events, longest first.

    downtime_min counts minutes inside shifts; duration_min is the real
    duration from the original start to end, including any time outside
    the shift calendar.
    """
    s = analysis.segments
    mask = s["state"] == "STOP"
    if not include_micro:
        mask &= ~s["is_micro"]
    columns = [
        "machine", "start", "end", "reason", "downtime_min", "duration_min",
        "micro", "row",
    ]
    if not mask.any():
        return pd.DataFrame(columns=columns)
    t = (
        s[mask].groupby("event_id")
        .agg(
            machine=("machine", "first"),
            start=("event_start", "first"),
            end=("event_end", "first"),
            reason=("reason", "first"),
            downtime_min=("minutes", "sum"),
            micro=("is_micro", "first"),
            row=("row", "first"),
        )
        .sort_values(["downtime_min", "start"], ascending=[False, True])
        .reset_index(drop=True)
    )
    t["duration_min"] = (t["end"] - t["start"]) / MINUTE
    return t[columns]


def top_stops(analysis: Analysis, n: int = 10) -> pd.DataFrame:
    """The n longest unplanned stops, ranked by minutes inside shifts."""
    return stop_events(analysis).head(n).drop(columns="micro")


@dataclass
class DataQuality:
    stops_total: int
    stops_without_reason: int
    shift_min: float
    unlogged_min: float
    outside_shift_min: float
    machine_days_without_events: pd.DataFrame
    shifts_performance_over_1: pd.DataFrame
    dropped_rows: pd.DataFrame

    @property
    def stops_without_reason_pct(self) -> float:
        return 100 * self.stops_without_reason / self.stops_total if self.stops_total else 0.0

    @property
    def unlogged_pct(self) -> float:
        return 100 * self.unlogged_min / self.shift_min if self.shift_min else 0.0

    def rows(self) -> pd.DataFrame:
        """One line per check, for display in the dashboard and the PDF.

        status is "warning" or "ok" for checks that can point to bad data,
        and "info" for facts that are expected by design: time logged
        outside shift hours is simply not part of the analysis.
        """
        checks = [
            ("Stops without reason code",
             f"{self.stops_without_reason} of {self.stops_total} "
             f"({self.stops_without_reason_pct:.1f}%)",
             self.stops_without_reason > 0),
            ("Unlogged shift time",
             f"{self.unlogged_min:,.0f} min ({self.unlogged_pct:.1f}% of shift time)",
             self.unlogged_min > 0),
            ("Machine-days without events",
             str(len(self.machine_days_without_events)),
             not self.machine_days_without_events.empty),
            ("Shifts with Performance > 100% (capped)",
             str(len(self.shifts_performance_over_1)),
             not self.shifts_performance_over_1.empty),
            ("Rows dropped while loading", str(len(self.dropped_rows)),
             not self.dropped_rows.empty),
        ]
        rows = [(c, v, "warning" if w else "ok") for c, v, w in checks]
        rows.append(("Outside shift hours",
                     f"{self.outside_shift_min:,.0f} min — ignored by design", "info"))
        table = pd.DataFrame(rows, columns=["check", "value", "status"])
        table["warning"] = table["status"] == "warning"
        return table


def data_quality(analysis: Analysis) -> DataQuality:
    s = analysis.segments
    unplanned_first = s[(s["state"] == "STOP") & s["is_first"]]
    shifts = analysis.shifts
    over = shifts[shifts["perf_over_1_shifts"] > 0]
    return DataQuality(
        stops_total=len(unplanned_first),
        stops_without_reason=int(unplanned_first["reason_missing"].sum()),
        shift_min=float(shifts["shift_min"].sum()),
        unlogged_min=float(shifts["unlogged_min"].sum()),
        outside_shift_min=analysis.outside_min,
        machine_days_without_events=analysis.missing_machine_days,
        shifts_performance_over_1=over[["machine", "date", "shift", "performance_raw"]]
        .reset_index(drop=True),
        dropped_rows=analysis.dropped_rows,
    )


def _missing_machine_days(segments: pd.DataFrame) -> pd.DataFrame:
    """(machine, date) pairs with no event on a day when other machines ran."""
    present = segments[["machine", "date"]].drop_duplicates()
    grid = pd.MultiIndex.from_product(
        [sorted(present["machine"].unique()), sorted(present["date"].unique())],
        names=["machine", "date"],
    ).to_frame(index=False)
    missing = grid.merge(present, how="left", indicator=True)
    missing = missing[missing["_merge"] == "left_only"].drop(columns="_merge")
    return missing.assign(week=iso_week(missing["date"])).reset_index(drop=True)


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
        if df.empty:
            return m
        if machines is not None:
            m &= df["machine"].isin(machines)
        if start is not None:
            m &= df["date"] >= pd.Timestamp(start)
        if end is not None:
            m &= df["date"] <= pd.Timestamp(end)
        if shifts is not None and "shift" in df:
            m &= df["shift"].isin(shifts)
        if week is not None:
            m &= df["week"] == week
        return m

    missing = analysis.missing_machine_days
    return replace(
        analysis,
        segments=analysis.segments[mask(analysis.segments)],
        shifts=analysis.shifts[mask(analysis.shifts)],
        missing_machine_days=missing[mask(missing)],
    )


def iso_week(dates: pd.Series) -> pd.Series:
    iso = dates.dt.isocalendar()
    return iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)


def downtime_by_hour(analysis: Analysis) -> pd.DataFrame:
    """Unplanned stop minutes (micro-stops excluded) by weekday x hour of day.

    Stops are cut at hour boundaries, so a stop from 09:50 to 10:20 adds
    10 minutes to 09:00 and 20 minutes to 10:00.
    """
    s = analysis.segments
    stops = s[(s["state"] == "STOP") & ~s["is_micro"]][["start", "end"]]
    columns = ["weekday", "hour", "minutes"]
    if stops.empty:
        return pd.DataFrame(columns=columns)
    first = stops["start"].dt.floor("h")
    n_hours = ((stops["end"] - pd.Timedelta(microseconds=1)).dt.floor("h") - first)
    n_hours = (n_hours / pd.Timedelta(hours=1)).astype(int) + 1
    rep = stops.loc[stops.index.repeat(n_hours)].copy()
    rep["hour_start"] = first.loc[rep.index] + pd.to_timedelta(
        rep.groupby(level=0).cumcount(), unit="h"
    )
    rep = rep.reset_index(drop=True)
    hour_end = rep["hour_start"] + pd.Timedelta(hours=1)
    lo = rep["start"].where(rep["start"] > rep["hour_start"], rep["hour_start"])
    hi = rep["end"].where(rep["end"] < hour_end, hour_end)
    rep["minutes"] = (hi - lo) / MINUTE
    rep["weekday"] = rep["hour_start"].dt.day_name()
    rep["hour"] = rep["hour_start"].dt.hour
    return rep.groupby(["weekday", "hour"], as_index=False)["minutes"].sum()[columns]
