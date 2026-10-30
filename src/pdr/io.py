"""Load event and production logs from CSV or XLSX into canonical tables.

All problems found in a file are collected and raised together as an
InputError, each with the spreadsheet row number (header = row 1), so a
user can fix the file in one pass.

Canonical events: machine, start, end, state, reason, reason_missing, row.
- state is one of RUN / STOP / SETUP / PLANNED, mapped from the source
  labels through Config.state_values.
- An empty reason on a non-RUN event becomes "Unspecified" and
  reason_missing is set, so those stops still show up in the Pareto.

Canonical production: machine, date, shift, ideal_cycle_s, good_parts,
scrap_parts, row. Several rows per machine and shift are allowed (e.g.
one per part number); they are summed downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

import pandas as pd

from .config import EVENT_FIELDS, PRODUCTION_FIELDS, STATES, Config

UNSPECIFIED = "Unspecified"
MAX_LISTED_PROBLEMS = 20

Source = str | Path | BinaryIO


class InputError(ValueError):
    """The input file cannot be analysed; `problems` lists every reason."""

    def __init__(self, what: str, problems: list[str]):
        self.what = what
        self.problems = sorted(problems, key=_row_of)
        listed = problems[:MAX_LISTED_PROBLEMS]
        lines = [f"{what}: {len(problems)} problem(s) found"]
        lines += [f"  - {p}" for p in listed]
        if len(problems) > len(listed):
            lines.append(f"  ... and {len(problems) - len(listed)} more")
        super().__init__("\n".join(lines))


def _row_of(problem: str) -> float:
    """Sort key: problems about a row by row number, general ones last."""
    head = problem.split(":", 1)[0]
    if head.startswith("row ") and head[4:].isdigit():
        return int(head[4:])
    return float("inf")


@dataclass
class Dataset:
    events: pd.DataFrame
    production: pd.DataFrame | None
    config: Config
    notes: list[str] = field(default_factory=list)
    # Rows skipped by the loader: columns what, row, reason.
    dropped_rows: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["what", "row", "reason"])
    )


def load_dataset(
    events_source: Source,
    production_source: Source | None = None,
    config: Config | None = None,
) -> Dataset:
    """Load events and, optionally, production data.

    An XLSX events source may hold both tables in separate sheets; then
    production is read from the same workbook unless given explicitly.
    """
    config = config or Config()
    notes: list[str] = []
    dropped: list[pd.DataFrame] = []

    if _is_excel(events_source):
        sheets = _read_workbook(events_source)
        events_raw, events_what = _pick_sheet(
            sheets, config.events_sheet, index=0, what="events"
        )
        if production_source is None and (config.production_sheet or len(sheets) > 1):
            production_raw, production_what = _pick_sheet(
                sheets, config.production_sheet, index=1, what="production"
            )
        else:
            production_raw = production_what = None
    else:
        events_raw, events_what = _read_csv(events_source), "events"
        production_raw = production_what = None

    if production_source is not None:
        if _is_excel(production_source):
            production_raw, production_what = _pick_sheet(
                _read_workbook(production_source),
                config.production_sheet,
                index=0,
                what="production",
            )
        else:
            production_raw, production_what = _read_csv(production_source), "production"

    events = parse_events(events_raw, config, events_what, dropped)
    production = None
    if production_raw is not None:
        production = parse_production(production_raw, config, production_what)
    else:
        notes.append(
            "No production data: Performance, Quality and OEE are not available."
        )
    dropped_rows = (
        pd.concat(dropped, ignore_index=True)
        if dropped
        else pd.DataFrame(columns=["what", "row", "reason"])
    )
    return Dataset(events, production, config, notes, dropped_rows)


def parse_events(
    raw: pd.DataFrame,
    config: Config,
    what: str = "events",
    dropped: list[pd.DataFrame] | None = None,
) -> pd.DataFrame:
    df = _select_columns(raw, config.event_columns, EVENT_FIELDS, what)
    rows = _row_numbers(df)
    problems: list[str] = []
    columns = config.event_columns

    machine = _clean_text(df["machine"])
    for i in machine.index[machine == ""]:
        problems.append(f"row {rows[i]}: missing machine ('{columns['machine']}')")

    start = _parse_datetimes(
        df["start"], config, rows, columns["start"], problems
    )
    end = _parse_datetimes(df["end"], config, rows, columns["end"], problems)
    reversed_ = start.notna() & end.notna() & (end < start)
    for i in reversed_.index[reversed_]:
        problems.append(
            f"row {rows[i]}: end ({end[i]}) is before start ({start[i]})"
        )

    label = _clean_text(df["state"])
    state = label.str.casefold().map(config.state_values)
    unknown = state.isna()
    if unknown.any():
        problems.append(_unknown_states_message(label[unknown], rows, columns["state"]))

    if problems:
        raise InputError(what, problems)

    reason = _clean_text(df["reason"])
    reason_missing = (reason == "") & (state != "RUN")
    reason = reason.mask(reason_missing, UNSPECIFIED)
    events = pd.DataFrame({
        "machine": machine,
        "start": start,
        "end": end,
        "state": state,
        "reason": reason,
        "reason_missing": reason_missing,
        "row": rows,
    })

    zero = events["end"] == events["start"]
    if zero.any() and dropped is not None:
        dropped.append(pd.DataFrame({
            "what": what, "row": events.loc[zero, "row"], "reason": "zero-length event",
        }))
    events = events[~zero].sort_values(["machine", "start"]).reset_index(drop=True)

    overlaps = _overlap_problems(events)
    if overlaps:
        raise InputError(what, overlaps)
    return events


def parse_production(
    raw: pd.DataFrame, config: Config, what: str = "production"
) -> pd.DataFrame:
    df = _select_columns(raw, config.production_columns, PRODUCTION_FIELDS, what)
    rows = _row_numbers(df)
    problems: list[str] = []
    columns = config.production_columns

    machine = _clean_text(df["machine"])
    for i in machine.index[machine == ""]:
        problems.append(f"row {rows[i]}: missing machine ('{columns['machine']}')")

    day = _parse_datetimes(
        df["date"], config, rows, columns["date"], problems, fmt=config.date_format
    ).dt.normalize()

    shift = _clean_text(df["shift"])
    known = {s.name for s in config.shifts}
    for i in shift.index[~shift.isin(known)]:
        problems.append(
            f"row {rows[i]}: unknown shift '{shift[i]}' in '{columns['shift']}'"
            f" (configured shifts: {', '.join(sorted(known))})"
        )

    numbers = {}
    for name, positive in [
        ("ideal_cycle_s", True), ("good_parts", False), ("scrap_parts", False),
    ]:
        numbers[name] = _parse_numbers(
            df[name], rows, columns[name], positive, problems
        )

    if problems:
        raise InputError(what, problems)
    return pd.DataFrame({
        "machine": machine,
        "date": day,
        "shift": shift,
        **numbers,
        "row": rows,
    })


# --- reading -------------------------------------------------------------


def _is_excel(source: Source) -> bool:
    name = str(getattr(source, "name", source))
    return Path(name).suffix.lower() in {".xlsx", ".xlsm", ".xls"}


def _read_csv(source: Source) -> pd.DataFrame:
    # Read as text so that parsing (and its error messages) stays in our hands.
    return pd.read_csv(source, dtype=str, keep_default_na=False, sep=None, engine="python")


def _read_workbook(source: Source) -> dict[str, pd.DataFrame]:
    return pd.read_excel(source, sheet_name=None, dtype=object)


def _pick_sheet(
    sheets: dict[str, pd.DataFrame], name: str | None, index: int, what: str
) -> tuple[pd.DataFrame, str]:
    if name is not None:
        if name not in sheets:
            raise InputError(
                what,
                [f"sheet '{name}' not found; sheets in the workbook: {', '.join(sheets)}"],
            )
        return sheets[name], f"{what} (sheet '{name}')"
    if index >= len(sheets):
        raise InputError(what, [f"workbook has no sheet number {index + 1}"])
    sheet_name = list(sheets)[index]
    return sheets[sheet_name], f"{what} (sheet '{sheet_name}')"


# --- cleaning and validation --------------------------------------------


def _select_columns(
    raw: pd.DataFrame, columns: dict[str, str], fields: tuple[str, ...], what: str
) -> pd.DataFrame:
    raw = raw.rename(columns=lambda c: str(c).strip())
    missing = [columns[f] for f in fields if columns[f] not in raw.columns]
    if missing:
        raise InputError(what, [
            f"missing column(s): {', '.join(missing)}; "
            f"columns in the file: {', '.join(map(str, raw.columns))}. "
            "Check the column names or the mapping file."
        ])
    return pd.DataFrame({f: raw[columns[f]] for f in fields})


def _row_numbers(df: pd.DataFrame) -> pd.Series:
    """Spreadsheet row of each record: header is row 1, data starts at 2."""
    return pd.Series(range(2, len(df) + 2), index=df.index)


def _clean_text(s: pd.Series) -> pd.Series:
    return s.astype(object).where(s.notna(), "").astype(str).str.strip()


def _parse_datetimes(
    s: pd.Series,
    config: Config,
    rows: pd.Series,
    column: str,
    problems: list[str],
    fmt: str | None = None,
) -> pd.Series:
    fmt = fmt if fmt is not None else config.datetime_format
    text = _clean_text(s)
    blank = text == ""
    # Excel cells arrive as datetimes already. Text must be ISO 8601 unless
    # the mapping sets a format or dayfirst: guessing would silently read
    # 01/09/2026 as 9 January.
    values = s.astype(object).where(~blank, None)
    values = values.map(lambda v: v.strip() if isinstance(v, str) else v)
    if fmt:
        hint = f" (expected format {fmt})"
    elif config.dayfirst:
        fmt, hint = "mixed", " (expected day first, e.g. 31/08/2026 06:00)"
    else:
        fmt = "ISO8601"
        hint = (
            " (expected YYYY-MM-DD HH:MM[:SS]; for other formats set "
            "datetime_format in the mapping file)"
        )
    parsed = pd.to_datetime(
        values, format=fmt, dayfirst=config.dayfirst, errors="coerce"
    ).astype("datetime64[ns]")
    for i in parsed.index[parsed.isna() & ~blank]:
        problems.append(
            f"row {rows[i]}: invalid date/time '{text[i]}' in '{column}'{hint}"
        )
    for i in blank.index[blank]:
        problems.append(f"row {rows[i]}: missing date/time in '{column}'")
    return parsed


def _parse_numbers(
    s: pd.Series, rows: pd.Series, column: str, positive: bool, problems: list[str]
) -> pd.Series:
    text = _clean_text(s)
    values = pd.to_numeric(text.str.replace(",", ".", regex=False), errors="coerce")
    bad = values.isna() | (values <= 0 if positive else values < 0)
    rule = "a number > 0" if positive else "a number >= 0"
    for i in bad.index[bad]:
        problems.append(f"row {rows[i]}: '{text[i]}' in '{column}' is not {rule}")
    return values.astype(float)


def _unknown_states_message(labels: pd.Series, rows: pd.Series, column: str) -> str:
    found = []
    for value, idx in labels.groupby(labels).groups.items():
        first = rows[idx].min()
        found.append(f"'{value}' ({len(idx)} rows, first at row {first})")
    return (
        f"unknown value(s) in '{column}': {', '.join(found)}. "
        f"Each state must map to one of {', '.join(STATES)}; "
        "add them under events.state_values in the mapping file."
    )


def _overlap_problems(events: pd.DataFrame) -> list[str]:
    """Events on the same machine must not overlap (input sorted by start)."""
    problems = []
    previous_end = events.groupby("machine")["end"].shift()
    previous_row = events.groupby("machine")["row"].shift()
    overlap = events["start"] < previous_end
    for i in events.index[overlap]:
        problems.append(
            f"row {events.at[i, 'row']}: overlaps row {int(previous_row[i])} "
            f"on machine {events.at[i, 'machine']}"
        )
    return problems
