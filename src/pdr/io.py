"""Load event and production logs from CSV or XLSX into canonical tables.

All problems found in the files are collected and raised together as an
InputError, each with the row number the user sees when the file is
opened in Excel (header = row 1, blank rows counted), so it can be fixed
in one pass.

Problems tied to one row (unreadable date, end before start, overlap,
bad number...) can instead be skipped with skip_invalid_rows=True: those
rows are left out and listed in Dataset.dropped_rows. Problems about the
file as a whole (missing columns, unmapped state labels) are never
skipped, since skipping them would silently drop data.

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

import numpy as np
import pandas as pd

from .config import EVENT_FIELDS, PRODUCTION_FIELDS, STATES, Config

UNSPECIFIED = "Unspecified"
MAX_LISTED_PROBLEMS = 20
ROW_PREFIX = "Excel row "
DROPPED_COLUMNS = ["what", "row", "reason"]

Source = str | Path | BinaryIO


@dataclass(frozen=True)
class Problem:
    """One reason why a file cannot be analysed as it is."""

    message: str
    row: int | None = None  # Excel row; None for problems about the whole file
    column: str | None = None  # column name as in the file
    what: str = ""  # which table, e.g. "events (sheet 'Eventi')"

    def __str__(self) -> str:
        return f"{ROW_PREFIX}{self.row}: {self.message}" if self.row else self.message


class InputError(ValueError):
    """The input cannot be analysed; `issues` lists every reason.

    `problems` holds the same reasons as text, sorted by table and row.
    """

    def __init__(self, what: str, problems: list[Problem | str]):
        issues = [p if isinstance(p, Problem) else Problem(p) for p in problems]
        issues = [p if p.what else Problem(p.message, p.row, p.column, what) for p in issues]
        order = list(dict.fromkeys(p.what for p in issues)) or [what]
        self.issues = sorted(
            issues,
            key=lambda p: (order.index(p.what), p.row if p.row else float("inf")),
        )
        self.what = " and ".join(order)
        self.problems = [str(p) for p in self.issues]
        lines: list[str] = []
        for table in order:
            listed = [str(p) for p in self.issues if p.what == table]
            lines.append(f"{table}: {len(listed)} problem(s) found")
            lines += [f"  - {p}" for p in listed[:MAX_LISTED_PROBLEMS]]
            if len(listed) > MAX_LISTED_PROBLEMS:
                lines.append(f"  ... and {len(listed) - MAX_LISTED_PROBLEMS} more")
        super().__init__("\n".join(lines))

    @property
    def rows_only(self) -> bool:
        """True if every problem concerns a single row, so rows can be skipped."""
        return all(p.row is not None for p in self.issues)

    @property
    def bad_rows(self) -> int:
        return len({(p.what, p.row) for p in self.issues if p.row is not None})

    def table(self) -> pd.DataFrame:
        """One line per problem: file, Excel row, column, problem."""
        return pd.DataFrame(
            [(p.what, p.row, p.column or "", p.message) for p in self.issues],
            columns=["file", "Excel row", "column", "problem"],
        ).astype({"Excel row": "Int64"})


@dataclass
class Dataset:
    events: pd.DataFrame
    production: pd.DataFrame | None
    config: Config
    notes: list[str] = field(default_factory=list)
    # Rows left out by the loader: columns what, row, reason.
    dropped_rows: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=DROPPED_COLUMNS)
    )


@dataclass
class RawTables:
    """The input tables as read from the files, before any validation."""

    events: pd.DataFrame
    events_what: str
    production: pd.DataFrame | None = None
    production_what: str | None = None


def read_tables(
    events_source: Source,
    production_source: Source | None = None,
    config: Config | None = None,
) -> RawTables:
    """Read the raw tables. An XLSX events source may hold both tables in
    separate sheets; then production is read from the same workbook unless
    given explicitly."""
    config = config or Config()
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
    return RawTables(events_raw, events_what, production_raw, production_what)


def load_dataset(
    events_source: Source,
    production_source: Source | None = None,
    config: Config | None = None,
    skip_invalid_rows: bool = False,
) -> Dataset:
    """Load events and, optionally, production data.

    Problems in both tables are collected before raising. With
    skip_invalid_rows, rows with a row-level problem are left out and
    listed in dropped_rows instead.
    """
    config = config or Config()
    raw = read_tables(events_source, production_source, config)
    return parse_tables(raw, config, skip_invalid_rows)


def parse_tables(raw: RawTables, config: Config, skip_invalid_rows: bool = False) -> Dataset:
    notes: list[str] = []
    dropped: list[pd.DataFrame] = []
    errors: list[InputError] = []

    events = production = None
    try:
        events = parse_events(raw.events, config, raw.events_what, dropped, skip_invalid_rows)
    except InputError as exc:
        errors.append(exc)
    if raw.production is not None:
        try:
            production = parse_production(
                raw.production, config, raw.production_what, dropped, skip_invalid_rows
            )
        except InputError as exc:
            errors.append(exc)
    else:
        notes.append(
            "No production data: Performance, Quality and OEE are not available."
        )
    if errors:
        if len(errors) == 1:
            raise errors[0]
        raise InputError(errors[0].what, [p for e in errors for p in e.issues])

    dropped_rows = (
        pd.concat(dropped, ignore_index=True)
        if dropped
        else pd.DataFrame(columns=DROPPED_COLUMNS)
    )
    return Dataset(events, production, config, notes, dropped_rows)


def parse_events(
    raw: pd.DataFrame,
    config: Config,
    what: str = "events",
    dropped: list[pd.DataFrame] | None = None,
    skip_invalid_rows: bool = False,
) -> pd.DataFrame:
    df = _select_columns(raw, config.event_columns, EVENT_FIELDS, what)
    rows = _row_numbers(df)
    problems: list[Problem] = []
    columns = config.event_columns

    machine = _clean_text(df["machine"])
    for i in machine.index[machine == ""]:
        problems.append(Problem(f"missing machine in '{columns['machine']}'", rows[i], columns["machine"]))

    start = _parse_datetimes(df["start"], config, rows, columns["start"], problems)
    end = _parse_datetimes(df["end"], config, rows, columns["end"], problems)
    reversed_ = start.notna() & end.notna() & (end < start)
    for i in reversed_.index[reversed_]:
        problems.append(Problem(
            f"end ({end[i]}) is before start ({start[i]})", rows[i], columns["end"]
        ))

    label = _clean_text(df["state"])
    state = label.str.casefold().map(config.state_values)
    unknown = state.isna()
    if unknown.any():
        problems.append(_unknown_states_problem(label[unknown], rows, columns["state"]))

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

    # Overlaps are checked among the rows that are otherwise valid, so the
    # user sees every problem at once.
    bad = {p.row for p in problems}
    zero = events["end"] == events["start"]
    candidates = events[~events["row"].isin(bad) & ~zero]
    candidates = candidates.sort_values(["machine", "start", "row"])
    problems += _overlap_problems(candidates, columns["start"])

    _raise_or_skip(problems, what, skip_invalid_rows, dropped)
    if zero.any() and dropped is not None:
        dropped.append(pd.DataFrame({
            "what": what, "row": events.loc[zero, "row"], "reason": "zero-length event",
        }))
    bad = {p.row for p in problems}
    events = events[~events["row"].isin(bad) & ~zero]
    return events.sort_values(["machine", "start"]).reset_index(drop=True)


def parse_production(
    raw: pd.DataFrame,
    config: Config,
    what: str = "production",
    dropped: list[pd.DataFrame] | None = None,
    skip_invalid_rows: bool = False,
) -> pd.DataFrame:
    df = _select_columns(raw, config.production_columns, PRODUCTION_FIELDS, what)
    rows = _row_numbers(df)
    problems: list[Problem] = []
    columns = config.production_columns

    machine = _clean_text(df["machine"])
    for i in machine.index[machine == ""]:
        problems.append(Problem(f"missing machine in '{columns['machine']}'", rows[i], columns["machine"]))

    day = _parse_datetimes(
        df["date"], config, rows, columns["date"], problems, fmt=config.date_format
    ).dt.normalize()

    shift = _clean_text(df["shift"])
    known = {s.name for s in config.shifts}
    for i in shift.index[~shift.isin(known)]:
        problems.append(Problem(
            f"unknown shift '{shift[i]}' in '{columns['shift']}'"
            f" (configured shifts: {', '.join(sorted(known))})",
            rows[i], columns["shift"],
        ))

    numbers = {}
    for name, positive in [
        ("ideal_cycle_s", True), ("good_parts", False), ("scrap_parts", False),
    ]:
        numbers[name] = _parse_numbers(
            df[name], rows, columns[name], positive, problems
        )

    _raise_or_skip(problems, what, skip_invalid_rows, dropped)
    production = pd.DataFrame({
        "machine": machine,
        "date": day,
        "shift": shift,
        **numbers,
        "row": rows,
    })
    return production[~production["row"].isin({p.row for p in problems})].reset_index(
        drop=True
    )


def _raise_or_skip(
    problems: list[Problem],
    what: str,
    skip_invalid_rows: bool,
    dropped: list[pd.DataFrame] | None,
) -> None:
    """Raise unless every problem is about a row and rows may be skipped;
    then record the skipped rows, one line per row."""
    if not problems:
        return
    if not skip_invalid_rows or any(p.row is None for p in problems):
        raise InputError(what, problems)
    if dropped is not None:
        by_row: dict[int, list[str]] = {}
        for p in problems:
            by_row.setdefault(p.row, []).append(p.message)
        dropped.append(pd.DataFrame({
            "what": what,
            "row": sorted(by_row),
            "reason": ["; ".join(by_row[r]) for r in sorted(by_row)],
        }))


# --- reading -------------------------------------------------------------


def _is_excel(source: Source) -> bool:
    name = str(getattr(source, "name", source))
    return Path(name).suffix.lower() in {".xlsx", ".xlsm", ".xls"}


def _read_csv(source: Source) -> pd.DataFrame:
    # Read as text so that parsing (and its error messages) stays in our hands.
    # Blank lines are kept so that row numbers match Excel; _select_columns
    # drops them afterwards.
    return pd.read_csv(
        source, dtype=str, keep_default_na=False, sep=None, engine="python",
        skip_blank_lines=False,
    )


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
    df = pd.DataFrame({f: raw[columns[f]] for f in fields}).reset_index(drop=True)
    # Blank rows (e.g. a gap left in Excel) carry no data; the index keeps
    # the original position, so row numbers of the other rows stay right.
    blank = df.apply(lambda s: _clean_text(s) == "").all(axis=1)
    return df[~blank]


def _row_numbers(df: pd.DataFrame) -> pd.Series:
    """Excel row of each record: header is row 1, data starts at row 2."""
    return pd.Series(df.index + 2, index=df.index)


def _clean_text(s: pd.Series) -> pd.Series:
    return s.astype(object).where(s.notna(), "").astype(str).str.strip()


DATE_HINTS = {
    "iso": " (expected YYYY-MM-DD HH:MM[:SS]; for other formats choose the "
           "date format or set date_order in the mapping file)",
    "dmy": " (expected DD/MM/YYYY HH:MM[:SS] or YYYY-MM-DD HH:MM[:SS])",
    "mdy": " (expected MM/DD/YYYY HH:MM[:SS] or YYYY-MM-DD HH:MM[:SS])",
}


def _order_formats(order: str) -> list[str]:
    """Explicit formats tried after ISO for a day- or month-first order."""
    day_month = "%d{0}%m{0}%Y" if order == "dmy" else "%m{0}%d{0}%Y"
    return [
        day_month.format(sep) + time
        for sep in "/-."
        for time in ("", " %H:%M", " %H:%M:%S")
    ]


def parse_dates(values: pd.Series, order: str = "iso", fmt: str | None = None) -> pd.Series:
    """Parse dates strictly: unreadable values become NaT, nothing is guessed.

    Values may be text or datetimes (Excel cells). ISO 8601 is always
    accepted; "dmy" / "mdy" also accept 31/08/2026 / 08/31/2026 with an
    optional HH:MM[:SS] time and / - . as separators. An explicit strftime
    format replaces all of this.
    """
    values = values.astype(object).map(lambda v: v.strip() if isinstance(v, str) else v)
    if fmt:
        return pd.to_datetime(values, format=fmt, errors="coerce").astype("datetime64[ns]")
    parsed = pd.to_datetime(values, format="ISO8601", errors="coerce").astype("datetime64[ns]")
    if order != "iso":
        for f in _order_formats(order):
            if not parsed.isna().any():
                break
            parsed = parsed.fillna(
                pd.to_datetime(values, format=f, errors="coerce").astype("datetime64[ns]")
            )
    return parsed


def _parse_datetimes(
    s: pd.Series,
    config: Config,
    rows: pd.Series,
    column: str,
    problems: list[Problem],
    fmt: str | None = None,
) -> pd.Series:
    fmt = fmt if fmt is not None else config.datetime_format
    text = _clean_text(s)
    blank = text == ""
    # Excel cells arrive as datetimes already. Text must be ISO 8601 unless
    # the mapping sets a format or a date order: guessing would silently
    # read 01/09/2026 as 9 January.
    values = s.astype(object).where(~blank, None)
    hint = f" (expected format {fmt})" if fmt else DATE_HINTS[config.date_order]
    parsed = parse_dates(values, config.date_order, fmt)
    for i in parsed.index[parsed.isna() & ~blank]:
        problems.append(
            Problem(f"invalid date/time '{text[i]}' in '{column}'{hint}", rows[i], column)
        )
    for i in blank.index[blank]:
        problems.append(Problem(f"missing date/time in '{column}'", rows[i], column))
    return parsed


def _parse_numbers(
    s: pd.Series, rows: pd.Series, column: str, positive: bool, problems: list[Problem]
) -> pd.Series:
    text = _clean_text(s)
    values = pd.to_numeric(text.str.replace(",", ".", regex=False), errors="coerce")
    bad = values.isna() | (values <= 0 if positive else values < 0)
    rule = "a number > 0" if positive else "a number >= 0"
    for i in bad.index[bad]:
        problems.append(Problem(f"'{text[i]}' in '{column}' is not {rule}", rows[i], column))
    return values.astype(float)


def _unknown_states_problem(labels: pd.Series, rows: pd.Series, column: str) -> Problem:
    found = []
    for value, idx in labels.groupby(labels).groups.items():
        first = rows[idx].min()
        rows_word = "row" if len(idx) == 1 else "rows"
        found.append(f"'{value}' ({len(idx)} {rows_word}, first at Excel row {first})")
    return Problem(
        f"unknown value(s) in '{column}': {', '.join(found)}. "
        f"Each state must map to one of {', '.join(STATES)}; "
        "map them in the dashboard or under events.state_values in the mapping file.",
        column=column,
    )


def _overlap_problems(events: pd.DataFrame, column: str) -> list[Problem]:
    """Events on the same machine must not overlap (input sorted by start).

    An event is kept if it starts after every kept event before it has
    ended; otherwise it is reported. Leaving out the reported rows
    therefore leaves no overlap.
    """
    problems = []
    for machine, part in events.groupby("machine", sort=False):
        starts, ends, rows = (part[c].to_numpy() for c in ("start", "end", "row"))
        latest_end, latest_row = np.datetime64("NaT"), None
        for start, end, row in zip(starts, ends, rows):
            if latest_row is not None and start < latest_end:
                problems.append(Problem(
                    f"overlaps Excel row {latest_row} on machine {machine}", int(row), column
                ))
                continue
            if latest_row is None or end > latest_end:
                latest_end, latest_row = end, int(row)
    return problems
