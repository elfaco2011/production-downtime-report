"""Input mapping and analysis settings, loaded from a YAML file.

A mapping file tells the loader how a plant's own export is laid out:
column names, the labels it uses for machine states, sheet names in a
workbook, and date format. It also defines the shift calendar and the
micro-stop threshold. Every key is optional; see
config/mapping.example.yaml for a complete example.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from pathlib import Path

import yaml

STATES = ("RUN", "STOP", "SETUP", "PLANNED")
EVENT_FIELDS = ("machine", "start", "end", "state", "reason")
PRODUCTION_FIELDS = (
    "machine", "date", "shift", "ideal_cycle_s", "good_parts", "scrap_parts",
)


class ConfigError(ValueError):
    """The mapping file is malformed."""


@dataclass(frozen=True)
class Shift:
    """A daily shift. If end <= start the shift ends on the next day."""

    name: str
    start: time
    end: time

    @property
    def start_min(self) -> int:
        return self.start.hour * 60 + self.start.minute

    @property
    def duration_min(self) -> int:
        end_min = self.end.hour * 60 + self.end.minute
        return (end_min - self.start_min) % (24 * 60) or 24 * 60


DEFAULT_SHIFTS = (
    Shift("A", time(6, 0), time(14, 0)),
    Shift("B", time(14, 0), time(22, 0)),
)


@dataclass(frozen=True)
class Config:
    # Source column name for each internal field.
    event_columns: dict[str, str] = field(
        default_factory=lambda: {f: f for f in EVENT_FIELDS}
    )
    production_columns: dict[str, str] = field(
        default_factory=lambda: {f: f for f in PRODUCTION_FIELDS}
    )
    # Source state label (case-insensitive) -> internal state.
    state_values: dict[str, str] = field(
        default_factory=lambda: {s.casefold(): s for s in STATES}
    )
    events_sheet: str | None = None
    production_sheet: str | None = None
    datetime_format: str | None = None
    date_format: str | None = None
    dayfirst: bool = False
    shifts: tuple[Shift, ...] = DEFAULT_SHIFTS
    micro_stop_min: float = 2.0

    def shift(self, name: str) -> Shift:
        return next(s for s in self.shifts if s.name == name)


def load_config(path: str | Path | None) -> Config:
    """Load a mapping YAML file; None returns the default configuration."""
    if path is None:
        return Config()
    try:
        raw = yaml.safe_load(Path(path).read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read mapping file {path}: {exc}") from exc
    return parse_config(raw)


def parse_config(raw: dict) -> Config:
    if not isinstance(raw, dict):
        raise ConfigError("mapping file must be a YAML mapping")
    _check_keys(raw, {"events", "production", "shifts", "micro_stop_minutes"}, "top level")
    events = raw.get("events") or {}
    production = raw.get("production") or {}
    _check_keys(
        events,
        {"sheet", "columns", "state_values", "datetime_format", "dayfirst"},
        "events",
    )
    _check_keys(production, {"sheet", "columns", "date_format"}, "production")

    kwargs = {
        "event_columns": _columns(events.get("columns"), EVENT_FIELDS, "events"),
        "production_columns": _columns(
            production.get("columns"), PRODUCTION_FIELDS, "production"
        ),
        "state_values": _state_values(events.get("state_values")),
        "events_sheet": events.get("sheet"),
        "production_sheet": production.get("sheet"),
        "datetime_format": events.get("datetime_format"),
        "date_format": production.get("date_format"),
        "dayfirst": bool(events.get("dayfirst", False)),
    }
    if "shifts" in raw:
        kwargs["shifts"] = _shifts(raw["shifts"])
    if "micro_stop_minutes" in raw:
        value = raw["micro_stop_minutes"]
        if not isinstance(value, (int, float)) or value < 0:
            raise ConfigError("micro_stop_minutes must be a number >= 0")
        kwargs["micro_stop_min"] = float(value)
    return Config(**kwargs)


def _check_keys(section: dict, allowed: set[str], where: str) -> None:
    if not isinstance(section, dict):
        raise ConfigError(f"{where}: expected a mapping")
    unknown = set(section) - allowed
    if unknown:
        raise ConfigError(
            f"{where}: unknown keys {sorted(unknown)}; allowed: {sorted(allowed)}"
        )


def _columns(raw: dict | None, fields: tuple[str, ...], where: str) -> dict[str, str]:
    columns = {f: f for f in fields}
    if raw is None:
        return columns
    _check_keys(raw, set(fields), f"{where}.columns")
    columns.update({k: str(v) for k, v in raw.items()})
    return columns


def _state_values(raw: dict | None) -> dict[str, str]:
    """Build the label -> state lookup. Internal names always map to themselves."""
    lookup: dict[str, str] = {}
    if raw is not None:
        _check_keys(raw, set(STATES), "events.state_values")
        for state, labels in raw.items():
            for label in labels if isinstance(labels, list) else [labels]:
                key = str(label).strip().casefold()
                if key in lookup and lookup[key] != state:
                    raise ConfigError(
                        f"events.state_values: '{label}' is mapped to both "
                        f"{lookup[key]} and {state}"
                    )
                lookup[key] = state
    for state in STATES:
        lookup.setdefault(state.casefold(), state)
    return lookup


def _shifts(raw: dict) -> tuple[Shift, ...]:
    if not isinstance(raw, dict) or not raw:
        raise ConfigError("shifts: expected a mapping like A: {start: '06:00', end: '14:00'}")
    shifts = []
    for name, spec in raw.items():
        if not isinstance(spec, dict) or set(spec) != {"start", "end"}:
            raise ConfigError(f"shifts.{name}: expected keys 'start' and 'end'")
        shifts.append(Shift(str(name), _time(spec["start"], name), _time(spec["end"], name)))
    # Overlapping shifts would count the same minutes twice.
    day = 24 * 60
    taken: dict[int, str] = {}
    for s in shifts:
        for m in range(s.start_min, s.start_min + s.duration_min):
            if m % day in taken:
                raise ConfigError(f"shifts {taken[m % day]} and {s.name} overlap")
            taken[m % day] = s.name
    return tuple(shifts)


def _time(value, shift_name) -> time:
    # PyYAML reads unquoted 06:00 as the sexagesimal integer 360 (minutes).
    if isinstance(value, int):
        return time(value // 60 % 24, value % 60)
    try:
        hours, minutes = str(value).split(":")
        return time(int(hours), int(minutes))
    except ValueError as exc:
        raise ConfigError(f"shifts.{shift_name}: invalid time '{value}', use HH:MM") from exc
