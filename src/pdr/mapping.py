"""Guided mapping: build a mapping file from a plant's own export.

When a file does not use the expected column names, state labels or date
format, the dashboard asks the user instead of failing. This module holds
the logic behind those questions, so it can be tested without a browser:

- suggest_columns: preselect the column whose name looks like each field
  ("Macchina" -> machine, "Inizio" -> start).
- unknown_state_labels / suggest_state: the state labels that still need a
  meaning, and a likely one ("Alarm" -> STOP, "In marcia" -> RUN).
- guess_date_order / preview_dates: the date order that reads the most
  values, and a few dates as they would be read, so the user sees at once
  whether 01/09 is September.
- diagnose: whether the files need any of this.
- build_mapping / mapping_yaml: the answers as a validated mapping file,
  the same format as config/mapping.example.yaml.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field

import pandas as pd
import yaml

from .config import (
    DATE_ORDERS,
    EVENT_FIELDS,
    PRODUCTION_FIELDS,
    STATES,
    Config,
    ConfigError,
    parse_config,
)
from .io import RawTables, parse_dates

# Column names a plant export may use for each field, already normalised
# (see _normalise). English, Italian, German, Spanish and French.
COLUMN_SYNONYMS: dict[str, list[str]] = {
    "machine": [
        "machine", "machine id", "machine name", "macchina", "macchinario",
        "impianto", "asset", "equipment", "resource", "risorsa", "press",
        "pressa", "work center", "workcenter", "maschine", "anlage", "maquina",
    ],
    "start": [
        "start", "start time", "start date", "started", "begin", "from",
        "inizio", "ora inizio", "data inizio", "dalle", "beginn", "startzeit",
        "inicio", "debut",
    ],
    "end": [
        "end", "end time", "end date", "ended", "finish", "until", "to",
        "fine", "ora fine", "data fine", "alle", "ende", "endzeit", "fin",
    ],
    "state": [
        "state", "status", "machine state", "machine status", "mode", "stato",
        "stato macchina", "condizione", "zustand", "estado", "etat",
    ],
    "reason": [
        "reason", "reason code", "downtime reason", "cause", "stop reason",
        "description", "comment", "note", "notes", "causa", "causale",
        "motivo", "descrizione", "causa fermo", "grund", "ursache", "razon",
    ],
    "date": [
        "date", "day", "production date", "shift date", "data", "giorno",
        "datum", "fecha",
    ],
    "shift": ["shift", "shift name", "crew", "turno", "schicht", "equipo"],
    "ideal_cycle_s": [
        "ideal cycle", "ideal cycle s", "ideal cycle time", "ideal cycle time s",
        "cycle time", "standard cycle", "target cycle", "ciclo ideale",
        "tempo ciclo", "tempo ciclo ideale", "ciclo", "taktzeit", "zykluszeit",
    ],
    "good_parts": [
        "good parts", "good", "good count", "good qty", "ok parts", "pezzi buoni",
        "buoni", "pezzi ok", "gutteile", "piezas buenas",
    ],
    "scrap_parts": [
        "scrap", "scrap parts", "scrap count", "rejects", "rejected", "bad parts",
        "nok", "scarti", "scarto", "pezzi scarto", "ausschuss", "desecho",
    ],
}

# Words that suggest a state, checked in this order: "planned stop" is
# PLANNED, not STOP. Phrases match whole words of the normalised label.
STATE_KEYWORDS: list[tuple[str, list[str]]] = [
    ("SETUP", [
        "setup", "set up", "changeover", "change over", "mold change",
        "mould change", "tool change", "attrezzaggio", "riattrezzaggio",
        "cambio stampo", "cambio formato", "cambio produzione", "rusten",
    ]),
    ("PLANNED", [
        "planned", "scheduled", "break", "lunch", "meeting", "no order",
        "no orders", "preventive", "holiday", "pianificato", "pianificata",
        "programmato", "programmata", "pausa", "pranzo", "riunione",
        "preventiva", "mancanza ordini", "geplant", "pause",
    ]),
    ("STOP", [
        "stop", "stopped", "down", "downtime", "alarm", "fault", "error",
        "failure", "breakdown", "waiting", "idle", "blocked", "starved", "jam",
        "emergency", "fermo", "ferma", "fermata", "allarme", "guasto", "errore",
        "attesa", "blocco", "emergenza", "stillstand", "storung", "parada",
    ]),
    ("RUN", [
        "run", "running", "production", "producing", "working", "auto",
        "automatic", "operating", "in marcia", "marcia", "produzione",
        "in produzione", "lavoro", "automatico", "lauft", "produktion",
    ]),
]

DATE_ORDER_LABELS = {"iso": "Auto (ISO)", "dmy": "DD/MM/YYYY", "mdy": "MM/DD/YYYY"}


def _normalise(name: object) -> str:
    """Lower case, no accents, no units in brackets, words separated by one space."""
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    text = re.sub(r"\(.*?\)|\[.*?\]", " ", text.casefold())
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def _contains_phrase(text: str, phrase: str) -> bool:
    return f" {phrase} " in f" {text} "


def _column_score(column: str, field: str) -> float:
    name = _normalise(column)
    if not name:
        return 0.0
    synonyms = [field.replace("_", " "), *COLUMN_SYNONYMS[field]]
    if name in synonyms:
        return 1.0
    if any(_contains_phrase(name, s) for s in synonyms if len(s) > 2):
        return 0.8
    ratio = max(difflib.SequenceMatcher(None, name, s).ratio() for s in synonyms)
    return 0.7 * ratio if ratio >= 0.85 else 0.0


def suggest_columns(columns: list[str], fields: tuple[str, ...]) -> dict[str, str | None]:
    """The most likely column for each field, or None.

    Best matches are assigned first and every column is used at most once,
    so "Data inizio" goes to start even though it also contains "data".
    """
    scored = sorted(
        ((_column_score(c, f), i, c, f) for f in fields for i, c in enumerate(columns)),
        key=lambda t: (-t[0], t[1]),
    )
    chosen: dict[str, str | None] = dict.fromkeys(fields)
    used: set[str] = set()
    for score, _, column, field_ in scored:
        if score > 0 and chosen[field_] is None and column not in used:
            chosen[field_] = column
            used.add(column)
    return chosen


def suggest_state(label: str) -> str | None:
    """A likely internal state for a source label, or None if unclear."""
    text = _normalise(label)
    if text.upper() in STATES:
        return text.upper()
    for state, phrases in STATE_KEYWORDS:
        if any(_contains_phrase(text, p) for p in phrases):
            return state
    return None


def unknown_state_labels(values: pd.Series, config: Config) -> list[str]:
    """Distinct labels with no state in config, most frequent first (ties: file order)."""
    text = values.astype(object).where(values.notna(), "").astype(str).str.strip()
    text = text[text != ""]
    unknown = text[~text.str.casefold().isin(config.state_values.keys())]
    # Labels differing only in case are one label for the loader.
    first = unknown.groupby(unknown.str.casefold(), sort=False).agg(["first", "size"])
    return list(first.sort_values("size", ascending=False, kind="stable")["first"])


def _date_values(values: pd.Series) -> pd.Series:
    text = values.astype(object).where(values.notna(), "").astype(str).str.strip()
    return values[text != ""]


def guess_date_order(values: pd.Series) -> str:
    """The date order that reads the most values; ISO, then day-first, on a tie."""
    values = _date_values(values)
    counts = {order: int(parse_dates(values, order).notna().sum()) for order in DATE_ORDERS}
    return max(DATE_ORDERS, key=lambda order: counts[order])


def preview_dates(values: pd.Series, order: str, n: int = 3) -> pd.DataFrame:
    """The first n distinct dates as written in the file and as they would be read.

    Dates where day and month could be swapped are shown first, since those
    are the ones that tell DD/MM from MM/DD.
    """
    values = _date_values(values).astype(object).drop_duplicates()
    text = values.astype(str).str.strip()
    numbers = text.str.extract(r"^(\d{1,2})[/.-](\d{1,2})[/.-]")
    ambiguous = numbers.notna().all(axis=1) & (numbers[0] != numbers[1])
    sample = pd.concat([values[ambiguous], values[~ambiguous]]).head(n)
    parsed = parse_dates(sample, order)
    return pd.DataFrame({
        "in the file": sample.astype(str).str.strip().to_list(),
        "read as": [
            "not readable" if pd.isna(p) else f"{p:%a %d %b %Y}" + (
                f" {p:%H:%M}" if (p.hour, p.minute, p.second) != (0, 0, 0) else ""
            )
            for p in parsed
        ],
    })


@dataclass
class MappingNeeds:
    """What the dashboard must ask before the files can be loaded."""

    missing_event_fields: list[str] = field(default_factory=list)
    missing_production_fields: list[str] = field(default_factory=list)
    unknown_states: list[str] = field(default_factory=list)
    # The date order that reads more dates than the configured one, if any.
    date_order: str | None = None

    @property
    def any(self) -> bool:
        return bool(
            self.missing_event_fields or self.missing_production_fields
            or self.unknown_states or self.date_order
        )


def _columns_of(raw: pd.DataFrame) -> list[str]:
    return [str(c).strip() for c in raw.columns]


def diagnose(tables: RawTables, config: Config) -> MappingNeeds:
    """Missing columns, unmapped state labels and a better date order."""
    needs = MappingNeeds()
    events = tables.events.rename(columns=lambda c: str(c).strip())
    needs.missing_event_fields = [
        f for f in EVENT_FIELDS if config.event_columns[f] not in events.columns
    ]
    if tables.production is not None:
        production = tables.production.rename(columns=lambda c: str(c).strip())
        needs.missing_production_fields = [
            f for f in PRODUCTION_FIELDS
            if config.production_columns[f] not in production.columns
        ]
    if "state" not in needs.missing_event_fields:
        needs.unknown_states = unknown_state_labels(
            events[config.event_columns["state"]], config
        )
    if not config.datetime_format and "start" not in needs.missing_event_fields:
        dates = events[config.event_columns["start"]]
        current = int(parse_dates(_date_values(dates), config.date_order).notna().sum())
        best = guess_date_order(dates)
        if int(parse_dates(_date_values(dates), best).notna().sum()) > current:
            needs.date_order = best
    return needs


def build_mapping(
    event_columns: dict[str, str | None],
    production_columns: dict[str, str | None] | None,
    state_values: dict[str, str | None],
    date_order: str,
    base: dict | None = None,
) -> dict:
    """The user's answers as a mapping dict, validated by parse_config.

    base is an uploaded mapping file (or {}); its other settings (sheets,
    shifts, micro-stop options...) are kept. Raises ConfigError with every
    unanswered question.
    """
    raw = {k: (dict(v) if isinstance(v, dict) else v) for k, v in (base or {}).items()}
    problems = _unanswered("events", event_columns)
    if production_columns is not None:
        problems += _unanswered("production", production_columns)
    unmapped = [label for label, state in state_values.items() if state not in STATES]
    if unmapped:
        problems.append(f"choose a state for: {', '.join(unmapped)}")
    if date_order not in DATE_ORDERS:
        problems.append(f"date format must be one of: {', '.join(DATE_ORDERS)}")
    if problems:
        raise ConfigError("; ".join(problems))

    events = raw.setdefault("events", {}) or {}
    raw["events"] = events
    events["columns"] = dict(event_columns)
    if state_values:
        merged: dict[str, list[str]] = {}
        for state, labels in (events.get("state_values") or {}).items():
            merged[state] = list(labels) if isinstance(labels, list) else [labels]
        for label, state in state_values.items():
            merged.setdefault(state, []).append(label)
        events["state_values"] = {s: merged[s] for s in STATES if s in merged}
    if production_columns is not None:
        production = raw.get("production") or {}
        production["columns"] = dict(production_columns)
        raw["production"] = production
    raw["date_order"] = date_order
    parse_config(raw)  # raises ConfigError, e.g. one label mapped to two states
    return raw


def _unanswered(table: str, columns: dict[str, str | None]) -> list[str]:
    problems = []
    missing = [f for f, c in columns.items() if not c]
    if missing:
        problems.append(f"choose the {table} column for: {', '.join(missing)}")
    chosen = [c for c in columns.values() if c]
    repeated = sorted({c for c in chosen if chosen.count(c) > 1})
    if repeated:
        problems.append(
            f"each {table} column can be used once; used more than once: {', '.join(repeated)}"
        )
    return problems


def mapping_yaml(raw: dict) -> str:
    """The mapping as a YAML file to keep and reuse."""
    header = (
        "# Mapping created in the Production Downtime dashboard.\n"
        "# Upload it next time with your files, or use it with the command line:\n"
        "#   python -m pdr report --input <events file> --mapping mapping.yaml\n"
    )
    return header + yaml.safe_dump(raw, sort_keys=False, allow_unicode=True)
