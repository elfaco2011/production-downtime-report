import io
from pathlib import Path

import pandas as pd
import pytest
import yaml

from pdr.config import EVENT_FIELDS, PRODUCTION_FIELDS, Config, ConfigError, parse_config
from pdr.io import load_dataset, read_tables
from pdr.mapping import (
    build_mapping,
    diagnose,
    guess_date_order,
    mapping_yaml,
    preview_dates,
    suggest_columns,
    suggest_state,
    unknown_state_labels,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample"

ITALIAN_EVENTS = """\
Macchina;Inizio;Fine;Stato;Causa
P-01;01/09/2026 06:00;01/09/2026 07:30;In marcia;
P-01;01/09/2026 07:30;01/09/2026 07:45;Fermo;stampo bloccato
P-01;01/09/2026 07:45;01/09/2026 13:00;In marcia;
P-01;01/09/2026 13:00;01/09/2026 14:00;Allarme;temperatura
P-01;13/09/2026 06:00;13/09/2026 14:00;In marcia;
"""


def named(text: str, name: str) -> io.BytesIO:
    buf = io.BytesIO(text.encode())
    buf.name = name
    return buf


def test_suggest_columns_italian_and_english_names():
    assert suggest_columns(["Macchina", "Inizio", "Fine", "Stato", "Causa"], EVENT_FIELDS) == {
        "machine": "Macchina", "start": "Inizio", "end": "Fine",
        "state": "Stato", "reason": "Causa",
    }
    assert suggest_columns(
        ["Machine ID", "Start Time", "End Time", "Status", "Downtime reason", "Operator"],
        EVENT_FIELDS,
    ) == {
        "machine": "Machine ID", "start": "Start Time", "end": "End Time",
        "state": "Status", "reason": "Downtime reason",
    }


def test_suggest_columns_production_with_units_and_accents():
    columns = ["Macchina", "Data", "Turno", "Ciclo ideale (s)", "Pezzi buoni", "Scarti"]
    assert list(suggest_columns(columns, PRODUCTION_FIELDS).values()) == columns


def test_suggest_columns_uses_each_column_once_and_leaves_unknown_empty():
    found = suggest_columns(["Data inizio", "Data fine", "Codice"], EVENT_FIELDS)
    assert found == {
        "machine": None, "start": "Data inizio", "end": "Data fine",
        "state": None, "reason": None,
    }


@pytest.mark.parametrize("label, state", [
    ("Alarm", "STOP"), ("Waiting", "STOP"), ("Fermo", "STOP"), ("Guasto motore", "STOP"),
    ("In marcia", "RUN"), ("Running", "RUN"), ("Automatico", "RUN"),
    ("Attrezzaggio", "SETUP"), ("Mold change", "SETUP"),
    ("Pianificato", "PLANNED"), ("Planned stop", "PLANNED"), ("Pausa pranzo", "PLANNED"),
    ("run", "RUN"), ("Xyz", None),
])
def test_suggest_state(label, state):
    assert suggest_state(label) == state


def test_unknown_state_labels_most_frequent_first_case_insensitive():
    values = pd.Series(["RUN", "Waiting", "Alarm", "alarm", "ALARM", " ", None, "stop"])
    assert unknown_state_labels(values, Config()) == ["Alarm", "Waiting"]
    config = parse_config({"events": {"state_values": {"STOP": "Alarm"}}})
    assert unknown_state_labels(values, config) == ["Waiting"]


def test_guess_date_order():
    assert guess_date_order(pd.Series(["2026-09-01 06:00", "2026-09-13 07:00"])) == "iso"
    assert guess_date_order(pd.Series(["01/09/2026 06:00", "13/09/2026 07:00"])) == "dmy"
    assert guess_date_order(pd.Series(["09/01/2026 06:00", "09/13/2026 07:00"])) == "mdy"
    # Cannot tell: day-first is offered, and the preview shows how it reads.
    assert guess_date_order(pd.Series(["01/09/2026 06:00"])) == "dmy"


def test_preview_dates_shows_how_ambiguous_dates_are_read():
    values = pd.Series(["2026-09-01 06:00", "01/09/2026 07:30", "01/09/2026 07:30", "x", ""])
    dmy = preview_dates(values, "dmy")
    assert dmy.to_dict("records") == [
        {"in the file": "01/09/2026 07:30", "read as": "Tue 01 Sep 2026 07:30"},
        {"in the file": "2026-09-01 06:00", "read as": "Tue 01 Sep 2026 06:00"},
        {"in the file": "x", "read as": "not readable"},
    ]
    mdy = preview_dates(values, "mdy")
    assert mdy.at[0, "read as"] == "Fri 09 Jan 2026 07:30"


def test_diagnose_italian_file():
    tables = read_tables(named(ITALIAN_EVENTS, "eventi.csv"))
    needs = diagnose(tables, Config())
    assert needs.any
    assert needs.missing_event_fields == list(EVENT_FIELDS)
    assert needs.unknown_states == []  # cannot be listed before the state column is known
    assert needs.date_order is None


def test_diagnose_sample_needs_nothing():
    tables = read_tables(SAMPLE / "events.csv", SAMPLE / "production.csv")
    assert not diagnose(tables, Config()).any


def test_diagnose_dates_and_states_once_columns_are_known():
    text = ITALIAN_EVENTS.replace("Macchina;Inizio;Fine;Stato;Causa",
                                  "machine;start;end;state;reason")
    needs = diagnose(read_tables(named(text, "events.csv")), Config())
    assert needs.missing_event_fields == []
    assert needs.unknown_states == ["In marcia", "Fermo", "Allarme"]
    assert needs.date_order == "dmy"


def italian_mapping(base=None) -> dict:
    tables = read_tables(named(ITALIAN_EVENTS, "eventi.csv"))
    columns = suggest_columns(list(tables.events.columns), EVENT_FIELDS)
    labels = unknown_state_labels(tables.events[columns["state"]], Config())
    return build_mapping(
        columns, None, {label: suggest_state(label) for label in labels},
        guess_date_order(tables.events[columns["start"]]), base,
    )


def test_italian_file_loads_with_the_suggested_answers_only():
    raw = italian_mapping()
    assert raw["events"]["state_values"] == {"RUN": ["In marcia"], "STOP": ["Fermo", "Allarme"]}
    assert raw["date_order"] == "dmy"
    events = load_dataset(named(ITALIAN_EVENTS, "eventi.csv"), config=parse_config(raw)).events
    assert list(events["state"]) == ["RUN", "STOP", "RUN", "STOP", "RUN"]
    assert events.at[4, "start"] == pd.Timestamp("2026-09-13 06:00")


def test_mapping_yaml_round_trip_keeps_base_settings():
    base = {"shifts": {"A": {"start": "06:00", "end": "14:00"}},
            "events": {"state_values": {"PLANNED": "Pausa"}}}
    raw = italian_mapping(base)
    text = mapping_yaml(raw)
    assert text.startswith("# Mapping created in the Production Downtime dashboard.")
    again = yaml.safe_load(text)
    assert again == raw
    config = parse_config(again)
    assert [s.name for s in config.shifts] == ["A"]
    assert config.state_values["pausa"] == "PLANNED"
    assert config.state_values["allarme"] == "STOP"


def test_build_mapping_lists_every_unanswered_question():
    with pytest.raises(ConfigError) as exc:
        build_mapping(
            {"machine": "A", "start": "B", "end": "B", "state": None, "reason": "C"},
            None, {"Alarm": None}, "dmy",
        )
    message = str(exc.value)
    assert "choose the events column for: state" in message
    assert "used more than once: B" in message
    assert "choose a state for: Alarm" in message


def test_build_mapping_rejects_conflicting_labels():
    base = {"events": {"state_values": {"RUN": "Fermo"}}}
    with pytest.raises(ConfigError, match="mapped to both"):
        build_mapping(
            dict(zip(EVENT_FIELDS, ["Macchina", "Inizio", "Fine", "Stato", "Causa"])),
            None, {"Fermo": "STOP"}, "dmy", base,
        )
