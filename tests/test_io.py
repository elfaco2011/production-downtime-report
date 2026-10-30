import io
from pathlib import Path

import pandas as pd
import pytest

from pdr.config import Config, ConfigError, load_config, parse_config
from pdr.io import UNSPECIFIED, InputError, load_dataset

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample"
MAPPING = ROOT / "config" / "mapping.example.yaml"

HEADER = "machine,start,end,state,reason\n"
ISO_HINT = (
    " (expected YYYY-MM-DD HH:MM[:SS]; for other formats set "
    "datetime_format in the mapping file)"
)


def csv(text: str) -> io.BytesIO:
    buf = io.BytesIO(text.encode())
    buf.name = "events.csv"
    return buf


def load_events(text: str, config=None) -> pd.DataFrame:
    return load_dataset(csv(text), config=config).events


def problems(text: str, config=None) -> list[str]:
    with pytest.raises(InputError) as exc:
        load_events(text, config)
    return exc.value.problems


def test_csv_and_mapped_xlsx_are_identical():
    a = load_dataset(SAMPLE / "events.csv", SAMPLE / "production.csv")
    b = load_dataset(SAMPLE / "plant_export.xlsx", config=load_config(MAPPING))
    pd.testing.assert_frame_equal(a.events, b.events)
    pd.testing.assert_frame_equal(a.production, b.production)


def test_empty_reason_becomes_unspecified():
    events = load_events(
        HEADER
        + "M1,2026-09-01 06:00,2026-09-01 06:10,RUN,\n"
        + "M1,2026-09-01 06:10,2026-09-01 06:20,STOP,\n"
        + "M1,2026-09-01 06:20,2026-09-01 06:30,STOP,robot fault\n"
    )
    assert list(events["reason"]) == ["", UNSPECIFIED, "robot fault"]
    assert list(events["reason_missing"]) == [False, True, False]


def test_state_labels_are_case_insensitive():
    events = load_events(HEADER + "M1,2026-09-01 06:00,2026-09-01 06:10, run ,\n")
    assert events.at[0, "state"] == "RUN"


def test_missing_columns_are_listed():
    with pytest.raises(InputError, match="missing column.*end.*columns in the file"):
        load_events("machine,start,state,reason\nM1,2026-09-01 06:00,RUN,\n")


def test_invalid_and_missing_dates_report_rows():
    found = problems(
        HEADER
        + "M1,2026-09-01 06:00,2026-09-01 06:10,RUN,\n"
        + "M1,not a date,2026-09-01 06:20,RUN,\n"
        + "M1,2026-09-01 06:20,,RUN,\n"
    )
    assert found == [
        "row 3: invalid date/time 'not a date' in 'start'" + ISO_HINT,
        "row 4: missing date/time in 'end'",
    ]


def test_non_iso_dates_are_rejected_not_guessed():
    # Without a format, 01/09/2026 could be 1 September or 9 January.
    found = problems(
        HEADER
        + "M1,2026-09-01 07:00,2026-09-01 06:50,STOP,x\n"
        + "M1,01/09/2026 07:00,2026-09-01 08:00,RUN,\n"
        + "M1,2026-09-01 08:00,2026-09-01 08:20,Alarm,\n"
    )
    assert found == [
        "row 2: end (2026-09-01 06:50:00) is before start (2026-09-01 07:00:00)",
        "row 3: invalid date/time '01/09/2026 07:00' in 'start'" + ISO_HINT,
        found[2],
    ]
    assert found[2].startswith("unknown value(s) in 'state'")


def test_dayfirst_option():
    config = parse_config({"events": {"dayfirst": True}})
    events = load_events(HEADER + "M1,01/09/2026 07:00,01/09/2026 08:00,RUN,\n", config)
    assert events.at[0, "start"] == pd.Timestamp("2026-09-01 07:00")


def test_end_before_start_reports_row():
    found = problems(HEADER + "M1,2026-09-01 07:00,2026-09-01 06:00,STOP,x\n")
    assert found == [
        "row 2: end (2026-09-01 06:00:00) is before start (2026-09-01 07:00:00)"
    ]


def test_overlapping_events_report_rows():
    found = problems(
        HEADER
        + "M1,2026-09-01 06:00,2026-09-01 06:30,RUN,\n"
        + "M1,2026-09-01 06:20,2026-09-01 06:40,STOP,x\n"
        + "M2,2026-09-01 06:20,2026-09-01 06:40,STOP,x\n"
    )
    assert found == ["row 3: overlaps row 2 on machine M1"]


def test_unmapped_state_values_are_listed():
    found = problems(
        HEADER
        + "M1,2026-09-01 06:00,2026-09-01 06:10,Marcia,\n"
        + "M1,2026-09-01 06:10,2026-09-01 06:20,Allarme,x\n"
        + "M1,2026-09-01 06:20,2026-09-01 06:30,Allarme,x\n"
    )
    assert len(found) == 1
    assert "'Allarme' (2 rows, first at row 3)" in found[0]
    assert "'Marcia' (1 row, first at row 2)" in found[0]
    assert "events.state_values" in found[0]


def test_state_values_mapping():
    config = parse_config({"events": {"state_values": {"RUN": ["Marcia", "Auto"]}}})
    events = load_events(
        HEADER
        + "M1,2026-09-01 06:00,2026-09-01 06:10,marcia,\n"
        + "M1,2026-09-01 06:10,2026-09-01 06:20,AUTO,\n"
        + "M1,2026-09-01 06:20,2026-09-01 06:30,STOP,x\n",
        config,
    )
    assert list(events["state"]) == ["RUN", "RUN", "STOP"]


def test_custom_datetime_format():
    config = parse_config({"events": {"datetime_format": "%d/%m/%Y %H:%M"}})
    events = load_events(HEADER + "M1,03/09/2026 06:00,03/09/2026 06:10,RUN,\n", config)
    assert events.at[0, "start"] == pd.Timestamp("2026-09-03 06:00")
    found = problems(HEADER + "M1,2026-09-03 06:00,03/09/2026 06:10,RUN,\n", config)
    assert found == [
        "row 2: invalid date/time '2026-09-03 06:00' in 'start'"
        " (expected format %d/%m/%Y %H:%M)"
    ]


def test_production_errors_report_rows(tmp_path):
    production = tmp_path / "production.csv"
    production.write_text(
        "machine,date,shift,ideal_cycle_s,good_parts,scrap_parts\n"
        "M1,2026-09-01,A,30,700,20\n"
        "M1,2026-09-01,C,0,-5,x\n"
    )
    with pytest.raises(InputError) as exc:
        load_dataset(csv(HEADER + "M1,2026-09-01 06:00,2026-09-01 06:10,RUN,\n"), production)
    assert exc.value.problems == [
        "row 3: unknown shift 'C' in 'shift' (configured shifts: A, B)",
        "row 3: '0' in 'ideal_cycle_s' is not a number > 0",
        "row 3: '-5' in 'good_parts' is not a number >= 0",
        "row 3: 'x' in 'scrap_parts' is not a number >= 0",
    ]


def test_missing_sheet_lists_available_sheets():
    config = parse_config({"events": {"sheet": "Events"}})
    with pytest.raises(InputError, match="sheet 'Events' not found.*Eventi, Produzione"):
        load_dataset(SAMPLE / "plant_export.xlsx", config=config)


def test_no_production_is_noted():
    dataset = load_dataset(csv(HEADER + "M1,2026-09-01 06:00,2026-09-01 06:10,RUN,\n"))
    assert dataset.production is None
    assert any("No production data" in n for n in dataset.notes)


def test_config_rejects_unknown_keys_and_conflicts():
    with pytest.raises(ConfigError, match="unknown keys"):
        parse_config({"events": {"colums": {}}})
    with pytest.raises(ConfigError, match="mapped to both"):
        parse_config({"events": {"state_values": {"RUN": "X", "STOP": "x"}}})


def test_config_shifts():
    config = parse_config({
        "shifts": {"N": {"start": "22:00", "end": "06:00"}, "D": {"start": 360, "end": "22:00"}},
    })
    assert config.shift("N").duration_min == 480
    assert config.shift("D").duration_min == 960


def test_config_rejects_overlapping_shifts():
    with pytest.raises(ConfigError, match="shifts A and B overlap"):
        parse_config({"shifts": {
            "A": {"start": "06:00", "end": "14:00"},
            "B": {"start": "13:00", "end": "22:00"},
        }})


def test_minutes_and_seconds_timestamps():
    events = load_events(
        HEADER
        + "M1,2026-09-01 06:00,2026-09-01 06:00:45,RUN,\n"
        + "M1,2026-09-01 06:00:45,2026-09-01 06:02,STOP,x\n"
    )
    assert events.at[0, "end"] == pd.Timestamp("2026-09-01 06:00:45")
    assert events.at[1, "end"] == pd.Timestamp("2026-09-01 06:02:00")


def test_zero_length_events_are_dropped_and_reported():
    dataset = load_dataset(csv(
        HEADER
        + "M1,2026-09-01 06:00,2026-09-01 06:10,RUN,\n"
        + "M1,2026-09-01 06:10,2026-09-01 06:10,STOP,x\n"
    ))
    assert len(dataset.events) == 1
    assert dataset.dropped_rows.to_dict("records") == [
        {"what": "events", "row": 3, "reason": "zero-length event"}
    ]


def test_config_loss_options():
    config = parse_config({"micro_stops_as": "availability", "unlogged_time": "exclude"})
    assert (config.micro_stops_as, config.unlogged_time) == ("availability", "exclude")
    assert Config().micro_stops_as == "performance"
    with pytest.raises(ConfigError, match="micro_stops_as must be one of"):
        parse_config({"micro_stops_as": "quality"})
