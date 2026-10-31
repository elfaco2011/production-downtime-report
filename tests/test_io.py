import io
from pathlib import Path

import pandas as pd
import pytest
import yaml

from pdr.config import Config, ConfigError, load_config, parse_config
from pdr.io import UNSPECIFIED, InputError, load_dataset

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample"
MAPPING = ROOT / "config" / "mapping.example.yaml"

HEADER = "machine,start,end,state,reason\n"
ISO_HINT = (
    " (expected YYYY-MM-DD HH:MM[:SS]; for other formats choose the "
    "date format or set date_order in the mapping file)"
)


def yaml_of(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


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
        "Excel row 3: invalid date/time 'not a date' in 'start'" + ISO_HINT,
        "Excel row 4: missing date/time in 'end'",
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
        "Excel row 2: end (2026-09-01 06:50:00) is before start (2026-09-01 07:00:00)",
        "Excel row 3: invalid date/time '01/09/2026 07:00' in 'start'" + ISO_HINT,
        found[2],
    ]
    assert found[2].startswith("unknown value(s) in 'state'")


def test_error_message_lists_problems_in_row_order():
    with pytest.raises(InputError) as exc:
        load_events(
            HEADER
            + "M1,2026-09-01 07:00,2026-09-01 06:50,STOP,x\n"
            + "M1,bad,2026-09-01 08:00,RUN,\n"
        )
    lines = str(exc.value).splitlines()
    assert lines[1].startswith("  - Excel row 2:") and lines[2].startswith("  - Excel row 3:")


BLANK_ROWS = [
    ["machine", "start", "end", "state", "reason"],
    ["M1", "2026-09-01 06:00", "2026-09-01 06:10", "RUN", None],
    [None] * 5,  # a gap left in the sheet: Excel row 3
    ["M1", "2026-09-01 06:10", "2026-09-01 06:05", "STOP", "x"],  # Excel row 4
]


def test_row_numbers_match_excel_in_csv_with_blank_lines():
    text = "\n".join(",".join(c or "" for c in r) for r in BLANK_ROWS) + "\n"
    text = text.replace("\n,,,,\n", "\n\n")  # a truly empty line, as Excel saves it
    assert problems(text) == [
        "Excel row 4: end (2026-09-01 06:05:00) is before start (2026-09-01 06:10:00)"
    ]


def test_row_numbers_match_excel_in_xlsx_with_blank_rows():
    import openpyxl

    wb = openpyxl.Workbook()
    for r in BLANK_ROWS:
        wb.active.append(r)
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    buf.name = "events.xlsx"
    with pytest.raises(InputError) as exc:
        load_dataset(buf)
    assert exc.value.problems == [
        "Excel row 4: end (2026-09-01 06:05:00) is before start (2026-09-01 06:10:00)"
    ]


@pytest.mark.parametrize("order, expected", [
    ("dmy", "2026-09-01 07:00"), ("mdy", "2026-01-09 07:00"),
])
def test_date_order(order, expected):
    config = parse_config({"date_order": order})
    events = load_events(
        HEADER
        + "M1,01/09/2026 07:00,01/09/2026 08:00:30,RUN,\n"
        + "M2,2026-09-01 07:00,2026-09-01 08:00,RUN,\n",  # ISO is always accepted
        config,
    )
    assert events.at[0, "start"] == pd.Timestamp(expected)
    assert events.at[1, "start"] == pd.Timestamp("2026-09-01 07:00")


def test_date_order_is_strict():
    # Neither guessed nor partly read: "7h" is not a time.
    found = problems(HEADER + "M1,01/09/2026 7h,2026-09-01 08:00,RUN,\n",
                     parse_config({"date_order": "dmy"}))
    assert found == [
        "Excel row 2: invalid date/time '01/09/2026 7h' in 'start'"
        " (expected DD/MM/YYYY HH:MM[:SS] or YYYY-MM-DD HH:MM[:SS])"
    ]


def test_date_order_applies_to_production_and_excel_cells(tmp_path):
    config = parse_config({"date_order": "dmy"})
    production = tmp_path / "production.csv"
    production.write_text(
        "machine,date,shift,ideal_cycle_s,good_parts,scrap_parts\n"
        "M1,01/09/2026,A,30,700,20\n"
    )
    dataset = load_dataset(
        csv(HEADER + "M1,01/09/2026 06:00,01/09/2026 06:10,RUN,\n"), production, config
    )
    assert dataset.production.at[0, "date"] == pd.Timestamp("2026-09-01")
    mapped = load_config(MAPPING)
    xlsx = load_dataset(SAMPLE / "plant_export.xlsx",
                        config=parse_config({**yaml_of(MAPPING), "date_order": "dmy"}))
    pd.testing.assert_frame_equal(xlsx.events, load_dataset(
        SAMPLE / "plant_export.xlsx", config=mapped).events)


def test_end_before_start_reports_row():
    found = problems(HEADER + "M1,2026-09-01 07:00,2026-09-01 06:00,STOP,x\n")
    assert found == [
        "Excel row 2: end (2026-09-01 06:00:00) is before start (2026-09-01 07:00:00)"
    ]


def test_overlapping_events_report_rows():
    found = problems(
        HEADER
        + "M1,2026-09-01 06:00,2026-09-01 06:30,RUN,\n"
        + "M1,2026-09-01 06:20,2026-09-01 06:40,STOP,x\n"
        + "M2,2026-09-01 06:20,2026-09-01 06:40,STOP,x\n"
    )
    assert found == ["Excel row 3: overlaps Excel row 2 on machine M1"]


BAD_ROWS = (
    HEADER
    + "M1,2026-09-01 06:00,2026-09-01 08:00,RUN,\n"           # 2 kept
    + "M1,2026-09-01 06:30,2026-09-01 06:45,STOP,x\n"         # 3 overlaps 2
    + "M1,2026-09-01 07:00,2026-09-01 07:30,STOP,x\n"         # 4 overlaps 2, not 3
    + "M1,2026-09-01 08:00,,STOP,x\n"                         # 5 missing end
    + "M1,2026-09-01 08:30,2026-09-01 08:20,STOP,x\n"         # 6 end before start
    + "M1,2026-09-01 09:00,2026-09-01 10:00,RUN,\n"           # 7 kept
)


def test_invalid_rows_are_listed_with_column():
    with pytest.raises(InputError) as exc:
        load_events(BAD_ROWS)
    assert exc.value.rows_only and exc.value.bad_rows == 4
    table = exc.value.table()
    assert list(table.columns) == ["file", "Excel row", "column", "problem"]
    assert table[["Excel row", "column"]].values.tolist() == [
        [3, "start"], [4, "start"], [5, "end"], [6, "end"],
    ]
    assert table.at[0, "problem"] == "overlaps Excel row 2 on machine M1"
    # Overlaps are checked among valid rows only: row 4 overlaps row 2, not row 3.
    assert table.at[1, "problem"] == "overlaps Excel row 2 on machine M1"


def test_skip_invalid_rows():
    dataset = load_dataset(csv(BAD_ROWS), skip_invalid_rows=True)
    assert list(dataset.events["row"]) == [2, 7]
    assert dataset.dropped_rows.to_dict("records") == [
        {"what": "events", "row": 3, "reason": "overlaps Excel row 2 on machine M1"},
        {"what": "events", "row": 4, "reason": "overlaps Excel row 2 on machine M1"},
        {"what": "events", "row": 5, "reason": "missing date/time in 'end'"},
        {"what": "events", "row": 6, "reason":
            "end (2026-09-01 08:20:00) is before start (2026-09-01 08:30:00)"},
    ]


def test_unmapped_states_are_never_skipped():
    text = HEADER + "M1,2026-09-01 06:00,2026-09-01 06:10,Alarm,\n"
    with pytest.raises(InputError) as exc:
        load_dataset(csv(text), skip_invalid_rows=True)
    assert not exc.value.rows_only
    assert exc.value.issues[0].column == "state" and exc.value.issues[0].row is None


def test_events_and_production_problems_are_reported_together(tmp_path):
    production = tmp_path / "production.csv"
    production.write_text(
        "machine,date,shift,ideal_cycle_s,good_parts,scrap_parts\n"
        "M1,2026-09-01,A,30,700,x\n"
    )
    with pytest.raises(InputError) as exc:
        load_dataset(csv(HEADER + "M1,2026-09-01 07:00,2026-09-01 06:00,STOP,x\n"), production)
    assert exc.value.what == "events and production"
    assert [(p.what, p.row) for p in exc.value.issues] == [("events", 2), ("production", 2)]
    assert str(exc.value).splitlines()[0] == "events: 1 problem(s) found"
    dataset = load_dataset(
        csv(HEADER + "M1,2026-09-01 07:00,2026-09-01 06:00,STOP,x\n"
            + "M1,2026-09-01 06:00,2026-09-01 06:10,RUN,\n"),
        production, skip_invalid_rows=True,
    )
    assert dataset.production.empty
    assert dataset.dropped_rows[["what", "row"]].values.tolist() == [
        ["events", 2], ["production", 2],
    ]


def test_unmapped_state_values_are_listed():
    found = problems(
        HEADER
        + "M1,2026-09-01 06:00,2026-09-01 06:10,Marcia,\n"
        + "M1,2026-09-01 06:10,2026-09-01 06:20,Allarme,x\n"
        + "M1,2026-09-01 06:20,2026-09-01 06:30,Allarme,x\n"
    )
    assert len(found) == 1
    assert "'Allarme' (2 rows, first at Excel row 3)" in found[0]
    assert "'Marcia' (1 row, first at Excel row 2)" in found[0]
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
        "Excel row 2: invalid date/time '2026-09-03 06:00' in 'start'"
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
        "Excel row 3: unknown shift 'C' in 'shift' (configured shifts: A, B)",
        "Excel row 3: '0' in 'ideal_cycle_s' is not a number > 0",
        "Excel row 3: '-5' in 'good_parts' is not a number >= 0",
        "Excel row 3: 'x' in 'scrap_parts' is not a number >= 0",
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
