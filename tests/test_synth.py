import pandas as pd
import pytest

from pdr import synth


@pytest.fixture(scope="module")
def data():
    return synth.generate()


def test_reproducible(data):
    events, production = synth.generate()
    pd.testing.assert_frame_equal(events, data[0])
    pd.testing.assert_frame_equal(production, data[1])


def test_shapes(data):
    events, production = data
    assert list(events.columns) == synth.EVENT_COLUMNS
    assert list(production.columns) == synth.PRODUCTION_COLUMNS
    # 3 machines x 8 weeks x 5 days x 2 shifts
    assert len(production) == 3 * 8 * 5 * 2
    assert set(events["machine"]) == set(synth.MACHINES)


def test_weekdays_only(data):
    events, _ = data
    assert (events["start"].dt.day_of_week < 5).all()


def test_no_overlaps_and_within_shifts(data):
    synth.validate_events(data[0])  # raises on failure


def _clip_to_shifts(events):
    """Minutes of each event inside each shift window, keyed by shift."""
    rows = []
    for day in synth._working_days():
        for name, (s, e) in synth.SHIFTS.items():
            lo = pd.Timestamp.combine(day, s)
            hi = pd.Timestamp.combine(day, e)
            overlap = (
                events["end"].clip(upper=hi) - events["start"].clip(lower=lo)
            ) / pd.Timedelta(minutes=1)
            part = events.assign(day=day, shift=name, minutes=overlap)
            rows.append(part[part["minutes"] > 0])
    return pd.concat(rows)


def test_each_shift_fully_covered(data):
    events, _ = data
    per_shift = _clip_to_shifts(events).groupby(["machine", "day", "shift"])[
        "minutes"
    ].sum()
    assert len(per_shift) == 3 * 8 * 5 * 2
    assert (per_shift == synth.SHIFT_MIN).all()


def test_some_long_stops_carry_over(data):
    events, _ = data
    stops = events[events["state"] == "STOP"]
    start_shift = synth._shift_index(stops["start"], closed="left")
    end_shift = synth._shift_index(stops["end"], closed="right")
    crosses = start_shift != end_shift
    long = (stops["end"] - stops["start"]) >= pd.Timedelta(minutes=30)
    share = (crosses & long).sum() / long.sum()
    assert 0.05 <= share <= 0.20
    # Both kinds occur: into the next shift and into the next day.
    next_day = stops["end"].dt.date != stops["start"].dt.date
    assert (crosses & next_day).any()
    assert (crosses & ~next_day).any()
    # Never over a weekend.
    assert ((stops["end"] - stops["start"]) < pd.Timedelta(hours=24)).all()


def test_states_and_reasons(data):
    events, _ = data
    by_state = events.groupby("state")["reason"].unique()
    assert set(by_state["RUN"]) == {""}
    assert set(by_state["SETUP"]) == {"mold change"}
    assert set(by_state["PLANNED"]) == {"preventive maintenance", "break"}
    assert set(by_state["STOP"]) == set(synth.BASE_CAUSES)


def test_one_break_per_shift_near_mid_shift(data):
    events, production = data
    breaks = events[events["reason"] == "break"]
    assert len(breaks) == len(production)
    assert ((breaks["end"] - breaks["start"]) == pd.Timedelta(minutes=30)).all()
    shift_start = breaks["start"].dt.normalize() + pd.to_timedelta(
        (breaks["start"].dt.hour >= 14) * 8 + 6, unit="h"
    )
    center = (breaks["start"] - shift_start) / pd.Timedelta(minutes=1) + 15
    assert center.between(240 - 30, 240 + 30).all()


def test_imm02_mold_temperature_story(data):
    events, _ = data
    stops = events[events["state"] == "STOP"]
    minutes = (stops["end"] - stops["start"]).dt.total_seconds()
    share = (
        minutes[stops["reason"] == "mold temperature alarm"]
        .groupby(stops["machine"]).sum()
        / minutes.groupby(stops["machine"]).sum()
    )
    assert share["IMM-02"] > 0.3
    assert share["IMM-02"] > 2 * share.drop("IMM-02").max()


def test_validate_rejects_overlap():
    t = pd.Timestamp("2026-09-01 06:00")
    bad = pd.DataFrame({
        "machine": ["IMM-01", "IMM-01"],
        "start": [t, t + pd.Timedelta(minutes=5)],
        "end": [t + pd.Timedelta(minutes=10), t + pd.Timedelta(minutes=20)],
        "state": ["RUN", "STOP"],
        "reason": ["", "nozzle clog"],
    })
    with pytest.raises(ValueError, match="overlapping"):
        synth.validate_events(bad)


def test_validate_rejects_outside_shift():
    t = pd.Timestamp("2026-09-01 22:30")
    bad = pd.DataFrame({
        "machine": ["IMM-01"],
        "start": [t],
        "end": [t + pd.Timedelta(minutes=10)],
        "state": ["RUN"],
        "reason": [""],
    })
    with pytest.raises(ValueError, match="outside shift"):
        synth.validate_events(bad)


def test_write_sample_roundtrip(tmp_path):
    paths = synth.write_sample(tmp_path)
    assert [p.name for p in paths] == [
        "events.csv", "production.csv", "plant_export.xlsx",
    ]
    sheets = pd.read_excel(tmp_path / "plant_export.xlsx", sheet_name=None)
    assert set(sheets) == {"Eventi", "Produzione"}
    assert list(sheets["Eventi"].columns) == list(synth.XLSX_EVENT_COLUMNS.values())
    assert len(sheets["Eventi"]) == len(pd.read_csv(tmp_path / "events.csv"))
