"""KPI tests on tiny datasets whose expected values are computed by hand."""

import io
from pathlib import Path

import pandas as pd
import pytest

from pdr.config import load_config, parse_config
from pdr.io import load_dataset
from pdr.kpi import (
    analyze,
    data_quality,
    filter_analysis,
    kpi_table,
    micro_stops,
    pareto,
    summary,
    top_stops,
)

EVENTS_HEADER = "machine,start,end,state,reason\n"
PRODUCTION_HEADER = "machine,date,shift,ideal_cycle_s,good_parts,scrap_parts\n"

# One machine, one shift (A, 06:00-14:00 = 480 min):
#   RUN      06:00-07:00   60
#   STOP     07:00-07:10   10  nozzle clog
#   RUN      07:10-09:00  110
#   STOP     09:00-09:01    1  robot fault     (micro-stop, < 2 min)
#   RUN      09:01-10:00   59
#   PLANNED  10:00-10:30   30  break
#   SETUP    10:30-11:00   30  mold change
#   RUN      11:00-12:00   60
#   STOP     12:00-12:20   20  (no reason)    -> Unspecified
#   RUN      12:20-14:00  100
ONE_SHIFT = EVENTS_HEADER + """\
M1,2026-09-01 06:00,2026-09-01 07:00,RUN,
M1,2026-09-01 07:00,2026-09-01 07:10,STOP,nozzle clog
M1,2026-09-01 07:10,2026-09-01 09:00,RUN,
M1,2026-09-01 09:00,2026-09-01 09:01,STOP,robot fault
M1,2026-09-01 09:01,2026-09-01 10:00,RUN,
M1,2026-09-01 10:00,2026-09-01 10:30,PLANNED,break
M1,2026-09-01 10:30,2026-09-01 11:00,SETUP,mold change
M1,2026-09-01 11:00,2026-09-01 12:00,RUN,
M1,2026-09-01 12:00,2026-09-01 12:20,STOP,
M1,2026-09-01 12:20,2026-09-01 14:00,RUN,
"""
# With micro_stops_as=performance (default) the 1-minute micro-stop counts
# as run time: run = 389 + 1 = 390 min.
# 30 s ideal cycle, 700 good + 20 scrap = 720 parts -> 360 min of ideal time.
ONE_SHIFT_PRODUCTION = PRODUCTION_HEADER + "M1,2026-09-01,A,30,700,20\n"


def _file(text: str, name: str) -> io.BytesIO:
    buf = io.BytesIO(text.encode())
    buf.name = name
    return buf


def run(events: str, production: str | None = None, config: dict | None = None):
    dataset = load_dataset(
        _file(events, "events.csv"),
        _file(production, "production.csv") if production else None,
        parse_config(config or {}),
    )
    return analyze(dataset)


@pytest.fixture
def one_shift():
    return run(ONE_SHIFT, ONE_SHIFT_PRODUCTION)


def test_segments_cover_only_real_events(one_shift):
    s = one_shift.segments
    assert len(s) == 10  # one shift, no event crosses a boundary
    assert s.notna().all().all()


def test_hand_calculated_oee(one_shift):
    s = summary(one_shift)
    assert s["shift_min"] == 480
    assert s["planned_min"] == 30
    assert s["planned_production_min"] == 450
    assert s["run_min"] == 390  # 60 + 110 + 59 + 60 + 100 + 1 micro
    assert s["stop_min"] == 30  # 10 + 20, micro-stop excluded
    assert s["setup_min"] == 30
    assert s["micro_min"] == 1
    assert s["downtime_min"] == 60  # stops + setup
    assert s["availability"] == pytest.approx(390 / 450)  # 0.8667
    assert s["performance"] == pytest.approx(360 / 390)  # 0.9231
    assert s["quality"] == pytest.approx(700 / 720)  # 0.9722
    # A x P x Q = good parts x ideal cycle / planned production time
    assert s["oee"] == pytest.approx(700 * 30 / (450 * 60))  # 0.7778
    assert not s["performance_capped"]


def test_hand_calculated_oee_micro_stops_as_availability():
    s = summary(run(ONE_SHIFT, ONE_SHIFT_PRODUCTION, {"micro_stops_as": "availability"}))
    assert s["run_min"] == 389
    assert s["downtime_min"] == 61  # stops + setup + micro
    assert s["availability"] == pytest.approx(389 / 450)  # 0.8644
    assert s["performance"] == pytest.approx(360 / 389)  # 0.9254
    assert s["oee"] == pytest.approx(700 * 30 / (450 * 60))  # unchanged
    assert s["mtbf_min"] == pytest.approx(389 / 2)


def test_hand_calculated_mtbf_mttr(one_shift):
    s = summary(one_shift)
    assert s["stop_count"] == 2  # the micro-stop is not counted
    assert s["mtbf_min"] == pytest.approx(390 / 2)  # 195
    assert s["mttr_min"] == pytest.approx(30 / 2)  # 15


def test_micro_stops_and_missing_reasons(one_shift):
    m = micro_stops(one_shift)
    assert (m["count"], m["minutes"]) == (1, 1)
    assert m["top_causes"].to_dict("records") == [
        {"reason": "robot fault", "count": 1, "minutes": 1.0}
    ]
    s = summary(one_shift)
    assert s["micro_count"] == 1
    assert s["missing_reason_count"] == 1


def test_micro_stop_threshold_is_configurable():
    s = summary(run(ONE_SHIFT, ONE_SHIFT_PRODUCTION, {"micro_stop_minutes": 15}))
    assert s["micro_count"] == 2  # 10 min and 1 min
    assert s["micro_min"] == 11
    assert s["stop_count"] == 1
    assert s["availability"] == pytest.approx((389 + 11) / 450)


def test_hand_calculated_pareto_unplanned(one_shift):
    p = pareto(one_shift)
    assert list(p["reason"]) == ["Unspecified", "nozzle clog"]
    assert list(p["category"]) == ["unplanned", "unplanned"]
    assert list(p["minutes"]) == [20, 10]
    assert list(p["count"]) == [1, 1]
    assert p["cum_pct"].tolist() == pytest.approx([200 / 3, 100])


def test_hand_calculated_pareto_all_losses(one_shift):
    p = pareto(one_shift, view="all_losses")
    assert list(p["reason"]) == ["mold change", "Unspecified", "nozzle clog", "Micro-stops"]
    assert list(p["category"]) == ["setup", "unplanned", "unplanned", "micro"]
    assert list(p["minutes"]) == [30, 20, 10, 1]
    assert list(p["count"]) == [1, 1, 1, 1]
    assert p["pct"].tolist() == pytest.approx([3000 / 61, 2000 / 61, 1000 / 61, 100 / 61])
    assert p["cum_pct"].tolist() == pytest.approx([49.180, 81.967, 98.361, 100.0], abs=1e-3)


def test_top_stops(one_shift):
    t = top_stops(one_shift)
    assert list(t["reason"]) == ["Unspecified", "nozzle clog"]
    assert list(t["downtime_min"]) == [20, 10]
    assert list(t["row"]) == [10, 3]


def test_performance_above_one_is_capped_and_flagged():
    # 1000 parts x 30 s = 500 min of ideal time in 390 min of run time.
    a = run(ONE_SHIFT, PRODUCTION_HEADER + "M1,2026-09-01,A,30,1000,0\n")
    s = summary(a)
    assert s["performance_raw"] == pytest.approx(500 / 390)
    assert s["performance"] == 1.0
    assert s["performance_capped"]
    over = data_quality(a).shifts_performance_over_1
    assert over[["machine", "shift"]].values.tolist() == [["M1", "A"]]


def test_without_production_only_availability():
    s = summary(run(ONE_SHIFT))
    assert s["availability"] == pytest.approx(390 / 450)
    assert s["performance"] is None
    assert s["oee"] is None


def test_stop_across_shift_boundary_is_split():
    # Stop 13:50-14:30: 10 min in shift A, 30 min in shift B, counted once (in A).
    a = run(EVENTS_HEADER + """\
M1,2026-09-01 06:00,2026-09-01 13:50,RUN,
M1,2026-09-01 13:50,2026-09-01 14:30,STOP,material shortage
M1,2026-09-01 14:30,2026-09-01 22:00,RUN,
""")
    by_shift = kpi_table(a.shifts, ["shift"]).set_index("shift")
    assert by_shift.loc["A", "stop_min"] == 10
    assert by_shift.loc["B", "stop_min"] == 30
    assert by_shift.loc["A", "run_min"] == 470
    assert by_shift.loc["B", "run_min"] == 450
    assert by_shift.loc["A", "stop_count"] == 1
    assert by_shift.loc["B", "stop_count"] == 0
    assert by_shift.loc["A", "availability"] == pytest.approx(470 / 480)
    assert by_shift.loc["B", "availability"] == pytest.approx(450 / 480)

    p = pareto(a)
    assert p.loc[0, "minutes"] == 40 and p.loc[0, "count"] == 1
    t = top_stops(a)
    assert t.loc[0, "downtime_min"] == 40 and t.loc[0, "duration_min"] == 40
    assert summary(a)["mttr_min"] == 40


def test_stop_across_days_ignores_time_outside_shifts():
    # Stop 21:40 -> 06:25 next day: 20 min in day 1 shift B, 25 min in day 2
    # shift A; the 8 h between 22:00 and 06:00 is outside the shift calendar.
    a = run(EVENTS_HEADER + """\
M1,2026-09-01 14:00,2026-09-01 21:40,RUN,
M1,2026-09-01 21:40,2026-09-02 06:25,STOP,no operator
M1,2026-09-02 06:25,2026-09-02 14:00,RUN,
""")
    by_day = kpi_table(a.shifts, ["date", "shift"])
    assert by_day[["shift", "stop_min", "stop_count"]].values.tolist() == [
        ["B", 20, 1], ["A", 25, 0],
    ]
    t = top_stops(a)
    assert t.loc[0, "downtime_min"] == 45
    assert t.loc[0, "duration_min"] == 525
    assert data_quality(a).outside_shift_min == 480


def test_night_shift_crossing_midnight():
    a = run(
        EVENTS_HEADER
        + "M1,2026-09-01 22:00,2026-09-02 03:00,RUN,\n"
        + "M1,2026-09-02 03:00,2026-09-02 06:00,STOP,robot fault\n",
        config={"shifts": {"N": {"start": "22:00", "end": "06:00"}}},
    )
    t = kpi_table(a.shifts, ["date", "shift"])
    assert len(t) == 1
    assert t.loc[0, "date"] == pd.Timestamp("2026-09-01")
    assert t.loc[0, "availability"] == pytest.approx(300 / 480)


HALF_LOGGED = EVENTS_HEADER + "M1,2026-09-01 06:00,2026-09-01 10:00,RUN,\n"


def test_unlogged_time_as_loss():
    a = run(HALF_LOGGED)
    s = summary(a)
    assert s["unlogged_min"] == 240
    assert s["availability"] == pytest.approx(0.5)
    assert pareto(a, "all_losses").to_dict("records") == [{
        "reason": "Unlogged", "category": "unlogged", "minutes": 240.0,
        "count": 1, "pct": 100.0, "cum_pct": 100.0,
    }]
    assert data_quality(a).unlogged_pct == pytest.approx(50)


def test_unlogged_time_excluded():
    a = run(HALF_LOGGED, config={"unlogged_time": "exclude"})
    s = summary(a)
    assert s["planned_production_min"] == 240
    assert s["availability"] == pytest.approx(1.0)
    assert pareto(a, "all_losses").empty
    assert data_quality(a).unlogged_min == 240  # still reported


def test_machine_days_without_events_are_reported_not_counted():
    a = run(
        ONE_SHIFT
        + "M1,2026-09-02 06:00,2026-09-02 14:00,RUN,\n"
        + "M2,2026-09-01 06:00,2026-09-01 14:00,RUN,\n"
    )
    missing = data_quality(a).machine_days_without_events
    assert missing[["machine", "date"]].values.tolist() == [
        ["M2", pd.Timestamp("2026-09-02")]
    ]
    assert summary(a)["shift_min"] == 3 * 480
    assert filter_analysis(a, machines=["M1"]).missing_machine_days.empty


def test_data_quality_rows(one_shift):
    dq = data_quality(one_shift)
    assert (dq.stops_total, dq.stops_without_reason) == (3, 1)
    assert dq.stops_without_reason_pct == pytest.approx(100 / 3)
    rows = dq.rows().set_index("check")
    assert rows.loc["Stops without reason code", "value"] == "1 of 3 (33.3%)"
    assert rows.loc["Stops without reason code", "warning"]
    assert not rows.loc["Rows dropped while loading", "warning"]


def test_unmatched_production_rows_are_reported():
    a = run(ONE_SHIFT, ONE_SHIFT_PRODUCTION + "M9,2026-09-01,A,30,1,0\n")
    assert data_quality(a).dropped_rows.to_dict("records") == [{
        "what": "production", "row": 3,
        "reason": "no events for this machine, date and shift",
    }]


def test_kpis_aggregate_sums_not_averages():
    # Shift A: 390/450 available; shift B fully running 480/480.
    events = ONE_SHIFT + "M1,2026-09-01 14:00,2026-09-01 22:00,RUN,\n"
    s = summary(run(events))
    assert s["availability"] == pytest.approx((390 + 480) / (450 + 480))


def test_filters():
    events = ONE_SHIFT + "M2,2026-09-02 14:00,2026-09-02 22:00,RUN,\n"
    a = run(events)
    assert summary(filter_analysis(a, machines=["M2"]))["availability"] == 1.0
    assert summary(filter_analysis(a, shifts=["A"]))["run_min"] == 390
    only_day_2 = filter_analysis(a, start=pd.Timestamp("2026-09-02").date())
    assert set(only_day_2.segments["machine"]) == {"M2"}
    assert summary(filter_analysis(a, week="2026-W36"))["shift_min"] == 960


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample"


def sample(**settings):
    config = parse_config(settings)
    return analyze(load_dataset(SAMPLE / "events.csv", SAMPLE / "production.csv", config))


def test_sample_csv_and_mapped_xlsx_give_identical_kpis():
    a = sample()
    b = analyze(load_dataset(
        SAMPLE / "plant_export.xlsx",
        config=load_config(ROOT / "config" / "mapping.example.yaml"),
    ))
    for by in (["machine", "week"], ["machine", "date", "shift"]):
        pd.testing.assert_frame_equal(kpi_table(a.shifts, by), kpi_table(b.shifts, by))
    for view in ("unplanned", "all_losses"):
        pd.testing.assert_frame_equal(pareto(a, view), pareto(b, view))
    pd.testing.assert_frame_equal(top_stops(a), top_stops(b))


def test_oee_identical_in_both_micro_stop_modes():
    perf = kpi_table(sample().shifts, ["machine"])
    avail = kpi_table(sample(micro_stops_as="availability").shifts, ["machine"])
    assert summary(sample())["oee"] == pytest.approx(
        summary(sample(micro_stops_as="availability"))["oee"]
    )
    assert perf["oee"].tolist() == pytest.approx(avail["oee"].tolist())
    assert (perf["availability"] > avail["availability"]).all()
    assert (perf["performance"] < avail["performance"]).all()


def test_sample_has_no_spurious_unlogged_time():
    a = sample()
    assert data_quality(a).unlogged_min == 0
    assert "Unlogged" not in set(pareto(a, "all_losses")["reason"])


def test_downtime_by_hour_splits_at_hour_boundaries():
    from pdr.kpi import downtime_by_hour

    a = run(EVENTS_HEADER + """\
M1,2026-09-01 06:00,2026-09-01 09:50,RUN,
M1,2026-09-01 09:50,2026-09-01 10:20,STOP,robot fault
M1,2026-09-01 10:20,2026-09-01 10:21,STOP,micro
M1,2026-09-01 10:21,2026-09-01 14:00,RUN,
""")
    h = downtime_by_hour(a)
    assert h.to_dict("records") == [
        {"weekday": "Tuesday", "hour": 9, "minutes": 10.0},
        {"weekday": "Tuesday", "hour": 10, "minutes": 20.0},
    ]


def test_stop_events_can_include_micro_stops(one_shift):
    from pdr.kpi import stop_events

    assert len(stop_events(one_shift)) == 2
    events = stop_events(one_shift, include_micro=True)
    assert events["micro"].tolist() == [False, False, True]
