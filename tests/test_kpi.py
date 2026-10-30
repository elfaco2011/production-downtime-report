"""KPI tests on tiny datasets whose expected values are computed by hand."""

import io

import pandas as pd
import pytest

from pdr.config import parse_config
from pdr.io import load_dataset
from pdr.kpi import analyze, filter_analysis, kpi_table, pareto, summary, top_stops

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
    assert s["run_min"] == 389  # 60 + 110 + 59 + 60 + 100
    assert s["stop_min"] == 31  # 10 + 1 + 20
    assert s["setup_min"] == 30
    assert s["downtime_min"] == 61
    assert s["availability"] == pytest.approx(389 / 450)  # 0.8644
    assert s["performance"] == pytest.approx(360 / 389)  # 0.9254
    assert s["quality"] == pytest.approx(700 / 720)  # 0.9722
    # A x P x Q = good parts x ideal cycle / planned production time
    assert s["oee"] == pytest.approx(700 * 30 / (450 * 60))  # 0.7778
    assert not s["performance_capped"]


def test_hand_calculated_mtbf_mttr(one_shift):
    s = summary(one_shift)
    assert s["stop_count"] == 3
    assert s["mtbf_min"] == pytest.approx(389 / 3)  # 129.67
    assert s["mttr_min"] == pytest.approx(31 / 3)  # 10.33


def test_micro_stops_and_missing_reasons(one_shift):
    s = summary(one_shift)
    assert s["micro_count"] == 1
    assert s["micro_min"] == 1
    assert s["missing_reason_count"] == 1


def test_micro_stop_threshold_is_configurable():
    s = summary(run(ONE_SHIFT, ONE_SHIFT_PRODUCTION, {"micro_stop_minutes": 15}))
    assert s["micro_count"] == 2  # 10 min and 1 min
    assert s["micro_min"] == 11
    assert s["availability"] == pytest.approx(389 / 450)  # unchanged


def test_hand_calculated_pareto(one_shift):
    p = pareto(one_shift.segments)
    assert list(p["reason"]) == ["mold change", "Unspecified", "nozzle clog", "robot fault"]
    assert list(p["category"]) == ["setup", "unplanned", "unplanned", "unplanned"]
    assert list(p["minutes"]) == [30, 20, 10, 1]
    assert list(p["count"]) == [1, 1, 1, 1]
    assert p["pct"].tolist() == pytest.approx([3000 / 61, 2000 / 61, 1000 / 61, 100 / 61])
    assert p["cum_pct"].tolist() == pytest.approx([49.180, 81.967, 98.361, 100.0], abs=1e-3)


def test_top_stops(one_shift):
    t = top_stops(one_shift.segments)
    assert list(t["reason"]) == ["Unspecified", "nozzle clog", "robot fault"]
    assert list(t["downtime_min"]) == [20, 10, 1]
    assert list(t["row"]) == [10, 3, 5]


def test_performance_above_one_is_capped_and_flagged():
    # 1000 parts x 30 s = 500 min of ideal time in 389 min of run time.
    a = run(ONE_SHIFT, PRODUCTION_HEADER + "M1,2026-09-01,A,30,1000,0\n")
    s = summary(a)
    assert s["performance_raw"] == pytest.approx(500 / 389)
    assert s["performance"] == 1.0
    assert s["performance_capped"]
    assert s["perf_over_1_shifts"] == 1
    assert any("Performance > 100%" in n for n in a.notes)


def test_without_production_only_availability():
    s = summary(run(ONE_SHIFT))
    assert s["availability"] == pytest.approx(389 / 450)
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

    p = pareto(a.segments)
    assert p.loc[0, "minutes"] == 40 and p.loc[0, "count"] == 1
    t = top_stops(a.segments)
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
    t = top_stops(a.segments)
    assert t.loc[0, "downtime_min"] == 45
    assert t.loc[0, "duration_min"] == 525
    assert any("480 logged minutes fall outside" in n for n in a.notes)


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


def test_unlogged_time_counts_as_loss():
    a = run(EVENTS_HEADER + "M1,2026-09-01 06:00,2026-09-01 10:00,RUN,\n")
    s = summary(a)
    assert s["unlogged_min"] == 240
    assert s["availability"] == pytest.approx(0.5)
    assert any("no event logged" in n for n in a.notes)


def test_kpis_aggregate_sums_not_averages():
    # Shift A: 389/450 available; shift B fully running 480/480.
    events = ONE_SHIFT + "M1,2026-09-01 14:00,2026-09-01 22:00,RUN,\n"
    s = summary(run(events))
    assert s["availability"] == pytest.approx((389 + 480) / (450 + 480))


def test_filters():
    events = ONE_SHIFT + "M2,2026-09-02 14:00,2026-09-02 22:00,RUN,\n"
    a = run(events)
    assert summary(filter_analysis(a, machines=["M2"]))["availability"] == 1.0
    assert summary(filter_analysis(a, shifts=["A"]))["run_min"] == 389
    only_day_2 = filter_analysis(a, start=pd.Timestamp("2026-09-02").date())
    assert set(only_day_2.segments["machine"]) == {"M2"}
    assert summary(filter_analysis(a, week="2026-W36"))["shift_min"] == 960


def test_sample_csv_and_mapped_xlsx_give_identical_kpis():
    from pathlib import Path

    from pdr.config import load_config

    root = Path(__file__).resolve().parents[1]
    sample = root / "data" / "sample"
    a = analyze(load_dataset(sample / "events.csv", sample / "production.csv"))
    b = analyze(load_dataset(
        sample / "plant_export.xlsx",
        config=load_config(root / "config" / "mapping.example.yaml"),
    ))
    for by in (["machine", "week"], ["machine", "date", "shift"]):
        pd.testing.assert_frame_equal(kpi_table(a.shifts, by), kpi_table(b.shifts, by))
    pd.testing.assert_frame_equal(pareto(a.segments), pareto(b.segments))
    pd.testing.assert_frame_equal(top_stops(a.segments), top_stops(b.segments))
