"""Smoke tests for the Streamlit dashboard (headless, no browser)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def run_app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def test_demo_renders_without_errors():
    at = run_app()
    assert not at.exception
    assert not at.error
    labels = [m.label for m in at.metric]
    assert labels == [
        "OEE", "Availability", "Performance", "Quality", "Downtime", "MTBF", "MTTR",
    ]
    assert at.metric[0].value.endswith("%")


def test_pareto_toggle_and_micro_stop_setting():
    at = run_app()
    pareto_radio = next(r for r in at.radio if r.label == "Pareto view")
    pareto_radio.set_value("All losses").run()
    assert not at.exception
    oee_before = at.metric[0].value
    micro_radio = next(r for r in at.sidebar.radio if r.label.startswith("Micro-stops"))
    micro_radio.set_value("availability").run()
    assert not at.exception
    assert at.metric[0].value == oee_before  # OEE does not depend on the mode


def test_filter_to_one_machine():
    at = run_app()
    machines = next(m for m in at.sidebar.multiselect if m.label == "Machines")
    machines.set_value(["IMM-02"]).run()
    assert not at.exception
    assert not at.error


def test_download_buttons_present():
    at = run_app()
    assert not at.exception
    labels = [b.proto.label for b in at.get("download_button")]
    assert labels == ["Download weekly PDF report", "Download Excel (current filters)"]


ITALIAN_EVENTS = """\
Macchina;Inizio;Fine;Stato;Causa
P-01;01/09/2026 06:00;01/09/2026 07:30;In marcia;
P-01;01/09/2026 07:30;01/09/2026 07:45;Fermo;stampo bloccato
P-01;01/09/2026 07:45;01/09/2026 14:00;In marcia;
P-01;13/09/2026 06:00;13/09/2026 14:00;In marcia;
"""

BAD_EVENTS = """\
machine,start,end,state,reason
IMM-01,2026-09-01 06:00,2026-09-01 07:00,RUN,
IMM-01,2026-09-01 07:00,2026-09-01 06:50,STOP,nozzle clog
IMM-01,01/09/2026 7h,2026-09-01 08:00,RUN,
IMM-01,2026-09-01 08:00,2026-09-01 08:20,Alarm,
IMM-01,2026-09-01 08:20,,Alarm,robot fault
IMM-02,2026-09-01 06:00,2026-09-01 06:30,Waiting,
IMM-02,2026-09-01 06:30,2026-09-01 14:00,RUN,
"""


def uploaded_app(root: str, events: str, name: str):
    """The page with an uploaded events file (AppTest cannot upload files)."""
    import sys

    sys.path.insert(0, root)
    import app

    app.main(((events.encode(), name), None, ""))


def run_uploaded(events: str, name: str = "events.csv") -> AppTest:
    at = AppTest.from_function(
        uploaded_app, args=(str(Path(APP).parent), events, name), default_timeout=60
    )
    at.run()
    assert not at.exception
    return at


def button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


def test_italian_file_reaches_the_charts_with_the_menus_only():
    at = run_uploaded(ITALIAN_EVENTS)
    assert [s.value for s in at.subheader] == ["Map your columns"]
    columns = {s.label: s.value for s in at.selectbox}
    assert columns["Machine"] == "Macchina" and columns["Reason"] == "Causa"
    assert columns["“In marcia”"] == "RUN" and columns["“Fermo”"] == "STOP"
    dates = next(r for r in at.radio if r.label == "Dates in the file")
    assert dates.value == "dmy"
    button(at, "Apply").click().run()
    assert not at.exception and not at.error
    assert [m.label for m in at.metric][0] == "OEE"
    assert any(s.value == "Downtime by cause" for s in at.subheader)


def test_bad_rows_can_be_skipped_after_mapping():
    at = run_uploaded(BAD_EVENTS)
    states = {s.label: s for s in at.selectbox if s.label.startswith("“")}
    assert {k: s.value for k, s in states.items()} == {"“Alarm”": "STOP", "“Waiting”": "STOP"}
    next(r for r in at.radio if r.label == "Dates in the file").set_value("dmy").run()
    button(at, "Apply").click().run()
    assert at.error[0].value.startswith("**3 rows have errors.**")
    table = at.dataframe[0].value
    assert list(table["Excel row"]) == [3, 4, 6]
    button(at, "Analyse anyway, skipping these rows").click().run()
    assert not at.exception and not at.error
    assert [m.label for m in at.metric][0] == "OEE"
    quality = next(e for e in at.expander if e.label.startswith("Data quality"))
    rows = quality.dataframe[0].value.set_index("check")
    assert rows.loc["Rows dropped while loading", "value"] == "3 (events: Excel rows 3, 4, 6)"


def test_unanswered_question_blocks_apply():
    at = run_uploaded(BAD_EVENTS.replace("Waiting", "Zzz"))
    assert next(s for s in at.selectbox if s.label == "“Zzz”").value is None
    assert button(at, "Apply").disabled
    assert "choose a state for: Zzz" in at.warning[0].value
