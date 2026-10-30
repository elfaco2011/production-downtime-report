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
