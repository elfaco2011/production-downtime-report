import re
from pathlib import Path

import pandas as pd
import pytest

from pdr.config import parse_config
from pdr.io import load_dataset
from pdr.kpi import analyze
from pdr.report import build_weekly, previous_week, render_excel, render_pdf

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample"


@pytest.fixture(scope="module")
def analysis():
    return analyze(load_dataset(SAMPLE / "events.csv", SAMPLE / "production.csv"))


def test_previous_week_crosses_year():
    assert previous_week("2026-W43") == "2026-W42"
    assert previous_week("2026-W01") == "2025-W52"


def test_weekly_report_content(analysis):
    r = build_weekly(analysis, "2026-W43")
    assert r.previous_week == "2026-W42" and r.previous is not None
    assert len(r.top_stops) == 5
    assert len(r.observations) == 3
    # Every observation quotes a computed number for a real machine or cause.
    assert r.observations[0].startswith("IMM-02: mold temperature alarm = ")
    share = r.by_machine.set_index("machine").loc["IMM-02", "top_cause_pct"]
    assert f"{share:.0f}%" in r.observations[0]


def test_first_week_has_no_comparison(analysis):
    r = build_weekly(analysis, "2026-W36")
    assert r.previous is None
    assert r.by_machine["oee_prev"].isna().all()


def test_unknown_week_is_rejected(analysis):
    with pytest.raises(ValueError, match="2026-W50 is not in the data"):
        build_weekly(analysis, "2026-W50")


def test_pdf_is_one_page(analysis):
    pdf = render_pdf(build_weekly(analysis), source="events.csv")
    assert pdf.startswith(b"%PDF")
    # The page tree object holds the total page count.
    assert re.findall(rb"/Count (\d+)", pdf) == [b"1"]


def test_pdf_handles_non_latin_text():
    events = (
        "machine,start,end,state,reason\n"
        "Presse-1,2026-09-01 06:00,2026-09-01 07:00,RUN,\n"
        "Presse-1,2026-09-01 07:00,2026-09-01 07:30,STOP,Störung – Düse ✓\n"
        "Presse-1,2026-09-01 07:30,2026-09-01 14:00,RUN,\n"
    )
    import io

    buf = io.BytesIO(events.encode())
    buf.name = "e.csv"
    a = analyze(load_dataset(buf, config=parse_config({})))
    assert render_pdf(build_weekly(a)).startswith(b"%PDF")


def test_excel_has_one_sheet_per_view(analysis):
    sheets = pd.read_excel(pd.io.common.BytesIO(render_excel(analysis)), sheet_name=None)
    assert list(sheets) == [
        "Summary", "By machine", "By week", "By day", "By shift", "Pareto unplanned",
        "Pareto all losses", "MTBF MTTR", "Micro-stops", "Stops", "Data quality", "Settings",
    ]
    assert len(sheets["By week"]) == 3 * 8
