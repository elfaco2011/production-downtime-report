from pathlib import Path

import pandas as pd

from pdr.__main__ import main

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample"


def test_cli_csv_and_mapped_xlsx(tmp_path, capsys):
    assert main([
        "report", "--input", str(SAMPLE / "events.csv"),
        "--production", str(SAMPLE / "production.csv"),
        "--week", "2026-W43", "--output", str(tmp_path / "csv"),
    ]) == 0
    assert main([
        "report", "--input", str(SAMPLE / "plant_export.xlsx"),
        "--mapping", str(ROOT / "config" / "mapping.example.yaml"),
        "--output", str(tmp_path / "xlsx"),
    ]) == 0
    out = capsys.readouterr().out
    assert out.count("IMM-02: mold temperature alarm") == 2
    for folder in ("csv", "xlsx"):
        assert (tmp_path / folder / "downtime_report_2026-W43.pdf").exists()
        assert (tmp_path / folder / "downtime_kpis_2026-W43.xlsx").exists()
    a = pd.read_excel(tmp_path / "csv" / "downtime_kpis_2026-W43.xlsx", sheet_name=None)
    b = pd.read_excel(tmp_path / "xlsx" / "downtime_kpis_2026-W43.xlsx", sheet_name=None)
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])


def test_cli_reports_input_errors(tmp_path, capsys):
    bad = tmp_path / "bad.csv"
    bad.write_text("machine,start,end,state,reason\nM1,yesterday,2026-09-01 07:00,RUN,\n")
    assert main(["report", "--input", str(bad), "--output", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert "Excel row 2: invalid date/time 'yesterday' in 'start'" in err
    assert main(["report", "--input", str(tmp_path / "missing.csv")]) == 2
    assert "file not found" in capsys.readouterr().err
