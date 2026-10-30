"""Command line interface.

    python -m pdr report --input data/sample/events.csv \
        --production data/sample/production.csv --week 2026-W43 --output output/
    python -m pdr report --input data/sample/plant_export.xlsx \
        --mapping config/mapping.example.yaml --output output/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ConfigError, load_config
from .io import InputError, load_dataset
from .kpi import analyze, filter_analysis
from .report import build_weekly, render_excel, render_pdf, weeks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pdr", description="Production downtime and OEE report.")
    sub = parser.add_subparsers(dest="command", required=True)
    rep = sub.add_parser("report", help="write the weekly PDF report and the Excel KPI workbook")
    rep.add_argument("--input", required=True, type=Path,
                     help="events file (CSV or XLSX; an XLSX may also hold the production sheet)")
    rep.add_argument("--production", type=Path, help="production file, if separate")
    rep.add_argument("--mapping", type=Path, help="YAML column mapping and settings")
    rep.add_argument("--week", help="ISO week such as 2026-W43 (default: last week in the data)")
    rep.add_argument("--output", type=Path, default=Path("output"), help="output folder")
    args = parser.parse_args(argv)

    try:
        config = load_config(args.mapping)
        dataset = load_dataset(args.input, args.production, config)
    except (InputError, ConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"error: file not found: {exc.filename}", file=sys.stderr)
        return 2

    analysis = analyze(dataset)
    try:
        report = build_weekly(analysis, args.week)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.output.mkdir(parents=True, exist_ok=True)
    pdf_path = args.output / f"downtime_report_{report.week}.pdf"
    xlsx_path = args.output / f"downtime_kpis_{report.week}.xlsx"
    pdf_path.write_bytes(render_pdf(report, source=args.input.name))
    xlsx_path.write_bytes(render_excel(filter_analysis(analysis, week=report.week)))

    print(f"Week {report.week} ({len(weeks(analysis))} weeks in the data)")
    for text in report.observations:
        print(f"  - {text}")
    for note in analysis.notes:
        print(f"  note: {note}")
    print(f"wrote {pdf_path}")
    print(f"wrote {xlsx_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
