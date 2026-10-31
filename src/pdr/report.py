"""Weekly PDF report and Excel KPI workbook.

Used by both the CLI (python -m pdr report) and the dashboard download
buttons. Every sentence in the report is built from computed numbers by
the rules in observations(); nothing is written free-form.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from fpdf import FPDF  # noqa: E402

from .kpi import (  # noqa: E402
    Analysis,
    data_quality,
    filter_analysis,
    kpi_table,
    micro_stops,
    pareto,
    stop_events,
    summary,
    top_stops,
)

# Rule thresholds for automatic observations.
DOMINANT_CAUSE_SHARE = 30.0  # % of a machine's unplanned downtime
OEE_CHANGE_POINTS = 2.0  # percentage points week over week
MICRO_STOP_SHARE = 15.0  # % of all losses
MISSING_REASON_SHARE = 3.0  # % of stops without reason code
N_OBSERVATIONS = 3

CATEGORY_COLORS = {"unplanned": "#2a78d6", "setup": "#eb6834", "micro": "#1baf7a", "unlogged": "#eda100"}
NEUTRAL = "#8a8985"


@dataclass
class WeeklyReport:
    week: str
    previous_week: str
    first_day: date
    last_day: date
    current: dict
    previous: dict | None
    by_machine: pd.DataFrame
    pareto: pd.DataFrame
    top_stops: pd.DataFrame
    observations: list[str]
    data_quality: pd.DataFrame
    micro: dict
    micro_stops_as: str
    micro_stop_min: float
    has_production: bool


def weeks(analysis: Analysis) -> list[str]:
    return sorted(analysis.shifts["week"].unique())


def previous_week(week: str) -> str:
    monday = date.fromisocalendar(int(week[:4]), int(week[-2:]), 1)
    year, num, _ = (monday - timedelta(days=7)).isocalendar()
    return f"{year}-W{num:02d}"


def build_weekly(analysis: Analysis, week: str | None = None) -> WeeklyReport:
    """Collect everything the weekly report shows. Default: last week in the data."""
    available = weeks(analysis)
    if not available:
        raise ValueError("no data to report on")
    week = week or available[-1]
    if week not in available:
        raise ValueError(f"week {week} is not in the data; available: {', '.join(available)}")
    prev = previous_week(week)
    current = filter_analysis(analysis, week=week)
    before = filter_analysis(analysis, week=prev) if prev in available else None
    monday = date.fromisocalendar(int(week[:4]), int(week[-2:]), 1)

    by_machine = kpi_table(current.shifts, ["machine"])
    by_machine["top_cause"], by_machine["top_cause_pct"] = zip(*[
        _top_cause(filter_analysis(current, machines=[m])) for m in by_machine["machine"]
    ])
    if before is not None:
        prev_oee = kpi_table(before.shifts, ["machine"]).set_index("machine")["oee"]
        by_machine["oee_prev"] = by_machine["machine"].map(prev_oee)
    else:
        by_machine["oee_prev"] = float("nan")

    report = WeeklyReport(
        week=week,
        previous_week=prev,
        first_day=monday,
        last_day=monday + timedelta(days=6),
        current=summary(current),
        previous=summary(before) if before is not None else None,
        by_machine=by_machine,
        pareto=pareto(current, "unplanned"),
        top_stops=top_stops(current, n=5),
        observations=[],
        data_quality=data_quality(current).rows(),
        micro=micro_stops(current),
        micro_stops_as=analysis.config.micro_stops_as,
        micro_stop_min=analysis.config.micro_stop_min,
        has_production=analysis.has_production,
    )
    report.observations = observations(report, current)
    return report


def _top_cause(a: Analysis) -> tuple[str, float]:
    p = pareto(a, "unplanned")
    if p.empty:
        return ("-", 0.0)
    return (p.loc[0, "reason"], float(p.loc[0, "pct"]))


def observations(report: WeeklyReport, current: Analysis) -> list[str]:
    """Up to three findings from fixed rules in priority order, one per rule."""
    found: list[str] = []
    m = report.by_machine

    # 1. The cause that most dominates one machine's unplanned downtime.
    dominant = m[m["top_cause_pct"] >= DOMINANT_CAUSE_SHARE].sort_values(
        "top_cause_pct", ascending=False
    )
    for row in dominant.head(1).itertuples():
        minutes = pareto(filter_analysis(current, machines=[row.machine])).loc[0, "minutes"]
        found.append(
            f"{row.machine}: {row.top_cause} = {row.top_cause_pct:.0f}% of unplanned "
            f"downtime ({minutes:,.0f} min)."
        )

    # 2. Largest week-over-week OEE change per machine.
    if report.has_production:
        change = (100 * (m["oee"] - m["oee_prev"])).dropna()
        if not change.empty:
            i = change.abs().idxmax()
            if abs(change[i]) >= OEE_CHANGE_POINTS:
                found.append(
                    f"{m.loc[i, 'machine']}: OEE {change[i]:+.1f} pts vs "
                    f"{report.previous_week} ({100 * m.loc[i, 'oee_prev']:.1f}% to "
                    f"{100 * m.loc[i, 'oee']:.1f}%)."
                )

    # 3. Lowest OEE machine against the plant figure.
    if report.has_production and len(m) > 1 and report.current["oee"] is not None:
        low = m.loc[m["oee"].idxmin()]
        found.append(
            f"{low['machine']} has the lowest OEE: {100 * low['oee']:.1f}% "
            f"(all machines: {100 * report.current['oee']:.1f}%)."
        )

    # 4. Micro-stops as a share of all losses.
    losses = pareto(current, "all_losses")
    micro = losses[losses["category"] == "micro"]
    if not micro.empty and micro["pct"].iloc[0] >= MICRO_STOP_SHARE:
        top = report.micro["top_causes"]
        cause = f"; most frequent: {top.loc[0, 'reason']}" if not top.empty else ""
        found.append(
            f"Micro-stops: {report.micro['count']:,} events, "
            f"{micro['pct'].iloc[0]:.0f}% of all lost time{cause}."
        )

    # 5. Missing reason codes.
    dq = data_quality(current)
    if dq.stops_without_reason_pct >= MISSING_REASON_SHARE:
        found.append(
            f"{dq.stops_without_reason} stops ({dq.stops_without_reason_pct:.1f}%) "
            "have no reason code."
        )
    return found[:N_OBSERVATIONS]


# --- charts ------------------------------------------------------------------


# Bars at least this tall (% of the axis) show their label inside.
INSIDE_LABEL_MIN_PCT = 12


def pareto_png(table: pd.DataFrame, width_in: float = 7.0, height_in: float = 2.4) -> bytes:
    """Pareto as PNG: share per cause (bars) and cumulative share (line), one axis."""
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=200)
    x = range(len(table))
    colors = [CATEGORY_COLORS.get(c, NEUTRAL) for c in table["category"]]
    ax.bar(x, table["pct"], color=colors, width=0.7, zorder=2)
    ax.plot(x, table["cum_pct"], color=NEUTRAL, linewidth=2, marker="o", markersize=4, zorder=3)
    for i, (pct, minutes) in enumerate(zip(table["pct"], table["minutes"])):
        # The cumulative marker sits on top of the first bar, so tall bars carry
        # their label inside; short bars are far below the line.
        if pct >= INSIDE_LABEL_MIN_PCT:
            ax.text(i, pct - 2, f"{minutes:,.0f}", ha="center", va="top", fontsize=7,
                    color="white", fontweight="bold", zorder=4)
        else:
            ax.text(i, pct + 2, f"{minutes:,.0f}", ha="center", va="bottom", fontsize=7,
                    color="#52514e", zorder=4)
    ax.set_xticks(list(x))
    ax.set_xticklabels([_wrap(r) for r in table["reason"]], fontsize=7)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", color="#e5e4e0", linewidth=0.6, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.set_title("Share of unplanned downtime (bars, labelled in minutes) and cumulative share (line)",
                 fontsize=8, loc="left", color="#52514e")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def _wrap(label: str, width: int = 12) -> str:
    lines, line = [], ""
    for word in str(label).split():
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    lines.append(line)
    return "\n".join(lines)


# --- PDF -----------------------------------------------------------------------


def _pct(v) -> str:
    return "-" if v is None or pd.isna(v) else f"{100 * v:.1f}%"


def _num(v, unit: str = "", digits: int = 0) -> str:
    return "-" if v is None or pd.isna(v) else f"{v:,.{digits}f}{unit}"


def _latin1(text: str) -> str:
    """Core PDF fonts are Latin-1; replace anything else instead of failing."""
    text = str(text).replace("\u2014", "-").replace("\u2013", "-")
    return text.encode("latin-1", "replace").decode("latin-1")


class _PDF(FPDF):
    title_text = ""
    generated = ""

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", size=7)
        self.set_text_color(120)
        self.cell(0, 5, _latin1(self.generated))
        self.set_x(self.l_margin)
        self.cell(0, 5, _latin1(f"{self.title_text}  |  page {self.page_no()}"), align="R")


def render_pdf(report: WeeklyReport, source: str = "") -> bytes:
    pdf = _PDF(format="A4")
    pdf.title_text = f"Weekly downtime report {report.week}"
    pdf.generated = f"Generated {datetime.now():%Y-%m-%d %H:%M}"
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 9, _latin1(f"Weekly downtime report - {report.week}"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=9)
    pdf.set_text_color(90)
    machines = ", ".join(report.by_machine["machine"])
    period = f"{report.first_day:%d %b %Y} to {report.last_day:%d %b %Y}"
    line = f"{period}  |  {machines}"
    if source:
        line += f"  |  source: {source}"
    pdf.multi_cell(0, 5, _latin1(line), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0)
    pdf.ln(2)

    _heading(pdf, "Key figures")
    cur, prev = report.current, report.previous or {}
    rows = [
        ("OEE", "oee", _pct, "pts"),
        ("Availability", "availability", _pct, "pts"),
        ("Performance", "performance", _pct, "pts"),
        ("Quality", "quality", _pct, "pts"),
        ("Downtime (h)", "downtime_min", lambda v: _num(None if v is None else v / 60, digits=1), "h"),
        ("Unplanned stops", "stop_count", _num, ""),
        ("MTBF (min)", "mtbf_min", _num, "min"),
        ("MTTR (min)", "mttr_min", lambda v: _num(v, digits=1), "min"),
        ("Micro-stops", "micro_count", _num, ""),
    ]
    table = [["", report.week, report.previous_week if report.previous else "Previous week", "Change"]]
    for label, key, fmt, unit in rows:
        table.append([label, fmt(cur.get(key)), fmt(prev.get(key)) if prev else "-",
                      _change(cur.get(key), prev.get(key), unit)])
    _table(pdf, table, [55, 40, 40, 45])
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(90)
    pdf.multi_cell(
        0, 4,
        _latin1(
            f"MTBF and MTTR exclude micro-stops (< {report.micro_stop_min:g} min). "
            f"Micro-stops are counted as a {report.micro_stops_as} loss. "
            "Downtime = unplanned stops + setups"
            + (" + micro-stops." if report.micro_stops_as == "availability" else ".")
        ),
        new_x="LMARGIN", new_y="NEXT",
    )
    pdf.set_text_color(0)
    pdf.ln(2)

    _heading(pdf, "Observations")
    pdf.set_font("Helvetica", size=9)
    for text in report.observations or ["No rule-based observation applies to this week."]:
        pdf.multi_cell(0, 5, _latin1(f"-  {text}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    _heading(pdf, "By machine")
    machine_rows = [["Machine", "OEE", "Prev. week", "Availability", "Performance", "Quality",
                     "Downtime (h)", "Top cause"]]
    for r in report.by_machine.itertuples():
        machine_rows.append([
            r.machine, _pct(r.oee), _pct(r.oee_prev), _pct(r.availability), _pct(r.performance),
            _pct(r.quality), _num(r.downtime_min / 60, digits=1),
            f"{r.top_cause} ({r.top_cause_pct:.0f}%)" if r.top_cause != "-" else "-",
        ])
    _table(pdf, machine_rows, [18, 16, 19, 22, 23, 17, 21, 44], size=8, left=[0, 7])
    pdf.ln(2)

    _heading(pdf, "Downtime by cause (unplanned stops)")
    if report.pareto.empty:
        pdf.set_font("Helvetica", size=9)
        pdf.cell(0, 6, "No unplanned stops this week.", new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.image(io.BytesIO(pareto_png(report.pareto)), w=180)
    pdf.ln(2)

    _heading(pdf, "Top 5 longest stops")
    stop_rows = [["Machine", "Start", "End", "Reason", "Min. in shift", "Real duration"]]
    for r in report.top_stops.itertuples():
        stop_rows.append([
            r.machine, f"{r.start:%a %d %b %H:%M}", f"{r.end:%a %d %b %H:%M}", r.reason,
            _num(r.downtime_min), _duration(r.duration_min),
        ])
    _table(pdf, stop_rows, [18, 32, 32, 50, 22, 26], size=8, left=[0, 3])
    pdf.ln(2)

    _heading(pdf, "Data quality")
    pdf.set_font("Helvetica", size=8)
    items = [
        f"[{'check' if r.status == 'warning' else r.status}]  {r.check}: {r.value}"
        for r in report.data_quality.itertuples()
    ]
    half = (len(items) + 1) // 2
    for left, right in zip(items[:half], items[half:] + [""]):
        pdf.cell(90, 4.5, _latin1(left))
        pdf.cell(90, 4.5, _latin1(right), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


def _duration(minutes: float) -> str:
    if minutes < 120:
        return f"{minutes:,.0f} min"
    return f"{minutes / 60:,.1f} h"


def _change(cur, prev, unit: str) -> str:
    if cur is None or prev is None or pd.isna(cur) or pd.isna(prev):
        return "-"
    if unit == "pts":
        return f"{100 * (cur - prev):+.1f} pts"
    if unit == "h":
        return f"{(cur - prev) / 60:+.1f} h"
    diff = cur - prev
    return f"{diff:+,.0f}" + (f" {unit}" if unit else "")


def _heading(pdf: FPDF, text: str) -> None:
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30)
    pdf.cell(0, 7, _latin1(text), new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0)


def _table(
    pdf: FPDF,
    rows: list[list[str]],
    widths: list[float],
    size: int = 9,
    left: list[int] | None = None,
) -> None:
    """Simple table; columns listed in `left` are left-aligned, others right."""
    left = left if left is not None else [0]
    pdf.set_font("Helvetica", size=size)
    with pdf.table(
        col_widths=widths,
        width=sum(widths),
        align="LEFT",
        line_height=size * 0.55,
        borders_layout="HORIZONTAL_LINES",
        first_row_as_headings=True,
        headings_style=_headings_style(),
        text_align=["LEFT" if i in left else "RIGHT" for i in range(len(widths))],
    ) as table:
        for row in rows:
            r = table.row()
            for value in row:
                r.cell(_latin1(value))


def _headings_style():
    from fpdf.fonts import FontFace

    return FontFace(emphasis="BOLD", fill_color=(240, 239, 236))


# --- Excel ---------------------------------------------------------------------


def render_excel(analysis: Analysis) -> bytes:
    """One sheet per KPI view of the (filtered) analysis."""
    kpi_columns = [
        "oee", "availability", "performance", "quality", "planned_production_min",
        "run_min", "downtime_min", "stop_min", "setup_min", "micro_min",
        "unlogged_min", "stop_count", "mtbf_min", "mttr_min", "micro_count",
        "good_parts", "scrap_parts", "performance_capped",
    ]

    def kpis(by: list[str]) -> pd.DataFrame:
        t = kpi_table(analysis.shifts, by)
        return t[by + kpi_columns] if by else t[kpi_columns]

    micro = micro_stops(analysis)
    dq = data_quality(analysis)
    sheets = {
        "Summary": kpis([]),
        "By machine": kpis(["machine"]),
        "By week": kpis(["machine", "week"]),
        "By day": kpis(["machine", "date"]),
        "By shift": kpis(["machine", "date", "shift"]),
        "Pareto unplanned": pareto(analysis, "unplanned"),
        "Pareto all losses": pareto(analysis, "all_losses"),
        "MTBF MTTR": kpis(["machine"])[["machine", "stop_count", "run_min", "stop_min",
                                        "mtbf_min", "mttr_min"]],
        "Micro-stops": micro["top_causes"],
        "Stops": stop_events(analysis, include_micro=True),
        "Data quality": dq.rows().drop(columns="warning"),
        "Settings": pd.DataFrame([
            ("micro_stop_minutes", analysis.config.micro_stop_min),
            ("micro_stops_as", analysis.config.micro_stops_as),
            ("unlogged_time", analysis.config.unlogged_time),
            ("shifts", ", ".join(
                f"{s.name} {s.start:%H:%M}-{s.end:%H:%M}" for s in analysis.config.shifts
            )),
            ("note", "MTBF, MTTR and stop counts exclude micro-stops."),
        ], columns=["setting", "value"]),
    }
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df = df.copy()
            for col in df.select_dtypes("datetimetz").columns:
                df[col] = df[col].dt.tz_localize(None)
            df.to_excel(writer, sheet_name=name, index=False)
            _format_sheet(writer.sheets[name], df)
    return buf.getvalue()


PERCENT_COLUMNS = {"oee", "availability", "performance", "quality"}


def _format_sheet(ws, df: pd.DataFrame) -> None:
    from openpyxl.utils import get_column_letter

    ws.freeze_panes = "A2"
    for i, col in enumerate(df.columns, start=1):
        letter = get_column_letter(i)
        width = max(len(str(col)), *(len(str(v)) for v in df[col].head(200))) if len(df) else len(str(col))
        ws.column_dimensions[letter].width = min(max(10, width + 2), 45)
        if col in PERCENT_COLUMNS:
            fmt = "0.0%"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            fmt = "yyyy-mm-dd hh:mm:ss" if col in {"start", "end"} else "yyyy-mm-dd"
        elif pd.api.types.is_float_dtype(df[col]):
            fmt = "#,##0.0"
        else:
            continue
        for cell in ws[letter][1:]:
            cell.number_format = fmt
