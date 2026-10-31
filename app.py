"""Streamlit dashboard: production downtime and OEE.

Run with:  streamlit run app.py

Uploaded files are processed in memory only; nothing is written to disk
or sent anywhere.
"""

from __future__ import annotations

import io
from pathlib import Path

import streamlit as st
import yaml

from pdr import charts
from pdr.config import ConfigError, parse_config
from pdr.io import InputError, load_dataset
from pdr.kpi import (
    Analysis,
    analyze,
    data_quality,
    downtime_by_hour,
    filter_analysis,
    kpi_table,
    micro_stops,
    pareto,
    stop_events,
    summary,
)
from pdr.report import build_weekly, render_excel, render_pdf, weeks

SAMPLE = Path(__file__).parent / "data" / "sample"

st.set_page_config(page_title="Production Downtime Report", page_icon="🏭", layout="wide")


# --- loading ---------------------------------------------------------------


def _named(data: bytes, name: str) -> io.BytesIO:
    buf = io.BytesIO(data)
    buf.name = name
    return buf


@st.cache_data(show_spinner="Analysing…")
def run_analysis(
    events: tuple[bytes, str],
    production: tuple[bytes, str] | None,
    mapping_text: str,
    micro_stops_as: str,
) -> Analysis:
    raw = (yaml.safe_load(mapping_text) if mapping_text else None) or {}
    raw = {**raw, "micro_stops_as": micro_stops_as}
    config = parse_config(raw)
    dataset = load_dataset(
        _named(*events),
        _named(*production) if production else None,
        config,
    )
    return analyze(dataset)


def demo_inputs():
    return (
        ((SAMPLE / "events.csv").read_bytes(), "events.csv"),
        ((SAMPLE / "production.csv").read_bytes(), "production.csv"),
        "",
    )


def sidebar_inputs():
    st.sidebar.header("Data")
    source = st.sidebar.radio(
        "Source", ["Demo data", "Upload my own files"], label_visibility="collapsed"
    )
    if source == "Demo data":
        st.sidebar.caption(
            "Synthetic data: 3 injection molding machines, 8 weeks, 2 shifts."
        )
        return demo_inputs()

    events = st.sidebar.file_uploader(
        "Machine events (CSV or XLSX)", type=["csv", "xlsx"],
        help="One row per state change: machine, start, end, state, reason. "
        "An XLSX file may also contain the production sheet.",
    )
    production = st.sidebar.file_uploader(
        "Production per shift (CSV or XLSX, optional)", type=["csv", "xlsx"],
        help="machine, date, shift, ideal cycle (s), good parts, scrap parts. "
        "Needed for Performance, Quality and OEE.",
    )
    mapping = st.sidebar.file_uploader(
        "Column mapping (YAML, optional)", type=["yaml", "yml"],
        help="Maps your column names and state labels; see config/mapping.example.yaml.",
    )
    st.sidebar.caption("Files are processed in memory and never leave this app.")
    if events is None:
        return None
    return (
        (events.getvalue(), events.name),
        (production.getvalue(), production.name) if production else None,
        mapping.getvalue().decode("utf-8") if mapping else "",
    )


def show_input_error(exc: Exception) -> None:
    if isinstance(exc, InputError):
        st.error(f"**The {exc.what} could not be read.** Please fix these rows and upload again.")
        st.markdown("\n".join(f"- {p}" for p in exc.problems[:50]))
        if len(exc.problems) > 50:
            st.caption(f"… and {len(exc.problems) - 50} more.")
    else:
        st.error(f"**The mapping file is not valid:** {exc}")


# --- page sections ----------------------------------------------------------


def fmt_pct(value) -> str:
    return "–" if value is None else f"{100 * value:.1f}%"


def fmt_min(value) -> str:
    return "–" if value is None else f"{value:,.0f} min"


def kpi_cards(a: Analysis) -> None:
    s = summary(a)
    cols = st.columns(7)
    cols[0].metric("OEE", fmt_pct(s["oee"]))
    cols[1].metric("Availability", fmt_pct(s["availability"]))
    cols[2].metric(
        "Performance", fmt_pct(s["performance"]),
        help="Capped at 100%; see Data quality." if s["performance_capped"] else None,
    )
    cols[3].metric("Quality", fmt_pct(s["quality"]))
    cols[4].metric("Downtime", f"{s['downtime_min'] / 60:,.1f} h",
                   help="Unplanned stops + setups"
                   + (" + micro-stops." if a.config.micro_stops_as == "availability" else "."))
    cols[5].metric("MTBF", fmt_min(s["mtbf_min"]), help="Micro-stops excluded.")
    cols[6].metric("MTTR", fmt_min(s["mttr_min"]), help="Micro-stops excluded.")
    m = micro_stops(a)
    st.caption(
        f"MTBF / MTTR: micro-stops excluded · {s['stop_count']:,.0f} stops · "
        f"micro-stops (< {m['threshold_min']:g} min): {m['count']:,} events, "
        f"{m['minutes']:,.0f} min, counted as {m['counted_as']} loss."
    )


def pareto_section(a: Analysis) -> None:
    st.subheader("Downtime by cause")
    view = st.radio(
        "Pareto view", ["Unplanned stops", "All losses"], horizontal=True,
        help="All losses adds setups, micro-stops and unlogged time.",
    )
    table = pareto(a, "unplanned" if view == "Unplanned stops" else "all_losses")
    if table.empty:
        st.info("No downtime in the selection.")
        return
    st.plotly_chart(charts.pareto_chart(table), width="stretch")
    with st.expander("Pareto table"):
        st.dataframe(
            table.rename(columns={"pct": "%", "cum_pct": "cumulative %"}).round(1),
            hide_index=True, width="stretch",
        )


def trend_section(a: Analysis, colors: dict[str, str]) -> None:
    st.subheader("Weekly OEE by machine")
    if not a.has_production:
        st.info("Upload production data to see OEE.")
        return
    weekly = kpi_table(a.shifts, ["machine", "week"]).dropna(subset=["oee"])
    if weekly.empty:
        st.info("No production in the selection.")
        return
    st.plotly_chart(charts.oee_trend_chart(weekly, colors), width="stretch")


def heatmap_section(a: Analysis) -> None:
    st.subheader("When do stops happen?")
    by_hour = downtime_by_hour(a)
    if by_hour.empty:
        st.info("No unplanned stops in the selection.")
        return
    st.plotly_chart(charts.heatmap_chart(by_hour), width="stretch")
    st.caption("Minutes of unplanned stops per weekday and hour (micro-stops excluded).")


def stops_section(a: Analysis) -> None:
    st.subheader("Stops")
    c1, c2, c3 = st.columns([3, 1, 1])
    include_micro = c3.toggle("Include micro-stops")
    events = stop_events(a, include_micro=include_micro)
    reasons = c1.multiselect("Reason", sorted(events["reason"].unique()))
    min_minutes = c2.number_input("Min. minutes", min_value=0.0, value=0.0, step=5.0)
    if reasons:
        events = events[events["reason"].isin(reasons)]
    events = events[events["downtime_min"] >= min_minutes]
    st.dataframe(
        events.rename(columns={
            "downtime_min": "minutes in shift",
            "duration_min": "real duration (min)",
            "row": "file row",
        }).round({"minutes in shift": 1, "real duration (min)": 1}),
        hide_index=True, width="stretch", height=360,
    )
    st.caption(f"{len(events):,} stops. 'minutes in shift' excludes time outside the shift calendar.")


def data_quality_section(a: Analysis) -> None:
    dq = data_quality(a)
    rows = dq.rows()
    warnings = int(rows["warning"].sum())
    with st.expander(f"Data quality · {warnings} item(s) to check", expanded=False):
        shown = rows.assign(
            status=rows["status"].map({"warning": "⚠️ check", "ok": "✅ ok", "info": "ℹ️ info"})
        )
        st.dataframe(shown[["status", "check", "value"]], hide_index=True, width="stretch")
        if not dq.machine_days_without_events.empty:
            st.markdown("**Machine-days without events** (not counted in KPIs)")
            st.dataframe(dq.machine_days_without_events, hide_index=True)
        if not dq.shifts_performance_over_1.empty:
            st.markdown("**Shifts with Performance > 100%** (check the ideal cycle time)")
            st.dataframe(dq.shifts_performance_over_1.round(3), hide_index=True)
        if not dq.dropped_rows.empty:
            st.markdown("**Rows not used**")
            st.dataframe(dq.dropped_rows, hide_index=True)
        for note in a.notes:
            st.caption(note)


def downloads_section(a: Analysis, source: str) -> None:
    """Report downloads; files are generated in memory when clicked."""
    st.subheader("Reports")
    available = weeks(a)
    c1, c2, c3 = st.columns([1, 1, 1])
    week = c1.selectbox("Week for the PDF report", available, index=len(available) - 1)
    c2.download_button(
        "Download weekly PDF report",
        data=lambda: render_pdf(build_weekly(a, week), source=source),
        file_name=f"downtime_report_{week}.pdf",
        mime="application/pdf",
        icon=":material/picture_as_pdf:",
        width="stretch",
    )
    c3.download_button(
        "Download Excel (current filters)",
        data=lambda: render_excel(a),
        file_name="downtime_kpis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        icon=":material/table_view:",
        width="stretch",
    )
    st.caption(
        "The PDF covers one ISO week of the selected machines and shifts and compares "
        "it with the week before. The Excel file holds one sheet per KPI view."
    )


def sidebar_filters(a: Analysis) -> Analysis:
    st.sidebar.header("Filters")
    machines = sorted(a.shifts["machine"].unique())
    chosen = st.sidebar.multiselect("Machines", machines, default=machines)
    first, last = a.shifts["date"].min().date(), a.shifts["date"].max().date()
    dates = st.sidebar.date_input("Dates", (first, last), min_value=first, max_value=last)
    start, end = (dates if isinstance(dates, tuple) and len(dates) == 2 else (first, last))
    shift_names = [s.name for s in a.config.shifts]
    shifts = st.sidebar.multiselect("Shifts", shift_names, default=shift_names)
    return filter_analysis(a, machines=chosen, start=start, end=end, shifts=shifts)


# --- page -------------------------------------------------------------------


def main() -> None:
    st.title("Production downtime & OEE")
    st.markdown(
        "Upload your machine event log to see **where production time is lost**: "
        "OEE, the biggest causes of downtime, when stops happen, and how each "
        "machine trends week by week."
    )

    inputs = sidebar_inputs()
    st.sidebar.header("Settings")
    micro_as = st.sidebar.radio(
        "Micro-stops count as a loss of",
        ["performance", "availability"],
        help="Short stops (< 2 min by default). OEE is the same either way; "
        "only the split between Availability and Performance changes.",
    )
    if inputs is None:
        st.info("Upload a machine event file in the sidebar, or switch to the demo data.")
        return
    try:
        analysis = run_analysis(*inputs, micro_as)
    except (InputError, ConfigError) as exc:
        show_input_error(exc)
        return
    except yaml.YAMLError as exc:
        st.error(f"**The mapping file is not valid YAML:** {exc}")
        return

    colors = charts.machine_colors(list(analysis.shifts["machine"].unique()))
    view = sidebar_filters(analysis)
    if view.shifts.empty:
        st.warning("No data for the selected filters.")
        return

    kpi_cards(view)
    data_quality_section(view)
    pareto_section(view)
    left, right = st.columns(2)
    with left:
        trend_section(view, colors)
    with right:
        heatmap_section(view)
    stops_section(view)
    downloads_section(view, source=inputs[0][1])


main()
