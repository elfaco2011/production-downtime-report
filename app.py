"""Streamlit dashboard: production downtime and OEE.

Run with:  streamlit run app.py

Uploaded files are processed in memory only; nothing is written to disk
or sent anywhere. When an upload does not match the expected layout, the
page asks the user to map columns, state labels and the date format
(logic in pdr.mapping) instead of stopping with an error.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from pdr import charts
from pdr.config import (
    DATE_ORDERS,
    EVENT_FIELDS,
    PRODUCTION_FIELDS,
    STATES,
    Config,
    ConfigError,
    parse_config,
)
from pdr.io import InputError, RawTables, load_dataset, read_tables
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
from pdr.mapping import (
    DATE_ORDER_LABELS,
    build_mapping,
    diagnose,
    mapping_yaml,
    preview_dates,
    suggest_columns,
    suggest_date_order,
    suggest_state,
    unknown_state_labels,
)
from pdr.report import build_weekly, render_excel, render_pdf, weeks

SAMPLE = Path(__file__).parent / "data" / "sample"
UPWORK_URL = "https://www.upwork.com/freelancers/~0113002181207316dd"
CONTACT = (
    "Need this configured for your plant's export? "
    f"[Contact me on Upwork]({UPWORK_URL})."
)
FIELD_LABELS = {
    "machine": "Machine", "start": "Start time", "end": "End time", "state": "State",
    "reason": "Reason", "date": "Date", "shift": "Shift",
    "ideal_cycle_s": "Ideal cycle time (s)", "good_parts": "Good parts",
    "scrap_parts": "Scrap parts",
}

# (events, production, mapping YAML text); files as (bytes, name).
Inputs = tuple[tuple[bytes, str], tuple[bytes, str] | None, str]


# --- loading ---------------------------------------------------------------


def _named(data: bytes, name: str) -> io.BytesIO:
    buf = io.BytesIO(data)
    buf.name = name
    return buf


@st.cache_data(show_spinner="Reading files…")
def read_raw(
    events: tuple[bytes, str], production: tuple[bytes, str] | None, mapping_text: str
) -> RawTables:
    config = parse_config(yaml.safe_load(mapping_text) or {})
    return read_tables(_named(*events), _named(*production) if production else None, config)


@st.cache_data(show_spinner="Analysing…")
def run_analysis(
    events: tuple[bytes, str],
    production: tuple[bytes, str] | None,
    mapping_text: str,
    micro_stops_as: str,
    skip_invalid_rows: bool = False,
) -> Analysis:
    raw = (yaml.safe_load(mapping_text) if mapping_text else None) or {}
    raw = {**raw, "micro_stops_as": micro_stops_as}
    config = parse_config(raw)
    dataset = load_dataset(
        _named(*events),
        _named(*production) if production else None,
        config,
        skip_invalid_rows=skip_invalid_rows,
    )
    return analyze(dataset)


def demo_inputs() -> Inputs:
    return (
        ((SAMPLE / "events.csv").read_bytes(), "events.csv"),
        ((SAMPLE / "production.csv").read_bytes(), "production.csv"),
        "",
    )


def sidebar_inputs() -> Inputs | None:
    st.sidebar.header("Data")
    source = st.sidebar.radio(
        "Source", ["Demo data", "Upload my own files"], label_visibility="collapsed",
        key="source",
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
        help="A mapping.yaml downloaded from this page, or see "
        "config/mapping.example.yaml. Without one, the page asks.",
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
    elif isinstance(exc, yaml.YAMLError):
        st.error(f"**The mapping file is not valid YAML:** {exc}")
    else:
        st.error(f"**The mapping file is not valid:** {exc}")
    st.markdown(CONTACT)


# --- guided mapping ---------------------------------------------------------


def _upload_key(inputs: Inputs) -> str:
    """Identifies the uploaded files; answers are reset when they change."""
    digest = hashlib.sha1()
    for part in (inputs[0], inputs[1]):
        if part:
            digest.update(part[1].encode() + part[0])
    digest.update(inputs[2].encode())
    return digest.hexdigest()[:12]


def _column_selects(
    title: str, raw: pd.DataFrame, fields: tuple[str, ...], current: dict[str, str], key: str
) -> dict[str, str | None]:
    columns = [str(c).strip() for c in raw.columns]
    suggested = suggest_columns(columns, fields)
    st.markdown(f"**{title}**: which column holds…")
    chosen = {}
    cells = st.columns(len(fields))
    for cell, f in zip(cells, fields):
        default = current[f] if current[f] in columns else suggested[f]
        chosen[f] = cell.selectbox(
            FIELD_LABELS[f], columns,
            index=columns.index(default) if default else None,
            placeholder="Choose a column", key=f"{key}_{f}",
        )
    return chosen


def mapping_panel(tables: RawTables, base: dict, current: Config, key: str) -> None:
    st.subheader("Map your columns")
    st.markdown(
        "This file uses its own column names, state labels or date format. "
        "Tell the dashboard what they mean; the most likely answers are already "
        "selected. You can download the answers as a mapping file and upload it "
        "next time."
    )
    reference = parse_config(base)
    events = tables.events.rename(columns=lambda c: str(c).strip())
    event_columns = _column_selects(
        "Machine events", events, EVENT_FIELDS, current.event_columns, f"{key}_ev"
    )
    production_columns = None
    if tables.production is not None:
        production_columns = _column_selects(
            "Production", tables.production, PRODUCTION_FIELDS,
            current.production_columns, f"{key}_pr",
        )

    states: dict[str, str | None] = {}
    if event_columns["state"]:
        labels = unknown_state_labels(events[event_columns["state"]], reference)
        if labels:
            st.markdown("**Machine states**: what does each value mean?")
            st.caption(
                "RUN = producing · STOP = unplanned stop (alarm, fault, waiting) · "
                "SETUP = changeover, mold change · PLANNED = break, planned maintenance, no orders"
            )
            cells = st.columns(4)
            for i, label in enumerate(labels):
                default = current.state_values.get(label.casefold()) or suggest_state(label)
                states[label] = cells[i % 4].selectbox(
                    f"“{label}”", STATES,
                    index=STATES.index(default) if default else None,
                    placeholder="Choose a state", key=f"{key}_state_{label}",
                )

    st.markdown("**Date format**")
    start = events[event_columns["start"]] if event_columns["start"] else None
    default_order = (
        suggest_date_order(start, current.date_order) if start is not None
        else current.date_order
    )
    left, right = st.columns([1, 2])
    order = left.radio(
        "Dates in the file", DATE_ORDERS, index=DATE_ORDERS.index(default_order),
        format_func=DATE_ORDER_LABELS.get, key=f"{key}_dates",
        help="Auto reads YYYY-MM-DD only. Pick the order your plant uses: "
        "01/09/2026 is 1 September as DD/MM and 9 January as MM/DD.",
    )
    if start is not None:
        right.dataframe(preview_dates(start, order), hide_index=True, width="stretch")

    try:
        answers, unanswered = build_mapping(
            event_columns, production_columns, states, order, base
        ), None
    except ConfigError as exc:
        answers, unanswered = None, str(exc)
    if unanswered:
        st.warning(f"Still to answer: {unanswered}")
    apply_col, download_col, _ = st.columns([1, 1, 2])
    if apply_col.button("Apply", type="primary", disabled=answers is None, width="stretch"):
        st.session_state.update(guided=answers, skip_rows=False, edit_mapping=False)
        st.rerun()
    download_col.download_button(
        "Download mapping.yaml",
        data=mapping_yaml(answers) if answers else "",
        file_name="mapping.yaml",
        mime="application/x-yaml",
        disabled=answers is None,
        icon=":material/download:",
        width="stretch",
    )
    st.caption(CONTACT)


def row_errors_panel(exc: InputError) -> None:
    n = exc.bad_rows
    st.error(
        f"**{n} {'row has' if n == 1 else 'rows have'} errors.** Fix them in the file "
        "and upload it again, or analyse the other rows. Skipped rows are listed "
        "under Data quality and in the PDF report."
    )
    st.dataframe(exc.table(), hide_index=True, width="stretch")
    if st.button("Analyse anyway, skipping these rows", type="primary"):
        st.session_state.skip_rows = True
        st.rerun()
    st.markdown(CONTACT)


def load_uploaded(inputs: Inputs, micro_as: str) -> Analysis | None:
    """Analysis of the uploaded files, or None while the page asks the user."""
    state = st.session_state
    key = _upload_key(inputs)
    if state.get("upload_key") != key:
        state.update(upload_key=key, guided=None, skip_rows=False, edit_mapping=False)
    try:
        base = (yaml.safe_load(inputs[2]) if inputs[2] else None) or {}
        applied = state.guided if state.guided is not None else base
        config = parse_config(applied)
        mapping_text = yaml.safe_dump(applied, sort_keys=False, allow_unicode=True)
        tables = read_raw(inputs[0], inputs[1], mapping_text)
    except (InputError, ConfigError, yaml.YAMLError) as exc:
        show_input_error(exc)
        return None

    if state.edit_mapping or diagnose(tables, config).any:
        mapping_panel(tables, base, config, key)
        return None
    try:
        return run_analysis(inputs[0], inputs[1], mapping_text, micro_as, state.skip_rows)
    except InputError as exc:
        if exc.rows_only and not state.skip_rows:
            row_errors_panel(exc)
        else:
            show_input_error(exc)
    except ConfigError as exc:
        show_input_error(exc)
    return None


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
            "row": "Excel row",
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
            st.dataframe(
                dq.dropped_rows.rename(columns={"what": "file", "row": "Excel row"}),
                hide_index=True,
            )
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


def main(inputs: Inputs | None = None) -> None:
    """The page. inputs bypasses the sidebar uploads (used by the tests)."""
    st.set_page_config(page_title="Production Downtime Report", page_icon="🏭", layout="wide")
    try:
        page(inputs)
    finally:
        st.sidebar.divider()
        st.sidebar.caption(CONTACT)


def page(inputs: Inputs | None) -> None:
    st.title("Production downtime & OEE")
    st.markdown(
        "Upload your machine event log to see **where production time is lost**: "
        "OEE, the biggest causes of downtime, when stops happen, and how each "
        "machine trends week by week."
    )

    demo = inputs is None and st.session_state.get("source", "Demo data") == "Demo data"
    if inputs is None:
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
    if demo:
        analysis = run_analysis(*inputs, micro_as)
    else:
        analysis = load_uploaded(inputs, micro_as)
        if analysis is None:
            return
        if st.sidebar.button("Edit column mapping", icon=":material/table_edit:"):
            st.session_state.edit_mapping = True
            st.rerun()

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


if __name__ == "__main__":
    main()
