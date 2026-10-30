"""Plotly figures for the dashboard.

Colors follow the entity, not its rank: each machine keeps its slot even
when filters hide other machines. The palette is a validated categorical
order (first three slots are colorblind-safe for any pair).
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CATEGORY_COLORS = {
    "unplanned": SERIES[0],
    "setup": SERIES[1],
    "micro": SERIES[2],
    "unlogged": SERIES[3],
}
CATEGORY_LABELS = {
    "unplanned": "Unplanned stop",
    "setup": "Setup",
    "micro": "Micro-stops",
    "unlogged": "Unlogged time",
}
NEUTRAL = "#8a8985"
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def machine_colors(machines: list[str]) -> dict[str, str]:
    """Stable color per machine, from the full (unfiltered) machine list."""
    return {m: SERIES[i % len(SERIES)] for i, m in enumerate(sorted(machines))}


def _wrap(label: str, width: int = 12) -> str:
    """Break a long axis label over several lines instead of rotating it."""
    lines, line = [], ""
    for word in str(label).split():
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    lines.append(line)
    return "<br>".join(lines)


def _layout(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        bargap=0.25,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridwidth=1, zeroline=False)
    return fig


def pareto_chart(pareto: pd.DataFrame) -> go.Figure:
    """Share of downtime per cause (bars) and cumulative share (line).

    Both use the same 0-100 % axis, so there is no second y-axis; minutes
    and counts are in the tooltip.
    """
    fig = go.Figure()
    categories = list(dict.fromkeys(pareto["category"]))
    for category in categories:
        part = pareto[pareto["category"] == category]
        fig.add_bar(
            x=part["reason"],
            y=part["pct"],
            name=CATEGORY_LABELS.get(category, category),
            marker=dict(color=CATEGORY_COLORS.get(category, NEUTRAL), cornerradius=4),
            customdata=part[["minutes", "count"]],
            hovertemplate=(
                "<b>%{x}</b><br>%{y:.1f}% of downtime<br>"
                "%{customdata[0]:,.0f} min · %{customdata[1]:,} events<extra></extra>"
            ),
        )
    fig.add_scatter(
        x=pareto["reason"],
        y=pareto["cum_pct"],
        name="Cumulative %",
        mode="lines+markers",
        line=dict(color=NEUTRAL, width=2),
        marker=dict(size=8),
        hovertemplate="<b>%{x}</b><br>cumulative %{y:.1f}%<extra></extra>",
    )
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=list(pareto["reason"]),
        tickmode="array",
        tickvals=list(pareto["reason"]),
        ticktext=[_wrap(r) for r in pareto["reason"]],
        tickangle=0,
        tickfont=dict(size=11),
    )
    fig.update_yaxes(range=[0, 105], ticksuffix="%", title=None)
    fig.update_layout(barmode="overlay", showlegend=True)
    return _layout(fig, height=420)


def oee_trend_chart(weekly: pd.DataFrame, colors: dict[str, str]) -> go.Figure:
    """Weekly OEE per machine, direct-labeled at the last week."""
    fig = go.Figure()
    for machine, part in weekly.groupby("machine"):
        part = part.sort_values("week")
        fig.add_scatter(
            x=part["week"],
            y=100 * part["oee"],
            name=machine,
            mode="lines+markers",
            line=dict(color=colors[machine], width=2),
            marker=dict(size=8),
            hovertemplate=f"<b>{machine}</b> · %{{x}}<br>OEE %{{y:.1f}}%<extra></extra>",
        )
        last = part.iloc[-1]
        fig.add_annotation(
            x=last["week"], y=100 * last["oee"], text=machine, showarrow=False,
            xanchor="left", xshift=8, font=dict(size=12),
        )
    weeks = sorted(weekly["week"].unique())
    fig.update_yaxes(ticksuffix="%", title=None)
    # 2026-W43 -> W43; the year is in the tooltip.
    fig.update_xaxes(
        title=None, tickmode="array", tickvals=weeks, ticktext=[w[-3:] for w in weeks]
    )
    fig.update_layout(hovermode="x unified")
    return _layout(fig)


def heatmap_chart(by_hour: pd.DataFrame) -> go.Figure:
    """Unplanned stop minutes by weekday (rows) and hour of day (columns)."""
    days = [d for d in WEEKDAYS if d in set(by_hour["weekday"])]
    grid = (
        by_hour.pivot(index="weekday", columns="hour", values="minutes")
        .reindex(index=days)
        .reindex(columns=range(int(by_hour["hour"].min()), int(by_hour["hour"].max()) + 1))
        .fillna(0)
    )
    fig = go.Figure(go.Heatmap(
        z=grid.values,
        x=[f"{h:02d}" for h in grid.columns],
        y=grid.index,
        colorscale=[[i / (len(SEQUENTIAL) - 1), c] for i, c in enumerate(SEQUENTIAL)],
        xgap=2,
        ygap=2,
        colorbar=dict(title="min", thickness=12),
        hovertemplate="%{y} %{x}<br>%{z:,.0f} min of stops<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(
        title=dict(text="hour of day", font=dict(size=11)), tickangle=0, type="category"
    )
    return _layout(fig, height=340)
