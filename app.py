from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from aircraft_dashboard import load_aircraft_data, pareto_front


ROOT = Path(__file__).resolve().parent

METRIC_LABELS = {
    "mass_kg": "Mass (kg)",
    "max_distance_m": "Maximum distance (m)",
    "hover_time_s": "Hover time (s)",
    "max_speed_mps": "Maximum speed (m/s)",
    "interferences": "Interferences",
    "battery_capacity_mah": "Battery capacity (mAh)",
    "battery_cells_s": "Battery cells (S)",
    "motor_count": "Motor count",
    "prop_count": "Propeller count",
    "wing_count": "Wing count",
    "arm_count": "Arm count",
    "arm_length_mean_mm": "Mean arm length (mm)",
    "wing_span_total_mm": "Total wing span (mm)",
    "fuselage_length_mm": "Fuselage length (mm)",
    "fuselage_horz_diameter_mm": "Fuselage width (mm)",
    "hub_label": "Hub family",
    "fuselage_label": "Fuselage family",
    "battery_cells_label": "Battery cells",
    "wing_count_label": "Wing count",
    "flight_status": "Flight status",
    "wing_presence": "Wing presence",
}

EXPLORER_NUMERIC_COLUMNS = [
    "mass_kg",
    "max_distance_m",
    "hover_time_s",
    "max_speed_mps",
    "interferences",
    "battery_capacity_mah",
    "battery_cells_s",
    "motor_count",
    "wing_count",
    "arm_count",
    "arm_length_mean_mm",
    "wing_span_total_mm",
    "fuselage_length_mm",
]

EXPLORER_COLOR_COLUMNS = [
    "flight_status",
    "hub_label",
    "fuselage_label",
    "battery_cells_label",
    "wing_count_label",
]


def label(column: str) -> str:
    return METRIC_LABELS.get(column, column.replace("_", " ").title())


def hub_label(raw: str) -> str:
    number = "".join(ch for ch in raw if ch.isdigit())
    if raw == "ConnectedHub6_Sym":
        return "6-arm symmetric hub"
    if raw == "ConnectedHub6_Sym_Aligned":
        return "6-arm symmetric aligned hub"
    if raw == "ConnectedHub6_2_2_2":
        return "6-arm segmented hub (2-2-2)"
    if raw == "ConnectedHub6_1_2_2_1":
        return "6-arm segmented hub (1-2-2-1)"
    if raw == "ConnectedHub4_Sym":
        return "4-arm symmetric hub"
    if raw == "ConnectedHub4_Sym_Aligned":
        return "4-arm symmetric aligned hub"
    if raw == "ConnectedHub4_2_2":
        return "4-arm segmented hub (2-2)"
    if raw == "ConnectedHub4_1_2_1":
        return "4-arm segmented hub (1-2-1)"
    if raw == "ConnectedHub3_2_1":
        return "3-arm segmented hub (2-1)"
    if raw == "ConnectedHub2_Sym_Wide":
        return "2-arm wide symmetric hub"
    return f"{number}-arm hub family" if number else raw


def hub_short_label(raw: str) -> str:
    if raw == "ConnectedHub6_Sym":
        return "6-arm symmetric"
    if raw == "ConnectedHub6_Sym_Aligned":
        return "6-arm aligned"
    if raw == "ConnectedHub6_2_2_2":
        return "6-arm segmented (2-2-2)"
    if raw == "ConnectedHub6_1_2_2_1":
        return "6-arm segmented (1-2-2-1)"
    if raw == "ConnectedHub4_Sym":
        return "4-arm symmetric"
    if raw == "ConnectedHub4_Sym_Aligned":
        return "4-arm aligned"
    if raw == "ConnectedHub4_2_2":
        return "4-arm segmented (2-2)"
    if raw == "ConnectedHub4_1_2_1":
        return "4-arm segmented (1-2-1)"
    if raw == "ConnectedHub3_2_1":
        return "3-arm segmented (2-1)"
    if raw == "ConnectedHub2_Sym_Wide":
        return "2-arm wide"
    return hub_label(raw)


def fuselage_label(raw: str) -> str:
    if "SingleBattery" in raw:
        return "Single-battery fuselage"
    if "DualBattery" in raw:
        return "Dual-battery fuselage"
    return raw


def hub_checkbox_key(hub: str) -> str:
    return f"hub_checkbox_{hub}"


def apply_all_hubs_choice(hubs: list[str]) -> None:
    value = st.session_state["all_hubs_selected"]
    for hub in hubs:
        st.session_state[hub_checkbox_key(hub)] = value


def sync_all_hubs_checkbox(hubs: list[str]) -> None:
    st.session_state["all_hubs_selected"] = all(
        st.session_state.get(hub_checkbox_key(hub), False) for hub in hubs
    )


def status_checkbox_key(status: str) -> str:
    return f"status_checkbox_{status}"


def apply_all_status_choice(statuses: list[str]) -> None:
    value = st.session_state["all_status_selected"]
    for status in statuses:
        st.session_state[status_checkbox_key(status)] = value


def sync_all_status_checkbox(statuses: list[str]) -> None:
    st.session_state["all_status_selected"] = all(
        st.session_state.get(status_checkbox_key(status), False) for status in statuses
    )


def battery_checkbox_key(cell: str) -> str:
    return f"battery_checkbox_{cell}"


def apply_all_batteries_choice(cells: list[str]) -> None:
    value = st.session_state["all_batteries_selected"]
    for cell in cells:
        st.session_state[battery_checkbox_key(cell)] = value


def sync_all_batteries_checkbox(cells: list[str]) -> None:
    st.session_state["all_batteries_selected"] = all(
        st.session_state.get(battery_checkbox_key(cell), False) for cell in cells
    )


@st.cache_data(show_spinner=False)
def get_data() -> tuple[pd.DataFrame, str]:
    frame, data_dir = load_aircraft_data(ROOT)
    return enrich_data(frame), str(data_dir)


def enrich_data(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = frame.copy()
    enriched["hub_label"] = enriched["hub_type"].map(hub_label)
    enriched["fuselage_label"] = enriched["fuselage_type"].map(fuselage_label)

    enriched["flight_status"] = np.select(
        [enriched["can_hover"], (~enriched["can_hover"]) & enriched["can_fly"]],
        ["Hover-capable", "Forward-flight only"],
        default="No viable flight",
    )
    enriched["battery_cells_label"] = enriched["battery_cells_s"].astype(int).astype(str) + "S"
    enriched["wing_count_label"] = enriched["wing_count"].astype(int).astype(str) + " wing(s)"
    enriched["wing_presence"] = np.where(enriched["wing_count"] > 0, "Winged", "Wingless")

    for column in ["mass_kg", "max_distance_m", "hover_time_s", "max_speed_mps", "interferences"]:
        enriched[f"{column}_plot"] = enriched[column].clip(upper=float(enriched[column].quantile(0.99)))

    enriched["recipe_label"] = (
        enriched["hub_label"]
        + " | "
        + enriched["battery_cells_label"]
        + " | "
        + np.where(
            enriched["fuselage_type"].str.contains("Single"),
            "Single battery",
            "Dual battery",
        )
    )
    return enriched


def apply_filters(frame: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("### Global Filters")
    statuses = sorted(frame["flight_status"].unique().tolist())
    st.sidebar.markdown("<div class='sidebar-section-title'>Flight Status</div>", unsafe_allow_html=True)
    if "all_status_selected" not in st.session_state:
        st.session_state["all_status_selected"] = True
    for status in statuses:
        key = status_checkbox_key(status)
        if key not in st.session_state:
            st.session_state[key] = True

    st.sidebar.checkbox(
        "All flight statuses",
        key="all_status_selected",
        on_change=apply_all_status_choice,
        args=(statuses,),
    )
    with st.sidebar.expander("Flight status subset", expanded=False):
        for status in statuses:
            st.checkbox(
                status,
                key=status_checkbox_key(status),
                on_change=sync_all_status_checkbox,
                args=(statuses,),
            )
    selected_status = [status for status in statuses if st.session_state.get(status_checkbox_key(status), False)]
    if not selected_status:
        selected_status = statuses

    st.sidebar.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

    hubs = sorted(frame["hub_type"].unique().tolist(), key=hub_label)
    st.sidebar.markdown("<div class='sidebar-section-title'>Hub Families</div>", unsafe_allow_html=True)
    if "all_hubs_selected" not in st.session_state:
        st.session_state["all_hubs_selected"] = True
    for hub in hubs:
        key = hub_checkbox_key(hub)
        if key not in st.session_state:
            st.session_state[key] = True

    st.sidebar.checkbox(
        "All hub families",
        key="all_hubs_selected",
        on_change=apply_all_hubs_choice,
        args=(hubs,),
    )

    with st.sidebar.expander("Hub family subset", expanded=False):
        for hub in hubs:
            st.checkbox(
                hub_short_label(hub),
                key=hub_checkbox_key(hub),
                on_change=sync_all_hubs_checkbox,
                args=(hubs,),
            )

    selected_hubs = [hub for hub in hubs if st.session_state.get(hub_checkbox_key(hub), False)]
    if not selected_hubs:
        selected_hubs = hubs

    cells = sorted(frame["battery_cells_label"].unique().tolist(), key=lambda s: int(s[:-1]))
    st.sidebar.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='sidebar-section-title'>Battery Cells</div>", unsafe_allow_html=True)
    if "all_batteries_selected" not in st.session_state:
        st.session_state["all_batteries_selected"] = True
    for cell in cells:
        key = battery_checkbox_key(cell)
        if key not in st.session_state:
            st.session_state[key] = True

    st.sidebar.checkbox(
        "All battery cells",
        key="all_batteries_selected",
        on_change=apply_all_batteries_choice,
        args=(cells,),
    )
    with st.sidebar.expander("Battery cell subset", expanded=False):
        for cell in cells:
            st.checkbox(
                cell,
                key=battery_checkbox_key(cell),
                on_change=sync_all_batteries_checkbox,
                args=(cells,),
            )
    selected_cells = [cell for cell in cells if st.session_state.get(battery_checkbox_key(cell), False)]
    if not selected_cells:
        selected_cells = cells

    st.sidebar.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div class='sidebar-section-title'>Mass</div>", unsafe_allow_html=True)

    mass_q995 = float(frame["mass_kg"].quantile(0.995))
    mass_range = st.sidebar.slider(
        "Mass range (kg)",
        min_value=float(frame["mass_kg"].min()),
        max_value=mass_q995,
        value=(float(frame["mass_kg"].quantile(0.01)), float(frame["mass_kg"].quantile(0.95))),
    )

    filtered = frame[
        frame["flight_status"].isin(selected_status)
        & frame["hub_type"].isin(selected_hubs)
        & frame["battery_cells_label"].isin(selected_cells)
        & frame["mass_kg"].between(*mass_range)
    ].reset_index(drop=True)

    st.sidebar.caption(
        f"{len(filtered):,} designs shown • {len(selected_hubs)} hub family"
        f"{'' if len(selected_hubs) == 1 else 'ies'}"
    )
    return filtered


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(255, 181, 82, 0.14), transparent 24%),
                radial-gradient(circle at 85% 9%, rgba(34, 164, 156, 0.10), transparent 20%),
                linear-gradient(180deg, #0d1b26 0%, #102433 100%);
            color: #eef3f7;
        }
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #102736 0%, #163a4d 100%);
        }
        [data-testid="stHeader"] {
            background: transparent;
        }
        [data-testid="stAppViewContainer"] > .main {
            background: transparent;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        .hero {
            background: linear-gradient(135deg, rgba(255,245,224,0.98), rgba(255,224,178,0.92));
            border-radius: 22px;
            padding: 1.4rem 1.5rem;
            box-shadow: 0 18px 42px rgba(16, 31, 45, 0.12);
            border: 1px solid rgba(16, 31, 45, 0.08);
            color: #10212d;
        }
        .finding {
            background: rgba(255, 250, 242, 0.9);
            border-radius: 16px;
            padding: 1rem;
            border-top: 4px solid #d17c3f;
            color: #233c48;
            min-height: 132px;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.3rem;
        }
        section[data-testid="stSidebar"] [data-baseweb="select"],
        section[data-testid="stSidebar"] [data-baseweb="popover"] > div,
        section[data-testid="stSidebar"] .stSlider {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 0.2rem 0.45rem;
        }
        section[data-testid="stSidebar"] [data-baseweb="tag"] {
            background: rgba(255, 214, 153, 0.16) !important;
            border-radius: 999px !important;
            max-width: 100%;
        }
        section[data-testid="stSidebar"] [data-baseweb="tag"] span {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            line-height: 1.2 !important;
        }
        section[data-testid="stSidebar"] label, 
        section[data-testid="stSidebar"] .stMarkdown {
            color: #f4f7fa !important;
        }
        section[data-testid="stSidebar"] details {
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
            margin-top: 0.1rem;
            margin-bottom: 0.45rem;
        }
        section[data-testid="stSidebar"] details summary {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 10px;
            padding: 0.32rem 0.55rem;
        }
        .sidebar-section-title {
            margin: 0.2rem 0 0.35rem 0;
            font-size: 0.84rem;
            font-weight: 600;
            color: #f4f7fa;
            letter-spacing: 0.02em;
        }
        .sidebar-divider {
            height: 1px;
            background: rgba(255,255,255,0.12);
            margin: 0.75rem 0 0.7rem 0;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 999px;
            color: #eef3f7;
            padding: 0.35rem 0.9rem;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(255, 214, 153, 0.22) !important;
            color: #fff5e6 !important;
        }
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 16px;
            padding: 0.8rem 1rem;
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] div,
        .stMarkdown, .stText, .stCaption, p, li, label, h2, h3 {
            color: inherit;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def recipe_summary(frame: pd.DataFrame, min_designs: int = 80) -> pd.DataFrame:
    summary = (
        frame.groupby(["hub_type", "hub_label", "battery_cells_label", "fuselage_type", "fuselage_label"], as_index=False)
        .agg(
            designs=("design_name", "count"),
            fly_rate=("can_fly", "mean"),
            hover_rate=("can_hover", "mean"),
            median_distance=("max_distance_m", "median"),
            p90_distance=("max_distance_m", lambda s: s.quantile(0.9)),
            median_hover=("hover_time_s", "median"),
            p90_hover=("hover_time_s", lambda s: s.quantile(0.9)),
            median_speed=("max_speed_mps", "median"),
            median_mass=("mass_kg", "median"),
        )
    )
    summary["battery_setup"] = np.where(
        summary["fuselage_type"].str.contains("Single"),
        "Single battery",
        "Dual battery",
    )
    summary["recipe"] = summary["hub_label"] + " | " + summary["battery_cells_label"] + " | " + summary["battery_setup"]
    summary = summary[summary["designs"] >= min_designs].sort_values(
        ["hover_rate", "median_distance"], ascending=False
    )
    return summary


def key_findings(frame: pd.DataFrame) -> list[str]:
    hub_stats = (
        frame.groupby(["hub_type", "hub_label"])
        .agg(designs=("design_name", "count"), hover_rate=("can_hover", "mean"), median_distance=("max_distance_m", "median"))
        .query("designs >= 500")
    )
    best_hover_hub = hub_stats.sort_values("hover_rate", ascending=False).iloc[0]

    fuselage_stats = frame.groupby(["fuselage_type", "fuselage_label"]).agg(
        designs=("design_name", "count"),
        hover_rate=("can_hover", "mean"),
        p90_distance=("max_distance_m", lambda s: s.quantile(0.9)),
    )
    better_fuselage = fuselage_stats.sort_values(["hover_rate", "p90_distance"], ascending=False).iloc[0]

    wing_stats = frame.groupby("wing_presence").agg(
        designs=("design_name", "count"),
        fly_rate=("can_fly", "mean"),
        hover_rate=("can_hover", "mean"),
        median_distance=("max_distance_m", "median"),
    )
    wingless = wing_stats.loc["Wingless"]
    winged = wing_stats.loc["Winged"]

    findings = [
        f"{best_hover_hub.name[1]} is the strongest large hub family, with a hover-capable share of {best_hover_hub['hover_rate']:.1%} and median range of {best_hover_hub['median_distance']:,.0f} m.",
        f"{better_fuselage.name[1]} performs better overall than the alternative on the current filters, making fuselage layout a useful comparison dimension.",
        f"Wingless designs dominate this dataset: {wingless['hover_rate']:.1%} hover-capable versus {winged['hover_rate']:.1%} for winged designs.",
    ]
    return findings


def render_glossary() -> None:
    with st.expander("Glossary: what the dataset terms mean"):
        st.markdown(
            """
            - `6-arm symmetric hub`, `4-arm segmented hub (2-2)`, etc.: human-readable names for aircraft layout families from the generator. The number indicates the main arm / connection count. `Sym` means symmetrical, while patterns such as `2-2-2` describe how the arms are arranged.
            - `Single-battery fuselage` / `Dual-battery fuselage`: whether the aircraft body contains one battery or two.
            - `6S`, `4S`, `3S`: battery cell count. Higher values usually mean higher voltage systems.
            - `Hover-capable`: the design achieved a hover time greater than zero in the simulation.
            - `Forward-flight capable`: the design achieved a maximum distance greater than zero.
            - `Wingless` / `Winged`: whether a design has any wings in the tree definition.
            - `Median`: the middle value in a group. Good for typical performance.
            - `90th percentile (p90)`: a "strong but not extreme" upper-end value. Useful because it is less distorted by a few crazy outliers than the maximum.
            - `Recipe`: a recurring combination of hub family + battery cell count + single/dual battery fuselage.
            - `Pareto front`: designs where you cannot improve one objective, like range, without making another one, like hover time, worse.
            """
        )


def render_header(frame: pd.DataFrame, data_dir: str) -> None:
    st.markdown(
        """
        <div class="hero">
            <div style="display:inline-block; padding:0.25rem 0.65rem; border-radius:999px; background:#103649; color:#f6efe0; font-size:0.82rem; text-transform:uppercase; letter-spacing:0.05em;">
                AircraftVerse Dashboard
            </div>
            <h1 style="margin:0.55rem 0 0.25rem 0;">Exploring the aircraft design space</h1>
            <p style="margin:0;">
                Explore which aircraft designs are viable, which design families perform well, and which recurring design recipes produce the strongest trade-offs.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Data source: `{data_dir}`")

    total = len(frame)
    viable_share = 100 * frame["can_fly"].mean()
    hover_share = 100 * frame["can_hover"].mean()
    best_distance = frame["max_distance_m"].max()

    cols = st.columns(4)
    cols[0].metric("Designs in current view", f"{total:,}")
    cols[1].metric("Forward-flight capable", f"{viable_share:.1f}%")
    cols[2].metric("Hover-capable", f"{hover_share:.1f}%")
    cols[3].metric("Best max distance", f"{best_distance:,.0f} m")

    findings = key_findings(frame)
    fact_cols = st.columns(3)
    for idx, finding in enumerate(findings):
        fact_cols[idx].markdown(f'<div class="finding">{finding}</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 0.9rem;'></div>", unsafe_allow_html=True)
    render_glossary()


def render_landscape_tab(frame: pd.DataFrame) -> None:
    st.subheader("1. Viability Landscape")
    st.write("Start with the broadest question: how many simulated aircraft actually work, and what separates successful designs from failures?")

    left, right = st.columns([0.9, 1.1])
    with left:
        status = frame["flight_status"].value_counts().rename_axis("status").reset_index(name="designs")
        fig = px.bar(
            status,
            x="status",
            y="designs",
            color="status",
            color_discrete_map={
                "Hover-capable": "#1d7f6f",
                "Forward-flight only": "#d08c38",
                "No viable flight": "#c7522a",
            },
            title="Most designs are not viable",
            template="plotly_white",
        )
        fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=50, b=10), xaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        family = (
            frame.groupby(["hub_type", "hub_label"], as_index=False)
            .agg(designs=("design_name", "count"), fly_rate=("can_fly", "mean"), hover_rate=("can_hover", "mean"))
            .sort_values("hover_rate", ascending=False)
        )
        fig = px.scatter(
            family,
            x="fly_rate",
            y="hover_rate",
            size="designs",
            color="hub_label",
            hover_name="hub_label",
            labels={"fly_rate": "Forward-flight capable share", "hover_rate": "Hover-capable share"},
            title="Hub families split clearly by success rate",
            template="plotly_white",
        )
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    dist_cols = st.columns(3)
    specs = [
        ("max_distance_m_plot", "#1f9d8a", "Range distribution"),
        ("hover_time_s_plot", "#d08c38", "Hover distribution"),
        ("mass_kg_plot", "#4b6b8a", "Mass distribution"),
    ]
    for idx, (column, color, title) in enumerate(specs):
        fig = px.histogram(frame, x=column, nbins=50, template="plotly_white", title=title)
        fig.update_traces(marker_color=color)
        fig.update_layout(margin=dict(l=10, r=10, t=45, b=10), yaxis_title="Design count", xaxis_title=label(column.replace("_plot", "")))
        dist_cols[idx].plotly_chart(fig, use_container_width=True)

def family_summary(frame: pd.DataFrame, group_col: str) -> pd.DataFrame:
    return (
        frame.groupby(group_col, as_index=False)
        .agg(
            designs=("design_name", "count"),
            fly_rate=("can_fly", "mean"),
            hover_rate=("can_hover", "mean"),
            median_distance=("max_distance_m", "median"),
            p90_distance=("max_distance_m", lambda s: s.quantile(0.9)),
            median_hover=("hover_time_s", "median"),
            p90_hover=("hover_time_s", lambda s: s.quantile(0.9)),
            median_speed=("max_speed_mps", "median"),
            median_mass=("mass_kg", "median"),
        )
        .sort_values("designs", ascending=False)
    )


def render_families_tab(frame: pd.DataFrame) -> None:
    st.subheader("2. Design Families")
    st.write("The most defensible comparisons in this dataset come from meaningful families: hub layouts, battery cell count, fuselage setup, and wing presence.")

    selector_col, metric_col = st.columns(2)
    with selector_col:
        family_label = st.selectbox("Family dimension", ["Hub family", "Battery cells", "Fuselage family", "Wing presence"])
    with metric_col:
        metric = st.selectbox("Primary metric", ["p90_distance", "hover_rate", "p90_hover", "median_speed"])

    family_map = {
        "Hub family": "hub_label",
        "Battery cells": "battery_cells_label",
        "Fuselage family": "fuselage_label",
        "Wing presence": "wing_presence",
    }
    group_col = family_map[family_label]
    summary = family_summary(frame, group_col)
    summary = summary[summary["designs"] >= max(25, int(0.002 * len(frame)))]

    left, right = st.columns([1.1, 1])
    with left:
        fig = px.bar(
            summary.sort_values(metric, ascending=False),
            x=group_col,
            y=metric,
            color="designs",
            color_continuous_scale="YlGnBu",
            title=f"{family_label} ranked by {metric.replace('_', ' ')}",
            template="plotly_white",
        )
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), xaxis_title="", yaxis_title=metric.replace("_", " ").title())
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.scatter(
            summary,
            x="median_mass",
            y="p90_distance",
            size="designs",
            color="hover_rate",
            hover_name=group_col,
            color_continuous_scale="Viridis",
            title="Mass vs. long-range potential",
            labels={"median_mass": "Median mass (kg)", "p90_distance": "90th percentile distance (m)", "hover_rate": "Hover rate"},
            template="plotly_white",
        )
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    top_groups = summary[group_col].head(min(6, len(summary))).tolist()
    detail = frame[frame[group_col].isin(top_groups)].copy()
    fig = px.box(
        detail,
        x=group_col,
        y="max_distance_m",
        color=group_col,
        points=False,
        title="Range spread within the strongest family groups",
        template="plotly_white",
    )
    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=50, b=10), xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(summary.round(2), use_container_width=True, hide_index=True)


def render_recipes_tab(frame: pd.DataFrame) -> None:
    st.subheader("3. High-Performing Recipes")
    st.write("Instead of hunting single outliers, compare recurring design recipes with enough support to be credible.")

    min_designs = st.slider("Minimum designs per recipe", min_value=25, max_value=250, value=80, step=5)
    recipes = recipe_summary(frame, min_designs=min_designs)

    left, right = st.columns([1.05, 1.15])
    with left:
        fig = px.scatter(
            recipes,
            x="hover_rate",
            y="median_distance",
            size="designs",
            color="hub_label",
            hover_name="recipe",
            title="Recipes with both support and performance",
            labels={"hover_rate": "Hover-capable share", "median_distance": "Median distance (m)"},
            template="plotly_white",
        )
        fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.dataframe(
            recipes[
                [
                    "recipe",
                    "designs",
                    "fly_rate",
                    "hover_rate",
                    "median_distance",
                    "p90_distance",
                    "median_hover",
                    "median_mass",
                ]
            ].round(2).head(20),
            use_container_width=True,
            hide_index=True,
        )

    viable = frame[frame["can_fly"]].copy()
    viable["pareto_optimal"] = pareto_front(viable, "max_distance_m", "hover_time_s", "max", "max")
    viable["pareto_label"] = np.where(viable["pareto_optimal"], "Pareto front", "Other viable designs")
    viable["size_plot"] = viable["max_speed_mps"].clip(upper=float(viable["max_speed_mps"].quantile(0.98)))

    fig = px.scatter(
        viable,
        x="max_distance_m_plot",
        y="hover_time_s_plot",
        color="pareto_label",
        size="size_plot",
        hover_name="design_name",
        hover_data={"hub_label": True, "battery_cells_label": True, "mass_kg":":.2f", "wing_count": True},
        color_discrete_map={"Pareto front": "#c7522a", "Other viable designs": "#2b6178"},
        title="Trade-off frontier: range versus hover time",
        template="plotly_white",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)


def render_explorer_tab(frame: pd.DataFrame) -> None:
    st.subheader("4. Free Explorer")
    st.write("This section is mainly for demonstration, custom exploration, and presentation Q&A.")

    ctrl = st.columns(4)
    with ctrl[0]:
        x_col = st.selectbox("X-axis", EXPLORER_NUMERIC_COLUMNS, index=0, format_func=label)
    with ctrl[1]:
        y_col = st.selectbox("Y-axis", EXPLORER_NUMERIC_COLUMNS, index=1, format_func=label)
    with ctrl[2]:
        color_col = st.selectbox("Color by", EXPLORER_COLOR_COLUMNS, index=0, format_func=label)
    with ctrl[3]:
        sample_size = st.slider("Point sample", min_value=2000, max_value=min(20000, len(frame)), value=min(8000, len(frame)), step=1000)

    plot_frame = frame if len(frame) <= sample_size else frame.sample(sample_size, random_state=42)
    fig = px.scatter(
        plot_frame,
        x=x_col,
        y=y_col,
        color=color_col,
        hover_name="design_name",
        hover_data=["hub_label", "fuselage_label", "battery_cells_label", "wing_count"],
        template="plotly_white",
        title=f"{label(x_col)} vs. {label(y_col)}",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        frame[
            [
                "design_name",
                "flight_status",
                "hub_label",
                "fuselage_label",
                "battery_cells_label",
                "wing_count",
                "mass_kg",
                "max_distance_m",
                "hover_time_s",
                "max_speed_mps",
            ]
        ].round(2),
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="AircraftVerse Design Explorer",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    frame, data_dir = get_data()
    filtered = apply_filters(frame)

    if filtered.empty:
        st.error("No designs match the current filters.")
        return

    render_header(filtered, data_dir)

    landscape_tab, families_tab, recipes_tab, explorer_tab = st.tabs(
        ["Viability Landscape", "Design Families", "High-Performing Recipes", "Free Explorer"]
    )

    with landscape_tab:
        render_landscape_tab(filtered)
    with families_tab:
        render_families_tab(filtered)
    with recipes_tab:
        render_recipes_tab(filtered)
    with explorer_tab:
        render_explorer_tab(filtered)


if __name__ == "__main__":
    main()
