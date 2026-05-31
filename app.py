from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from aircraft_dashboard import load_aircraft_data, pareto_front


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"
HERO_AIRCRAFT_IMAGE = ASSET_DIR / "hero_aircraft_formation.png"
BLUEPRINT_AIRCRAFT_IMAGE = ASSET_DIR / "blueprint_aircraft.png"

ACCENT_BLUE = "#0668f6"
ACCENT_TEAL = "#23b6a8"
ACCENT_PURPLE = "#7257e8"
DARK_NAVY = "#06192b"

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


def image_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def display_path(path_text: str) -> str:
    path = Path(path_text)
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path_text


def configure_plotly_theme() -> None:
    px.defaults.template = "plotly_white"
    px.defaults.color_discrete_sequence = [
        ACCENT_BLUE,
        ACCENT_TEAL,
        ACCENT_PURPLE,
        "#ff9f1c",
        "#e45757",
        "#415a77",
    ]


def polish_figure(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(color="#102033", family="sans-serif", size=13),
        title=dict(font=dict(size=18, color=DARK_NAVY), x=0.02, xanchor="left"),
        legend=dict(
            bgcolor="rgba(255,255,255,0.72)",
            bordercolor="rgba(8,31,53,0.08)",
            borderwidth=1,
        ),
    )
    fig.update_xaxes(
        gridcolor="rgba(8,31,53,0.08)",
        zerolinecolor="rgba(8,31,53,0.12)",
        linecolor="rgba(8,31,53,0.12)",
        title_font=dict(color="#506174"),
        tickfont=dict(color="#506174"),
    )
    fig.update_yaxes(
        gridcolor="rgba(8,31,53,0.08)",
        zerolinecolor="rgba(8,31,53,0.12)",
        linecolor="rgba(8,31,53,0.12)",
        title_font=dict(color="#506174"),
        tickfont=dict(color="#506174"),
    )
    return fig


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
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-mark">AV</div>
            <div>
                <div class="brand-title">AIRCRAFTVERSE</div>
                <div class="brand-subtitle">DASHBOARD</div>
            </div>
        </div>
        <div class="sidebar-main-title">GLOBAL FILTERS</div>
        """,
        unsafe_allow_html=True,
    )
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
    if BLUEPRINT_AIRCRAFT_IMAGE.exists():
        st.sidebar.image(str(BLUEPRINT_AIRCRAFT_IMAGE), width="stretch")
    return filtered


def inject_styles() -> None:
    css = """
        <style>
        :root {
            --navy: #06192b;
            --navy-2: #09243c;
            --blue: #0668f6;
            --teal: #23b6a8;
            --purple: #7257e8;
            --ink: #081f35;
            --muted: #5c6b7d;
            --card: rgba(255, 255, 255, 0.84);
            --line: rgba(8, 31, 53, 0.10);
            --shadow: 0 22px 55px rgba(25, 67, 120, 0.13);
        }
"""
    st.markdown(
        (
            css
            + """
        .stApp {
            background:
                radial-gradient(circle at 84% 4%, rgba(6, 104, 246, 0.11), transparent 28%),
                radial-gradient(circle at 28% 0%, rgba(35, 182, 168, 0.10), transparent 22%),
                linear-gradient(180deg, #f7fbff 0%, #eef5ff 52%, #f8fbff 100%);
            color: var(--ink);
        }
        section[data-testid="stSidebar"] {
            background:
                radial-gradient(circle at 86% 18%, rgba(75, 181, 255, 0.19), transparent 25%),
                radial-gradient(circle at 35% 64%, rgba(21, 89, 145, 0.24), transparent 30%),
                linear-gradient(180deg, #041221 0%, #08213a 52%, #05182b 100%);
            border-right: 1px solid rgba(117, 193, 255, 0.14);
        }
        [data-testid="stHeader"] {
            background: transparent;
        }
        [data-testid="stAppViewContainer"] > .main {
            background: transparent;
        }
        .block-container {
            padding-top: 1.55rem;
            padding-bottom: 3rem;
            max-width: 1420px;
        }
        .hero {
            position: relative;
            min-height: 370px;
            overflow: hidden;
            border-radius: 30px;
            padding: 2.5rem 2.55rem;
            box-shadow: var(--shadow);
            border: 1px solid rgba(120, 158, 205, 0.18);
            color: var(--ink);
            background:
                linear-gradient(90deg, rgba(255,255,255,0.98) 0%, rgba(255,255,255,0.92) 44%, rgba(255,255,255,0.28) 71%, transparent 100%),
                radial-gradient(circle at 77% 24%, rgba(6,104,246,0.12), transparent 30%),
                radial-gradient(circle at 38% 0%, rgba(35,182,168,0.10), transparent 26%),
                linear-gradient(135deg, #ffffff 0%, #eaf5ff 100%);
        }
        .hero::before {
            content: "";
            position: absolute;
            inset: 0;
            pointer-events: none;
            background:
                radial-gradient(circle at 20% 30%, rgba(6,104,246,0.10), transparent 20%),
                linear-gradient(120deg, rgba(255,255,255,0.12), transparent 55%);
        }
        .hero-content {
            position: relative;
            max-width: 670px;
            z-index: 2;
        }
        .hero-visual {
            position: absolute;
            z-index: 1;
            inset: -3.5rem -3.2rem -4.2rem 31%;
            pointer-events: none;
        }
        .hero-visual img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            object-position: right center;
            display: block;
            filter: drop-shadow(0 24px 34px rgba(25, 79, 145, 0.20));
        }
        .hero-visual::after {
            content: "";
            position: absolute;
            inset: 0;
            pointer-events: none;
            background:
                radial-gradient(circle at 58% 36%, rgba(255,255,255,0.26), transparent 23%),
                linear-gradient(90deg, rgba(255,255,255,0.16), transparent 34%);
        }
        .eyebrow {
            display: flex;
            gap: 0.75rem;
            align-items: center;
            color: var(--blue);
            font-weight: 800;
            font-size: 0.86rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }
        .eyebrow-dots {
            letter-spacing: 0.3em;
            color: #65a8ff;
        }
        .hero h1 {
            margin: 0.65rem 0 0.75rem 0;
            font-size: clamp(2.55rem, 5vw, 5.1rem);
            line-height: 0.96;
            letter-spacing: -0.065em;
            color: var(--ink);
            max-width: 690px;
        }
        .hero h1 span {
            color: var(--blue);
        }
        .hero p {
            margin: 0;
            max-width: 650px;
            color: #405267;
            font-size: 1.05rem;
            line-height: 1.62;
        }
        .source-pill {
            display: inline-flex;
            gap: 0.55rem;
            align-items: center;
            margin-top: 1.35rem;
            padding: 0.58rem 0.8rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.76);
            border: 1px solid rgba(8,31,53,0.08);
            box-shadow: 0 10px 30px rgba(25,67,120,0.08);
            color: #52667a;
            font-size: 0.86rem;
        }
        .source-pill code {
            background: rgba(6,104,246,0.08);
            border-radius: 999px;
            padding: 0.22rem 0.55rem;
            color: #425164;
        }
        .kpi-card {
            position: relative;
            min-height: 104px;
            overflow: hidden;
            background: var(--card);
            border: 1px solid rgba(90, 134, 190, 0.16);
            border-radius: 20px;
            padding: 1rem 1.08rem;
            box-shadow: 0 16px 36px rgba(23, 61, 113, 0.11);
            backdrop-filter: blur(18px);
        }
        .kpi-card::after {
            content: "";
            position: absolute;
            right: -2.8rem;
            bottom: -3.4rem;
            width: 8rem;
            height: 8rem;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(6,104,246,0.13), rgba(6,104,246,0.02) 64%, transparent 65%);
            pointer-events: none;
        }
        .kpi-card.purple::after {
            background: radial-gradient(circle, rgba(114,87,232,0.14), rgba(114,87,232,0.02) 64%, transparent 65%);
        }
        .kpi-card.teal::after {
            background: radial-gradient(circle, rgba(35,182,168,0.14), rgba(35,182,168,0.02) 64%, transparent 65%);
        }
        .kpi-top {
            display: flex;
            gap: 0.78rem;
            align-items: center;
            position: relative;
            z-index: 1;
        }
        .kpi-icon {
            display: grid;
            place-items: center;
            width: 2.85rem;
            height: 2.85rem;
            flex: 0 0 2.85rem;
            border-radius: 999px;
            background: linear-gradient(135deg, var(--blue), #04a6ff);
            color: white;
            font-weight: 900;
            letter-spacing: -0.03em;
            box-shadow: 0 12px 24px rgba(6, 104, 246, 0.28);
        }
        .kpi-icon svg {
            width: 1.42rem;
            height: 1.42rem;
            stroke: currentColor;
            stroke-width: 2;
            fill: none;
            stroke-linecap: round;
            stroke-linejoin: round;
        }
        .kpi-card.purple .kpi-icon {
            background: linear-gradient(135deg, var(--purple), #9f8cff);
            box-shadow: 0 12px 24px rgba(114, 87, 232, 0.25);
        }
        .kpi-card.teal .kpi-icon {
            background: linear-gradient(135deg, var(--teal), #48d6c8);
            box-shadow: 0 12px 24px rgba(35, 182, 168, 0.22);
        }
        .kpi-label {
            color: #304155;
            font-weight: 760;
            font-size: 0.84rem;
        }
        .kpi-value {
            color: var(--ink);
            font-size: 1.72rem;
            font-weight: 850;
            line-height: 1.1;
            margin-top: 0.1rem;
        }
        .finding {
            position: relative;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            gap: 1rem;
            overflow: hidden;
            background: rgba(255, 255, 255, 0.76);
            border-radius: 22px;
            padding: 1.2rem 1.25rem;
            border: 1px solid rgba(79, 128, 190, 0.16);
            border-bottom: 4px solid var(--blue);
            box-shadow: 0 16px 36px rgba(23, 61, 113, 0.10);
            color: #263d52;
            min-height: 218px;
            height: 218px;
            backdrop-filter: blur(18px);
        }
        .finding::after {
            content: "";
            position: absolute;
            right: -2.4rem;
            top: -2.4rem;
            width: 8rem;
            height: 8rem;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(6,104,246,0.10), transparent 66%);
            pointer-events: none;
        }
        .finding.teal::after {
            background: radial-gradient(circle, rgba(35,182,168,0.11), transparent 66%);
        }
        .finding.purple::after {
            background: radial-gradient(circle, rgba(114,87,232,0.11), transparent 66%);
        }
        .finding.teal {
            border-bottom-color: var(--teal);
        }
        .finding.purple {
            border-bottom-color: var(--purple);
        }
        .finding-badge {
            display: inline-grid;
            place-items: center;
            width: 2.7rem;
            height: 2.7rem;
            border-radius: 999px;
            background: rgba(6, 104, 246, 0.12);
            color: var(--blue);
            font-weight: 850;
            position: relative;
            z-index: 1;
        }
        .finding-badge svg {
            width: 1.35rem;
            height: 1.35rem;
            stroke: currentColor;
            stroke-width: 2;
            fill: none;
            stroke-linecap: round;
            stroke-linejoin: round;
        }
        .finding-text {
            position: relative;
            z-index: 1;
            line-height: 1.52;
        }
        .finding-label {
            position: relative;
            z-index: 1;
            color: #788799;
            font-size: 0.72rem;
            font-weight: 820;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .finding.teal .finding-badge {
            background: rgba(35, 182, 168, 0.13);
            color: #168d82;
        }
        .finding.purple .finding-badge {
            background: rgba(114, 87, 232, 0.12);
            color: var(--purple);
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 0.95rem;
        }
        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 0.65rem;
            margin: 0.05rem 0 0.7rem 0;
            padding: 0.72rem 0.78rem;
            border-radius: 18px;
            background: rgba(255,255,255,0.055);
            border: 1px solid rgba(145,209,255,0.13);
        }
        .brand-mark {
            display: grid;
            place-items: center;
            width: 2.55rem;
            height: 2.55rem;
            border-radius: 999px;
            background:
                radial-gradient(circle, rgba(77,184,255,0.26), transparent 62%),
                linear-gradient(135deg, rgba(255,255,255,0.16), rgba(255,255,255,0.04));
            border: 1px solid rgba(137, 208, 255, 0.35);
            color: #dff4ff;
            font-weight: 900;
            letter-spacing: -0.05em;
        }
        .brand-title {
            color: #f8fbff;
            font-size: 0.9rem;
            font-weight: 850;
            letter-spacing: 0.045em;
        }
        .brand-subtitle {
            color: #b8d7ef;
            font-size: 0.74rem;
            letter-spacing: 0.07em;
        }
        .sidebar-main-title {
            margin: 0.2rem 0 0.65rem 0;
            color: #5eb7ff;
            font-size: 0.82rem;
            font-weight: 850;
            letter-spacing: 0.08em;
        }
        section[data-testid="stSidebar"] [data-baseweb="select"],
        section[data-testid="stSidebar"] [data-baseweb="popover"] > div,
        section[data-testid="stSidebar"] .stSlider {
            background: rgba(255,255,255,0.07);
            border: 1px solid rgba(145,209,255,0.13);
            border-radius: 14px;
            padding: 0.2rem 0.45rem;
        }
        section[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"],
        section[data-testid="stSidebar"] .stSlider [data-baseweb="tooltip"],
        section[data-testid="stSidebar"] .stSlider [role="tooltip"],
        section[data-testid="stSidebar"] [data-baseweb="tooltip"] {
            display: none;
        }
        section[data-testid="stSidebar"] [data-baseweb="tag"] {
            background: rgba(63, 165, 255, 0.18) !important;
            border-radius: 999px !important;
            max-width: 100%;
        }
        section[data-testid="stSidebar"] [data-baseweb="tag"] span {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            line-height: 1.2 !important;
        }
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] .stCaption,
        section[data-testid="stSidebar"] div[data-testid="stCaptionContainer"],
        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        section[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
        section[data-testid="stSidebar"] [data-testid="stCheckbox"] label,
        section[data-testid="stSidebar"] [data-testid="stCheckbox"] p,
        section[data-testid="stSidebar"] details summary,
        section[data-testid="stSidebar"] details summary *,
        section[data-testid="stSidebar"] [data-baseweb="select"],
        section[data-testid="stSidebar"] [data-baseweb="select"] *,
        section[data-testid="stSidebar"] [data-baseweb="tag"],
        section[data-testid="stSidebar"] [data-baseweb="tag"] * {
            color: rgba(244, 247, 250, 0.96) !important;
            fill: rgba(244, 247, 250, 0.96) !important;
        }
        section[data-testid="stSidebar"] .stCaption,
        section[data-testid="stSidebar"] div[data-testid="stCaptionContainer"],
        section[data-testid="stSidebar"] div[data-testid="stCaptionContainer"] * {
            color: rgba(244, 247, 250, 0.68) !important;
        }
        section[data-testid="stSidebar"] details {
            border: none !important;
            background: transparent !important;
            box-shadow: none !important;
            margin-top: 0.04rem;
            margin-bottom: 0.25rem;
        }
        section[data-testid="stSidebar"] details summary {
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(145,209,255,0.13);
            border-radius: 13px;
            padding: 0.34rem 0.58rem;
        }
        .sidebar-section-title {
            margin: 0.08rem 0 0.24rem 0;
            font-size: 0.81rem;
            font-weight: 760;
            color: #f4f7fa;
            letter-spacing: 0.03em;
        }
        .sidebar-divider {
            height: 1px;
            background: rgba(145,209,255,0.15);
            margin: 0.52rem 0 0.5rem 0;
        }
        section[data-testid="stSidebar"] img {
            border-radius: 18px;
            border: 1px solid rgba(145,209,255,0.16);
            box-shadow: 0 18px 35px rgba(0,0,0,0.22);
            margin-top: 0.45rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
            background: rgba(255,255,255,0.62);
            padding: 0.35rem;
            border-radius: 999px;
            border: 1px solid rgba(8,31,53,0.08);
            box-shadow: 0 12px 30px rgba(23, 61, 113, 0.08);
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            border-radius: 999px;
            color: #53667b;
            padding: 0.45rem 1rem;
            font-weight: 750;
        }
        .stTabs [aria-selected="true"] {
            background: var(--navy) !important;
            color: #ffffff !important;
            box-shadow: 0 10px 22px rgba(6,25,43,0.20);
        }
        div[data-testid="stPlotlyChart"],
        div[data-testid="stDataFrame"] {
            background: rgba(255, 255, 255, 0.78);
            border: 1px solid rgba(79, 128, 190, 0.12);
            border-radius: 22px;
            box-shadow: 0 14px 34px rgba(23, 61, 113, 0.08);
            padding: 0.6rem;
            backdrop-filter: blur(18px);
        }
        div[data-testid="stDataFrame"] {
            padding: 0.35rem;
        }
        div[data-testid="stDataFrame"] * {
            color-scheme: light;
        }
        div[data-testid="stDataFrame"] [role="grid"],
        div[data-testid="stDataFrame"] canvas,
        div[data-testid="stDataFrame"] iframe {
            background: #ffffff !important;
            border-radius: 16px;
        }
        [data-testid="stAppViewContainer"] .main div[data-testid="stSelectbox"] [data-baseweb="select"],
        [data-testid="stAppViewContainer"] .main div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
        [data-testid="stAppViewContainer"] .main div[data-testid="stSelectbox"] [data-baseweb="select"] > div > div {
            background: #f7fbff !important;
            background-color: #f7fbff !important;
            border-color: rgba(79,128,190,0.18) !important;
            border-radius: 14px !important;
            color: var(--ink) !important;
            box-shadow: 0 10px 24px rgba(23,61,113,0.07);
        }
        [data-testid="stAppViewContainer"] .main div[data-testid="stSelectbox"] [data-baseweb="select"] input,
        [data-testid="stAppViewContainer"] .main div[data-testid="stSelectbox"] [data-baseweb="select"] svg,
        [data-testid="stAppViewContainer"] .main div[data-testid="stSelectbox"] [data-baseweb="select"] span {
            color: var(--ink) !important;
            fill: var(--ink) !important;
        }
        [data-testid="stAppViewContainer"] .main div[data-testid="stSelectbox"] label,
        [data-testid="stAppViewContainer"] .main div[data-testid="stSelectbox"] p {
            color: #22364b !important;
            font-weight: 760 !important;
        }
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] > div {
            background: #ffffff !important;
            color: var(--ink) !important;
            border: 1px solid rgba(79,128,190,0.16) !important;
            border-radius: 14px !important;
            box-shadow: 0 18px 42px rgba(23,61,113,0.16) !important;
        }
        div[data-baseweb="popover"] li,
        div[data-baseweb="popover"] div[role="option"] {
            background: #ffffff !important;
            color: var(--ink) !important;
        }
        div[data-baseweb="popover"] li:hover,
        div[data-baseweb="popover"] div[role="option"]:hover {
            background: #eef6ff !important;
        }
        .light-table-wrap {
            overflow: auto;
            max-height: 440px;
            background: rgba(255,255,255,0.86);
            border: 1px solid rgba(79,128,190,0.14);
            border-radius: 20px;
            box-shadow: 0 14px 34px rgba(23,61,113,0.08);
            padding: 0.55rem;
        }
        .light-table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
            color: var(--ink);
            font-size: 0.86rem;
        }
        .light-table th {
            position: sticky;
            top: 0;
            z-index: 1;
            background: #edf6ff;
            color: #17324d;
            font-weight: 820;
            text-align: left;
            padding: 0.62rem 0.7rem;
            border-bottom: 1px solid rgba(79,128,190,0.16);
        }
        .light-table td {
            background: #ffffff;
            color: #25384c;
            padding: 0.55rem 0.7rem;
            border-bottom: 1px solid rgba(79,128,190,0.09);
            white-space: nowrap;
        }
        .light-table tr:nth-child(even) td {
            background: #f7fbff;
        }
        .table-note {
            margin: 0.45rem 0 0.2rem 0;
            color: #5f7184;
            font-size: 0.82rem;
        }
        h2, h3 {
            color: var(--ink);
            letter-spacing: -0.025em;
        }
        .stMarkdown p {
            color: #405267;
        }
        [data-testid="stAppViewContainer"] .main .stExpander,
        [data-testid="stAppViewContainer"] .main div[data-testid="stExpander"] {
            background: rgba(255,255,255,0.80) !important;
            border: 1px solid rgba(79,128,190,0.14) !important;
            border-radius: 18px !important;
            overflow: hidden;
            box-shadow: 0 12px 28px rgba(23,61,113,0.08);
        }
        [data-testid="stAppViewContainer"] .main div[data-testid="stExpander"] details,
        [data-testid="stAppViewContainer"] .main div[data-testid="stExpander"] details[open],
        [data-testid="stAppViewContainer"] .main div[data-testid="stExpander"] summary {
            background: rgba(255,255,255,0.86) !important;
            color: var(--ink) !important;
        }
        [data-testid="stAppViewContainer"] .main div[data-testid="stExpander"] code,
        .stMarkdown code {
            background: rgba(6,104,246,0.08) !important;
            color: #17324d !important;
            border: 1px solid rgba(6,104,246,0.10);
            border-radius: 7px;
            padding: 0.1rem 0.28rem;
        }
        .stMarkdown, .stText, .stCaption, p, li, label, h2, h3 {
            color: inherit;
        }
        @media (max-width: 900px) {
            .hero {
                min-height: 530px;
                padding: 1.75rem;
            }
            .hero-visual {
                inset: 11rem -4rem -3rem -12%;
                opacity: 0.9;
            }
        }
        </style>
        """
        ),
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


ICON_SVGS = {
    "clipboard": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M9 4h6l1 2h3v14H5V6h3l1-2Z"></path>
            <path d="M9 10h6"></path><path d="M9 14h4"></path>
        </svg>
    """,
    "plane": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M3 12 21 4l-5 16-4-7-9-1Z"></path>
            <path d="m12 13 9-9"></path>
        </svg>
    """,
    "rotor": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <circle cx="12" cy="12" r="2.3"></circle>
            <path d="M12 9V3"></path><path d="M12 21v-6"></path>
            <path d="M15 12h6"></path><path d="M3 12h6"></path>
            <path d="M14 10l4-4"></path><path d="M10 14l-4 4"></path>
        </svg>
    """,
    "range": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M4 18 9 7l4 7 3-4 4 8H4Z"></path>
            <path d="M15 7h4v4"></path>
        </svg>
    """,
    "hub": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <circle cx="12" cy="12" r="2.5"></circle>
            <circle cx="5" cy="7" r="1.8"></circle>
            <circle cx="19" cy="7" r="1.8"></circle>
            <circle cx="5" cy="17" r="1.8"></circle>
            <circle cx="19" cy="17" r="1.8"></circle>
            <path d="M10 11 6.5 8.2M14 11l3.5-2.8M10 13l-3.5 2.8M14 13l3.5 2.8"></path>
        </svg>
    """,
    "battery": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M7 7h9a2 2 0 0 1 2 2v6a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2Z"></path>
            <path d="M18 10h2v4h-2"></path>
            <path d="M10 10v4"></path><path d="M8 12h4"></path>
        </svg>
    """,
    "wing": """
        <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M3 14c5-5 11-8 18-8-4 5-8 9-15 11l-3-3Z"></path>
            <path d="M9 15 7 20"></path>
        </svg>
    """,
}


def render_kpi_card(container, label_text: str, value_text: str, icon_name: str, variant: str = "") -> None:
    class_name = f"kpi-card {variant}".strip()
    container.markdown(
        f"""
        <div class="{class_name}">
            <div class="kpi-top">
                <div class="kpi-icon">{ICON_SVGS[icon_name]}</div>
                <div>
                    <div class="kpi-label">{label_text}</div>
                    <div class="kpi-value">{value_text}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_light_table(frame: pd.DataFrame, max_rows: int = 250) -> None:
    shown = frame.head(max_rows).copy()
    html = shown.to_html(classes="light-table", index=False, escape=True, border=0)
    st.markdown(f'<div class="light-table-wrap">{html}</div>', unsafe_allow_html=True)
    if len(frame) > max_rows:
        st.markdown(
            f'<div class="table-note">Showing first {max_rows:,} of {len(frame):,} rows.</div>',
            unsafe_allow_html=True,
        )


def render_header(frame: pd.DataFrame, data_dir: str) -> None:
    hero_image_html = ""
    if HERO_AIRCRAFT_IMAGE.exists():
        hero_image_html = (
            f'<div class="hero-visual"><img src="{image_data_uri(HERO_AIRCRAFT_IMAGE)}" '
            'alt="Futuristic aircraft visualization"></div>'
        )
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-content">
                <div class="eyebrow">
                    <span class="eyebrow-dots">•••</span>
                    <span>AircraftVerse Dashboard</span>
                </div>
                <h1>Exploring and Visualizing <span>27’714 Aircraft Designs</span></h1>
                <p>
                    Explore which aircraft designs are viable, which design families perform well,
                    and which recurring design recipes produce the strongest trade-offs.
                </p>
            </div>
            {hero_image_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="source-pill">
            <strong>Data source</strong>
            <code>{display_path(data_dir)}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total = len(frame)
    viable_share = 100 * frame["can_fly"].mean()
    hover_share = 100 * frame["can_hover"].mean()
    best_distance = frame["max_distance_m"].max()

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    cols = st.columns(4)
    render_kpi_card(cols[0], "Designs in current view", f"{total:,}", "clipboard")
    render_kpi_card(cols[1], "Forward-flight capable", f"{viable_share:.1f}%", "plane", "teal")
    render_kpi_card(cols[2], "Hover-capable", f"{hover_share:.1f}%", "rotor")
    render_kpi_card(cols[3], "Best max distance", f"{best_distance:,.0f} m", "range", "purple")

    st.markdown("<div style='height: 1.15rem;'></div>", unsafe_allow_html=True)
    findings = key_findings(frame)
    fact_cols = st.columns(3)
    variants = ["", "teal", "purple"]
    icons = ["hub", "battery", "wing"]
    labels = ["Hub insight", "Fuselage insight", "Design trend"]
    for idx, finding in enumerate(findings):
        fact_cols[idx].markdown(
            f"""
            <div class="finding {variants[idx]}">
                <div class="finding-badge">{ICON_SVGS[icons[idx]]}</div>
                <div class="finding-text">{finding}</div>
                <div class="finding-label">{labels[idx]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
        st.plotly_chart(polish_figure(fig), width="stretch")

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
        st.plotly_chart(polish_figure(fig), width="stretch")

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
        dist_cols[idx].plotly_chart(polish_figure(fig), width="stretch")

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
        st.plotly_chart(polish_figure(fig), width="stretch")

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
        st.plotly_chart(polish_figure(fig), width="stretch")

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
    st.plotly_chart(polish_figure(fig), width="stretch")

    render_light_table(summary.round(2), max_rows=80)


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
        st.plotly_chart(polish_figure(fig), width="stretch")

    with right:
        render_light_table(
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
            max_rows=20,
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
    st.plotly_chart(polish_figure(fig), width="stretch")


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
    st.plotly_chart(polish_figure(fig), width="stretch")

    render_light_table(
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
        max_rows=250,
    )


def main() -> None:
    st.set_page_config(
        page_title="Exploring and Visualizing 27’714 Aircraft Designs",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    configure_plotly_theme()
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
