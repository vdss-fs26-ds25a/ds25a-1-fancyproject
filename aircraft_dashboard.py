from __future__ import annotations

import os
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


OUTPUT_RENAMES = {
    "Mass": "mass_kg",
    "Max_Distance": "max_distance_m",
    "Hover_Time": "hover_time_s",
    "Max_Speed": "max_speed_mps",
    "Interferences": "interferences",
    "Distance_MxSpd": "distance_at_max_speed_m",
    "Power_MFD": "power_at_max_distance_w",
    "Power_MxSpd": "power_at_max_speed_w",
    "Speed_MFD": "speed_at_max_distance_mps",
}

PERFORMANCE_COLUMNS = [
    "mass_kg",
    "max_distance_m",
    "hover_time_s",
    "max_speed_mps",
    "interferences",
]

OUTPUT_ONLY_COLUMNS = {
    "distance_at_max_speed_m",
    "power_at_max_distance_w",
    "power_at_max_speed_w",
    "speed_at_max_distance_mps",
    "distance_per_kg",
    "hover_per_kg",
    "mission_score",
    "can_hover",
    "can_fly",
}


def locate_data_dir(root: Path) -> Path:
    candidates = [
        root / "data_raw" / "aircraftverse_slim",
        root / "data_raw" / "data_slim",
        root / "AircraftVerse-main" / "data_full",
        root / "AircraftVerse-main" / "data",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No AircraftVerse data directory found.")


def parse_capacity_mah(battery_type: str | None) -> float | None:
    if not battery_type:
        return None
    match = re.search(r"(\d+)mAh", battery_type)
    return float(match.group(1)) if match else None


def parse_cells(battery_type: str | None) -> float | None:
    if not battery_type:
        return None
    match = re.search(r"(\d+)S", battery_type)
    return float(match.group(1)) if match else None


def parse_prop_measurement(prop_type: str | None, index: int) -> float | None:
    if not prop_type:
        return None
    match = re.search(r"(\d+(?:_\d+)?)x(\d+(?:_\d+)?)", prop_type)
    if not match:
        return None
    raw = match.group(index).replace("_", ".")
    return float(raw)


def _walk_tree(obj: Any, stats: dict[str, Any]) -> None:
    if isinstance(obj, dict):
        node_type = obj.get("node_type")
        if node_type:
            stats["node_types"][node_type] += 1
        battery_type = obj.get("batteryType")
        if battery_type:
            stats["battery_types"][battery_type] += 1
        motor_type = obj.get("motorType")
        if motor_type:
            stats["motor_types"][motor_type] += 1
        prop_type = obj.get("propType")
        if prop_type:
            stats["prop_types"][prop_type] += 1
        if "armLength" in obj and isinstance(obj["armLength"], (int, float)):
            stats["arm_lengths"].append(float(obj["armLength"]))
        if "span" in obj and isinstance(obj["span"], (int, float)):
            stats["wing_spans"].append(float(obj["span"]))
        if "chordInner" in obj and isinstance(obj["chordInner"], (int, float)):
            stats["wing_chord_inner"].append(float(obj["chordInner"]))
        if "chordOuter" in obj and isinstance(obj["chordOuter"], (int, float)):
            stats["wing_chord_outer"].append(float(obj["chordOuter"]))
        for value in obj.values():
            _walk_tree(value, stats)
    elif isinstance(obj, list):
        for value in obj:
            _walk_tree(value, stats)


def _safe_stat(values: list[float], fn) -> float | None:
    return float(fn(values)) if values else None


def _build_design_record(design_dir: Path) -> dict[str, Any]:
    tree = json.loads((design_dir / "design_tree.json").read_text())
    output = json.loads((design_dir / "output.json").read_text())

    stats = {
        "node_types": Counter(),
        "battery_types": Counter(),
        "motor_types": Counter(),
        "prop_types": Counter(),
        "arm_lengths": [],
        "wing_spans": [],
        "wing_chord_inner": [],
        "wing_chord_outer": [],
    }
    _walk_tree(tree, stats)

    fuselage = tree["fuselageWithComponents"]["fuselage"]
    battery_type = next(iter(stats["battery_types"]), None)
    prop_type = next(iter(stats["prop_types"]), None)

    record = {
        "design_name": design_dir.name,
        "hub_type": tree["hub"]["node_type"],
        "fuselage_type": tree["fuselageWithComponents"]["node_type"],
        "battery_type": battery_type,
        "battery_capacity_mah": parse_capacity_mah(battery_type),
        "battery_cells_s": parse_cells(battery_type),
        "battery_count": sum(stats["battery_types"].values()),
        "motor_count": sum(stats["motor_types"].values()),
        "prop_count": sum(stats["prop_types"].values()),
        "wing_count": len(stats["wing_spans"]),
        "component_variety": len(stats["node_types"]),
        "primary_motor_type": next(iter(stats["motor_types"]), None),
        "primary_prop_type": prop_type,
        "prop_diameter_in": parse_prop_measurement(prop_type, 1),
        "prop_pitch_in": parse_prop_measurement(prop_type, 2),
        "arm_count": len(stats["arm_lengths"]),
        "arm_length_mean_mm": _safe_stat(stats["arm_lengths"], np.mean),
        "arm_length_max_mm": _safe_stat(stats["arm_lengths"], np.max),
        "arm_length_min_mm": _safe_stat(stats["arm_lengths"], np.min),
        "wing_span_total_mm": float(sum(stats["wing_spans"])),
        "wing_span_mean_mm": _safe_stat(stats["wing_spans"], np.mean),
        "wing_chord_inner_mean_mm": _safe_stat(stats["wing_chord_inner"], np.mean),
        "wing_chord_outer_mean_mm": _safe_stat(stats["wing_chord_outer"], np.mean),
        "wing_area_proxy": float(
            sum(
                span * ((inner + outer) / 2.0)
                for span, inner, outer in zip(
                    stats["wing_spans"],
                    stats["wing_chord_inner"],
                    stats["wing_chord_outer"],
                )
            )
        ),
        "fuselage_length_mm": float(fuselage["length"]),
        "fuselage_horz_diameter_mm": float(fuselage["horzDiameter"]),
        "fuselage_vert_diameter_mm": float(fuselage["vertDiameter"]),
        "fuselage_floor_height_mm": float(fuselage["floorHeight"]),
    }

    for old_key, new_key in OUTPUT_RENAMES.items():
        if old_key in output:
            record[new_key] = output[old_key]

    record["can_hover"] = bool(record.get("hover_time_s", 0) > 0)
    record["can_fly"] = bool(record.get("max_distance_m", 0) > 0 or record.get("max_speed_mps", 0) > 0)
    distance = record.get("max_distance_m", 0) or 0
    hover = record.get("hover_time_s", 0) or 0
    mass = record.get("mass_kg", 0) or 0
    record["distance_per_kg"] = distance / mass if mass else None
    record["hover_per_kg"] = hover / mass if mass else None
    record["mission_score"] = (
        distance / 1000.0
        + hover / 60.0
        + record.get("max_speed_mps", 0) / 10.0
        - mass / 5.0
        - record.get("interferences", 0) * 0.25
    )
    return record


def load_aircraft_data(root: Path) -> tuple[pd.DataFrame, Path]:
    data_dir = locate_data_dir(root)
    design_dirs = sorted(
        path for path in data_dir.iterdir() if path.is_dir() and path.name.startswith("design_")
    )
    rows = [_build_design_record(path) for path in design_dirs]
    frame = pd.DataFrame(rows).sort_values("mission_score", ascending=False).reset_index(drop=True)
    return frame, data_dir


def numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {"mission_score"}
    columns = [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]
    return columns


def correlation_ranking(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    numeric = frame[numeric_feature_columns(frame)]
    min_non_null = max(4, math.ceil(len(frame) * 0.3))
    excluded = set(PERFORMANCE_COLUMNS) | OUTPUT_ONLY_COLUMNS | {target}
    eligible = [
        column
        for column in numeric.columns
        if column not in excluded and numeric[column].notna().sum() >= min_non_null
    ]
    correlations = (
        numeric[eligible]
        .assign(**{target: frame[target]})
        .corr(numeric_only=True)[target]
        .dropna()
        .drop(target, errors="ignore")
    )
    ranking = (
        correlations.abs()
        .sort_values(ascending=False)
        .rename("absolute_correlation")
        .to_frame()
        .assign(correlation=correlations)
        .reset_index(names="feature")
    )
    return ranking


def pareto_front(frame: pd.DataFrame, x_col: str, y_col: str, x_goal: str, y_goal: str) -> pd.Series:
    data = frame[[x_col, y_col]].fillna(0).to_numpy(dtype=float)
    signed = data.copy()
    if x_goal == "min":
        signed[:, 0] *= -1
    if y_goal == "min":
        signed[:, 1] *= -1

    efficient = np.ones(len(signed), dtype=bool)
    for i, point in enumerate(signed):
        if not efficient[i]:
            continue
        dominates = np.all(signed >= point, axis=1) & np.any(signed > point, axis=1)
        efficient[dominates] = False
    return pd.Series(efficient, index=frame.index)


def cluster_projection(frame: pd.DataFrame, features: list[str], cluster_count: int) -> pd.DataFrame:
    usable = frame[features].copy()
    usable = usable.dropna(axis=1, how="all").fillna(usable.median(numeric_only=True))
    if usable.shape[1] < 2:
        raise ValueError("Not enough numeric features available for clustering.")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(usable)

    cluster_count = max(2, min(cluster_count, len(usable)))
    kmeans = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled)

    pca = PCA(n_components=2, random_state=42)
    projection = pca.fit_transform(scaled)

    result = frame.copy()
    result["cluster"] = labels.astype(str)
    result["pca_1"] = projection[:, 0]
    result["pca_2"] = projection[:, 1]
    result["cluster_size"] = result.groupby("cluster")["design_name"].transform("count")
    result.attrs["explained_variance_ratio"] = pca.explained_variance_ratio_
    return result
