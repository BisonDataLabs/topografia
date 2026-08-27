from __future__ import annotations

import math

import geopandas as gpd
import numpy as np
import pandas as pd


COMPASS = ["N", "NE", "E", "SE", "S", "SO", "O", "NO"]


def _direction_label(bearing: float) -> str:
    return COMPASS[int((bearing + 22.5) // 45) % 8]


def monitor_dashboard(points: gpd.GeoDataFrame, boundary) -> dict:
    selected = points.loc[points.geometry.within(boundary)].copy()
    if selected.empty:
        raise ValueError("No hay puntos dentro del lote para resumir.")
    elevation = selected["elev_m_raw"].to_numpy(dtype=float)
    center_lat = boundary.centroid.y
    center_lon = boundary.centroid.x
    x_m = (selected["lon"].to_numpy() - center_lon) * 111_320 * math.cos(math.radians(center_lat))
    y_m = (selected["lat"].to_numpy() - center_lat) * 110_574
    design = np.column_stack([x_m, y_m, np.ones(len(selected))])
    east_gradient, north_gradient, _ = np.linalg.lstsq(design, elevation, rcond=None)[0]
    plane_slope_percent = 100 * math.hypot(east_gradient, north_gradient)
    down_east, down_north = -east_gradient, -north_gradient
    bearing = (math.degrees(math.atan2(down_east, down_north)) + 360) % 360
    norm = max(math.hypot(down_east, down_north), 1e-12)
    distance = (x_m * down_east + y_m * down_north) / norm

    profile_source = pd.DataFrame({"distance_m": distance, "elevation_m": elevation})
    profile_source["segment"] = pd.cut(profile_source["distance_m"], bins=20, duplicates="drop")
    profile = (
        profile_source.groupby("segment", observed=True)
        .agg(distancia_m=("distance_m", "median"), cota_mediana_m=("elevation_m", "median"), puntos=("elevation_m", "size"))
        .reset_index(drop=True)
        .sort_values("distancia_m")
    )
    profile["distancia_m"] -= profile["distancia_m"].min()

    counts, edges = np.histogram(elevation, bins=24)
    histogram = pd.DataFrame(
        {"cota_m": (edges[:-1] + edges[1:]) / 2, "puntos": counts}
    )
    low_mask = elevation <= np.quantile(elevation, 0.03)
    high_mask = elevation >= np.quantile(elevation, 0.97)
    markers = pd.DataFrame(
        [
            {
                "tipo": "Bajo relativo",
                "lon": float(selected.loc[low_mask, "lon"].median()),
                "lat": float(selected.loc[low_mask, "lat"].median()),
                "elevacion_m": float(np.median(elevation[low_mask])),
                "color": [36, 99, 235, 230],
            },
            {
                "tipo": "Alto relativo",
                "lon": float(selected.loc[high_mask, "lon"].median()),
                "lat": float(selected.loc[high_mask, "lat"].median()),
                "elevacion_m": float(np.median(elevation[high_mask])),
                "color": [234, 88, 12, 230],
            },
        ]
    )
    return {
        "rows": int(len(selected)),
        "elevation_min": float(np.min(elevation)),
        "elevation_p01": float(np.quantile(elevation, 0.01)),
        "elevation_median": float(np.median(elevation)),
        "elevation_p99": float(np.quantile(elevation, 0.99)),
        "elevation_max": float(np.max(elevation)),
        "elevation_iqr": float(np.quantile(elevation, 0.75) - np.quantile(elevation, 0.25)),
        "relief_p01_p99": float(np.quantile(elevation, 0.99) - np.quantile(elevation, 0.01)),
        "plane_slope_percent": float(plane_slope_percent),
        "downslope_bearing": float(bearing),
        "downslope_direction": _direction_label(bearing),
        "histogram": histogram,
        "profile": profile,
        "markers": markers,
    }


def quality_grade(diagnostics: dict, total_uploaded_rows: int, area_ha: float) -> tuple[str, str]:
    valid_share = 100 * diagnostics["rows_numeric"] / max(total_uploaded_rows, 1)
    density = diagnostics["rows_inside"] / max(area_ha, 1e-9)
    if diagnostics["suspicious_coordinate_copy"]:
        return "No utilizable", "La elevación parece duplicar una coordenada."
    if diagnostics["inside_percent"] >= 95 and valid_share >= 95 and density >= 20 and diagnostics["unique_elevations"] >= 20:
        return "Buena para diagnóstico", "Cobertura, densidad y variación suficientes para interpretar la macro-topografía."
    if diagnostics["inside_percent"] >= 80 and valid_share >= 80 and diagnostics["unique_elevations"] >= 10:
        return "Utilizable con cautela", "Permite reconocer tendencias, pero conviene revisar cobertura y precisión vertical."
    return "Insuficiente", "La cobertura o la variación vertical no alcanzan para sostener conclusiones topográficas."
