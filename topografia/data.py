from __future__ import annotations

import numpy as np
import pandas as pd

def feature_name(feature: dict) -> str:
    properties = feature.get("properties", {})
    return str(properties.get("lote") or properties.get("name") or properties.get("nombre") or "Lote")


def polygons(geometry: dict) -> list[list[list[list[float]]]]:
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"]]
    if geometry["type"] == "MultiPolygon":
        return geometry["coordinates"]
    raise ValueError(f"Geometría no soportada: {geometry['type']}")


def geometry_area_ha(geometry: dict) -> float:
    coords = [point for polygon in polygons(geometry) for ring in polygon for point in ring]
    mean_lat = float(np.mean([point[1] for point in coords]))
    metres_x = 111_320 * np.cos(np.radians(mean_lat))
    metres_y = 110_574
    total = 0.0
    for polygon in polygons(geometry):
        for index, ring in enumerate(polygon):
            projected = np.asarray([(x * metres_x, y * metres_y) for x, y in ring])
            signed = 0.5 * np.sum(
                projected[:-1, 0] * projected[1:, 1]
                - projected[1:, 0] * projected[:-1, 1]
            )
            total += abs(signed) if index == 0 else -abs(signed)
    return total / 10_000


def display_sample(frame: pd.DataFrame, maximum: int = 10_000) -> pd.DataFrame:
    valid = frame.loc[frame["dentro_poligono"]].copy()
    if len(valid) > maximum:
        valid = valid.sample(maximum, random_state=26)
    low, high = valid["elevacion_m"].quantile([0.01, 0.99])
    scale = np.clip((valid["elevacion_m"] - low) / max(high - low, 0.001), 0, 1)
    valid["color_r"] = (48 + 222 * scale).astype(int)
    valid["color_g"] = (96 + 110 * (1 - np.abs(scale - 0.5) * 2)).astype(int)
    valid["color_b"] = (220 - 170 * scale).astype(int)
    return valid
