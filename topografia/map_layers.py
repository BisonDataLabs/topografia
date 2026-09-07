from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib
import numpy as np
import pydeck as pdk
import rasterio
from matplotlib import colormaps
from PIL import Image
from pydeck.types import String
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds

matplotlib.use("Agg")


BASEMAPS = {
    "Topográfico con nombres": [
        ("https://a.tile.opentopomap.org/{z}/{x}/{y}.png", "OpenTopoMap"),
    ],
    "Satelital": [
        (
            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            "Esri World Imagery",
        ),
    ],
    "Híbrido": [
        (
            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            "Esri World Imagery",
        ),
        (
            "https://services.arcgisonline.com/ArcGIS/rest/services/Reference/"
            "World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
            "Esri Reference",
        ),
    ],
    "Calles": [("https://tile.openstreetmap.org/{z}/{x}/{y}.png", "OpenStreetMap")],
}

PALETTE_GRADIENTS = {
    "terrain": "#333399, #00a6ca, #7fd34e, #d9c27a, #ffffff",
    "viridis": "#440154, #3b528b, #21918c, #5ec962, #fde725",
    "Blues": "#f7fbff, #c6dbef, #6baed6, #2171b5, #08306b",
    "YlOrRd": "#ffffcc, #fed976, #fd8d3c, #e31a1c, #800026",
    "RdBu_r": "#2166ac, #67a9cf, #f7f7f7, #ef8a62, #b2182b",
    "magma": "#000004, #3b0f70, #8c2981, #de4968, #fcfdbf",
    "gray": "#000000, #ffffff",
    "twilight": "#e2d9e2, #6276ba, #2f1436, #9c3b4c, #e2d9e2",
}


def basemap_layers(name: str, opacity: float) -> list[pdk.Layer]:
    return [
        pdk.Layer(
            "TileLayer",
            id=f"base-{index}-{name}",
            data=url,
            min_zoom=0,
            max_zoom=19,
            tile_size=256,
            opacity=opacity,
            pickable=False,
        )
        for index, (url, _) in enumerate(BASEMAPS[name])
    ]


def _rgba(array: np.ndarray, valid: np.ndarray, layer: dict[str, Any]) -> np.ndarray:
    legend = layer.get("legend", [])
    if legend:
        pixels = np.zeros((*array.shape, 4), dtype="uint8")
        for item in legend:
            try:
                selected = valid & np.isclose(array, float(item["value"]))
            except (TypeError, ValueError):
                continue
            pixels[selected] = _hex_color(item["color"], 235)
        return pixels
    values = np.log1p(np.clip(array, 0, None)) if layer.get("scale") == "log" else array
    stats = layer.get("stats", {})
    low = stats.get("p02")
    high = stats.get("p98")
    if layer.get("scale") == "log":
        low = math.log1p(max(float(low or 0), 0))
        high = math.log1p(max(float(high or 0), 0))
    if low is None or high is None or float(high) <= float(low):
        low, high = np.nanpercentile(values[valid], [2, 98])
    scaled = np.clip((values - float(low)) / max(float(high) - float(low), 1e-9), 0, 1)
    pixels = (colormaps.get_cmap(layer.get("palette", "viridis"))(scaled) * 255).astype("uint8")
    pixels[..., 3] = np.where(valid, 235, 0).astype("uint8")
    return pixels


def raster_layer(
    path: Path, layer: dict[str, Any], opacity: float
) -> tuple[pdk.Layer, tuple[float, float, float, float]]:
    with rasterio.open(path) as source:
        scale = min(1.0, math.sqrt(1_200_000 / max(source.width * source.height, 1)))
        width = max(2, round(source.width * scale))
        height = max(2, round(source.height * scale))
        array = source.read(1, out_shape=(height, width), resampling=Resampling.nearest).astype("float64")
        valid = np.isfinite(array)
        if source.nodata is not None:
            valid &= array != source.nodata
        pixels = _rgba(array, valid, layer)
        output = io.BytesIO()
        Image.fromarray(pixels, "RGBA").save(output, "PNG")
        affine = source.transform * source.transform.scale(source.width / width, source.height / height)
        west, south, east, north = array_bounds(height, width, affine)
        bounds = transform_bounds(source.crs, "EPSG:4326", west, south, east, north)
    data_url = f"data:image/png;base64,{base64.b64encode(output.getvalue()).decode('ascii')}"
    return (
        pdk.Layer(
            "BitmapLayer",
            id=layer["id"],
            image=String(data_url),
            bounds=list(bounds),
            opacity=opacity,
            pickable=True,
        ),
        tuple(float(value) for value in bounds),
    )


def _hex_color(value: str, alpha: int) -> list[int]:
    clean = value.lstrip("#")
    return [int(clean[index : index + 2], 16) for index in (0, 2, 4)] + [alpha]


def vector_layer(
    path: Path, layer: dict[str, Any], opacity: float
) -> tuple[pdk.Layer, tuple[float, float, float, float]]:
    frame = gpd.read_parquet(path) if path.suffix == ".parquet" else gpd.read_file(path)
    if len(frame) > 40_000:
        frame = frame.sample(40_000, random_state=42)
    frame = frame.to_crs(4326)
    alpha = round(255 * opacity)
    style = layer.get("style", {})
    renderer = style.get("renderer", "fill")
    field = style.get("field")
    legend = {str(item["value"]): item["color"] for item in layer.get("legend", [])}
    if renderer == "graduated_points" and field in frame:
        values = frame[field].astype("float64")
        limits = style.get("range") or [values.quantile(0.02), values.quantile(0.98)]
        scaled = np.clip(
            (values - float(limits[0])) / max(float(limits[1]) - float(limits[0]), 1e-9),
            0,
            1,
        )
        colors = colormaps.get_cmap(style.get("palette", "terrain"))(scaled)
        frame["_fill"] = [[round(channel * 255) for channel in color[:3]] + [alpha] for color in colors]
    elif legend and field in frame:
        frame["_fill"] = frame[field].map(lambda value: _hex_color(legend.get(str(value), "#2878b5"), alpha))
    else:
        fill = style.get("fill", "#2878b5")
        frame["_fill"] = [_hex_color(fill, alpha)] * len(frame)
    if renderer == "categorized_lines" and legend and field in frame:
        frame["_line"] = frame[field].map(lambda value: _hex_color(legend.get(str(value), "#243746"), alpha))
        frame["_fill"] = [[0, 0, 0, 0]] * len(frame)
    else:
        line = style.get("color", style.get("stroke", "#192d3c"))
        frame["_line"] = [_hex_color(line, alpha)] * len(frame)
    data = json.loads(frame.to_json(drop_id=True))
    return (
        pdk.Layer(
            "GeoJsonLayer",
            id=layer["id"],
            data=data,
            pickable=True,
            stroked=True,
            filled=renderer not in {"line", "categorized_lines"},
            get_fill_color="properties._fill",
            get_line_color="properties._line",
            get_radius=style.get("point_radius_px", 2) * 2,
            radius_min_pixels=1,
            line_width_min_pixels=style.get("width_px", 1),
        ),
        tuple(float(value) for value in frame.total_bounds),
    )


def view_state(bounds: list[tuple[float, float, float, float]]) -> pdk.ViewState:
    west = min(item[0] for item in bounds)
    south = min(item[1] for item in bounds)
    east = max(item[2] for item in bounds)
    north = max(item[3] for item in bounds)
    center_lat = (south + north) / 2
    width_km = max((east - west) * 111.32 * math.cos(math.radians(center_lat)), 0.01)
    height_km = max((north - south) * 110.57, 0.01)
    zoom = max(3.0, min(17.0, 14.2 - math.log2(max(width_km, height_km))))
    return pdk.ViewState(latitude=center_lat, longitude=(west + east) / 2, zoom=zoom)
