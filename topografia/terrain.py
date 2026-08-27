from __future__ import annotations

import base64
import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "topografia-matplotlib"))
import matplotlib
import numpy as np
from matplotlib import colormaps
from PIL import Image
from rasterio.io import MemoryFile
from rasterio.fill import fillnodata
from rasterio.features import geometry_mask
from rasterio.mask import mask
from rasterio.transform import array_bounds
from rasterio.warp import transform, transform_bounds, transform_geom
from shapely import contains_xy
from shapely.geometry import mapping

matplotlib.use("Agg")


@dataclass(frozen=True)
class TerrainOverlay:
    image_data_url: str
    bounds_wgs84: tuple[float, float, float, float]
    elevation_p02_m: float
    elevation_p98_m: float
    relief_m: float
    mean_slope_percent: float
    area_over_2_percent: float
    area_over_5_percent: float
    elevation_histogram: tuple[tuple[float, int], ...]
    slope_classes: tuple[tuple[str, float], ...]
    markers: tuple[dict, ...]


@dataclass(frozen=True)
class MonitorOverlay:
    image_data_url: str
    bounds_wgs84: tuple[float, float, float, float]
    elevation_p01_m: float
    elevation_p99_m: float


def _rgba(values: np.ndarray, valid: np.ndarray, cmap_name: str, low: float, high: float) -> np.ndarray:
    scale = np.clip((values - low) / max(high - low, 1e-9), 0, 1)
    rgba = (colormaps[cmap_name](scale) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(valid, 205, 0).astype(np.uint8)
    return rgba


def analyze_monitor(points, boundary_wgs84, size: int = 240) -> MonitorOverlay:
    inside = points.geometry.within(boundary_wgs84)
    selected = points.loc[inside]
    if selected.empty:
        raise ValueError("No hay puntos del monitor dentro del lote.")
    west, south, east, north = boundary_wgs84.bounds
    width = size
    height = max(80, round(size * (north - south) / max(east - west, 1e-9)))
    height = min(height, 420)
    x_index = np.clip(((selected["lon"].to_numpy() - west) / (east - west) * (width - 1)).astype(int), 0, width - 1)
    y_index = np.clip(((north - selected["lat"].to_numpy()) / (north - south) * (height - 1)).astype(int), 0, height - 1)
    totals = np.zeros((height, width), dtype="float64")
    counts = np.zeros((height, width), dtype="uint32")
    np.add.at(totals, (y_index, x_index), selected["elev_m_raw"].to_numpy())
    np.add.at(counts, (y_index, x_index), 1)
    observed = counts > 0
    grid = np.divide(totals, counts, out=np.zeros_like(totals), where=observed)
    grid = fillnodata(grid.astype("float32"), mask=observed.astype("uint8"), max_search_distance=12)

    x_centers = np.linspace(west, east, width)
    y_centers = np.linspace(north, south, height)
    mesh_x, mesh_y = np.meshgrid(x_centers, y_centers)
    inside_grid = contains_xy(boundary_wgs84, mesh_x, mesh_y)
    valid = inside_grid & np.isfinite(grid) & (grid != 0)
    low, high = np.nanpercentile(selected["elev_m_raw"], [1, 99])
    pixels = _rgba(grid, valid, "terrain", float(low), float(high))
    pixels[..., 3] = np.where(valid, 178, 0).astype(np.uint8)
    image = Image.fromarray(pixels, "RGBA")
    output = io.BytesIO()
    image.save(output, "PNG")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return MonitorOverlay(
        image_data_url=f"data:image/png;base64,{encoded}",
        bounds_wgs84=(float(west), float(south), float(east), float(north)),
        elevation_p01_m=float(low),
        elevation_p99_m=float(high),
    )


def analyze_dem(content: bytes, boundary_wgs84, mode: str = "Elevación", display_geometry_wgs84=None) -> TerrainOverlay:
    with MemoryFile(content) as memory:
        with memory.open() as source:
            if source.crs is None:
                raise ValueError("El GeoTIFF no declara un sistema de coordenadas.")
            display_geometry_wgs84 = display_geometry_wgs84 or boundary_wgs84
            display_geometry = transform_geom("EPSG:4326", source.crs, mapping(display_geometry_wgs84))
            analysis_geometry = transform_geom("EPSG:4326", source.crs, mapping(boundary_wgs84))
            raster, affine = mask(source, [display_geometry], crop=True, filled=False)
            elevation = raster[0].astype("float64")
            valid = ~np.ma.getmaskarray(elevation) & np.isfinite(elevation.filled(np.nan))
            if not valid.any():
                raise ValueError("El DEM no contiene celdas válidas dentro del lote.")
            values = elevation.filled(np.nan)
            inside = geometry_mask([analysis_geometry], values.shape, affine, invert=True, all_touched=True) & valid
            if not inside.any():
                raise ValueError("El DEM no contiene celdas válidas dentro de la selección.")
            valid_elevation = values[inside]
            low, high = np.nanpercentile(valid_elevation, [2, 98])

            center_lat = boundary_wgs84.centroid.y
            x_resolution_m = abs(affine.a) * (111_320 * np.cos(np.radians(center_lat)))
            y_resolution_m = abs(affine.e) * 110_574
            if not source.crs.is_geographic:
                x_resolution_m = abs(affine.a)
                y_resolution_m = abs(affine.e)
            extended = fillnodata(
                np.where(valid, values, 0).astype("float32"),
                mask=valid.astype("uint8"),
                max_search_distance=max(values.shape),
            ).astype("float64")
            padded = np.pad(extended, 1, mode="edge")
            smooth = sum(
                padded[row : row + extended.shape[0], col : col + extended.shape[1]]
                for row in range(3) for col in range(3)
            ) / 9
            dz_dy, dz_dx = np.gradient(smooth, max(y_resolution_m, 0.01), max(x_resolution_m, 0.01))
            slope_percent = np.hypot(dz_dx, dz_dy) * 100

            if mode == "Pendiente":
                pixels = np.zeros((*slope_percent.shape, 4), dtype=np.uint8)
                slope_palette = [
                    (0, 1, [198, 226, 187, 225]),
                    (1, 2, [255, 237, 160, 225]),
                    (2, 5, [254, 178, 76, 225]),
                    (5, 10, [240, 59, 32, 225]),
                    (10, np.inf, [117, 107, 177, 225]),
                ]
                for lower, upper, color in slope_palette:
                    pixels[valid & (slope_percent >= lower) & (slope_percent < upper)] = color
            elif mode == "Sombreado":
                slope_rad = np.arctan(np.hypot(dz_dx, dz_dy))
                aspect = np.arctan2(-dz_dx, dz_dy)
                azimuth, altitude = np.radians(315), np.radians(45)
                shade = np.sin(altitude) * np.cos(slope_rad) + np.cos(altitude) * np.sin(slope_rad) * np.cos(azimuth - aspect)
                shade_low, shade_high = np.nanpercentile(shade[valid], [2, 98])
                pixels = _rgba(shade, valid, "gray", float(shade_low), float(shade_high))
                pixels[..., 3] = np.where(valid, 230, 0).astype(np.uint8)
            else:
                pixels = _rgba(values, valid, "terrain", float(low), float(high))

            marker_rows, marker_cols = np.indices(values.shape)
            marker_items = []
            for label, percentile, comparison, color in [
                ("Bajo relativo DEM", 2, np.less_equal, [30, 100, 210, 245]),
                ("Alto relativo DEM", 98, np.greater_equal, [230, 90, 25, 245]),
            ]:
                limit = float(np.nanpercentile(valid_elevation, percentile))
                candidate = inside & comparison(values, limit)
                candidate_rows = marker_rows[candidate]
                candidate_cols = marker_cols[candidate]
                center_row = float(np.median(candidate_rows))
                center_col = float(np.median(candidate_cols))
                representative = int(np.argmin((candidate_rows - center_row) ** 2 + (candidate_cols - center_col) ** 2))
                row = int(candidate_rows[representative])
                col = int(candidate_cols[representative])
                point_x, point_y = affine * (col + 0.5, row + 0.5)
                point_lon, point_lat = transform(source.crs, "EPSG:4326", [point_x], [point_y])
                marker_items.append(
                    {
                        "tipo": label,
                        "detalle": f"{values[row, col]:.2f} m",
                        "lon": float(point_lon[0]),
                        "lat": float(point_lat[0]),
                        "elevacion_m": float(values[row, col]),
                        "color": color,
                    }
                )

            image = Image.fromarray(pixels, "RGBA")
            output = io.BytesIO()
            image.save(output, "PNG")
            west, south, east, north = array_bounds(values.shape[0], values.shape[1], affine)
            if source.crs.to_epsg() != 4326:
                west, south, east, north = transform_bounds(source.crs, "EPSG:4326", west, south, east, north)

    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    valid_slope = slope_percent[inside]
    elevation_counts, elevation_edges = np.histogram(valid_elevation, bins=24)
    elevation_histogram = tuple(
        (float((left + right) / 2), int(count))
        for left, right, count in zip(elevation_edges[:-1], elevation_edges[1:], elevation_counts)
    )
    slope_classes = tuple(
        (
            label,
            float(100 * np.mean((valid_slope >= lower) & (valid_slope < upper))),
        )
        for label, lower, upper in [
            ("0 a 1 % · casi plano", 0, 1),
            ("1 a 2 % · suave", 1, 2),
            ("2 a 5 % · moderado", 2, 5),
            (">5 % · sensible", 5, np.inf),
        ]
    )
    return TerrainOverlay(
        image_data_url=f"data:image/png;base64,{encoded}",
        bounds_wgs84=(float(west), float(south), float(east), float(north)),
        elevation_p02_m=float(low),
        elevation_p98_m=float(high),
        relief_m=float(high - low),
        mean_slope_percent=float(np.nanmean(valid_slope)),
        area_over_2_percent=float(100 * np.mean(valid_slope > 2)),
        area_over_5_percent=float(100 * np.mean(valid_slope > 5)),
        elevation_histogram=elevation_histogram,
        slope_classes=slope_classes,
        markers=tuple(marker_items),
    )
