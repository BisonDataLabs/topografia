from __future__ import annotations

import base64
import io
import math
import tempfile
import zipfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import requests
from PIL import Image
from matplotlib import colormaps
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.fill import fillnodata
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds, transform_geom
from shapely import make_valid
from shapely.geometry import mapping


BOUNDARY_EXTENSIONS = ["geojson", "json", "gpkg", "kml", "kmz", "zip"]
HAND_SERVICE_URL = "https://gis.asf.alaska.edu/arcgis/rest/services/GlobalHAND/GLO30_HAND/ImageServer"


@dataclass(frozen=True)
class DemResult:
    content: bytes
    dataset: str
    native_resolution_m: int


@dataclass(frozen=True)
class RasterOverlay:
    image_data_url: str
    bounds_wgs84: tuple[float, float, float, float]
    legend: tuple[tuple[str, str], ...]
    metrics: dict[str, float | str]


def read_boundaries(filename: str, payload: bytes) -> gpd.GeoDataFrame:
    suffix = Path(filename).suffix.lower()
    with tempfile.TemporaryDirectory(prefix="topografia-limites-") as temp_name:
        temp = Path(temp_name)
        source = temp / Path(filename).name
        source.write_bytes(payload)
        if suffix == ".kmz":
            with zipfile.ZipFile(source) as archive:
                archive.extractall(temp / "kmz")
            kml_files = list((temp / "kmz").rglob("*.kml"))
            if not kml_files:
                raise ValueError("El KMZ no contiene un archivo KML.")
            frame = gpd.read_file(kml_files[0])
        elif suffix == ".zip":
            frame = gpd.read_file(f"zip://{source}")
        else:
            frame = gpd.read_file(source)
    if frame.empty:
        raise ValueError("El archivo no contiene lotes.")
    if frame.crs is None:
        raise ValueError(
            "El archivo no declara su sistema de coordenadas. "
            "Generar nuevamente el archivo con esa información incorporada."
        )
    frame = frame.loc[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
    frame["geometry"] = frame.geometry.map(make_valid)
    frame = frame.loc[frame.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if frame.empty:
        raise ValueError("No se encontraron polígonos válidos.")
    frame = frame.to_crs(4326).reset_index(drop=True)
    frame["__field_id"] = [f"lote_{index + 1:03d}" for index in range(len(frame))]
    return frame


def area_hectares(geometry) -> float:
    series = gpd.GeoSeries([geometry], crs=4326)
    return float(series.to_crs(series.estimate_utm_crs()).area.iloc[0] / 10_000)


def buffered_geometry(geometry, distance_m: float):
    series = gpd.GeoSeries([geometry], crs=4326)
    metric = series.estimate_utm_crs()
    return series.to_crs(metric).buffer(distance_m).to_crs(4326).iloc[0]


def expanded_bounds(bounds: tuple[float, float, float, float], buffer_km: float) -> tuple[float, float, float, float]:
    west, south, east, north = bounds
    mean_lat = (south + north) / 2
    latitude_pad = buffer_km / 110.574
    longitude_pad = buffer_km / max(111.320 * math.cos(math.radians(mean_lat)), 1)
    return (
        west - longitude_pad,
        south - latitude_pad,
        east + longitude_pad,
        north + latitude_pad,
    )


def _tile_xy(longitude: float, latitude: float, zoom: int) -> tuple[int, int]:
    latitude = max(-85.05112878, min(85.05112878, latitude))
    scale = 2**zoom
    x = int((longitude + 180.0) / 360.0 * scale)
    latitude_rad = math.radians(latitude)
    y = int((1.0 - math.asinh(math.tan(latitude_rad)) / math.pi) / 2.0 * scale)
    return max(0, min(scale - 1, x)), max(0, min(scale - 1, y))


def terrain_tile_indices(bounds: tuple[float, float, float, float], zoom: int = 12) -> list[tuple[int, int, int]]:
    west, south, east, north = bounds
    x_min, y_max = _tile_xy(west, south, zoom)
    x_max, y_min = _tile_xy(east, north, zoom)
    return [(zoom, x, y) for x in range(x_min, x_max + 1) for y in range(y_min, y_max + 1)]


def fetch_open_dem(
    bounds: tuple[float, float, float, float],
    zoom: int = 12,
    maximum_tiles: int = 64,
) -> DemResult:
    indices = terrain_tile_indices(bounds, zoom)
    if len(indices) > maximum_tiles:
        raise ValueError(
            f"La selección requiere {len(indices)} teselas y supera el máximo de "
            f"{maximum_tiles}. Reducir la cantidad de lotes analizados en conjunto."
        )
    memories: list[MemoryFile] = []
    datasets = []
    try:
        for level, x, y in indices:
            url = f"https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{level}/{x}/{y}.tif"
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            memory = MemoryFile(response.content)
            memories.append(memory)
            datasets.append(memory.open())
        mosaic, affine = merge(datasets, nodata=-9999.0)
        profile = datasets[0].profile.copy()
        profile.update(
            driver="GTiff",
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=affine,
            count=1,
            compress="deflate",
            nodata=-9999.0,
        )
        with MemoryFile() as destination_memory:
            with destination_memory.open(**profile) as destination:
                destination.write(mosaic[:1])
            content = destination_memory.read()
    finally:
        for dataset in datasets:
            dataset.close()
        for memory in memories:
            memory.close()
    resolution = round(40_075_016.686 / (2**zoom * 512))
    return DemResult(
        content=content,
        dataset=f"Terrain Tiles z{zoom}",
        native_resolution_m=max(1, resolution),
    )


def _data_url(pixels: np.ndarray) -> str:
    output = io.BytesIO()
    Image.fromarray(pixels, "RGBA").save(output, "PNG")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _continuous_rgba(
    values: np.ndarray,
    valid: np.ndarray,
    palette: str,
    low: float,
    high: float,
) -> np.ndarray:
    scaled = np.clip((values - low) / max(high - low, 1e-9), 0, 1)
    pixels = (colormaps.get_cmap(palette)(scaled) * 255).astype("uint8")
    pixels[..., 3] = np.where(valid, 235, 0).astype("uint8")
    return pixels


def analyze_dem(
    content: bytes,
    analysis_geometry,
    display_geometry,
    mode: str,
) -> RasterOverlay:
    with MemoryFile(content) as memory:
        with memory.open() as source:
            display_shape = transform_geom("EPSG:4326", source.crs, mapping(display_geometry))
            analysis_shape = transform_geom("EPSG:4326", source.crs, mapping(analysis_geometry))
            raster, affine = mask(source, [display_shape], crop=True, filled=False)
            masked = raster[0].astype("float64")
            values = masked.filled(np.nan)
            valid = ~np.ma.getmaskarray(masked) & np.isfinite(values)
            inside = geometry_mask([analysis_shape], values.shape, affine, invert=True, all_touched=True) & valid
            if not inside.any():
                raise ValueError("El DEM no contiene celdas dentro de la selección.")
            selected = values[inside]
            low, high = np.nanpercentile(selected, [2, 98])

            center_latitude = analysis_geometry.centroid.y
            if source.crs.is_geographic:
                dx = abs(affine.a) * 111_320 * math.cos(math.radians(center_latitude))
                dy = abs(affine.e) * 110_574
            else:
                dx, dy = abs(affine.a), abs(affine.e)
            extended = fillnodata(
                np.where(valid, values, 0).astype("float32"),
                mask=valid.astype("uint8"),
                max_search_distance=max(values.shape),
            ).astype("float64")
            padded = np.pad(extended, 1, mode="edge")
            smooth = (
                sum(
                    padded[row : row + extended.shape[0], col : col + extended.shape[1]]
                    for row in range(3)
                    for col in range(3)
                )
                / 9
            )
            dz_dy, dz_dx = np.gradient(smooth, max(dy, 0.01), max(dx, 0.01))
            slope = np.hypot(dz_dx, dz_dy) * 100

            if mode == "slope":
                pixels = np.zeros((*values.shape, 4), dtype="uint8")
                classes = (
                    (0, 1, "#c6e2bb"),
                    (1, 2, "#ffeda0"),
                    (2, 5, "#feb24c"),
                    (5, 10, "#f03b20"),
                    (10, np.inf, "#756bb1"),
                )
                for lower, upper, color in classes:
                    rgb = [int(color[index : index + 2], 16) for index in (1, 3, 5)]
                    pixels[valid & (slope >= lower) & (slope < upper)] = [*rgb, 235]
                legend = tuple(
                    (label, color)
                    for label, color in (
                        ("0 a 1 %", "#c6e2bb"),
                        ("1 a 2 %", "#ffeda0"),
                        ("2 a 5 %", "#feb24c"),
                        ("5 a 10 %", "#f03b20"),
                        ("Más de 10 %", "#756bb1"),
                    )
                )
            elif mode == "hillshade":
                slope_radians = np.arctan(np.hypot(dz_dx, dz_dy))
                aspect = np.arctan2(-dz_dx, dz_dy)
                azimuth, altitude = np.radians(315), np.radians(45)
                shade = np.sin(altitude) * np.cos(slope_radians) + np.cos(altitude) * np.sin(slope_radians) * np.cos(
                    azimuth - aspect
                )
                shade_low, shade_high = np.nanpercentile(shade[valid], [2, 98])
                pixels = _continuous_rgba(shade, valid, "gray", float(shade_low), float(shade_high))
                legend = (("Sombra", "#252525"), ("Iluminación", "#f7f7f7"))
            else:
                pixels = _continuous_rgba(values, valid, "terrain", float(low), float(high))
                legend = (
                    (f"Baja · {low:.1f} m", "#333399"),
                    ("Intermedia", "#7fd34e"),
                    (f"Alta · {high:.1f} m", "#ffffff"),
                )

            west, south, east, north = array_bounds(*values.shape, affine)
            if source.crs.to_epsg() != 4326:
                west, south, east, north = transform_bounds(source.crs, "EPSG:4326", west, south, east, north)
    selected_slope = slope[inside]
    return RasterOverlay(
        image_data_url=_data_url(pixels),
        bounds_wgs84=tuple(float(value) for value in (west, south, east, north)),
        legend=legend,
        metrics={
            "elevation_p02_m": float(low),
            "elevation_p98_m": float(high),
            "relief_m": float(high - low),
            "mean_slope_percent": float(np.nanmean(selected_slope)),
            "area_over_2_percent": float(100 * np.mean(selected_slope > 2)),
        },
    )


def analyze_flow(content: bytes, analysis_geometry, display_geometry) -> RasterOverlay:
    with MemoryFile(content) as memory:
        with memory.open() as source:
            scale = min(1.0, math.sqrt(1_200_000 / max(source.width * source.height, 1)))
            width = max(2, round(source.width * scale))
            height = max(2, round(source.height * scale))
            elevation = source.read(1, out_shape=(height, width), resampling=Resampling.bilinear).astype("float64")
            affine = source.transform * source.transform.scale(source.width / width, source.height / height)
            valid = np.isfinite(elevation)
            if source.nodata is not None:
                valid &= elevation != source.nodata
            analysis_shape = transform_geom("EPSG:4326", source.crs, mapping(analysis_geometry))
            display_shape = transform_geom("EPSG:4326", source.crs, mapping(display_geometry))
            inside = geometry_mask([analysis_shape], (height, width), affine, invert=True) & valid
            display = geometry_mask([display_shape], (height, width), affine, invert=True) & valid

            center_latitude = analysis_geometry.centroid.y
            if source.crs.is_geographic:
                dx = abs(affine.a) * 111_320 * math.cos(math.radians(center_latitude))
                dy = abs(affine.e) * 110_574
            else:
                dx, dy = abs(affine.a), abs(affine.e)
            cell_area_ha = max(dx * dy / 10_000, 1e-9)
            destinations = np.full((height, width), -1, dtype="int64")
            best_drop = np.zeros((height, width), dtype="float64")
            rows, columns = np.indices((height, width))
            offsets = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
            for row_shift, column_shift in offsets:
                distance = math.hypot(dx, dy) if row_shift and column_shift else (dy if row_shift else dx)
                neighbor = np.full_like(elevation, np.nan)
                source_rows = slice(max(0, row_shift), min(height, height + row_shift))
                source_columns = slice(max(0, column_shift), min(width, width + column_shift))
                target_rows = slice(max(0, -row_shift), min(height, height - row_shift))
                target_columns = slice(max(0, -column_shift), min(width, width - column_shift))
                neighbor[target_rows, target_columns] = elevation[source_rows, source_columns]
                drop = (elevation - neighbor) / max(distance, 1e-9)
                better = valid & np.isfinite(neighbor) & (drop > best_drop)
                neighbor_rows = rows + row_shift
                neighbor_columns = columns + column_shift
                destinations[better] = neighbor_rows[better] * width + neighbor_columns[better]
                best_drop[better] = drop[better]
            flat_destinations = destinations.ravel()
            flat_valid = valid.ravel()
            has_destination = flat_valid & (flat_destinations >= 0)
            indegree = np.zeros(height * width, dtype="int32")
            np.add.at(indegree, flat_destinations[has_destination], 1)
            accumulation = np.where(flat_valid, 1.0, 0.0)
            queue = deque(np.flatnonzero(flat_valid & (indegree == 0)).tolist())
            while queue:
                cell = queue.popleft()
                destination = flat_destinations[cell]
                if destination >= 0:
                    accumulation[destination] += accumulation[cell]
                    indegree[destination] -= 1
                    if indegree[destination] == 0:
                        queue.append(int(destination))
            area = accumulation.reshape(height, width) * cell_area_ha
            selected = area[inside & np.isfinite(area)]
            positive = area[display & (area > 0)]
            low, high = np.nanpercentile(positive, [70, 99.5])
            scaled = np.clip(
                (np.log1p(area) - np.log1p(low)) / max(np.log1p(high) - np.log1p(low), 1e-9),
                0,
                1,
            )
            pixels = (colormaps.get_cmap("Blues")(scaled) * 255).astype("uint8")
            pixels[..., 3] = np.where(display & (area >= low), 45 + 190 * scaled, 0).astype("uint8")
            west, south, east, north = array_bounds(height, width, affine)
            if source.crs.to_epsg() != 4326:
                west, south, east, north = transform_bounds(source.crs, "EPSG:4326", west, south, east, north)
    return RasterOverlay(
        image_data_url=_data_url(pixels),
        bounds_wgs84=tuple(float(value) for value in (west, south, east, north)),
        legend=(
            (f"Inicio visible · {low:.2f} ha", "#9ecae1"),
            ("Concentración media", "#3182bd"),
            (f"Alta · {high:.2f} ha", "#08519c"),
        ),
        metrics={"max_contributing_area_ha": float(np.nanmax(selected))},
    )


def fetch_hand(
    bounds: tuple[float, float, float, float],
    analysis_geometry,
    size: int = 1_000,
) -> RasterOverlay:
    west, south, east, north = bounds
    response = requests.get(
        f"{HAND_SERVICE_URL}/exportImage",
        params={
            "bbox": f"{west},{south},{east},{north}",
            "bboxSR": "4326",
            "imageSR": "4326",
            "size": f"{size},{size}",
            "format": "tiff",
            "pixelType": "F32",
            "interpolation": "RSP_NearestNeighbor",
            "f": "image",
        },
        timeout=90,
    )
    response.raise_for_status()
    with MemoryFile(response.content) as memory:
        with memory.open() as source:
            raster = source.read(1, masked=True)
            values = raster.astype("float64").filled(np.nan)
            valid = ~np.ma.getmaskarray(raster) & np.isfinite(values) & (values >= 0) & (values < 10_000)
            analysis_shape = transform_geom("EPSG:4326", source.crs, mapping(analysis_geometry))
            inside = (
                geometry_mask([analysis_shape], values.shape, source.transform, invert=True, all_touched=True) & valid
            )
            selected = values[inside]
            if not len(selected):
                raise ValueError("HAND no contiene celdas dentro de la selección.")
            pixels = np.zeros((*values.shape, 4), dtype="uint8")
            classes = (
                (0, 1, "#08519c"),
                (1, 2, "#3182bd"),
                (2, 5, "#6baed6"),
                (5, 10, "#c6dbef"),
                (10, np.inf, "#fee391"),
            )
            for lower, upper, color in classes:
                rgb = [int(color[index : index + 2], 16) for index in (1, 3, 5)]
                pixels[valid & (values >= lower) & (values < upper)] = [*rgb, 225]
            display_bounds = array_bounds(*values.shape, source.transform)
            if source.crs.to_epsg() != 4326:
                display_bounds = transform_bounds(source.crs, "EPSG:4326", *display_bounds)
    return RasterOverlay(
        image_data_url=_data_url(pixels),
        bounds_wgs84=tuple(float(value) for value in display_bounds),
        legend=tuple(
            (label, color)
            for label, color in (
                ("0 a 1 m", "#08519c"),
                ("1 a 2 m", "#3182bd"),
                ("2 a 5 m", "#6baed6"),
                ("5 a 10 m", "#c6dbef"),
                ("Más de 10 m", "#fee391"),
            )
        ),
        metrics={
            "median_m": float(np.nanmedian(selected)),
            "area_below_2m_percent": float(100 * np.mean(selected < 2)),
        },
    )
