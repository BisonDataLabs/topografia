from __future__ import annotations

from dataclasses import dataclass
import io
import math

import requests
from rasterio.io import MemoryFile
from rasterio.merge import merge


@dataclass(frozen=True)
class DemResult:
    content: bytes
    dataset: str
    provider: str
    source_url: str
    horizontal_crs: str
    vertical_reference: str
    native_resolution_m: int


def expanded_bounds(bounds: tuple[float, float, float, float], buffer_km: float) -> tuple[float, float, float, float]:
    west, south, east, north = bounds
    mean_lat = (south + north) / 2
    lat_pad = buffer_km / 110.574
    lon_pad = buffer_km / max(111.320 * __import__("math").cos(__import__("math").radians(mean_lat)), 1)
    return west - lon_pad, south - lat_pad, east + lon_pad, north + lat_pad


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


def fetch_aws_terrain(
    bounds: tuple[float, float, float, float], zoom: int = 12, timeout_seconds: int = 60, maximum_tiles: int = 36
) -> DemResult:
    indices = terrain_tile_indices(bounds, zoom)
    if len(indices) > maximum_tiles:
        raise ValueError(
            f"El área requiere {len(indices)} teselas; el máximo público configurado es {maximum_tiles}. "
            "Es necesario reducir el buffer o dividir el área."
        )
    memories: list[MemoryFile] = []
    datasets = []
    try:
        for level, x, y in indices:
            url = f"https://s3.amazonaws.com/elevation-tiles-prod/geotiff/{level}/{x}/{y}.tif"
            response = requests.get(url, timeout=timeout_seconds)
            response.raise_for_status()
            memory = MemoryFile(response.content)
            memories.append(memory)
            datasets.append(memory.open())
        mosaic, transform = merge(datasets, nodata=-9999.0)
        profile = datasets[0].profile.copy()
        profile.update(
            driver="GTiff",
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=transform,
            count=mosaic.shape[0],
            compress="deflate",
            nodata=-9999.0,
        )
        output = io.BytesIO()
        with MemoryFile() as destination_memory:
            with destination_memory.open(**profile) as destination:
                destination.write(mosaic)
            output.write(destination_memory.read())
    finally:
        for dataset in datasets:
            dataset.close()
        for memory in memories:
            memory.close()
    resolution = round(40_075_016.686 / (2**zoom * 512))
    return DemResult(
        content=output.getvalue(),
        dataset=f"Terrain Tiles z{zoom}",
        provider="AWS Registry of Open Data / Mapzen",
        source_url="https://registry.opendata.aws/terrain-tiles/",
        horizontal_crs="Web Mercator / EPSG:3857",
        vertical_reference="Compilación multi-fuente; consultar metadatos de Terrain Tiles",
        native_resolution_m=max(1, resolution),
    )
