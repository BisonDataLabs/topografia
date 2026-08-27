from __future__ import annotations

import base64
import io
from dataclasses import dataclass

import numpy as np
import requests
from PIL import Image
from rasterio.features import geometry_mask
from rasterio.io import MemoryFile
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds, transform_geom
from shapely.geometry import mapping


SERVICE_URL = "https://gis.asf.alaska.edu/arcgis/rest/services/GlobalHAND/GLO30_HAND/ImageServer"


@dataclass(frozen=True)
class HandOverlay:
    image_data_url: str
    raster_content: bytes
    bounds_wgs84: tuple[float, float, float, float]
    median_m: float
    p10_m: float
    p90_m: float
    area_below_2m_percent: float
    area_below_5m_percent: float
    histogram: tuple[tuple[str, float], ...]
    source_url: str


def fetch_hand(
    bounds: tuple[float, float, float, float],
    boundary_wgs84,
    size: int = 900,
    timeout_seconds: int = 90,
) -> HandOverlay:
    """Obtener y resumir HAND, altura vertical sobre el drenaje más cercano."""
    west, south, east, north = bounds
    response = requests.get(
        f"{SERVICE_URL}/exportImage",
        params={
            "bbox": f"{west},{south},{east},{north}",
            "bboxSR": "4326",
            "imageSR": "4326",
            "size": f"{size},{size}",
            "format": "tiff",
            "pixelType": "F32",
            "interpolation": "RSP_BilinearInterpolation",
            "f": "image",
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    if "tiff" not in response.headers.get("content-type", ""):
        raise RuntimeError("El servicio HAND no devolvió un GeoTIFF.")

    with MemoryFile(response.content) as memory:
        with memory.open() as source:
            geometry = transform_geom("EPSG:4326", source.crs, mapping(boundary_wgs84))
            raster = source.read(1, masked=True)
            affine = source.transform
            values = raster.astype("float64").filled(np.nan)
            valid = ~np.ma.getmaskarray(raster) & np.isfinite(values) & (values >= 0) & (values < 10_000)
            inside = geometry_mask([geometry], values.shape, affine, invert=True, all_touched=True) & valid
            if not inside.any():
                raise ValueError("HAND no contiene celdas válidas dentro del lote.")
            selected = values[inside]

            pixels = np.zeros((*values.shape, 4), dtype=np.uint8)
            classes = [
                (0, 1, [8, 81, 156, 230]),
                (1, 2, [49, 130, 189, 225]),
                (2, 5, [107, 174, 214, 220]),
                (5, 10, [198, 219, 239, 215]),
                (10, np.inf, [254, 227, 145, 210]),
            ]
            for lower, upper, color in classes:
                pixels[valid & (values >= lower) & (values < upper)] = color

            output = io.BytesIO()
            Image.fromarray(pixels, "RGBA").save(output, "PNG")
            image_content = output.getvalue()
            display_bounds = array_bounds(values.shape[0], values.shape[1], affine)
            if source.crs.to_epsg() != 4326:
                display_bounds = transform_bounds(source.crs, "EPSG:4326", *display_bounds)

    labels = ["0 a 1 m", "1 a 2 m", "2 a 5 m", "5 a 10 m", "Más de 10 m"]
    histogram = tuple(
        (label, float(100 * np.mean((selected >= lower) & (selected < upper))))
        for label, (lower, upper, _) in zip(labels, classes)
    )
    return HandOverlay(
        image_data_url=f"data:image/png;base64,{base64.b64encode(image_content).decode('ascii')}",
        raster_content=response.content,
        bounds_wgs84=tuple(float(value) for value in display_bounds),
        median_m=float(np.nanmedian(selected)),
        p10_m=float(np.nanpercentile(selected, 10)),
        p90_m=float(np.nanpercentile(selected, 90)),
        area_below_2m_percent=float(100 * np.mean(selected < 2)),
        area_below_5m_percent=float(100 * np.mean(selected < 5)),
        histogram=histogram,
        source_url=SERVICE_URL,
    )
