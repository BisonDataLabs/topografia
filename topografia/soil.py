from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import requests
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.warp import transform_geom
from shapely.geometry import mapping


WCS_URL = "https://maps.isric.org/mapserv"
PROPERTIES = {
    "clay": ("Arcilla", 10.0, "%"),
    "silt": ("Limo", 10.0, "%"),
    "sand": ("Arena", 10.0, "%"),
    "soc": ("Carbono orgánico", 10.0, "g/kg"),
}


@dataclass(frozen=True)
class SoilSummary:
    clay_percent: float
    silt_percent: float
    sand_percent: float
    organic_carbon_gkg: float
    texture_label: str
    erosion_screening: str
    table: tuple[dict, ...]
    source_url: str
    resolution_m: int


def _coverage(property_code: str, bounds: tuple[float, float, float, float], timeout_seconds: int) -> bytes:
    west, south, east, north = bounds
    parameters = [
        ("map", f"/map/{property_code}.map"),
        ("SERVICE", "WCS"),
        ("VERSION", "2.0.1"),
        ("REQUEST", "GetCoverage"),
        ("COVERAGEID", f"{property_code}_0-5cm_Q0.5"),
        ("FORMAT", "GEOTIFF_INT16"),
        ("SUBSETTINGCRS", "http://www.opengis.net/def/crs/EPSG/0/4326"),
        ("OUTPUTCRS", "http://www.opengis.net/def/crs/EPSG/0/4326"),
        ("SUBSET", f"Long({west},{east})"),
        ("SUBSET", f"Lat({south},{north})"),
    ]
    response = requests.get(WCS_URL, params=parameters, timeout=timeout_seconds)
    response.raise_for_status()
    if "tiff" not in response.headers.get("content-type", ""):
        raise RuntimeError(f"SoilGrids no devolvió una grilla válida para {property_code}.")
    return response.content


def _median_inside(content: bytes, boundary_wgs84) -> float:
    with MemoryFile(content) as memory:
        with memory.open() as source:
            geometry = transform_geom("EPSG:4326", source.crs, mapping(boundary_wgs84))
            raster, _ = mask(source, [geometry], crop=True, filled=False, all_touched=True)
            values = raster[0].astype("float64").filled(np.nan)
            valid = ~np.ma.getmaskarray(raster[0]) & np.isfinite(values) & (values > 0)
            if not valid.any():
                raise ValueError("SoilGrids no contiene celdas válidas dentro del lote.")
            return float(np.nanmedian(values[valid]))


def _texture(clay: float, silt: float, sand: float) -> str:
    if clay >= 40:
        return "Predominio arcilloso"
    if sand >= 70 and clay < 15:
        return "Predominio arenoso"
    if silt >= 60 and clay < 27:
        return "Predominio limoso"
    return "Textura mixta o franca"


def fetch_soilgrids(
    bounds: tuple[float, float, float, float], boundary_wgs84, timeout_seconds: int = 90
) -> SoilSummary:
    raw = {}
    for code in PROPERTIES:
        raw[code] = _median_inside(_coverage(code, bounds, timeout_seconds), boundary_wgs84) / PROPERTIES[code][1]

    texture = _texture(raw["clay"], raw["silt"], raw["sand"])
    if raw["silt"] >= 50 or raw["soc"] < 10:
        screening = "Sensibilidad potencial alta por textura fina o bajo carbono orgánico"
    elif raw["silt"] >= 35 or raw["soc"] < 20:
        screening = "Sensibilidad potencial intermedia; contrastar con pendiente y cobertura"
    else:
        screening = "Sensibilidad textural comparativamente menor; la pendiente y la cobertura siguen siendo decisivas"

    table = tuple(
        {
            "Propiedad": PROPERTIES[code][0],
            "Mediana": round(raw[code], 1),
            "Unidad": PROPERTIES[code][2],
            "Profundidad": "0 a 5 cm",
        }
        for code in PROPERTIES
    )
    return SoilSummary(
        clay_percent=raw["clay"],
        silt_percent=raw["silt"],
        sand_percent=raw["sand"],
        organic_carbon_gkg=raw["soc"],
        texture_label=texture,
        erosion_screening=screening,
        table=table,
        source_url="https://docs.isric.org/globaldata/soilgrids/wcs.html",
        resolution_m=250,
    )
