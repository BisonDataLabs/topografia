from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timezone

import requests


@dataclass(frozen=True)
class RadarOverlay:
    image_data_url: str
    image_content: bytes
    bounds_wgs84: tuple[float, float, float, float]
    polarization: str
    acquisition_date: str
    scene_count: int
    source_url: str


def fetch_nasa_opera_radar(
    bounds: tuple[float, float, float, float], polarization: str = "VV", size: int = 900, timeout_seconds: int = 90
) -> RadarOverlay:
    polarization = polarization.upper()
    if polarization not in {"VV", "VH"}:
        raise ValueError("La polarización debe ser VV o VH.")
    west, south, east, north = bounds
    service = (
        "https://gis.earthdata.nasa.gov/image/rest/services/"
        f"OPERA_L2_RTC_S1_V1/OPERA_L2_RTC_S1_V1_{polarization}/ImageServer"
    )
    query = requests.get(
        f"{service}/query",
        params={
            "geometry": f"{west},{south},{east},{north}",
            "geometryType": "esriGeometryEnvelope",
            "inSR": "4326",
            "where": "1=1",
            "outFields": "objectid,name,startdate,enddate,polarization",
            "returnGeometry": "false",
            "orderByFields": "startdate DESC",
            "resultRecordCount": "30",
            "f": "json",
        },
        timeout=timeout_seconds,
    )
    query.raise_for_status()
    features = query.json().get("features", [])
    if not features:
        raise ValueError("No se encontraron escenas NASA OPERA Sentinel-1 para el área.")
    latest_ms = max(item["attributes"]["startdate"] for item in features)
    latest_day = datetime.fromtimestamp(latest_ms / 1000, tz=timezone.utc).date()
    latest = [
        item for item in features
        if datetime.fromtimestamp(item["attributes"]["startdate"] / 1000, tz=timezone.utc).date() == latest_day
    ]
    export = requests.get(
        f"{service}/exportImage",
        params={
            "bbox": f"{west},{south},{east},{north}",
            "bboxSR": "4326",
            "imageSR": "4326",
            "size": f"{size},{size}",
            "format": "png32",
            "transparent": "true",
            "interpolation": "RSP_BilinearInterpolation",
            "f": "image",
        },
        timeout=timeout_seconds,
    )
    export.raise_for_status()
    if "image" not in export.headers.get("content-type", ""):
        raise RuntimeError("El servicio radar no devolvió una imagen.")
    return RadarOverlay(
        image_data_url=f"data:image/png;base64,{base64.b64encode(export.content).decode('ascii')}",
        image_content=export.content,
        bounds_wgs84=tuple(float(value) for value in bounds),
        polarization=polarization,
        acquisition_date=latest_day.isoformat(),
        scene_count=len(latest),
        source_url=service,
    )
