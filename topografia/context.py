from __future__ import annotations

from dataclasses import dataclass

import requests


@dataclass(frozen=True)
class OsmContext:
    roads: dict
    waterways: dict
    road_count: int
    waterway_count: int
    source_url: str


def _feature(element: dict, kind: str) -> dict | None:
    coordinates = [[point["lon"], point["lat"]] for point in element.get("geometry", [])]
    if len(coordinates) < 2:
        return None
    tags = element.get("tags", {})
    if kind == "waterway":
        color = [15, 112, 188, 230]
        width = 2.4 if tags.get("waterway") in {"river", "canal"} else 1.6
        category = tags.get("waterway", "curso")
    else:
        road_type = tags.get("highway", "camino")
        if road_type in {"motorway", "trunk", "primary", "secondary", "tertiary"}:
            color, width = [196, 74, 46, 230], 2.8
        elif road_type in {"track", "path"}:
            color, width = [151, 106, 56, 225], 1.8
        else:
            color, width = [82, 82, 82, 220], 2.0
        category = road_type
    return {
        "type": "Feature",
        "properties": {
            "tipo": "Curso de agua" if kind == "waterway" else "Camino existente",
            "detalle": tags.get("name") or category,
            "categoria": category,
            "color": color,
            "width": width,
            "osm_id": element.get("id"),
        },
        "geometry": {"type": "LineString", "coordinates": coordinates},
    }


def fetch_osm_context(bounds: tuple[float, float, float, float], timeout_seconds: int = 45) -> OsmContext:
    west, south, east, north = bounds
    if max(east - west, north - south) > 0.75:
        raise ValueError("El área de contexto es demasiado extensa para una consulta pública responsable.")
    query = f"""
    [out:json][timeout:30];
    (
      way["highway"]({south},{west},{north},{east});
      way["waterway"]({south},{west},{north},{east});
    );
    out tags geom;
    """
    endpoint = "https://overpass-api.de/api/interpreter"
    response = requests.post(
        endpoint,
        data={"data": query},
        timeout=timeout_seconds,
        headers={"User-Agent": "DiagnosticoTopografico/1.0"},
    )
    response.raise_for_status()
    elements = response.json().get("elements", [])
    roads = []
    waterways = []
    for element in elements:
        tags = element.get("tags", {})
        if "waterway" in tags:
            feature = _feature(element, "waterway")
            if feature:
                waterways.append(feature)
        elif "highway" in tags:
            feature = _feature(element, "road")
            if feature:
                roads.append(feature)
    return OsmContext(
        roads={"type": "FeatureCollection", "features": roads},
        waterways={"type": "FeatureCollection", "features": waterways},
        road_count=len(roads),
        waterway_count=len(waterways),
        source_url="https://www.openstreetmap.org/",
    )
