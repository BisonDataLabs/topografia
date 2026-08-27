from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import make_valid


BOUNDARY_EXTENSIONS = ["geojson", "json", "gpkg", "kml", "kmz", "zip"]
MONITOR_EXTENSIONS = ["geojson", "json", "gpkg", "kml", "kmz", "zip", "shp", "shx", "dbf", "prj", "cpg"]
SHAPEFILE_COMPONENT_EXTENSIONS = {".shp", ".shx", ".dbf", ".prj", ".cpg"}


def _read_vector(filename: str, payload: bytes) -> gpd.GeoDataFrame:
    suffix = Path(filename).suffix.lower()
    with tempfile.TemporaryDirectory(prefix="topografia-vector-") as temp_name:
        temp = Path(temp_name)
        source = temp / Path(filename).name
        source.write_bytes(payload)
        if suffix == ".kmz":
            with zipfile.ZipFile(source) as archive:
                archive.extractall(temp / "kmz")
            kml_files = list((temp / "kmz").rglob("*.kml"))
            if not kml_files:
                raise ValueError("El KMZ no contiene un archivo KML.")
            return gpd.read_file(kml_files[0])
        if suffix == ".zip":
            return gpd.read_file(f"zip://{source}")
        return gpd.read_file(source)


def read_boundaries(filename: str, payload: bytes, fallback_crs: str | None = None) -> gpd.GeoDataFrame:
    frame = _read_vector(filename, payload)

    if frame.empty:
        raise ValueError("El archivo de límites no contiene entidades.")
    if frame.crs is None:
        if not fallback_crs:
            raise ValueError("El archivo no declara CRS. Es necesario indicar el EPSG de origen.")
        frame = frame.set_crs(fallback_crs)
    frame = frame.loc[frame.geometry.notna()].copy()
    frame["geometry"] = frame.geometry.map(make_valid)
    frame = frame.loc[frame.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if frame.empty:
        raise ValueError("No se encontraron polígonos o multipolígonos válidos.")
    frame = frame.to_crs("EPSG:4326")
    frame["__field_id"] = [f"lote_{index + 1:03d}" for index in range(len(frame))]
    return frame


def read_monitor_vector(filename: str, payload: bytes, fallback_crs: str | None = None) -> gpd.GeoDataFrame:
    frame = _read_vector(filename, payload)
    if frame.empty:
        raise ValueError("El archivo de monitor no contiene entidades.")
    frame = frame.loc[frame.geometry.notna() & ~frame.geometry.is_empty].copy()
    if frame.crs is None:
        if not fallback_crs:
            raise ValueError("El monitor no declara CRS. Es necesario indicar el EPSG de origen.")
        frame = frame.set_crs(fallback_crs)
    return frame.to_crs("EPSG:4326")


def read_monitor_shapefile_components(
    files: tuple[tuple[str, bytes], ...], fallback_crs: str | None = None
) -> gpd.GeoDataFrame:
    with tempfile.TemporaryDirectory(prefix="topografia-monitor-shp-") as temp_name:
        temp = Path(temp_name)
        for filename, payload in files:
            (temp / Path(filename).name).write_bytes(payload)
        shapefiles = [source for source in temp.iterdir() if source.suffix.lower() == ".shp"]
        if not shapefiles:
            raise ValueError("La selección no contiene el componente .shp.")
        available_dbf = {source.stem.lower() for source in temp.iterdir() if source.suffix.lower() == ".dbf"}
        missing_attributes = [source.name for source in shapefiles if source.stem.lower() not in available_dbf]
        if missing_attributes:
            raise ValueError("El Shapefile necesita su componente DBF para leer la columna de elevación.")
        frames = [gpd.read_file(source) for source in shapefiles]
    normalized = []
    for frame in frames:
        if frame.crs is None:
            if not fallback_crs:
                raise ValueError("El Shapefile no declara CRS. Es necesario indicar el EPSG de origen.")
            frame = frame.set_crs(fallback_crs)
        normalized.append(frame.to_crs("EPSG:4326"))
    combined = gpd.GeoDataFrame(pd.concat(normalized, ignore_index=True, sort=False), geometry="geometry", crs="EPSG:4326")
    if combined.empty:
        raise ValueError("El Shapefile de monitor no contiene entidades.")
    return combined.loc[combined.geometry.notna() & ~combined.geometry.is_empty].copy()


def column_guess(columns: list[str], concepts: list[str]) -> str | None:
    normalized = {column: str(column).lower().replace("_", " ") for column in columns}
    for concept in concepts:
        for column, value in normalized.items():
            if concept in value:
                return column
    return None


def normalize_monitor_vector(
    frame: gpd.GeoDataFrame,
    elevation_column: str,
    optional_columns: dict[str, str | None] | None = None,
) -> gpd.GeoDataFrame:
    points = frame.copy().explode(index_parts=False, ignore_index=True)
    points["elev_m_raw"] = pd.to_numeric(points[elevation_column], errors="coerce")
    points = points.dropna(subset=["elev_m_raw"]).copy()
    if points.empty:
        raise ValueError("La columna seleccionada no contiene elevaciones numéricas.")
    point_mask = points.geom_type == "Point"
    if not point_mask.all():
        points.loc[~point_mask, "geometry"] = points.loc[~point_mask, "geometry"].representative_point()
    points = points.to_crs("EPSG:4326")
    points["lon"] = points.geometry.x
    points["lat"] = points.geometry.y
    for target, source in (optional_columns or {}).items():
        if source:
            points[target] = points[source]
    return points


def monitor_diagnostics(points: gpd.GeoDataFrame, boundary_geometry) -> dict:
    inside = points.geometry.within(boundary_geometry)
    valid = points.loc[inside, "elev_m_raw"]
    corr_lon = points["elev_m_raw"].corr(points["lon"])
    corr_lat = points["elev_m_raw"].corr(points["lat"])
    suspicious_coordinate_copy = bool(
        (np.isfinite(corr_lon) and abs(corr_lon) > 0.9999)
        or (np.isfinite(corr_lat) and abs(corr_lat) > 0.9999)
    )
    return {
        "rows_numeric": int(len(points)),
        "rows_inside": int(inside.sum()),
        "inside_percent": float(100 * inside.mean()) if len(points) else 0.0,
        "unique_elevations": int(valid.nunique()),
        "elevation_p01": float(valid.quantile(0.01)) if len(valid) else None,
        "elevation_median": float(valid.median()) if len(valid) else None,
        "elevation_p99": float(valid.quantile(0.99)) if len(valid) else None,
        "suspicious_coordinate_copy": suspicious_coordinate_copy,
        "inside_mask": inside,
    }
