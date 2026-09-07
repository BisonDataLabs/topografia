from pathlib import Path

import geopandas as gpd
from shapely.geometry import LineString, Point

from topografia.map_layers import vector_layer
from topografia.map_layers import raster_layer

import numpy as np
import rasterio
from rasterio.transform import from_origin


def test_colors_points_by_elevation(tmp_path: Path) -> None:
    path = tmp_path / "points.gpkg"
    gpd.GeoDataFrame(
        {"elevation_m": [30.0, 40.0]},
        geometry=[Point(-58.7, -33.3), Point(-58.69, -33.29)],
        crs="EPSG:4326",
    ).to_file(path, driver="GPKG")
    layer, _ = vector_layer(
        path,
        {
            "id": "points",
            "style": {
                "renderer": "graduated_points",
                "field": "elevation_m",
                "palette": "terrain",
                "range": [30, 40],
            },
        },
        0.8,
    )
    colors = [feature["properties"]["_fill"] for feature in layer.data["features"]]
    assert colors[0] != colors[1]
    assert all(color[3] == 204 for color in colors)


def test_colors_relative_contours_by_category(tmp_path: Path) -> None:
    path = tmp_path / "contours.gpkg"
    gpd.GeoDataFrame(
        {"level": [1, 2]},
        geometry=[
            LineString([(-58.7, -33.3), (-58.69, -33.29)]),
            LineString([(-58.7, -33.3), (-58.68, -33.28)]),
        ],
        crs="EPSG:4326",
    ).to_file(path, driver="GPKG")
    layer, _ = vector_layer(
        path,
        {
            "id": "contours",
            "style": {"renderer": "categorized_lines", "field": "level"},
            "legend": [
                {"value": 1, "color": "#6c757d"},
                {"value": 2, "color": "#d97706"},
            ],
        },
        1.0,
    )
    colors = [feature["properties"]["_line"] for feature in layer.data["features"]]
    assert colors == [[108, 117, 125, 255], [217, 119, 6, 255]]


def test_bitmap_image_is_serialized_as_a_literal_url(tmp_path: Path) -> None:
    path = tmp_path / "elevation.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(-58.7, -33.2, 0.001, 0.001),
    ) as destination:
        destination.write(np.array([[30, 31], [32, 33]], dtype="float32"), 1)

    layer, _ = raster_layer(
        path,
        {
            "id": "elevation",
            "stats": {"p02": 30, "p98": 33},
            "palette": "terrain",
        },
        0.8,
    )

    serialized = layer.to_json()
    assert '"image": "data:image/png;base64,' in serialized
    assert '"@@=' not in serialized
