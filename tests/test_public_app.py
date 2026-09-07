import hashlib
import io
import json
import zipfile

import pytest

from topografia.map_layers import BASEMAPS, view_state
from topografia.package_io import load_package


def _package() -> bytes:
    layer_content = b"example"
    files = [
        {
            "path": "rasters/elevation.tif",
            "size_bytes": len(layer_content),
            "sha256": hashlib.sha256(layer_content).hexdigest(),
        }
    ]
    catalog = {
        "schema": "topografia-layer-catalog",
        "field": {"id": "field-1", "name": "Example"},
        "layers": [
            {
                "id": "elevation",
                "title": "Elevación",
                "path": "rasters/elevation.tif",
                "format": "cog",
                "kind": "raster",
                "group": "relief",
                "opacity": 0.8,
                "visible": True,
            }
        ],
    }
    manifest = {
        "schema": "topografia-streamlit-package",
        "field": catalog["field"],
        "files": files,
    }
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w") as archive:
        archive.writestr("layers.json", json.dumps(catalog))
        archive.writestr("package.json", json.dumps(manifest))
        archive.writestr("rasters/elevation.tif", layer_content)
    return result.getvalue()


def test_loads_valid_package_and_verifies_integrity():
    package = load_package(_package())

    assert package.catalog["field"]["name"] == "Example"
    assert package.package["files"][0]["size_bytes"] == 7


def test_rejects_zip_path_traversal():
    content = io.BytesIO()
    with zipfile.ZipFile(content, "w") as archive:
        archive.writestr("../private.txt", "x")

    with pytest.raises(ValueError, match="rutas no admitidas"):
        load_package(content.getvalue())


def test_basemaps_need_no_key_and_view_fits_bounds():
    assert {"Topográfico con nombres", "Satelital", "Híbrido", "Calles"} <= BASEMAPS.keys()
    assert all("key=" not in url for layers in BASEMAPS.values() for url, _ in layers)
    view = view_state([(-60.0, -33.0, -59.99, -32.99)])
    assert view.latitude == pytest.approx(-32.995)
    assert 3 <= view.zoom <= 17
