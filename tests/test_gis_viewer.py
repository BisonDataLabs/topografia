from topografia.gis_viewer import viewer_html


def test_viewer_keeps_gis_controls_inside_map() -> None:
    html = viewer_html(
        [
            {
                "id": "boundary",
                "original_id": "boundary",
                "title": "Límite",
                "group": "reference",
                "kind": "vector",
                "geojson": {"type": "FeatureCollection", "features": []},
                "bounds": [-59, -33, -58, -32],
                "renderer": "line",
                "opacity": 1,
                "visible": True,
            }
        ],
        [-59, -33, -58, -32],
    )

    assert "Topográfico con nombres" in html
    assert "Satelital" in html
    assert "Mapas y capas" in html
    assert "slider.addEventListener('input'" in html
    assert "base-opacity" in html
    assert "fillOpacity:boundary?0" in html
    assert "pane:'rasters'" in html
    assert "data-topografia-leaflet" in html
    assert "dataset.ready" in html
