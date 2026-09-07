from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_opens_without_a_preloaded_package() -> None:
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run(timeout=30)

    assert not app.exception
    assert app.title[0].value == "Topografía"
    assert len(app.get("file_uploader")) == 1
    assert [option.value for option in app.radio] == ["Paquete topográfico"]


def test_public_entry_has_no_monitor_or_manual_crs() -> None:
    source = (Path(__file__).parents[1] / "universal_app.py").read_text(encoding="utf-8")

    assert "Monitor opcional" not in source
    assert "CRS si el archivo" not in source
    assert "Radar satelital" not in source
    assert "Caracterizar suelo" not in source
    assert "Resumir lluvia" not in source
    assert '"Indicadores"' not in source
    assert '"Calidad y supuestos"' not in source
    assert 'st.tabs(["Resumen", "Mapa y capas"])' in source
    assert "Paquete completo" not in source
