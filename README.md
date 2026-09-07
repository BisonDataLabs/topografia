# Topografía

Aplicación pública de Streamlit para explorar información topográfica de uno o varios lotes. No contiene límites, datos ni resultados precargados.

## Formas de uso

La pantalla inicial ofrece dos recorridos.

### Paquete topográfico

Carga conjunta de uno o más ZIP generados por el paquete técnico. Cada ZIP puede incorporar:

- evidencia del equipo de siembra por campaña;
- puntos aceptados y observaciones descartadas;
- superficies interpoladas y respaldo local;
- DEM público y versión alineada para comparación;
- desacuerdo entre campañas;
- microrelieve y altura relativa;
- pendiente, sombreado y curvas de nivel;
- altura sobre el drenaje y concentración de flujo;
- métricas e imágenes descargables.

La app permite filtrar lotes, prender varias capas simultáneamente y cambiar sus opacidades. El visor concentra el selector de mapa base, las capas, sus leyendas y sus transparencias dentro del propio mapa. Estos cambios se resuelven en el navegador y no vuelven a ejecutar el análisis. El paquete ya contiene las capas, métricas e imágenes, por lo que este recorrido no presenta descargas redundantes.

Para grupos numerosos, la selección inicial muestra superficie interpolada, curvas y límites de todos los lotes. La opción `Lotes con todas las capas` permite incorporar el detalle técnico únicamente donde hace falta, evitando cargar decenas de capas por cada lote. El DEM público se identifica como una fuente de aproximadamente 30 m aunque haya sido regrillado para compararlo con otras superficies. Ese regrillado no agrega detalle espacial.

### Sólo límites de lotes

Carga de GeoJSON, KML, KMZ, GeoPackage o un Shapefile completo dentro de un ZIP. El archivo debe declarar su sistema de coordenadas.

Este recorrido permite:

- analizar un lote o un grupo;
- obtener el DEM público sin clave;
- visualizar elevación, pendiente, sombreado y acumulación de flujo;
- consultar HAND con un contexto hídrico de 2, 5 o 10 km;
- descargar límites, DEM e imágenes analíticas.

No existe carga directa de monitor ni selección manual de CRS. Los datos del equipo de siembra ingresan solamente mediante un paquete topográfico validado.

## Alcance

La aplicación no incluye radar satelital, suelo, lluvia, clasificación de cobertura ni evaluación de caminos. Tampoco genera visores HTML autónomos.

## Publicar en Streamlit Community Cloud

1. Crear un repositorio nuevo en GitHub.
2. Subir el contenido de esta carpeta a la raíz del repositorio.
3. Crear una aplicación en Streamlit Community Cloud seleccionando ese repositorio.
4. Seleccionar `app.py` como archivo principal.
5. Seleccionar Python 3.13 en la configuración avanzada.

La aplicación no necesita secretos. Los mapas topográfico, satelital, híbrido y de calles utilizan teselas públicas sin clave.

## Ejecutar localmente

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
streamlit run app.py
```

## Verificar

```bash
make check
```

## Contenido del repositorio

- `app.py`: entrada de Streamlit.
- `universal_app.py`: interfaz consolidada.
- `topografia/package_io.py`: validación segura de paquetes ZIP.
- `topografia/gis_viewer.py`: visor Leaflet, mapas base, capas, leyendas y opacidades.
- `topografia/map_layers.py`: compatibilidad de representación de COG, GeoPackage y GeoParquet.
- `topografia/public_analysis.py`: límites, DEM público, relieve, flujo y HAND.
- `.streamlit/config.toml`: tema blanco y configuración de carga.
- `requirements.txt`: dependencias de producción.
- `tests/`: pruebas automatizadas.
- `.github/workflows/ci.yml`: verificación automática con Python 3.13.
