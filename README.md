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


