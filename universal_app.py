from __future__ import annotations

import base64
import json
import math
from pathlib import Path

import altair as alt
import geopandas as gpd
import pandas as pd
import pydeck as pdk
import streamlit as st
from pydeck.types import String
from shapely import from_wkb, union_all

from topografia.analytics import monitor_dashboard, quality_grade
from topografia.climate import fetch_rainfall
from topografia.context import fetch_osm_context
from topografia.data import display_sample, geometry_area_ha
from topografia.exports import (
    bar_chart_png,
    context_map_png,
    map_png,
    overlay_map_png,
    result_zip,
    series_chart_png,
    standalone_html,
)
from topografia.hydrology import analyze_flow
from topografia.hand import fetch_hand
from topografia.ingestion import (
    BOUNDARY_EXTENSIONS,
    MONITOR_EXTENSIONS,
    SHAPEFILE_COMPONENT_EXTENSIONS,
    column_guess,
    monitor_diagnostics,
    normalize_monitor_vector,
    read_boundaries,
    read_monitor_vector,
    read_monitor_shapefile_components,
)
from topografia.providers import expanded_bounds, fetch_aws_terrain, terrain_tile_indices
from topografia.radar import fetch_nasa_opera_radar
from topografia.soil import fetch_soilgrids
from topografia.terrain import analyze_dem, analyze_monitor


BASEMAPS = {
    "Topográfico · OpenTopoMap": "topo-style.json",
    "Satelital · Esri World Imagery": "satellite-style.json",
    "Híbrido · Esri Imagery + rótulos": "hybrid-style.json",
    "Calles · OpenStreetMap": "light-style.json",
}
ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="Diagnóstico topográfico", page_icon="🗺️", layout="wide")


@st.cache_data(show_spinner=False)
def parse_boundaries(filename: str, payload: bytes, fallback_crs: str | None) -> gpd.GeoDataFrame:
    return read_boundaries(filename, payload, fallback_crs)


@st.cache_data(show_spinner=False)
def parse_monitor(filename: str, payload: bytes, fallback_crs: str | None) -> gpd.GeoDataFrame:
    return read_monitor_vector(filename, payload, fallback_crs)


@st.cache_data(show_spinner=False)
def parse_monitor_shapefile(files: tuple[tuple[str, bytes], ...], fallback_crs: str | None) -> gpd.GeoDataFrame:
    return read_monitor_shapefile_components(files, fallback_crs)


@st.cache_data(ttl=86_400, show_spinner=False)
def get_open_dem(bounds: tuple[float, float, float, float], zoom: int):
    return fetch_aws_terrain(bounds, zoom=zoom)


@st.cache_data(ttl=21_600, show_spinner=False)
def get_open_context(bounds: tuple[float, float, float, float]):
    return fetch_osm_context(bounds)


@st.cache_data(ttl=21_600, show_spinner=False)
def get_radar_overlay(bounds: tuple[float, float, float, float], polarization: str):
    return fetch_nasa_opera_radar(bounds, polarization)


@st.cache_data(ttl=86_400, show_spinner=False)
def get_hand_overlay(bounds: tuple[float, float, float, float], boundary_wkb: bytes):
    return fetch_hand(bounds, from_wkb(boundary_wkb))


@st.cache_data(ttl=604_800, show_spinner=False)
def get_soil_summary(bounds: tuple[float, float, float, float], boundary_wkb: bytes):
    return fetch_soilgrids(bounds, from_wkb(boundary_wkb))


@st.cache_data(ttl=86_400, show_spinner=False)
def get_rainfall_summary(latitude: float, longitude: float):
    return fetch_rainfall(latitude, longitude)


@st.cache_data(max_entries=8, show_spinner=False)
def get_flow_analysis(content: bytes, boundary_wkb: bytes, visual_wkb: bytes):
    return analyze_flow(content, from_wkb(boundary_wkb), from_wkb(visual_wkb))


@st.cache_data(max_entries=24, show_spinner=False)
def get_terrain_analysis(content: bytes, boundary_wkb: bytes, visual_wkb: bytes, mode: str):
    return analyze_dem(content, from_wkb(boundary_wkb), mode, from_wkb(visual_wkb))


@st.cache_data(show_spinner=False)
def map_style_url(filename: str, opacity: float) -> str:
    style = json.loads((ROOT / "static" / filename).read_text(encoding="utf-8"))
    for layer in style.get("layers", []):
        if layer.get("type") == "raster":
            layer.setdefault("paint", {})["raster-opacity"] = opacity
    encoded = base64.b64encode(json.dumps(style, separators=(",", ":")).encode("utf-8")).decode("ascii")
    return f"data:application/json;base64,{encoded}"


def choose_index(options: list[str], guess: str | None) -> int:
    return options.index(guess) if guess in options else 0


def boundary_feature(frame: gpd.GeoDataFrame, index: int, label: str) -> dict:
    one = gpd.GeoDataFrame({"lote": [label]}, geometry=[frame.iloc[index].geometry], crs=frame.crs)
    return json.loads(one.to_json())["features"][0]


def geometry_feature(geometry, label: str) -> dict:
    one = gpd.GeoDataFrame({"lote": [label]}, geometry=[geometry], crs="EPSG:4326")
    return json.loads(one.to_json())["features"][0]


def lot_view(geometry) -> pdk.ViewState:
    west, south, east, north = geometry.bounds
    effective_span = max(max(east - west, 0.0002), max(north - south, 0.0002) * 1.6)
    zoom = max(8.0, min(17.0, math.log2(360 / effective_span) + 0.2))
    return pdk.ViewState(longitude=(west + east) / 2, latitude=(south + north) / 2, zoom=zoom)


def buffered_geometry(geometry, distance_m: float = 500):
    source = gpd.GeoSeries([geometry], crs="EPSG:4326")
    metric_crs = source.estimate_utm_crs()
    return source.to_crs(metric_crs).buffer(distance_m).to_crs("EPSG:4326").iloc[0]


def canonical_points(points: gpd.GeoDataFrame, label: str, diagnostics: dict) -> pd.DataFrame:
    result = pd.DataFrame(points.drop(columns="geometry")).copy()
    result["elevacion_m"] = result["elev_m_raw"]
    result["dentro_poligono"] = diagnostics["inside_mask"].to_numpy()
    defaults = {"fuente": "monitor", "campana": "", "cultivo": "", "lote_sima": label}
    for column, value in defaults.items():
        if column not in result:
            result[column] = value
    return result


def format_number(value: float | int, decimals: int = 1) -> str:
    return f"{value:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def color_legend(title: str, items: list[tuple[str, str]], note: str) -> None:
    swatches = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:.35rem;margin:.2rem .9rem .2rem 0">'
        f'<span style="width:16px;height:10px;border-radius:2px;background:{color};border:1px solid #94a3b8"></span>{label}</span>'
        for label, color in items
    )
    st.markdown(
        f'<div style="border:1px solid #dbe2ea;border-radius:8px;padding:.65rem .8rem;background:#fff">'
        f'<strong>{title}</strong><div style="margin-top:.25rem">{swatches}</div>'
        f'<div style="color:#64748b;font-size:.82rem;margin-top:.2rem">{note}</div></div>',
        unsafe_allow_html=True,
    )


def render_download_bundle(bundle: dict) -> None:
    downloads = st.columns(2)
    downloads[0].download_button(
        "Informe HTML completo", bundle["html"], bundle["html_name"], "text/html", width="stretch", on_click="ignore"
    )
    downloads[1].download_button(
        "Paquete completo ZIP", bundle["package"], bundle["zip_name"], "application/zip", width="stretch", on_click="ignore"
    )
    downloads[0].download_button(
        "Mapa principal PNG", bundle["main_map"], bundle["map_name"], "image/png", width="stretch", on_click="ignore"
    )
    downloads[1].download_button(
        "Límites normalizados", bundle["boundaries"], "limites_normalizados.geojson", "application/geo+json", width="stretch", on_click="ignore"
    )
    downloads[0].download_button(
        "Resumen JSON", bundle["summary"], bundle["summary_name"], "application/json", width="stretch", on_click="ignore"
    )
    if bundle.get("dem") is not None:
        downloads[1].download_button(
            "DEM abierto GeoTIFF", bundle["dem"], bundle["dem_name"], "image/tiff", width="stretch", on_click="ignore"
        )
    st.caption(
        f"Contenido del informe: {bundle['product_count']} mapas y gráficos. "
        f"El ZIP contiene el informe completo y {bundle['file_count']} archivos derivados."
    )


st.title("Diagnóstico topográfico")
st.caption("Cargar límites · integrar monitor si existe · completar el relieve con datos abiertos sin clave")

with st.sidebar:
    st.header("1 · Límites")
    uploaded_boundary = st.file_uploader(
        "Archivo de lotes",
        type=BOUNDARY_EXTENSIONS,
        help="GeoJSON/JSON, KML/KMZ, GeoPackage o Shapefile completo dentro de un ZIP.",
    )
    fallback_epsg = st.text_input("CRS si el archivo no lo declara", value="EPSG:4326")

if not uploaded_boundary:
    st.info("Carga de un archivo de límites para iniciar un análisis nuevo. La aplicación pública no contiene lotes precargados.")
    start_a, start_b, start_c = st.columns(3)
    start_a.subheader("1 · Delimitar")
    start_a.write("Selección del lote y verificación de superficie, geometría y sistema de coordenadas.")
    start_b.subheader("2 · Completar")
    start_b.write("Integración de datos de monitor o descarga automática de un DEM abierto, sin cuenta ni clave.")
    start_c.subheader("3 · Evaluar")
    start_c.write("Comparación de relieve, pendiente, cobertura y calidad para evaluar caminos y escurrimiento.")
    st.stop()

try:
    boundaries = parse_boundaries(uploaded_boundary.name, uploaded_boundary.getvalue(), fallback_epsg)
except Exception as error:
    st.error(f"No se pudo leer el archivo de límites: {error}")
    st.stop()

attribute_columns = [column for column in boundaries.columns if column != "geometry"]
preferred_names = ["nombre", "name", "lote", "field", "id", "__field_id"]
name_guess = next((column for column in preferred_names if column in attribute_columns), "__field_id")
with st.sidebar:
    name_column = st.selectbox("Columna con el nombre", attribute_columns, index=choose_index(attribute_columns, name_guess))

boundaries = boundaries.reset_index(drop=True).copy()
labels = boundaries[name_column].fillna(boundaries["__field_id"]).astype(str).tolist()
boundaries["__display_name"] = labels
with st.sidebar:
    analysis_scale = st.radio("Escala de análisis", ["Un lote", "Grupo / campo"], horizontal=True)
    if analysis_scale == "Un lote":
        selected_lot = st.selectbox("Lote a analizar", labels)
        selected_indices = [labels.index(selected_lot)]
        selected_label = selected_lot
    else:
        group_columns = [column for column in attribute_columns if column not in {name_column, "__field_id"}]
        grouping = st.selectbox("Forma de agrupar", ["Selección manual"] + group_columns)
        if grouping == "Selección manual":
            selected_names = st.multiselect("Lotes del grupo", labels, default=labels[: min(3, len(labels))])
            if not selected_names:
                st.warning("La selección del grupo necesita al menos un lote.")
                st.stop()
            selected_indices = [index for index, label in enumerate(labels) if label in selected_names]
            selected_label = f"Grupo de {len(selected_indices)} lotes"
        else:
            group_values = boundaries[grouping].fillna("Sin dato").astype(str)
            selected_group = st.selectbox("Grupo / campo a analizar", sorted(group_values.unique()))
            selected_indices = boundaries.index[group_values == selected_group].tolist()
            selected_label = f"{grouping}: {selected_group}"

selected_frame = boundaries.iloc[selected_indices].copy()
selected_geometry = union_all(selected_frame.geometry.to_numpy())
visual_geometry = buffered_geometry(selected_geometry, 500)
selected_feature = geometry_feature(selected_geometry, selected_label)
display_frame = selected_frame[["__display_name", "geometry"]].rename(columns={"__display_name": "lote"})
selected_feature_collection = json.loads(display_frame.to_json())
area_ha = geometry_area_ha(selected_feature["geometry"])
boundary_signature = selected_geometry.wkb_hex

with st.sidebar:
    st.header("2 · Monitor opcional")
    uploaded_monitors = st.file_uploader(
        "Datos de maquinaria",
        type=MONITOR_EXTENSIONS,
        accept_multiple_files=True,
        help="GeoJSON, GeoPackage, KML/KMZ, Shapefile dentro de ZIP o selección conjunta de SHP, SHX, DBF, PRJ y CPG.",
    )
    monitor_fallback_epsg = st.text_input("CRS si el monitor no lo declara", value="EPSG:4326")

monitor_table = None
if uploaded_monitors:
    tables = []
    shapefile_uploads = [
        uploaded for uploaded in uploaded_monitors if Path(uploaded.name).suffix.lower() in SHAPEFILE_COMPONENT_EXTENSIONS
    ]
    if shapefile_uploads:
        component_files = tuple((uploaded.name, uploaded.getvalue()) for uploaded in shapefile_uploads)
        table = parse_monitor_shapefile(component_files, monitor_fallback_epsg).copy()
        table["__archivo_origen"] = "Shapefile"
        tables.append(table)
    for uploaded in uploaded_monitors:
        if Path(uploaded.name).suffix.lower() in SHAPEFILE_COMPONENT_EXTENSIONS:
            continue
        table = parse_monitor(uploaded.name, uploaded.getvalue(), monitor_fallback_epsg).copy()
        table["__archivo_origen"] = uploaded.name
        tables.append(table)
    monitor_table = gpd.GeoDataFrame(pd.concat(tables, ignore_index=True, sort=False), geometry="geometry", crs="EPSG:4326")

points = None
diagnostics = None
canonical = None
monitor_stats = None
mapping_error = None
if monitor_table is not None:
    columns = [str(column) for column in monitor_table.columns if str(column) != "geometry"]
    elev_guess = column_guess(columns, ["elevacion", "elevation", "altura", "altitude", "cota", "elev"])
    with st.sidebar:
        elevation_column = st.selectbox("Columna de elevación", columns, index=choose_index(columns, elev_guess))
        optional_options = ["Sin seleccionar"] + columns
        source_column = st.selectbox("Fuente / máquina", optional_options)
        pass_column = st.selectbox("Fecha / pasada", optional_options)
    try:
        points = normalize_monitor_vector(
            monitor_table,
            elevation_column,
            {"fuente": None if source_column == "Sin seleccionar" else source_column, "pasada": None if pass_column == "Sin seleccionar" else pass_column},
        )
        diagnostics = monitor_diagnostics(points, selected_geometry)
        canonical = canonical_points(points, selected_label, diagnostics)
        monitor_stats = monitor_dashboard(points, selected_geometry) if diagnostics["rows_inside"] else None
    except Exception as error:
        mapping_error = str(error)

with st.sidebar:
    st.header("3 · Mapa y relieve")
    basemap_label = st.selectbox("Mapa base", list(BASEMAPS))
    basemap_opacity = st.slider("Opacidad del mapa base", 20, 100, 100, 5, format="%d %%")
    show_context = st.checkbox("Mostrar otros lotes", value=False)
    visual_buffer_km = 0.5
    st.caption("Contexto visual común · 500 m alrededor de la selección")
    dem_zoom = st.select_slider(
        "Detalle del DEM",
        options=[11, 12, 13],
        value=12,
        format_func=lambda value: {11: "regional · ~38 m", 12: "equilibrado · ~19 m", 13: "detalle · ~10 m"}[value],
    )
    open_bounds = expanded_bounds(tuple(selected_geometry.bounds), visual_buffer_km)
    tile_count = len(terrain_tile_indices(open_bounds, dem_zoom))
    st.caption(f"Fuente sin clave · {tile_count} tesela(s) públicas")
    if st.button("Analizar con DEM abierto", type="primary", width="stretch"):
        try:
            with st.spinner("Descargando y uniendo teselas abiertas…"):
                dem = get_open_dem(open_bounds, dem_zoom)
                st.session_state["dem_result"] = dem
                st.session_state["dem_boundary_signature"] = boundary_signature
                st.session_state["analytic_layer_selector"] = "Pendiente DEM"
        except Exception as error:
            st.error(f"No se pudo obtener el DEM abierto: {error}")
    st.header("4 · Contexto abierto")
    st.caption("Caminos y cursos de agua registrados en OpenStreetMap")
    context_bounds = expanded_bounds(tuple(selected_geometry.bounds), visual_buffer_km)
    if st.button("Cargar contexto vial e hídrico", width="stretch"):
        try:
            with st.spinner("Consultando infraestructura y drenajes abiertos…"):
                osm_context_result = get_open_context(context_bounds)
                st.session_state["osm_context"] = osm_context_result
                st.session_state["osm_boundary_signature"] = boundary_signature
                st.session_state["show_osm_context"] = True
        except Exception as error:
            st.error(f"No se pudo obtener el contexto abierto: {error}")
    st.header("5 · Radar satelital")
    radar_polarization = st.selectbox(
        "Polarización Sentinel-1",
        ["VV · humedad y rugosidad", "VH · vegetación y volumen"],
    )
    radar_code = radar_polarization[:2]
    radar_bounds = expanded_bounds(tuple(selected_geometry.bounds), 0.5)
    if st.button("Cargar radar NASA OPERA", width="stretch"):
        try:
            with st.spinner("Consultando la escena radar corregida más reciente…"):
                radar_result = get_radar_overlay(radar_bounds, radar_code)
                st.session_state["radar_result"] = radar_result
                st.session_state["radar_boundary_signature"] = boundary_signature
                st.session_state["radar_polarization"] = radar_code
                st.session_state["analytic_layer_selector"] = f"Radar Sentinel-1 {radar_code}"
        except Exception as error:
            st.error(f"No se pudo obtener la escena radar: {error}")
    st.header("6 · Cuenca, suelo y lluvia")
    st.caption("Fuentes globales abiertas para completar el riesgo de erosión")
    environment_bounds = expanded_bounds(tuple(selected_geometry.bounds), visual_buffer_km)
    if st.button("Calcular altura sobre el drenaje", width="stretch"):
        try:
            with st.spinner("Consultando el modelo HAND de 30 m…"):
                hand_result = get_hand_overlay(environment_bounds, selected_geometry.wkb)
                st.session_state["hand_result"] = hand_result
                st.session_state["hand_boundary_signature"] = boundary_signature
                st.session_state["analytic_layer_selector"] = "Altura sobre el drenaje HAND"
        except Exception as error:
            st.error(f"No se pudo obtener HAND: {error}")
    if st.button("Caracterizar suelo superficial", width="stretch"):
        try:
            with st.spinner("Recortando arcilla, limo, arena y carbono de SoilGrids…"):
                soil_result = get_soil_summary(environment_bounds, selected_geometry.wkb)
                st.session_state["soil_result"] = soil_result
                st.session_state["soil_boundary_signature"] = boundary_signature
        except Exception as error:
            st.error(f"No se pudo obtener SoilGrids: {error}")
    if st.button("Resumir lluvia histórica", width="stretch"):
        try:
            centroid = selected_geometry.centroid
            with st.spinner("Resumiendo diez años completos de lluvia de reanálisis…"):
                rainfall_result = get_rainfall_summary(float(centroid.y), float(centroid.x))
                st.session_state["rainfall_result"] = rainfall_result
                st.session_state["rainfall_boundary_signature"] = boundary_signature
        except Exception as error:
            st.error(f"No se pudo obtener la lluvia histórica: {error}")

available_dem = (
    st.session_state.get("dem_result")
    if st.session_state.get("dem_boundary_signature") == boundary_signature
    else None
)
osm_context = (
    st.session_state.get("osm_context")
    if st.session_state.get("osm_boundary_signature") == boundary_signature
    else None
)
radar = (
    st.session_state.get("radar_result")
    if st.session_state.get("radar_boundary_signature") == boundary_signature
    and st.session_state.get("radar_polarization") == radar_code
    else None
)
hand = (
    st.session_state.get("hand_result")
    if st.session_state.get("hand_boundary_signature") == boundary_signature
    else None
)
soil = (
    st.session_state.get("soil_result")
    if st.session_state.get("soil_boundary_signature") == boundary_signature
    else None
)
rainfall = (
    st.session_state.get("rainfall_result")
    if st.session_state.get("rainfall_boundary_signature") == boundary_signature
    else None
)
terrain_elevation = get_terrain_analysis(available_dem.content, selected_geometry.wkb, visual_geometry.wkb, "Elevación") if available_dem else None
flow = None
flow_error = None
if available_dem:
    try:
        flow = get_flow_analysis(available_dem.content, selected_geometry.wkb, visual_geometry.wkb)
    except Exception as error:
        flow_error = str(error)
quality_label = "Sin monitor"
quality_text = "El diagnóstico se apoyará en el DEM abierto cuando esté disponible."
if diagnostics:
    quality_label, quality_text = quality_grade(diagnostics, len(monitor_table), area_ha)

st.subheader(selected_label)
source_label = "Monitor + DEM abierto" if monitor_stats and available_dem else "Monitor" if monitor_stats else "DEM abierto" if available_dem else "Sólo límites"
main_relief = terrain_elevation.relief_m if terrain_elevation else monitor_stats["relief_p01_p99"] if monitor_stats else None
main_slope = terrain_elevation.mean_slope_percent if terrain_elevation else monitor_stats["plane_slope_percent"] if monitor_stats else None
headline = st.columns(5)
headline[0].metric("Superficie", f"{format_number(area_ha)} ha")
headline[1].metric("Fuente analítica", source_label)
headline[2].metric("Relieve robusto", f"{format_number(main_relief, 2)} m" if main_relief is not None else "Pendiente")
headline[3].metric("Pendiente", f"{format_number(main_slope, 1)} %" if main_slope is not None else "Pendiente")
headline[4].metric("Calidad monitor", quality_label)

if mapping_error:
    st.error(f"El monitor no pudo interpretarse: {mapping_error}")
if diagnostics and diagnostics["suspicious_coordinate_copy"]:
    st.error("La elevación elegida parece duplicar una coordenada. Es necesario corregir el mapeo antes de interpretar el relieve.")
if flow_error:
    st.warning(f"El DEM se cargó, pero no se pudo calcular el escurrimiento: {flow_error}")

tab_summary, tab_map, tab_environment, tab_quality, tab_downloads = st.tabs(
    ["Resumen ejecutivo", "Mapa y capas", "Indicadores", "Calidad y supuestos", "Descargas"]
)

with tab_summary:
    st.subheader("Qué se puede afirmar hoy")
    answers = [
        {
            "Pregunta": "¿Cuál es la magnitud del relieve?",
            "Respuesta": f"{format_number(main_relief, 2)} m entre percentiles robustos." if main_relief is not None else "Falta monitor o DEM.",
            "Estado": "Respondida" if main_relief is not None else "Falta dato",
        },
        {
            "Pregunta": "¿Hacia dónde cae el terreno en general?",
            "Respuesta": (
                f"Hacia {flow.downslope_direction} ({flow.downslope_bearing:.0f}°), según la tendencia general del DEM."
                if flow else f"Hacia {monitor_stats['downslope_direction']} ({monitor_stats['downslope_bearing']:.0f}°), según el plano ajustado al monitor."
                if monitor_stats else "Se responde al descargar el DEM o incorporar un monitor."
            ),
            "Estado": "Respondida" if flow or monitor_stats else "Falta dato",
        },
        {
            "Pregunta": "¿Dónde la pendiente puede agravar erosión?",
            "Respuesta": (
                f"{format_number(terrain_elevation.area_over_2_percent)} % del lote supera 2 % y "
                f"{format_number(terrain_elevation.area_over_5_percent)} % supera 5 %."
                if terrain_elevation else "Hace falta descargar el DEM abierto."
            ),
            "Estado": "Respondida" if terrain_elevation else "Falta dato",
        },
        {
            "Pregunta": "¿Por dónde se concentra el agua?",
            "Respuesta": (
                f"La mayor convergencia dentro del lote recibe ~{format_number(flow.max_contributing_area_ha, 2)} ha; "
                f"la capa azul muestra los corredores de mayor aporte."
                if flow else "Se responde con la capa de acumulación al descargar el DEM abierto."
            ),
            "Estado": "Preliminar" if flow else "Falta dato",
        },
        {
            "Pregunta": "¿Ya se puede definir un camino?",
            "Respuesta": (
                "Ya se pueden descartar corredores que crucen flujo concentrado o pendientes sensibles. La traza final debe sumar suelo, accesos y control de campo."
                if flow and terrain_elevation else "Completar el DEM para evaluar pendiente y concentración de flujo."
            ),
            "Estado": "Criterio definido" if flow and terrain_elevation else "Falta dato",
        },
        {
            "Pregunta": "¿Qué sectores están más próximos verticalmente al drenaje?",
            "Respuesta": (
                f"HAND estima {format_number(hand.area_below_2m_percent)} % del lote a menos de 2 m verticales del drenaje más cercano y una mediana de {format_number(hand.median_m, 2)} m."
                if hand else "La capa HAND de 30 m permite reconocer bajos conectados al drenaje, aun cuando la elevación absoluta cambie poco."
            ),
            "Estado": "Respondida" if hand else "Consulta disponible",
        },
        {
            "Pregunta": "¿Qué aporta el suelo al riesgo de erosión?",
            "Respuesta": (
                f"{soil.texture_label}. {soil.erosion_screening}. Resolución global de {soil.resolution_m} m."
                if soil else "SoilGrids permite contextualizar textura y carbono superficial; su resolución sirve para diagnóstico regional, no para diseñar una obra puntual."
            ),
            "Estado": "Contexto disponible" if soil else "Consulta disponible",
        },
        {
            "Pregunta": "¿Cuál es la presión histórica de lluvia?",
            "Respuesta": (
                f"Promedio anual de {format_number(rainfall.annual_mean_mm, 0)} mm, {format_number(rainfall.heavy_days_mean, 1)} días por año con al menos 30 mm y máximo diario de {format_number(rainfall.max_daily_mm, 1)} mm entre {rainfall.start_date[:4]} y {rainfall.end_date[:4]}."
                if rainfall else "La serie ERA5-Land permite dimensionar frecuencia de días húmedos y eventos diarios fuertes en diez años completos."
            ),
            "Estado": "Respondida" if rainfall else "Consulta disponible",
        },
        {
            "Pregunta": "¿Qué infraestructura o drenajes registrados atraviesan el entorno?",
            "Respuesta": (
                f"OpenStreetMap registra {osm_context.road_count} caminos y {osm_context.waterway_count} cursos de agua en el área de contexto."
                if osm_context else "La consulta opcional de contexto abierto permite incorporar caminos y cursos registrados."
            ),
            "Estado": "Respondida" if osm_context else "Consulta disponible",
        },
        {
            "Pregunta": "¿Qué aporta el radar Sentinel-1?",
            "Respuesta": (
                f"Escena NASA OPERA {radar.polarization} del {radar.acquisition_date}. Permite observar contrastes de humedad, rugosidad o vegetación; una anomalía requiere comparación temporal."
                if radar else "La capa opcional NASA OPERA permite observar humedad, rugosidad y vegetación aun con nubosidad."
            ),
            "Estado": "Contexto disponible" if radar else "Consulta disponible",
        },
    ]
    st.dataframe(pd.DataFrame(answers), hide_index=True, width="stretch")
    if terrain_elevation and terrain_elevation.area_over_5_percent > 20:
        st.warning("Una proporción importante presenta pendiente mayor a 5 %. Conviene priorizar trazas por divisorias o pendientes suaves y evitar la concentración de escorrentía.")
    elif terrain_elevation:
        st.info("Predominan pendientes moderadas o suaves. La comparación de cualquier traza con la acumulación de flujo resulta necesaria antes de mover caminos.")
    elif monitor_stats:
        st.info("El monitor permite leer la macro-topografía, pero no reemplaza un DEM continuo para pendiente y drenaje.")
    else:
        st.info("La descarga del DEM abierto o la incorporación de un monitor permiten transformar el límite en un diagnóstico.")

    chart_left, chart_right = st.columns(2)
    if monitor_stats:
        with chart_left:
            st.markdown("#### Cómo se distribuyen las cotas del monitor")
            chart = alt.Chart(monitor_stats["histogram"]).mark_bar(color="#2878B5").encode(
                x=alt.X("cota_m:Q", title="Elevación (m)", bin=alt.Bin(maxbins=24)),
                y=alt.Y("sum(puntos):Q", title="Puntos", scale=alt.Scale(zero=True)),
                tooltip=[alt.Tooltip("cota_m:Q", format=".2f"), alt.Tooltip("puntos:Q", format=",")],
            ).properties(height=280)
            st.altair_chart(chart, width="stretch")
        with chart_right:
            st.markdown(f"#### Perfil medio hacia {monitor_stats['downslope_direction']}")
            chart = alt.Chart(monitor_stats["profile"]).mark_line(point=True, color="#D97706").encode(
                x=alt.X("distancia_m:Q", title="Distancia sobre el eje de caída (m)"),
                y=alt.Y("cota_mediana_m:Q", title="Cota mediana (m)", scale=alt.Scale(zero=False)),
                tooltip=[alt.Tooltip("distancia_m:Q", format=".0f"), alt.Tooltip("cota_mediana_m:Q", format=".2f")],
            ).properties(height=280)
            st.altair_chart(chart, width="stretch")
    elif terrain_elevation:
        elevation_hist = pd.DataFrame(terrain_elevation.elevation_histogram, columns=["cota_m", "celdas"])
        slope_classes = pd.DataFrame(terrain_elevation.slope_classes, columns=["clase", "porcentaje"])
        with chart_left:
            st.markdown("#### Distribución de elevación del DEM")
            chart = alt.Chart(elevation_hist).mark_bar(color="#2878B5").encode(
                x=alt.X("cota_m:Q", title="Elevación (m)", bin=alt.Bin(maxbins=24)),
                y=alt.Y("sum(celdas):Q", title="Celdas", scale=alt.Scale(zero=True)),
            ).properties(height=280)
            st.altair_chart(chart, width="stretch")
        with chart_right:
            st.markdown("#### Superficie por clase de pendiente")
            chart = alt.Chart(slope_classes).mark_bar(color="#D97706").encode(
                x=alt.X("porcentaje:Q", title="Superficie (%)", scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("clase:N", title=None, sort=None),
                tooltip=[alt.Tooltip("porcentaje:Q", format=".1f")],
            ).properties(height=280)
            st.altair_chart(chart, width="stretch")

with tab_environment:
    indicators = []
    if hand:
        indicators.append({"Indicador": "HAND bajo 2 m", "Resultado": f"{format_number(hand.area_below_2m_percent)} %", "Alcance": "Calculado dentro de la selección"})
    if soil:
        indicators.append({"Indicador": "Textura superficial", "Resultado": soil.texture_label, "Alcance": "SoilGrids 0 a 5 cm"})
    if rainfall:
        indicators.append({"Indicador": "Lluvia anual media", "Resultado": f"{format_number(rainfall.annual_mean_mm, 0)} mm", "Alcance": f"{rainfall.start_date[:4]} a {rainfall.end_date[:4]}"})
    if indicators:
        st.dataframe(pd.DataFrame(indicators), hide_index=True, width="stretch")
    else:
        st.info("HAND, suelo superficial y lluvia histórica quedan disponibles como consultas opcionales.")
    st.caption("Las clases de pendiente y HAND se consultan geográficamente desde Mapa y capas, con una leyenda común y 500 m de contexto visual.")

with tab_map:
    analytic_options = ["Sólo mapa base"]
    if monitor_stats:
        analytic_options += ["Elevación del monitor", "Puntos crudos del monitor"]
    if available_dem:
        analytic_options += ["Acumulación de flujo", "Elevación DEM", "Pendiente DEM", "Sombreado DEM"]
    if radar:
        analytic_options += [f"Radar Sentinel-1 {radar.polarization}"]
    if hand:
        analytic_options += ["Altura sobre el drenaje HAND"]
    default_layer = "Acumulación de flujo" if flow else "Pendiente DEM" if available_dem else "Elevación del monitor" if monitor_stats else "Sólo mapa base"
    map_controls = st.columns([2, 1, 1, 1])
    if st.session_state.get("analytic_layer_selector") not in analytic_options:
        st.session_state["analytic_layer_selector"] = default_layer
    analytic_layer = map_controls[0].selectbox(
        "Capa analítica", analytic_options, key="analytic_layer_selector"
    )
    show_boundary = map_controls[1].checkbox("Límite", value=True)
    marker_available = bool(monitor_stats or terrain_elevation)
    show_markers = map_controls[2].checkbox("Altos y bajos", value=False, disabled=not marker_available)
    if not osm_context:
        st.session_state["show_osm_context"] = False
    elif "show_osm_context" not in st.session_state:
        st.session_state["show_osm_context"] = True
    show_osm_context = map_controls[3].checkbox(
        "Caminos y agua", disabled=not bool(osm_context), key="show_osm_context"
    )

    opacity_controls = st.columns(4)
    analytic_opacity = opacity_controls[0].slider(
        "Opacidad analítica", 0, 100, 78, 5, format="%d %%", disabled=analytic_layer == "Sólo mapa base"
    ) / 100
    boundary_opacity = opacity_controls[1].slider("Opacidad del límite", 0, 100, 100, 5, format="%d %%") / 100
    context_opacity = opacity_controls[2].slider(
        "Opacidad de contexto", 0, 100, 70, 5, format="%d %%", disabled=not (show_context or show_osm_context)
    ) / 100
    marker_opacity = opacity_controls[3].slider(
        "Opacidad de puntos", 0, 100, 100, 5, format="%d %%", disabled=not show_markers
    ) / 100

    layers = []
    active_terrain = None
    if analytic_layer == "Elevación del monitor" and points is not None:
        surface = analyze_monitor(points, selected_geometry)
        layers.append(pdk.Layer("BitmapLayer", id="elevacion_monitor", image=String(surface.image_data_url), bounds=list(surface.bounds_wgs84), opacity=analytic_opacity))
    elif analytic_layer == "Puntos crudos del monitor" and canonical is not None:
        sample = display_sample(canonical, maximum=4_000)
        layers.append(pdk.Layer(
            "ScatterplotLayer", sample, id="puntos_monitor", get_position="[lon, lat]", radius_units="pixels", get_radius=1.2,
            get_fill_color="[color_r, color_g, color_b, 190]", opacity=analytic_opacity, pickable=True,
        ))
    elif analytic_layer == "Acumulación de flujo" and flow:
        layers.append(pdk.Layer("BitmapLayer", id="acumulacion_flujo", image=String(flow.image_data_url), bounds=list(flow.bounds_wgs84), opacity=analytic_opacity))
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            [{"lon": flow.outlet_lon, "lat": flow.outlet_lat, "tipo": "Mayor convergencia", "detalle": f"Aporte estimado: {format_number(flow.max_contributing_area_ha, 2)} ha", "color": [8, 81, 156, round(245 * marker_opacity)]}],
            id="salida_flujo",
            get_position="[lon, lat]", get_fill_color="color", get_radius=1, radius_units="meters",
            radius_min_pixels=5, radius_max_pixels=5, stroked=True,
            get_line_color=[255, 255, 255, round(255 * marker_opacity)], line_width_min_pixels=2, pickable=True,
        ))
    elif analytic_layer in {"Elevación DEM", "Pendiente DEM", "Sombreado DEM"}:
        mode = {"Elevación DEM": "Elevación", "Pendiente DEM": "Pendiente", "Sombreado DEM": "Sombreado"}[analytic_layer]
        active_terrain = get_terrain_analysis(available_dem.content, selected_geometry.wkb, visual_geometry.wkb, mode)
        layers.append(pdk.Layer("BitmapLayer", id=f"dem_{mode.lower()}", image=String(active_terrain.image_data_url), bounds=list(active_terrain.bounds_wgs84), opacity=analytic_opacity))
    elif radar and analytic_layer == f"Radar Sentinel-1 {radar.polarization}":
        layers.append(pdk.Layer("BitmapLayer", id="radar_sentinel1", image=String(radar.image_data_url), bounds=list(radar.bounds_wgs84), opacity=analytic_opacity))
    elif hand and analytic_layer == "Altura sobre el drenaje HAND":
        layers.append(pdk.Layer("BitmapLayer", id="hand", image=String(hand.image_data_url), bounds=list(hand.bounds_wgs84), opacity=analytic_opacity))
    if show_context and len(boundaries) > 1:
        context = boundaries.drop(index=selected_frame.index)
        layers.append(pdk.Layer(
            "GeoJsonLayer", json.loads(context.to_json()), id="otros_lotes", stroked=True, filled=True,
            get_fill_color=[100, 116, 139, round(35 * context_opacity)],
            get_line_color=[71, 85, 105, round(210 * context_opacity)], line_width_min_pixels=1,
        ))
    if show_osm_context and osm_context:
        layers.append(pdk.Layer(
            "GeoJsonLayer", osm_context.roads, id="caminos_osm", stroked=True, filled=False,
            get_line_color="properties.color", get_line_width="properties.width", line_width_units="pixels",
            opacity=context_opacity, pickable=True,
        ))
        layers.append(pdk.Layer(
            "GeoJsonLayer", osm_context.waterways, id="agua_osm", stroked=True, filled=False,
            get_line_color="properties.color", get_line_width="properties.width", line_width_units="pixels",
            opacity=context_opacity, pickable=True,
        ))
    if show_boundary:
        layers.append(pdk.Layer(
            "GeoJsonLayer", selected_feature_collection, id="limites_seleccionados", stroked=True, filled=True,
            get_fill_color=[37, 99, 235, round(22 * boundary_opacity)],
            get_line_color=[15, 74, 117, round(245 * boundary_opacity)], line_width_min_pixels=3, pickable=True,
        ))
    if show_markers:
        if monitor_stats:
            marker_data = monitor_stats["markers"].copy()
            marker_data["detalle"] = marker_data["elevacion_m"].map(lambda value: f"{value:.2f} m · monitor")
            marker_records = marker_data.to_dict("records")
        else:
            marker_records = [dict(item) for item in terrain_elevation.markers]
        for item in marker_records:
            item["color"] = [*item["color"][:3], round(item["color"][3] * marker_opacity)]
        layers.append(pdk.Layer(
            "ScatterplotLayer", marker_records, id="altos_bajos", get_position="[lon, lat]", get_fill_color="color",
            get_radius=1, radius_units="meters", radius_min_pixels=6, radius_max_pixels=6, stroked=True,
            get_line_color=[255, 255, 255, round(255 * marker_opacity)], line_width_min_pixels=2, pickable=True,
        ))
    st.pydeck_chart(
        pdk.Deck(
            layers=layers, initial_view_state=lot_view(visual_geometry),
            map_style=map_style_url(BASEMAPS[basemap_label], basemap_opacity / 100),
            map_provider="maplibre",
            tooltip={"html": "<b>{tipo}</b><br>{detalle}"},
        ), width="stretch", height=620,
    )
    if analytic_layer in {"Elevación DEM", "Elevación del monitor", "Puntos crudos del monitor"}:
        legend_low = terrain_elevation.elevation_p02_m if analytic_layer == "Elevación DEM" else surface.elevation_p01_m if analytic_layer == "Elevación del monitor" else monitor_stats["elevation_p01"]
        legend_high = terrain_elevation.elevation_p98_m if analytic_layer == "Elevación DEM" else surface.elevation_p99_m if analytic_layer == "Elevación del monitor" else monitor_stats["elevation_p99"]
        color_legend(
            "Elevación",
            [(f"Baja · {format_number(legend_low, 2)} m", "#2b83ba"), ("Intermedia", "#d9ef8b"), (f"Alta · {format_number(legend_high, 2)} m", "#d7191c")],
            "Escala recortada por percentiles para reducir el efecto de valores extremos.",
        )
    elif analytic_layer == "Pendiente DEM":
        color_legend(
            "Pendiente del terreno",
            [("0 a 1 %", "#c6e2bb"), ("1 a 2 %", "#ffeda0"), ("2 a 5 %", "#feb24c"), ("5 a 10 %", "#f03b20"), ("Más de 10 %", "#756bb1")],
            "Clases discretas orientadas a lectura agronómica. Los tonos rojos y violetas indican mayor sensibilidad.",
        )
    elif analytic_layer == "Sombreado DEM":
        color_legend(
            "Relieve sombreado",
            [("Ladera en sombra", "#252525"), ("Tono medio", "#969696"), ("Ladera iluminada", "#f7f7f7")],
            "Iluminación simulada desde el noroeste. La capa muestra forma del relieve, no elevación absoluta.",
        )
    elif analytic_layer == "Acumulación de flujo" and flow:
        color_legend(
            "Concentración topográfica de flujo",
            [(f"Inicio visible · {format_number(flow.display_low_ha, 2)} ha", "#9ecae1"), ("Concentración media", "#3182bd"), (f"Alta · {format_number(flow.display_high_ha, 2)} ha", "#08519c")],
            "Área aportante estimada con D8. El punto marcado representa la mayor convergencia dentro del lote.",
        )
    elif radar and analytic_layer == f"Radar Sentinel-1 {radar.polarization}":
        radar_note = (
            "En VV, las superficies lisas y el agua calma suelen verse oscuras; humedad, rugosidad y estructuras pueden aumentar el retorno."
            if radar.polarization == "VV"
            else "En VH, los retornos más claros suelen asociarse con mayor dispersión volumétrica de vegetación. El agua calma suele verse oscura."
        )
        color_legend(
            f"Radar Sentinel-1 {radar.polarization} · {radar.acquisition_date}",
            [("Retorno bajo", "#151515"), ("Retorno medio", "#888888"), ("Retorno alto", "#f2f2f2")],
            radar_note,
        )
    elif hand and analytic_layer == "Altura sobre el drenaje HAND":
        color_legend(
            "Altura sobre el drenaje más cercano",
            [("0 a 1 m", "#08519c"), ("1 a 2 m", "#3182bd"), ("2 a 5 m", "#6baed6"), ("5 a 10 m", "#c6dbef"), ("Más de 10 m", "#fee391")],
            "HAND de 30 m. Los azules oscuros indican mayor conexión topográfica potencial con el drenaje.",
        )
    else:
        color_legend("Referencias", [("Límite seleccionado", "#0f4a75")], "Cartografía de contexto sin capa analítica activa.")
    st.caption(f"Mapa base: {basemap_label}. Fuente analítica: {source_label}.")

with tab_quality:
    st.subheader("Qué tan confiable es el análisis")
    checks = [
        {"Control": "Geometría", "Resultado": f"{selected_geometry.geom_type} válido" if selected_geometry.is_valid else "Geometría inválida", "Estado": "OK" if selected_geometry.is_valid else "Revisar"},
        {"Control": "CRS interno", "Resultado": "WGS84 / EPSG:4326", "Estado": "OK"},
        {"Control": "DEM abierto", "Resultado": available_dem.dataset if available_dem else "No descargado", "Estado": "OK" if available_dem else "Pendiente"},
        {"Control": "Contexto vial e hídrico", "Resultado": f"{osm_context.road_count} caminos · {osm_context.waterway_count} cursos" if osm_context else "No consultado", "Estado": "OK" if osm_context else "Opcional"},
        {"Control": "Radar Sentinel-1", "Resultado": f"NASA OPERA {radar.polarization} · {radar.acquisition_date}" if radar else "No consultado", "Estado": "OK" if radar else "Opcional"},
        {"Control": "Altura sobre drenaje", "Resultado": "GLO-30 HAND · 30 m" if hand else "No consultado", "Estado": "OK" if hand else "Opcional"},
        {"Control": "Suelo modelado", "Resultado": f"SoilGrids · {soil.resolution_m} m · 0 a 5 cm" if soil else "No consultado", "Estado": "Contexto" if soil else "Opcional"},
        {"Control": "Lluvia histórica", "Resultado": f"{rainfall.model} · {rainfall.start_date} a {rainfall.end_date}" if rainfall else "No consultada", "Estado": "OK" if rainfall else "Opcional"},
    ]
    if diagnostics:
        checks += [
            {"Control": "Filas numéricas", "Resultado": f"{format_number(diagnostics['rows_numeric'], 0)} de {format_number(len(monitor_table), 0)}", "Estado": "OK" if diagnostics["rows_numeric"] / len(monitor_table) >= 0.95 else "Revisar"},
            {"Control": "Cobertura espacial", "Resultado": f"{format_number(diagnostics['inside_percent'])} % dentro", "Estado": "OK" if diagnostics["inside_percent"] >= 95 else "Revisar"},
            {"Control": "Variación vertical", "Resultado": f"{diagnostics['unique_elevations']} valores únicos", "Estado": "OK" if diagnostics["unique_elevations"] >= 20 else "Revisar"},
            {"Control": "Coordenada como elevación", "Resultado": "No detectada" if not diagnostics["suspicious_coordinate_copy"] else "Posible copia", "Estado": "OK" if not diagnostics["suspicious_coordinate_copy"] else "Bloqueante"},
        ]
    st.dataframe(pd.DataFrame(checks), hide_index=True, width="stretch")
    st.markdown("#### Supuestos que no deben quedar ocultos")
    st.write(
        "- Terrain Tiles es una compilación global multi-fuente; su datum y precisión varían según la fuente original.\n"
        "- Interpolar el monitor mejora la lectura espacial, no la precisión vertical del receptor.\n"
        "- La pendiente depende de la resolución efectiva del DEM y no representa microrelieve centimétrico.\n"
        "- HAND representa posición vertical frente al drenaje modelado, no probabilidad observada de inundación.\n"
        "- SoilGrids a 250 m caracteriza contexto edáfico y no reemplaza muestreo del lote.\n"
        "- La lluvia diaria de reanálisis no representa intensidades subdiarias ni calcula erosividad USLE.\n"
        "- Recomendar caminos exige cuenca aportante, acumulación de flujo, suelo, obras existentes y control de campo."
    )
    if diagnostics:
        st.info(f"Evaluación del monitor: {quality_label}. {quality_text}")

def render_download_tab() -> None:
    st.subheader("Informe y paquete integral")
    monitor_download_signature = tuple(
        (uploaded.name, len(uploaded.getvalue())) for uploaded in (uploaded_monitors or [])
    )
    download_signature = json.dumps(
        {
            "boundary": boundary_signature,
            "lot": selected_label,
            "monitor": monitor_download_signature,
            "dem": bool(available_dem),
            "context": bool(osm_context),
            "radar": f"{radar.polarization}:{radar.acquisition_date}" if radar else None,
            "hand": bool(hand),
            "soil": bool(soil),
            "rainfall": rainfall.end_date if rainfall else None,
        },
        sort_keys=True,
    )
    prepared_downloads = st.session_state.get("prepared_downloads")
    if prepared_downloads and prepared_downloads.get("signature") == download_signature:
        render_download_bundle(prepared_downloads)
        return
    if not st.button("Preparar informe y paquete completo", type="primary", width="stretch"):
        st.info(
            "La preparación reúne todas las capas disponibles, mapas, gráficos y tablas en el HTML y el ZIP. "
            "El cálculo se ejecuta una sola vez y no vuelve a bloquear el mapa al cambiar el fondo."
        )
        return

    st.session_state.pop("prepared_downloads", None)
    st.caption("Preparando todos los productos disponibles…")
    summary = {
        "lote": selected_label,
        "area_ha": round(area_ha, 3),
        "fuente_analitica": source_label,
        "relieve_robusto_m": round(main_relief, 3) if main_relief is not None else None,
        "pendiente_media_o_general_pct": round(main_slope, 3) if main_slope is not None else None,
        "calidad_monitor": quality_label,
        "dem": available_dem.dataset if available_dem else None,
        "direccion_caida_dem": flow.downslope_direction if flow else None,
        "aporte_maximo_estimado_ha": round(flow.max_contributing_area_ha, 3) if flow else None,
        "metodo_hidrologico": "D8 preliminar sin acondicionamiento de depresiones" if flow else None,
        "caminos_osm_en_contexto": osm_context.road_count if osm_context else None,
        "cursos_osm_en_contexto": osm_context.waterway_count if osm_context else None,
        "radar_producto": f"NASA OPERA RTC-S1 {radar.polarization}" if radar else None,
        "radar_fecha": radar.acquisition_date if radar else None,
        "hand_mediana_m": round(hand.median_m, 3) if hand else None,
        "hand_superficie_bajo_2m_pct": round(hand.area_below_2m_percent, 2) if hand else None,
        "suelo_textura": soil.texture_label if soil else None,
        "suelo_arcilla_pct": round(soil.clay_percent, 1) if soil else None,
        "suelo_limo_pct": round(soil.silt_percent, 1) if soil else None,
        "suelo_arena_pct": round(soil.sand_percent, 1) if soil else None,
        "suelo_carbono_organico_gkg": round(soil.organic_carbon_gkg, 1) if soil else None,
        "lluvia_periodo": f"{rainfall.start_date} a {rainfall.end_date}" if rainfall else None,
        "lluvia_anual_media_mm": round(rainfall.annual_mean_mm, 1) if rainfall else None,
        "lluvia_maxima_diaria_mm": round(rainfall.max_daily_mm, 1) if rainfall else None,
    }
    valid = canonical.loc[canonical["dentro_poligono"]] if canonical is not None else None
    products = []
    files = {}

    if available_dem:
        dem_layers = {
            "Elevación DEM": terrain_elevation,
            "Pendiente DEM": get_terrain_analysis(available_dem.content, selected_geometry.wkb, visual_geometry.wkb, "Pendiente"),
            "Sombreado DEM": get_terrain_analysis(available_dem.content, selected_geometry.wkb, visual_geometry.wkb, "Sombreado"),
        }
        dem_captions = {
            "Elevación DEM": f"Escala robusta: {format_number(terrain_elevation.elevation_p02_m, 2)} a {format_number(terrain_elevation.elevation_p98_m, 2)} m.",
            "Pendiente DEM": "Clases: 0 a 1 %, 1 a 2 %, 2 a 5 %, 5 a 10 % y más de 10 %.",
            "Sombreado DEM": "Iluminación simulada desde el noroeste para destacar la forma del relieve.",
        }
        for filename, (title, overlay) in zip(
            ["mapas/elevacion_dem.png", "mapas/pendiente_dem.png", "mapas/sombreado_dem.png"], dem_layers.items()
        ):
            content = overlay_map_png(selected_feature, overlay.image_data_url, overlay.bounds_wgs84, f"{selected_label} · {title}")
            products.append({"title": title, "caption": dem_captions[title], "content": content})
            files[filename] = content
        elevation_chart = series_chart_png(
            [item[0] for item in terrain_elevation.elevation_histogram],
            [item[1] for item in terrain_elevation.elevation_histogram],
            "Distribución de elevación del DEM", "Elevación (m)", "Celdas",
        )
        slope_chart = bar_chart_png(
            [item[0] for item in terrain_elevation.slope_classes],
            [item[1] for item in terrain_elevation.slope_classes],
            "Superficie por clase de pendiente", "Superficie (%)",
        )
        products += [
            {"title": "Distribución de elevación", "caption": "Frecuencia de celdas por cota.", "content": elevation_chart},
            {"title": "Clases de pendiente", "caption": "Participación porcentual dentro del lote.", "content": slope_chart},
        ]
        files["graficos/distribucion_elevacion_dem.png"] = elevation_chart
        files["graficos/clases_pendiente_dem.png"] = slope_chart
        files["datos/dem_abierto.tif"] = available_dem.content

    if flow:
        flow_png = overlay_map_png(selected_feature, flow.image_data_url, flow.bounds_wgs84, f"{selected_label} · acumulación de flujo")
        products.append({
            "title": "Acumulación de flujo",
            "caption": f"Área aportante visible entre {format_number(flow.display_low_ha, 2)} y {format_number(flow.display_high_ha, 2)} ha; mayor convergencia estimada: {format_number(flow.max_contributing_area_ha, 2)} ha.",
            "content": flow_png,
        })
        files["mapas/acumulacion_flujo.png"] = flow_png

    if monitor_stats and points is not None:
        monitor_surface = analyze_monitor(points, selected_geometry)
        monitor_surface_png = overlay_map_png(
            selected_feature, monitor_surface.image_data_url, monitor_surface.bounds_wgs84, f"{selected_label} · elevación del monitor"
        )
        monitor_points_png = map_png(selected_feature, valid, f"{selected_label} · puntos del monitor")
        monitor_histogram_png = series_chart_png(
            monitor_stats["histogram"]["cota_m"].tolist(), monitor_stats["histogram"]["puntos"].tolist(),
            "Distribución de elevación del monitor", "Elevación (m)", "Puntos",
        )
        monitor_profile_png = series_chart_png(
            monitor_stats["profile"]["distancia_m"].tolist(), monitor_stats["profile"]["cota_mediana_m"].tolist(),
            f"Perfil medio hacia {monitor_stats['downslope_direction']}", "Distancia (m)", "Cota mediana (m)", kind="line",
        )
        monitor_products = [
            ("Elevación del monitor", "Superficie interpolada a partir de puntos dentro del lote.", monitor_surface_png, "mapas/elevacion_monitor.png"),
            ("Puntos del monitor", "Observaciones válidas ubicadas dentro del límite.", monitor_points_png, "mapas/puntos_monitor.png"),
            ("Distribución del monitor", "Frecuencia de observaciones por cota.", monitor_histogram_png, "graficos/distribucion_monitor.png"),
            ("Perfil del monitor", "Perfil mediano sobre el eje general de caída.", monitor_profile_png, "graficos/perfil_monitor.png"),
        ]
        for title, caption, content, filename in monitor_products:
            products.append({"title": title, "caption": caption, "content": content})
            files[filename] = content
        files["datos/perfil_topografico.csv"] = monitor_stats["profile"].to_csv(index=False).encode("utf-8")

    if osm_context:
        context_png = context_map_png(selected_feature, osm_context.roads, osm_context.waterways, f"{selected_label} · caminos y cursos registrados")
        products.append({
            "title": "Contexto vial e hídrico",
            "caption": f"{osm_context.road_count} caminos y {osm_context.waterway_count} cursos registrados en OpenStreetMap.",
            "content": context_png,
        })
        files["mapas/contexto_vial_hidrico.png"] = context_png
        files["datos/caminos_osm.geojson"] = json.dumps(osm_context.roads, ensure_ascii=False).encode("utf-8")
        files["datos/cursos_osm.geojson"] = json.dumps(osm_context.waterways, ensure_ascii=False).encode("utf-8")

    if radar:
        radar_png = overlay_map_png(
            selected_feature, radar.image_data_url, radar.bounds_wgs84,
            f"{selected_label} · Sentinel-1 {radar.polarization} · {radar.acquisition_date}",
        )
        radar_caption = (
            "Retrodifusión VV corregida por terreno. Contraste sensible a humedad, rugosidad y geometría superficial."
            if radar.polarization == "VV"
            else "Retrodifusión VH corregida por terreno. Contraste sensible a vegetación y dispersión volumétrica."
        )
        products.append({"title": f"Radar Sentinel-1 {radar.polarization}", "caption": radar_caption, "content": radar_png})
        files[f"mapas/radar_sentinel1_{radar.polarization.lower()}.png"] = radar_png
        files[f"datos/radar_nasa_opera_{radar.polarization.lower()}_origen.png"] = radar.image_content

    if hand:
        hand_png = overlay_map_png(
            selected_feature, hand.image_data_url, hand.bounds_wgs84,
            f"{selected_label} · altura sobre el drenaje HAND",
        )
        hand_chart = bar_chart_png(
            [item[0] for item in hand.histogram],
            [item[1] for item in hand.histogram],
            "Superficie por altura sobre el drenaje", "Superficie (%)",
        )
        products += [
            {
                "title": "Altura sobre el drenaje HAND",
                "caption": f"Mediana de {format_number(hand.median_m, 2)} m; {format_number(hand.area_below_2m_percent)} % del lote por debajo de 2 m verticales.",
                "content": hand_png,
            },
            {
                "title": "Distribución HAND",
                "caption": "Participación del lote por clase de altura vertical sobre el drenaje más cercano.",
                "content": hand_chart,
            },
        ]
        files["mapas/altura_sobre_drenaje_hand.png"] = hand_png
        files["graficos/distribucion_hand.png"] = hand_chart
        files["datos/hand_origen.tif"] = hand.raster_content
        files["datos/clases_hand.csv"] = pd.DataFrame(hand.histogram, columns=["clase", "superficie_pct"]).to_csv(index=False).encode("utf-8")

    if soil:
        soil_frame = pd.DataFrame(soil.table)
        texture_frame = soil_frame.loc[soil_frame["Propiedad"].isin(["Arcilla", "Limo", "Arena"])]
        soil_chart = bar_chart_png(
            texture_frame["Propiedad"].tolist(), texture_frame["Mediana"].tolist(),
            "Composición textural superficial", "Proporción mediana (%)",
        )
        products.append({
            "title": "Suelo superficial SoilGrids",
            "caption": f"{soil.texture_label}. {soil.erosion_screening}. Resolución global de {soil.resolution_m} m.",
            "content": soil_chart,
        })
        files["graficos/composicion_suelo.png"] = soil_chart
        files["datos/suelo_soilgrids.csv"] = soil_frame.to_csv(index=False).encode("utf-8")

    if rainfall:
        annual_rain = pd.DataFrame(rainfall.annual)
        monthly_rain = pd.DataFrame(rainfall.monthly)
        annual_chart = series_chart_png(
            annual_rain["anio"].tolist(), annual_rain["precipitacion_mm"].tolist(),
            "Precipitación anual de reanálisis", "Año", "Precipitación (mm)",
        )
        monthly_chart = series_chart_png(
            monthly_rain["mes"].tolist(), monthly_rain["precipitacion_media_mm"].tolist(),
            "Promedio mensual de precipitación", "Mes", "Precipitación media (mm)", kind="line",
        )
        products += [
            {
                "title": "Lluvia anual histórica",
                "caption": f"Media de {format_number(rainfall.annual_mean_mm, 0)} mm por año entre {rainfall.start_date[:4]} y {rainfall.end_date[:4]}.",
                "content": annual_chart,
            },
            {
                "title": "Estacionalidad de la lluvia",
                "caption": f"Promedio mensual de reanálisis. Máximo diario del período: {format_number(rainfall.max_daily_mm, 1)} mm.",
                "content": monthly_chart,
            },
        ]
        files["graficos/lluvia_anual.png"] = annual_chart
        files["graficos/lluvia_mensual.png"] = monthly_chart
        files["datos/lluvia_anual.csv"] = annual_rain.to_csv(index=False).encode("utf-8")
        files["datos/lluvia_mensual.csv"] = monthly_rain.to_csv(index=False).encode("utf-8")

    if not products:
        boundary_png = map_png(selected_feature, valid, selected_label)
        products.append({"title": "Límite del lote", "caption": "Geometría normalizada en WGS84.", "content": boundary_png})
        files["mapas/limite.png"] = boundary_png

    files["datos/respuestas_para_decision.csv"] = pd.DataFrame(answers).to_csv(index=False).encode("utf-8")
    files["datos/controles_calidad.csv"] = pd.DataFrame(checks).to_csv(index=False).encode("utf-8")
    html = standalone_html(selected_feature, summary, products, answers, checks)
    package = result_zip(selected_feature, canonical, summary, html, files)
    safe = "".join(character if character.isalnum() or character in "-_" else "_" for character in selected_label)
    prepared_downloads = {
        "signature": download_signature,
        "html": html,
        "html_name": f"{safe}.html",
        "package": package,
        "zip_name": f"{safe}.zip",
        "main_map": products[0]["content"],
        "map_name": f"{safe}.png",
        "boundaries": boundaries.to_json(),
        "summary": json.dumps(summary, ensure_ascii=False, indent=2),
        "summary_name": f"{safe}_resumen.json",
        "dem": available_dem.content if available_dem else None,
        "dem_name": f"{safe}_dem_abierto.tif",
        "product_count": len(products),
        "file_count": len(files),
    }
    st.session_state["prepared_downloads"] = prepared_downloads
    render_download_bundle(prepared_downloads)


with tab_downloads:
    render_download_tab()
