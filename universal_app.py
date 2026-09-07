from __future__ import annotations

import base64
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from shapely import from_wkb, union_all

from topografia.gis_viewer import PALETTE_GRADIENTS, raster_payload, vector_payload, viewer_html
from topografia.package_io import TopographyPackage, load_package
from topografia.public_analysis import (
    BOUNDARY_EXTENSIONS,
    analyze_dem,
    analyze_flow,
    area_hectares,
    buffered_geometry,
    expanded_bounds,
    fetch_hand,
    fetch_open_dem,
    read_boundaries,
)


st.set_page_config(page_title="Topografía", layout="wide")
st.markdown(
    """
    <style>
    .stApp { background: #ffffff; color: #17212b; }
    [data-testid="stFileUploaderDropzone"] button {font-size:0 !important;}
    [data-testid="stFileUploaderDropzone"] button * {display:none !important;}
    [data-testid="stFileUploaderDropzone"] button::after {content:"Seleccionar";font-size:.875rem;}
    .layer-legend {border:1px solid #dce3ea;border-radius:8px;padding:.55rem .7rem;background:#fff;}
    </style>
    """,
    unsafe_allow_html=True,
)

PACKAGE_LAYER_IDS = {
    "public-dem",
    "public-dem-aligned",
    "consensus-elevation",
    "campaign-disagreement",
    "surface-confidence",
    "microrelief-residual",
    "relative-elevation",
    "slope-percent",
    "slope-class",
    "multidirectional-hillshade",
    "height-above-drainage",
    "flow-accumulation",
    "drainage-potential",
    "drainage-potential-zones",
    "relative-contours",
    "boundary",
    "rejected-observations",
}
GROUP_LABELS = {
    "evidence": "1 · Evidencia topográfica",
    "comparison": "2 · Comparación de superficies",
    "microrelief": "3 · Microrelieve y cotas",
    "basin": "4 · Cuenca y drenaje",
    "reference": "5 · Referencias",
}
TITLE_OVERRIDES = {
    "public-dem": "DEM público · fuente de aproximadamente 30 m",
    "public-dem-aligned": "DEM público alineado · sólo comparación",
    "consensus-elevation": "Superficie topográfica interpolada",
    "campaign-disagreement": "Desacuerdo entre campañas",
    "microrelief-residual": "Microrelieve respecto de la tendencia local",
    "relative-elevation": "Altura relativa dentro del lote",
    "relative-contours": "Curvas de nivel relativas cada 1 m",
    "height-above-drainage": "Altura sobre el drenaje HAND",
    "rejected-observations": "Observaciones descartadas",
}


@st.cache_resource(show_spinner=False)
def open_package(content: bytes) -> TopographyPackage:
    return load_package(content)


@st.cache_data(show_spinner=False)
def parse_boundaries(filename: str, content: bytes) -> gpd.GeoDataFrame:
    return read_boundaries(filename, content)


@st.cache_data(ttl=86_400, show_spinner=False)
def open_dem(bounds: tuple[float, float, float, float]):
    return fetch_open_dem(bounds, zoom=12)


@st.cache_data(max_entries=24, show_spinner=False)
def terrain_overlay(content: bytes, analysis_wkb: bytes, display_wkb: bytes, mode: str):
    return analyze_dem(content, from_wkb(analysis_wkb), from_wkb(display_wkb), mode)


@st.cache_data(max_entries=8, show_spinner=False)
def flow_overlay(content: bytes, analysis_wkb: bytes, display_wkb: bytes):
    return analyze_flow(content, from_wkb(analysis_wkb), from_wkb(display_wkb))


@st.cache_data(ttl=86_400, show_spinner=False)
def hand_overlay(bounds: tuple[float, float, float, float], analysis_wkb: bytes):
    return fetch_hand(bounds, from_wkb(analysis_wkb))


@st.cache_data(max_entries=256, show_spinner=False)
def prepared_package_layer(path: str, layer_json: str) -> dict:
    layer = json.loads(layer_json)
    payload = raster_payload(Path(path), layer) if layer["kind"] == "raster" else vector_payload(Path(path), layer)
    payload.update(layer)
    return payload


def number(value: object, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def format_number(value: object, decimals: int = 1) -> str:
    return f"{number(value):,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def layer_group(layer_id: str) -> str:
    if layer_id in {"public-dem", "public-dem-aligned", "consensus-elevation", "campaign-disagreement"}:
        return "comparison"
    if layer_id.startswith("campaign-") or layer_id in {"rejected-observations", "surface-confidence"}:
        return "evidence"
    if layer_id in {
        "microrelief-residual",
        "relative-elevation",
        "slope-percent",
        "slope-class",
        "multidirectional-hillshade",
        "relative-contours",
    }:
        return "microrelief"
    if layer_id in {"height-above-drainage", "flow-accumulation", "drainage-potential", "drainage-potential-zones"}:
        return "basin"
    return "reference"


def package_layer_allowed(layer: dict) -> bool:
    layer_id = str(layer.get("id", ""))
    return layer_id in PACKAGE_LAYER_IDS or layer_id.startswith("campaign-")


def package_payload(entry: dict, visible: bool, detailed: bool) -> dict:
    serializable = {
        key: value for key, value in entry.items() if key not in {"package", "field_id", "field_name", "group_ui"}
    }
    serializable.update(
        {
            "group": entry["group_ui"],
            "title": f"{entry['field_name']} · {entry['title']}",
            "visible": visible,
            "opacity": float(entry.get("opacity", 0.78)),
            "gradient": PALETTE_GRADIENTS.get(entry.get("palette", "")),
            "display_max_pixels": 650_000 if detailed else 180_000,
            "display_max_features": 45_000 if detailed else 8_000,
        }
    )
    path = entry["package"].root / entry["path"]
    return prepared_package_layer(str(path), json.dumps(serializable, ensure_ascii=False, sort_keys=True))


def public_raster_payload(layer_id: str, title: str, group: str, overlay, visible: bool) -> dict:
    colors = [color for _, color in overlay.legend]
    return {
        "id": layer_id,
        "original_id": layer_id,
        "title": title,
        "group": group,
        "kind": "raster",
        "image": overlay.image_data_url,
        "bounds": list(overlay.bounds_wgs84),
        "opacity": 0.78,
        "visible": visible,
        "gradient": ",".join(colors),
    }


def package_mode(uploaded_files) -> None:
    packages: list[tuple[object, TopographyPackage]] = []
    for uploaded in uploaded_files:
        try:
            packages.append((uploaded, open_package(uploaded.getvalue())))
        except Exception as error:
            st.error(f"{uploaded.name}: {error}")
    if not packages:
        st.stop()

    entries, summary_rows = [], []
    for package_index, (uploaded, package) in enumerate(packages):
        field = package.catalog.get("field", {})
        field_id = str(field.get("id") or f"lote-{package_index + 1}")
        field_name = str(field.get("name") or field_id)
        terrain = read_json(package.root / "metrics" / "terrain.json")
        stability = read_json(package.root / "metrics" / "campaign_stability.json")
        quality = read_json(package.root / "metrics" / "quality_control.json")
        summary_rows.append(
            {
                "Lote": field_name,
                "Superficie (ha)": number(terrain.get("area_ha")),
                "Relieve robusto (m)": number(terrain.get("robust_relief_m")),
                "Pendiente mediana (%)": number(terrain.get("slope_median_percent")),
                "Observaciones aceptadas (%)": number(quality.get("accepted_percent")),
                "Desacuerdo P90 (m)": number(stability.get("p90_disagreement_m")),
            }
        )
        for layer in package.catalog.get("layers", []):
            if not package_layer_allowed(layer):
                continue
            item = dict(layer)
            original_id = str(item["id"])
            item.update(
                {
                    "original_id": original_id,
                    "id": f"p{package_index}-{original_id}",
                    "title": TITLE_OVERRIDES.get(original_id, str(item.get("title", original_id))),
                    "field_id": field_id,
                    "field_name": field_name,
                    "package": package,
                    "group_ui": layer_group(original_id),
                }
            )
            entries.append(item)

    st.subheader("Paquete topográfico")
    headline = st.columns(3)
    headline[0].metric("Lotes", len(summary_rows))
    headline[1].metric("Superficie", f"{format_number(sum(row['Superficie (ha)'] for row in summary_rows), 1)} ha")
    headline[2].metric("Capas disponibles", len(entries))
    summary_tab, map_tab = st.tabs(["Resumen", "Mapa y capas"])

    with summary_tab:
        st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")
        group_rows = [
            {"Grupo": label, "Capas": sum(entry["group_ui"] == group_id for entry in entries)}
            for group_id, label in GROUP_LABELS.items()
        ]
        st.markdown("#### Contenido disponible")
        st.dataframe(pd.DataFrame(group_rows), hide_index=True, width="stretch")

    with map_tab:
        field_names = [row["Lote"] for row in summary_rows]
        selected_fields = st.multiselect("Lotes incluidos en el visor", field_names, default=field_names)
        detail_fields = st.multiselect(
            "Lotes con todas las capas",
            selected_fields,
            default=selected_fields[:1],
            help="Los demás lotes conservan superficie, curvas y límite para reducir el peso del visor.",
        )
        overview_layers = {"consensus-elevation", "relative-contours", "boundary"}
        available = [
            entry
            for entry in entries
            if entry["field_name"] in selected_fields
            and (entry["field_name"] in detail_fields or entry["original_id"] in overview_layers)
        ]
        defaults = {"consensus-elevation", "relative-contours", "boundary"}
        order = {name: index for index, name in enumerate(GROUP_LABELS)}
        available.sort(key=lambda entry: (order[entry["group_ui"]], entry["field_name"], entry["title"]))
        with st.spinner("Preparando las capas del visor…"):
            map_layers = [
                package_payload(
                    entry,
                    entry["original_id"] in defaults,
                    entry["field_name"] in detail_fields,
                )
                for entry in available
            ]
        boundary_layers = [layer for layer in map_layers if layer["original_id"] == "boundary"]
        fitting = boundary_layers or map_layers
        if fitting:
            west = min(layer["bounds"][0] for layer in fitting)
            south = min(layer["bounds"][1] for layer in fitting)
            east = max(layer["bounds"][2] for layer in fitting)
            north = max(layer["bounds"][3] for layer in fitting)
            components.html(viewer_html(map_layers, [west, south, east, north]), height=780, scrolling=False)
        else:
            st.info("La selección no contiene capas cartográficas.")


def boundary_mode(uploaded) -> None:
    try:
        boundaries = parse_boundaries(uploaded.name, uploaded.getvalue())
    except Exception as error:
        st.error(f"No se pudo leer el archivo de lotes: {error}")
        st.stop()

    columns = [column for column in boundaries.columns if column != "geometry"]
    preferred = next(
        (column for column in ("nombre", "name", "lote", "field", "id") if column in columns), "__field_id"
    )
    with st.sidebar:
        st.header("Selección")
        name_column = st.selectbox("Columna con el nombre", columns, index=columns.index(preferred))
    boundaries = boundaries.copy()
    boundaries["__name"] = boundaries[name_column].fillna(boundaries["__field_id"]).astype(str)
    names = boundaries["__name"].tolist()
    with st.sidebar:
        scale = st.radio("Escala", ["Un lote", "Grupo de lotes"], horizontal=True)
        chosen = [st.selectbox("Lote", names)] if scale == "Un lote" else st.multiselect("Lotes", names, default=names)
        if not chosen:
            st.info("La selección requiere al menos un lote.")
            st.stop()
    selected_frame = boundaries.loc[boundaries["__name"].isin(chosen)].copy()
    geometry = union_all(selected_frame.geometry.to_numpy())
    visual_geometry = buffered_geometry(geometry, 500)
    signature = geometry.wkb_hex

    with st.sidebar:
        st.header("Relieve público")
        if st.button("Analizar con DEM abierto", type="primary", width="stretch"):
            try:
                with st.spinner("Obteniendo el DEM público…"):
                    st.session_state["public-dem"] = open_dem(expanded_bounds(geometry.bounds, 0.5))
                    st.session_state["public-dem-signature"] = signature
            except Exception as error:
                st.error(f"No se pudo obtener el DEM: {error}")
        context_km = st.select_slider(
            "Extensión del contexto hídrico", [2, 5, 10], value=5, format_func=lambda value: f"{value} km"
        )
        if st.button("Calcular altura sobre el drenaje", width="stretch"):
            try:
                with st.spinner("Obteniendo HAND para el entorno hídrico…"):
                    st.session_state["public-hand"] = hand_overlay(
                        expanded_bounds(geometry.bounds, context_km), geometry.wkb
                    )
                    st.session_state["public-hand-signature"] = signature
                    st.session_state["public-hand-context"] = context_km
            except Exception as error:
                st.error(f"No se pudo obtener HAND: {error}")

    dem = st.session_state.get("public-dem") if st.session_state.get("public-dem-signature") == signature else None
    hand = st.session_state.get("public-hand") if st.session_state.get("public-hand-signature") == signature else None
    overlays = {}
    if dem:
        try:
            overlays = {
                "Elevación DEM": terrain_overlay(dem.content, geometry.wkb, visual_geometry.wkb, "elevation"),
                "Pendiente DEM": terrain_overlay(dem.content, geometry.wkb, visual_geometry.wkb, "slope"),
                "Sombreado DEM": terrain_overlay(dem.content, geometry.wkb, visual_geometry.wkb, "hillshade"),
                "Acumulación de flujo": flow_overlay(dem.content, geometry.wkb, visual_geometry.wkb),
            }
        except Exception as error:
            st.error(f"No se pudo completar el análisis del DEM: {error}")
    if hand:
        overlays["Altura sobre el drenaje HAND"] = hand

    st.subheader("Análisis público")
    headline = st.columns(4)
    headline[0].metric("Lotes", len(selected_frame))
    headline[1].metric("Superficie", f"{format_number(area_hectares(geometry), 1)} ha")
    if overlays.get("Elevación DEM"):
        metrics = overlays["Elevación DEM"].metrics
        headline[2].metric("Relieve robusto", f"{format_number(metrics['relief_m'], 2)} m")
        headline[3].metric("Pendiente media", f"{format_number(metrics['mean_slope_percent'], 2)} %")
    else:
        headline[2].metric("Relieve robusto", "Pendiente")
        headline[3].metric("Pendiente media", "Pendiente")

    summary_tab, map_tab, downloads_tab = st.tabs(["Resumen", "Mapa y capas", "Descargas"])
    with summary_tab:
        rows = selected_frame[["__name", "geometry"]].copy()
        metric_crs = rows.estimate_utm_crs()
        rows["Superficie (ha)"] = rows.to_crs(metric_crs).area / 10_000
        st.dataframe(rows.drop(columns="geometry").rename(columns={"__name": "Lote"}), hide_index=True, width="stretch")
        if hand:
            hand_columns = st.columns(2)
            hand_columns[0].metric("HAND mediano", f"{format_number(hand.metrics['median_m'], 2)} m")
            hand_columns[1].metric(
                "Superficie bajo 2 m HAND", f"{format_number(hand.metrics['area_below_2m_percent'], 1)} %"
            )

    with map_tab:
        map_layers = []
        public_groups = {
            "Elevación DEM": "comparison",
            "Pendiente DEM": "microrelief",
            "Sombreado DEM": "microrelief",
            "Acumulación de flujo": "basin",
            "Altura sobre el drenaje HAND": "basin",
        }
        for index, (name, overlay) in enumerate(overlays.items()):
            map_layers.append(
                public_raster_payload(f"public-{index}", name, public_groups[name], overlay, name == "Elevación DEM")
            )
        boundary_geojson = json.loads(
            selected_frame[["__name", "geometry"]].rename(columns={"__name": "lote"}).to_json()
        )
        map_layers.append(
            {
                "id": "public-boundaries",
                "original_id": "boundary",
                "title": "Límites",
                "group": "reference",
                "kind": "vector",
                "geojson": boundary_geojson,
                "bounds": [float(value) for value in selected_frame.total_bounds],
                "renderer": "line",
                "width": 2.5,
                "opacity": 1.0,
                "visible": True,
            }
        )
        components.html(
            viewer_html(map_layers, [float(value) for value in selected_frame.total_bounds]),
            height=780,
            scrolling=False,
        )

    with downloads_tab:
        boundary_bytes = (
            selected_frame[["__name", "geometry"]].rename(columns={"__name": "lote"}).to_json().encode("utf-8")
        )
        st.download_button(
            "Límites seleccionados",
            boundary_bytes,
            "limites-seleccionados.geojson",
            "application/geo+json",
            width="stretch",
        )
        if dem:
            st.download_button("DEM público GeoTIFF", dem.content, "dem-publico.tif", "image/tiff", width="stretch")
        for name, overlay in overlays.items():
            image_bytes = base64.b64decode(overlay.image_data_url.split(",", 1)[1])
            st.download_button(
                f"Imagen · {name}",
                image_bytes,
                f"{name.lower().replace(' ', '-')}.png",
                "image/png",
                key=f"public-download-{name}",
                width="stretch",
            )


st.title("Topografía")
st.caption("Visualización conjunta de lotes, relieve, microrelieve, cotas y drenaje")
source_mode = st.radio(
    "Información disponible",
    ["Paquete topográfico", "Sólo límites de lotes"],
    horizontal=True,
    captions=[
        "Capas técnicas preparadas con relevamientos y DEM público.",
        "Análisis topográfico común mediante fuentes públicas.",
    ],
)
if source_mode == "Paquete topográfico":
    uploads = st.file_uploader(
        "Paquetes topográficos",
        type=["zip"],
        accept_multiple_files=True,
        help="Selección conjunta de uno o más ZIP generados por el paquete técnico.",
    )
    if not uploads:
        st.info("Carga de los paquetes correspondientes a los lotes de interés.")
        st.stop()
    package_mode(uploads)
else:
    upload = st.file_uploader(
        "Archivo de lotes",
        type=BOUNDARY_EXTENSIONS,
        help="GeoJSON, KML, KMZ, GeoPackage o Shapefile completo dentro de un ZIP.",
    )
    if upload is None:
        st.info("Carga de un archivo con uno o más límites de lotes.")
        st.stop()
    boundary_mode(upload)
