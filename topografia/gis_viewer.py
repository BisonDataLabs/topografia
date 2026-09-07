from __future__ import annotations

import base64
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from matplotlib import colormaps
from PIL import Image
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds


PALETTE_GRADIENTS = {
    "terrain": "#333399,#00a6ca,#7fd34e,#d9c27a,#ffffff",
    "viridis": "#440154,#3b528b,#21918c,#5ec962,#fde725",
    "Blues": "#f7fbff,#c6dbef,#6baed6,#2171b5,#08306b",
    "YlOrRd": "#ffffcc,#fed976,#fd8d3c,#e31a1c,#800026",
    "RdBu_r": "#2166ac,#67a9cf,#f7f7f7,#ef8a62,#b2182b",
    "magma": "#000004,#3b0f70,#8c2981,#de4968,#fcfdbf",
    "gray": "#000000,#ffffff",
    "twilight": "#e2d9e2,#6276ba,#2f1436,#9c3b4c,#e2d9e2",
}


def _hex_color(value: str) -> list[int]:
    clean = value.lstrip("#")
    return [int(clean[index : index + 2], 16) for index in (0, 2, 4)]


def _rgba(array: np.ndarray, valid: np.ndarray, layer: dict[str, Any]) -> np.ndarray:
    legend = layer.get("legend", [])
    if legend:
        pixels = np.zeros((*array.shape, 4), dtype="uint8")
        for item in legend:
            try:
                selected = valid & np.isclose(array, float(item["value"]))
            except (TypeError, ValueError):
                continue
            pixels[selected] = [*_hex_color(item["color"]), 235]
        return pixels
    values = np.log1p(np.clip(array, 0, None)) if layer.get("scale") == "log" else array
    stats = layer.get("stats", {})
    low, high = stats.get("p02"), stats.get("p98")
    if layer.get("scale") == "log":
        low = math.log1p(max(float(low or 0), 0))
        high = math.log1p(max(float(high or 0), 0))
    if low is None or high is None or float(high) <= float(low):
        low, high = np.nanpercentile(values[valid], [2, 98])
    scaled = np.clip((values - float(low)) / max(float(high) - float(low), 1e-9), 0, 1)
    pixels = (colormaps.get_cmap(layer.get("palette", "viridis"))(scaled) * 255).astype("uint8")
    pixels[..., 3] = np.where(valid, 235, 0).astype("uint8")
    return pixels


def raster_payload(path: Path, layer: dict[str, Any]) -> dict[str, Any]:
    with rasterio.open(path) as source:
        maximum_pixels = int(layer.get("display_max_pixels", 650_000))
        scale = min(1.0, math.sqrt(maximum_pixels / max(source.width * source.height, 1)))
        width = max(2, round(source.width * scale))
        height = max(2, round(source.height * scale))
        array = source.read(1, out_shape=(height, width), resampling=Resampling.nearest).astype("float64")
        valid = np.isfinite(array)
        if source.nodata is not None:
            valid &= array != source.nodata
        pixels = _rgba(array, valid, layer)
        output = io.BytesIO()
        Image.fromarray(pixels, "RGBA").save(output, "PNG", optimize=True)
        affine = source.transform * source.transform.scale(source.width / width, source.height / height)
        west, south, east, north = array_bounds(height, width, affine)
        bounds = transform_bounds(source.crs, "EPSG:4326", west, south, east, north)
    return {
        "kind": "raster",
        "image": f"data:image/png;base64,{base64.b64encode(output.getvalue()).decode('ascii')}",
        "bounds": [float(value) for value in bounds],
    }


def vector_payload(path: Path, layer: dict[str, Any]) -> dict[str, Any]:
    frame = gpd.read_parquet(path) if path.suffix == ".parquet" else gpd.read_file(path)
    renderer = layer.get("style", {}).get("renderer", "fill")
    default_maximum = 25_000 if renderer == "graduated_points" else 45_000
    maximum = int(layer.get("display_max_features", default_maximum))
    if len(frame) > maximum:
        frame = frame.sample(maximum, random_state=42)
    frame = frame.to_crs(4326)
    style = layer.get("style", {})
    field = style.get("field")
    legend = {str(item["value"]): item["color"] for item in layer.get("legend", [])}
    if renderer == "graduated_points" and field in frame:
        values = frame[field].astype("float64")
        limits = style.get("range") or [values.quantile(0.02), values.quantile(0.98)]
        scaled = np.clip((values - float(limits[0])) / max(float(limits[1]) - float(limits[0]), 1e-9), 0, 1)
        colors = colormaps.get_cmap(style.get("palette", "terrain"))(scaled)
        frame["_color"] = [f"rgb({round(c[0] * 255)},{round(c[1] * 255)},{round(c[2] * 255)})" for c in colors]
    elif legend and field in frame:
        frame["_color"] = frame[field].map(lambda value: legend.get(str(value), "#2878b5"))
    else:
        frame["_color"] = style.get("color", style.get("stroke", style.get("fill", "#2878b5")))
    properties = [column for column in frame.columns if column != "geometry"]
    safe = frame[properties + ["geometry"]].copy()
    for column in properties:
        if safe[column].dtype == "object":
            safe[column] = safe[column].map(lambda value: str(value) if value is not None else "")
    return {
        "kind": "vector",
        "geojson": json.loads(safe.to_json(drop_id=True)),
        "bounds": [float(value) for value in frame.total_bounds],
        "renderer": renderer,
        "radius": int(style.get("point_radius_px", 2)),
        "width": float(style.get("width_px", 1.4)),
    }


def viewer_html(layers: list[dict[str, Any]], bounds: list[float]) -> str:
    payload = json.dumps(layers, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    viewer_id = f"gis-{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12]}"
    west, south, east, north = bounds
    return f"""
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>
#{viewer_id}{{height:760px;position:relative;background:#eef2f5;font-family:system-ui,-apple-system,sans-serif}}
#{viewer_id} .gis-map{{height:100%;margin:0;background:#eef2f5}}
.leaflet-control-layers{{display:none}}
.gis-panel{{position:absolute;z-index:1000;top:12px;right:12px;width:320px;max-height:calc(100% - 24px);overflow:auto;background:rgba(255,255,255,.97);border:1px solid #cfd8df;border-radius:8px;box-shadow:0 2px 14px #20304033}}
.gis-panel summary{{cursor:pointer;padding:10px 12px;font-weight:700;list-style:none}}
.gis-body{{padding:0 12px 12px}}
.gis-section{{font-size:12px;font-weight:700;color:#52606d;text-transform:uppercase;letter-spacing:.04em;margin:10px 0 5px;border-top:1px solid #e3e8ec;padding-top:9px}}
.gis-row{{padding:6px 0;border-bottom:1px solid #edf0f2}}
.gis-name{{display:flex;gap:7px;align-items:flex-start;font-size:13px;line-height:1.25}}
.gis-name input{{margin-top:2px}}
.gis-opacity{{display:flex;align-items:center;gap:7px;padding:4px 0 0 21px;color:#697783;font-size:11px}}
.gis-opacity input{{width:100%}}
.gis-basemap{{width:100%;padding:6px;border:1px solid #ccd5dc;border-radius:5px;background:#fff}}
.gis-legend{{height:7px;border-radius:2px;margin:5px 0 0 21px}}
.gis-range{{display:flex;justify-content:space-between;margin:2px 0 0 21px;color:#697783;font-size:10px}}
.gis-classes{{margin:4px 0 0 21px;color:#697783;font-size:10px;line-height:1.4}}
.gis-swatch{{display:inline-block;width:9px;height:7px;margin-right:3px;border:1px solid #9aa7b2}}
.leaflet-control-scale-line{{background:#ffffffcc}}
@media(max-width:700px){{.gis-panel{{width:260px}}}}
</style><div id="{viewer_id}"><div class="gis-map"></div>
<details class="gis-panel" open><summary>Mapas y capas</summary><div class="gis-body">
<div class="gis-section" style="border-top:0;margin-top:0">Mapa base</div>
<select data-role="basemap" class="gis-basemap"><option value="topo">Topográfico con nombres</option><option value="satellite">Satelital</option><option value="hybrid">Híbrido</option><option value="streets">Calles</option></select>
<label class="gis-opacity" style="padding-left:0"><span>Opacidad</span><input data-role="base-opacity" type="range" min="0" max="100" value="100"></label>
<div data-role="layer-list"></div></div></details></div>
<script>
(()=>{{
const initialize=()=>{{
const root=document.getElementById('{viewer_id}');if(!root||root.dataset.ready)return;root.dataset.ready='true';
const map=L.map(root.querySelector('.gis-map'),{{zoomControl:true,preferCanvas:true}});
map.createPane('rasters');map.getPane('rasters').style.zIndex=300;
map.createPane('vectors');map.getPane('vectors').style.zIndex=400;
map.createPane('references');map.getPane('references').style.zIndex=450;
const attribution='';
const topo=L.tileLayer('https://{{s}}.tile.opentopomap.org/{{z}}/{{x}}/{{y}}.png',{{maxZoom:17,attribution:'© OpenStreetMap, SRTM · OpenTopoMap'}});
const satellite=L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',{{maxZoom:19,attribution:'Esri World Imagery'}});
const labels=L.tileLayer('https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{{z}}/{{y}}/{{x}}',{{maxZoom:19}});
const streets=L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{maxZoom:19,attribution:'© OpenStreetMap'}});
const hybrid=L.layerGroup([satellite,labels]);
const bases={{topo,satellite,hybrid,streets}}; let activeBase=topo.addTo(map);
let baseOpacity=1;function setBaseOpacity(layer,value){{if(layer.eachLayer)layer.eachLayer(item=>item.setOpacity(value));else layer.setOpacity(value)}}
root.querySelector('[data-role="basemap"]').addEventListener('change',e=>{{map.removeLayer(activeBase);activeBase=bases[e.target.value];setBaseOpacity(activeBase,baseOpacity);activeBase.addTo(map);activeBase.bringToBack?.();}});
root.querySelector('[data-role="base-opacity"]').addEventListener('input',e=>{{baseOpacity=Number(e.target.value)/100;setBaseOpacity(activeBase,baseOpacity)}});
const defs={payload}; const layerObjects={{}}; const groups={{evidence:'1 · Evidencia topográfica',comparison:'2 · Comparación de superficies',microrelief:'3 · Microrelieve y cotas',basin:'4 · Cuenca y drenaje',reference:'5 · Referencias'}};
function styleFeature(def,feature,opacity){{const boundary=def.original_id==='boundary';const line=feature.properties._color||'#284b63';return {{color:line,weight:boundary?2.5:(def.width||1.4),opacity:opacity,fillColor:line,fillOpacity:boundary?0:(def.renderer==='line'?0:opacity*.55)}}}}
function escapeText(value){{return String(value).replace(/[&<>"']/g,char=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[char]))}}
function makeLayer(def){{if(def.kind==='raster')return L.imageOverlay(def.image,[[def.bounds[1],def.bounds[0]],[def.bounds[3],def.bounds[2]]],{{opacity:def.opacity,interactive:false,pane:'rasters'}});const pane=def.original_id==='boundary'?'references':'vectors';return L.geoJSON(def.geojson,{{pane,style:f=>({{...styleFeature(def,f,def.opacity),pane}}),pointToLayer:(f,ll)=>L.circleMarker(ll,{{pane,radius:def.radius||2,...styleFeature(def,f,def.opacity)}}),onEachFeature:(f,l)=>{{const p=f.properties||{{}};const text=p.elevation_m??p.level??p.reason??p.lote;if(text!==undefined)l.bindTooltip(escapeText(text));}}}})}}
function setOpacity(def,obj,value){{def.opacity=value;if(def.kind==='raster')obj.setOpacity(value);else obj.setStyle(f=>styleFeature(def,f,value));}}
const list=root.querySelector('[data-role="layer-list"]');let current='';defs.forEach((def,i)=>{{if(def.group!==current){{current=def.group;const h=document.createElement('div');h.className='gis-section';h.textContent=groups[current]||current;list.appendChild(h)}}const obj=makeLayer(def);layerObjects[def.id]=obj;if(def.visible)obj.addTo(map);const row=document.createElement('div');row.className='gis-row';const label=document.createElement('label');label.className='gis-name';const check=document.createElement('input');check.type='checkbox';check.checked=!!def.visible;check.addEventListener('change',()=>check.checked?obj.addTo(map):map.removeLayer(obj));const text=document.createElement('span');text.textContent=def.title;label.append(check,text);row.appendChild(label);const op=document.createElement('label');op.className='gis-opacity';const opacityText=document.createElement('span');opacityText.textContent='Opacidad';op.appendChild(opacityText);const slider=document.createElement('input');slider.type='range';slider.min=0;slider.max=100;slider.value=Math.round(def.opacity*100);slider.addEventListener('input',()=>setOpacity(def,obj,Number(slider.value)/100));op.appendChild(slider);row.appendChild(op);if(def.gradient){{const grad=document.createElement('div');grad.className='gis-legend';grad.style.background='linear-gradient(90deg,'+def.gradient+')';row.appendChild(grad)}}if(def.stats){{const low=def.stats.p02??def.stats.min;const high=def.stats.p98??def.stats.max;if(low!==undefined&&high!==undefined){{const range=document.createElement('div');range.className='gis-range';const lowLabel=document.createElement('span');const highLabel=document.createElement('span');lowLabel.textContent=Number(low).toLocaleString('es-AR',{{maximumFractionDigits:2}})+' '+(def.units||'');highLabel.textContent=Number(high).toLocaleString('es-AR',{{maximumFractionDigits:2}})+' '+(def.units||'');range.append(lowLabel,highLabel);row.appendChild(range)}}}}if(def.legend&&def.legend.length){{const classes=document.createElement('div');classes.className='gis-classes';def.legend.forEach((item,index)=>{{const line=document.createElement('span');const swatch=document.createElement('i');swatch.className='gis-swatch';swatch.style.background=item.color;line.append(swatch,document.createTextNode(item.label));classes.appendChild(line);if(index<def.legend.length-1)classes.appendChild(document.createElement('br'))}});row.appendChild(classes)}}list.appendChild(row)}});
L.control.scale({{imperial:false}}).addTo(map);map.fitBounds([[{south},{west}],[{north},{east}]],{{padding:[22,22]}});
setTimeout(()=>map.invalidateSize(),0);
}};
if(window.L)initialize();else{{let script=document.querySelector('script[data-topografia-leaflet]');if(script)script.addEventListener('load',initialize,{{once:true}});else{{script=document.createElement('script');script.src='https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';script.dataset.topografiaLeaflet='true';script.addEventListener('load',initialize,{{once:true}});document.head.appendChild(script)}}}}
}})();
</script>"""
