from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from html import escape
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "topografia-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from .data import feature_name, polygons


def map_png(feature: dict, points: pd.DataFrame | None, title: str) -> bytes:
    figure, axis = plt.subplots(figsize=(8, 8), dpi=180)
    axis.set_facecolor("#FFFFFF")
    figure.patch.set_facecolor("#FFFFFF")
    if points is not None and len(points):
        sample = points if len(points) <= 60_000 else points.sample(60_000, random_state=26)
        scatter = axis.scatter(
            sample["lon"], sample["lat"], c=sample["elevacion_m"], cmap="terrain",
            s=1.2, alpha=0.65, rasterized=True,
        )
        colorbar = figure.colorbar(scatter, ax=axis, fraction=0.035, pad=0.02)
        colorbar.set_label("Elevación observada (m)", color="#17212B")
        colorbar.ax.tick_params(colors="#17212B")
    for polygon in polygons(feature["geometry"]):
        for index, ring in enumerate(polygon):
            x = [point[0] for point in ring]
            y = [point[1] for point in ring]
            axis.plot(x, y, color="#174A75" if index == 0 else "#718096", linewidth=1.6)
    axis.set_title(title, color="#17212B", fontsize=14, weight="bold")
    axis.set_xlabel("Longitud", color="#344054")
    axis.set_ylabel("Latitud", color="#344054")
    axis.tick_params(colors="#475467")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(color="#D0D5DD", linewidth=0.5, alpha=0.7)
    figure.tight_layout()
    output = io.BytesIO()
    figure.savefig(output, format="png", facecolor=figure.get_facecolor(), bbox_inches="tight")
    plt.close(figure)
    return output.getvalue()


def overlay_map_png(feature: dict, image_data_url: str, bounds: tuple[float, float, float, float], title: str) -> bytes:
    import base64

    encoded = image_data_url.split(",", 1)[1]
    raster = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGBA")
    west, south, east, north = bounds
    figure, axis = plt.subplots(figsize=(9, 7), dpi=180)
    figure.patch.set_facecolor("#FFFFFF")
    axis.set_facecolor("#F8FAFC")
    axis.imshow(raster, extent=[west, east, south, north], origin="upper")
    for polygon in polygons(feature["geometry"]):
        for index, ring in enumerate(polygon):
            x = [point[0] for point in ring]
            y = [point[1] for point in ring]
            axis.plot(x, y, color="#0F4A75" if index == 0 else "#718096", linewidth=2.0)
    axis.set_title(title, color="#17212B", fontsize=14, weight="bold")
    axis.set_xlabel("Longitud")
    axis.set_ylabel("Latitud")
    axis.set_aspect("equal", adjustable="box")
    axis.grid(color="#D0D5DD", linewidth=0.45, alpha=0.6)
    figure.tight_layout()
    output = io.BytesIO()
    figure.savefig(output, format="png", facecolor="#FFFFFF", bbox_inches="tight")
    plt.close(figure)
    return output.getvalue()


def bar_chart_png(labels: list[str], values: list[float], title: str, x_label: str) -> bytes:
    figure, axis = plt.subplots(figsize=(8, 4.8), dpi=180)
    positions = range(len(labels))
    bars = axis.barh(list(positions), values, color="#2878B5")
    axis.set_yticks(list(positions), labels)
    axis.set_xlabel(x_label)
    axis.set_title(title, loc="left", weight="bold")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="x", color="#E2E8F0", linewidth=0.7)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, values):
        axis.text(value, bar.get_y() + bar.get_height() / 2, f" {value:,.1f}", va="center", fontsize=9)
    figure.tight_layout()
    output = io.BytesIO()
    figure.savefig(output, format="png", facecolor="#FFFFFF", bbox_inches="tight")
    plt.close(figure)
    return output.getvalue()


def series_chart_png(
    x_values: list[float], y_values: list[float], title: str, x_label: str, y_label: str, kind: str = "bar"
) -> bytes:
    figure, axis = plt.subplots(figsize=(8, 4.8), dpi=180)
    if kind == "line":
        axis.plot(x_values, y_values, color="#D97706", linewidth=2.2, marker="o", markersize=3)
    else:
        width = (max(x_values) - min(x_values)) / max(len(x_values), 1) * 0.86 if len(x_values) > 1 else 1
        axis.bar(x_values, y_values, width=width, color="#2878B5", edgecolor="#FFFFFF")
    axis.set_title(title, loc="left", weight="bold")
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#E2E8F0", linewidth=0.7)
    axis.set_axisbelow(True)
    figure.tight_layout()
    output = io.BytesIO()
    figure.savefig(output, format="png", facecolor="#FFFFFF", bbox_inches="tight")
    plt.close(figure)
    return output.getvalue()


def context_map_png(feature: dict, roads: dict, waterways: dict, title: str) -> bytes:
    figure, axis = plt.subplots(figsize=(9, 7), dpi=180)
    figure.patch.set_facecolor("#FFFFFF")
    axis.set_facecolor("#F8FAFC")
    for item in roads.get("features", []):
        coordinates = item["geometry"]["coordinates"]
        axis.plot([p[0] for p in coordinates], [p[1] for p in coordinates], color="#976A38", linewidth=0.8, alpha=0.75)
    for item in waterways.get("features", []):
        coordinates = item["geometry"]["coordinates"]
        axis.plot([p[0] for p in coordinates], [p[1] for p in coordinates], color="#0F70BC", linewidth=1.2, alpha=0.9)
    for polygon in polygons(feature["geometry"]):
        for index, ring in enumerate(polygon):
            axis.plot([p[0] for p in ring], [p[1] for p in ring], color="#0F4A75" if index == 0 else "#718096", linewidth=2)
    axis.set_title(title, loc="left", weight="bold")
    axis.set_xlabel("Longitud")
    axis.set_ylabel("Latitud")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(color="#E2E8F0", linewidth=0.5)
    figure.tight_layout()
    output = io.BytesIO()
    figure.savefig(output, format="png", facecolor="#FFFFFF", bbox_inches="tight")
    plt.close(figure)
    return output.getvalue()


def standalone_html(
    feature: dict,
    summary: dict,
    products: list[dict] | None = None,
    answers: list[dict] | None = None,
    quality_checks: list[dict] | None = None,
) -> bytes:
    import base64

    name = feature_name(feature)
    summary_rows = "".join(
        f"<tr><th>{escape(str(key).replace('_', ' ').title())}</th><td>{escape(str(value))}</td></tr>"
        for key, value in summary.items() if value is not None
    )
    answer_rows = "".join(
        f"<tr><th>{escape(str(item.get('Pregunta', '')))}</th><td>{escape(str(item.get('Respuesta', '')))}</td>"
        f"<td><span class='status'>{escape(str(item.get('Estado', '')))}</span></td></tr>"
        for item in (answers or [])
    )
    quality_rows = "".join(
        f"<tr><th>{escape(str(item.get('Control', '')))}</th><td>{escape(str(item.get('Resultado', '')))}</td>"
        f"<td><span class='status'>{escape(str(item.get('Estado', '')))}</span></td></tr>"
        for item in (quality_checks or [])
    )
    gallery = "".join(
        f"<figure><img src='data:image/png;base64,{base64.b64encode(item['content']).decode('ascii')}' "
        f"alt='{escape(item['title'])}'><figcaption><strong>{escape(item['title'])}</strong>"
        f"<span>{escape(item.get('caption', ''))}</span></figcaption></figure>"
        for item in (products or [])
    )
    document = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{escape(name)} · control topográfico</title>
<style>
:root{{--ink:#17212b;--muted:#64748b;--line:#dbe2ea;--soft:#f5f8fb;--blue:#0f4a75}}
*{{box-sizing:border-box}}body{{margin:0;background:#fff;color:var(--ink);font:15px system-ui;line-height:1.5}}
main{{max-width:1180px;margin:auto;padding:36px}}h1{{margin:.2rem 0 0;font-size:2.1rem}}h2{{margin-top:2.2rem}}
.tag{{color:var(--blue);font-weight:700;letter-spacing:.08em;font-size:.78rem}}.meta{{color:var(--muted);margin-top:.25rem}}
table{{border-collapse:collapse;width:100%;margin-top:12px}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}
th{{color:#475467;width:34%}}.status{{display:inline-block;background:#e8f1f8;color:#174a75;padding:.15rem .45rem;border-radius:999px;white-space:nowrap}}
.gallery{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}}figure{{margin:0;border:1px solid var(--line);border-radius:12px;overflow:hidden;background:#fff}}
img{{display:block;width:100%;background:#fff}}figcaption{{display:flex;flex-direction:column;gap:.2rem;padding:12px 14px}}figcaption span{{color:var(--muted);font-size:.88rem}}
@media(max-width:760px){{main{{padding:20px}}.gallery{{grid-template-columns:1fr}}th{{width:42%}}}}
@media print{{main{{max-width:none;padding:12mm}}figure{{break-inside:avoid}}}}
</style></head><body><main><div class="tag">DIAGNÓSTICO TOPOGRÁFICO</div><h1>{escape(name)}</h1>
<p class="meta">Informe integral de relieve, pendiente, escurrimiento y contexto disponible</p>
<h2>Indicadores principales</h2><table>{summary_rows}</table>
<h2>Preguntas para decisión</h2><table>{answer_rows}</table>
<h2>Mapas y gráficos</h2><div class="gallery">{gallery}</div>
<h2>Calidad y procedencia</h2><table>{quality_rows}</table>
</main></body></html>"""
    return document.encode("utf-8")


def result_zip(
    feature: dict,
    frame: pd.DataFrame | None,
    summary: dict,
    html: bytes,
    files: dict[str, bytes] | None = None,
) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("resumen.json", json.dumps(summary, ensure_ascii=False, indent=2))
        archive.writestr(
            "lote.geojson",
            json.dumps({"type": "FeatureCollection", "features": [feature]}, ensure_ascii=False),
        )
        archive.writestr("informe_completo.html", html)
        for filename, content in (files or {}).items():
            archive.writestr(filename, content)
        if frame is not None:
            columns = ["lon", "lat", "elevacion_m", "fuente", "campana", "cultivo", "lote_sima"]
            archive.writestr("puntos_dentro_lote.csv", frame.loc[frame["dentro_poligono"], columns].to_csv(index=False))
    return output.getvalue()
