from __future__ import annotations

import json
from pathlib import Path

from .package_io import TopographyPackage


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _number(value: object, default: float = 0.0) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def package_summary_rows(package: TopographyPackage) -> list[dict]:
    field = package.catalog.get("field", {})
    group_metrics = _read_json(package.root / "metrics" / "group.json")
    if field.get("type") == "group" and group_metrics.get("fields"):
        return [
            {
                "Lote": str(member.get("field_name") or member.get("field_id")),
                "Superficie (ha)": _number(member.get("area_ha")),
                "Relieve robusto (m)": _number(member.get("robust_relief_m")),
                "Pendiente mediana (%)": _number(member.get("slope_median_percent")),
                "Observaciones aceptadas (%)": None,
                "Desacuerdo P90 (m)": None,
            }
            for member in group_metrics["fields"]
        ]

    terrain = _read_json(package.root / "metrics" / "terrain.json")
    stability = _read_json(package.root / "metrics" / "campaign_stability.json")
    quality = _read_json(package.root / "metrics" / "quality_control.json")
    field_name = str(field.get("name") or field.get("id") or "Lote")
    return [
        {
            "Lote": field_name,
            "Superficie (ha)": _number(terrain.get("area_ha")),
            "Relieve robusto (m)": _number(terrain.get("robust_relief_m")),
            "Pendiente mediana (%)": _number(terrain.get("slope_median_percent")),
            "Observaciones aceptadas (%)": _number(quality.get("accepted_percent")),
            "Desacuerdo P90 (m)": _number(stability.get("p90_disagreement_m")),
        }
    ]
