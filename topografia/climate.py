from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import requests


ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


@dataclass(frozen=True)
class RainfallSummary:
    start_date: str
    end_date: str
    annual_mean_mm: float
    wet_days_mean: float
    heavy_days_mean: float
    max_daily_mm: float
    p95_wet_day_mm: float
    annual: tuple[dict, ...]
    monthly: tuple[dict, ...]
    source_url: str
    model: str


def fetch_rainfall(latitude: float, longitude: float, years: int = 10, timeout_seconds: int = 90) -> RainfallSummary:
    final_year = date.today().year - 1
    first_year = final_year - years + 1
    response = requests.get(
        ARCHIVE_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "start_date": f"{first_year}-01-01",
            "end_date": f"{final_year}-12-31",
            "daily": "precipitation_sum",
            "timezone": "auto",
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    daily = payload.get("daily", {})
    if not daily.get("time") or not daily.get("precipitation_sum"):
        raise ValueError("Open-Meteo no devolvió lluvia histórica para la ubicación.")
    frame = pd.DataFrame({"fecha": pd.to_datetime(daily["time"]), "precipitacion_mm": daily["precipitation_sum"]})
    frame["precipitacion_mm"] = pd.to_numeric(frame["precipitacion_mm"], errors="coerce")
    frame = frame.dropna()
    if frame.empty:
        raise ValueError("La serie de lluvia histórica está vacía.")
    frame["anio"] = frame["fecha"].dt.year
    frame["mes"] = frame["fecha"].dt.month
    annual = frame.groupby("anio").agg(
        precipitacion_mm=("precipitacion_mm", "sum"),
        dias_lluvia=("precipitacion_mm", lambda values: int((values >= 1).sum())),
        dias_mayor_30mm=("precipitacion_mm", lambda values: int((values >= 30).sum())),
        maximo_diario_mm=("precipitacion_mm", "max"),
    ).reset_index()
    monthly = frame.groupby("mes", as_index=False)["precipitacion_mm"].sum()
    monthly["precipitacion_media_mm"] = monthly["precipitacion_mm"] / annual["anio"].nunique()
    wet = frame.loc[frame["precipitacion_mm"] >= 1, "precipitacion_mm"]
    return RainfallSummary(
        start_date=frame["fecha"].min().date().isoformat(),
        end_date=frame["fecha"].max().date().isoformat(),
        annual_mean_mm=float(annual["precipitacion_mm"].mean()),
        wet_days_mean=float(annual["dias_lluvia"].mean()),
        heavy_days_mean=float(annual["dias_mayor_30mm"].mean()),
        max_daily_mm=float(frame["precipitacion_mm"].max()),
        p95_wet_day_mm=float(np.nanpercentile(wet, 95)) if len(wet) else 0.0,
        annual=tuple(annual.round(1).to_dict("records")),
        monthly=tuple(monthly[["mes", "precipitacion_media_mm"]].round(1).to_dict("records")),
        source_url="https://open-meteo.com/en/docs/historical-weather-api",
        model="Open-Meteo Best Match · ERA5, ERA5-Land e IFS",
    )
