from __future__ import annotations

from datetime import UTC, datetime
import logging

import httpx
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)
URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARS = ["temperature_2m", "shortwave_radiation", "wind_speed_10m", "cloud_cover"]


def fetch_weather() -> pd.DataFrame:
    response = httpx.get(
        URL,
        params={
            "latitude": 51.51,
            "longitude": -0.13,
            "hourly": ",".join(HOURLY_VARS),
            "timezone": "UTC",
            "past_days": 7,
            "forecast_days": 3,
        },
        timeout=30,
    )
    response.raise_for_status()
    hourly = response.json().get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        raise RuntimeError("Open-Meteo returned no hourly times")
    data = {"date": pd.to_datetime(times, utc=True)}
    for variable in HOURLY_VARS:
        values = hourly.get(variable)
        if values is None or len(values) != len(times):
            raise ValueError(f"Open-Meteo variable {variable} is missing or mis-sized")
        data[variable] = values
    frame = pd.DataFrame(data)
    path = settings.data_dir / "raw" / "weather" / f"weather_{datetime.now(UTC):%Y%m%d%H}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, engine="pyarrow", index=False)
    logger.info("Weather data written to %s", path)
    return frame
