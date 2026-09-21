from __future__ import annotations

from datetime import UTC, datetime
import logging

import httpx
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)
ZONE = "GB"


def fetch_electricity_maps(timestamp: datetime | None = None) -> pd.DataFrame:
    target = timestamp or datetime.now(UTC)
    if target.tzinfo is None:
        target = target.replace(tzinfo=UTC)
    target_string = target.astimezone(UTC).strftime("%Y-%m-%dT%H:00:00Z")
    response = httpx.get(
        "https://api.electricitymaps.com/v3/price-day-ahead/past",
        params={
            "zone": ZONE,
            "datetime": target_string,
            "temporalGranularity": "hourly",
        },
        headers={"auth-token": settings.em_api_key},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or "value" not in payload:
        raise ValueError(f"Unexpected Electricity Maps response: {payload!r}")
    frame = pd.DataFrame([payload])
    path = settings.data_dir / "raw" / "em_day_ahead" / (
        f"em_day_ahead_{target.astimezone(UTC).strftime('%Y%m%d%H')}.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, engine="pyarrow", index=False)
    logger.info("Day-ahead data written to %s", path)
    return frame
