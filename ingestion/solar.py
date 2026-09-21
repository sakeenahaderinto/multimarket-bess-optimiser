from __future__ import annotations

from datetime import UTC, datetime, timedelta
import logging

import httpx
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)
BASE_URL = "https://api.pvlive.uk/pvlive/api/v4"


def fetch_solar(hours: int = 2) -> pd.DataFrame:
    if hours <= 0:
        raise ValueError("hours must be positive")
    end = datetime.now(UTC)
    start = end - timedelta(hours=hours)
    response = httpx.get(
        f"{BASE_URL}/gsp/0",
        params={"start": start.isoformat(), "end": end.isoformat()},
        headers={"Accept-Encoding": "gzip, deflate"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    data = payload.get("data")
    meta = payload.get("meta")
    if not isinstance(data, list) or not isinstance(meta, list):
        raise ValueError("Unexpected PVLive response")
    frame = pd.DataFrame(data, columns=meta)
    if frame.empty:
        raise RuntimeError("PVLive returned no rows")
    path = settings.data_dir / "raw" / "solar" / f"solar_{end:%Y%m%d%H}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, engine="pyarrow", index=False)
    logger.info("Solar data written to %s", path)
    return frame
