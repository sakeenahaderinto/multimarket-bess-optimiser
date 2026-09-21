from __future__ import annotations

from datetime import UTC, datetime
import logging

import httpx
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)
BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"


def fetch_bmrs(target_date: str | None = None) -> pd.DataFrame:
    date_string = target_date or datetime.now(UTC).strftime("%Y-%m-%d")
    response = httpx.get(
        f"{BASE_URL}/balancing/settlement/system-prices/{date_string}",
        params={"format": "json"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data", [])
    if not rows:
        raise RuntimeError(f"BMRS returned no system-price rows for {date_string}")
    frame = pd.DataFrame(rows).convert_dtypes()
    path = settings.data_dir / "raw" / "bmrs" / f"system_prices_{date_string}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, engine="pyarrow", index=False)
    logger.info("BMRS data written to %s", path)
    return frame
