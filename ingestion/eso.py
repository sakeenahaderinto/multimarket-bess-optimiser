from __future__ import annotations

from datetime import UTC, datetime, timedelta
import logging

import httpx
import pandas as pd

from config import settings

logger = logging.getLogger(__name__)
RESOURCE_ID = "596f29ac-0387-4ba4-a6d3-95c243140707"


def fetch_eso(days: int = 7) -> pd.DataFrame:
    if days <= 0:
        raise ValueError("days must be positive")
    start = (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%S")
    sql = (
        f'SELECT * FROM "{RESOURCE_ID}" '
        f'WHERE "deliveryStart" >= \'{start}\' '
        'ORDER BY "deliveryStart" DESC'
    )
    response = httpx.get(
        "https://api.neso.energy/api/3/action/datastore_search_sql",
        params={"sql": sql},
        timeout=60,
    )
    response.raise_for_status()
    rows = response.json().get("result", {}).get("records", [])
    if not rows:
        raise RuntimeError("NESO returned no auction rows")
    frame = pd.DataFrame(rows)
    path = settings.data_dir / "raw" / "eso" / f"eso_{datetime.now(UTC):%Y%m%d%H}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, engine="pyarrow", index=False)
    logger.info("NESO data written to %s", path)
    return frame
