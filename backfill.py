"""Historical-data backfill with chunking and explicit partial-failure checks."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
import logging
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from config import settings
from ingestion.weather import HOURLY_VARS

logger = logging.getLogger(__name__)
START_DATE = date(2020, 1, 1)
END_DATE = date(2025, 12, 31)


def _output_path(*parts: str) -> Path:
    path = settings.data_dir.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


async def _get_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    attempts: int = 4,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            last_error = exc
            if attempt == attempts - 1:
                break
            await asyncio.sleep(2**attempt)
    raise RuntimeError(f"Request failed after {attempts} attempts: {url}") from last_error


async def backfill_day_ahead() -> None:
    records: list[dict] = []
    failures: list[str] = []
    current = START_DATE
    headers = {"auth-token": settings.em_api_key}

    async with httpx.AsyncClient(timeout=60.0) as client:
        while current <= END_DATE:
            chunk_end = min(current + timedelta(days=9), END_DATE)
            params = {
                "zone": "GB",
                "start": f"{current.isoformat()}T00:00:00.000Z",
                "end": f"{chunk_end.isoformat()}T23:59:59.999Z",
            }
            try:
                payload = await _get_json(
                    client,
                    "https://api.electricitymaps.com/v3/price-day-ahead/past-range",
                    params=params,
                    headers=headers,
                )
                chunk = payload.get("data", []) if isinstance(payload, dict) else []
                if not isinstance(chunk, list):
                    raise ValueError("Electricity Maps data field is not a list")
                records.extend(chunk)
                logger.info("Day-ahead %s to %s: %d rows", current, chunk_end, len(chunk))
            except Exception as exc:
                label = f"{current} to {chunk_end}: {exc}"
                failures.append(label)
                logger.exception("Day-ahead chunk failed: %s", label)
            current = chunk_end + timedelta(days=1)
            await asyncio.sleep(0.25)

    if failures:
        raise RuntimeError(
            f"Day-ahead backfill had {len(failures)} failed chunks. First: {failures[0]}"
        )
    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError("Day-ahead backfill returned no rows")
    frame.to_parquet(
        _output_path("raw", "em_day_ahead", "em_day_ahead_backfill.parquet"),
        engine="pyarrow",
        index=False,
    )


async def backfill_bmrs() -> None:
    failures: list[str] = []
    current_year = START_DATE.year
    yearly_frames: list[pd.DataFrame] = []

    def write_year(year: int, frames: list[pd.DataFrame]) -> None:
        if not frames:
            raise RuntimeError(f"BMRS produced no rows for {year}")
        frame = pd.concat(frames, ignore_index=True)
        frame.to_parquet(
            _output_path("raw", "bmrs", f"bmrs_{year}.parquet"),
            engine="pyarrow",
            index=False,
        )
        logger.info("BMRS %d written with %d rows", year, len(frame))

    async with httpx.AsyncClient(timeout=45.0) as client:
        current = START_DATE
        while current <= END_DATE:
            if current.year != current_year:
                write_year(current_year, yearly_frames)
                current_year = current.year
                yearly_frames = []

            try:
                payload = await _get_json(
                    client,
                    "https://data.elexon.co.uk/bmrs/api/v1/balancing/settlement/system-prices/"
                    f"{current.isoformat()}",
                    params={"format": "json"},
                )
                rows = payload.get("data", []) if isinstance(payload, dict) else []
                if rows:
                    yearly_frames.append(pd.DataFrame(rows).convert_dtypes())
                else:
                    failures.append(f"{current}: empty response")
            except Exception as exc:
                failures.append(f"{current}: {exc}")
                logger.exception("BMRS day failed: %s", current)

            current += timedelta(days=1)
            await asyncio.sleep(0.05)

    write_year(current_year, yearly_frames)
    if failures:
        raise RuntimeError(f"BMRS had {len(failures)} failed or empty days. First: {failures[0]}")


async def backfill_weather() -> None:
    frames: list[pd.DataFrame] = []
    failures: list[str] = []
    current = START_DATE

    async with httpx.AsyncClient(timeout=90.0) as client:
        while current <= END_DATE:
            chunk_end = min(current + timedelta(days=365), END_DATE)
            params = {
                "latitude": 51.51,
                "longitude": -0.13,
                "hourly": ",".join(HOURLY_VARS),
                "timezone": "UTC",
                "start_date": current.isoformat(),
                "end_date": chunk_end.isoformat(),
            }
            try:
                payload = await _get_json(
                    client,
                    "https://archive-api.open-meteo.com/v1/archive",
                    params=params,
                )
                hourly = payload.get("hourly", {})
                times = hourly.get("time", [])
                if not times:
                    raise ValueError("Open-Meteo response has no hourly times")
                data = {"date": pd.to_datetime(times, utc=True)}
                for variable in HOURLY_VARS:
                    values = hourly.get(variable)
                    if values is None or len(values) != len(times):
                        raise ValueError(f"Open-Meteo variable {variable} is missing or mis-sized")
                    data[variable] = values
                frames.append(pd.DataFrame(data))
            except Exception as exc:
                failures.append(f"{current} to {chunk_end}: {exc}")
                logger.exception("Weather chunk failed")
            current = chunk_end + timedelta(days=1)

    if failures:
        raise RuntimeError(f"Weather had {len(failures)} failed chunks. First: {failures[0]}")
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.drop_duplicates("date", keep="last").sort_values("date")
    frame.to_parquet(
        _output_path("raw", "weather", "weather_backfill.parquet"),
        engine="pyarrow",
        index=False,
    )


async def backfill_solar() -> None:
    frames: list[pd.DataFrame] = []
    failures: list[str] = []
    current = START_DATE

    async with httpx.AsyncClient(timeout=90.0) as client:
        while current <= END_DATE:
            chunk_end = min(current + timedelta(days=30), END_DATE)
            params = {
                "start": f"{current.isoformat()}T00:00:00Z",
                "end": f"{chunk_end.isoformat()}T23:30:00Z",
            }
            try:
                payload = await _get_json(
                    client,
                    "https://api.pvlive.uk/pvlive/api/v4/gsp/0",
                    params=params,
                    headers={"Accept-Encoding": "gzip, deflate"},
                )
                data = payload.get("data")
                meta = payload.get("meta")
                if not isinstance(data, list) or not isinstance(meta, list):
                    raise ValueError("PVLive response does not contain data and meta lists")
                frames.append(pd.DataFrame(data, columns=meta))
            except Exception as exc:
                failures.append(f"{current} to {chunk_end}: {exc}")
                logger.exception("Solar chunk failed")
            current = chunk_end + timedelta(days=1)
            await asyncio.sleep(0.1)

    if failures:
        raise RuntimeError(f"Solar had {len(failures)} failed chunks. First: {failures[0]}")
    frame = pd.concat(frames, ignore_index=True)
    frame.to_parquet(
        _output_path("raw", "solar", "solar_backfill.parquet"),
        engine="pyarrow",
        index=False,
    )


async def backfill_eso() -> None:
    resource_id = "596f29ac-0387-4ba4-a6d3-95c243140707"
    limit = 10_000
    offset = 0
    rows: list[dict] = []

    async with httpx.AsyncClient(timeout=90.0) as client:
        while True:
            sql = (
                f'SELECT * FROM "{resource_id}" '
                'ORDER BY "deliveryStart" ASC '
                f"LIMIT {limit} OFFSET {offset}"
            )
            payload = await _get_json(
                client,
                "https://api.neso.energy/api/3/action/datastore_search_sql",
                params={"sql": sql},
            )
            chunk = payload.get("result", {}).get("records", [])
            if not isinstance(chunk, list):
                raise ValueError("NESO response records field is not a list")
            rows.extend(chunk)
            logger.info("NESO offset %d: %d rows", offset, len(chunk))
            if len(chunk) < limit:
                break
            offset += limit

    if not rows:
        raise RuntimeError("NESO backfill returned no rows")
    pd.DataFrame(rows).to_parquet(
        _output_path("raw", "eso", "eso_backfill.parquet"),
        engine="pyarrow",
        index=False,
    )


async def main() -> None:
    if settings.em_api_key in {"", "changeme-set-in-env"}:
        raise RuntimeError("Set EM_API_KEY in .env before running the backfill")

    for name, function in (
        ("day-ahead", backfill_day_ahead),
        ("weather", backfill_weather),
        ("solar", backfill_solar),
        ("NESO", backfill_eso),
        ("BMRS", backfill_bmrs),
    ):
        logger.info("Starting %s backfill", name)
        await function()
    logger.info("Backfill complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(main())
