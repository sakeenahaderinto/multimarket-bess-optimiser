"""Evaluate sign accuracy of the forecast BM-minus-DA spread."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from backtest.engine import _normalise_to_30min
from features.pipeline_da_hourly import upsample_da_forecast
from market_calendar import aggregate_to_complete_utc_days, aggregate_to_efa_blocks
from run_backtest import _load_actual, _load_bm_actual, _load_forecast


def _magnitude_weighted_sign_accuracy(actual: np.ndarray, forecast: np.ndarray) -> float:
    weights = np.abs(actual)
    return float(
        np.sum(weights * (np.sign(actual) == np.sign(forecast))) / np.sum(weights)
    ) if np.sum(weights) > 0 else float("nan")


def _aligned_spreads(da_actual, bm_actual, da_forecast, bm_forecast, spread_forecast):
    actual = bm_actual["value"] - da_actual["value"]
    predicted = (
        spread_forecast["q50"]
        if spread_forecast is not None
        else bm_forecast["q50"] - da_forecast["q50"]
    )
    return pd.concat([actual.rename("actual"), predicted.rename("forecast")], axis=1).dropna()


def _complete_interval_spreads(
    aligned: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate aligned spreads on complete EFA blocks and UTC days."""
    block = aggregate_to_efa_blocks(aligned, complete_only=True)
    daily = aggregate_to_complete_utc_days(aligned)
    daily["actual_volatility"] = (
        aligned["actual"].groupby(aligned.index.normalize()).std().reindex(daily.index)
    )
    return block, daily


def _volatility_groups(volatility: pd.Series) -> pd.Series:
    """Return ordered, tie-safe empirical volatility groups."""
    if volatility.empty:
        raise ValueError("No complete UTC days are available for volatility grouping")
    values = pd.to_numeric(volatility, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Daily spread volatility contains non-finite values")

    if values.nunique() == 1:
        labels = ["All days tied"]
        assigned = pd.Series(labels[0], index=values.index)
    else:
        codes = pd.qcut(values, 4, labels=False, duplicates="drop")
        group_count = int(codes.max()) + 1
        labels_by_count = {
            2: ["Q1-Q2 calm", "Q3-Q4 volatile"],
            3: ["Q1 calm", "Q2-Q3", "Q4 volatile"],
            4: ["Q1 calm", "Q2", "Q3", "Q4 volatile"],
        }
        labels = labels_by_count.get(
            group_count,
            [f"Group {number + 1}" for number in range(group_count)],
        )
        assigned = codes.astype(int).map(dict(enumerate(labels)))

    return pd.Series(
        pd.Categorical(assigned, categories=labels, ordered=True),
        index=values.index,
        name="volatility_quartile",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", choices=["cv", "oos"], default="cv")
    parser.add_argument("--use-hourly-da", action="store_true")
    parser.add_argument("--use-spread-model", action="store_true")
    args = parser.parse_args()

    da_actual = _load_actual("processed/day_ahead_processed.parquet", "value").to_frame("value")
    da_actual = _normalise_to_30min(da_actual, "da_actual", fill_limit=1)
    bm_actual = _load_bm_actual()

    da_forecast = (
        upsample_da_forecast(_load_forecast("da_hourly", args.period))
        if args.use_hourly_da
        else _load_forecast("da", args.period)
    )
    bm_forecast = _load_forecast("bm", args.period)
    spread_forecast = _load_forecast("spread", args.period) if args.use_spread_model else None

    aligned = _aligned_spreads(
        da_actual,
        bm_actual,
        da_forecast,
        bm_forecast,
        spread_forecast,
    )
    if aligned.empty:
        raise RuntimeError("No aligned spread observations were found")

    period_score = _magnitude_weighted_sign_accuracy(
        aligned["actual"].to_numpy(float),
        aligned["forecast"].to_numpy(float),
    )

    block, daily = _complete_interval_spreads(aligned)
    block_score = _magnitude_weighted_sign_accuracy(
        block["actual"].to_numpy(float),
        block["forecast"].to_numpy(float),
    )

    daily_score = _magnitude_weighted_sign_accuracy(
        daily["actual"].to_numpy(float),
        daily["forecast"].to_numpy(float),
    )
    daily["sign_match"] = np.sign(daily["actual"]) == np.sign(daily["forecast"])
    daily["volatility_quartile"] = _volatility_groups(daily["actual_volatility"])

    print(f"Period-level magnitude-weighted sign accuracy: {period_score * 100:.2f}%")
    print(f"EFA-block magnitude-weighted sign accuracy: {block_score * 100:.2f}%")
    print(f"Daily magnitude-weighted sign accuracy: {daily_score * 100:.2f}%")
    print("\nDaily mismatch rate by volatility quartile")
    print(
        daily.groupby("volatility_quartile", observed=True)["sign_match"]
        .agg(["count", "mean"])
        .assign(mismatch_rate_pct=lambda frame: (1 - frame["mean"]) * 100)
        .to_string()
    )


if __name__ == "__main__":
    main()
