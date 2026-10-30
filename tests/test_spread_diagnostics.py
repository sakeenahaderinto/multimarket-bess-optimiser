import numpy as np
import pandas as pd

from check_q50_spread_sign import _complete_interval_spreads, _volatility_groups


def test_spread_diagnostics_exclude_partial_blocks_and_days():
    complete = pd.date_range("2025-01-01", periods=48, freq="30min", tz="UTC")
    incomplete = pd.date_range("2025-01-02", periods=47, freq="30min", tz="UTC")
    index = complete.append(incomplete)
    aligned = pd.DataFrame(
        {
            "actual": np.arange(len(index), dtype=float),
            "forecast": np.arange(len(index), dtype=float),
        },
        index=index,
    )

    blocks, days = _complete_interval_spreads(aligned)

    assert days.index.tolist() == [pd.Timestamp("2025-01-01", tz="UTC")]
    assert days.iloc[0]["actual"] == np.mean(np.arange(48, dtype=float))
    assert all(timestamp.minute == 0 for timestamp in blocks.index)
    assert len(blocks) == 11


def test_volatility_groups_collapse_tied_quantiles_without_crashing():
    volatility = pd.Series(
        [1.0, 1.0, 2.0, 2.0],
        index=pd.date_range("2025-01-01", periods=4, freq="D", tz="UTC"),
    )

    groups = _volatility_groups(volatility)

    assert groups.tolist() == [
        "Q1-Q2 calm",
        "Q1-Q2 calm",
        "Q3-Q4 volatile",
        "Q3-Q4 volatile",
    ]
    assert groups.cat.categories.tolist() == ["Q1-Q2 calm", "Q3-Q4 volatile"]
    assert groups.cat.ordered


def test_volatility_groups_report_one_group_when_all_days_are_tied():
    volatility = pd.Series(
        [3.0, 3.0, 3.0],
        index=pd.date_range("2025-01-01", periods=3, freq="D", tz="UTC"),
    )

    groups = _volatility_groups(volatility)

    assert groups.tolist() == ["All days tied"] * 3
    assert groups.cat.categories.tolist() == ["All days tied"]


def test_volatility_groups_preserve_four_ordered_quartiles():
    volatility = pd.Series(
        [1.0, 2.0, 3.0, 4.0],
        index=pd.date_range("2025-01-01", periods=4, freq="D", tz="UTC"),
    )

    groups = _volatility_groups(volatility)

    assert groups.tolist() == ["Q1 calm", "Q2", "Q3", "Q4 volatile"]
    assert groups.cat.categories.tolist() == [
        "Q1 calm",
        "Q2",
        "Q3",
        "Q4 volatile",
    ]


def test_volatility_groups_require_complete_daily_sample():
    with np.testing.assert_raises_regex(ValueError, "No complete UTC days"):
        _volatility_groups(pd.Series(dtype=float))


def test_volatility_groups_reject_non_finite_values():
    with np.testing.assert_raises_regex(ValueError, "non-finite"):
        _volatility_groups(pd.Series([1.0, np.nan]))
