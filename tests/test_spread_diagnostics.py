import numpy as np
import pandas as pd

from check_q50_spread_sign import _complete_interval_spreads


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
