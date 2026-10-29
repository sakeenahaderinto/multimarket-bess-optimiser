import numpy as np
import pandas as pd
import pytest

from baselines.scenarios import build_median_scenario


@pytest.mark.parametrize(
    ("window_start", "expected_dc"),
    [("2023-10-15", 0.0), ("2025-01-15", -2.0)],
)
def test_median_baseline_applies_dc_price_regime(window_start: str, expected_dc: float):
    start = pd.Timestamp(window_start, tz="UTC")
    index = pd.date_range(start, periods=8, freq="30min")
    energy = pd.DataFrame({"q50": np.full(len(index), 50.0)}, index=index)
    negative_dc = pd.DataFrame({"q50": np.full(len(index), -2.0)}, index=index)

    output = build_median_scenario(
        start,
        len(index),
        energy,
        energy,
        negative_dc,
        negative_dc,
    )

    assert np.all(np.asarray(output["dc_low"]) == expected_dc)
    assert np.all(np.asarray(output["dc_high"]) == expected_dc)
