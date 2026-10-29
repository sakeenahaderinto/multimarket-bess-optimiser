import numpy as np
import pandas as pd
from scipy.stats import norm

from market_calendar import efa_block_groups
from optimiser.scenarios import (
    _two_piece_normal_quantile,
    sample_scenarios_multimarket,
)


def _forecast(index, centre, width):
    return pd.DataFrame(
        {
            "q10": np.full(len(index), centre - width),
            "q50": np.full(len(index), centre),
            "q90": np.full(len(index), centre + width),
        },
        index=index,
    )


def test_two_piece_normal_matches_supplied_quantiles():
    latent = np.array([[norm.ppf(0.1), 0.0, norm.ppf(0.9)]])
    q10 = np.array([[10.0, 10.0, 10.0]])
    q50 = np.array([[20.0, 20.0, 20.0]])
    q90 = np.array([[40.0, 40.0, 40.0]])
    mapped = _two_piece_normal_quantile(latent, q10, q50, q90)
    assert np.allclose(mapped, [[10.0, 20.0, 40.0]])


def test_scenario_generation_is_reproducible_and_block_constant_for_dc():
    index = pd.date_range("2025-01-15", periods=48, freq="30min", tz="UTC")
    da = _forecast(index, 80.0, 20.0)
    bm = _forecast(index, 100.0, 40.0)
    dc_low = _forecast(index, 5.0, 2.0)
    dc_high = _forecast(index, 4.0, 1.0)
    correlation = np.eye(4)

    first = sample_scenarios_multimarket(
        da,
        bm,
        dc_low,
        dc_high,
        n=12,
        seed=123,
        corr_matrix=correlation,
    )
    second = sample_scenarios_multimarket(
        da,
        bm,
        dc_low,
        dc_high,
        n=12,
        seed=123,
        corr_matrix=correlation,
    )

    for market in ("da", "bm", "dc_low", "dc_high"):
        assert np.asarray(first[market]).shape == (12, 48)
        assert np.allclose(first[market], second[market])
        assert np.isfinite(first[market]).all()

    for start, end in efa_block_groups(index[0], len(index)):
        assert np.allclose(first["dc_low"][:, start:end], first["dc_low"][:, start : start + 1])
        assert np.allclose(first["dc_high"][:, start:end], first["dc_high"][:, start : start + 1])


def test_spread_mode_constructs_bm_as_da_plus_spread():
    index = pd.date_range("2025-01-15", periods=8, freq="30min", tz="UTC")
    da = _forecast(index, 80.0, 20.0)
    unused_bm = _forecast(index, 999.0, 1.0)
    spread = _forecast(index, 15.0, 10.0)
    dc_low = _forecast(index, 5.0, 2.0)
    dc_high = _forecast(index, 4.0, 1.0)

    output = sample_scenarios_multimarket(
        da,
        unused_bm,
        dc_low,
        dc_high,
        n=5,
        seed=1,
        corr_matrix=np.eye(4),
        spread_fc=spread,
    )
    implied_spread = np.asarray(output["bm"]) - np.asarray(output["da"])
    assert np.all(np.isfinite(implied_spread))
    assert np.mean(implied_spread) < 50.0


def test_temporal_correlation_creates_persistent_energy_price_paths():
    index = pd.date_range("2025-01-15", periods=48, freq="30min", tz="UTC")
    forecasts = [_forecast(index, centre, 10.0) for centre in (80.0, 90.0, 5.0, 4.0)]

    output = sample_scenarios_multimarket(
        *forecasts,
        n=2_000,
        seed=9,
        corr_matrix=np.eye(4),
        temporal_correlation=0.8,
    )

    da = np.asarray(output["da"])
    empirical_lag_one = np.corrcoef(da[:, :-1].ravel(), da[:, 1:].ravel())[0, 1]
    assert 0.75 < empirical_lag_one < 0.85


def test_temporal_correlation_must_be_stationary():
    index = pd.date_range("2025-01-15", periods=2, freq="30min", tz="UTC")
    forecasts = [_forecast(index, centre, 10.0) for centre in (80.0, 90.0, 5.0, 4.0)]

    with np.testing.assert_raises_regex(ValueError, "strictly between -1 and 1"):
        sample_scenarios_multimarket(
            *forecasts,
            n=2,
            corr_matrix=np.eye(4),
            temporal_correlation=1.0,
        )


def test_post_eac_dc_scenarios_preserve_negative_prices():
    index = pd.date_range("2025-01-15", periods=8, freq="30min", tz="UTC")
    energy = _forecast(index, 50.0, 0.0)
    negative_dc = _forecast(index, -2.0, 0.0)

    output = sample_scenarios_multimarket(
        energy,
        energy,
        negative_dc,
        negative_dc,
        n=3,
        seed=4,
        corr_matrix=np.eye(4),
    )

    assert np.all(np.asarray(output["dc_low"]) == -2.0)
    assert np.all(np.asarray(output["dc_high"]) == -2.0)


def test_pre_eac_dc_scenarios_retain_historical_zero_floor():
    index = pd.date_range("2023-10-15", periods=8, freq="30min", tz="UTC")
    energy = _forecast(index, 50.0, 0.0)
    negative_dc = _forecast(index, -2.0, 0.0)

    output = sample_scenarios_multimarket(
        energy,
        energy,
        negative_dc,
        negative_dc,
        n=3,
        seed=4,
        corr_matrix=np.eye(4),
    )

    assert np.all(np.asarray(output["dc_low"]) == 0.0)
    assert np.all(np.asarray(output["dc_high"]) == 0.0)
