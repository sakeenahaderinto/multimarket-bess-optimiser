import numpy as np
import pandas as pd
import pytest

from features.point_in_time import build_bm_asof_features


def _row(start, created, volume, price, imbalance):
    return {
        "startTime": start,
        "createdDateTime": created,
        "totalAcceptedOfferVolume": volume,
        "systemSellPrice": price,
        "netImbalanceVolume": imbalance,
    }


def test_later_publications_cannot_change_earlier_features():
    delivery_index = pd.date_range("2025-01-03", periods=48, freq="30min", tz="UTC")
    records = pd.DataFrame(
        [
            _row("2025-01-02 07:30Z", "2025-01-02 08:20Z", 4.0, 40.0, 400.0),
            _row("2025-01-02 08:00Z", "2025-01-02 08:50Z", 5.0, 50.0, 500.0),
            # This period and the revision above are both published after the
            # 09:00 D-1 origin and must not affect any feature for 3 January.
            _row("2025-01-02 08:30Z", "2025-01-02 09:20Z", 6.0, 60.0, 600.0),
            _row("2025-01-02 08:00Z", "2025-01-02 10:00Z", 999.0, 999.0, 999.0),
        ]
    )

    before = build_bm_asof_features(
        records,
        delivery_index,
        resolution="30min",
        snapshot_lag=48,
        long_lags=(),
        rolling_windows=(2,),
    )
    changed = records.copy()
    late = pd.to_datetime(changed["createdDateTime"], utc=True) > pd.Timestamp(
        "2025-01-02 09:00Z"
    )
    changed.loc[late, [
        "totalAcceptedOfferVolume",
        "systemSellPrice",
        "netImbalanceVolume",
    ]] = 1_000_000.0
    after = build_bm_asof_features(
        changed,
        delivery_index,
        resolution="30min",
        snapshot_lag=48,
        long_lags=(),
        rolling_windows=(2,),
    )

    pd.testing.assert_frame_equal(before, after)
    assert (before["accepted_offer_volume_lag_48"] == 5.0).all()
    assert (before["bm_price_lag_48"] == 50.0).all()
    assert (before["bm_imbalance_volume_lag_48"] == 500.0).all()
    assert (
        before["bm_snapshot_start_time"] == pd.Timestamp("2025-01-02 08:00Z")
    ).all()
    assert (
        before["bm_feature_max_created_time"] <= before["bm_feature_origin"]
    ).all()


def test_latest_published_vintage_is_used_for_an_exact_long_lag():
    delivery_index = pd.DatetimeIndex([pd.Timestamp("2025-01-03 00:00Z")])
    records = pd.DataFrame(
        [
            _row("2025-01-01 00:00Z", "2025-01-01 01:00Z", 5.0, 50.0, 500.0),
            _row("2025-01-01 00:00Z", "2025-01-02 10:00Z", 9.0, 90.0, 900.0),
            _row("2025-01-02 08:00Z", "2025-01-02 08:50Z", 7.0, 70.0, 700.0),
        ]
    )
    features = build_bm_asof_features(
        records,
        delivery_index,
        resolution="30min",
        snapshot_lag=48,
        long_lags=(96,),
        rolling_windows=(),
    )

    assert features.iloc[0]["accepted_offer_volume_lag_96"] == 5.0
    assert features.iloc[0]["bm_price_lag_96"] == 50.0


def test_hourly_snapshot_requires_both_published_half_hours():
    delivery_index = pd.date_range("2025-01-03", periods=24, freq="h", tz="UTC")
    records = pd.DataFrame(
        [
            _row("2025-01-02 07:00Z", "2025-01-02 07:40Z", 2.0, 40.0, 20.0),
            _row("2025-01-02 07:30Z", "2025-01-02 08:10Z", 3.0, 60.0, 30.0),
            _row("2025-01-02 08:00Z", "2025-01-02 08:50Z", 4.0, 80.0, 40.0),
            _row("2025-01-02 08:30Z", "2025-01-02 09:20Z", 5.0, 100.0, 50.0),
        ]
    )
    features = build_bm_asof_features(
        records,
        delivery_index,
        resolution="1h",
        snapshot_lag=24,
        long_lags=(),
        rolling_windows=(),
    )

    assert (features["accepted_offer_volume_lag_24"] == 5.0).all()
    assert (features["bm_price_lag_24"] == 50.0).all()
    assert (features["bm_imbalance_volume_lag_24"] == 50.0).all()
    assert (
        features["bm_snapshot_start_time"] == pd.Timestamp("2025-01-02 07:00Z")
    ).all()


def test_publication_metadata_is_required():
    records = pd.DataFrame(
        {
            "startTime": ["2025-01-01 00:00Z"],
            "totalAcceptedOfferVolume": [1.0],
            "systemSellPrice": [2.0],
            "netImbalanceVolume": [3.0],
        }
    )
    with pytest.raises(ValueError, match="publication metadata"):
        build_bm_asof_features(
            records,
            pd.DatetimeIndex([pd.Timestamp("2025-01-03 00:00Z")]),
            resolution="30min",
            snapshot_lag=48,
            long_lags=(),
            rolling_windows=(),
        )


def test_unavailable_exact_lag_is_missing_not_backfilled():
    delivery_index = pd.DatetimeIndex([pd.Timestamp("2025-01-03 00:00Z")])
    records = pd.DataFrame(
        [
            _row("2025-01-01 00:00Z", "2025-01-02 10:00Z", 9.0, 90.0, 900.0),
            _row("2025-01-02 08:00Z", "2025-01-02 08:50Z", 7.0, 70.0, 700.0),
        ]
    )
    features = build_bm_asof_features(
        records,
        delivery_index,
        resolution="30min",
        snapshot_lag=48,
        long_lags=(96,),
        rolling_windows=(),
    )

    assert np.isnan(features.iloc[0]["accepted_offer_volume_lag_96"])
    assert np.isnan(features.iloc[0]["bm_price_lag_96"])
