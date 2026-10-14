import pandas as pd

from features.pipeline import _latest_bm_targets


def test_latest_bm_target_keeps_value_and_publication_time_from_same_vintage() -> None:
    records = pd.DataFrame(
        {
            "startTime": pd.to_datetime(
                ["2024-01-01 00:00Z", "2024-01-01 00:00Z", "2024-01-01 00:30Z"]
            ),
            "createdDateTime": pd.to_datetime(
                ["2024-01-01 00:50Z", "2024-01-02 00:00Z", "2024-01-01 01:20Z"]
            ),
            "systemSellPrice": [50.0, 75.0, 60.0],
        }
    )

    targets = _latest_bm_targets(records)

    first = targets.loc[pd.Timestamp("2024-01-01 00:00Z")]
    assert first["systemSellPrice"] == 75.0
    assert first["createdDateTime"] == pd.Timestamp("2024-01-02 00:00Z")
    assert targets.index.is_unique
