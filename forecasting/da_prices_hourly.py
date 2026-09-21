"""Native-hourly day-ahead price forecaster."""

import pandas as pd

from config import settings
from forecasting.base import BaseForecaster

DA_HOURLY_FEATURES = [
    "hour_sin",
    "hour_cos",
    "dayofweek_sin",
    "dayofweek_cos",
    "year_sin",
    "year_cos",
    "accepted_offer_volume_lag_24",
    "accepted_offer_volume_lag_48",
    "accepted_offer_volume_lag_72",
    "accepted_offer_volume_lag_168",
    "price_lag_24",
    "price_lag_48",
    "price_lag_72",
    "price_lag_168",
    "accepted_offer_volume_roll_mean_4",
    "accepted_offer_volume_roll_std_4",
    "price_roll_mean_4",
    "price_roll_std_4",
    "accepted_offer_volume_roll_mean_24",
    "accepted_offer_volume_roll_std_24",
    "price_roll_mean_24",
    "price_roll_std_24",
    "accepted_offer_volume_roll_mean_168",
    "accepted_offer_volume_roll_std_168",
    "price_roll_mean_168",
    "price_roll_std_168",
    "wind_speed",
    "cloud_cover",
    "bm_imbalance_volume_lag_24",
    "wind_change_lag_24",
    "spread_lag_24",
]


class DAPricesForecasterHourly(BaseForecaster):
    model_id = "da_hourly"
    target_col = "price"
    feature_cols = DA_HOURLY_FEATURES

    MIN_TRAIN = 24 * 28
    FOLD_STEP = 24 * 7
    VAL_WINDOW = 24 * 7
    MIN_FOLD_ROWS = 24 * 7

    GATE_ORIGIN_COLS: frozenset[str] = frozenset(
        {
            "accepted_offer_volume_lag_24",
            "bm_imbalance_volume_lag_24",
            "wind_change_lag_24",
            "spread_lag_24",
        }
    )

    def load_features(self) -> pd.DataFrame:
        return pd.read_parquet(
            settings.data_dir / "features" / "features_da_hourly.parquet"
        )
