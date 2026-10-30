import pandas as pd

from forecasting.base import BASE_FEATURES, BaseForecaster
from market_calendar import efa_block_keys


class _DCForecaster(BaseForecaster):
    """Keep every repeated EFA-block clearing outcome in one validation side."""

    def _validation_group_label(self, timestamp: pd.Timestamp) -> object:
        return efa_block_keys(pd.DatetimeIndex([timestamp]))[0]


class DCLowForecaster(_DCForecaster):
    model_id = "dc_low"
    target_col = "dc_low_price"
    feature_cols = BASE_FEATURES


class DCHighForecaster(_DCForecaster):
    model_id = "dc_high"
    target_col = "dc_high_price"
    feature_cols = BASE_FEATURES
