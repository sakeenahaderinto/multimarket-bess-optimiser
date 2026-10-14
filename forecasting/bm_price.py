from forecasting.base import BASE_FEATURES, BaseForecaster


class BMPriceForecaster(BaseForecaster):
    model_id = "bm"
    target_col = "bm_price"
    TARGET_AVAILABILITY_COL = "bm_target_created_time"
    feature_cols = BASE_FEATURES + ["wind_change_lag_48", "spread_lag_48"]
