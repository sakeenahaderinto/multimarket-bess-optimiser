import numpy as np
import pandas as pd

from features.pipeline_da_hourly import add_weather_features_hourly
from features.weather_features import add_weather_features


def _weather(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "temperature_2m": np.arange(len(index), dtype=float) + 10.0,
            "wind_speed_10m": np.arange(len(index), dtype=float) + 5.0,
            "cloud_cover": np.arange(len(index), dtype=float) + 30.0,
            "shortwave_radiation": np.arange(len(index), dtype=float) + 100.0,
        },
        index=index,
    )


def test_weather_features_use_exact_last_complete_hour():
    delivery = pd.DataFrame(
        index=pd.DatetimeIndex([pd.Timestamp("2025-01-02 00:00Z")])
    )
    exact_time = pd.DatetimeIndex([pd.Timestamp("2025-01-01 08:00Z")])

    result = add_weather_features(delivery.copy(), _weather(exact_time))

    assert result.loc[delivery.index[0], "temperature"] == 10.0
    assert result.loc[delivery.index[0], "wind_speed"] == 5.0
    assert result.loc[delivery.index[0], "cloud_cover"] == 30.0
    assert result.loc[delivery.index[0], "irradiance"] == 100.0


def test_weather_features_do_not_forward_fill_stale_observations():
    delivery = pd.DataFrame(
        index=pd.DatetimeIndex([pd.Timestamp("2025-01-02 00:00Z")])
    )
    stale_time = pd.DatetimeIndex([pd.Timestamp("2025-01-01 07:00Z")])

    half_hourly = add_weather_features(delivery.copy(), _weather(stale_time))
    hourly = add_weather_features_hourly(delivery.copy(), _weather(stale_time))

    assert half_hourly[["temperature", "wind_speed", "cloud_cover", "irradiance"]].isna().all().all()
    assert hourly[["wind_speed", "cloud_cover"]].isna().all().all()
