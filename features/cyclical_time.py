import numpy as np
import pandas as pd


def add_cyclical_time(df: pd.DataFrame) -> pd.DataFrame:
    """Add half-hour-aware hour, weekday and annual cyclical features."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame index must be a DatetimeIndex.")
    fractional_hour = df.index.hour + df.index.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * fractional_hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * fractional_hour / 24.0)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7.0)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7.0)
    df["year_sin"] = np.sin(2 * np.pi * df.index.dayofyear / 365.2425)
    df["year_cos"] = np.cos(2 * np.pi * df.index.dayofyear / 365.2425)
    return df
