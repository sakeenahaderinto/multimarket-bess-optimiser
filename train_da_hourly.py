"""Build native-hourly DA features and train the hourly DA forecaster."""

import logging

import pandas as pd

from config import settings
from features.pipeline_da_hourly import build_da_hourly_features
from forecasting.da_prices_hourly import DAPricesForecasterHourly

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_and_save_features() -> pd.DataFrame:
    bm_df = pd.read_parquet(settings.data_dir / "processed" / "bmrs_processed.parquet")
    da_df = pd.read_parquet(settings.data_dir / "processed" / "day_ahead_processed.parquet")

    weather_files = sorted((settings.data_dir / "raw" / "weather").glob("*.parquet"))
    if not weather_files:
        raise FileNotFoundError("No weather parquet files were found")
    weather_df = pd.concat(
        [pd.read_parquet(path) for path in weather_files],
        ignore_index=True,
    )

    da_df["datetime"] = pd.to_datetime(da_df["datetime"], utc=True)
    da_df = da_df.set_index("datetime").sort_index()
    da_df = da_df[~da_df.index.duplicated(keep="last")]

    bm_df["startTime"] = pd.to_datetime(bm_df["startTime"], utc=True)
    bm_df = bm_df.set_index("startTime").sort_index()
    bm_df = bm_df[~bm_df.index.duplicated(keep="last")]

    weather_df["date"] = pd.to_datetime(weather_df["date"], utc=True)
    weather_df = weather_df.set_index("date").sort_index()
    weather_df = weather_df[~weather_df.index.duplicated(keep="last")]

    features = build_da_hourly_features(
        da_df_hourly=da_df,
        accepted_offer_volume_30min=bm_df["totalAcceptedOfferVolume"],
        bm_imbalance_30min=bm_df["netImbalanceVolume"],
        bm_sell_price_30min=bm_df["systemSellPrice"],
        weather_df=weather_df,
    )

    output_path = settings.data_dir / "features" / "features_da_hourly.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, engine="pyarrow")
    logger.info("Hourly DA features written to %s with shape %s", output_path, features.shape)
    return features


def main() -> None:
    build_and_save_features()
    DAPricesForecasterHourly().run()


if __name__ == "__main__":
    main()
