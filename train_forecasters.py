import logging

from features.pipeline import build_features
from forecasting.bm_price import BMPriceForecaster
from forecasting.da_prices import DAPricesForecaster
from forecasting.dc_tender import DCHighForecaster, DCLowForecaster

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def main() -> None:
    build_features()
    DAPricesForecaster().run()
    BMPriceForecaster().run()
    DCLowForecaster().run()
    DCHighForecaster().run()


if __name__ == "__main__":
    main()
