import logging

from forecasting.bm_da_spread import BMDASpreadForecaster

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


if __name__ == "__main__":
    BMDASpreadForecaster().run()
