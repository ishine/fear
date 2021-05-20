# just a test right now
from channels.alpaca import Alpaca
import pandas as pd, numpy as np

import logging, os

os.environ["NUMEXPR_MAX_THREADS"] = "8"

logging.basicConfig(
    format="[%(name)s] %(levelname)s %(asctime)s %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


class FEDNN:  # feature engineering deep neural network
    def __init__(
        self,
        dfcols: dict = {
            "close": "close",
            "open": "open",
            "low": "low",
            "high": "high",
            "volume": "volume",
        },
    ) -> None:
        """dfcols is the labels for the dataframe that you recieve"""
        self.close = dfcols["close"]
        self.open = dfcols["open"]
        self.low = dfcols["low"]
        self.high = dfcols["high"]
        self.volume = dfcols["volume"]

    def _create_returns(self, data: pd.DataFrame):
        """Create the log returns"""
        try:
            data["return"] = np.log(data[self.close] / data[self.close].shift(1))
        except Exception as e:
            logger.warning(f"Unknown error occured while generating returns ({e})")
        return data

    def _create_direction(self, data: pd.DataFrame):
        """Create direction from returns"""
        try:
            data["direction"] = np.where(data["return"] > 0, 1, 0)
        except Exception as e:
            logger.warning(f"Unknown error occured while generating direction ({e})")
        return data

    def _create_lags(self, data: pd.DataFrame, lags: int = 5):
        """Create the lags"""
        try:
            for lag in range(1, lags + 1):
                col = f"lag_{lag}"
                data[col] = data["return"].shift(lag)
            data.dropna(inplace=True)
        except Exception as e:
            logger.warning(f"Unknown error occured while generating lags ({e})")
        return data

    def _create_features(self, data: pd.DataFrame):
        """Create features for the predictor"""
        try:
            data["momentum"] = data["return"].rolling(5).mean().shift(1)
            data["volatility"] = data["return"].rolling(20).std().shift(1)
            data["distance"] = (
                data[self.close] - data[self.close].rolling(50).mean()
            ).shift(1)
            data.dropna(inplace=True)
        except Exception as e:
            logger.warning(f"Unknown error occured while generating features ({e})")
        return data

    def prime_data(self, data: pd.DataFrame, prune: bool = False):
        """Prime the data for the network"""
        if prune:
            data = data[[self.close]]
        data = self._create_returns(data)
        data = self._create_direction(data)
        data = self._create_lags(data)
        data = self._create_features(data)
        print(data)
        return data


if __name__ == "__main__":
    ape = Alpaca()
    fednn = FEDNN()
    data = ape.get_bars("pg")
    fednn.prime_data(data)
