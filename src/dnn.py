# just a test right now
from channels.alpaca import Alpaca
import pandas as pd, numpy as np
from finta import TA as ta
import logging, os
from tqdm import trange, tqdm

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
        mv_avgs: list = [5, 10, 20, 30, 50, 100, 200],
    ) -> None:
        """dfcols is the labels for the dataframe that you recieve"""
        self.close = dfcols["close"]
        self.open = dfcols["open"]
        self.low = dfcols["low"]
        self.high = dfcols["high"]
        self.volume = dfcols["volume"]
        self.mv_avgs = mv_avgs

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

    def _append_data(
        self, maindf: pd.DataFrame, dataarray: list, namesarray: list = None
    ):
        if namesarray == None:
            return maindf.join(pd.DataFrame(dataarray), how="outer")
        return maindf.join(pd.DataFrame(dataarray, columns=namesarray), how="outer")

    def _create_features(self, data: pd.DataFrame):
        # create copy
        df = data.copy()
        try:
            ### Oscillators ###
            ## RSI
            df = self._append_data(df, ta.RSI(df))
            ## Sto-%K
            df = self._append_data(df, ta.STOCH(df))
            ## CCI
            df = self._append_data(df, ta.CCI(df))
            ## ADX
            df = self._append_data(df, ta.ADX(df))
            ## DMI (Added to aid in interpreting ADX)
            df = self._append_data(df, ta.DMI(df, 14))
            ## Awesome
            df = self._append_data(df, ta.AO(df))
            ## Momentum
            df = self._append_data(df, ta.MOM(df, 10))
            ## MACD (We rename the undescriptive "SIGNAL" here)
            df = self._append_data(df, ta.MACD(df)).rename(
                columns={"SIGNAL": "MACD SIGNAL"}
            )
            ## Sto-RSI
            df = self._append_data(df, ta.STOCHRSI(df))
            ## Williams %R
            df = self._append_data(df, ta.WILLIAMS(df))
            ## Bull-Bear Power
            df = self._append_data(df, ta.EBBP(df))
            ## Ultimate (FinTA does not name this column, so we must)
            df = self._append_data(df, ta.UO(df), ["UO"])

            ### Moving Averages ###
            for i in self.mv_avgs:
                df = self._append_data(df, ta.SMA(df, i))
                df = self._append_data(df, ta.EMA(df, i))
            ## VWMA
            df = self._append_data(df, ta.VAMA(df, 20))
            ## Hull
            df = self._append_data(df, ta.HMA(df, 9))
            # Ichimoku -- Base (Kijun) and Conversion (Tenkan) Only
            df = self._append_data(
                df, ta.ICHIMOKU(df).drop(["senkou_span_a", "SENKOU", "CHIKOU"], axis=1)
            )
            ### Custom
            df["momentum"] = df["return"].rolling(5).mean().shift(1)
            df["volatility"] = df["return"].rolling(20).std().shift(1)
            df["distance"] = (df[self.close] - df[self.close].rolling(50).mean()).shift(
                1
            )
            ### drop na
            df.dropna(inplace=True)
        except Exception as e:
            logger.warning(f"Unknown error occured while generating features ({e})")
        return df

    def prime_data(self, data: pd.DataFrame, prune: bool = False):
        """Prime the data for the network"""
        if prune:
            data = data[[self.close]]
        data = self._create_returns(data)
        data = self._create_direction(data)
        data = self._create_lags(data)
        data = self._create_features(data)

        return data


if __name__ == "__main__":
    ape = Alpaca()
    fednn = FEDNN()
    data = ape.get_bars("pg")
    fednn.prime_data(data)
