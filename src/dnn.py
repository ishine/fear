# just a test right now
from keras import optimizers
from keras.engine import training
from channels.alpaca import Alpaca
import pandas as pd, numpy as np
from finta import TA as ta
import logging, os
from tqdm import trange, tqdm
from sklearn import preprocessing
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam, RMSprop

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
        units: int = 32,
        learning_rate: float = 0.0001,
        lags: int = 5,
        epochs=25,
    ) -> None:
        """dfcols is the labels for the dataframe that you recieve"""
        self.close = dfcols["close"]
        self.open = dfcols["open"]
        self.low = dfcols["low"]
        self.high = dfcols["high"]
        self.volume = dfcols["volume"]
        self.mv_avgs = mv_avgs

        # for ml
        self.units = units
        self.optimizer = Adam(learning_rate=learning_rate)

        self.lags = lags  # number of lags
        self.cols = [f"lag_{l}" for l in range(1, self.lags + 1)]  # col labels
        self.epochs = epochs

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

    def _create_lags(self, data: pd.DataFrame):
        """Create the lags"""
        try:
            for lag in range(1, self.lags + 1):
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
            """
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
            """
            ### Custom
            df["momentum"] = df["return"].rolling(5).mean().shift(1)
            df["volatility"] = df["return"].rolling(20).std().shift(1)
            df["distance"] = (df[self.close] - df[self.close].rolling(50).mean()).shift(
                1
            )

            self.cols.extend(["momentum", "volatility", "distance"])
            self.cols = list(set(self.cols))  # remove duplicates
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
        data = self._create_features(data)
        data = self._create_lags(data)
        return data

    def fit_scaler(self, data: pd.DataFrame):
        """Fit the min max scaler - MUST BE CALLED BEFORE SCALING"""
        self.mu, self.std = data.mean(), data.std()

    def normalize(self, data: pd.DataFrame):
        """Normalize the data and return"""
        data = data.copy()
        normalized = (data - self.mu) / self.std
        return normalized

    def build(self, data: pd.DataFrame, summary: bool = False):
        """Compile model"""
        print(self.cols)
        self.fit_scaler(data)
        self.model = Sequential()
        self.model.add(
            Dense(self.units, activation="relu", input_shape=(len(self.cols),))
        )
        self.model.add(Dense(self.units, activation="relu"))
        self.model.add(Dense(1, activation="sigmoid"))
        self.model.compile(
            optimizer=self.optimizer, loss="binary_crossentropy", metrics=["accuracy"]
        )
        if summary:
            self.model.summary()
        return self.model

    def train(self, data: pd.DataFrame):
        """Train the model"""
        data_normalized = self.normalize(data)  # maybe move to outside funct or sum
        self.model.fit(
            data_normalized[self.cols],
            data_normalized["direction"],
            epochs=self.epochs,
            verbose=False,
        )
        self.model.evaluate(data_normalized[self.cols], data["direction"])
        return self.model


if __name__ == "__main__":
    ape = Alpaca()

    symbol = "pg"
    data = ape.get_bars(symbol)
    logger.info(f"Got data for {symbol}")

    fednn = FEDNN()
    # prime data
    data = fednn.prime_data(data)
    # make model
    fednn.build(data, summary=True)
    # train model
    fednn.train(data)
    # predict
    # p = fednn.predict(data)
    # print(p)
