import logging
import os
import re
from datetime import datetime, timedelta
from inspect import CO_GENERATOR
from random import shuffle
from threading import Thread

from numpy.core.numeric import NaN

from channels.alpaca import Alpaca, TimeFrame
from channels.binanceus import BinanceUS
from channels.screener import Screener

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd
import tensorflow as tf
from finta import TA as ta

# just a test right now
from keras import optimizers
from keras.engine import training
from sklearn import preprocessing
from tqdm import tqdm, trange
from mlbase import BaseTrader

tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(1)

import plotly.express as px
import plotly.graph_objects as go
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras.utils.vis_utils import plot_model
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None  # default='warn'

os.environ["NUMEXPR_MAX_THREADS"] = "8"

logging.basicConfig(
    # filename,
    format="[%(name)s] %(levelname)s %(asctime)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MODELPATH = "models"
CHARTPATH = "chart"


class FEDNN(BaseTrader):  # feature engineering deep neural network
    def __init__(
        self,
        units: int = 64,
        learning_rate: float = 0.0001,
        lags: int = 7,
        cols: list = ["return"],
        epochs: int = 25,
        mv_avgs: list = [5, 10, 20, 30, 50, 100, 200],
    ) -> None:
        super(FEDNN, self).__init__(lags=lags, cols=cols)
        self.mv_avgs = mv_avgs

        # for ml
        self.units = units
        self.optimizer = Adam(learning_rate=learning_rate)

        self.epochs = epochs

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
            df["volume_ma"] = df["volume"].rolling(5).mean().shift(1)
            df["volatility"] = df["return"].rolling(20).std().shift(1)
            df["distance"] = (df["close"] - df["close"].rolling(50).mean()).shift(1)
            # print(df.columns)
            self._add_columns(
                "momentum", "volatility", "distance", "14 period RSI", "volume"
            )
            ### drop na
            df.dropna(inplace=True)
        except Exception as e:
            logger.warning(
                f"Unknown error occured while generating features ({e})", exc_info=True
            )
        return df

    def _create_direction(self, data: pd.DataFrame):
        """Create direction from returns"""
        try:
            data["direction"] = np.where(data["return"] > 0, 1, 0)
        except Exception as e:
            logger.warning(f"Unknown error occured while generating direction ({e})")
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
        data = self.prime_data(data)
        logger.info(f"Building model using features {self.cols}")
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
        # save model

        if not os.path.exists(MODELPATH):
            os.mkdir(MODELPATH)

        self.model.save(os.path.join(MODELPATH, "latest"))
        plot_model(
            self.model,
            to_file=os.path.join(MODELPATH, "model_plot.png"),
            show_shapes=True,
            show_layer_names=True,
        )

        if summary:
            self.model.summary()

    def train(self, data: pd.DataFrame):
        """Train the model"""
        data = self.prime_data(data)
        logger.info(f"Training model on {data.shape[0]} records")
        data_normalized = self.normalize(data)  # maybe move to outside funct or sum
        self.model.fit(
            data_normalized[self.cols],
            data_normalized["direction"],
            epochs=self.epochs,
            verbose=False,
        )
        self.model.evaluate(data_normalized[self.cols], data["direction"])

    def predict(self, data: pd.DataFrame, strict_hold=False):
        """Predict
        Strict hold = introduce hold signals (0) as well, not just 1 and -1
        """
        data = self.prime_data(data)
        data_normalized = self.normalize(data)
        pred = np.where(self.model.predict(data_normalized[self.cols]) > 0.5, 1, 0)
        data["prediction"] = np.where(pred > 0, 1, -1)
        if strict_hold:
            data["prediction"] = (
                data["prediction"] - data["prediction"].shift(1)
            ) / 2  # add holding as 0
            data.loc[data.index[0], "prediction"] = 0
        data["prediction"] = data["prediction"].astype(int)
        return data

    def get_signal(self, data: pd.DataFrame, strict_hold=False):
        """
        Get a signal from an already trained model
        """
        truncated = data.copy()
        predset = self.predict(truncated, strict_hold=strict_hold)

        prediction = predset.iloc[-1].prediction
        return prediction

    def evaluate(
        self,
        data: pd.DataFrame,
        tt_split: int = 0.8,
        strict_hold=False,
        securityname: str = None,
    ):
        """Vectorize evaluate - split data into train and test, build, and evaluate"""
        # prime data happens inside each function

        # split
        train, test = (
            data[: int(data.shape[0] * tt_split)],
            data[int(data.shape[0] * tt_split) :],
        )

        # train test sizes
        logger.info(
            f"Train: {train.index[0]} - {train.index[-1]} ({train.shape[0]}, {len(self.cols)})"
        )
        logger.info(
            f"Test: {test.index[0]} - {test.index[-1]} ({test.shape[0]}, {len(self.cols)})"
        )
        # make model
        self.build(train)

        # train model
        self.train(train)

        # predict
        predictions = self.predict(test, strict_hold=strict_hold)[
            ["close", "return", "prediction"]
        ]

        # calcluate returns
        predictions["strategy"] = predictions["prediction"] * predictions["return"]

        predictions["buys"] = (
            np.where(predictions["prediction"] == -1, 1, NaN) * predictions["close"]
        )
        predictions["sells"] = (
            np.where(predictions["prediction"] == 1, 1, NaN) * predictions["close"]
        )

        # count trades
        num_trades = (predictions["prediction"] != 0).sum()

        logger.info(f"Trades made: {num_trades}")

        # define returns
        predictions["return"] = predictions["return"].cumsum().apply(np.exp)
        predictions["strategy"] = predictions["strategy"].cumsum().apply(np.exp)
        returns = predictions[["return", "strategy"]]  # .cumsum().apply(np.exp)

        logger.info(f"Returns [{securityname}]:\n{returns.tail(1)}")

        # write to csv and stuff
        if securityname:
            self.save_plot(predictions, securityname)
        return returns

    def test_w_stocks(self, symbols: list, strict_hold=False):
        shuffle(symbols)
        for symbol in symbols:
            try:
                data = self.alpaca.get_bars(
                    symbol,
                    timeframe=TimeFrame.Minute,
                    start_time=datetime.now() - timedelta(days=10),
                    end_time=datetime.now(),
                )
                # evaluate
                self.evaluate(
                    data, tt_split=0.7, securityname=symbol, strict_hold=strict_hold
                )
            except Exception as e:
                logging.warning(f"Couldn't do {symbol} ({e})")


def test_w_crypto(symbols, strict_hold=False):
    bnc = BinanceUS()
    shuffle(symbols)
    for symbol in symbols:
        try:
            data = bnc.get_bars(
                symbol,
                start_time=datetime.now() - timedelta(days=10),
                end_time=datetime.now(),
            )

            # create fednn
            fednn = FEDNN(epochs=25)
            # evaluate
            fednn.evaluate(
                data, tt_split=0.8, securityname=symbol, strict_hold=strict_hold
            )
        except Exception as e:
            logging.warning(f"Couldn't do {symbol} ({e})")


if __name__ == "__main__":
    screener = Screener()
    cryptosymbols = [
        "BTCUSD",
        "ETHUSD",
        "ADAUSD",
    ]
    stocksymbols = screener.get_active()["symbol"]
    fednn = FEDNN()
    fednn.test_w_stocks(stocksymbols, strict_hold=False)
    test_w_crypto(cryptosymbols, strict_hold=True)
