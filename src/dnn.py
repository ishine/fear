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


class FEDNN:  # feature engineering deep neural network
    def __init__(
        self,
        units: int = 32,
        learning_rate: float = 0.0001,
        lags: int = 5,
        cols: list = ["return"],
        epochs: int = 25,
        mv_avgs: list = [5, 10, 20, 30, 50, 100, 200],
    ) -> None:
        """dfcols is the labels for the dataframe that you recieve"""
        self.mv_avgs = mv_avgs

        # for ml
        self.units = units
        self.optimizer = Adam(learning_rate=learning_rate)

        self.lags = lags  # number of lags
        self.cols = cols  # col labels
        self.epochs = epochs

    def _add_columns(self, *args):
        """Add columns to self.columns"""
        self.cols.extend(list(args))
        self.cols = list(set(self.cols))  # remove duplicates

    def _create_returns(self, data: pd.DataFrame):
        """Create the log returns"""
        try:
            data["return"] = np.log(data["close"] / data["close"].shift(1))
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
                self.cols.append(col)
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

    def prime_data(self, data: pd.DataFrame, prune: bool = False):
        """Prime the data for the network"""
        if prune:
            data = data[["close"]]
        data = self._create_returns(data)
        data = self._create_direction(data)
        data = self._create_features(data)
        data = self._create_lags(data)
        self.cols = list(set(self.cols))  # remove duplicates
        # logger.info("Primed data")
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
            self.save_plot(data)

    def write_info(self, text: str, filepath: str):
        """Write a string to a file"""
        with open(filepath, "w+") as f:
            f.write(text)

    def generate_text(self, predictions: pd.DataFrame, securityname: str = ""):
        """Writes the hyper params to file"""
        out_str = f"[{securityname}]\n"
        out_str += f"features={self.cols}\n"
        # out_str += f"train_size={train.shape[0]}\n"
        out_str += f"test_size={predictions.shape[0]}\n"
        out_str += f"units={self.units}\n"
        out_str += f"epochs={self.epochs}\n"
        out_str += f"num_buys={predictions['buys'].count()}\n"
        out_str += f"num_sells={predictions['sells'].count()}\n"
        out_str += f"return_b_and_h={predictions['return'][-1]}\n"
        out_str += f"return_strategy={predictions['strategy'][-1]}\n"
        return out_str

    def save_plot(self, predictions: pd.DataFrame, securityname: str = ""):
        """
        Save the plot to files
        """
        fig = self.generate_plot(predictions, securityname)
        info = self.generate_text(predictions, securityname)

        if not os.path.exists(CHARTPATH):
            os.mkdir(CHARTPATH)

        path = os.path.join(CHARTPATH, securityname)

        if not os.path.exists(path):
            os.mkdir(path)
        dt = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

        imgname = securityname + "_" + dt + ".png"
        htmname = securityname + "_" + dt + ".html"
        returnname = securityname + "_" + dt + ".csv"
        infoname = securityname + "_" + dt + ".txt"

        imgpath = os.path.join(path, imgname)
        htmpath = os.path.join(path, htmname)
        returnpath = os.path.join(path, returnname)
        infopath = os.path.join(path, infoname)

        fig.write_image(imgpath, scale=2)
        fig.write_html(htmpath)
        predictions.to_csv(returnpath)
        self.write_info(info, infopath)

    def generate_plot(self, predictions: pd.DataFrame, securityname: str = ""):
        """Generate a plot from the returns dataframe with columns ['return', 'strategy']"""
        markersize = 8
        xticks = 40
        yticks = 25

        ### make signal marks

        buy_count = predictions["buys"].count()
        sell_count = predictions["sells"].count()
        trade_count = buy_count + sell_count

        predictions["human-return"] = predictions["return"] * 100 - 100
        predictions["human-strategy"] = predictions["strategy"] * 100 - 100

        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=(
                f"{securityname.upper()} predictions",
                f"{securityname.upper()} Close Price",
            ),
            vertical_spacing=0.08,
        )

        fig.add_trace(
            go.Scatter(
                x=predictions.index,
                y=predictions["human-strategy"],
                name="FEAR Strategy",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=predictions.index,
                y=predictions["human-return"],
                name="Buy & Hold",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=predictions.index,
                y=predictions["close"],
                name=f"{securityname.upper()} Close Price",
                line=dict(color="#4e5cdc"),
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=predictions.index,
                y=predictions["buys"],
                name="Buy Signals",
                marker=dict(
                    color="#5bcf5b",
                    size=markersize,
                ),
                mode="markers",
                marker_symbol="triangle-up",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=predictions.index,
                y=predictions["sells"],
                name="Sell Signals",
                marker=dict(
                    color="Red",
                    size=markersize,
                ),
                mode="markers",
                marker_symbol="triangle-down",
            ),
            row=2,
            col=1,
        )
        fig.add_annotation(
            x=predictions.index[-1],
            y=predictions["human-return"][-1],
            text=f"{predictions['human-return'][-1]:.3f}%",
            showarrow=True,
            ax=20,
            ay=-30,
            bordercolor="#c7c7c7",
            borderwidth=2,
            borderpad=4,
            bgcolor="#2e2e2e",
            opacity=0.8,
            font=dict(color="#ffffff"),
            align="center",
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#636363",
            row=1,
            col=1,
        )
        fig.add_annotation(
            x=predictions.index[-1],
            y=predictions["human-strategy"][-1],
            text=f"{predictions['human-strategy'][-1]:.3f}%",
            showarrow=True,
            ax=20,
            ay=30,
            bordercolor="#c7c7c7",
            borderwidth=2,
            borderpad=4,
            bgcolor="#2e2e2e",
            opacity=0.8,
            font=dict(color="#ffffff"),
            align="center",
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#636363",
            row=1,
            col=1,
        )
        fig.add_annotation(
            x=predictions.index[0],
            y=predictions["human-strategy"][0],
            text=f"{predictions['human-strategy'][0]:.3f}%",
            showarrow=True,
            ax=20,
            ay=30,
            bordercolor="#c7c7c7",
            borderwidth=2,
            borderpad=4,
            bgcolor="#2e2e2e",
            opacity=0.8,
            font=dict(color="#ffffff"),
            align="center",
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#636363",
            row=1,
            col=1,
        )
        fig.add_annotation(
            x=predictions.index[0],
            y=predictions["close"][0],
            text=f"${predictions['close'][0]:.2f}",
            showarrow=True,
            ax=20,
            ay=30,
            bordercolor="#c7c7c7",
            borderwidth=2,
            borderpad=4,
            bgcolor="#2e2e2e",
            opacity=0.8,
            font=dict(color="#ffffff"),
            align="center",
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#636363",
            row=2,
            col=1,
        )
        fig.add_annotation(
            x=predictions.index[-1],
            y=predictions["close"][-1],
            text=f"${predictions['close'][-1]:.2f}",
            showarrow=True,
            ax=20,
            ay=-30,
            bordercolor="#c7c7c7",
            borderwidth=2,
            borderpad=4,
            bgcolor="#2e2e2e",
            opacity=0.8,
            font=dict(color="#ffffff"),
            align="center",
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#636363",
            row=2,
            col=1,
        )

        # add counts
        fig.add_annotation(
            xref="x domain",
            yref="y domain",
            x=0.02,
            y=0.97,
            text=f"# of buys: {buy_count}<br># of sells: {sell_count}",
            showarrow=False,
            bordercolor="#c7c7c7",
            borderwidth=2,
            borderpad=4,
            bgcolor="#2e2e2e",
            opacity=0.8,
            font=dict(color="#ffffff"),
            align="center",
            row=1,
            col=1,
        )

        # plot scatter
        fig.update_layout(
            width=1920,
            height=1080,
            title=f"{securityname.upper()} FEAR Strategy vs Buy & Hold predictions from {predictions.index[0]} to {predictions.index[-1]}",
            template="plotly_dark",
        )
        fig.update_xaxes(nticks=xticks, row=1, col=1)
        fig.update_xaxes(nticks=xticks, title_text="Date", row=2, col=1)
        fig.update_yaxes(nticks=yticks, title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(nticks=yticks, title_text="Price ($)", row=2, col=1)
        return fig


def test_w_stocks(symbols, strict_hold=False):
    ape = Alpaca()
    shuffle(symbols)
    for symbol in symbols:
        try:
            data = ape.get_bars(
                symbol,
                timeframe=TimeFrame.Minute,
                start_time=datetime.now() - timedelta(days=10),
                end_time=datetime.now(),
            )

            # create fednn
            fednn = FEDNN(epochs=25, units=64)
            # evaluate
            fednn.evaluate(
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
    test_w_stocks(stocksymbols, strict_hold=False)
    test_w_crypto(cryptosymbols, strict_hold=True)
