import logging
import os
from datetime import datetime, timedelta
from random import shuffle

from numpy.core.numeric import NaN

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd
import tensorflow as tf
from finta import TA as ta

tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(1)

from abc import abstractmethod

import plotly.graph_objects as go
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

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


class BasePredictor:
    def __init__(
        self,
        lags: int = 5,
        cols: list = ["return"],
        momentum: list = [15, 30, 60, 120],
    ) -> None:
        self.lags = lags  # number of lags
        self.cols = cols  # col labels
        self.momentum = momentum

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
            data["direction"] = np.where(data["return"] > 0, 1, -1)
        except Exception as e:
            logger.warning(f"Unknown error occured while generating direction ({e})")
        return data

    def _create_next_direction(self, data: pd.DataFrame):
        """Create direction from returns"""
        try:
            data["next_direction"] = np.where(
                data["close"].shift(-1) > data["close"], 1, -1
            )
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

    def _create_momentum(self, data: pd.DataFrame):
        """Create the lags"""
        try:
            col = f"position_{self.momentum}"
            data[col] = np.sign(data["return"].rolling(self.momentum).mean())
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

    @abstractmethod
    def _create_features(self, data: pd.DataFrame):
        """To be implemented"""
        return data

    def prime_data(self, data: pd.DataFrame, prune: bool = False):
        """Prime the data for the network"""
        if prune:
            data = data[["close"]]
        data = self._create_returns(data)
        data = self._create_direction(data)
        data = self._create_features(data)
        data = self._create_lags(data)
        self.cols = list(set(self.cols))  # remove duplicates
        return data

    def fit_scaler(self, data: pd.DataFrame):
        """Fit the min max scaler - MUST BE CALLED BEFORE SCALING"""
        self.mu, self.std = data.mean(), data.std()

    def normalize(self, data: pd.DataFrame):
        """Normalize the data and return"""
        data = data.copy()
        normalized = (data - self.mu) / self.std
        return normalized

    @abstractmethod
    def build(self, data: pd.DataFrame):
        """Compile model"""
        pass

    @abstractmethod
    def train(self, data: pd.DataFrame):
        """Train the model"""
        pass

    @abstractmethod
    def evaluate(self, data: pd.DataFrame):
        """Evaluate the model"""
        pass

    def write_info(self, text: str, filepath: str):
        """Write a string to a file"""
        with open(filepath, "w+") as f:
            f.write(text)

    @abstractmethod
    def generate_text(self, predictions: pd.DataFrame, securityname: str = ""):
        """Writes the hyper params to file"""
        pass

    def save_plot(self, predictions: pd.DataFrame, securityname: str = ""):
        """
        Save the plot to files
        """
        if securityname == "":
            securityname = "unknown"
        fig = self.generate_plot(predictions, securityname)
        info = self.generate_text(predictions, securityname)

        if not os.path.exists(CHARTPATH):
            os.mkdir(CHARTPATH)

        path = os.path.join(CHARTPATH, securityname)

        if not os.path.exists(path):
            os.mkdir(path)
        dt = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")

        modeltype = self.__class__.__name__

        imgname = securityname + "_" + modeltype + "_" + dt + ".png"
        htmname = securityname + "_" + modeltype + "_" + dt + ".html"
        returnname = securityname + "_" + modeltype + "_" + dt + ".csv"
        infoname = securityname + "_" + modeltype + "_" + dt + ".txt"

        imgpath = os.path.join(path, imgname)
        htmpath = os.path.join(path, htmname)
        returnpath = os.path.join(path, returnname)
        infopath = os.path.join(path, infoname)

        fig.write_image(imgpath, scale=2)
        fig.write_html(htmpath)
        predictions.to_csv(returnpath)
        # self.write_info(info, infopath)

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


class FEDNNPredictor(BasePredictor):  # feature engineering deep neural network
    def __init__(
        self,
        units: int = 64,
        learning_rate: float = 0.0001,
        lags: int = 7,
        cols: list = ["return"],
        epochs: int = 25,
        mv_avgs: list = [5, 10, 20, 30, 50, 100, 200],
    ) -> None:
        super(FEDNNPredictor, self).__init__(lags=lags, cols=cols)
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
                columns={"SIGNAL": "MACD_SIGNAL"}
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
                "momentum",
                "volatility",
                "distance",
                "14 period RSI",
                "volume",
                "MACD",
                "MACD_SIGNAL",
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
        validation = np.where(self.model.predict(data[self.cols]) > 0.5, 1, 0)
        accuracy_in_sample = accuracy_score(data["direction"], validation)
        logger.info(f"In-sample accuracy={accuracy_in_sample:.4f}")

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


class FEKNNPredictor(BasePredictor):
    def __init__(
        self,
        window: int = 20,
        lags: int = 6,
        n_neighbors: int = 65,
    ) -> None:
        """
        window: analysis window
        """
        super(FEKNNPredictor, self).__init__(lags=lags)
        self.n_neighbors = n_neighbors
        self.window = window

    def _create_features(self, data: pd.DataFrame):
        """Create the features"""
        data = data.copy()
        data["vol"] = data["return"].rolling(self.window).std()
        data["mom"] = np.sign(data["return"].rolling(self.window).mean())
        data["sma"] = data["close"].rolling(self.window).mean()
        data["min"] = data["close"].rolling(self.window).min()
        data["max"] = data["close"].rolling(self.window).max()

        data["MA_Fast"] = data["close"].ewm(span=13, min_periods=13).mean()
        data["MA_Slow"] = data["close"].ewm(span=24, min_periods=24).mean()
        data["MACD"] = data["MA_Fast"] - data["MA_Slow"]
        data["MACD_signal"] = data["MACD"].ewm(span=9, min_periods=9).mean()

        data = self._append_data(data, ta.RSI(data))
        # self.features.extend(["vol", "mom", "sma", "min", "max", "14 period RSI"])
        self._add_columns(
            "close",
            "MA_Fast",
            "MA_Slow",
            "MACD",
            "MACD_signal",
            "14 period RSI",
            "vol",
            "mom",
            "sma",
            "min",
            "max",
        )
        return data

    def build(self, data: pd.DataFrame, summary: bool = False):
        """Compile model"""
        data = data.copy()
        data = self.prime_data(data)
        logger.info(f"Building model using features {self.cols}")

        self.model_unfitted = KNeighborsClassifier(n_neighbors=self.n_neighbors)

    def train(self, data: pd.DataFrame):
        """Train the model"""
        data = data.copy()
        data = self.prime_data(data)
        logger.info(f"Training model on {data.shape[0]} records")
        self.model = self.model_unfitted.fit(data[self.cols], data["direction"])
        validation = self.model.predict(data[self.cols])
        accuracy_in_sample = accuracy_score(data["direction"], validation)
        logger.info(f"In-sample accuracy={accuracy_in_sample:.4f}")

    def predict(self, data: pd.DataFrame, strict_hold: bool = False):
        """Predict
        Strict hold = introduce hold signals (0) as well, not just 1 and -1
        """
        data = data.copy()
        data = self.prime_data(data)

        data["prediction"] = self.model.predict(data[self.cols])
        if strict_hold:
            data["prediction"] = (
                data["prediction"] - data["prediction"].shift(1)
            ) / 2  # add holding as 0
            data.loc[data.index[0], "prediction"] = 0
        return data

    def get_signal(self, data: pd.DataFrame):
        """
        Get a signal from an already trained model
        """
        truncated = data.copy()
        predset = self.predict(truncated)

        prediction = predset.iloc[-1].prediction
        return prediction

    def evaluate(
        self,
        data: pd.DataFrame,
        tt_split: int = 0.8,
        strict_hold: bool = False,
        securityname: str = None,
    ):
        """Vectorize evaluate - split data into train and test, build, and evaluate"""
        # prime data happens inside each function
        data = data.copy()

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

        # count trades
        num_trades = (predictions["prediction"] != 0).sum()

        logger.info(f"Trades made: {num_trades}")

        # define returns
        predictions["return"] = predictions["return"].cumsum().apply(np.exp)
        predictions["strategy"] = predictions["strategy"].cumsum().apply(np.exp)

        predictions["buys"] = (
            np.where(predictions["prediction"] == -1, 1, NaN) * predictions["close"]
        )
        predictions["sells"] = (
            np.where(predictions["prediction"] == 1, 1, NaN) * predictions["close"]
        )

        returns = predictions[["return", "strategy"]]  # .cumsum().apply(np.exp)

        logger.info(f"Returns [{securityname}]:\n{returns.tail(1)}")
        # write to csv and stuff
        if securityname:
            self.save_plot(predictions, securityname)
        return returns

    def tune(self, data: pd.DataFrame, k_range: range = range(1, 70)):
        """Search a k range and find the best param"""
        values = pd.DataFrame(columns=["k", "strategy"])
        for k in k_range:
            self.n_neighbors = k
            returns = self.evaluate(data, tt_split=0.7).tail(1)[["strategy"]]
            returns["k"] = k
            values = values.append(returns.iloc[[-1]])
        best = values.iloc[values["strategy"].values.argmax()]["k"]
        self.n_neighbors = best
        return best
