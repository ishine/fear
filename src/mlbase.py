"""
Class designed for inheritance
"""
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


class BaseTrader:
    def __init__(
        self,
        lags: int = 5,
        cols: list = ["return"],
    ) -> None:
        self.lags = lags  # number of lags
        self.cols = cols  # col labels

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

    def build(self, data: pd.DataFrame):
        """Compile model"""
        pass

    def train(self, data: pd.DataFrame):
        """Train the model"""
        pass

    def write_info(self, text: str, filepath: str):
        """Write a string to a file"""
        with open(filepath, "w+") as f:
            f.write(text)

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