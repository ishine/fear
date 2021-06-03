"""
Class designed for inheritance
"""
import logging
import os
from datetime import datetime, timedelta
from threading import Thread

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from time import sleep

import numpy as np
import pandas as pd
import tensorflow as tf

tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(1)

import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


class BaseStrategy:
    def __init__(
        self,
        lags: int = 5,
        cols: list = ["return"],
        api_key: str = open("keys/alpaca_paper_public").read().strip(),
        api_secret: str = open("keys/alpaca_paper_private").read().strip(),
        base_url: str = "https://paper-api.alpaca.markets",
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


class BaseCycler:
    def __init__(self, **kwargs) -> None:
        """
        Example:
        def __init__(self,
        api_key: str = open("keys/alpaca_paper_public").read().strip(),
        api_secret: str = open("keys/alpaca_paper_private").read().strip(),
        base_url: str = "https://paper-api.alpaca.markets",):

            self.alpaca = Alpaca(api_key=api_key, api_secret=api_secret, base_url=base_url)
        """
        self.track = pd.DataFrame(columns=["price", "prediction"])

    def _add_track(self, price: float, prediction: int):
        self.track = self.track.append(
            pd.DataFrame({"price": [price], "prediction": [prediction]})
        )

    def trade(self, ticker: str, side: str, price: float, qty: int):
        """Trades if all checks pass"""
        buying_power = self.get_buying_power()

        try:
            if side == "buy":
                if buying_power >= price * qty:
                    self.submit_limit_order(
                        ticker=ticker, side=side, price=price, qty=qty
                    )
                    return True
                else:
                    logger.warning(
                        f"Not enough balance to buy ({buying_power} < {price*qty})"
                    )
            elif side == "sell":
                self.submit_limit_order(ticker=ticker, side=side, price=price, qty=qty)
                return True
            elif side == "hold":  # do nothing
                return True
        except Exception as e:
            logger.warning(
                f"Couldn't place order ({e}). Is shorting enabled on your account?"
            )
        return False

    def cycle_trades(self, ticker: str):
        """
        Cycles between predicting and trading

        Params
        ------
        ticker : str
            stock ticker
        spend_amt : int = 1000
            max amount of money to spend
        """

        while True:
            # get the data
            data = self.get_data(symbol=ticker)
            close = data.close
            price = close[-1]
            qty = 1

            prediction = self.get_signal(data)

            if prediction == -1:
                signal = "buy"
            elif prediction == 1:
                signal = "sell"
            else:
                signal = "hold"

            self._add_track(price, prediction)
            logger.info(f"{signal} {qty} @ {price:.2f}")
            # act
            # self.trade(ticker, signal, price, qty)
            # sleep til next min start
            logger.info(f"Sleeping {60 - datetime.now().second} s ...")
            sleep(60 - datetime.now().second)

    def cycle_train(self, ticker: str, train_interval: int = 30):
        """
        Cycles training the prediction model

        Params
        ------
        ticker : str
            stock ticker
        train_interval : int
            how many minutes between trains
        """
        # model must have been built first
        interval = 60 * train_interval  # convert to seconds

        # build the model
        data = self.get_data(ticker=ticker)
        self.build(data)

        while True:
            data = self.get_data(ticker=ticker)
            self.train(data)
            logger.info(f"Sleeping {interval} mins ...")
            sleep(interval)

    def cycle(self, ticker: str, train_interval: int = 30):
        """
        first train then start
        # build the model
        data = self.binanceus.get_bars(
            ticker,
            # timeframe=TimeFrame.Minute,
            start_time=datetime.now() - timedelta(days=14),
        )
        self.build(data)
        self.train(data)

        self.cycle_trades(ticker)

        """
        logger.info("Starting training thread")
        training_thread = Thread(
            target=self.cycle_train,
            args=(
                ticker,
                train_interval,
            ),
        )
        training_thread.start()
        sleep(60)
        logger.info("Starting trading thread")
        trading_thread = Thread(
            target=self.cycle_trades,
            args=(ticker,),
        )
        trading_thread.start()
        # join
        training_thread.join()
        trading_thread.join()

    def build_models(self, data):
        """
        Build model to be implemented
        """
        pass

    def train_models(self, data):
        """
        Train model to be implemented
        """
        pass

    def get_data(self, ticker: str, **kwargs):
        """
        Train model to be implemented
        Example:
            data = self.alpaca.get_bars(
                ticker,
                timeframe=TimeFrame.Minute,
                start_time=datetime.now() - timedelta(days=14),
            )

        """
        pass

    def get_signal(self, datà):
        """Get the signals"""
        pass

    def submit_limit_order(self, ticker: str, side: str, price: float, qty: int = 1):
        """Submit order
        Example:
        self.alpaca.submit_limit_order(
            ticker=ticker, side=side, price=price, qty=qty
        )
        """
        pass