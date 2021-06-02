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
from time import sleep

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